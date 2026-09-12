import json
import os
import random
from pathlib import Path

import pandas as pd
import torch

from live_torch_mbon_policy import LiveTorchMBONPolicy, CON_FILE
from fly_mb_policy import FlyMBPolicy, ACTIONS


BASE = Path(__file__).resolve().parent
MEMORY_FILE = BASE / "data" / "fly_memory.json"
RICH_PATTERN_FILE = BASE / "priority_kc_subpatterns_16.json"


class TrainableTorchMBONPolicy16(LiveTorchMBONPolicy):
    """
    Full 138k-neuron CUDA fly with 16 sensory states:

      - 4 cardinal states
      - 4 diagonal directions x {X_DOMINANT, BALANCED, Y_DOMINANT}

    The neural simulation is still the same full PyTorch connectome. The only
    change is which real KCs are activated for a sensory state and which of
    those real KC->MBON synapses are eligible for online plasticity.

    Learning is dopamine-style multiplicative plasticity, not backpropagation.
    Disk writes are only done by save(), which uses an atomic replace.
    """

    def __init__(
        self,
        run_time_ms=3,
        device="cuda",
        reward_multiplier=1.01,
        punishment_multiplier=0.99,
        min_weight=0.25,
        max_weight=4.0,
    ):
        # Build the same validated full neural CUDA model first.
        super().__init__(
            run_time_ms=run_time_ms,
            device=device,
        )

        print("Adding 16-state online KC->MBON plasticity...")

        self.brain = FlyMBPolicy()
        self.memory = self.brain.memory

        self.reward_multiplier = float(reward_multiplier)
        self.punishment_multiplier = float(punishment_multiplier)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

        if not RICH_PATTERN_FILE.exists():
            raise FileNotFoundError(
                f"Missing {RICH_PATTERN_FILE.name}. Put it beside this file."
            )

        with open(RICH_PATTERN_FILE, "r", encoding="utf-8") as f:
            self.rich_patterns = json.load(f)

        if len(self.rich_patterns) != 16:
            raise RuntimeError(
                f"Expected 16 rich sensory patterns, found {len(self.rich_patterns)}."
            )

        self.rich_states = list(self.rich_patterns.keys())

        # Make sure each KC belongs to only one rich state. This is important:
        # a reward for one sensory condition must not directly modify another.
        self.rich_kc_usage = {}
        for info in self.rich_patterns.values():
            for kc_idx in info["kc_indices"]:
                kc_idx = int(kc_idx)
                self.rich_kc_usage[kc_idx] = (
                    self.rich_kc_usage.get(kc_idx, 0) + 1
                )

        overlaps = [
            kc for kc, count in self.rich_kc_usage.items()
            if count != 1
        ]
        if overlaps:
            raise RuntimeError(
                f"16-state pattern file has {len(overlaps)} KCs shared between states."
            )

        # Biological base weights are needed to build the effective learned
        # correction tensor used by the next full neural simulation.
        df = pd.read_parquet(CON_FILE)
        self.pair_weight = {}

        for pre, post, value in zip(
            df["Presynaptic_Index"].to_numpy(),
            df["Postsynaptic_Index"].to_numpy(),
            df["Excitatory x Connectivity"].to_numpy(),
        ):
            pair = (int(pre), int(post))
            self.pair_weight[pair] = (
                self.pair_weight.get(pair, 0.0) + float(value)
            )

        # Rich-state neural input and correction plans.
        self.rich_state_kcs = {}
        self.rich_baseline_plan = {}
        self.rich_learned_plan = {}

        for rich_state in self.rich_states:
            info = self.rich_patterns[rich_state]

            self.rich_state_kcs[rich_state] = torch.tensor(
                [int(x) for x in info["kc_indices"]],
                dtype=torch.long,
                device=self.device,
            )

            self.rich_baseline_plan[rich_state] = {}
            self.rich_learned_plan[rich_state] = {}

            for action in ACTIONS:
                self._rebuild_rich_action(
                    rich_state,
                    action,
                    rebuild_baseline=True,
                )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Warm one rich-state simulation.
        self._run_once_rich(self.rich_states[0], learned=False)

        print("Caching 16-state unlearned GPU baselines...")
        self.rich_baseline_cache = {
            state: self._run_once_rich(state, learned=False)
            for state in self.rich_states
        }

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        print("16-state trainable CUDA fly ready.")

    @staticmethod
    def _direction_key(direction):
        return f"{int(direction[0])},{int(direction[1])}"

    def _rich_info(self, rich_state):
        if rich_state not in self.rich_patterns:
            raise ValueError(
                f"Unknown 16-state sensory key: {rich_state}"
            )
        return self.rich_patterns[rich_state]

    def direct_connection_counts(self, rich_state):
        """Return real direct KC->action-MBON edge counts for diagnostics."""
        info = self._rich_info(rich_state)
        direction_key = self._direction_key(info["direction"])
        active_kcs = {int(x) for x in info["kc_indices"]}

        counts = {}
        for action in ACTIONS:
            counts[action] = sum(
                1
                for conn in self.brain.real_map[direction_key][action]["connections"]
                if int(conn["kc_index"]) in active_kcs
            )
        return counts

    def _build_rich_action_plan(self, rich_state, action, learned):
        info = self._rich_info(rich_state)
        direction_key = self._direction_key(info["direction"])
        active_kcs = {int(x) for x in info["kc_indices"]}

        post_idx = int(self.brain.action_mbon_indices[action])
        calibration = float(
            self.calibration[direction_key][action]["factor"]
        )

        mods = self.memory.get_weight_multipliers()

        pres = []
        deltas = []

        for connection in self.brain.real_map[direction_key][action]["connections"]:
            pre_idx = int(connection["kc_index"])

            if pre_idx not in active_kcs:
                continue

            biological_weight = self.pair_weight.get(
                (pre_idx, post_idx),
                0.0,
            )

            if biological_weight == 0.0:
                continue

            multiplier = 1.0
            if learned:
                multiplier = float(
                    mods.get((pre_idx, post_idx), 1.0)
                )

            # The sparse connectome already contributes biological_weight.
            # Add a correction so the effective direct edge becomes:
            # biological_weight * calibration * learned_multiplier.
            delta = biological_weight * (
                calibration * multiplier - 1.0
            )

            pres.append(pre_idx)
            deltas.append(delta)

        pre_tensor = torch.tensor(
            pres,
            dtype=torch.long,
            device=self.device,
        )
        delta_tensor = torch.tensor(
            deltas,
            dtype=torch.float32,
            device=self.device,
        )

        return pre_tensor, delta_tensor, post_idx

    def _rebuild_rich_action(
        self,
        rich_state,
        action,
        rebuild_baseline=False,
    ):
        if rebuild_baseline:
            self.rich_baseline_plan[rich_state][action] = (
                self._build_rich_action_plan(
                    rich_state,
                    action,
                    learned=False,
                )
            )

        self.rich_learned_plan[rich_state][action] = (
            self._build_rich_action_plan(
                rich_state,
                action,
                learned=True,
            )
        )

    @torch.no_grad()
    def _run_once_rich(self, rich_state, learned):
        self._rich_info(rich_state)

        conductance, delay_buffer, spikes, v, refrac = self.model.state_init()

        # Same validated sensory stimulation method: selected KCs start just
        # above threshold, but now only the KCs for this 16-state condition.
        v[:, self.rich_state_kcs[rich_state]] = -44.0

        peak = torch.full(
            (len(self.action_order),),
            -float("inf"),
            device=self.device,
        )

        plan = (
            self.rich_learned_plan[rich_state]
            if learned
            else self.rich_baseline_plan[rich_state]
        )

        for _ in range(self.num_steps):
            weighted = torch.matmul(
                spikes,
                self.weights.transpose(0, 1),
            )

            self._apply_correction(
                weighted,
                spikes,
                plan,
            )

            conductance, delay_buffer, spikes, v, refrac = self.model.neurons(
                self.model.scale * weighted,
                conductance,
                delay_buffer,
                spikes,
                v,
                refrac,
            )

            peak = torch.maximum(
                peak,
                conductance[0, self.action_indices],
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        values = peak.cpu().tolist()

        return {
            action: float(values[i])
            for i, action in enumerate(self.action_order)
        }

    def get_scores(self, rich_state):
        learned = self._run_once_rich(
            rich_state,
            learned=True,
        )
        baseline = self.rich_baseline_cache[rich_state]

        return {
            action: learned[action] - baseline[action]
            for action in self.action_order
        }

    def choose_action(self, rich_state, epsilon=0.0):
        # Always run the real full neural model. Exploration only changes the
        # selected training action after neural scores have been computed.
        scores = self.get_scores(rich_state)

        if random.random() < float(epsilon):
            action = random.choice(ACTIONS)
        else:
            best = max(scores.values())
            winners = [
                action
                for action, score in scores.items()
                if score == best
            ]
            action = random.choice(winners)

        return action, scores

    def learn(self, rich_state, action, reward):
        info = self._rich_info(rich_state)

        if action not in ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        multiplier = (
            self.reward_multiplier
            if float(reward) > 0
            else self.punishment_multiplier
        )
        signal = "reward" if float(reward) > 0 else "punishment"

        direction_key = self._direction_key(info["direction"])
        active_kcs = {int(x) for x in info["kc_indices"]}
        mbon_idx = int(self.brain.action_mbon_indices[action])

        mods = self.memory.memory["weight_modifications"]

        changed = 0

        for connection in self.brain.real_map[direction_key][action]["connections"]:
            kc_idx = int(connection["kc_index"])

            if kc_idx not in active_kcs:
                continue

            # Rich patterns are required to be disjoint, so this synapse is
            # specific to exactly one of the 16 sensory states.
            if self.rich_kc_usage.get(kc_idx, 0) != 1:
                continue

            memory_key = f"{kc_idx}:{mbon_idx}"
            old = float(mods.get(memory_key, 1.0))
            new = old * multiplier
            new = max(self.min_weight, min(self.max_weight, new))
            mods[memory_key] = round(new, 5)
            changed += 1

        # The very next full neural simulation uses the new synaptic strength.
        self._rebuild_rich_action(
            rich_state,
            action,
            rebuild_baseline=False,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        return {
            "signal_type": signal,
            "multiplier": multiplier,
            "changed": changed,
        }

    def save(self):
        """Atomically persist the current synaptic memory."""
        temp_file = MEMORY_FILE.with_suffix(".json.tmp")

        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(self.memory.memory, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_file, MEMORY_FILE)
