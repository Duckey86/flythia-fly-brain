import sys, json, time
import numpy as np
import pandas as pd
import pyarrow
import torch
from pathlib import Path

from dopamine_learning import FlyMemory
from fly_mb_policy import FlyMBPolicy, ACTION_MBONS, STATES

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "code"))

from run_pytorch import TorchModel, MODEL_PARAMS, DT, get_weights

COMP_FILE = BASE / "data" / "2025_Completeness_783.csv"
CON_FILE = BASE / "data" / "2025_Connectivity_783.parquet"
CAL_FILE = BASE / "mbon_calibration.json"
WEIGHT_DIR = BASE / "data"


class LiveTorchMBONPolicy:
    def __init__(self, run_time_ms=3.0, device="cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        self.device = torch.device(device)
        self.run_time_ms = float(run_time_ms)
        self.num_steps = max(1, int(round(self.run_time_ms / DT)))

        print("Loading full PyTorch fly connectome on", self.device)

        self.brain = FlyMBPolicy()
        self.memory = FlyMemory()

        with open(CAL_FILE, "r", encoding="utf-8") as f:
            self.calibration = json.load(f)

        self.weights = get_weights(
            str(CON_FILE),
            str(COMP_FILE),
            str(WEIGHT_DIR),
            csr=True,
        ).to(self.device)

        self.num_neurons = int(self.weights.shape[0])

        self.model = TorchModel(
            1,
            self.num_neurons,
            DT,
            MODEL_PARAMS,
            self.weights,
            device=str(self.device),
        )

        self.action_order = list(ACTION_MBONS.keys())
        self.action_indices = torch.tensor(
            [self.brain.action_mbon_indices[a] for a in self.action_order],
            dtype=torch.long,
            device=self.device,
        )

        df = pd.read_parquet(CON_FILE)
        pair_weight = {}
        for pre, post, val in zip(
            df["Presynaptic_Index"].to_numpy(),
            df["Postsynaptic_Index"].to_numpy(),
            df["Excitatory x Connectivity"].to_numpy(),
        ):
            k = (int(pre), int(post))
            pair_weight[k] = pair_weight.get(k, 0.0) + float(val)

        mods = self.memory.get_weight_multipliers()

        self.state_kcs = {}
        self.baseline_plan = {}
        self.learned_plan = {}

        for state in STATES:
            key = self._key(state)

            self.state_kcs[key] = torch.tensor(
                [int(x) for x in self.brain.get_state_kcs(state)],
                dtype=torch.long,
                device=self.device,
            )

            bplan = {}
            lplan = {}

            for action in self.action_order:
                post_idx = int(self.brain.action_mbon_indices[action])
                cal = float(self.calibration[key][action]["factor"])

                pres = []
                bdelta = []
                ldelta = []

                for conn in self.brain.real_map[key][action]["connections"]:
                    pre_idx = int(conn["kc_index"])
                    base = pair_weight.get((pre_idx, post_idx), 0.0)

                    if base == 0.0:
                        continue

                    mult = float(mods.get((pre_idx, post_idx), 1.0))

                    pres.append(pre_idx)
                    bdelta.append(base * (cal - 1.0))
                    ldelta.append(base * (cal * mult - 1.0))

                pre_t = torch.tensor(
                    pres, dtype=torch.long, device=self.device
                )

                bplan[action] = (
                    pre_t,
                    torch.tensor(bdelta, dtype=torch.float32, device=self.device),
                    post_idx,
                )
                lplan[action] = (
                    pre_t,
                    torch.tensor(ldelta, dtype=torch.float32, device=self.device),
                    post_idx,
                )

            self.baseline_plan[key] = bplan
            self.learned_plan[key] = lplan

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        print(
            f"GPU model ready: {self.num_neurons} neurons, "
            f"{self.num_steps} steps ({self.run_time_ms:g} ms)"
        )

        # Warm-up.
        self._run_once(STATES[0], learned=False)

        print("Caching unlearned GPU baselines...")
        t0 = time.perf_counter()

        self.baseline_cache = {
            self._key(state): self._run_once(state, learned=False)
            for state in STATES
        }

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        print(f"Baseline cache ready in {time.perf_counter()-t0:.3f}s")

    @staticmethod
    def _key(state):
        return f"{int(state[0])},{int(state[1])}"

    def _apply_correction(self, weighted, spikes, plan):
        for action in self.action_order:
            pre_idx, delta, post_idx = plan[action]

            if pre_idx.numel() == 0:
                continue

            weighted[:, post_idx] += (
                spikes[:, pre_idx] * delta.unsqueeze(0)
            ).sum(dim=1)

    @torch.no_grad()
    def _run_once(self, state, learned):
        key = self._key(state)

        conductance, delay_buffer, spikes, v, refrac = self.model.state_init()

        # Match the Brian2 test: state KCs start just above threshold.
        v[:, self.state_kcs[key]] = -44.0

        peak = torch.full(
            (len(self.action_order),),
            -float("inf"),
            device=self.device,
        )

        plan = self.learned_plan[key] if learned else self.baseline_plan[key]

        for _ in range(self.num_steps):
            weighted = torch.matmul(
                spikes,
                self.weights.transpose(0, 1),
            )

            self._apply_correction(weighted, spikes, plan)

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

        vals = peak.cpu().tolist()

        return {
            action: float(vals[i])
            for i, action in enumerate(self.action_order)
        }

    def get_scores(self, state):
        key = self._key(state)
        learned = self._run_once(state, learned=True)
        base = self.baseline_cache[key]

        return {
            action: learned[action] - base[action]
            for action in self.action_order
        }

    def choose_action(self, state, epsilon=0.0):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        scores = self.get_scores(state)
        action = max(scores, key=scores.get)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        print("GPU MBON delta peak-g:", {k: round(v, 3) for k, v in scores.items()})
        print(f"GPU neural decision: {action} ({elapsed:.4f}s)")

        return action, scores

    def learn(self, state, action, reward):
        return None

    def reset_flythia(self):
        raise RuntimeError("Uses the already-trained fly_memory.json.")
