import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

from brian2 import Network, StateMonitor, ms, mV

from dopamine_learning import FlyMemory
from fly_mb_policy import FlyMBPolicy, ACTION_MBONS, STATES


BASE = Path(__file__).resolve().parent
CODE_DIR = BASE / "code" / "paper-phil-drosophila"
sys.path.insert(0, str(CODE_DIR))

from model import create_model, default_params


COMP_FILE = BASE / "data" / "2025_Completeness_783.csv"
CON_FILE = BASE / "data" / "2025_Connectivity_783.parquet"
CALIBRATION_FILE = BASE / "mbon_calibration.json"


class LiveNeuralMBONPolicyCached:
    """
    Live Brian2 MBON policy with cached UNLEARNED baselines.

    Startup:
      - build the full Brian2 brain once
      - compute unlearned baseline peak_g for all 8 states once

    Every move:
      - run ONE fresh learned Brian2 simulation
      - subtract cached unlearned baseline
      - choose largest delta_peak_g

    The action itself is still computed live from Brian2 every move.
    """

    def __init__(self, run_time_ms=50):
        self.run_time_ms = float(run_time_ms)
        self.run_time = self.run_time_ms * ms

        self.brain = FlyMBPolicy()
        self.memory = FlyMemory()

        with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
            self.calibration = json.load(f)

        print("Building persistent Brian2 brain...")

        params = dict(default_params)
        params["t_run"] = self.run_time
        self.params = params

        self.neu, self.syn, self.spike_monitor = create_model(
            str(COMP_FILE),
            str(CON_FILE),
            params,
        )

        comp = pd.read_csv(COMP_FILE, index_col=0)
        flyid2i = {
            int(fid): i
            for i, fid in enumerate(comp.index)
        }

        self.action_indices = {
            action: flyid2i[int(fly_id)]
            for action, fly_id in ACTION_MBONS.items()
        }

        self.action_order = list(ACTION_MBONS.keys())

        self.state_monitor = StateMonitor(
            self.neu,
            ["g"],
            record=[
                self.action_indices[action]
                for action in self.action_order
            ],
        )

        self.net = Network(
            self.neu,
            self.syn,
            self.spike_monitor,
            self.state_monitor,
        )

        self.pre = np.asarray(
            self.syn.i[:],
            dtype=np.int64,
        )

        self.post = np.asarray(
            self.syn.j[:],
            dtype=np.int64,
        )

        # Incoming synapses to the four action MBONs.
        action_posts = np.array(
            [
                self.brain.action_mbon_indices[action]
                for action in self.action_order
            ],
            dtype=np.int64,
        )

        self.action_synapse_indices = np.flatnonzero(
            np.isin(self.post, action_posts)
        )

        cand_idx = self.action_synapse_indices
        cand_pre = self.pre[cand_idx]
        cand_post = self.post[cand_idx]

        # Precompute saved learning multipliers.
        mods = self.memory.get_weight_multipliers()

        learned_indices = []
        learned_multipliers = []

        for syn_idx, pre_idx, post_idx in zip(
            cand_idx,
            cand_pre,
            cand_post,
        ):
            mult = mods.get(
                (int(pre_idx), int(post_idx))
            )

            if mult is not None and not np.isclose(mult, 1.0):
                learned_indices.append(int(syn_idx))
                learned_multipliers.append(float(mult))

        self.learned_indices = np.asarray(
            learned_indices,
            dtype=np.int64,
        )

        self.learned_multipliers = np.asarray(
            learned_multipliers,
            dtype=float,
        )

        # Precompute calibration synapse indices.
        self.calibration_plan = {}

        for state_key in self.brain.real_map:
            per_action = {}

            for action in self.action_order:
                post_idx = self.brain.action_mbon_indices[action]

                kcs = np.asarray(
                    [
                        int(c["kc_index"])
                        for c in self.brain.real_map[
                            state_key
                        ][action]["connections"]
                    ],
                    dtype=np.int64,
                )

                mask = (
                    np.isin(cand_pre, kcs)
                    &
                    (cand_post == post_idx)
                )

                indices = cand_idx[mask]

                factor = float(
                    self.calibration[
                        state_key
                    ][action]["factor"]
                )

                per_action[action] = (
                    indices,
                    factor,
                )

            self.calibration_plan[state_key] = per_action

        # Save pristine network once.
        self.net.store("pristine")

        print(
            "Precomputed learned action-MBON synapses:",
            len(self.learned_indices),
        )

        # Cache unlearned baselines for ALL 8 states.
        print("Caching unlearned MBON baselines for all 8 states...")
        baseline_start = time.perf_counter()

        self.baseline_cache = {}

        for state in STATES:
            key = self._state_key(state)

            peak_g = self._run_once(
                state,
                learned=False,
            )

            self.baseline_cache[key] = peak_g

            print(
                f"  {state}:",
                {
                    k: round(v, 3)
                    for k, v in peak_g.items()
                },
            )

        elapsed = time.perf_counter() - baseline_start

        print(
            f"Baseline cache ready in {elapsed:.2f}s."
        )
        print(
            "Live policy ready: only ONE Brian2 simulation per move."
        )

    @staticmethod
    def _state_key(state):
        return f"{int(state[0])},{int(state[1])}"

    def _restore_pristine(self):
        self.net.restore("pristine")

    def _apply_calibration(self, state):
        state_key = self._state_key(state)

        if state_key not in self.calibration_plan:
            raise ValueError(
                f"Unknown state {state}. Expected one of the 8 "
                "non-zero direction states."
            )

        for action in self.action_order:
            indices, factor = (
                self.calibration_plan[
                    state_key
                ][action]
            )

            if len(indices) > 0:
                self.syn.w[indices] = (
                    self.syn.w[indices]
                    * factor
                )

    def _apply_learning(self):
        if len(self.learned_indices) == 0:
            return

        self.syn.w[
            self.learned_indices
        ] = (
            self.syn.w[
                self.learned_indices
            ]
            * self.learned_multipliers
        )

    def _activate_state_kcs(self, state):
        for kc_idx in self.brain.get_state_kcs(state):
            self.neu.v[int(kc_idx)] = -44 * mV

    def _run_once(self, state, learned):
        self._restore_pristine()
        self._apply_calibration(state)

        if learned:
            self._apply_learning()

        self._activate_state_kcs(state)

        self.net.run(
            self.run_time,
        )

        peak_g = {}

        for row, action in enumerate(self.action_order):
            g = np.asarray(
                self.state_monitor.g[row] / mV,
                dtype=float,
            )

            peak_g[action] = (
                float(np.max(g))
                if len(g)
                else 0.0
            )

        return peak_g

    def get_scores(self, state):
        state_key = self._state_key(state)

        if state_key not in self.baseline_cache:
            raise ValueError(
                f"Unknown state {state}."
            )

        learned_peak = self._run_once(
            state,
            learned=True,
        )

        baseline = self.baseline_cache[
            state_key
        ]

        return {
            action: (
                learned_peak[action]
                - baseline[action]
            )
            for action in self.action_order
        }

    def choose_action(self, state, epsilon=0.0):
        start = time.perf_counter()

        scores = self.get_scores(state)

        action = max(
            scores,
            key=scores.get,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            "Live MBON Δpeak_g:",
            {
                k: round(v, 3)
                for k, v in scores.items()
            },
        )

        print(
            f"Live neural decision: {action} "
            f"({elapsed:.2f}s)"
        )

        return action, scores

    def learn(self, state, action, reward):
        return None

    def reset_flythia(self):
        raise RuntimeError(
            "This live policy uses the trained fly_memory.json. "
            "Do not reset it during gameplay."
        )
