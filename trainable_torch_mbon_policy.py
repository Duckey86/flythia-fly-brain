import random
import os
import json

from pathlib import Path

import pandas as pd
import torch

BASE = Path(__file__).resolve().parent

MEMORY_FILE = (
    BASE
    / "data"
    / "fly_memory.json"
)

from live_torch_mbon_policy import (
    LiveTorchMBONPolicy,
    CON_FILE,
)

from fly_mb_policy import (
    FlyMBPolicy,
    ACTIONS,
    STATES,
)

class TrainableTorchMBONPolicy(
    LiveTorchMBONPolicy
):

    def __init__(
        self,
        run_time_ms=3,
        device="cuda",
    ):

        # Build the SAME full neural CUDA model
        # that we've already validated.
        super().__init__(
            run_time_ms=run_time_ms,
            device=device,
        )

        print(
            "Adding online KC->MBON plasticity..."
        )

        # Use the safer learner:
        # unique KCs only, bounded weights.
        self.brain = FlyMBPolicy()

        # IMPORTANT:
        # same persistent memory object used
        # by learning.
        self.memory = self.brain.memory
        self.kc_usage_count = {}

        for pattern in self.brain.anatomical_patterns.values():

            for kc_idx in pattern["kc_indices"]:

                self.kc_usage_count[kc_idx] = (
                    self.kc_usage_count.get(
                        kc_idx,
                        0
                    )
                    + 1
                )


        # We need biological base weights so
        # learned GPU correction tensors can
        # be rebuilt immediately.
        df = pd.read_parquet(
            CON_FILE
        )

        self.pair_weight = {}

        for pre, post, value in zip(

            df[
                "Presynaptic_Index"
            ].to_numpy(),

            df[
                "Postsynaptic_Index"
            ].to_numpy(),

            df[
                "Excitatory x Connectivity"
            ].to_numpy(),
        ):

            pair = (
                int(pre),
                int(post),
            )

            self.pair_weight[pair] = (
                self.pair_weight.get(
                    pair,
                    0.0,
                )
                + float(value)
            )


        # Rebuild all learned plans once so
        # they're guaranteed to correspond
        # to the fixed plasticity memory.
        for state in STATES:

            for action in ACTIONS:

                self._rebuild_action(
                    state,
                    action,
                )


        print(
            "Trainable CUDA fly ready."
        )


    # -------------------------------------------------
    # REBUILD ONE GPU KC -> MBON LEARNING PATH
    # -------------------------------------------------

    def _rebuild_action(
        self,
        state,
        action,
    ):

        state_key = self._key(
            state
        )

        post_idx = int(
            self.brain
            .action_mbon_indices[
                action
            ]
        )

        calibration = float(
            self.calibration[
                state_key
            ][action]["factor"]
        )


        # This converts fly_memory string keys
        # into (pre, post) tuples.
        mods = (
            self.memory
            .get_weight_multipliers()
        )


        pres = []
        deltas = []


        connections = (
            self.brain
            .real_map[
                state_key
            ][action][
                "connections"
            ]
        )


        for connection in connections:

            pre_idx = int(
                connection[
                    "kc_index"
                ]
            )

            biological_weight = (
                self.pair_weight.get(
                    (
                        pre_idx,
                        post_idx,
                    ),
                    0.0,
                )
            )


            if biological_weight == 0:
                continue


            learned_multiplier = float(

                mods.get(
                    (
                        pre_idx,
                        post_idx,
                    ),
                    1.0,
                )

            )


            # Existing sparse connectome already
            # contributes biological_weight.
            #
            # This correction changes it into:
            #
            # biological_weight
            # * calibration
            # * learned_multiplier

            corrected = (

                biological_weight
                * calibration
                * learned_multiplier

            )


            delta = (
                corrected
                - biological_weight
            )


            pres.append(
                pre_idx
            )

            deltas.append(
                delta
            )


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


        # THIS tensor is used during the next
        # full neural simulation.

        self.learned_plan[
            state_key
        ][action] = (

            pre_tensor,
            delta_tensor,
            post_idx,

        )


    # -------------------------------------------------
    # NEURAL DECISION + EXPLORATION
    # -------------------------------------------------

    def choose_action(
        self,
        state,
        epsilon=0.0,
    ):

        # ALWAYS run the actual neural model first.
        scores = self.get_scores(
            state
        )


        # Occasionally explore during training.
        if random.random() < epsilon:

            action = random.choice(
                ACTIONS
            )

        else:

            action = max(
                scores,
                key=scores.get,
            )


        return (
            action,
            scores,
        )


    # -------------------------------------------------
    # ACTUAL ONLINE PLASTICITY
    # -------------------------------------------------

    def learn(
        self,
        state,
        action,
        reward,
    ):

        if state not in STATES:
            raise ValueError(
                f"Invalid state: {state}"
            )

        if action not in ACTIONS:
            raise ValueError(
                f"Invalid action: {action}"
            )


        # Gentle online plasticity
        if reward > 0:

            multiplier = 1.01
            signal = "reward"

        else:

            multiplier = 0.99
            signal = "punishment"


        state_key = self._key(
            state
        )


        connections = (
            self.brain
            .real_map[
                state_key
            ][action][
                "connections"
            ]
        )


        mbon_idx = int(
            self.brain
            .action_mbon_indices[
                action
            ]
        )


        mods = (
            self.memory.memory[
                "weight_modifications"
            ]
        )


        changed = 0
        skipped = 0


        for connection in connections:

            kc_idx = int(
                connection[
                    "kc_index"
                ]
            )


            # Only state-unique KCs learn
            if (
                self.kc_usage_count.get(
                    kc_idx,
                    0
                )
                != 1
            ):

                skipped += 1
                continue


            memory_key = (
                f"{kc_idx}:{mbon_idx}"
            )


            old = float(
                mods.get(
                    memory_key,
                    1.0
                )
            )


            new = (
                old
                * multiplier
            )


            new = max(
                0.25,
                min(
                    4.0,
                    new
                )
            )


            mods[memory_key] = round(
                new,
                5
            )


            changed += 1


        # Update GPU weights immediately
        self._rebuild_action(
            state,
            action,
        )


        if self.device.type == "cuda":

            torch.cuda.synchronize()


        return {

            "signal_type":
                signal,

            "multiplier":
                multiplier,

            "changed":
                changed,

            "skipped":
                skipped,
        }
    
    def save(self):

        temp_file = (
            MEMORY_FILE.with_suffix(
                ".json.tmp"
            )
        )


        with open(
            temp_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.memory.memory,
                f,
                indent=2
            )

            f.flush()

            os.fsync(
                f.fileno()
            )


        os.replace(
            temp_file,
            MEMORY_FILE
        )