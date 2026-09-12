import json
import pandas as pd
from pathlib import Path

from dopamine_learning import FlyMemory
from fly_mb_policy import (
    FlyMBPolicy,
    ACTION_MBONS,
    ACTIONS,
    STATES,
)


BASE = Path(__file__).resolve().parent

CON_FILE = (
    BASE
    / "data"
    / "2025_Connectivity_783.parquet"
)


brain = FlyMBPolicy()
memory = FlyMemory()

mods = memory.memory[
    "weight_modifications"
]


print("Loading connectivity...")

con = pd.read_parquet(
    CON_FILE
)


# ------------------------------------------------
# Build fast lookup using the EXACT weight
# Brian2 uses:
#
# Excitatory x Connectivity
# ------------------------------------------------

weight_lookup = {}

for pre, post, weight in zip(
    con["Presynaptic_Index"],
    con["Postsynaptic_Index"],
    con["Excitatory x Connectivity"]
):
    key = (
        int(pre),
        int(post)
    )

    weight_lookup[key] = (
        weight_lookup.get(key, 0.0)
        + float(weight)
    )


print()
print("=" * 90)
print("EFFECTIVE KC -> MBON DRIVE")
print("=" * 90)


for state in STATES:

    state_key = (
        f"{state[0]},{state[1]}"
    )

    expected, scores = (
        brain.choose_action(
            state,
            epsilon=0.0
        )
    )

    print()
    print("=" * 70)
    print("STATE:", state)
    print("EXPECTED:", expected)
    print("=" * 70)

    for action in ACTIONS:

        mbon_idx = (
            brain.action_mbon_indices[
                action
            ]
        )

        connections = (
            brain.real_map[
                state_key
            ][action][
                "connections"
            ]
        )

        baseline_drive = 0.0
        learned_drive = 0.0

        real_synapses = 0

        for connection in connections:

            kc_idx = connection[
                "kc_index"
            ]

            pair = (
                kc_idx,
                mbon_idx
            )

            base_weight = (
                weight_lookup.get(
                    pair,
                    0.0
                )
            )

            if base_weight == 0:
                continue

            real_synapses += 1

            memory_key = (
                f"{kc_idx}:{mbon_idx}"
            )

            multiplier = mods.get(
                memory_key,
                1.0
            )

            baseline_drive += (
                base_weight
            )

            learned_drive += (
                base_weight
                * multiplier
            )

        delta = (
            learned_drive
            - baseline_drive
        )

        if baseline_drive != 0:
            ratio = (
                learned_drive
                / baseline_drive
            )
        else:
            ratio = 0.0

        marker = (
            "  <-- EXPECTED"
            if action == expected
            else ""
        )

        print(
            f"{action:5s} | "
            f"synapses={real_synapses:2d} | "
            f"base={baseline_drive:8.2f} | "
            f"learned={learned_drive:8.2f} | "
            f"delta={delta:+8.2f} | "
            f"x{ratio:.2f}"
            f"{marker}"
        )