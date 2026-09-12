import json
import pandas as pd
from pathlib import Path

from fly_mb_policy import (
    FlyMBPolicy,
    ACTIONS,
    STATES,
)


BASE = Path(__file__).resolve().parent

CON_FILE = (
    BASE
    / "data"
    / "2025_Connectivity_783.parquet"
)

OUTPUT_FILE = (
    BASE
    / "mbon_calibration.json"
)


brain = FlyMBPolicy()


print("Loading connectivity...")

con = pd.read_parquet(
    CON_FILE
)


# ------------------------------------------------
# Exact signed weights used by Brian2
# ------------------------------------------------

weight_lookup = {}

for pre, post, weight in zip(
    con["Presynaptic_Index"],
    con["Postsynaptic_Index"],
    con["Excitatory x Connectivity"]
):

    pair = (
        int(pre),
        int(post)
    )

    weight_lookup[pair] = (
        weight_lookup.get(
            pair,
            0.0
        )
        + float(weight)
    )


calibration = {}


print()
print("=" * 80)
print("MBON HOMEOSTATIC CALIBRATION")
print("=" * 80)


for state in STATES:

    key = (
        f"{state[0]},{state[1]}"
    )

    baseline = {}

    # -----------------------------------------
    # Calculate baseline drive for each action
    # -----------------------------------------

    for action in ACTIONS:

        mbon_idx = (
            brain.action_mbon_indices[
                action
            ]
        )

        connections = (
            brain.real_map[
                key
            ][action][
                "connections"
            ]
        )

        drive = 0.0

        for connection in connections:

            kc_idx = connection[
                "kc_index"
            ]

            drive += weight_lookup.get(
                (
                    kc_idx,
                    mbon_idx
                ),
                0.0
            )

        baseline[action] = drive

    # -----------------------------------------
    # Use the mean drive as the target
    # -----------------------------------------

    positive = [
        x
        for x in baseline.values()
        if x > 0
    ]

    target = (
        sum(positive)
        / len(positive)
    )

    calibration[key] = {}

    print()
    print(
        "STATE",
        state
    )

    print(
        "Target baseline:",
        round(target, 2)
    )

    for action in ACTIONS:

        drive = baseline[
            action
        ]

        if drive <= 0:
            factor = 1.0
        else:
            factor = (
                target / drive
            )

        calibration[key][action] = {
            "baseline_drive": drive,
            "target_drive": target,
            "factor": factor,
        }

        print(
            f"{action:5s} "
            f"base={drive:7.2f} "
            f"factor={factor:7.3f} "
            f"calibrated="
            f"{drive * factor:7.2f}"
        )


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        calibration,
        f,
        indent=2
    )


print()
print(
    "Saved:",
    OUTPUT_FILE
)