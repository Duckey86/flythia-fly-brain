import json
import pandas as pd
from pathlib import Path


BASE = Path(__file__).resolve().parent


ACTION_MBONS = {
    "LEFT":  720575940639697827,
    "RIGHT": 720575940621828443,
    "UP":    720575940624539284,
    "DOWN":  720575940607155890,
}


# ------------------------------------------------
# Load anatomical KC patterns
# ------------------------------------------------

with open(
    BASE / "anatomical_kc_patterns.json",
    "r",
    encoding="utf-8"
) as f:
    patterns = json.load(f)


# ------------------------------------------------
# Brian index -> FlyWire ID
# ------------------------------------------------

comp = pd.read_csv(
    BASE / "data" / "2025_Completeness_783.csv",
    index_col=0
)

i2flyid = {
    i: int(fid)
    for i, fid in enumerate(comp.index)
}


# ------------------------------------------------
# Connectivity
# ------------------------------------------------

print("Loading connectivity...")

con = pd.read_parquet(
    BASE / "data" / "2025_Connectivity_783.parquet"
)


print()
print("=" * 90)
print("REAL KC -> ACTION MBON CONNECTIONS")
print("=" * 90)


for state, pattern in patterns.items():

    kc_indices = pattern["kc_indices"]

    kc_ids = [
        i2flyid[idx]
        for idx in kc_indices
    ]

    print()
    print("STATE:", state)
    print(
        "Input:",
        pattern["cell_type"],
        pattern["input_neuron"]
    )

    for action, mbon_id in ACTION_MBONS.items():

        rows = con[
            con["Presynaptic_ID"].isin(kc_ids)
            &
            (con["Postsynaptic_ID"] == mbon_id)
        ]

        connected_kcs = (
            rows["Presynaptic_ID"].nunique()
        )

        total_weight = (
            rows["Connectivity"].sum()
            if len(rows)
            else 0
        )

        print(
            f"  {action:5s} "
            f"connected KCs={connected_kcs:2d}/60   "
            f"weight={total_weight}"
        )