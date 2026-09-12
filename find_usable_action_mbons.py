import json
import pandas as pd
from pathlib import Path


BASE = Path(__file__).resolve().parent


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


# Convert every state pattern to FlyWire IDs
state_kcs = {}

for state, pattern in patterns.items():

    state_kcs[state] = {
        i2flyid[idx]
        for idx in pattern["kc_indices"]
    }


# ------------------------------------------------
# Load mushroom body MBONs
# ------------------------------------------------

with open(
    BASE / "data" / "mushroom_body_neurons.json",
    "r",
    encoding="utf-8"
) as f:
    mb = json.load(f)


mbon_names = {}

for name, ids in mb["mbon"].items():

    for neuron_id in ids:
        mbon_names[int(neuron_id)] = name


# ------------------------------------------------
# Connectivity
# ------------------------------------------------

print("Loading connectivity...")

con = pd.read_parquet(
    BASE / "data" / "2025_Connectivity_783.parquet"
)


# Only KC -> MBON connections are relevant
all_pattern_kcs = set()

for ids in state_kcs.values():
    all_pattern_kcs.update(ids)


kc_mbon = con[
    con["Presynaptic_ID"].isin(all_pattern_kcs)
    &
    con["Postsynaptic_ID"].isin(mbon_names.keys())
].copy()


# ------------------------------------------------
# Score every MBON
# ------------------------------------------------

results = []


for mbon_id, mbon_name in mbon_names.items():

    state_counts = {}
    state_weights = {}

    for state, kc_ids in state_kcs.items():

        rows = kc_mbon[
            kc_mbon["Presynaptic_ID"].isin(kc_ids)
            &
            (kc_mbon["Postsynaptic_ID"] == mbon_id)
        ]

        count = rows[
            "Presynaptic_ID"
        ].nunique()

        weight = (
            rows["Connectivity"].sum()
            if len(rows)
            else 0
        )

        state_counts[state] = int(count)
        state_weights[state] = float(weight)

    # Important:
    # an action MBON should be reachable from EVERY state
    min_count = min(
        state_counts.values()
    )

    avg_count = sum(
        state_counts.values()
    ) / len(state_counts)

    total_weight = sum(
        state_weights.values()
    )

    results.append(
        {
            "id": mbon_id,
            "name": mbon_name,
            "min_count": min_count,
            "avg_count": avg_count,
            "total_weight": total_weight,
            "counts": state_counts,
        }
    )


# Prioritize:
# 1. MBON has inputs from every state
# 2. good average KC coverage
# 3. strong total connectivity

results.sort(
    key=lambda r: (
        r["min_count"],
        r["avg_count"],
        r["total_weight"],
    ),
    reverse=True
)


# ------------------------------------------------
# Print best candidates
# ------------------------------------------------

print()
print("=" * 100)
print("BEST ACTION-MBON CANDIDATES")
print("=" * 100)


for rank, result in enumerate(
    results[:20],
    start=1
):

    print()
    print(
        f"{rank:2d}. "
        f"{result['name']} "
        f"id={result['id']}"
    )

    print(
        f"    minimum connected KCs/state: "
        f"{result['min_count']}"
    )

    print(
        f"    average connected KCs/state: "
        f"{result['avg_count']:.1f}"
    )

    print(
        f"    total weight: "
        f"{result['total_weight']:.0f}"
    )

    print(
        "    state coverage:"
    )

    for state, count in (
        result["counts"].items()
    ):

        print(
            f"        {state:5s}: "
            f"{count:2d}/60"
        )