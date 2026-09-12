import json
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent

ACTION_MBONS = {
    "LEFT":  720575940617552340,
    "RIGHT": 720575940624185095,
    "UP":    720575940623201833,
    "DOWN":  720575940629422086,
}

with open(
    BASE / "anatomical_kc_patterns.json",
    "r",
    encoding="utf-8"
) as f:
    patterns = json.load(f)

comp = pd.read_csv(
    BASE / "data" / "2025_Completeness_783.csv",
    index_col=0
)

i2flyid = {
    i: int(fid)
    for i, fid in enumerate(comp.index)
}

flyid2i = {
    int(fid): i
    for i, fid in enumerate(comp.index)
}

print("Loading connectivity...")

con = pd.read_parquet(
    BASE / "data" / "2025_Connectivity_783.parquet"
)

output = {}

for state, pattern in patterns.items():

    output[state] = {}

    kc_indices = pattern["kc_indices"]

    kc_ids = {
        i2flyid[idx]
        for idx in kc_indices
    }

    print()
    print("STATE", state)

    for action, mbon_id in ACTION_MBONS.items():

        rows = con[
            con["Presynaptic_ID"].isin(kc_ids)
            &
            (con["Postsynaptic_ID"] == mbon_id)
        ]

        # Combine duplicate edges from same KC
        grouped = (
            rows
            .groupby("Presynaptic_ID")[
                "Connectivity"
            ]
            .sum()
        )

        connections = []

        for kc_fly_id, weight in grouped.items():

            kc_fly_id = int(kc_fly_id)

            connections.append({
                "kc_index": flyid2i[kc_fly_id],
                "weight": float(weight),
            })

        output[state][action] = {
            "mbon_id": mbon_id,
            "connections": connections,
        }

        print(
            f"{action:5s}: "
            f"{len(connections):2d} real KC→MBON synapses, "
            f"weight={sum(x['weight'] for x in connections):.0f}"
        )

with open(
    BASE / "real_kc_mbon_map.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2
    )

print()
print("Saved real_kc_mbon_map.json")