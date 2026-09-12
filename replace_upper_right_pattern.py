import json
import pandas as pd
from pathlib import Path


BASE = Path(__file__).resolve().parent

PN_ID = 720575940620437339
STATE = "1,-1"
TOP_KCS = 60


with open(
    BASE / "anatomical_kc_patterns.json",
    "r",
    encoding="utf-8"
) as f:
    patterns = json.load(f)


con = pd.read_parquet(
    BASE / "data" / "2025_Connectivity_783.parquet"
)

comp = pd.read_csv(
    BASE / "data" / "2025_Completeness_783.csv",
    index_col=0
)

flyid2i = {
    int(fid): i
    for i, fid in enumerate(comp.index)
}


# Get this PN's real KC targets
rows = con[
    con["Presynaptic_ID"] == PN_ID
]


# Load KC IDs
with open(
    BASE / "data" / "mushroom_body_neurons.json",
    "r",
    encoding="utf-8"
) as f:
    mb = json.load(f)


kc_ids = set()

for ids in mb["kenyon_cells"].values():
    kc_ids.update(ids)


rows = rows[
    rows["Postsynaptic_ID"].isin(kc_ids)
]


ranked = (
    rows
    .groupby("Postsynaptic_ID")["Connectivity"]
    .sum()
    .sort_values(ascending=False)
)


chosen_fly_ids = [
    int(fid)
    for fid in ranked.head(TOP_KCS).index
]

chosen_indices = [
    flyid2i[fid]
    for fid in chosen_fly_ids
]


patterns[STATE] = {
    "input_neuron": PN_ID,
    "cell_type": "DM6_adPN",
    "kc_indices": sorted(chosen_indices),
}


with open(
    BASE / "anatomical_kc_patterns.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        patterns,
        f,
        indent=2
    )


print(
    f"Replaced state ({STATE}) "
    f"with DM6_adPN {PN_ID}"
)

print(
    "KC count:",
    len(chosen_indices)
)