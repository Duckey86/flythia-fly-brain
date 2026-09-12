import json
import pandas as pd
from pathlib import Path


BASE = Path(__file__).resolve().parent


# ------------------------------------------------
# Load mushroom-body neuron IDs
# ------------------------------------------------

with open(
    BASE / "data" / "mushroom_body_neurons.json",
    "r",
    encoding="utf-8"
) as f:
    mb = json.load(f)


kc_ids = set()

for ids in mb["kenyon_cells"].values():
    kc_ids.update(ids)


print(
    "Kenyon cells:",
    len(kc_ids)
)


# ------------------------------------------------
# Load connectivity
# ------------------------------------------------

print("Loading connectivity...")

con = pd.read_parquet(
    BASE / "data" / "2025_Connectivity_783.parquet"
)


# ------------------------------------------------
# Find ALL neurons directly feeding KCs
# ------------------------------------------------

print("Finding direct KC inputs...")

kc_inputs = con[
    con["Postsynaptic_ID"].isin(kc_ids)
].copy()


print(
    "KC input connections:",
    len(kc_inputs)
)


# ------------------------------------------------
# Sum connection strength per presynaptic neuron
# ------------------------------------------------

ranked = (
    kc_inputs
    .groupby("Presynaptic_ID")
    .agg(
        total_weight=(
            "Connectivity",
            "sum"
        ),

        connections=(
            "Connectivity",
            "count"
        ),

        kc_targets=(
            "Postsynaptic_ID",
            "nunique"
        )
    )
    .reset_index()
    .sort_values(
        "total_weight",
        ascending=False
    )
)


# ------------------------------------------------
# Load annotations
# ------------------------------------------------

annotation_file = (
    BASE
    / "data"
    / "flywire_annotations.tsv"
)

annotations = pd.read_csv(
    annotation_file,
    sep="\t",
    low_memory=False
)

type_map = dict(
    zip(
        annotations["root_id"],
        annotations["cell_type"]
    )
)


# ------------------------------------------------
# Remove mushroom-body neurons themselves
# ------------------------------------------------

all_mb_ids = set()

for group in mb["kenyon_cells"].values():
    all_mb_ids.update(group)

for group in mb["mbon"].values():
    all_mb_ids.update(group)

for group in mb[
    "dan_pam_reward"
].values():
    all_mb_ids.update(group)

for group in mb[
    "dan_ppl_punishment"
].values():
    all_mb_ids.update(group)


ranked = ranked[
    ~ranked[
        "Presynaptic_ID"
    ].isin(all_mb_ids)
]


# ------------------------------------------------
# Print strongest candidates
# ------------------------------------------------

print()
print("=" * 100)
print("STRONGEST NON-MB INPUTS TO KENYON CELLS")
print("=" * 100)

for _, row in ranked.head(50).iterrows():

    neuron_id = int(
        row["Presynaptic_ID"]
    )

    neuron_type = type_map.get(
        neuron_id,
        "unknown"
    )

    print(
        f"{neuron_id}   "
        f"weight={int(row['total_weight']):6d}   "
        f"KC targets={int(row['kc_targets']):4d}   "
        f"connections={int(row['connections']):4d}   "
        f"type={neuron_type}"
    )