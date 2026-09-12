import json
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent

# -------------------------
# Load atlas
# -------------------------

with open(BASE / "neuron_atlas.json", "r", encoding="utf-8") as f:
    atlas = json.load(f)

outputs = atlas["output_neurons"]

# Motor neurons we care about
targets = {
    "TURN_LEFT": outputs["DNa02_left"]["id"],
    "TURN_RIGHT": outputs["DNa02_right"]["id"],
    "FORWARD_LEFT": outputs["P9_oDN1_left"]["id"],
    "FORWARD_RIGHT": outputs["P9_oDN1_right"]["id"],
}

# -------------------------
# Load connectivity
# -------------------------

con = pd.read_parquet(
    BASE / "data" / "2025_Connectivity_783.parquet"
)

# Optional annotations
annotation_path = BASE / "data" / "flywire_annotations.tsv"

if annotation_path.exists():
    annot = pd.read_csv(
        annotation_path,
        sep="\t",
        low_memory=False
    )

    type_map = dict(
        zip(
            annot["root_id"],
            annot["cell_type"]
        )
    )
else:
    type_map = {}


# -------------------------
# Find strongest direct inputs
# -------------------------

for name, target_id in targets.items():

    print()
    print("=" * 80)
    print(name, target_id)
    print("=" * 80)

    rows = con[
        con["Postsynaptic_ID"] == target_id
    ].copy()

    rows = rows.sort_values(
        "Connectivity",
        ascending=False
    ).head(30)

    for _, row in rows.iterrows():

        pre = int(row["Presynaptic_ID"])
        strength = row["Connectivity"]

        cell_type = type_map.get(
            pre,
            "unknown"
        )

        print(
            f"{pre}   "
            f"weight={strength:<8}   "
            f"type={cell_type}"
        )