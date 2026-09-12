import json
import pandas as pd
from pathlib import Path


BASE = Path(__file__).resolve().parent

TOP_KCS = 60

STATES = [
    (-1, -1),
    (0, -1),
    (1, -1),

    (-1, 0),
    (1, 0),

    (-1, 1),
    (0, 1),
    (1, 1),
]


# ------------------------------------------------
# Load mushroom-body neuron IDs
# ------------------------------------------------

with open(
    BASE / "data" / "mushroom_body_neurons.json",
    "r",
    encoding="utf-8"
) as f:
    mb = json.load(f)


kc_fly_ids = set()

for ids in mb["kenyon_cells"].values():
    kc_fly_ids.update(ids)


# ------------------------------------------------
# FlyWire ID -> Brian neuron index
# ------------------------------------------------

comp = pd.read_csv(
    BASE / "data" / "2025_Completeness_783.csv",
    index_col=0
)

flyid2i = {
    int(fid): i
    for i, fid in enumerate(comp.index)
}


# ------------------------------------------------
# Connectivity
# ------------------------------------------------

print("Loading connectivity...")

con = pd.read_parquet(
    BASE / "data" / "2025_Connectivity_783.parquet"
)


# Only direct inputs to Kenyon cells
kc_con = con[
    con["Postsynaptic_ID"].isin(kc_fly_ids)
].copy()


# ------------------------------------------------
# Annotations
# ------------------------------------------------

annotations = pd.read_csv(
    BASE / "data" / "flywire_annotations.tsv",
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
# Build candidate PN patterns
# ------------------------------------------------

patterns = []

for pre_id, group in kc_con.groupby(
    "Presynaptic_ID"
):

    pre_id = int(pre_id)

    cell_type = type_map.get(
        pre_id,
        ""
    )

    if not isinstance(
        cell_type,
        str
    ):
        continue

    # Keep projection-neuron-like types
    if not (
        "PN" in cell_type
        or "adPN" in cell_type
        or "lPN" in cell_type
    ):
        continue

    # Merge repeated connections to same KC
    ranked = (
        group
        .groupby("Postsynaptic_ID")[
            "Connectivity"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    if len(ranked) < TOP_KCS:
        continue

    # Strongest anatomically connected KCs
    chosen_fly_ids = [
        int(x)
        for x in ranked.head(
            TOP_KCS
        ).index
    ]

    chosen_indices = {
        flyid2i[fid]
        for fid in chosen_fly_ids
        if fid in flyid2i
    }

    if len(chosen_indices) < TOP_KCS:
        continue

    patterns.append(
        {
            "input_id": pre_id,
            "cell_type": cell_type,
            "kc_indices": chosen_indices,
            "total_weight": float(
                ranked.head(
                    TOP_KCS
                ).sum()
            ),
        }
    )


print(
    "Candidate projection neurons:",
    len(patterns)
)


# ------------------------------------------------
# Jaccard similarity
# ------------------------------------------------

def overlap(a, b):

    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


# ------------------------------------------------
# Greedily choose 8 DISTINCT patterns
# ------------------------------------------------

patterns.sort(
    key=lambda x: x["total_weight"],
    reverse=True
)

selected = []

for candidate in patterns:

    if not selected:

        selected.append(
            candidate
        )

        continue

    worst_overlap = max(
        overlap(
            candidate["kc_indices"],
            existing["kc_indices"]
        )
        for existing in selected
    )

    # Require reasonably distinct KC populations
    if worst_overlap <= 0.25:

        selected.append(
            candidate
        )

    if len(selected) == 8:
        break


if len(selected) < 8:

    raise RuntimeError(
        f"Only found {len(selected)} "
        f"sufficiently distinct patterns."
    )


# ------------------------------------------------
# Assign selected PNs to the 8 game states
# ------------------------------------------------

output = {}

print()
print("=" * 80)
print("SELECTED ANATOMICAL KC PATTERNS")
print("=" * 80)


for state, candidate in zip(
    STATES,
    selected
):

    key = f"{state[0]},{state[1]}"

    output[key] = {
        "input_neuron": (
            candidate["input_id"]
        ),
        "cell_type": (
            candidate["cell_type"]
        ),
        "kc_indices": sorted(
            candidate["kc_indices"]
        ),
    }

    print()
    print(
        state,
        "->",
        candidate["cell_type"],
        candidate["input_id"]
    )

    print(
        "KC count:",
        len(
            candidate["kc_indices"]
        )
    )


# ------------------------------------------------
# Pairwise overlap
# ------------------------------------------------

print()
print("=" * 80)
print("PAIRWISE KC OVERLAP")
print("=" * 80)


for i in range(len(selected)):

    for j in range(
        i + 1,
        len(selected)
    ):

        sim = overlap(
            selected[i]["kc_indices"],
            selected[j]["kc_indices"]
        )

        print(
            f"{STATES[i]} vs "
            f"{STATES[j]}: "
            f"{sim:.3f}"
        )


# ------------------------------------------------
# Save
# ------------------------------------------------

output_file = (
    BASE
    / "anatomical_kc_patterns.json"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2
    )


print()
print(
    "Saved:",
    output_file
)