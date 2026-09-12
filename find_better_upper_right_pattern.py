import json
import pandas as pd
from pathlib import Path

from fly_mb_policy import ACTION_MBONS


BASE = Path(__file__).resolve().parent

PATTERN_FILE = BASE / "anatomical_kc_patterns.json"
COMP_FILE = BASE / "data" / "2025_Completeness_783.csv"
CON_FILE = BASE / "data" / "2025_Connectivity_783.parquet"
ANNOTATION_FILE = BASE / "data" / "flywire_annotations.tsv"
MB_FILE = BASE / "data" / "mushroom_body_neurons.json"

TARGET_STATE = "1,-1"
TOP_KCS = 60


# ------------------------------------------------
# Load current patterns
# ------------------------------------------------

with open(
    PATTERN_FILE,
    "r",
    encoding="utf-8"
) as f:
    current_patterns = json.load(f)


# ------------------------------------------------
# Load KC IDs
# ------------------------------------------------

with open(
    MB_FILE,
    "r",
    encoding="utf-8"
) as f:
    mb = json.load(f)


kc_ids = set()

for ids in mb["kenyon_cells"].values():
    kc_ids.update(ids)


# ------------------------------------------------
# Brian index mappings
# ------------------------------------------------

comp = pd.read_csv(
    COMP_FILE,
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


# ------------------------------------------------
# Existing 7 patterns
# ------------------------------------------------

existing_patterns = {}

for state, info in current_patterns.items():

    if state == TARGET_STATE:
        continue

    existing_patterns[state] = {
        i2flyid[idx]
        for idx in info["kc_indices"]
    }


# ------------------------------------------------
# Annotations
# ------------------------------------------------

annotations = pd.read_csv(
    ANNOTATION_FILE,
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
# Connectivity
# ------------------------------------------------

print("Loading connectivity...")

con = pd.read_parquet(
    CON_FILE
)


# Direct PN -> KC connections
pn_kc = con[
    con["Postsynaptic_ID"].isin(kc_ids)
].copy()


# ------------------------------------------------
# Pattern overlap
# ------------------------------------------------

def overlap(a, b):

    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


# ------------------------------------------------
# Search candidates
# ------------------------------------------------

results = []


for pre_id, group in pn_kc.groupby(
    "Presynaptic_ID"
):

    pre_id = int(pre_id)

    cell_type = type_map.get(
        pre_id,
        ""
    )

    if not isinstance(cell_type, str):
        continue

    # Projection-neuron-like inputs only
    if not (
        "PN" in cell_type
        or "adPN" in cell_type
        or "lPN" in cell_type
    ):
        continue

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

    chosen_kcs = {
        int(x)
        for x in ranked.head(
            TOP_KCS
        ).index
    }

    # -----------------------------------------
    # Must remain distinct from the other
    # seven state patterns
    # -----------------------------------------

    max_overlap = max(
        overlap(
            chosen_kcs,
            existing
        )
        for existing in existing_patterns.values()
    )

    if max_overlap > 0.25:
        continue

    # -----------------------------------------
    # Calculate actual Brian2 drive to each
    # action MBON
    # -----------------------------------------

    drives = {}

    valid = True

    for action, mbon_id in ACTION_MBONS.items():

        rows = con[
            con["Presynaptic_ID"].isin(
                chosen_kcs
            )
            &
            (
                con["Postsynaptic_ID"]
                == mbon_id
            )
        ]

        drive = float(
            rows[
                "Excitatory x Connectivity"
            ].sum()
        )

        drives[action] = drive

        if drive <= 0:
            valid = False

    if not valid:
        continue

    smallest = min(
        drives.values()
    )

    largest = max(
        drives.values()
    )

    ratio = (
        largest / smallest
    )

    results.append({
        "input_id": pre_id,
        "cell_type": cell_type,
        "kc_ids": chosen_kcs,
        "drives": drives,
        "ratio": ratio,
        "max_overlap": max_overlap,
    })


# ------------------------------------------------
# Best = lowest baseline imbalance
# ------------------------------------------------

results.sort(
    key=lambda x: (
        x["ratio"],
        x["max_overlap"]
    )
)


print()
print("=" * 100)
print("BEST REPLACEMENTS FOR STATE (1,-1)")
print("=" * 100)


for rank, result in enumerate(
    results[:15],
    start=1
):

    print()
    print(
        f"{rank}. "
        f"{result['cell_type']} "
        f"id={result['input_id']}"
    )

    print(
        f"   baseline ratio: "
        f"{result['ratio']:.2f}x"
    )

    print(
        f"   max overlap with other states: "
        f"{result['max_overlap']:.3f}"
    )

    print(
        "   drives:",
        {
            action: round(value, 1)
            for action, value
            in result["drives"].items()
        }
    )