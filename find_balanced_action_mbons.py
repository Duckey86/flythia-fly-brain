import json
import itertools
import pandas as pd
from pathlib import Path


BASE = Path(__file__).resolve().parent

PATTERN_FILE = BASE / "anatomical_kc_patterns.json"
MB_FILE = BASE / "data" / "mushroom_body_neurons.json"
COMP_FILE = BASE / "data" / "2025_Completeness_783.csv"
CON_FILE = BASE / "data" / "2025_Connectivity_783.parquet"


# ------------------------------------------------
# Load state KC patterns
# ------------------------------------------------

with open(
    PATTERN_FILE,
    "r",
    encoding="utf-8"
) as f:
    patterns = json.load(f)


comp = pd.read_csv(
    COMP_FILE,
    index_col=0
)

i2flyid = {
    i: int(fid)
    for i, fid in enumerate(comp.index)
}


state_kcs = {}

for state, info in patterns.items():

    state_kcs[state] = {
        i2flyid[idx]
        for idx in info["kc_indices"]
    }


# ------------------------------------------------
# Load MBON IDs
# ------------------------------------------------

with open(
    MB_FILE,
    "r",
    encoding="utf-8"
) as f:
    mb = json.load(f)


mbon_names = {}

for name, ids in mb["mbon"].items():
    for fid in ids:
        mbon_names[int(fid)] = name


# ------------------------------------------------
# Load REAL signed Brian2 weights
# ------------------------------------------------

print("Loading connectivity...")

con = pd.read_parquet(
    CON_FILE
)


all_kcs = set()

for ids in state_kcs.values():
    all_kcs.update(ids)


kc_mbon = con[
    con["Presynaptic_ID"].isin(all_kcs)
    &
    con["Postsynaptic_ID"].isin(mbon_names)
].copy()


# ------------------------------------------------
# Compute baseline drive for every MBON/state
# ------------------------------------------------

candidates = []


for mbon_id, name in mbon_names.items():

    drives = {}
    counts = {}

    valid = True

    for state, kc_ids in state_kcs.items():

        rows = kc_mbon[
            kc_mbon["Presynaptic_ID"].isin(kc_ids)
            &
            (kc_mbon["Postsynaptic_ID"] == mbon_id)
        ]

        count = rows[
            "Presynaptic_ID"
        ].nunique()

        # EXACT signed value Brian2 uses
        drive = rows[
            "Excitatory x Connectivity"
        ].sum()

        counts[state] = int(count)
        drives[state] = float(drive)

        # We want each candidate reachable
        # with positive excitatory drive in all 8 states
        if count == 0 or drive <= 0:
            valid = False

    if not valid:
        continue

    candidates.append({
        "id": mbon_id,
        "name": name,
        "drives": drives,
        "counts": counts,
        "min_count": min(counts.values()),
        "mean_drive": (
            sum(drives.values())
            / len(drives)
        ),
    })


print(
    "Usable MBON candidates:",
    len(candidates)
)


# Prefer candidates with decent anatomical coverage
candidates.sort(
    key=lambda x: (
        x["min_count"],
        x["mean_drive"]
    ),
    reverse=True
)

# Keep search manageable
POOL = candidates[:20]


# ------------------------------------------------
# Score quartets
# ------------------------------------------------

def quartet_score(group):

    ratios = []
    absolute_spreads = []

    for state in state_kcs:

        values = [
            x["drives"][state]
            for x in group
        ]

        smallest = min(values)
        largest = max(values)

        ratio = largest / smallest

        ratios.append(ratio)

        absolute_spreads.append(
            largest - smallest
        )

    # Lower is better
    worst_ratio = max(ratios)
    mean_ratio = sum(ratios) / len(ratios)

    return (
        worst_ratio,
        mean_ratio
    )


results = []

for group in itertools.combinations(
    POOL,
    4
):

    worst_ratio, mean_ratio = (
        quartet_score(group)
    )

    results.append(
        (
            worst_ratio,
            mean_ratio,
            group
        )
    )


results.sort(
    key=lambda x: (
        x[0],
        x[1]
    )
)


# ------------------------------------------------
# Print best quartets
# ------------------------------------------------

print()
print("=" * 100)
print("BEST BALANCED MBON QUARTETS")
print("=" * 100)


for rank, (
    worst_ratio,
    mean_ratio,
    group
) in enumerate(
    results[:10],
    start=1
):

    print()
    print(
        f"QUARTET {rank}"
    )

    print(
        f"Worst baseline ratio: "
        f"{worst_ratio:.2f}x"
    )

    print(
        f"Average baseline ratio: "
        f"{mean_ratio:.2f}x"
    )

    for mbon in group:

        print(
            f"  {mbon['name']:8s} "
            f"id={mbon['id']} "
            f"minKC={mbon['min_count']}"
        )

    print("State drives:")

    for state in state_kcs:

        values = [
            round(
                mbon["drives"][state],
                1
            )
            for mbon in group
        ]

        print(
            f"  {state:5s}: "
            f"{values}"
        )