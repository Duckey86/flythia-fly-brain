import json
from pathlib import Path

BASE = Path(__file__).resolve().parent

SOURCE = (
    BASE
    / "priority_kc_subpatterns.json"
)

OUTPUT = (
    BASE
    / "priority_kc_subpatterns_16.json"
)


with open(
    SOURCE,
    "r",
    encoding="utf-8"
) as f:

    patterns = json.load(f)


output = {}


# --------------------------------
# CARDINAL STATES
# --------------------------------

CARDINALS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
]


for state in CARDINALS:

    base = (
        f"{state[0]},"
        f"{state[1]}"
    )

    merged = []

    for priority in [
        "X_DOMINANT",
        "BALANCED",
        "Y_DOMINANT",
    ]:

        key = (
            f"{base}|"
            f"{priority}"
        )

        merged.extend(
            patterns[
                key
            ][
                "kc_indices"
            ]
        )


    # remove duplicates, preserve order
    merged = list(
        dict.fromkeys(
            merged
        )
    )


    output[
        base
    ] = {

        "direction":
            list(state),

        "priority":
            "CARDINAL",

        "kc_indices":
            merged,
    }


# --------------------------------
# DIAGONAL STATES
# --------------------------------

DIAGONALS = [
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
]


for state in DIAGONALS:

    base = (
        f"{state[0]},"
        f"{state[1]}"
    )

    for priority in [
        "X_DOMINANT",
        "BALANCED",
        "Y_DOMINANT",
    ]:

        key = (
            f"{base}|"
            f"{priority}"
        )

        output[
            key
        ] = patterns[
            key
        ]


with open(
    OUTPUT,
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
    "16-state sensory map generated."
)

print(
    "Patterns:",
    len(output)
)

print(
    "Saved:",
    OUTPUT
)

print()


for key, info in output.items():

    print(
        f"{key:28s} "
        f"KCs={len(info['kc_indices'])}"
    )