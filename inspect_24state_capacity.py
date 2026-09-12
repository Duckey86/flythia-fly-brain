import json
import random
from pathlib import Path

from fly_mb_policy import (
    FlyMBPolicy,
    STATES,
    ACTIONS,
)

BASE = Path(__file__).resolve().parent

OUTPUT = (
    BASE
    / "priority_kc_subpatterns.json"
)

NUM_BUCKETS = 3
SEARCH_TRIES = 5000

PRIORITIES = [
    "X_DOMINANT",
    "BALANCED",
    "Y_DOMINANT",
]


brain = FlyMBPolicy()


# --------------------------------
# KC usage across direction states
# --------------------------------

usage = {}

for state in STATES:

    for kc in brain.get_state_kcs(
        state
    ):

        kc = int(kc)

        usage[kc] = (
            usage.get(kc, 0)
            + 1
        )


def state_key(state):

    return (
        f"{state[0]},"
        f"{state[1]}"
    )


def action_sets(state):

    key = state_key(
        state
    )

    return {

        action: {

            int(c["kc_index"])

            for c in (
                brain.real_map[
                    key
                ][action][
                    "connections"
                ]
            )

        }

        for action in ACTIONS

    }


def valid_actions(state):

    x, y = state

    result = set()

    if x < 0:
        result.add("LEFT")

    if x > 0:
        result.add("RIGHT")

    if y < 0:
        result.add("UP")

    if y > 0:
        result.add("DOWN")

    return result


def score_partition(
    buckets,
    sets,
    valid,
):

    # We care primarily that every bucket
    # can reach the CORRECT action MBONs.
    #
    # Missing an irrelevant/wrong MBON is
    # not automatically a problem.

    worst_valid = 999999

    for bucket in buckets:

        b = set(bucket)

        best_correct = max(

            len(
                b
                & sets[action]
            )

            for action
            in valid
        )

        worst_valid = min(
            worst_valid,
            best_correct,
        )

    return worst_valid


def partition(
    kcs,
    sets,
    valid,
):

    best = None
    best_score = -1

    rng = random.Random(
        54321
    )

    for _ in range(
        SEARCH_TRIES
    ):

        shuffled = list(kcs)

        rng.shuffle(
            shuffled
        )

        buckets = [
            []
            for _ in range(
                NUM_BUCKETS
            )
        ]

        for i, kc in enumerate(
            shuffled
        ):

            buckets[
                i % NUM_BUCKETS
            ].append(kc)

        score = score_partition(
            buckets,
            sets,
            valid,
        )

        if score > best_score:

            best_score = score
            best = buckets

    return (
        best,
        best_score,
    )


print()
print(
    "================================="
)
print(
    " 24-STATE KC CAPACITY CHECK"
)
print(
    "================================="
)
print()


output = {}

overall_min = 999999


for state in STATES:

    key = state_key(
        state
    )

    all_kcs = {

        int(kc)

        for kc in (
            brain.get_state_kcs(
                state
            )
        )
    }


    unique = {

        kc

        for kc in all_kcs

        if usage.get(
            kc,
            0
        ) == 1
    }


    sets = action_sets(
        state
    )

    valid = valid_actions(
        state
    )


    usable = {

        kc

        for kc in unique

        if any(

            kc
            in sets[action]

            for action
            in valid

        )
    }


    buckets, score = partition(

        usable,

        sets,

        valid,

    )


    overall_min = min(
        overall_min,
        score
    )


    print(
        f"STATE {state}"
    )

    print(
        f"  usable unique KCs: "
        f"{len(usable)}"
    )

    print(
        f"  KCs/substate: "
        f"{min(len(x) for x in buckets)}"
        f"-"
        f"{max(len(x) for x in buckets)}"
    )

    print(
        f"  worst VALID MBON coverage: "
        f"{score}"
    )

    print()


    for priority, bucket in zip(
        PRIORITIES,
        buckets,
    ):

        rich_key = (
            f"{key}|{priority}"
        )

        output[
            rich_key
        ] = {

            "direction":
                list(state),

            "priority":
                priority,

            "kc_indices":
                bucket,
        }


with open(
    OUTPUT,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output,
        f,
        indent=2,
    )


print(
    "================================="
)

print(
    "Patterns:",
    len(output)
)

print(
    "Worst valid MBON coverage:",
    overall_min
)

print(
    "Saved:",
    OUTPUT
)


if overall_min >= 3:

    print()
    print(
        "RESULT: GOOD"
    )

    print(
        "24-state neural fly is "
        "safe to build."
    )

elif overall_min >= 1:

    print()
    print(
        "RESULT: USABLE"
    )

    print(
        "24-state design is possible, "
        "but some pathways are sparse."
    )

else:

    print()
    print(
        "RESULT: FAILED"
    )