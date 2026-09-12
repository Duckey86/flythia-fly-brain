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
    / "rich_kc_subpatterns.json"
)

NUM_BUCKETS = 9
SEARCH_TRIES = 2000

DISTANCE_BINS = [
    "NEAR",
    "MID",
    "FAR",
]

TIME_BINS = [
    "EARLY",
    "SOON",
    "URGENT",
]


brain = FlyMBPolicy()


# ----------------------------------------
# Count how many direction states use KC
# ----------------------------------------

usage = {}

for state in STATES:

    for kc in brain.get_state_kcs(state):

        kc = int(kc)

        usage[kc] = (
            usage.get(kc, 0)
            + 1
        )


def key(state):

    return (
        f"{state[0]},{state[1]}"
    )


def build_action_sets(state):

    state_key = key(state)

    result = {}

    for action in ACTIONS:

        result[action] = {

            int(conn["kc_index"])

            for conn in (
                brain.real_map[
                    state_key
                ][action]["connections"]
            )
        }

    return result


def score_buckets(
    buckets,
    action_sets,
):

    counts = []

    for bucket in buckets:

        bucket_set = set(bucket)

        for action in ACTIONS:

            n = len(
                bucket_set
                & action_sets[action]
            )

            counts.append(n)

    return min(counts)


def find_best_partition(
    kcs,
    action_sets,
):

    best = None
    best_score = -1

    rng = random.Random(12345)

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


        # Round-robin distribution
        for i, kc in enumerate(
            shuffled
        ):

            buckets[
                i % NUM_BUCKETS
            ].append(kc)


        s = score_buckets(
            buckets,
            action_sets,
        )


        if s > best_score:

            best_score = s
            best = buckets


    return (
        best,
        best_score,
    )


print()
print(
    "======================================"
)
print(
    " RICH STATE KC CAPACITY CHECK"
)
print(
    "======================================"
)
print()

print(
    "Goal: 8 directions x "
    "3 distance x 3 timing = 72 states"
)

print()


output = {}

overall_min = 999999


for state in STATES:

    state_key = key(
        state
    )

    all_kcs = {

        int(x)

        for x in (
            brain.get_state_kcs(
                state
            )
        )
    }


    # Only use KCs unique to this direction.
    # This prevents learning for one direction
    # modifying another direction.

    unique_kcs = {

        kc

        for kc in all_kcs

        if usage.get(
            kc,
            0
        ) == 1
    }


    action_sets = (
        build_action_sets(
            state
        )
    )


    # Only keep unique KCs that actually
    # connect to at least one action MBON.

    usable = {

        kc

        for kc in unique_kcs

        if any(

            kc
            in action_sets[action]

            for action
            in ACTIONS

        )
    }


    buckets, minimum = (
        find_best_partition(
            usable,
            action_sets,
        )
    )


    overall_min = min(
        overall_min,
        minimum,
    )


    print(
        f"STATE {state}"
    )

    print(
        f"  total KCs:   "
        f"{len(all_kcs)}"
    )

    print(
        f"  unique KCs:  "
        f"{len(unique_kcs)}"
    )

    print(
        f"  usable KCs:  "
        f"{len(usable)}"
    )

    print(
        f"  KCs/bucket:  "
        f"{min(len(x) for x in buckets)}"
        f"-"
        f"{max(len(x) for x in buckets)}"
    )

    print(
        f"  worst KC->MBON coverage "
        f"per rich state: {minimum}"
    )

    print()


    bucket_index = 0

    for distance in DISTANCE_BINS:

        for timing in TIME_BINS:

            rich_key = (

                f"{state[0]},"
                f"{state[1]}|"
                f"{distance}|"
                f"{timing}"

            )


            output[
                rich_key
            ] = {

                "direction":
                    list(state),

                "distance":
                    distance,

                "timing":
                    timing,

                "kc_indices":
                    buckets[
                        bucket_index
                    ],

            }


            bucket_index += 1


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
    "======================================"
)

print(
    "72 patterns generated:",
    len(output),
)

print(
    "Worst real KC->MBON coverage:",
    overall_min,
)

print(
    "Saved:",
    OUTPUT,
)


if overall_min >= 3:

    print()

    print(
        "RESULT: GOOD"
    )

    print(
        "The connectome has enough "
        "separable KC pathways to try "
        "the 72-state neural fly."
    )


elif overall_min >= 1:

    print()

    print(
        "RESULT: USABLE BUT SPARSE"
    )

    print(
        "We can build it, but some "
        "rich states have weak "
        "KC->MBON coverage."
    )


else:

    print()

    print(
        "RESULT: NOT SAFE YET"
    )

    print(
        "At least one rich sensory "
        "pattern cannot reach every "
        "action MBON."
    )