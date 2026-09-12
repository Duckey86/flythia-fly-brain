import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent

# -------------------------
# Load mushroom body data
# -------------------------

with open(
    BASE / "data" / "mushroom_body_neurons.json",
    "r",
    encoding="utf-8"
) as f:
    mb = json.load(f)

print("Loading connectivity...")

con = pd.read_parquet(
    BASE / "data" / "2025_Connectivity_783.parquet"
)

print(f"Loaded {len(con):,} connections.")
print("Building forward graph...")

# presynaptic neuron -> [(postsynaptic neuron, weight), ...]
forward = defaultdict(list)

for pre, post, weight in zip(
    con["Presynaptic_ID"],
    con["Postsynaptic_ID"],
    con["Connectivity"]
):
    forward[int(pre)].append(
        (int(post), float(weight))
    )

# Free the large DataFrame once graph is built
del con

print(f"Graph ready: {len(forward):,} presynaptic neurons.")


# -------------------------
# Collect all MBON IDs
# -------------------------

mbon_to_name = {}

for name, ids in mb["mbon"].items():
    for neuron_id in ids:
        mbon_to_name[int(neuron_id)] = name

print(f"MBON neurons to test: {len(mbon_to_name)}")


# -------------------------
# Action pathways
# -------------------------

ACTION_TARGETS = {
    # AOTU025 left
    "LEFT": {
        720575940616012061
    },

    # AOTU025 right
    "RIGHT": {
        720575940639182424
    },

    # P9 forward pair
    "UP": {
        720575940627652358,
        720575940635872101
    },

    # MDNs
    "DOWN": {
        720575940616026939,
        720575940631082808,
        720575940640331472,
        720575940610236514
    },
}


# -------------------------
# Fast path scoring
# -------------------------

def score_mbon(mbon_id, targets, first_hop_limit=100):
    """
    Score an MBON's connectivity to an action pathway.

    direct:
        MBON -> target action neuron

    twohop:
        MBON -> intermediate neuron -> target action neuron

    twohop score is sum(w1 * w2) over matching 2-hop paths.
    """

    direct = 0.0
    twohop = 0.0

    first_connections = forward.get(mbon_id, [])

    # Keep strongest outgoing MBON connections.
    # This makes the search much faster while preserving the strongest routes.
    first_connections = sorted(
        first_connections,
        key=lambda x: x[1],
        reverse=True
    )[:first_hop_limit]

    for middle, w1 in first_connections:

        # Direct MBON -> action target
        if middle in targets:
            direct += w1

        # MBON -> middle -> action target
        for final, w2 in forward.get(middle, []):
            if final in targets:
                twohop += w1 * w2

    # Strongly prioritize a real direct connection if one exists.
    score = direct * 1000.0 + twohop

    return score, direct, twohop


# -------------------------
# Search
# -------------------------

all_results = {}

for action, targets in ACTION_TARGETS.items():

    print()
    print("=" * 90)
    print(action)
    print("=" * 90)

    results = []

    for mbon_id, mbon_name in mbon_to_name.items():

        score, direct, twohop = score_mbon(
            mbon_id,
            targets
        )

        if score > 0:
            results.append(
                (
                    score,
                    mbon_name,
                    mbon_id,
                    direct,
                    twohop
                )
            )

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    all_results[action] = results

    if not results:
        print("No direct or 2-hop MBON paths found.")
        continue

    for (
        score,
        name,
        neuron_id,
        direct,
        twohop
    ) in results[:10]:

        print(
            f"{name:10s} "
            f"id={neuron_id} "
            f"direct={direct:<8.1f} "
            f"twohop={twohop:<12.1f} "
            f"score={score:.1f}"
        )


# -------------------------
# Optional selectivity summary
# -------------------------

print()
print("=" * 90)
print("SELECTIVITY SUMMARY")
print("=" * 90)
print("Higher positive margin = more selective for that action.")
print()

# Collect every MBON that appeared for any action
seen_ids = set()

for results in all_results.values():
    for _, _, neuron_id, _, _ in results:
        seen_ids.add(neuron_id)

for action in ACTION_TARGETS:

    ranked = []

    for neuron_id in seen_ids:

        name = mbon_to_name[neuron_id]

        action_scores = {}

        for other_action, targets in ACTION_TARGETS.items():
            score, _, _ = score_mbon(
                neuron_id,
                targets
            )
            action_scores[other_action] = score

        wanted = action_scores[action]

        other_best = max(
            score
            for other_action, score in action_scores.items()
            if other_action != action
        )

        margin = wanted - other_best

        if wanted > 0:
            ranked.append(
                (
                    margin,
                    wanted,
                    name,
                    neuron_id,
                    action_scores
                )
            )

    ranked.sort(
        key=lambda x: x[0],
        reverse=True
    )

    print()
    print(f"{action} most selective candidates:")

    for (
        margin,
        wanted,
        name,
        neuron_id,
        scores
    ) in ranked[:5]:

        score_text = ", ".join(
            f"{k}={v:.0f}"
            for k, v in scores.items()
        )

        print(
            f"  {name:10s} "
            f"id={neuron_id} "
            f"margin={margin:.1f} "
            f"[{score_text}]"
        )
