import json
import random
from pathlib import Path

from chat_with_fly import run_simulation


BASE = Path(__file__).resolve().parent

ATLAS_FILE = BASE / "neuron_atlas.json"
OUTPUT_FILE = BASE / "sensory_kc_patterns.json"


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


with open(
    ATLAS_FILE,
    "r",
    encoding="utf-8"
) as f:
    atlas = json.load(f)


def get_ids(name):

    stim = atlas["stimuli"][name]

    if "neuron_ids" in stim:
        return stim["neuron_ids"]

    result = []

    for group in stim.get(
        "neuron_ids_groups",
        {}
    ).values():

        result.extend(group)

    return result


# ------------------------------------------------
# Build a pool of REAL sensory neurons
#
# Deliberately excluding locomotion neurons.
# Also avoiding LC4 looming for now because it
# strongly activates the escape circuit.
# ------------------------------------------------

sensory_pool = []

for stimulus in [
    "taste_sweet",
    "taste_bitter",
    "hearing",
    "courtship_song",
    "smell_danger",
]:

    sensory_pool.extend(
        get_ids(stimulus)
    )


# Remove duplicates
sensory_pool = list(
    dict.fromkeys(sensory_pool)
)

print(
    "Available sensory neurons:",
    len(sensory_pool)
)


# Always generate identical codes
rng = random.Random(42)
rng.shuffle(sensory_pool)


NEURONS_PER_STATE = 15

state_inputs = {}

for i, state in enumerate(STATES):

    start = i * NEURONS_PER_STATE
    end = start + NEURONS_PER_STATE

    state_inputs[state] = (
        sensory_pool[start:end]
    )


patterns = {}


for number, state in enumerate(
    STATES,
    start=1
):

    neurons = state_inputs[state]

    print()
    print("=" * 70)
    print(
        f"STATE {state} "
        f"({number}/8)"
    )
    print("=" * 70)

    print(
        "Stimulating",
        len(neurons),
        "sensory neurons"
    )

    result = run_simulation(
        neuron_ids=neurons,
        freq_hz=[250],
        duration_sec=0.1,

        # No learned weights influence this run
        use_memory=False
    )

    mb = result.get(
        "mushroom_body",
        {}
    )

    active_kcs = mb.get(
        "active_kc_indices",
        []
    )

    print(
        "Active KCs:",
        len(active_kcs)
    )

    print(
        "Total active neurons:",
        result["active_neurons"]
    )

    patterns[
        f"{state[0]},{state[1]}"
    ] = {
        "sensory_neurons": neurons,
        "active_kcs": active_kcs,
    }


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        patterns,
        f,
        indent=2
    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    "Saved:",
    OUTPUT_FILE
)