import json
import random

with open("neuron_atlas.json", "r", encoding="utf-8") as f:
    atlas = json.load(f)

lc4 = atlas["stimuli"]["vision_looming"]["neuron_ids"]

GRID_INPUTS = {}

# Give every position its own reproducible 15-neuron pattern
for position in range(9):
    rng = random.Random(1000 + position)
    GRID_INPUTS[position] = rng.sample(lc4, 15)


def get_target_neurons(row, col):
    position = row * 3 + col
    return GRID_INPUTS[position]


if __name__ == "__main__":
    for pos, neurons in GRID_INPUTS.items():
        print(f"Position {pos + 1}:")
        print(neurons)
        print()