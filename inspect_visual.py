import json

with open("neuron_atlas.json", "r", encoding="utf-8") as f:
    atlas = json.load(f)


KEYWORDS = [
    "visual",
    "vision",
    "LC",
    "T4",
    "T5",
    "left",
    "right",
]


def search(obj, path=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key

            text = f"{key} {value}"

            if any(word.lower() in text.lower() for word in KEYWORDS):
                print("\n", new_path)
                print(value)

            search(value, new_path)

    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            search(value, f"{path}[{i}]")


search(atlas)