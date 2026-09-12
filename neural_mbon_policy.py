import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
TABLE_FILE = BASE / "neural_action_table.json"

ACTIONS = ("LEFT", "RIGHT", "UP", "DOWN")


class NeuralMBONPolicy:
    """
    Fast gameplay policy using action winners measured from the real Brian2
    MBON learning-response test (largest learned - unlearned peak_g).

    This is a cached neural readout, not a fresh Brian2 simulation every move.
    """

    def __init__(self, table_file=TABLE_FILE):
        with open(table_file, "r", encoding="utf-8") as f:
            self.table = json.load(f)

    @staticmethod
    def _key(state):
        x, y = state
        return f"{int(x)},{int(y)}"

    def get_scores(self, state):
        key = self._key(state)

        if key not in self.table:
            raise ValueError(
                f"Unknown state {state}. Expected one of the 8 non-zero "
                "direction states."
            )

        chosen = self.table[key]

        return {
            action: 1.0 if action == chosen else 0.0
            for action in ACTIONS
        }

    def choose_action(self, state, epsilon=0.0):
        """
        Compatible with the existing rhythia_test.py interface:
        returns (action, scores).
        """
        scores = self.get_scores(state)
        action = max(scores, key=scores.get)
        return action, scores

    def learn(self, state, action, reward):
        # No online learning here; this table is the cached neural readout.
        return None

    def reset_flythia(self):
        raise RuntimeError(
            "NeuralMBONPolicy is a cached neural readout. "
            "Do not reset the trained fly memory from this policy."
        )
