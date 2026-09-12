import random
import json
from pathlib import Path

from dopamine_learning import FlyMemory


ACTIONS = ["LEFT", "RIGHT", "UP", "DOWN"]

# Four MBONs that have REAL KC inputs from all 8 anatomical state patterns.
# These are used as decision channels; fly_actions.py still executes movement.
ACTION_MBONS = {
    "LEFT":  720575940617552340,  # MBON02
    "RIGHT": 720575940624185095,  # MBON02
    "UP":    720575940623201833,  # MBON11
    "DOWN":  720575940629422086,  # MBON09
}

BASE = Path(__file__).resolve().parent

PATTERN_FILE = BASE / "anatomical_kc_patterns.json"
REAL_MAP_FILE = BASE / "real_kc_mbon_map.json"


# 8 possible target directions around the player
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


class FlyMBPolicy:

    def __init__(self):

        # Real PN-associated KC patterns for each game state
        with open(
            PATTERN_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            self.anatomical_patterns = json.load(f)

        # Real existing KC -> action-MBON synapses
        with open(
            REAL_MAP_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            self.real_map = json.load(f)

        # Persistent fly memory
        self.memory = FlyMemory()

        # Convert action MBON FlyWire IDs to internal Brian2 indices
        self.action_mbon_indices = {}
        
        # Count how many state patterns use each KC
        self.kc_usage_count = {}

        for pattern in self.anatomical_patterns.values():

            for kc_idx in pattern["kc_indices"]:

                self.kc_usage_count[kc_idx] = (
                    self.kc_usage_count.get(
                        kc_idx,
                        0
                    )
                    + 1
                )

        for action, fly_id in ACTION_MBONS.items():

            if fly_id not in self.memory.flyid2i:
                raise ValueError(
                    f"{action} MBON {fly_id} "
                    f"is not in the fly model."
                )

            idx = self.memory.flyid2i[fly_id]

            if idx not in self.memory.mbon_indices:
                raise ValueError(
                    f"{action} neuron {fly_id} "
                    f"is not recognised as an MBON."
                )

            self.action_mbon_indices[action] = idx

    # ------------------------------------------------
    # Anatomical KC population for a target direction
    # ------------------------------------------------

    def get_state_kcs(self, state):

        if state not in STATES:
            raise ValueError(
                f"Invalid state: {state}"
            )

        key = f"{state[0]},{state[1]}"

        pattern = self.anatomical_patterns.get(key)

        if pattern is None:
            raise ValueError(
                f"No anatomical KC pattern for state {state}"
            )

        return pattern["kc_indices"]

    # ------------------------------------------------
    # Read learned strength of REAL KC -> MBON synapses
    # ------------------------------------------------

    def get_scores(self, state):

        if state not in STATES:
            raise ValueError(
                f"Invalid state: {state}"
            )

        key = f"{state[0]},{state[1]}"

        mods = self.memory.memory[
            "weight_modifications"
        ]

        scores = {}

        for action in ACTIONS:

            info = self.real_map[key][action]

            connections = info["connections"]

            mbon_idx = self.action_mbon_indices[
                action
            ]

            weighted_score = 0.0
            total_weight = 0.0

            for connection in connections:

                kc_idx = connection["kc_index"]
                weight = connection["weight"]

                memory_key = (
                    f"{kc_idx}:{mbon_idx}"
                )

                multiplier = mods.get(
                    memory_key,
                    1.0
                )

                weighted_score += (
                    weight * multiplier
                )

                total_weight += weight

            # Should not be zero for the selected action MBONs,
            # but keep this guard just in case.
            if total_weight == 0:
                scores[action] = 0.0
            else:
                scores[action] = (
                    weighted_score / total_weight
                )

        return scores

    # ------------------------------------------------
    # Pick action
    # ------------------------------------------------

    def choose_action(
        self,
        state,
        epsilon=0.25
    ):

        scores = self.get_scores(state)

        # Exploration
        if random.random() < epsilon:
            action = random.choice(ACTIONS)
            return action, scores

        # Exploitation
        best_score = max(
            scores.values()
        )

        # Randomly break exact ties
        best_actions = [
            action
            for action, score in scores.items()
            if score == best_score
        ]

        action = random.choice(
            best_actions
        )

        return action, scores

    # ------------------------------------------------
    # Modify ONLY REAL KC -> chosen MBON synapses
    # ------------------------------------------------

    def learn(
        self,
        state,
        action,
        reward
    ):

        if state not in STATES:
            raise ValueError(
                f"Invalid state: {state}"
            )

        if action not in ACTIONS:
            raise ValueError(
                f"Invalid action: {action}"
            )

        key = f"{state[0]},{state[1]}"

        connections = (
            self.real_map[key][action][
                "connections"
            ]
        )

        mbon_idx = self.action_mbon_indices[
            action
        ]

        # Small multiplicative learning steps
        if reward >= 10:
            multiplier = 1.02
            signal = "reward"

        elif reward > 0:
            multiplier = 1.01
            signal = "reward"

        elif reward <= -1:
            multiplier = 0.99
            signal = "punishment"

        else:
            multiplier = 0.995
            signal = "punishment"

        mods = self.memory.memory[
            "weight_modifications"
        ]

        changed = 0

        # IMPORTANT:
        # Only modify KC -> MBON pairs that physically exist
        # in real_kc_mbon_map.json.
        for connection in connections:

            kc_idx = connection[
            "kc_index"
            ]

            # Don't modify KCs shared between multiple states.
            # This prevents one state from overwriting another.
            if self.kc_usage_count.get(
                kc_idx,
                0
            ) != 1:
                continue

            memory_key = (
                f"{kc_idx}:{mbon_idx}"
            )

            old = mods.get(
                memory_key,
                1.0
            )

            new = old * multiplier

            # Prevent runaway weights
            new = max(
                0.25,
                min(4.0, new)
            )

            mods[memory_key] = round(
                new,
                5
            )

            changed += 1

        # Record experience
        self.memory.memory[
            "experiences"
        ].append(
            {
                "label": "flythia",
                "state": list(state),
                "action": action,
                "reward": reward,
                "signal_type": signal,
                "target_mbon": int(
                    ACTION_MBONS[action]
                ),
                "real_synapses_modified": changed,
            }
        )

        self.memory.memory[
            "total_experiences"
        ] = len(
            self.memory.memory[
                "experiences"
            ]
        )

        self.memory.save_memory()

        return {
            "signal_type": signal,
            "real_synapses_modified": changed,
        }

    # ------------------------------------------------
    # Reset Flythia associations for CURRENT real map
    # ------------------------------------------------

    def reset_flythia(self):

        mods = self.memory.memory[
            "weight_modifications"
        ]

        removed = 0

        for state in STATES:

            state_key = (
                f"{state[0]},{state[1]}"
            )

            for action in ACTIONS:

                mbon_idx = (
                    self.action_mbon_indices[
                        action
                    ]
                )

                connections = (
                    self.real_map[
                        state_key
                    ][action][
                        "connections"
                    ]
                )

                for connection in connections:

                    kc_idx = connection[
                        "kc_index"
                    ]

                    memory_key = (
                        f"{kc_idx}:{mbon_idx}"
                    )

                    if memory_key in mods:
                        del mods[memory_key]
                        removed += 1

        # Remove Flythia experience records
        experiences = (
            self.memory.memory[
                "experiences"
            ]
        )

        self.memory.memory[
            "experiences"
        ] = [
            exp
            for exp in experiences
            if exp.get("label") != "flythia"
        ]

        self.memory.memory[
            "total_experiences"
        ] = len(
            self.memory.memory[
                "experiences"
            ]
        )

        self.memory.save_memory()

        print(
            f"Flythia mushroom-body memory reset. "
            f"Removed {removed} learned real-synapse entries."
        )


# ------------------------------------------------
# Test
# ------------------------------------------------

if __name__ == "__main__":

    brain = FlyMBPolicy()

    print()
    print("ACTION MBON INDICES")

    for action, idx in (
        brain.action_mbon_indices.items()
    ):
        print(
            action,
            "->",
            idx
        )

    print()
    print("CURRENT POLICY")

    for state in STATES:

        scores = brain.get_scores(
            state
        )

        print(
            state,
            {
                action: round(score, 3)
                for action, score in scores.items()
            }
        )
