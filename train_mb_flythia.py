import random
import math

from fly_mb_policy import FlyMBPolicy
from fly_actions import execute_action


brain = FlyMBPolicy()

EPISODES = 100

START_EPSILON = 0.30
MIN_EPSILON = 0.05


def sign(v):
    if v < 0:
        return -1

    if v > 0:
        return 1

    return 0


def get_state(
    player_x,
    player_y,
    target_x,
    target_y
):
    return (
        sign(target_x - player_x),
        sign(target_y - player_y)
    )


def distance(
    x1,
    y1,
    x2,
    y2
):
    return math.sqrt(
        (x1 - x2) ** 2
        +
        (y1 - y2) ** 2
    )


def move_player(
    x,
    y,
    mx,
    my
):

    x += mx
    y += my

    x = max(
        0,
        min(2, x)
    )

    y = max(
        0,
        min(2, y)
    )

    return x, y


def train_episode(epsilon):

    # Fly starts in centre
    player_x = 1
    player_y = 1

    # Random target
    while True:

        target_x = random.randint(
            0, 2
        )

        target_y = random.randint(
            0, 2
        )

        if (
            target_x != player_x
            or target_y != player_y
        ):
            break

    total_reward = 0

    for step in range(15):

        state = get_state(
            player_x,
            player_y,
            target_x,
            target_y
        )

        # -----------------------
        # MUSHROOM BODY DECISION
        # -----------------------

        action, scores = brain.choose_action(
            state,
            epsilon=epsilon
        )

        print()
        print(
            f"Player=({player_x},{player_y}) "
            f"Target=({target_x},{target_y})"
        )

        print(
            "State:",
            state
        )

        print(
            "MBON scores:",
            {
                k: round(v, 3)
                for k, v
                in scores.items()
            }
        )

        print(
            "Chosen:",
            action
        )

        old_distance = distance(
            player_x,
            player_y,
            target_x,
            target_y
        )

        # -----------------------
        # REAL FLY MOTOR CIRCUIT
        # -----------------------

        mx, my, result = execute_action(
            action
        )

        player_x, player_y = (
            move_player(
                player_x,
                player_y,
                mx,
                my
            )
        )

        new_distance = distance(
            player_x,
            player_y,
            target_x,
            target_y
        )

        # -----------------------
        # DOPAMINE-LIKE REWARD
        # -----------------------

        if new_distance == 0:

            reward = 10

        elif new_distance < old_distance:

            reward = 1

        elif new_distance > old_distance:

            reward = -1

        else:

            reward = -0.2

        total_reward += reward

        # -----------------------
        # KC -> CHOSEN MBON
        # PLASTICITY
        # -----------------------

        brain.learn(
            state,
            action,
            reward
        )

        print(
            f"Moved: ({mx},{my})"
        )

        print(
            f"Distance: "
            f"{old_distance:.2f} "
            f"-> {new_distance:.2f}"
        )

        print(
            "Reward:",
            reward
        )

        # Show updated association
        new_scores = brain.get_scores(
            state
        )

        print(
            "Updated:",
            {
                k: round(v, 3)
                for k, v
                in new_scores.items()
            }
        )

        if new_distance == 0:

            print(
                "TARGET REACHED"
            )

            return (
                total_reward,
                step + 1
            )

    return (
        total_reward,
        15
    )


def main():

    for episode in range(
        EPISODES
    ):

        epsilon = max(
            MIN_EPSILON,
            START_EPSILON
            * (
                1
                - episode
                / EPISODES
            )
        )

        print()
        print(
            "=" * 70
        )

        print(
            f"EPISODE "
            f"{episode + 1}/"
            f"{EPISODES}"
        )

        print(
            f"EPSILON: "
            f"{epsilon:.3f}"
        )

        print(
            "=" * 70
        )

        reward, steps = (
            train_episode(
                epsilon
            )
        )

        print()
        print(
            f"Episode result: "
            f"reward={reward}, "
            f"steps={steps}"
        )

    # -----------------------
    # FINAL LEARNED POLICY
    # -----------------------

    print()
    print(
        "=" * 70
    )

    print(
        "MUSHROOM BODY POLICY"
    )

    print(
        "=" * 70
    )

    for state in [
        (-1, -1),
        (0, -1),
        (1, -1),

        (-1, 0),
        (1, 0),

        (-1, 1),
        (0, 1),
        (1, 1),
    ]:

        scores = brain.get_scores(
            state
        )

        best = max(
            scores,
            key=scores.get
        )

        print()
        print(
            state,
            "->",
            best
        )

        print(
            {
                action: round(
                    score,
                    3
                )
                for action, score
                in scores.items()
            }
        )


if __name__ == "__main__":
    main()