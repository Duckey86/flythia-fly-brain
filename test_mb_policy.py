import random

from fly_mb_policy import FlyMBPolicy
from fly_actions import execute_action


brain = FlyMBPolicy()

TEST_EPISODES = 20
MAX_STEPS = 10


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


def move_player(x, y, mx, my):

    x += mx
    y += my

    x = max(0, min(2, x))
    y = max(0, min(2, y))

    return x, y


def test_episode():

    # Start in centre
    player_x = 1
    player_y = 1

    # Random target
    while True:

        target_x = random.randint(0, 2)
        target_y = random.randint(0, 2)

        if (
            target_x != player_x
            or target_y != player_y
        ):
            break

    print()
    print(
        f"START "
        f"Player=({player_x},{player_y}) "
        f"Target=({target_x},{target_y})"
    )

    for step in range(MAX_STEPS):

        state = get_state(
            player_x,
            player_y,
            target_x,
            target_y
        )

        # --------------------------------
        # Mushroom-body policy
        # NO EXPLORATION
        # --------------------------------

        action, scores = brain.choose_action(
            state,
            epsilon=0.0
        )

        print()
        print(
            f"Step {step + 1}"
        )

        print(
            "State:",
            state
        )

        print(
            "Scores:",
            {
                action_name: round(score, 3)
                for action_name, score
                in scores.items()
            }
        )

        print(
            "Chosen:",
            action
        )

        # --------------------------------
        # Actual fly motor circuit
        # --------------------------------

        mx, my, result = execute_action(
            action
        )

        player_x, player_y = move_player(
            player_x,
            player_y,
            mx,
            my
        )

        print(
            f"Movement: ({mx},{my})"
        )

        print(
            f"Player: "
            f"({player_x},{player_y})"
        )

        # --------------------------------
        # Did it reach target?
        # --------------------------------

        if (
            player_x == target_x
            and player_y == target_y
        ):

            print(
                "TARGET REACHED"
            )

            return True, step + 1

    print(
        "FAILED"
    )

    return False, MAX_STEPS


def main():

    successes = 0
    total_steps = 0

    print()
    print(
        "=" * 70
    )

    print(
        "MUSHROOM BODY MEMORY TEST"
    )

    print(
        "NO LEARNING / NO Q-TABLE / EPSILON = 0"
    )

    print(
        "=" * 70
    )

    for episode in range(
        TEST_EPISODES
    ):

        print()
        print(
            "=" * 70
        )

        print(
            f"TEST "
            f"{episode + 1}/"
            f"{TEST_EPISODES}"
        )

        print(
            "=" * 70
        )

        success, steps = (
            test_episode()
        )

        if success:
            successes += 1
            total_steps += steps

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 70
    )

    success_rate = (
        successes
        / TEST_EPISODES
        * 100
    )

    print(
        f"Success: "
        f"{successes}/"
        f"{TEST_EPISODES} "
        f"({success_rate:.1f}%)"
    )

    if successes > 0:

        average_steps = (
            total_steps
            / successes
        )

        print(
            f"Average steps: "
            f"{average_steps:.2f}"
        )


if __name__ == "__main__":
    main()