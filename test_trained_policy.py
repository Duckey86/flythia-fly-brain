import random

import train_flythia as fly


# No random exploration
fly.EPSILON = 0.0

# Load trained Q-table
fly.load_q()


def test_episode():

    player_x = 1
    player_y = 1

    # Random target
    target_x = random.randint(0, 2)
    target_y = random.randint(0, 2)

    while target_x == player_x and target_y == player_y:
        target_x = random.randint(0, 2)
        target_y = random.randint(0, 2)

    print()
    print(
        f"START: player=({player_x},{player_y}) "
        f"target=({target_x},{target_y})"
    )

    for step in range(10):

        state = fly.get_state(
            player_x,
            player_y,
            target_x,
            target_y
        )

        # IMPORTANT:
        # always choose best learned action
        action = max(
            fly.Q[state],
            key=fly.Q[state].get
        )

        print(
            f"Step {step + 1}: "
            f"state={state} "
            f"action={action}"
        )

        # Run actual fly brain
        mx, my, _ = fly.execute_action(action)

        player_x, player_y = fly.move_player(
            player_x,
            player_y,
            mx,
            my
        )

        print(
            f"Moved to: ({player_x},{player_y})"
        )

        if (
            player_x == target_x
            and player_y == target_y
        ):
            print("TARGET REACHED")
            return True, step + 1

    print("FAILED")
    return False, 10


def main():

    # Start with 20 because whole-brain simulation is slow
    episodes = 20

    successes = 0
    total_steps = 0

    for episode in range(episodes):

        print()
        print("=" * 60)
        print(f"TEST {episode + 1}/{episodes}")
        print("=" * 60)

        success, steps = test_episode()

        if success:
            successes += 1
            total_steps += steps

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"Success: {successes}/{episodes} "
        f"({successes / episodes * 100:.1f}%)"
    )

    if successes > 0:
        print(
            f"Average steps: "
            f"{total_steps / successes:.2f}"
        )


if __name__ == "__main__":
    main()