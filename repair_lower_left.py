import random
import math

from fly_mb_policy import FlyMBPolicy
from fly_actions import execute_action


brain = FlyMBPolicy()

TARGET_X = 0
TARGET_Y = 2

EPISODES = 20


def sign(v):
    if v < 0:
        return -1
    if v > 0:
        return 1
    return 0


def get_state(px, py):
    return (
        sign(TARGET_X - px),
        sign(TARGET_Y - py)
    )


def distance(x1, y1, x2, y2):
    return math.sqrt(
        (x1 - x2) ** 2
        + (y1 - y2) ** 2
    )


for episode in range(EPISODES):

    player_x = 1
    player_y = 1

    # Keep some exploration so it can discover LEFT/DOWN
    epsilon = 0.25

    print()
    print("=" * 60)
    print("EPISODE", episode + 1)
    print("=" * 60)

    for step in range(8):

        state = get_state(
            player_x,
            player_y
        )

        action, scores = brain.choose_action(
            state,
            epsilon=epsilon
        )

        print(
            state,
            action,
            {
                k: round(v, 3)
                for k, v in scores.items()
            }
        )

        old_dist = distance(
            player_x,
            player_y,
            TARGET_X,
            TARGET_Y
        )

        mx, my, _ = execute_action(
            action
        )

        player_x = max(
            0,
            min(2, player_x + mx)
        )

        player_y = max(
            0,
            min(2, player_y + my)
        )

        new_dist = distance(
            player_x,
            player_y,
            TARGET_X,
            TARGET_Y
        )

        if new_dist == 0:
            reward = 10

        elif new_dist < old_dist:
            reward = 1

        elif new_dist > old_dist:
            reward = -1

        else:
            reward = -0.5

        brain.learn(
            state,
            action,
            reward
        )

        print(
            "reward:",
            reward,
            "player:",
            (player_x, player_y)
        )

        if new_dist == 0:
            print("TARGET REACHED")
            break


print()
print("FINAL (-1,1) SCORES:")

print(
    brain.get_scores(
        (-1, 1)
    )
)