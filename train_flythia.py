import random
import math
import json
import os

from fly_actions import execute_action
from pathlib import Path


ACTIONS = ["LEFT", "RIGHT", "UP", "DOWN"]

# Q-table:
# key = (dx_sign, dy_sign)
# value = action values
Q = {}

Q_FILE = Path(__file__).resolve().parent / "q_table.json"

ALPHA = 0.3
GAMMA = 0.9
EPSILON = 0.25


def sign(v):
    if v < 0:
        return -1
    if v > 0:
        return 1
    return 0


def get_state(player_x, player_y, target_x, target_y):
    dx = sign(target_x - player_x)
    dy = sign(target_y - player_y)

    return (dx, dy)


def ensure_state(state):
    if state not in Q:
        Q[state] = {
            action: 0.0
            for action in ACTIONS
        }


def choose_action(state):
    ensure_state(state)

    if random.random() < EPSILON:
        return random.choice(ACTIONS)

    return max(
        Q[state],
        key=Q[state].get
    )


def distance(x1, y1, x2, y2):
    return math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )


def move_player(x, y, mx, my):
    x += mx
    y += my

    x = max(0, min(2, x))
    y = max(0, min(2, y))

    return x, y


def train_episode():

    # Start fly in center
    player_x = 1
    player_y = 1

    # Random target
    target_x = random.randint(0, 2)
    target_y = random.randint(0, 2)

    # Don't place target on fly
    while target_x == player_x and target_y == player_y:
        target_x = random.randint(0, 2)
        target_y = random.randint(0, 2)

    total_reward = 0

    for step in range(15):

        state = get_state(
            player_x,
            player_y,
            target_x,
            target_y
        )

        action = choose_action(state)

        old_dist = distance(
            player_x,
            player_y,
            target_x,
            target_y
        )

        print()
        print(
            f"Step {step} | "
            f"Player=({player_x},{player_y}) "
            f"Target=({target_x},{target_y})"
        )

        print("State:", state)
        print("Action:", action)

        mx, my, brain_result = execute_action(action)

        player_x, player_y = move_player(
            player_x,
            player_y,
            mx,
            my
        )

        new_dist = distance(
            player_x,
            player_y,
            target_x,
            target_y
        )

        # ------------------
        # Reward
        # ------------------

        if new_dist == 0:
            reward = 10

        elif new_dist < old_dist:
            reward = 1

        elif new_dist > old_dist:
            reward = -1

        else:
            reward = -0.2

        total_reward += reward

        new_state = get_state(
            player_x,
            player_y,
            target_x,
            target_y
        )

        ensure_state(new_state)

        # ------------------
        # Q-learning update
        # ------------------

        old_q = Q[state][action]

        if new_dist == 0:
            future_best = 0
        else:
            future_best = max(Q[new_state].values())

        Q[state][action] = (
            old_q
            + ALPHA * (
                reward
                + GAMMA * future_best
                - old_q
            )
        )

        print(
            f"Moved: ({mx},{my})"
        )

        print(
            f"Distance: {old_dist:.2f} → {new_dist:.2f}"
        )

        print(
            f"Reward: {reward}"
        )

        if new_dist == 0:
            print("TARGET REACHED")
            return total_reward, step + 1

    return total_reward, 15

def save_q():
    data = {}

    for (dx, dy), values in Q.items():
        data[f"{dx},{dy}"] = values

    with open(Q_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_q():
    global Q

    if not os.path.exists(Q_FILE):
        return

    with open(Q_FILE, "r") as f:
        data = json.load(f)

    Q = {}

    for key, values in data.items():
        dx, dy = map(int, key.split(","))
        Q[(dx, dy)] = values

    print(f"Loaded {len(Q)} learned states.")


def main():
    
    global EPSILON
    
    load_q()

    episodes = 100

    for episode in range(episodes):

        EPSILON = max(0.05, 0.30 * (1 - episode / episodes))
        print("EPSILON:", round(EPSILON, 3))

        print()
        print("=" * 70)
        print("EPISODE", episode + 1)
        print("=" * 70)

        reward, steps = train_episode()
        
        save_q()

        print()
        print(
            f"Episode result: "
            f"reward={reward}, "
            f"steps={steps}"
        )

    print()
    print("=" * 70)
    print("FINAL Q TABLE")
    print("=" * 70)

    for state, values in Q.items():
        print()
        print("State:", state)

        for action, value in values.items():
            print(
                f"  {action:5s}: "
                f"{value:.2f}"
            )
    
    print()
    print("=" * 70)
    print("LEARNED POLICY")
    print("=" * 70)

    for state in sorted(Q):

        if state == (0, 0):
            continue

        best = max(Q[state], key=Q[state].get)

        print(
            f"{state}: {best} "
            f"({Q[state][best]:.2f})"
        )


if __name__ == "__main__":
    main()