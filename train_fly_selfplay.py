import math
import random
import shutil
from datetime import datetime
from pathlib import Path

from fly_mb_policy import (
    FlyMBPolicy,
    STATES,
)

BASE = Path(__file__).resolve().parent
MEMORY = BASE / "data" / "fly_memory.json"

CHECKPOINT_DIR = (
    BASE / "data" / "selfplay_checkpoints"
)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# -----------------------------------------
# TRAINING SETTINGS
# -----------------------------------------

EPISODES = 10000

MOVE_STEP = 0.35

EPS_START = 0.50
EPS_END = 0.03

CHECK_EVERY = 200
STABLE_CHECKS_NEEDED = 8

MIN_MARGIN = 0.15


# Valid actions are NOT hardcoded to only one
# answer for diagonals.
#
# For example:
# upper-left can legitimately move UP or LEFT.
VALID_ACTIONS = {

    (-1, -1): {"LEFT", "UP"},
    (0, -1):  {"UP"},
    (1, -1):  {"RIGHT", "UP"},

    (-1, 0):  {"LEFT"},
    (1, 0):   {"RIGHT"},

    (-1, 1):  {"LEFT", "DOWN"},
    (0, 1):   {"DOWN"},
    (1, 1):   {"RIGHT", "DOWN"},
}


ACTION_MOVE = {

    "LEFT":  (-MOVE_STEP, 0.0),
    "RIGHT": ( MOVE_STEP, 0.0),

    "UP":    (0.0, -MOVE_STEP),
    "DOWN":  (0.0,  MOVE_STEP),
}


# -----------------------------------------
# GENERATE A RANDOM RHYTHIA-LIKE ERROR
# -----------------------------------------

def component(sign):

    if sign == -1:
        return random.uniform(
            -2.5,
            -0.45
        )

    if sign == 1:
        return random.uniform(
            0.45,
            2.5
        )

    # Cardinal direction:
    # keep other axis almost centered.
    return random.uniform(
        -0.10,
        0.10
    )


def make_error(state):

    sx, sy = state

    return (
        component(sx),
        component(sy)
    )


# -----------------------------------------
# ENVIRONMENT REWARD
# -----------------------------------------

def apply_action(dx, dy, action):

    move_x, move_y = (
        ACTION_MOVE[action]
    )

    # error = target - cursor
    #
    # moving the cursor changes error
    # in the opposite direction.
    new_dx = dx - move_x
    new_dy = dy - move_y

    return (
        new_dx,
        new_dy
    )


def distance(dx, dy):

    return math.hypot(
        dx,
        dy
    )


def environmental_reward(
    dx,
    dy,
    action
):

    before = distance(
        dx,
        dy
    )

    new_dx, new_dy = apply_action(
        dx,
        dy,
        action
    )

    after = distance(
        new_dx,
        new_dy
    )

    improvement = (
        before - after
    )

    # Actual reinforcement signal:
    #
    # closer to target = dopamine reward
    # farther from target = punishment

    if improvement > 0.01:
        return 1

    return -1


# -----------------------------------------
# EVALUATION
# -----------------------------------------

def evaluate(brain):

    correct = 0

    print()

    for state in STATES:

        scores = brain.get_scores(
            state
        )

        winner = max(
            scores,
            key=scores.get
        )

        valid = VALID_ACTIONS[
            state
        ]

        valid_score = max(
            scores[action]
            for action in valid
        )

        invalid_score = max(
            scores[action]
            for action in scores
            if action not in valid
        )

        margin = (
            valid_score
            - invalid_score
        )

        ok = (
            winner in valid
            and
            margin >= MIN_MARGIN
        )

        if ok:
            correct += 1

        print(
            f"{state} | "
            f"winner={winner:5s} | "
            f"valid={','.join(sorted(valid)):10s} | "
            f"margin={margin:+.3f} | "
            f"{'OK' if ok else 'LEARNING'}"
        )

    print(
        f"\nPolicy: {correct}/8"
    )

    return correct


# -----------------------------------------
# SAFETY BACKUP
# -----------------------------------------

if not MEMORY.exists():

    raise RuntimeError(
        "data/fly_memory.json not found"
    )


stamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

start_backup = (
    CHECKPOINT_DIR /
    f"before_selfplay_{stamp}.json"
)

shutil.copy2(
    MEMORY,
    start_backup
)

print()
print(
    "===================================="
)

print(
    "      FLY SELF-PLAY TRAINING"
)

print(
    "===================================="
)

print()

print(
    "Safety backup:",
    start_backup
)


# -----------------------------------------
# RESET ONLY FLYTHIA LEARNING
# -----------------------------------------

brain = FlyMBPolicy()

print()
print(
    "Resetting existing Flythia associations..."
)

brain.reset_flythia()

# Reload clean memory.
brain = FlyMBPolicy()

print(
    "Flythia memory reset."
)

print()
print(
    "The fly now has to relearn by trial and error."
)


# -----------------------------------------
# TRAIN
# -----------------------------------------

stable_checks = 0

reward_total = 0

positive = 0
negative = 0


for episode in range(
    1,
    EPISODES + 1
):

    progress = (
        episode /
        EPISODES
    )

    epsilon = (
        EPS_START
        +
        (
            EPS_END
            - EPS_START
        )
        * progress
    )


    # Random sensory situation.
    state = random.choice(
        STATES
    )

    dx, dy = make_error(
        state
    )


    # Fly decides.
    action, scores = (
        brain.choose_action(
            state,
            epsilon=epsilon
        )
    )


    # Environment judges the action.
    reward = (
        environmental_reward(
            dx,
            dy,
            action
        )
    )


    # Dopamine learning changes
    # actual KC -> MBON multipliers.
    brain.learn(
        state,
        action,
        reward
    )


    reward_total += reward

    if reward > 0:
        positive += 1
    else:
        negative += 1


    # ---------------------------------
    # PROGRESS CHECK
    # ---------------------------------

    if episode % CHECK_EVERY == 0:

        accuracy = evaluate(
            brain
        )

        success_rate = (
            positive
            /
            max(
                1,
                positive + negative
            )
        )


        print()

        print(
            f"episode={episode} | "
            f"epsilon={epsilon:.3f} | "
            f"reward={reward_total:+d} | "
            f"good actions="
            f"{success_rate * 100:.1f}%"
        )


        if accuracy == 8:

            stable_checks += 1

            print(
                f"Stable neural policy: "
                f"{stable_checks}/"
                f"{STABLE_CHECKS_NEEDED}"
            )

        else:

            stable_checks = 0


        # Checkpoint every 1000 experiences.
        if episode % 1000 == 0:

            checkpoint = (
                CHECKPOINT_DIR /
                f"episode_{episode}_"
                f"{stamp}.json"
            )

            shutil.copy2(
                MEMORY,
                checkpoint
            )

            print(
                "Checkpoint:",
                checkpoint
            )


        if (
            stable_checks
            >= STABLE_CHECKS_NEEDED
        ):

            print()
            print(
                "Fly has converged."
            )

            break


# -----------------------------------------
# FINAL CHECKPOINT
# -----------------------------------------

final_file = (
    CHECKPOINT_DIR /
    f"trained_{stamp}.json"
)

shutil.copy2(
    MEMORY,
    final_file
)


print()
print(
    "===================================="
)

print(
    "SELF-PLAY COMPLETE"
)

print(
    "===================================="
)

print(
    "Episodes:",
    episode
)

print(
    "Positive actions:",
    positive
)

print(
    "Punished actions:",
    negative
)

print(
    "Final checkpoint:",
    final_file
)


# -----------------------------------------
# FULL CUDA CONNECTOME VALIDATION
# -----------------------------------------

print()
print(
    "Loading 138k-neuron CUDA fly..."
)

from live_torch_mbon_policy import (
    LiveTorchMBONPolicy
)

gpu = LiveTorchMBONPolicy(
    run_time_ms=3,
    device="cuda"
)


gpu_correct = 0

print()
print(
    "========== CUDA VALIDATION =========="
)


for state in STATES:

    scores = gpu.get_scores(
        state
    )

    winner = max(
        scores,
        key=scores.get
    )

    valid = VALID_ACTIONS[
        state
    ]

    ok = (
        winner in valid
    )

    gpu_correct += int(
        ok
    )

    print(
        f"{state}: "
        f"{winner:5s} | "
        f"{'OK' if ok else 'WRONG'}"
    )


print()
print(
    f"CUDA result: "
    f"{gpu_correct}/8"
)


# -----------------------------------------
# AUTOMATIC ROLLBACK
# -----------------------------------------

if gpu_correct != 8:

    print()
    print(
        "CUDA VALIDATION FAILED."
    )

    print(
        "Restoring the memory from "
        "before this training run."
    )

    shutil.copy2(
        start_backup,
        MEMORY
    )

    print(
        "Original memory restored."
    )

else:

    print()
    print(
        "TRAINING SUCCESSFUL."
    )

    print(
        "New self-trained fly_memory.json "
        "is active."
    )