import math
import random
import shutil

from datetime import datetime
from pathlib import Path

from fly_mb_policy import (
    FlyMBPolicy,
    STATES,
)

from trainable_torch_mbon_policy import (
    TrainableTorchMBONPolicy,
)


BASE = Path(
    __file__
).resolve().parent


MEMORY = (
    BASE
    / "data"
    / "fly_memory.json"
)


CHECKPOINT_DIR = (
    BASE
    / "data"
    / "gpu_training_checkpoints"
)

CHECKPOINT_DIR.mkdir(
    exist_ok=True
)


# ==================================================
# SETTINGS
# ==================================================

MAX_EPISODES = 10000

MIN_EPISODES = 1000

CHECK_EVERY = 250

CHECKPOINT_EVERY = 1000

STABLE_CHECKS_NEEDED = 4


EPS_START = 0.60

EPS_END = 0.03


MOVE_STEP = 0.35


MIN_MARGIN = 0.05


# A diagonal can legitimately move along
# either useful axis.

VALID_ACTIONS = {

    (-1, -1):
        {"LEFT", "UP"},

    (0, -1):
        {"UP"},

    (1, -1):
        {"RIGHT", "UP"},


    (-1, 0):
        {"LEFT"},

    (1, 0):
        {"RIGHT"},


    (-1, 1):
        {"LEFT", "DOWN"},

    (0, 1):
        {"DOWN"},

    (1, 1):
        {"RIGHT", "DOWN"},
}


ACTION_MOVE = {

    "LEFT":
        (-MOVE_STEP, 0),

    "RIGHT":
        (MOVE_STEP, 0),

    "UP":
        (0, -MOVE_STEP),

    "DOWN":
        (0, MOVE_STEP),
}


# ==================================================
# ENVIRONMENT
# ==================================================

def component(sign):

    if sign < 0:

        return random.uniform(
            -2.5,
            -0.45,
        )

    if sign > 0:

        return random.uniform(
            0.45,
            2.5,
        )

    return random.uniform(
        -0.10,
        0.10,
    )


def make_error(state):

    sx, sy = state

    return (
        component(sx),
        component(sy),
    )


def distance(
    dx,
    dy,
):

    return math.hypot(
        dx,
        dy,
    )


def reward_action(
    dx,
    dy,
    action,
):

    before = distance(
        dx,
        dy,
    )


    mx, my = (
        ACTION_MOVE[
            action
        ]
    )


    # error = target - cursor
    #
    # move cursor by mx,my:
    # error shrinks by mx,my.

    new_dx = dx - mx
    new_dy = dy - my


    after = distance(
        new_dx,
        new_dy,
    )


    improvement = (
        before
        - after
    )


    # Environment does NOT tell it
    # "the correct button".
    #
    # It only says whether the movement
    # got closer to the note.

    if improvement > 0.01:

        return 1

    return -1


# ==================================================
# GPU EVALUATION
# ==================================================

def evaluate(brain):

    correct = 0

    print()


    for state in STATES:

        scores = brain.get_scores(
            state
        )


        winner = max(
            scores,
            key=scores.get,
        )


        valid = (
            VALID_ACTIONS[
                state
            ]
        )


        valid_score = max(

            scores[action]

            for action
            in valid

        )


        invalid_score = max(

            scores[action]

            for action
            in scores

            if action
            not in valid

        )


        margin = (
            valid_score
            - invalid_score
        )


        good = (

            winner in valid

            and

            margin >= MIN_MARGIN

        )


        correct += int(
            good
        )


        print(

            f"{state} | "

            f"winner="
            f"{winner:5s} | "

            f"margin="
            f"{margin:+8.3f} | "

            f"{'OK' if good else 'LEARNING'}"

        )


    print()

    print(
        f"GPU neural policy: "
        f"{correct}/8"
    )


    return correct


# ==================================================
# BACKUP
# ==================================================

stamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)


backup = (

    CHECKPOINT_DIR

    / f"before_gpu_training_"
      f"{stamp}.json"

)


shutil.copy2(
    MEMORY,
    backup,
)


print()

print(
    "Backup:",
    backup,
)


# ==================================================
# INTENTIONALLY MAKE FLY FORGET RHYTHIA
# ==================================================

print()

print(
    "Resetting Flythia associations..."
)


reset_brain = FlyMBPolicy()

reset_brain.reset_flythia()


print(
    "Fly has forgotten the learned "
    "Rhythia directions."
)


# ==================================================
# BUILD FULL GPU BRAIN
# ==================================================

print()

print(
    "Building trainable neural fly..."
)


brain = TrainableTorchMBONPolicy(

    run_time_ms=3,

    device="cuda",

)

def save_checkpoint(name):

    brain.save()

    checkpoint = (
        CHECKPOINT_DIR
        / f"{name}_{stamp}.json"
    )
    brain.save()
    shutil.copy2(
        MEMORY,
        checkpoint,
    )

    print(
        "Saved:",
        checkpoint,
    )

    return checkpoint


print()

print(
    "INITIAL UNTRAINED POLICY"
)


evaluate(
    brain
)


# ==================================================
# TRAIN
# ==================================================

positive = 0

negative = 0

stable = 0

try: 
    for episode in range(
        1,
        MAX_EPISODES + 1,
    ):


        progress = (

            episode
            /
            MAX_EPISODES

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


        # --------------------------
        # RANDOM SENSORY EXPERIENCE
        # --------------------------

        state = random.choice(
            STATES
        )


        dx, dy = make_error(
            state
        )


        # --------------------------
        # FULL NEURAL DECISION
        # --------------------------

        action, scores = (
            brain.choose_action(
                state,
                epsilon=epsilon,
            )
        )


        # --------------------------
        # ENVIRONMENTAL OUTCOME
        # --------------------------

        reward = reward_action(
            dx,
            dy,
            action,
        )


        # --------------------------
        # KC -> MBON PLASTICITY
        # --------------------------

        result = brain.learn(
            state,
            action,
            reward,
        )


        if reward > 0:

            positive += 1

        else:

            negative += 1


        # ==================================================
        # CHECK LEARNING
        # ==================================================

        if episode % CHECK_EVERY == 0:


            print()

            print(
                "================================"
            )

            print(
                "EPISODE",
                episode,
            )

            print(
                "================================"
            )


            accuracy = evaluate(
                brain
            )


            good_rate = (

                positive

                /

                max(
                    1,
                    positive + negative,
                )

            )


            print()

            print(
                f"epsilon="
                f"{epsilon:.3f}"
            )

            print(
                f"rewarded="
                f"{positive}"
            )

            print(
                f"punished="
                f"{negative}"
            )

            print(
                f"good actions="
                f"{good_rate * 100:.1f}%"
            )


            if (
                episode
                >= MIN_EPISODES

                and

                accuracy == 8
            ):

                stable += 1

            else:

                stable = 0


            print(
                f"stable="
                f"{stable}/"
                f"{STABLE_CHECKS_NEEDED}"
            )


            # --------------------------
            # CHECKPOINT
            # --------------------------

            if (
                episode
                % CHECKPOINT_EVERY
                == 0
            ):


                checkpoint = (

                    CHECKPOINT_DIR

                    / f"episode_"
                    f"{episode}_"
                    f"{stamp}.json"

                )

                brain.save()
                shutil.copy2(
                    MEMORY,
                    checkpoint,
                )


                print(
                    "Checkpoint:",
                    checkpoint,
                )


            # --------------------------
            # EARLY STOP
            # --------------------------

            if (
                stable
                >= STABLE_CHECKS_NEEDED
            ):

                print()

                print(
                    "NEURAL FLY CONVERGED."
                )

                break
except KeyboardInterrupt:

    print()
    print()
    print(
        "Ctrl+C detected."
    )

    print(
        "Saving current neural fly..."
    )

    save_checkpoint(
        "manual_stop"
    )

    print(
        "Memory saved safely."
    )

    raise SystemExit(0)


# ==================================================
# FINAL
# ==================================================
brain.save()
final_checkpoint = (

    CHECKPOINT_DIR

    / f"trained_gpu_"
      f"{stamp}.json"

)


shutil.copy2(
    MEMORY,
    final_checkpoint,
)


print()

print(
    "================================"
)

print(
    "FINAL GPU TEST"
)

print(
    "================================"
)


final_accuracy = evaluate(
    brain
)


print()

print(
    "Episodes:",
    episode,
)

print(
    "Rewards:",
    positive,
)

print(
    "Punishments:",
    negative,
)

print(
    "Saved:",
    final_checkpoint,
)


# ==================================================
# SAFETY ROLLBACK
# ==================================================

if final_accuracy != 8:

    print()

    print(
        "TRAINING FAILED."
    )

    print(
        "Restoring previous brain..."
    )


    shutil.copy2(
        backup,
        MEMORY,
    )


    print(
        "Rollback complete."
    )


else:

    print()

    print(
        "SUCCESS."
    )

    print(
        "The full neural CUDA fly "
        "learned the task."
    )