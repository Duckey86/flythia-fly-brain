import json
import shutil
from datetime import datetime
from pathlib import Path

from fly_mb_policy import FlyMBPolicy, STATES

BASE = Path(__file__).resolve().parent

MEMORY = BASE / "data" / "fly_memory.json"
BACKUP = BASE / "data" / "fly_memory_before_overnight.json"

CHECKPOINTS = BASE / "data" / "overnight_checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)

# This is the policy we've already validated with the real neural model.
EXPECTED = {
    (-1, -1): "UP",
    (0, -1):  "UP",
    (1, -1):  "UP",

    (-1, 0):  "LEFT",
    (1, 0):   "RIGHT",

    (-1, 1):  "DOWN",
    (0, 1):   "DOWN",
    (1, 1):   "DOWN",
}

TARGET_MARGIN = 0.20
MAX_EPOCHS = 12
STABLE_EPOCHS = 3


def margin(scores, correct):

    correct_score = scores[correct]

    best_wrong = max(
        score
        for action, score in scores.items()
        if action != correct
    )

    return correct_score - best_wrong


# ----------------------------------
# SAFETY
# ----------------------------------

if not MEMORY.exists():
    raise RuntimeError(
        "Can't find data/fly_memory.json"
    )

if not BACKUP.exists():
    raise RuntimeError(
        "Backup missing! Expected:\n"
        "data/fly_memory_before_overnight.json"
    )


stamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

start_checkpoint = (
    CHECKPOINTS /
    f"start_{stamp}.json"
)

shutil.copy2(
    MEMORY,
    start_checkpoint
)

print()
print("====================================")
print("   FLY OVERNIGHT TRAINING")
print("====================================")
print()

print(
    "Safety checkpoint:",
    start_checkpoint
)

print()


# ----------------------------------
# TRAIN
# ----------------------------------

brain = FlyMBPolicy()

stable = 0
total_updates = 0


for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    print()
    print(
        f"========== EPOCH {epoch} =========="
    )

    need_training = []


    for state in STATES:

        scores = brain.get_scores(
            state
        )

        expected = EXPECTED[
            state
        ]

        winner = max(
            scores,
            key=scores.get
        )

        m = margin(
            scores,
            expected
        )

        good = (
            winner == expected
            and
            m >= TARGET_MARGIN
        )


        print(
            f"{state}  "
            f"expected={expected:5}  "
            f"winner={winner:5}  "
            f"margin={m:+.3f}  "
            f"{'OK' if good else 'TRAIN'}"
        )


        if not good:

            need_training.append(
                (
                    state,
                    expected
                )
            )


    # ----------------------------------
    # CONVERGENCE CHECK
    # ----------------------------------

    if not need_training:

        stable += 1

        print()
        print(
            f"Stable pass "
            f"{stable}/{STABLE_EPOCHS}"
        )


        if stable >= STABLE_EPOCHS:

            print()
            print(
                "Directional policy converged."
            )

            break


    else:

        stable = 0


        print()
        print(
            "Reinforcing",
            len(need_training),
            "states..."
        )


        for state, action in need_training:

            # +1 uses the modest positive
            # reinforcement path.
            brain.learn(
                state,
                action,
                reward=1
            )

            total_updates += 1


    # ----------------------------------
    # PERIODIC CHECKPOINT
    # ----------------------------------

    if epoch % 4 == 0:

        cp = (
            CHECKPOINTS /
            f"epoch_{epoch}_{stamp}.json"
        )

        shutil.copy2(
            MEMORY,
            cp
        )

        print(
            "Checkpoint:",
            cp
        )


# ----------------------------------
# SAVE FINAL
# ----------------------------------

final_checkpoint = (
    CHECKPOINTS /
    f"final_{stamp}.json"
)

shutil.copy2(
    MEMORY,
    final_checkpoint
)

print()
print(
    "Training updates:",
    total_updates
)

print(
    "Final checkpoint:",
    final_checkpoint
)


# ----------------------------------
# REAL GPU NEURAL VALIDATION
# ----------------------------------

print()
print(
    "Loading real CUDA neural model "
    "for final validation..."
)

from live_torch_mbon_policy import (
    LiveTorchMBONPolicy
)


gpu = LiveTorchMBONPolicy(
    run_time_ms=3,
    device="cuda"
)


correct_count = 0


print()
print(
    "========== GPU VALIDATION =========="
)


for state in STATES:

    scores = gpu.get_scores(
        state
    )

    winner = max(
        scores,
        key=scores.get
    )

    expected = EXPECTED[
        state
    ]

    ok = (
        winner == expected
    )

    correct_count += int(
        ok
    )


    print(
        f"{state}: "
        f"{winner:5}  "
        f"expected={expected:5} "
        f"{'OK' if ok else 'WRONG'}"
    )


print()
print(
    f"GPU validation: "
    f"{correct_count}/8"
)


# ----------------------------------
# AUTOMATIC ROLLBACK IF BAD
# ----------------------------------

if correct_count != 8:

    print()
    print(
        "VALIDATION FAILED!"
    )

    print(
        "Restoring memory from "
        "before this training session..."
    )

    shutil.copy2(
        start_checkpoint,
        MEMORY
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
        "fly_memory.json contains "
        "the trained policy."
    )