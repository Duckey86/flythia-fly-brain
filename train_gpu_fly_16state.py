import json
import math
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

from fly_mb_policy import FlyMBPolicy, ACTIONS
from trainable_torch_mbon_policy_16 import (
    TrainableTorchMBONPolicy16,
    RICH_PATTERN_FILE,
)


BASE = Path(__file__).resolve().parent
MEMORY = BASE / "data" / "fly_memory.json"
CHECKPOINT_DIR = BASE / "data" / "gpu_training_checkpoints_16"
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MAX_EPISODES = 20000
MIN_EPISODES = 3000
CHECK_EVERY = 500
CHECKPOINT_EVERY = 1000
STABLE_CHECKS_NEEDED = 4

EPS_START = 0.65
EPS_END = 0.05

MOVE_STEP = 0.35
MIN_MARGIN = 0.05

# Same threshold we will later use in the live Rhythia controller.
PRIORITY_RATIO = 1.35

# When comparing two actions, treat almost-equal improvements as equally good.
REWARD_TOLERANCE = 0.03


ACTION_MOVE = {
    "LEFT": (-MOVE_STEP, 0.0),
    "RIGHT": (MOVE_STEP, 0.0),
    "UP": (0.0, -MOVE_STEP),
    "DOWN": (0.0, MOVE_STEP),
}


with open(RICH_PATTERN_FILE, "r", encoding="utf-8") as f:
    RICH_PATTERNS = json.load(f)

RICH_STATES = list(RICH_PATTERNS.keys())

if len(RICH_STATES) != 16:
    raise RuntimeError(
        f"Expected 16 rich states, found {len(RICH_STATES)}."
    )


# ============================================================
# STATE / ACTION DEFINITIONS
# ============================================================


def horizontal_action(sx):
    return "LEFT" if sx < 0 else "RIGHT"


def vertical_action(sy):
    return "UP" if sy < 0 else "DOWN"


def valid_actions_for(rich_state):
    info = RICH_PATTERNS[rich_state]
    sx, sy = [int(v) for v in info["direction"]]
    priority = info["priority"]

    if priority == "CARDINAL":
        if sx != 0:
            return {horizontal_action(sx)}
        return {vertical_action(sy)}

    if priority == "X_DOMINANT":
        return {horizontal_action(sx)}

    if priority == "Y_DOMINANT":
        return {vertical_action(sy)}

    if priority == "BALANCED":
        return {
            horizontal_action(sx),
            vertical_action(sy),
        }

    raise ValueError(f"Unknown priority: {priority}")


VALID_ACTIONS = {
    state: valid_actions_for(state)
    for state in RICH_STATES
}


# ============================================================
# SIMULATED RHYTHIA ERROR GENERATION
# ============================================================


def signed(value, sign):
    return abs(float(value)) * (1 if sign > 0 else -1)


def make_error(rich_state):
    """
    Generate dx,dy that really belongs to the requested 16-state condition.

    The fly is not told the correct button. It only sees the KC pattern and
    later gets reward/punishment based on how much its movement helped.
    """
    info = RICH_PATTERNS[rich_state]
    sx, sy = [int(v) for v in info["direction"]]
    priority = info["priority"]

    # Cardinal states: one substantial axis and one near-zero axis.
    if priority == "CARDINAL":
        if sx != 0:
            dx = signed(random.uniform(0.45, 2.5), sx)
            dy = random.uniform(-0.08, 0.08)
        else:
            dx = random.uniform(-0.08, 0.08)
            dy = signed(random.uniform(0.45, 2.5), sy)

        return dx, dy

    # Diagonal states.
    if priority == "X_DOMINANT":
        ax = random.uniform(0.8, 2.5)
        # Guarantee ax/ay > PRIORITY_RATIO with a safety margin.
        ay = ax * random.uniform(0.18, 0.68)

    elif priority == "Y_DOMINANT":
        ay = random.uniform(0.8, 2.5)
        ax = ay * random.uniform(0.18, 0.68)

    elif priority == "BALANCED":
        # Keep the two axes close enough that either useful movement is a
        # reasonable solution. This is intentionally inside the 1.35 band.
        base = random.uniform(0.55, 2.2)
        ratio = random.uniform(0.82, 1.22)
        ax = base * ratio
        ay = base

    else:
        raise ValueError(f"Unknown priority: {priority}")

    return signed(ax, sx), signed(ay, sy)


# ============================================================
# ENVIRONMENTAL REWARD
# ============================================================


def distance(dx, dy):
    return math.hypot(dx, dy)


def improvement_for(dx, dy, action):
    before = distance(dx, dy)
    mx, my = ACTION_MOVE[action]

    # error = target - cursor. Moving cursor by mx,my subtracts that movement
    # from the remaining error.
    after = distance(dx - mx, dy - my)
    return before - after


def reward_action(dx, dy, action):
    """
    Reward actions that are as useful as the best available movement.

    This is what teaches axis priority. Example: with a much larger x error,
    RIGHT improves distance more than UP, so RIGHT gets dopamine and UP does
    not. For balanced diagonals, near-equal useful axes can both be rewarded.
    """
    improvements = {
        candidate: improvement_for(dx, dy, candidate)
        for candidate in ACTIONS
    }

    best = max(improvements.values())
    chosen = improvements[action]

    if best > 0.01 and chosen >= best - REWARD_TOLERANCE:
        return 1

    return -1


# ============================================================
# SAFE RESET OF ONLY THE 16-STATE SYNAPSES
# ============================================================


def atomic_write_memory(memory_obj):
    temp = MEMORY.with_suffix(".json.tmp")

    with open(temp, "w", encoding="utf-8") as f:
        json.dump(memory_obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp, MEMORY)


def reset_16state_learning():
    """
    Remove learned multipliers only for real KC->MBON pairs used by the new
    16-state sensory patterns. The pre-run backup preserves the working
    8-state brain, and unrelated memory entries are left alone.
    """
    policy = FlyMBPolicy()
    mods = policy.memory.memory["weight_modifications"]

    removed = 0

    for info in RICH_PATTERNS.values():
        direction = info["direction"]
        direction_key = f"{int(direction[0])},{int(direction[1])}"
        active_kcs = {int(x) for x in info["kc_indices"]}

        for action in ACTIONS:
            post_idx = int(policy.action_mbon_indices[action])

            for connection in policy.real_map[direction_key][action]["connections"]:
                kc_idx = int(connection["kc_index"])

                if kc_idx not in active_kcs:
                    continue

                memory_key = f"{kc_idx}:{post_idx}"
                if memory_key in mods:
                    del mods[memory_key]
                    removed += 1

    experiences = policy.memory.memory.get("experiences", [])
    policy.memory.memory["experiences"] = [
        exp
        for exp in experiences
        if exp.get("label") != "flythia16"
    ]
    policy.memory.memory["total_experiences"] = len(
        policy.memory.memory["experiences"]
    )

    atomic_write_memory(policy.memory.memory)
    return removed


# ============================================================
# GPU EVALUATION
# ============================================================


def evaluate(brain):
    correct = 0
    print()

    for rich_state in RICH_STATES:
        scores = brain.get_scores(rich_state)
        winner = max(scores, key=scores.get)
        valid = VALID_ACTIONS[rich_state]

        valid_score = max(scores[action] for action in valid)
        invalid_score = max(
            scores[action]
            for action in scores
            if action not in valid
        )
        margin = valid_score - invalid_score

        good = winner in valid and margin >= MIN_MARGIN
        correct += int(good)

        print(
            f"{rich_state:28s} | "
            f"winner={winner:5s} | "
            f"valid={','.join(sorted(valid)):10s} | "
            f"margin={margin:+8.3f} | "
            f"{'OK' if good else 'LEARNING'}"
        )

    print()
    print(f"16-state GPU neural policy: {correct}/16")
    return correct


# ============================================================
# BACKUP + RESET
# ============================================================


if not MEMORY.exists():
    raise RuntimeError("data/fly_memory.json not found")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = CHECKPOINT_DIR / f"before_gpu16_training_{stamp}.json"
shutil.copy2(MEMORY, backup)

print()
print("====================================================")
print(" FULL CONNECTOME 16-STATE GPU REINFORCEMENT TRAINING")
print("====================================================")
print()
print("Backup:", backup)

print()
print("Resetting only the 16-state KC->MBON associations...")
removed = reset_16state_learning()
print(f"Removed {removed} previously learned 16-state synapse entries.")


# ============================================================
# BUILD FULL GPU BRAIN
# ============================================================


print()
print("Building 16-state trainable neural fly...")

brain = TrainableTorchMBONPolicy16(
    run_time_ms=3,
    device="cuda",
)


# Verify that every state has a direct plastic route to at least one valid
# action. For X/Y-dominant states, VALID_ACTIONS contains exactly the desired
# priority action, so this specifically checks that desired pathway exists.
bad_states = []

print()
print("DIRECT KC->MBON PATH CHECK")

for rich_state in RICH_STATES:
    counts = brain.direct_connection_counts(rich_state)
    valid = VALID_ACTIONS[rich_state]
    valid_count = max(counts[action] for action in valid)

    print(
        f"{rich_state:28s} | "
        f"valid_edges={valid_count:2d} | "
        f"all={counts}"
    )

    if valid_count == 0:
        bad_states.append(rich_state)

if bad_states:
    shutil.copy2(backup, MEMORY)
    raise RuntimeError(
        "These rich states have no direct real KC->desired-MBON pathway: "
        + ", ".join(bad_states)
        + ". Original fly_memory.json was restored."
    )


def save_checkpoint(name):
    brain.save()
    checkpoint = CHECKPOINT_DIR / f"{name}_{stamp}.json"
    shutil.copy2(MEMORY, checkpoint)
    print("Saved:", checkpoint)
    return checkpoint


print()
print("INITIAL UNTRAINED 16-STATE POLICY")
evaluate(brain)


# ============================================================
# TRAIN
# ============================================================


positive = 0
negative = 0
stable = 0
episode = 0

try:
    for episode in range(1, MAX_EPISODES + 1):
        progress = episode / MAX_EPISODES
        epsilon = EPS_START + (EPS_END - EPS_START) * progress

        rich_state = random.choice(RICH_STATES)
        dx, dy = make_error(rich_state)

        # Decision always comes from the full CUDA neural simulation.
        action, scores = brain.choose_action(
            rich_state,
            epsilon=epsilon,
        )

        # Environment provides only reward/punishment based on movement value.
        reward = reward_action(dx, dy, action)

        # Immediate online KC->MBON plasticity on this rich sensory subset.
        result = brain.learn(
            rich_state,
            action,
            reward,
        )

        if reward > 0:
            positive += 1
        else:
            negative += 1

        if episode % CHECK_EVERY == 0:
            print()
            print("========================================")
            print("EPISODE", episode)
            print("========================================")

            accuracy = evaluate(brain)
            good_rate = positive / max(1, positive + negative)

            print()
            print(f"epsilon={epsilon:.3f}")
            print(f"rewarded={positive}")
            print(f"punished={negative}")
            print(f"good actions={good_rate * 100:.1f}%")

            if episode >= MIN_EPISODES and accuracy == 16:
                stable += 1
            else:
                stable = 0

            print(f"stable={stable}/{STABLE_CHECKS_NEEDED}")

            if episode % CHECKPOINT_EVERY == 0:
                save_checkpoint(f"episode_{episode}")

            if stable >= STABLE_CHECKS_NEEDED:
                print()
                print("16-STATE NEURAL FLY CONVERGED.")
                break

except KeyboardInterrupt:
    print()
    print("Ctrl+C detected.")
    print("Saving current 16-state neural fly safely...")
    save_checkpoint("manual_stop")
    print("Memory saved safely.")
    raise SystemExit(0)


# ============================================================
# FINAL VALIDATION
# ============================================================


final_checkpoint = save_checkpoint("trained_gpu16")

print()
print("========================================")
print("FINAL 16-STATE GPU TEST")
print("========================================")

final_accuracy = evaluate(brain)

print()
print("Episodes:", episode)
print("Rewards:", positive)
print("Punishments:", negative)
print("Saved:", final_checkpoint)


# ============================================================
# SAFETY ROLLBACK
# ============================================================


if final_accuracy != 16:
    print()
    print("FINAL 16-STATE VALIDATION FAILED.")
    print("Restoring the working brain from before this run...")
    shutil.copy2(backup, MEMORY)
    print("Rollback complete.")
else:
    print()
    print("SUCCESS.")
    print("The full CUDA fly learned all 16 sensory states.")
