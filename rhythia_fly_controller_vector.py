import json
import socket
import time

import torch

from trainable_torch_mbon_policy_16 import TrainableTorchMBONPolicy16
from pathlib import Path


HOST = "127.0.0.1"
STATE_PORT = 50555
COMMAND_PORT = 50556

# Same hitbox-friendly deadzone as the working controller.
DEADZONE = 0.20

# Must match the 16-state training threshold.
PRIORITY_RATIO = 1.35

# Prevent one diagonal axis from disappearing completely. The neural MBON
# differences still determine the angle above this floor.
DIAGONAL_FLOOR = 0.30


ACTIONS = ("LEFT", "RIGHT", "UP", "DOWN")


def axis(v):
    if v > DEADZONE:
        return 1
    if v < -DEADZONE:
        return -1
    return 0


def classify_rich_state(dx, dy):
    """Convert Rhythia error into one of the trained 16 sensory states."""
    sx = axis(dx)
    sy = -axis(dy)

    if sx == 0 and sy == 0:
        return None, (0, 0), "CENTERED"

    if sx == 0 or sy == 0:
        key = f"{sx},{sy}"
        return key, (sx, sy), "CARDINAL"

    ax = abs(dx)
    ay = abs(dy)
    ratio = ax / max(ay, 1e-9)

    if ratio > PRIORITY_RATIO:
        priority = "X_DOMINANT"
    elif ratio < (1.0 / PRIORITY_RATIO):
        priority = "Y_DOMINANT"
    else:
        priority = "BALANCED"

    key = f"{sx},{sy}|{priority}"
    return key, (sx, sy), priority


def neural_motor_vector(scores, direction, dx, dy):
    """
    16-state fly still makes the neural decision.

    If the fly chooses a target-compatible MBON, the actual 2-D motor angle
    follows the real Rhythia dx/dy geometry instead of using MBON score
    magnitudes as an analog joystick.

    Positive X = RIGHT
    Negative X = LEFT

    Positive Y = DOWN in the fly/bridge motor convention
    Negative Y = UP
    """

    sx, sy = direction

    if sx == 0 and sy == 0:
        return 0.0, 0.0

    # -----------------------------------------
    # Let the neural fly make the decision first
    # -----------------------------------------
    winner = max(
        scores,
        key=scores.get,
    )

    valid_actions = set()

    if sx < 0:
        valid_actions.add("LEFT")

    if sx > 0:
        valid_actions.add("RIGHT")

    if sy < 0:
        valid_actions.add("UP")

    if sy > 0:
        valid_actions.add("DOWN")

    # -----------------------------------------
    # If the neural fly somehow chooses a
    # direction that does NOT point at the note,
    # respect that neural decision instead of
    # secretly correcting it with geometry.
    # -----------------------------------------
    if winner not in valid_actions:

        fallback = {
            "LEFT":  (-1.0, 0.0),
            "RIGHT": (1.0, 0.0),
            "UP":    (0.0, -1.0),
            "DOWN":  (0.0, 1.0),
        }

        return fallback[winner]

    # -----------------------------------------
    # Cardinal target
    # -----------------------------------------
    if sx == 0:
        return 0.0, float(sy)

    if sy == 0:
        return float(sx), 0.0

    # -----------------------------------------
    # Diagonal target:
    # use the ACTUAL target angle.
    #
    # Example:
    # dx = +1.30
    # dy = -1.00
    #
    # motor becomes approximately:
    # (+1.00, +0.77)
    #
    # Y uses sy because the bridge motor
    # convention is inverted from raw Rhythia Y.
    # -----------------------------------------
    ax = abs(dx)
    ay = abs(dy)

    peak = max(
        ax,
        ay,
        1e-9,
    )

    vx = (
        float(sx)
        * (ax / peak)
    )

    vy = (
        float(sy)
        * (ay / peak)
    )

    return vx, vy


print("Loading 16-state CUDA fly with neural vector motor output...")
brain = TrainableTorchMBONPolicy16(
    run_time_ms=3,
    device="cuda",
)
print("16-state vector fly ready.\n")

# ============================================================
# PRECOMPUTE ALL 16 FULL-NEURAL RESPONSES
# ============================================================

PATTERN_FILE = (
    Path(__file__).resolve().parent
    / "priority_kc_subpatterns_16.json"
)

with open(
    PATTERN_FILE,
    "r",
    encoding="utf-8",
) as f:
    all_patterns = json.load(f)


print("Precomputing 16 learned neural responses...")

NEURAL_CACHE = {}

cache_start = time.perf_counter()

for state_key in all_patterns.keys():

    t0 = time.perf_counter()

    scores = brain.get_scores(
        state_key
    )

    if brain.device.type == "cuda":
        torch.cuda.synchronize()

    NEURAL_CACHE[state_key] = scores

    elapsed = (
        time.perf_counter()
        - t0
    ) * 1000.0

    winner = max(
        scores,
        key=scores.get,
    )

    print(
        f"  {state_key:28s} "
        f"winner={winner:5s} "
        f"{elapsed:5.1f}ms"
    )


total_cache_ms = (
    time.perf_counter()
    - cache_start
) * 1000.0

print(
    f"Neural cache ready in "
    f"{total_cache_ms:.1f}ms\n"
)

rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rx.bind((HOST, STATE_PORT))
rx.setblocking(False)

tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Waiting for Rhythia...")
print("F10 still toggles fly control.\n")

latest = None
last_decision_key = None
last_scores = {a: 0.0 for a in ACTIONS}
last_vector = (0.0, 0.0)
last_winner = "CENTERED"
last_brain_ms = 0.0
last_print = 0.0

try:
    while True:
        got_packet = False

        # Drain stale ~60 Hz packets and use only the newest state.
        while True:
            try:
                data, _ = rx.recvfrom(65535)
                latest = json.loads(data.decode("utf-8"))
                got_packet = True
            except BlockingIOError:
                break

        if latest is None:
            time.sleep(0.001)
            continue

        p = latest
        dx = float(p["dx"])
        dy = float(p["dy"])
        note = int(p["note"])

        rich_state, direction, priority = classify_rich_state(dx, dy)
        decision_key = (note, rich_state)

        # Full neural simulation only when the note / sensory condition changes.
        if decision_key != last_decision_key:
            if rich_state is None:
                last_scores = {a: 0.0 for a in ACTIONS}
                last_vector = (0.0, 0.0)
                last_winner = "CENTERED"
                last_brain_ms = 0.0
            else:
                t0 = time.perf_counter()

                last_scores = NEURAL_CACHE[
                    rich_state
                ]

                last_brain_ms = (
                    time.perf_counter()
                    - t0
                ) * 1000.0

                last_winner = max(
                    last_scores,
                    key=last_scores.get
                )
                last_vector = neural_motor_vector(
                    last_scores,
                    direction,
                    dx,
                    dy,
                )

            last_decision_key = decision_key

        if got_packet:
            tx.sendto(
                json.dumps(
                    {
                        "vx": last_vector[0],
                        "vy": last_vector[1],
                    }
                ).encode("utf-8"),
                (HOST, COMMAND_PORT),
            )

        now = time.perf_counter()

        if got_packet and now - last_print >= 0.05:
            enabled = bool(p.get("fly_enabled", False))
            state_text = rich_state if rich_state is not None else "CENTERED"

            print(
                f"\r"
                f"{'FLY ON ' if enabled else 'FLY OFF'} | "
                f"note={note:4d} | "
                f"hit={float(p['time_to_note']):7.1f}ms | "
                f"err=({dx:+.2f},{dy:+.2f}) | "
                f"state={state_text:25s} | "
                f"winner={last_winner:5s} | "
                f"vec=({last_vector[0]:+.2f},{last_vector[1]:+.2f}) | "
                f"NN={last_brain_ms:5.1f}ms",
                end="",
                flush=True,
            )

            last_print = now

        time.sleep(0.001)

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    rx.close()
    tx.close()
