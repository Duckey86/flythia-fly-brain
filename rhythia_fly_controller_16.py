import json
import socket
import time

import torch

from trainable_torch_mbon_policy_16 import TrainableTorchMBONPolicy16


HOST = "127.0.0.1"
STATE_PORT = 50555
COMMAND_PORT = 50556

# Keep the cursor comfortably inside the note hitbox instead of wasting time
# chasing exact centre. This matches the setting that worked well for the
# 8-state controller.
DEADZONE = 0.40

# Must match the threshold used by train_gpu_fly_16state.py.
PRIORITY_RATIO = 1.35


def axis(v):
    if v > DEADZONE:
        return 1
    if v < -DEADZONE:
        return -1
    return 0


def classify_rich_state(dx, dy):
    """
    Convert raw Rhythia error into one of the 16 neural sensory states.

    Rhythia/Godot raw +Y is visually upward, while the trained fly state
    convention uses -1 for UP and +1 for DOWN, so Y is inverted here.
    """
    sx = axis(dx)
    sy = -axis(dy)

    if sx == 0 and sy == 0:
        return None, (0, 0), "CENTERED"

    # Cardinal state: one axis already lies inside the deadzone.
    if sx == 0 or sy == 0:
        key = f"{sx},{sy}"
        return key, (sx, sy), "CARDINAL"

    # Diagonal state: encode which axis needs more work.
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


print("Loading 16-state CUDA fly...")
brain = TrainableTorchMBONPolicy16(
    run_time_ms=3,
    device="cuda",
)
print("16-state fly ready.\n")

rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rx.bind((HOST, STATE_PORT))
rx.setblocking(False)

tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Waiting for Rhythia...")
print("Use your existing in-game F10 toggle for fly control.\n")

latest = None
last_decision_key = None
last_action = "CENTERED"
last_brain_ms = 0.0
last_print = 0.0

try:
    while True:
        got_packet = False

        # Drain stale ~60 Hz state packets and keep only the newest one.
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

        # Re-run the expensive neural simulation only when the note or
        # 16-state sensory condition changes.
        decision_key = (note, rich_state)

        if decision_key != last_decision_key:
            if rich_state is None:
                last_action = "CENTERED"
                last_brain_ms = 0.0
            else:
                t0 = time.perf_counter()
                scores = brain.get_scores(rich_state)
                if brain.device.type == "cuda":
                    torch.cuda.synchronize()
                last_brain_ms = (time.perf_counter() - t0) * 1000.0
                last_action = max(scores, key=scores.get)

            last_decision_key = decision_key

        if got_packet:
            tx.sendto(
                json.dumps({"action": last_action}).encode("utf-8"),
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
                f"brain={last_action:8s} | "
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
