import json
import socket
import time
import torch

from live_torch_mbon_policy import LiveTorchMBONPolicy

HOST = "127.0.0.1"
PORT = 50555

DEADZONE = 0.12

NAMES = {
    (-1, -1): "UPPER_LEFT",
    (0, -1):  "UP",
    (1, -1):  "UPPER_RIGHT",

    (-1, 0):  "LEFT",
    (0, 0):   "CENTERED",
    (1, 0):   "RIGHT",

    (-1, 1):  "LOWER_LEFT",
    (0, 1):   "DOWN",
    (1, 1):   "LOWER_RIGHT",
}


def axis(v):
    if v > DEADZONE:
        return 1

    if v < -DEADZONE:
        return -1

    return 0


# -------------------------
# LOAD REAL FLY BRAIN
# -------------------------

print("Loading trained fly brain...")

brain = LiveTorchMBONPolicy(
    run_time_ms=3,
    device="cuda"
)

print("Fly brain ready.\n")


# -------------------------
# RHYTHIA UDP
# -------------------------

sock = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM
)

sock.bind((HOST, PORT))

# Important:
# Rhythia sends ~1 packet every 16 ms,
# while the brain takes ~30-50 ms.
#
# We only want the NEWEST packet.
sock.setblocking(False)


latest = None

last_decision = None
last_action = "WAIT"
last_brain_ms = 0

print("Waiting for Rhythia...")
print("Play normally. Fly is observing only.\n")


while True:

    got_packet = False

    # Drain queued packets.
    # Keep only the newest game state.
    while True:

        try:
            data, _ = sock.recvfrom(65535)

            latest = json.loads(
                data.decode("utf-8")
            )

            got_packet = True

        except BlockingIOError:
            break


    if latest is None:
        time.sleep(0.001)
        continue


    p = latest

    dx = float(p["dx"])
    dy = float(p["dy"])


    # X is already correct.
    sx = axis(dx)

    # Rhythia +Y = visually UP.
    # Fly uses -1 = UP.
    sy = -axis(dy)

    state = (sx, sy)

    note = int(p["note"])


    # -------------------------
    # RUN FLY BRAIN
    # -------------------------

    decision_key = (
        note,
        state
    )


    # Don't run another 40 ms simulation if
    # literally nothing relevant changed.
    if decision_key != last_decision:

        if state == (0, 0):

            last_action = "CENTERED"
            last_brain_ms = 0

        else:

            start = time.perf_counter()

            scores = brain.get_scores(state)

            torch.cuda.synchronize()

            last_brain_ms = (
                time.perf_counter() - start
            ) * 1000

            last_action = max(
                scores,
                key=scores.get
            )


        last_decision = decision_key


    # -------------------------
    # DISPLAY
    # -------------------------

    if got_packet:

        print(
            f"\r"

            f"note={note:4d} | "

            f"hit in "
            f"{float(p['time_to_note']):7.1f}ms | "

            f"cursor=("
            f"{float(p['cursor_x']):+.2f},"
            f"{float(p['cursor_y']):+.2f}) | "

            f"target=("
            f"{float(p['target_x']):+.2f},"
            f"{float(p['target_y']):+.2f}) | "

            f"state="
            f"{NAMES[state]:11s} | "

            f"FLY="
            f"{last_action:8s} | "

            f"brain="
            f"{last_brain_ms:5.1f}ms",

            end="",
            flush=True
        )


    time.sleep(0.001)