import json
import math
import socket
import time

from live_torch_mbon_policy import LiveTorchMBONPolicy

HOST = "127.0.0.1"
PORT = 50555

DEADZONE = 0.12


def axis(v):
    if v > DEADZONE:
        return 1

    if v < -DEADZONE:
        return -1

    return 0


names = {
    (-1, -1): "UPPER_LEFT",
    (0, -1): "UP",
    (1, -1): "UPPER_RIGHT",

    (-1, 0): "LEFT",
    (0, 0): "CENTERED",
    (1, 0): "RIGHT",

    (-1, 1): "LOWER_LEFT",
    (0, 1): "DOWN",
    (1, 1): "LOWER_RIGHT",
}


sock = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM
)

sock.bind((HOST, PORT))

print("Waiting for Rhythia...")
print("Loading fly brain...")

brain = LiveTorchMBONPolicy(
    run_time_ms=3,
    device="cuda"
)

print("Fly brain ready.")


while True:

    data, addr = sock.recvfrom(65535)

    p = json.loads(
        data.decode("utf-8")
    )

    dx = p["dx"]
    dy = p["dy"]

    sx = axis(dx)
    sy = -axis(dy)
    
    state = (sx, sy)

    distance = math.hypot(dx, dy)
    
    if state != (0, 0):

        start = time.perf_counter()

        scores = brain.get_scores(state)

        action = max(
            scores,
            key=scores.get
        )

        brain_ms = (
            time.perf_counter() - start
        ) * 1000

    else:

        action = "CENTERED"
        brain_ms = 0

    print(
        f"\r"
        f"note={p['note']:4d} | "
        f"hit in {p['time_to_note']:7.1f} ms | "
        f"cursor=({p['cursor_x']:+.2f},{p['cursor_y']:+.2f}) | "
        f"target=({p['target_x']:+.2f},{p['target_y']:+.2f}) | "
        f"error=({dx:+.2f},{dy:+.2f}) | "
        f"dist={distance:.2f} | "
        f"state={names[(sx, sy)]}"
        f"state={NAMES[state]} | "
        f"FLY={action} | "
        f"brain={brain_ms:.1f}ms",
        end=""
    )