import json
import socket
import time

import torch

from live_torch_mbon_policy import LiveTorchMBONPolicy


HOST = "127.0.0.1"

STATE_PORT = 50555
COMMAND_PORT = 50556

DEADZONE = 0.12


NAMES = {

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


def axis(v):

    if v > DEADZONE:
        return 1

    if v < -DEADZONE:
        return -1

    return 0


# ---------------------------------
# LOAD ACTUAL FLY BRAIN
# ---------------------------------

print("Loading fly brain...")


brain = LiveTorchMBONPolicy(
    run_time_ms=3,
    device="cuda"
)


print("Fly brain ready.")


# ---------------------------------
# UDP
# ---------------------------------

rx = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM
)

rx.bind(
    (
        HOST,
        STATE_PORT
    )
)

rx.setblocking(False)


tx = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM
)


print()
print("Waiting for Rhythia...")
print("F10 = enable/disable fly control")
print()


latest = None

last_decision = None

last_action = "CENTERED"

last_brain_ms = 0


while True:

    got_packet = False


    # ---------------------------------
    # GET NEWEST RHYTHIA STATE
    # ---------------------------------

    while True:

        try:

            data, _ = rx.recvfrom(
                65535
            )

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


    dx = float(
        p["dx"]
    )

    dy = float(
        p["dy"]
    )


    # Rhythia X matches fly X.
    sx = axis(
        dx
    )


    # Rhythia world +Y is visually UP.
    # Fly learned -1 as UP.
    sy = -axis(
        dy
    )


    state = (
        sx,
        sy
    )


    note = int(
        p["note"]
    )


    # ---------------------------------
    # ACTUAL MBON DECISION
    # ---------------------------------

    decision_key = (
        note,
        state
    )


    if decision_key != last_decision:

        if state == (0, 0):

            last_action = (
                "CENTERED"
            )

            last_brain_ms = 0


        else:

            start = (
                time.perf_counter()
            )


            scores = (
                brain.get_scores(
                    state
                )
            )


            torch.cuda.synchronize()


            last_brain_ms = (
                time.perf_counter()
                - start
            ) * 1000


            last_action = max(
                scores,
                key=scores.get
            )


        last_decision = (
            decision_key
        )


    # ---------------------------------
    # SEND MOTOR INTENT TO RHYTHIA
    # ---------------------------------

    if got_packet:

        command = {

            "action":
                last_action
        }


        tx.sendto(

            json.dumps(
                command
            ).encode("utf-8"),

            (
                HOST,
                COMMAND_PORT
            )
        )


        enabled = bool(
            p.get(
                "fly_enabled",
                False
            )
        )


        print(

            f"\r"

            f"{'FLY ON ' if enabled else 'FLY OFF'} | "

            f"note={note:4d} | "

            f"hit="
            f"{float(p['time_to_note']):7.1f}ms | "

            f"err=("
            f"{dx:+.2f},"
            f"{dy:+.2f}) | "

            f"state="
            f"{NAMES[state]:11s} | "

            f"brain="
            f"{last_action:8s} | "

            f"NN="
            f"{last_brain_ms:5.1f}ms",

            end="",
            flush=True
        )


    time.sleep(
        0.001
    )