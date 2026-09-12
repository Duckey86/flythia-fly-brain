import json
import socket
import time
from pathlib import Path

import torch
import pandas as pd

from trainable_torch_mbon_policy_16 import TrainableTorchMBONPolicy16


HOST = "127.0.0.1"

STATE_PORT = 50555
COMMAND_PORT = 50556

# Separate debug-only UDP stream.
# This never touches the working cursor-control port.
HUD_PORT = 50557
HUD_INTERVAL_S = 1.0 / 30.0

DEADZONE = 0.40
PRIORITY_RATIO = 1.35

ACTIONS = ("LEFT", "RIGHT", "UP", "DOWN")

ACTION_MBONS = {
    "LEFT":  "720575940617552340",
    "RIGHT": "720575940624185095",
    "UP":    "720575940623201833",
    "DOWN":  "720575940629422086",
}

BASE = Path(__file__).resolve().parent
PATTERN_FILE = BASE / "priority_kc_subpatterns_16.json"
COMP_FILE = BASE / "data" / "2025_Completeness_783.csv"


def axis(v):
    if v > DEADZONE:
        return 1
    if v < -DEADZONE:
        return -1
    return 0


def classify_rich_state(dx, dy):
    """
    Convert Rhythia error into the trained 16-state fly sensory code.

    Godot raw +Y is visually upward.
    The trained fly convention uses:
        sy = -1 -> UP
        sy = +1 -> DOWN
    """
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
    Keep the 16-state fly's categorical neural decision, while making
    diagonal movement follow the actual target angle.
    """
    sx, sy = direction

    if sx == 0 and sy == 0:
        return 0.0, 0.0

    winner = max(scores, key=scores.get)

    valid_actions = set()

    if sx < 0:
        valid_actions.add("LEFT")
    if sx > 0:
        valid_actions.add("RIGHT")
    if sy < 0:
        valid_actions.add("UP")
    if sy > 0:
        valid_actions.add("DOWN")

    # Do not secretly correct a genuinely wrong neural winner.
    if winner not in valid_actions:
        fallback = {
            "LEFT":  (-1.0, 0.0),
            "RIGHT": (+1.0, 0.0),
            "UP":    (0.0, -1.0),
            "DOWN":  (0.0, +1.0),
        }
        return fallback[winner]

    if sx == 0:
        return 0.0, float(sy)

    if sy == 0:
        return float(sx), 0.0

    ax = abs(dx)
    ay = abs(dy)
    peak = max(ax, ay, 1e-9)

    vx = float(sx) * (ax / peak)
    vy = float(sy) * (ay / peak)

    return vx, vy


# ============================================================
# BUILD FULL CUDA BRAIN + CACHE ALL 16 LEARNED RESPONSES
# ============================================================

print("Loading 16-state CUDA fly...")

brain = TrainableTorchMBONPolicy16(
    run_time_ms=3,
    device="cuda",
)

with open(
    PATTERN_FILE,
    "r",
    encoding="utf-8",
) as f:
    all_patterns = json.load(f)

# Map the model's internal neuron indices back to real FlyWire v783 root IDs.
# Root IDs are sent as STRINGS over JSON so 64-bit IDs never lose precision.
comp = pd.read_csv(COMP_FILE, index_col=0)
index_to_root_id = [str(int(fid)) for fid in comp.index]

STATE_KC_ROOT_IDS = {}
for state_key, pattern in all_patterns.items():
    ids = []
    for idx in pattern.get("kc_indices", []):
        idx = int(idx)
        if 0 <= idx < len(index_to_root_id):
            ids.append(index_to_root_id[idx])
    STATE_KC_ROOT_IDS[state_key] = ids

print(
    "Spatial morphology stream: ",
    len(set(rid for ids in STATE_KC_ROOT_IDS.values() for rid in ids)),
    "unique real KCs + 4 action MBONs",
)
print()

print("Precomputing 16 learned neural responses...")

NEURAL_CACHE = {}

cache_start = time.perf_counter()

for state_key, pattern in all_patterns.items():

    t0 = time.perf_counter()

    raw_scores = brain.get_scores(
        state_key
    )

    if brain.device.type == "cuda":
        torch.cuda.synchronize()

    # Convert once at startup so the HUD never causes GPU synchronization.
    scores = {
        action: float(raw_scores[action])
        for action in ACTIONS
    }

    NEURAL_CACHE[state_key] = scores

    winner = max(
        scores,
        key=scores.get,
    )

    elapsed = (
        time.perf_counter()
        - t0
    ) * 1000.0

    print(
        f"  {state_key:28s} "
        f"winner={winner:5s} "
        f"KCs={len(pattern.get('kc_indices', [])):2d} "
        f"{elapsed:5.1f}ms"
    )

print(
    "Neural cache ready in "
    f"{(time.perf_counter() - cache_start) * 1000.0:.1f}ms\n"
)


# ============================================================
# UDP
# ============================================================

rx = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM,
)

rx.bind(
    (
        HOST,
        STATE_PORT,
    )
)

rx.setblocking(False)

tx = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM,
)

hud_tx = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM,
)

print("Waiting for Rhythia...")
print("F10 still toggles fly control.")
print("Morphology HUD output: UDP 50557 @ 30 FPS")
print()


latest = None

last_decision_key = None

last_scores = {
    action: 0.0
    for action in ACTIONS
}

last_vector = (
    0.0,
    0.0,
)

last_winner = "CENTERED"
last_brain_ms = 0.0

last_print = 0.0
last_hud_send = 0.0


try:

    while True:

        got_packet = False

        # Drain stale state packets and keep only newest state.
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

        note = int(
            p["note"]
        )

        time_to_note = float(
            p.get(
                "time_to_note",
                0.0,
            )
        )

        rich_state, direction, priority = (
            classify_rich_state(
                dx,
                dy,
            )
        )

        decision_key = (
            note,
            rich_state,
        )

        # The expensive full neural response is already cached.
        if (
            decision_key
            != last_decision_key
        ):

            t0 = time.perf_counter()

            if rich_state is None:

                last_scores = {
                    action: 0.0
                    for action in ACTIONS
                }

                last_vector = (
                    0.0,
                    0.0,
                )

                last_winner = "CENTERED"

            else:

                last_scores = (
                    NEURAL_CACHE[
                        rich_state
                    ]
                )

                last_winner = max(
                    last_scores,
                    key=last_scores.get,
                )

                last_vector = (
                    neural_motor_vector(
                        last_scores,
                        direction,
                        dx,
                        dy,
                    )
                )

            last_brain_ms = (
                time.perf_counter()
                - t0
            ) * 1000.0

            last_decision_key = (
                decision_key
            )

        # Cheap geometry update every fresh Rhythia state packet.
        if rich_state is not None:

            last_vector = (
                neural_motor_vector(
                    last_scores,
                    direction,
                    dx,
                    dy,
                )
            )

        else:

            last_vector = (
                0.0,
                0.0,
            )

        # ----------------------------------------------------
        # GAMEPLAY COMMAND — SAME SIMPLE PACKET AS BEFORE
        # ----------------------------------------------------

        if got_packet:

            tx.sendto(
                json.dumps(
                    {
                        "vx":
                            last_vector[0],

                        "vy":
                            last_vector[1],
                    }
                ).encode(
                    "utf-8"
                ),
                (
                    HOST,
                    COMMAND_PORT,
                ),
            )

        now = time.perf_counter()

        # ----------------------------------------------------
        # SEPARATE HUD PACKET — DEBUG ONLY, THROTTLED TO 30 FPS
        # ----------------------------------------------------

        if (
            got_packet
            and
            now - last_hud_send
            >= HUD_INTERVAL_S
        ):

            if rich_state is None:
                active_kcs = []
            else:
                active_kcs = STATE_KC_ROOT_IDS.get(
                    rich_state,
                    [],
                )

            winner_mbon = ACTION_MBONS.get(
                last_winner,
                "",
            )

            hud_packet = {
                "fly_enabled":
                    bool(
                        p.get(
                            "fly_enabled",
                            False,
                        )
                    ),

                "note":
                    note,

                "time_to_note":
                    time_to_note,

                "dx":
                    dx,

                "dy":
                    dy,

                "state":
                    (
                        rich_state
                        if rich_state is not None
                        else "CENTERED"
                    ),

                "priority":
                    priority,

                "winner":
                    last_winner,

                "scores":
                    last_scores,

                "mbon_ids":
                    ACTION_MBONS,

                "vx":
                    last_vector[0],

                "vy":
                    last_vector[1],

                "kc_count":
                    len(active_kcs),

                # Real FlyWire v783 root IDs for the currently stimulated KCs.
                # Keep them as strings because they are 64-bit identifiers.
                "active_kcs":
                    active_kcs,

                "winner_mbon":
                    winner_mbon,

                "cached":
                    True,

                "decision_ms":
                    last_brain_ms,
            }

            hud_tx.sendto(
                json.dumps(
                    hud_packet
                ).encode(
                    "utf-8"
                ),
                (
                    HOST,
                    HUD_PORT,
                ),
            )

            last_hud_send = now

        # ----------------------------------------------------
        # TERMINAL STATUS
        # ----------------------------------------------------

        if (
            got_packet
            and
            now - last_print
            >= 0.05
        ):

            enabled = bool(
                p.get(
                    "fly_enabled",
                    False,
                )
            )

            state_text = (
                rich_state
                if rich_state is not None
                else "CENTERED"
            )

            print(
                f"\r"
                f"{'FLY ON ' if enabled else 'FLY OFF'} | "
                f"note={note:4d} | "
                f"hit={time_to_note:7.1f}ms | "
                f"err=({dx:+.2f},{dy:+.2f}) | "
                f"state={state_text:25s} | "
                f"winner={last_winner:5s} | "
                f"vec=({last_vector[0]:+.2f},{last_vector[1]:+.2f}) | "
                f"NN={last_brain_ms:4.1f}ms",
                end="",
                flush=True,
            )

            last_print = now

        time.sleep(
            0.001
        )

except KeyboardInterrupt:

    print()
    print("Stopped.")

finally:

    rx.close()
    tx.close()
    hud_tx.close()
