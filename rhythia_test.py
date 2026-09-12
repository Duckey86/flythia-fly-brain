import tkinter as tk
import random
import threading
import time

from live_torch_mbon_policy import LiveTorchMBONPolicy
from fly_actions import execute_action


WIDTH = 600
HEIGHT = 600
CELL = WIDTH // 3

# ------------------------------------------------
# Load neural memory
# ------------------------------------------------

brain = LiveTorchMBONPolicy(
    run_time_ms=3,
    device="cuda"
)

# ------------------------------------------------
# Window
# ------------------------------------------------

root = tk.Tk()
root.title("Flythia - Mushroom Body Fly Brain")

canvas = tk.Canvas(
    root,
    width=WIDTH,
    height=HEIGHT,
    bg="black"
)
canvas.pack()

# Grid
for i in range(1, 3):

    canvas.create_line(
        i * CELL,
        0,
        i * CELL,
        HEIGHT,
        fill="white",
        width=2
    )

    canvas.create_line(
        0,
        i * CELL,
        WIDTH,
        i * CELL,
        fill="white",
        width=2
    )


# ------------------------------------------------
# State
# ------------------------------------------------

player_x = 1
player_y = 1

target_x = 0
target_y = 0

score = 0
moves = 0

ai_running = False
brain_busy = False


def sign(v):

    if v < 0:
        return -1

    if v > 0:
        return 1

    return 0


def get_state():

    return (
        sign(target_x - player_x),
        sign(target_y - player_y)
    )


def cell_center(x, y):

    return (
        x * CELL + CELL / 2,
        y * CELL + CELL / 2
    )


# ------------------------------------------------
# Fly cursor
# ------------------------------------------------

px, py = cell_center(
    player_x,
    player_y
)

cursor = canvas.create_oval(
    px - 14,
    py - 14,
    px + 14,
    py + 14,
    fill="cyan",
    outline=""
)


def draw_player():

    px, py = cell_center(
        player_x,
        player_y
    )

    canvas.coords(
        cursor,
        px - 14,
        py - 14,
        px + 14,
        py + 14
    )


# ------------------------------------------------
# Target
# ------------------------------------------------

target = None


def new_target():

    global target_x
    global target_y
    global target

    while True:

        target_x = random.randint(
            0, 2
        )

        target_y = random.randint(
            0, 2
        )

        if (
            target_x != player_x
            or target_y != player_y
        ):
            break

    tx, ty = cell_center(
        target_x,
        target_y
    )

    if target is not None:
        canvas.delete(target)

    target = canvas.create_oval(
        tx - 25,
        ty - 25,
        tx + 25,
        ty + 25,
        outline="red",
        width=5
    )

    update_labels()


# ------------------------------------------------
# UI
# ------------------------------------------------

status_label = tk.Label(
    root,
    text="Mushroom-body AI stopped",
    font=("Arial", 14)
)
status_label.pack()


score_label = tk.Label(
    root,
    text="Score: 0",
    font=("Arial", 14)
)
score_label.pack()


info_label = tk.Label(
    root,
    text="",
    font=("Arial", 12)
)
info_label.pack()


mbon_label = tk.Label(
    root,
    text="",
    font=("Arial", 11)
)
mbon_label.pack()


def update_labels():

    score_label.config(
        text=f"Score: {score}"
    )

    info_label.config(
        text=(
            f"Player: ({player_x},{player_y})   "
            f"Target: ({target_x},{target_y})   "
            f"Moves: {moves}"
        )
    )


# ------------------------------------------------
# Mushroom-body decision
# ------------------------------------------------

def choose_action():

    state = get_state()

    # ZERO exploration.
    # Only use learned mushroom-body memory.
    
    action, scores = brain.choose_action(
        state,
        epsilon=0.0
    )

    return state, action, scores


# ------------------------------------------------
# AI loop
# ------------------------------------------------

def ai_step():

    global brain_busy

    if not ai_running:
        return

    if brain_busy:
        return

    state, action, scores = (
        choose_action()
    )

    pretty_scores = "   ".join(
        f"{name}:{value:.2f}"
        for name, value
        in scores.items()
    )

    mbon_label.config(
        text=pretty_scores
    )

    status_label.config(
        text=(
            f"State {state} → "
            f"{action} | brain running..."
        )
    )

    print()
    print("=" * 60)

    print(
        "State:",
        state
    )

    print(
        "MBON scores:",
        {
            name: round(value, 3)
            for name, value
            in scores.items()
        }
    )

    print(
        "Chosen action:",
        action
    )

    brain_busy = True

    threading.Thread(
        target=run_motor_action,
        args=(
            state,
            action,
            scores
        ),
        daemon=True
    ).start()


def run_motor_action(
    state,
    action,
    scores
):

    try:

        # Actual whole-brain motor simulation
        mx, my, result = (
            execute_action(
                action
            )
        )

        root.after(
            0,
            lambda: finish_action(
                state,
                action,
                scores,
                mx,
                my
            )
        )

    except Exception as error:

        root.after(
            0,
            lambda: brain_error(
                str(error)
            )
        )


def brain_error(error):

    global brain_busy

    brain_busy = False

    status_label.config(
        text=f"Brain error: {error}"
    )


def finish_action(
    state,
    action,
    scores,
    mx,
    my
):

    global player_x
    global player_y
    global score
    global moves
    global brain_busy

    brain_busy = False

    # Motor pathway failed
    if mx == 0 and my == 0:

        print(
            "Motor pathway failed."
        )

        status_label.config(
            text=(
                f"{action} selected, "
                f"but motor neurons "
                f"did not fire"
            )
        )

        if ai_running:

            root.after(
                300,
                ai_step
            )

        return

    # Move one grid cell
    player_x += mx
    player_y += my

    player_x = max(
        0,
        min(2, player_x)
    )

    player_y = max(
        0,
        min(2, player_y)
    )

    moves += 1

    draw_player()
    update_labels()

    print(
        "Brain movement:",
        (mx, my)
    )

    print(
        "New player:",
        (player_x, player_y)
    )

    # Target reached
    if (
        player_x == target_x
        and player_y == target_y
    ):

        score += 1

        update_labels()

        status_label.config(
            text="TARGET REACHED!"
        )

        print(
            "TARGET REACHED"
        )

        if ai_running:

            root.after(
                600,
                continue_after_target
            )

        return

    status_label.config(
        text=(
            f"{action} → "
            f"({player_x},{player_y})"
        )
    )

    if ai_running:

        root.after(
            300,
            ai_step
        )


def continue_after_target():

    if not ai_running:
        return

    new_target()

    root.after(
        300,
        ai_step
    )


# ------------------------------------------------
# Buttons
# ------------------------------------------------

def start_ai():

    global ai_running

    if ai_running:
        return

    ai_running = True

    status_label.config(
        text="Mushroom-body AI started"
    )

    ai_step()


def stop_ai():

    global ai_running

    ai_running = False

    status_label.config(
        text="AI stopped"
    )


controls = tk.Frame(root)
controls.pack(pady=10)


tk.Button(
    controls,
    text="START MB AI",
    font=("Arial", 12),
    command=start_ai
).grid(
    row=0,
    column=0,
    padx=5
)


tk.Button(
    controls,
    text="STOP AI",
    font=("Arial", 12),
    command=stop_ai
).grid(
    row=0,
    column=1,
    padx=5
)


tk.Button(
    controls,
    text="NEW TARGET",
    font=("Arial", 12),
    command=new_target
).grid(
    row=0,
    column=2,
    padx=5
)


# ------------------------------------------------
# Start
# ------------------------------------------------

new_target()
update_labels()

root.mainloop()