import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

from brian2 import Network, StateMonitor, ms, mV

from dopamine_learning import FlyMemory, apply_learned_weights
from fly_mb_policy import FlyMBPolicy, ACTION_MBONS, STATES

BASE = Path(__file__).resolve().parent
CODE_DIR = BASE / "code" / "paper-phil-drosophila"
sys.path.insert(0, str(CODE_DIR))

from model import create_model, default_params

COMP_FILE = BASE / "data" / "2025_Completeness_783.csv"
CON_FILE = BASE / "data" / "2025_Connectivity_783.parquet"
CALIBRATION_FILE = BASE / "mbon_calibration.json"

# For diagonal states, either axis-reducing move is valid.
VALID_ACTIONS = {
    (-1, -1): {"LEFT", "UP"},
    (0, -1):  {"UP"},
    (1, -1):  {"RIGHT", "UP"},
    (-1, 0):  {"LEFT"},
    (1, 0):   {"RIGHT"},
    (-1, 1):  {"LEFT", "DOWN"},
    (0, 1):   {"DOWN"},
    (1, 1):   {"RIGHT", "DOWN"},
}

brain = FlyMBPolicy()
memory = FlyMemory()

with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
    calibration = json.load(f)

comp = pd.read_csv(COMP_FILE, index_col=0)

flyid2i = {
    int(fid): i
    for i, fid in enumerate(comp.index)
}

ACTION_INDICES = {
    action: flyid2i[fly_id]
    for action, fly_id in ACTION_MBONS.items()
}


def apply_state_calibration(syn, state):
    state_key = f"{state[0]},{state[1]}"

    pre_arr = np.asarray(syn.i[:], dtype=np.int64)
    post_arr = np.asarray(syn.j[:], dtype=np.int64)

    changed = 0

    for action in ACTION_MBONS:
        factor = float(
            calibration[state_key][action]["factor"]
        )

        mbon_idx = brain.action_mbon_indices[action]

        kc_indices = [
            int(connection["kc_index"])
            for connection
            in brain.real_map[state_key][action]["connections"]
        ]

        if not kc_indices:
            continue

        mask = (
            np.isin(pre_arr, kc_indices)
            & (post_arr == mbon_idx)
        )

        if mask.any():
            syn.w[mask] = syn.w[mask] * factor
            changed += int(mask.sum())

    return changed


def run_state(state):
    params = dict(default_params)
    params["t_run"] = 50 * ms

    neu, syn, spike_monitor = create_model(
        str(COMP_FILE),
        str(CON_FILE),
        params,
    )

    # Equalize baseline KC -> MBON drive for this state.
    calibrated = apply_state_calibration(
        syn,
        state,
    )

    # Apply the saved learned multipliers on top.
    learned = apply_learned_weights(
        syn,
        memory,
    )

    # Activate this state's anatomical KCs once.
    for kc_idx in brain.get_state_kcs(state):
        neu.v[kc_idx] = -44 * mV

    monitor_indices = [
        ACTION_INDICES[action]
        for action in ACTION_MBONS
    ]

    state_monitor = StateMonitor(
        neu,
        ["v", "g"],
        record=monitor_indices,
    )

    net = Network(
        neu,
        syn,
        spike_monitor,
        state_monitor,
    )

    net.run(params["t_run"])

    t_ms = np.asarray(
        state_monitor.t / ms,
        dtype=float
    )

    results = {}

    for row, action in enumerate(ACTION_MBONS):
        idx = ACTION_INDICES[action]

        g = np.asarray(
            state_monitor.g[row] / mV,
            dtype=float
        )

        v = np.asarray(
            state_monitor.v[row] / mV,
            dtype=float
        )

        # Positive synaptic drive only.
        positive_g = np.maximum(g, 0.0)

        if len(t_ms) >= 2:
            integrated_g = float(
                np.trapz(
                    positive_g,
                    t_ms
                )
            )
        else:
            integrated_g = 0.0

        results[action] = {
            "spikes": int(
                spike_monitor.count[idx]
            ),
            "peak_g": float(
                np.max(g)
            ),
            "integrated_g": integrated_g,
            "peak_v": float(
                np.max(v)
            ),
        }

    return results, calibrated, learned


print()
print("=" * 100)
print("ALL-STATE REAL MBON ACTIVITY TEST")
print("=" * 100)

peak_correct = 0
integrated_correct = 0
spike_correct = 0
unique_spike_states = 0

for state in STATES:
    valid = VALID_ACTIONS[state]

    print()
    print("=" * 75)
    print("STATE:", state)
    print(
        "VALID MOVE(S):",
        ", ".join(sorted(valid))
    )
    print("=" * 75)

    results, n_calibrated, n_learned = run_state(state)

    print(
        "Calibrated real synapses:",
        n_calibrated
    )
    print(
        "Learned synapses applied:",
        n_learned
    )
    print()

    for action, data in results.items():
        marker = (
            "  <-- VALID"
            if action in valid
            else ""
        )

        print(
            f"{action:5s} | "
            f"spikes={data['spikes']:2d} | "
            f"peak_g={data['peak_g']:8.3f} mV | "
            f"int_g={data['integrated_g']:9.3f} mV*ms | "
            f"peak_v={data['peak_v']:8.3f} mV"
            f"{marker}"
        )

    peak_winner = max(
        results,
        key=lambda a: results[a]["peak_g"]
    )

    integrated_winner = max(
        results,
        key=lambda a: results[a]["integrated_g"]
    )

    max_spikes = max(
        data["spikes"]
        for data in results.values()
    )

    spike_winners = [
        action
        for action, data in results.items()
        if data["spikes"] == max_spikes
    ]

    print()
    print(
        "PEAK-G WINNER:",
        peak_winner,
        "CORRECT" if peak_winner in valid else "WRONG"
    )

    print(
        "INTEGRATED-G WINNER:",
        integrated_winner,
        "CORRECT" if integrated_winner in valid else "WRONG"
    )

    if peak_winner in valid:
        peak_correct += 1

    if integrated_winner in valid:
        integrated_correct += 1

    if (
        max_spikes > 0
        and len(spike_winners) == 1
    ):
        unique_spike_states += 1
        spike_winner = spike_winners[0]

        print(
            "SPIKE WINNER:",
            spike_winner,
            "CORRECT" if spike_winner in valid else "WRONG"
        )

        if spike_winner in valid:
            spike_correct += 1
    else:
        print(
            "SPIKE WINNER: tie / no unique winner",
            spike_winners
        )


print()
print("=" * 100)
print("FINAL SUMMARY")
print("=" * 100)

print(
    "Peak-g correct:",
    f"{peak_correct}/{len(STATES)}"
)

print(
    "Integrated-g correct:",
    f"{integrated_correct}/{len(STATES)}"
)

print(
    "Unique spike winners:",
    f"{unique_spike_states}/{len(STATES)}"
)

print(
    "Correct unique spike winners:",
    f"{spike_correct}/{len(STATES)}"
)
