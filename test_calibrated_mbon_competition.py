import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

from brian2 import Network, StateMonitor, ms, mV

from dopamine_learning import FlyMemory, apply_learned_weights
from fly_mb_policy import FlyMBPolicy, ACTION_MBONS

BASE = Path(__file__).resolve().parent
CODE_DIR = BASE / "code" / "paper-phil-drosophila"
sys.path.insert(0, str(CODE_DIR))

from model import create_model, default_params

COMP_FILE = BASE / "data" / "2025_Completeness_783.csv"
CON_FILE = BASE / "data" / "2025_Connectivity_783.parquet"
CALIBRATION_FILE = BASE / "mbon_calibration.json"

TEST_STATES = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
]

EXPECTED = {
    (-1, 0): "LEFT",
    (1, 0): "RIGHT",
    (0, -1): "UP",
    (0, 1): "DOWN",
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

    calibrated = apply_state_calibration(syn, state)
    learned = apply_learned_weights(syn, memory)

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

    results = {}

    for row, action in enumerate(ACTION_MBONS):
        idx = ACTION_INDICES[action]

        spike_count = int(spike_monitor.count[idx])

        peak_g = float(
            np.max(np.asarray(state_monitor.g[row] / mV))
        )

        peak_v = float(
            np.max(np.asarray(state_monitor.v[row] / mV))
        )

        results[action] = {
            "spikes": spike_count,
            "peak_g": peak_g,
            "peak_v": peak_v,
        }

    return results, calibrated, learned


print()
print("=" * 90)
print("CALIBRATED REAL-MBON COMPETITION")
print("=" * 90)

correct_by_spikes = 0
clear_spike_winners = 0

for state in TEST_STATES:
    expected = EXPECTED[state]

    print()
    print("=" * 70)
    print("STATE:", state)
    print("EXPECTED:", expected)
    print("=" * 70)

    results, n_calibrated, n_learned = run_state(state)

    print("Calibrated real synapses:", n_calibrated)
    print("Learned synapses applied:", n_learned)
    print()

    for action, data in results.items():
        marker = "  <-- EXPECTED" if action == expected else ""

        print(
            f"{action:5s} | "
            f"spikes={data['spikes']:2d} | "
            f"peak_g={data['peak_g']:8.3f} mV | "
            f"peak_v={data['peak_v']:8.3f} mV"
            f"{marker}"
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

    if max_spikes > 0 and len(spike_winners) == 1:
        winner = spike_winners[0]
        clear_spike_winners += 1

        print()
        print("ACTUAL SPIKE WINNER:", winner)

        if winner == expected:
            correct_by_spikes += 1
            print("CORRECT")
        else:
            print("WRONG")

    else:
        drive_winner = max(
            results,
            key=lambda action: results[action]["peak_g"]
        )

        print()
        print("NO UNIQUE SPIKE WINNER.")
        print("Peak-g winner:", drive_winner)

print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)
print(
    "Unique spike winners:",
    f"{clear_spike_winners}/{len(TEST_STATES)}"
)
print(
    "Correct actual spike winners:",
    f"{correct_by_spikes}/{len(TEST_STATES)}"
)
