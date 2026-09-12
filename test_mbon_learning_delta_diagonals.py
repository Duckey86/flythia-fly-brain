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
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
]

VALID_ACTIONS = {
    (-1, -1): {"LEFT", "UP"},
    (1, -1): {"RIGHT", "UP"},
    (-1, 1): {"LEFT", "DOWN"},
    (1, 1): {"RIGHT", "DOWN"},
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

        synapse_indices = np.flatnonzero(mask)

        if len(synapse_indices) > 0:
            syn.w[synapse_indices] = (
                syn.w[synapse_indices] * factor
            )
            changed += len(synapse_indices)

    return changed


def run_state(state, learned):
    params = dict(default_params)
    params["t_run"] = 50 * ms

    neu, syn, spike_monitor = create_model(
        str(COMP_FILE),
        str(CON_FILE),
        params,
    )

    n_calibrated = apply_state_calibration(
        syn,
        state,
    )

    n_learned = 0
    if learned:
        n_learned = apply_learned_weights(
            syn,
            memory,
        )

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
        dtype=float,
    )

    results = {}

    for row, action in enumerate(ACTION_MBONS):
        idx = ACTION_INDICES[action]

        g = np.asarray(
            state_monitor.g[row] / mV,
            dtype=float,
        )

        positive_g = np.maximum(g, 0.0)

        integrated_g = (
            float(np.trapz(positive_g, t_ms))
            if len(t_ms) >= 2
            else 0.0
        )

        results[action] = {
            "spikes": int(spike_monitor.count[idx]),
            "peak_g": float(np.max(g)),
            "integrated_g": integrated_g,
        }

    return results, n_calibrated, n_learned


print()
print("=" * 100)
print("DIAGONAL LEARNED - UNLEARNED REAL MBON ACTIVITY")
print("=" * 100)

peak_correct = 0
integrated_correct = 0

for state in TEST_STATES:
    valid = VALID_ACTIONS[state]

    print()
    print("=" * 75)
    print("STATE:", state)
    print("VALID:", ", ".join(sorted(valid)))
    print("=" * 75)

    baseline, n_cal, _ = run_state(
        state,
        learned=False,
    )

    learned, _, n_learned = run_state(
        state,
        learned=True,
    )

    print("Calibration synapses:", n_cal)
    print("Learned synapses applied:", n_learned)
    print()

    deltas = {}

    for action in ACTION_MBONS:
        deltas[action] = {
            "spikes": (
                learned[action]["spikes"]
                - baseline[action]["spikes"]
            ),
            "peak_g": (
                learned[action]["peak_g"]
                - baseline[action]["peak_g"]
            ),
            "integrated_g": (
                learned[action]["integrated_g"]
                - baseline[action]["integrated_g"]
            ),
        }

        marker = "  <-- VALID" if action in valid else ""

        print(
            f"{action:5s} | "
            f"spikes {baseline[action]['spikes']}->{learned[action]['spikes']} "
            f"(Δ {deltas[action]['spikes']:+d}) | "
            f"peak_g {baseline[action]['peak_g']:7.3f}->{learned[action]['peak_g']:7.3f} "
            f"(Δ {deltas[action]['peak_g']:+8.3f}) | "
            f"int_g {baseline[action]['integrated_g']:8.3f}->{learned[action]['integrated_g']:8.3f} "
            f"(Δ {deltas[action]['integrated_g']:+8.3f})"
            f"{marker}"
        )

    peak_winner = max(
        deltas,
        key=lambda a: deltas[a]["peak_g"],
    )

    integrated_winner = max(
        deltas,
        key=lambda a: deltas[a]["integrated_g"],
    )

    print()
    print(
        "BIGGEST ΔPEAK_G:",
        peak_winner,
        "CORRECT" if peak_winner in valid else "WRONG",
    )

    print(
        "BIGGEST ΔINTEGRATED_G:",
        integrated_winner,
        "CORRECT" if integrated_winner in valid else "WRONG",
    )

    peak_correct += int(peak_winner in valid)
    integrated_correct += int(integrated_winner in valid)

print()
print("=" * 100)
print("FINAL SUMMARY")
print("=" * 100)
print(
    "ΔPeak-g valid:",
    f"{peak_correct}/{len(TEST_STATES)}",
)
print(
    "ΔIntegrated-g valid:",
    f"{integrated_correct}/{len(TEST_STATES)}",
)
