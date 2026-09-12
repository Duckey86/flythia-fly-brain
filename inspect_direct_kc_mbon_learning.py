import sys
import json
import numpy as np
from pathlib import Path
from brian2 import mV

from dopamine_learning import FlyMemory
from fly_mb_policy import FlyMBPolicy, ACTION_MBONS, STATES

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "code" / "paper-phil-drosophila"))

from model import create_model, default_params

COMP_FILE = BASE / "data" / "2025_Completeness_783.csv"
CON_FILE = BASE / "data" / "2025_Connectivity_783.parquet"
CAL_FILE = BASE / "mbon_calibration.json"

VALID_ACTIONS = {
    (-1, -1): {"LEFT", "UP"},
    (0, -1): {"UP"},
    (1, -1): {"RIGHT", "UP"},
    (-1, 0): {"LEFT"},
    (1, 0): {"RIGHT"},
    (-1, 1): {"LEFT", "DOWN"},
    (0, 1): {"DOWN"},
    (1, 1): {"RIGHT", "DOWN"},
}

brain = FlyMBPolicy()
memory = FlyMemory()
mods = memory.get_weight_multipliers()

with open(CAL_FILE, "r", encoding="utf-8") as f:
    calibration = json.load(f)

print("Building Brian2 connectivity once...")
params = dict(default_params)
neu, syn, spike_monitor = create_model(
    str(COMP_FILE),
    str(CON_FILE),
    params,
)

pre = np.asarray(syn.i[:], dtype=np.int64)
post = np.asarray(syn.j[:], dtype=np.int64)
base_w = np.asarray(syn.w[:] / mV, dtype=float)

print("Total saved memory multipliers:", len(mods))
print()

ratio_correct = 0
final_drive_correct = 0

for state in STATES:
    key = f"{state[0]},{state[1]}"
    valid = VALID_ACTIONS[state]

    print("=" * 100)
    print("STATE:", state, "| VALID:", ", ".join(sorted(valid)))
    print("=" * 100)

    policy_scores = brain.get_scores(state)
    rows = {}

    for action in ACTION_MBONS:
        mbon_idx = brain.action_mbon_indices[action]

        kc_indices = [
            int(c["kc_index"])
            for c in brain.real_map[key][action]["connections"]
        ]

        mask = np.isin(pre, kc_indices) & (post == mbon_idx)
        syn_idx = np.flatnonzero(mask)

        if len(syn_idx) == 0:
            rows[action] = {
                "count": 0,
                "modified": 0,
                "base": 0.0,
                "cal": 0.0,
                "learned": 0.0,
                "delta": 0.0,
                "ratio": 0.0,
                "policy": float(policy_scores[action]),
            }
            continue

        w0 = base_w[syn_idx]
        factor = float(calibration[key][action]["factor"])
        w_cal = w0 * factor

        mults = np.array([
            float(mods.get((int(pre[i]), int(post[i])), 1.0))
            for i in syn_idx
        ])

        w_learned = w_cal * mults

        base_drive = float(np.sum(np.abs(w0)))
        cal_drive = float(np.sum(np.abs(w_cal)))
        learned_drive = float(np.sum(np.abs(w_learned)))

        ratio = learned_drive / cal_drive if cal_drive != 0 else 0.0

        rows[action] = {
            "count": len(syn_idx),
            "modified": int(np.sum(~np.isclose(mults, 1.0))),
            "base": base_drive,
            "cal": cal_drive,
            "learned": learned_drive,
            "delta": learned_drive - cal_drive,
            "ratio": ratio,
            "policy": float(policy_scores[action]),
        }

    for action, r in rows.items():
        marker = " <-- VALID" if action in valid else ""
        print(
            f"{action:5s} | "
            f"edges={r['count']:2d} "
            f"learned_edges={r['modified']:2d} | "
            f"base={r['base']:8.3f} "
            f"cal={r['cal']:8.3f} "
            f"learned={r['learned']:8.3f} "
            f"delta={r['delta']:+8.3f} "
            f"ratio={r['ratio']:6.3f} | "
            f"policy_score={r['policy']:6.3f}"
            f"{marker}"
        )

    ratio_winner = max(rows, key=lambda a: rows[a]["ratio"])
    final_winner = max(rows, key=lambda a: rows[a]["learned"])

    print()
    print(
        "RATIO WINNER:",
        ratio_winner,
        "CORRECT" if ratio_winner in valid else "WRONG",
    )
    print(
        "FINAL DIRECT-DRIVE WINNER:",
        final_winner,
        "CORRECT" if final_winner in valid else "WRONG",
    )
    print()

    ratio_correct += int(ratio_winner in valid)
    final_drive_correct += int(final_winner in valid)

print("=" * 100)
print("FINAL SUMMARY")
print("=" * 100)
print(f"Learned/base ratio correct: {ratio_correct}/{len(STATES)}")
print(f"Calibrated learned direct-drive correct: {final_drive_correct}/{len(STATES)}")
print()
print(
    "If these are high while the full-network MBON test is low, "
    "the memory is on the correct real KC->MBON synapses and the remaining "
    "problem is downstream/recurrent network dynamics."
)
