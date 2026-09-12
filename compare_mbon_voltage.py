import sys
import pandas as pd
from pathlib import Path

from brian2 import (
    Network,
    StateMonitor,
    SpikeMonitor,
    ms,
    mV,
)

from dopamine_learning import (
    FlyMemory,
    apply_learned_weights,
)

from fly_mb_policy import (
    FlyMBPolicy,
    ACTION_MBONS,
    STATES,
)


BASE = Path(__file__).resolve().parent

CODE_DIR = (
    BASE
    / "code"
    / "paper-phil-drosophila"
)

sys.path.insert(
    0,
    str(CODE_DIR)
)

from model import (
    create_model,
    default_params,
)


COMP_FILE = (
    BASE
    / "data"
    / "2025_Completeness_783.csv"
)

CON_FILE = (
    BASE
    / "data"
    / "2025_Connectivity_783.parquet"
)


# ------------------------------------------------
# ID mapping
# ------------------------------------------------

comp = pd.read_csv(
    COMP_FILE,
    index_col=0
)

flyid2i = {
    int(fid): i
    for i, fid in enumerate(comp.index)
}


brain = FlyMBPolicy()
memory = FlyMemory()


ACTION_INDICES = {
    action: flyid2i[fly_id]
    for action, fly_id
    in ACTION_MBONS.items()
}


# ------------------------------------------------
# Run one state
# ------------------------------------------------

def run_state(state, learned):

    params = dict(
        default_params
    )

    params["t_run"] = 50 * ms

    neu, syn, old_spike_monitor = create_model(
        str(COMP_FILE),
        str(CON_FILE),
        params,
    )

    if learned:
        modified = apply_learned_weights(
            syn,
            memory
        )
    else:
        modified = 0

    # -----------------------------------------
    # Make this state's anatomical KCs fire once
    # -----------------------------------------

    kcs = brain.get_state_kcs(
        state
    )

    for kc_idx in kcs:
        neu.v[kc_idx] = -44 * mV

    # -----------------------------------------
    # Monitor four action MBONs
    # -----------------------------------------

    monitor_indices = [
        ACTION_INDICES[action]
        for action in ACTION_MBONS
    ]

    state_monitor = StateMonitor(
        neu,
        ["v", "g"],
        record=monitor_indices
    )

    spike_monitor = SpikeMonitor(
        neu
    )

    net = Network(
        neu,
        syn,
        state_monitor,
        spike_monitor
    )

    net.run(
        params["t_run"]
    )

    results = {}

    for monitor_row, action in enumerate(
        ACTION_MBONS
    ):

        idx = ACTION_INDICES[
            action
        ]

        voltages = (
            state_monitor.v[
                monitor_row
            ] / mV
        )

        conductance = (
            state_monitor.g[
                monitor_row
            ] / mV
        )

        spike_count = int(
            spike_monitor.count[
                idx
            ]
        )

        results[action] = {
            "spikes": spike_count,

            "peak_v": float(
                max(voltages)
            ),

            "mean_v": float(
                sum(voltages)
                / len(voltages)
            ),

            # g is the synaptic drive in this model
            "peak_g": float(
                max(conductance)
            ),

            "mean_g": float(
                sum(conductance)
                / len(conductance)
            ),
        }

    return results, modified


# ------------------------------------------------
# Compare
# ------------------------------------------------

for state in STATES:

    print()
    print("=" * 90)
    print("STATE:", state)
    print("=" * 90)

    baseline, _ = run_state(
        state,
        learned=False
    )

    learned, modified = run_state(
        state,
        learned=True
    )

    expected, scores = brain.choose_action(
        state,
        epsilon=0.0
    )

    print(
        "Expected:",
        expected
    )

    print(
        "Stored scores:",
        {
            k: round(v, 3)
            for k, v in scores.items()
        }
    )

    print(
        "Learned synapses applied:",
        modified
    )

    print()

    for action in ACTION_MBONS:

        before = baseline[
            action
        ]

        after = learned[
            action
        ]

        delta_g = (
            after["peak_g"]
            - before["peak_g"]
        )

        delta_v = (
            after["peak_v"]
            - before["peak_v"]
        )

        print(
            f"{action:5s} | "
            f"spikes {before['spikes']} -> {after['spikes']} | "
            f"peak g {before['peak_g']:.4f} -> "
            f"{after['peak_g']:.4f} "
            f"(Δ {delta_g:+.4f}) | "
            f"peak v {before['peak_v']:.4f} -> "
            f"{after['peak_v']:.4f} "
            f"(Δ {delta_v:+.4f})"
        )