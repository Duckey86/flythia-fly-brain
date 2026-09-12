import sys
import pandas as pd
from pathlib import Path

from brian2 import Network, ms, mV

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
    get_spk_trn,
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
# FlyWire ID -> Brian2 index
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


# ------------------------------------------------
# Run one neural simulation
# ------------------------------------------------

def run_state(state, learned):

    params = dict(
        default_params
    )

    params["t_run"] = 50 * ms

    neu, syn, monitor = create_model(
        str(COMP_FILE),
        str(CON_FILE),
        params,
    )

    modified = 0

    # Apply learned KC -> MBON weights
    if learned:

        modified = apply_learned_weights(
            syn,
            memory
        )

    # -----------------------------------------
    # Activate the anatomical KCs for this state
    # -----------------------------------------

    kc_indices = brain.get_state_kcs(
        state
    )

    for kc_idx in kc_indices:

        # One initial spike
        neu.v[kc_idx] = -44 * mV

    net = Network(
        neu,
        syn,
        monitor
    )

    net.run(
        params["t_run"]
    )

    spikes = get_spk_trn(
        monitor
    )

    # -----------------------------------------
    # Count spikes in four action MBONs
    # -----------------------------------------

    result = {}

    for action, fly_id in ACTION_MBONS.items():

        idx = flyid2i[
            fly_id
        ]

        result[action] = len(
            spikes.get(
                idx,
                []
            )
        )

    return result, modified


# ------------------------------------------------
# Compare baseline vs learned
# ------------------------------------------------

print()
print("=" * 90)
print("REAL MBON LEARNING COMPARISON")
print("=" * 90)


correct_improvements = 0


for state in STATES:

    print()
    print("=" * 70)
    print("STATE:", state)
    print("=" * 70)

    # Original brain
    baseline, _ = run_state(
        state,
        learned=False
    )

    # Brain with learned synaptic multipliers
    learned, modified = run_state(
        state,
        learned=True
    )

    expected, stored_scores = (
        brain.choose_action(
            state,
            epsilon=0.0
        )
    )

    print(
        "Expected action:",
        expected
    )

    print(
        "Stored scores:",
        {
            k: round(v, 3)
            for k, v
            in stored_scores.items()
        }
    )

    print()
    print(
        "UNLEARNED:",
        baseline
    )

    print(
        "LEARNED:  ",
        learned
    )

    print(
        "Applied learned synapses:",
        modified
    )

    # -----------------------------------------
    # Show change caused by learning
    # -----------------------------------------

    delta = {
        action:
            learned[action]
            - baseline[action]
        for action in ACTION_MBONS
    }

    print(
        "CHANGE:   ",
        delta
    )

    expected_delta = delta[
        expected
    ]

    if expected_delta > 0:

        correct_improvements += 1

        print(
            f"GOOD: {expected} gained "
            f"{expected_delta} spike(s)"
        )

    elif expected_delta < 0:

        print(
            f"BAD: {expected} lost "
            f"{abs(expected_delta)} spike(s)"
        )

    else:

        print(
            f"NO EFFECT on expected "
            f"{expected} MBON"
        )


print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)

print(
    "States where expected MBON "
    "increased after learning:",
    f"{correct_improvements}/{len(STATES)}"
)