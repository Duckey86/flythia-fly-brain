import sys
import pandas as pd
from pathlib import Path

from brian2 import Network, ms, mV

from dopamine_learning import (
    FlyMemory,
    apply_learned_weights,
)

from fly_mb_policy import (
    ACTION_MBONS,
    STATES,
    FlyMBPolicy,
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


# ---------------------------------------------
# Load neuron index mapping
# ---------------------------------------------

comp = pd.read_csv(
    BASE
    / "data"
    / "2025_Completeness_783.csv",
    index_col=0
)

flyid2i = {
    int(fid): i
    for i, fid
    in enumerate(comp.index)
}


brain = FlyMBPolicy()
memory = FlyMemory()


# ---------------------------------------------
# Test every state
# ---------------------------------------------

for state in STATES:

    print()
    print("=" * 70)
    print("STATE:", state)
    print("=" * 70)

    # Anatomical KC population representing state
    kc_indices = brain.get_state_kcs(
        state
    )

    params = dict(
        default_params
    )

    # Short neural response
    params["t_run"] = 50 * ms

    neu, syn, monitor = create_model(
        str(
            BASE
            / "data"
            / "2025_Completeness_783.csv"
        ),
        str(
            BASE
            / "data"
            / "2025_Connectivity_783.parquet"
        ),
        params,
    )

    # -----------------------------------------
    # Apply learned REAL KC -> MBON weights
    # -----------------------------------------

    modified = apply_learned_weights(
        syn,
        memory
    )

    print(
        "Learned synapses applied:",
        modified
    )

    # -----------------------------------------
    # Make state KCs fire ONCE
    # -----------------------------------------

    for kc_idx in kc_indices:

        # threshold = -45 mV
        # putting at -44 causes one initial spike
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
    # Count actual action-MBON spikes
    # -----------------------------------------

    action_spikes = {}

    for action, fly_id in (
        ACTION_MBONS.items()
    ):

        idx = flyid2i[
            fly_id
        ]

        count = len(
            spikes.get(
                idx,
                []
            )
        )

        action_spikes[action] = count

    print(
        "Actual MBON spikes:",
        action_spikes
    )

    best = max(
        action_spikes,
        key=action_spikes.get
    )

    best_count = action_spikes[
        best
    ]

    if best_count == 0:

        print(
            "RESULT: no action MBON fired"
        )

    else:

        print(
            "MBON WINNER:",
            best
        )

    # Compare against our current Python policy
    policy_action, scores = (
        brain.choose_action(
            state,
            epsilon=0.0
        )
    )

    print(
        "Expected policy:",
        policy_action
    )

    print(
        "Stored scores:",
        {
            k: round(v, 3)
            for k, v
            in scores.items()
        }
    )