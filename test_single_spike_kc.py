import sys
import json
import pandas as pd

from pathlib import Path

from brian2 import (
    Network,
    ms,
    mV,
)

from dopamine_learning import FlyMemory


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


TESTS = {
    "DP1m_A": 720575940618308825,
    "DP1m_B": 720575940622726271,

    "DM1_B": 720575940619071005,

    "DC1_A": 720575940637056887,
    "DC1_B": 720575940621529435,

    "VA2": 720575940611079236,
    "DM4": 720575940615366055,

    # These were interesting in the sparse test
    "DM2_A": 720575940630024566,
    "DL5_A": 720575940617207185,
    "VC1":   720575940637526190,
}


# ------------------------------------------------
# FlyWire ID -> Brian index
# ------------------------------------------------

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


# ------------------------------------------------
# KC indices
# ------------------------------------------------

memory = FlyMemory()

kc_indices = set(
    memory.kc_indices
)


# ------------------------------------------------
# Test each neuron
# ------------------------------------------------

patterns = {}


for name, fly_id in TESTS.items():

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    if fly_id not in flyid2i:

        print("Neuron not found.")
        patterns[name] = set()
        continue

    stim_idx = flyid2i[
        fly_id
    ]

    params = dict(
        default_params
    )

    params["t_run"] = (
        100 * ms
    )

    neu, syn, monitor = (
        create_model(
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
    )

    # --------------------------------------------
    # Give this neuron ONE initial spike.
    #
    # Threshold is -45 mV.
    # Put it just above threshold before simulation.
    # --------------------------------------------

    neu.v[stim_idx] = (
        -44 * mV
    )

    net = Network(
        neu,
        syn,
        monitor
    )

    net.run(
        params["t_run"]
    )

    spike_train = (
        get_spk_trn(
            monitor
        )
    )

    active_kcs = {
        idx
        for idx in spike_train
        if idx in kc_indices
    }

    patterns[name] = (
        active_kcs
    )

    print(
        "Active KCs:",
        len(active_kcs)
    )

    print(
        "Total active neurons:",
        len(spike_train)
    )

    print(
        "First KCs:",
        sorted(
            active_kcs
        )[:30]
    )


# ------------------------------------------------
# Pattern overlap
# ------------------------------------------------

print()
print("=" * 70)
print("SINGLE-SPIKE KC OVERLAP")
print("=" * 70)

names = list(
    patterns
)

for i in range(
    len(names)
):

    for j in range(
        i + 1,
        len(names)
    ):

        a_name = names[i]
        b_name = names[j]

        a = patterns[a_name]
        b = patterns[b_name]

        union = a | b

        if not union:

            similarity = 0.0

        else:

            similarity = (
                len(a & b)
                / len(union)
            )

        print(
            f"{a_name:8s} vs "
            f"{b_name:8s}: "
            f"{similarity:.3f}"
        )
print()
print("=" * 70)
print("KC COUNT SUMMARY")
print("=" * 70)

for name, pattern in patterns.items():
    print(
        f"{name:8s}: "
        f"{len(pattern)} active KCs"
    )