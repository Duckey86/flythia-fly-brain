import sys
import numpy as np

from pathlib import Path
from brian2 import mV

from dopamine_learning import (
    FlyMemory,
    apply_learned_weights
)


BASE = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(
        BASE
        / "code"
        / "paper-phil-drosophila"
    )
)

from model import (
    create_model,
    default_params
)


COMP = (
    BASE
    / "data"
    / "2025_Completeness_783.csv"
)

CON = (
    BASE
    / "data"
    / "2025_Connectivity_783.parquet"
)


memory = FlyMemory()

params = dict(
    default_params
)

neu, syn, monitor = create_model(
    str(COMP),
    str(CON),
    params
)


before = np.asarray(
    syn.w[:] / mV,
    dtype=float
).copy()


reported = apply_learned_weights(
    syn,
    memory
)


after = np.asarray(
    syn.w[:] / mV,
    dtype=float
)


changed = np.flatnonzero(
    ~np.isclose(
        before,
        after
    )
)


print(
    "Function reported:",
    reported
)

print(
    "Weights actually changed:",
    len(changed)
)


print()
print("FIRST 10 CHANGES")

for idx in changed[:10]:

    print(
        idx,
        ":",
        round(before[idx], 4),
        "->",
        round(after[idx], 4),
        "mV"
    )