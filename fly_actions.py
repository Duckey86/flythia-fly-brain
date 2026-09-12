import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from brian2 import (
    Network,
    PoissonInput,
    SpikeMonitor,
    Hz,
    ms,
)

# ------------------------------------------------
# Paths / original fly-brain model
# ------------------------------------------------

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
CODE_DIR = BASE / "code" / "paper-phil-drosophila"

sys.path.insert(0, str(CODE_DIR))

from model import create_model, default_params


# ------------------------------------------------
# Biological action neurons
# ------------------------------------------------

ACTIONS = {
    "LEFT": [
        720575940616012061,       # AOTU025 left
    ],

    "RIGHT": [
        720575940639182424,       # AOTU025 right
    ],

    "UP": [
        720575940627652358,
        720575940635872101,       # P9 forward pair
    ],

    "DOWN": [
        720575940616026939,
        720575940631082808,
        720575940640331472,
        720575940610236514,       # MDNs
    ],
}


OUTPUT_NAMES = [
    "DNa02_left",
    "DNa02_right",
    "P9_oDN1_left",
    "P9_oDN1_right",
    "MDN_1",
    "MDN_2",
    "MDN_3",
    "MDN_4",
]


# ------------------------------------------------
# Persistent Brian2 motor brain
# ------------------------------------------------

class PersistentMotorBrain:

    def __init__(
        self,
        freq_hz=200.0,
        duration_sec=0.1,
    ):

        self.freq_hz = float(freq_hz)
        self.duration_sec = float(duration_sec)

        self.lock = threading.Lock()

        print(
            "Building persistent Brian2 motor brain..."
        )

        start = time.perf_counter()

        comp_path = (
            DATA_DIR
            / "2025_Completeness_783.csv"
        )

        con_path = (
            DATA_DIR
            / "2025_Connectivity_783.parquet"
        )

        atlas_path = (
            BASE
            / "neuron_atlas.json"
        )

        self.df_comp = pd.read_csv(
            comp_path,
            index_col=0
        )

        self.flyid2i = {
            int(fid): i
            for i, fid
            in enumerate(
                self.df_comp.index
            )
        }

        with open(
            atlas_path,
            "r",
            encoding="utf-8"
        ) as f:

            atlas = json.load(f)

        # ----------------------------------------
        # Output neuron indices
        # ----------------------------------------

        self.output_indices = {}

        for name in OUTPUT_NAMES:

            fid = int(
                atlas[
                    "output_neurons"
                ][name]["id"]
            )

            if fid not in self.flyid2i:

                raise RuntimeError(
                    f"Output neuron {name} "
                    f"({fid}) not found "
                    f"in completeness table."
                )

            self.output_indices[name] = (
                self.flyid2i[fid]
            )

        # ----------------------------------------
        # Build full recurrent brain ONCE
        # ----------------------------------------

        self.params = dict(
            default_params
        )

        self.params["t_run"] = (
            self.duration_sec
            * 1000.0
            * ms
        )

        (
            self.neu,
            self.syn,
            _unused_full_monitor
        ) = create_model(
            str(comp_path),
            str(con_path),
            self.params
        )

        # We only record the 8 motor outputs that
        # Flythia actually checks.
        self.monitor = SpikeMonitor(
            self.neu,
            record=list(
                self.output_indices.values()
            ),
            name="flythia_motor_monitor"
        )

        # ----------------------------------------
        # Create all possible Poisson inputs ONCE
        #
        # They stay inactive until an action is run.
        # ----------------------------------------

        self.action_inputs = {}
        self.action_indices = {}

        all_inputs = []

        for action, fly_ids in ACTIONS.items():

            inputs = []
            indices = []

            for fid in fly_ids:

                if fid not in self.flyid2i:

                    raise RuntimeError(
                        f"Action neuron {fid} "
                        f"for {action} not found."
                    )

                idx = self.flyid2i[fid]

                p = PoissonInput(
                    target=self.neu[idx],
                    target_var="v",
                    N=1,
                    rate=self.freq_hz * Hz,
                    weight=(
                        self.params["w_syn"]
                        * self.params["f_poi"]
                    ),
                )

                p.active = False

                inputs.append(p)
                indices.append(idx)
                all_inputs.append(p)

            self.action_inputs[action] = (
                inputs
            )

            self.action_indices[action] = (
                indices
            )

        self.net = Network(
            self.neu,
            self.syn,
            self.monitor,
            *all_inputs
        )

        # Compile Brian2 code now rather than on
        # the first gameplay action.
        self.net.run(0 * ms)

        # Save pristine state:
        # resting neurons, empty spike monitor,
        # no delayed events pending.
        self.net.store(
            "flythia_motor_base"
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            "Persistent motor brain ready "
            f"in {elapsed:.3f}s."
        )

    # --------------------------------------------
    # Internal helpers
    # --------------------------------------------

    def _disable_all_inputs(self):

        for inputs in (
            self.action_inputs.values()
        ):

            for p in inputs:
                p.active = False

    def _prepare_action(self, action):

        # Restore neurons, monitor, time,
        # synaptic-delay queues, etc.
        #
        # Keep RNG moving forward so each trial
        # still gets fresh Poisson spikes.
        self.net.restore(
            "flythia_motor_base",
            restore_random_state=False
        )

        self._disable_all_inputs()

        # Original model gives stimulated neurons
        # zero refractory period.
        self.neu.rfc = (
            self.params["t_rfc"]
        )

        for idx in (
            self.action_indices[action]
        ):

            self.neu[idx].rfc = 0 * ms

        for p in (
            self.action_inputs[action]
        ):
            p.active = True

    def _collect_output(self):

        spike_indices = np.asarray(
            self.monitor.i[:],
            dtype=np.int64
        )

        output = {}

        for (
            name,
            neuron_index
        ) in self.output_indices.items():

            spikes = int(
                np.count_nonzero(
                    spike_indices
                    == neuron_index
                )
            )

            rate_hz = (
                spikes
                / self.duration_sec
            )

            output[name] = {
                "spikes": spikes,
                "rate_hz": round(
                    rate_hz,
                    1
                ),
            }

        return output

    # --------------------------------------------
    # Public motor simulation
    # --------------------------------------------

    def run(self, action):

        if action not in ACTIONS:

            raise ValueError(
                f"Unknown action: {action}"
            )

        with self.lock:

            start = time.perf_counter()

            self._prepare_action(
                action
            )

            self.net.run(
                self.duration_sec
                * 1000.0
                * ms
            )

            output = (
                self._collect_output()
            )

            elapsed = (
                time.perf_counter()
                - start
            )

        return {
            "status": "success",
            "duration_sec": (
                self.duration_sec
            ),
            "wall_time_sec": elapsed,
            "output_neuron_activity": (
                output
            ),
        }


# ------------------------------------------------
# Build once when fly_actions is imported
# ------------------------------------------------

_motor_brain = PersistentMotorBrain(
    freq_hz=250.0,
    duration_sec=0.04
)


# ------------------------------------------------
# Existing Flythia interface
# ------------------------------------------------

def rate(out, name):

    return (
        out
        .get(name, {})
        .get("rate_hz", 0.0)
    )


def execute_action(action):

    result = _motor_brain.run(
        action
    )

    out = result[
        "output_neuron_activity"
    ]

    dna_left = rate(
        out,
        "DNa02_left"
    )

    dna_right = rate(
        out,
        "DNa02_right"
    )

    p9_left = rate(
        out,
        "P9_oDN1_left"
    )

    p9_right = rate(
        out,
        "P9_oDN1_right"
    )

    mdns = [
        rate(out, "MDN_1"),
        rate(out, "MDN_2"),
        rate(out, "MDN_3"),
        rate(out, "MDN_4"),
    ]

    print()
    print(
        "ACTION:",
        action
    )

    print(
        "DNa02:",
        dna_left,
        dna_right
    )

    print(
        "P9:",
        p9_left,
        p9_right
    )

    print(
        "MDNs:",
        mdns
    )

    print(
        "Persistent motor sim:",
        f"{result['wall_time_sec'] * 1000:.1f} ms"
    )

    # Confirm that the intended biological
    # pathway actually fired.

    if action == "LEFT":

        success = (
            dna_left > dna_right
            and dna_left >= 20
        )

        movement = (
            -1,
            0
        )

    elif action == "RIGHT":

        success = (
            dna_right > dna_left
            and dna_right >= 20
        )

        movement = (
            1,
            0
        )

    elif action == "UP":

        success = (
            (
                p9_left
                + p9_right
            )
            / 2
            >= 10
        )

        movement = (
            0,
            -1
        )

    elif action == "DOWN":

        success = (
            sum(mdns)
            / 4
            >= 20
        )

        movement = (
            0,
            1
        )

    else:

        return (
            0,
            0,
            result
        )

    if success:

        return (
            movement[0],
            movement[1],
            result
        )

    return (
        0,
        0,
        result
    )
