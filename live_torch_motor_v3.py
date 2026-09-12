import json
import sys
import time
from pathlib import Path

import pyarrow  # keep repo import order compatible
import torch

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
CODE_DIR = BASE / "code"

sys.path.insert(0, str(CODE_DIR))

from run_pytorch import (
    TorchModel,
    MODEL_PARAMS,
    DT,
    get_weights,
    get_hash_tables,
)

ACTIONS = {
    "LEFT": [720575940616012061],
    "RIGHT": [720575940639182424],
    "UP": [
        720575940627652358,
        720575940635872101,
    ],
    "DOWN": [
        720575940616026939,
        720575940631082808,
        720575940640331472,
        720575940610236514,
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


class LiveTorchMotorV3:
    """
    CUDA motor simulation with Brian2-style external Poisson stimulation.

    Key change from v1/v2:
    Brian2's model.poi() uses PoissonInput(target_var='v') with
        weight = w_syn * f_poi = 0.275 * 250 = 68.75 mV
    so an external Poisson event changes the stimulated neuron's membrane
    voltage directly.

    V3 reproduces that direct-v injection. Recurrent connectome propagation
    still uses the repository's PyTorch alpha-synapse/delay implementation.
    """

    def __init__(self, run_time_ms=40.0, stim_hz=250.0, device="cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

        self.device = torch.device(device)
        self.run_time_ms = float(run_time_ms)
        self.stim_hz = float(stim_hz)
        self.num_steps = int(round(self.run_time_ms / DT))

        # Probability of one PoissonInput event in one DT=0.1 ms step.
        self.poisson_p = self.stim_hz * DT / 1000.0

        # Exact Brian2 PoissonInput voltage jump:
        # params['w_syn'] * params['f_poi'] = 0.275 * 250 = 68.75 mV
        self.direct_v_jump = (
            float(MODEL_PARAMS["wScale"])
            * float(MODEL_PARAMS["scalePoisson"])
        )

        self.w_scale = float(MODEL_PARAMS["wScale"])

        comp_path = DATA_DIR / "2025_Completeness_783.csv"
        con_path = DATA_DIR / "2025_Connectivity_783.parquet"

        print(
            f"Loading GPU motor v3 on {self.device} "
            f"({self.run_time_ms:g} ms @ {self.stim_hz:g} Hz)..."
        )

        self.flyid2i, _ = get_hash_tables(str(comp_path))

        weights = get_weights(
            str(con_path),
            str(comp_path),
            str(DATA_DIR),
            csr=True,
        ).to(self.device)

        self.num_neurons = int(weights.shape[0])

        self.model = TorchModel(
            1,
            self.num_neurons,
            DT,
            MODEL_PARAMS,
            weights,
            device=str(self.device),
        )

        # Cache transpose once.
        self.weights_t = self.model.weights.transpose(0, 1)

        self.action_index_tensors = {}
        for action, fly_ids in ACTIONS.items():
            idx = [self.flyid2i[fid] for fid in fly_ids]
            self.action_index_tensors[action] = torch.tensor(
                idx,
                dtype=torch.long,
                device=self.device,
            )

        with open(BASE / "neuron_atlas.json", "r", encoding="utf-8") as f:
            atlas = json.load(f)

        self.output_indices = {
            name: self.flyid2i[int(atlas["output_neurons"][name]["id"])]
            for name in OUTPUT_NAMES
        }

        self.output_names = list(self.output_indices)

        self.output_index_tensor = torch.tensor(
            [self.output_indices[name] for name in self.output_names],
            dtype=torch.long,
            device=self.device,
        )

        # Warm CUDA kernels once.
        self._run_once("LEFT")

        print(
            f"GPU motor v3 ready: {self.num_neurons} neurons, "
            f"{self.num_steps} steps, direct Poisson jump "
            f"{self.direct_v_jump:.2f} mV"
        )

    def _run_once(self, action):
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")

        action_idx = self.action_index_tensors[action]

        conductance, delay_buffer, spikes, v, refrac = self.model.state_init()

        counts = torch.zeros(
            len(self.output_names),
            dtype=torch.int32,
            device=self.device,
        )

        with torch.no_grad():
            for _ in range(self.num_steps):
                # Whole-connectome recurrent propagation.
                weighted_spikes = torch.matmul(
                    spikes,
                    self.weights_t,
                )

                # --------------------------------------------------------
                # Brian2-style PoissonInput(target_var='v')
                # --------------------------------------------------------
                #
                # Only 1-4 action neurons get external Poisson events.
                # An event jumps membrane voltage directly by 68.75 mV.
                #
                stim_events = (
                    torch.rand(
                        action_idx.numel(),
                        device=self.device,
                    ) < self.poisson_p
                )

                if stim_events.any():
                    v[0, action_idx[stim_events]] += self.direct_v_jump

                # Recurrent connections keep the normal repository
                # alpha-synapse + 1.8 ms delay dynamics.
                conductance, delay_buffer, spikes, v, refrac = self.model.neurons(
                    self.w_scale * weighted_spikes,
                    conductance,
                    delay_buffer,
                    spikes,
                    v,
                    refrac,
                )

                counts += (
                    spikes[0, self.output_index_tensor] > 0
                ).to(torch.int32)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        duration_sec = self.run_time_ms / 1000.0
        counts_cpu = counts.cpu().tolist()

        out = {}

        for name, n_spikes in zip(self.output_names, counts_cpu):
            out[name] = {
                "spikes": int(n_spikes),
                "rate_hz": round(float(n_spikes) / duration_sec, 1),
            }

        return out

    def run(self, action):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        out = self._run_once(action)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - t0

        return {
            "status": "success",
            "duration_sec": self.run_time_ms / 1000.0,
            "wall_time_sec": elapsed,
            "output_neuron_activity": out,
        }


def rate(out, name):
    return out.get(name, {}).get("rate_hz", 0.0)


def motor_success(action, out):
    dl = rate(out, "DNa02_left")
    dr = rate(out, "DNa02_right")

    pl = rate(out, "P9_oDN1_left")
    pr = rate(out, "P9_oDN1_right")

    mdns = [
        rate(out, "MDN_1"),
        rate(out, "MDN_2"),
        rate(out, "MDN_3"),
        rate(out, "MDN_4"),
    ]

    if action == "LEFT":
        return dl > dr and dl >= 20

    if action == "RIGHT":
        return dr > dl and dr >= 20

    if action == "UP":
        return (pl + pr) / 2 >= 10

    if action == "DOWN":
        return sum(mdns) / 4 >= 20

    return False
