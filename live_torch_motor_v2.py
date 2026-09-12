import json
import sys
import time
from pathlib import Path

import pyarrow  # keep import order compatible with repo
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


class LiveTorchMotorV2:
    """
    Faster CUDA motor simulator.

    Improvements over v1:
    - caches the transposed sparse weight matrix once
    - does NOT generate a 138k-neuron dense Bernoulli tensor every timestep
    - generates Poisson spikes only for the 1-4 stimulated action neurons
    - reuses the dense external-input tensor
    """

    def __init__(self, run_time_ms=40.0, stim_hz=250.0, device="cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

        self.device = torch.device(device)
        self.run_time_ms = float(run_time_ms)
        self.stim_hz = float(stim_hz)
        self.num_steps = int(round(self.run_time_ms / DT))
        self.poisson_p = self.stim_hz * DT / 1000.0
        self.poisson_scale = float(MODEL_PARAMS["scalePoisson"])
        self.w_scale = float(MODEL_PARAMS["wScale"])

        comp_path = DATA_DIR / "2025_Completeness_783.csv"
        con_path = DATA_DIR / "2025_Connectivity_783.parquet"

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

        # Avoid rebuilding transpose metadata every step.
        self.weights_t = self.model.weights.transpose(0, 1)

        self.action_indices = {}
        self.action_index_tensors = {}

        for action, fly_ids in ACTIONS.items():
            idx = [self.flyid2i[fid] for fid in fly_ids]
            self.action_indices[action] = idx
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

        # Reused external current vector.
        self.external = torch.zeros(
            1,
            self.num_neurons,
            device=self.device,
        )

        # Warm up CUDA.
        self._run_once("LEFT")

        print(
            f"GPU motor v2 ready: {self.num_neurons} neurons, "
            f"{self.num_steps} steps ({self.run_time_ms:g} ms), "
            f"stim={self.stim_hz:g} Hz"
        )

    def _run_once(self, action):
        action_idx = self.action_index_tensors[action]

        conductance, delay_buffer, spikes, v, refrac = self.model.state_init()

        counts = torch.zeros(
            len(self.output_names),
            dtype=torch.int32,
            device=self.device,
        )

        with torch.no_grad():
            for _ in range(self.num_steps):
                # Recurrent whole-connectome propagation.
                weighted_spikes = torch.matmul(
                    spikes,
                    self.weights_t,
                )

                # External Poisson input ONLY for stimulated action neurons.
                self.external.zero_()

                stim_spikes = (
                    torch.rand(
                        action_idx.numel(),
                        device=self.device,
                    ) < self.poisson_p
                ).to(torch.float32)

                self.external[0, action_idx] = (
                    stim_spikes * self.poisson_scale
                )

                conductance, delay_buffer, spikes, v, refrac = self.model.neurons(
                    self.w_scale * (self.external + weighted_spikes),
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
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")

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
    mdns = [rate(out, f"MDN_{i}") for i in range(1, 5)]

    if action == "LEFT":
        return dl > dr and dl >= 20
    if action == "RIGHT":
        return dr > dl and dr >= 20
    if action == "UP":
        return (pl + pr) / 2 >= 10
    if action == "DOWN":
        return sum(mdns) / 4 >= 20
    return False
