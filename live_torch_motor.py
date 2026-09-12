import json
import sys
import time
from pathlib import Path

import pyarrow  # keep import order compatible with repo
import torch

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
CODE_DIR = BASE / "code"
PAPER_DIR = CODE_DIR / "paper-phil-drosophila"

sys.path.insert(0, str(CODE_DIR))

from run_pytorch import (
    TorchModel,
    MODEL_PARAMS,
    DT,
    get_weights,
    get_hash_tables,
)

ACTIONS = {
    "LEFT": [
        720575940616012061,
    ],
    "RIGHT": [
        720575940639182424,
    ],
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


class LiveTorchMotor:
    """
    Standalone CUDA motor-pathway validator.

    Uses the repository's native PyTorch full-connectome backend.
    This is intentionally separate from fly_actions.py until it passes
    reliability + speed validation.
    """

    def __init__(
        self,
        run_time_ms=40.0,
        stim_hz=250.0,
        device="cuda",
    ):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

        self.device = torch.device(device)
        self.run_time_ms = float(run_time_ms)
        self.stim_hz = float(stim_hz)
        self.num_steps = int(round(self.run_time_ms / DT))

        comp_path = DATA_DIR / "2025_Completeness_783.csv"
        con_path = DATA_DIR / "2025_Connectivity_783.parquet"

        print(f"Loading GPU motor connectome on {self.device}...")

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

        self.rates = torch.zeros(
            1,
            self.num_neurons,
            device=self.device,
        )

        self.action_indices = {}
        for action, fly_ids in ACTIONS.items():
            indices = []
            for fid in fly_ids:
                if fid not in self.flyid2i:
                    raise RuntimeError(f"Action neuron {fid} not found")
                indices.append(self.flyid2i[fid])
            self.action_indices[action] = indices

        atlas_path = BASE / "neuron_atlas.json"
        with open(atlas_path, "r", encoding="utf-8") as f:
            atlas = json.load(f)

        self.output_indices = {}
        for name in OUTPUT_NAMES:
            fid = int(atlas["output_neurons"][name]["id"])
            if fid not in self.flyid2i:
                raise RuntimeError(f"Output neuron {name} ({fid}) not found")
            self.output_indices[name] = self.flyid2i[fid]

        self.output_names = list(self.output_indices)
        self.output_index_tensor = torch.tensor(
            [self.output_indices[name] for name in self.output_names],
            dtype=torch.long,
            device=self.device,
        )

        # Warm CUDA kernels.
        self._run_once("LEFT")

        print(
            f"GPU motor ready: {self.num_neurons} neurons, "
            f"{self.num_steps} steps ({self.run_time_ms:g} ms), "
            f"stim={self.stim_hz:g} Hz"
        )

    def _run_once(self, action):
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")

        self.rates.zero_()
        self.rates[:, self.action_indices[action]] = self.stim_hz

        conductance, delay_buffer, spikes, v, refrac = self.model.state_init()

        counts = torch.zeros(
            len(self.output_names),
            dtype=torch.int32,
            device=self.device,
        )

        with torch.no_grad():
            for _ in range(self.num_steps):
                conductance, delay_buffer, spikes, v, refrac = self.model(
                    self.rates,
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

        counts_cpu = counts.cpu().tolist()

        output = {}
        duration_sec = self.run_time_ms / 1000.0

        for name, n_spikes in zip(self.output_names, counts_cpu):
            output[name] = {
                "spikes": int(n_spikes),
                "rate_hz": round(float(n_spikes) / duration_sec, 1),
            }

        return output

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
    dna_left = rate(out, "DNa02_left")
    dna_right = rate(out, "DNa02_right")

    p9_left = rate(out, "P9_oDN1_left")
    p9_right = rate(out, "P9_oDN1_right")

    mdns = [
        rate(out, "MDN_1"),
        rate(out, "MDN_2"),
        rate(out, "MDN_3"),
        rate(out, "MDN_4"),
    ]

    if action == "LEFT":
        return dna_left > dna_right and dna_left >= 20

    if action == "RIGHT":
        return dna_right > dna_left and dna_right >= 20

    if action == "UP":
        return (p9_left + p9_right) / 2 >= 10

    if action == "DOWN":
        return sum(mdns) / 4 >= 20

    return False
