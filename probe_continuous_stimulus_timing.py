#!/usr/bin/env python3
"""
probe_continuous_stimulus_timing.py

Quick diagnostic for the Stage-3 continuous KC stimulus.

Why this exists:
The old encoder put selected KCs at v=-44 mV, so they spiked essentially
immediately. The continuous-drive encoder starts KCs near rest and lets
sensory current push them toward threshold. With the original 3 ms run,
a KC can spike too late for its signal to survive the model's 1.8 ms
synaptic delay and reach the action MBONs before the simulation ends.

This probe tests several simulation lengths and sensory gains WITHOUT
training or modifying any memory file.

It starts from:
    data/fly_memory_CONTINUOUS_2deg.json
"""

import math
from pathlib import Path

from train_continuous_kc_stimulus_1deg import (
    ContinuousKCStimulusTrainer,
    load_checkpoint_into_brain,
)

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "data" / "fly_memory_CONTINUOUS_2deg.json"

# Small representative angle set: cardinal, diagonal, and between-bin angles.
ANGLES = [
    0.0, 15.0, 30.0, 45.0,
    60.0, 75.0, 90.0, 120.0,
    150.0, 180.0, 225.0, 270.0,
    315.0, 345.0,
]

RUN_TIMES_MS = [3.0, 4.5, 6.0, 8.0]
GAINS = [400.0, 850.0, 1200.0, 1600.0, 2200.0, 3000.0]

DT_MS = 0.1


def evaluate_combo(brain, run_ms, gain):
    brain.num_steps = max(1, int(round(run_ms / DT_MS)))
    brain.run_time_ms = float(run_ms)
    brain.stimulus_gain = float(gain)
    brain.clear_continuous_caches()

    nonzero = 0
    errors = []
    magnitudes = []

    for angle in ANGLES:
        learned, _, _ = brain._run_continuous_stimulus(
            angle,
            learned=True,
        )
        baseline, _, _ = brain._run_continuous_stimulus(
            angle,
            learned=False,
        )

        scores = {
            action: learned[action] - baseline[action]
            for action in brain.action_order
        }

        x = float(scores["RIGHT"]) - float(scores["LEFT"])
        y = float(scores["DOWN"]) - float(scores["UP"])
        mag = math.hypot(x, y)
        magnitudes.append(mag)

        if mag > 1e-12:
            nonzero += 1
            pred = math.degrees(math.atan2(y, x)) % 360.0
            err = abs((pred - angle + 180.0) % 360.0 - 180.0)
            errors.append(err)

    mean_error = (
        sum(errors) / len(errors)
        if errors
        else 180.0
    )
    mean_mag = sum(magnitudes) / len(magnitudes)

    return nonzero, mean_error, mean_mag


def main():
    print()
    print("Loading Stage-3 timing probe...")
    print("No training. No files will be overwritten.")
    print()

    brain = ContinuousKCStimulusTrainer(
        run_time_ms=3.0,
        device="cuda",
        learning_rate=0.003,
        min_weight=0.25,
        max_weight=4.0,
        stimulus_gain=400.0,
    )

    load_checkpoint_into_brain(
        brain,
        SOURCE,
    )

    print()
    print(
        "run_ms  gain    nonzero   mean_error(nonzero)   mean_vector_mag"
    )
    print("-" * 70)

    best = None

    for run_ms in RUN_TIMES_MS:
        for gain in GAINS:
            nonzero, mean_error, mean_mag = evaluate_combo(
                brain,
                run_ms,
                gain,
            )

            print(
                f"{run_ms:5.1f}  "
                f"{gain:6.0f}  "
                f"{nonzero:2d}/{len(ANGLES):2d}      "
                f"{mean_error:8.2f} deg          "
                f"{mean_mag:.6g}"
            )

            candidate = (
                -nonzero,
                mean_error,
                -mean_mag,
                run_ms,
                gain,
            )

            if best is None or candidate < best[0]:
                best = (
                    candidate,
                    run_ms,
                    gain,
                    nonzero,
                    mean_error,
                    mean_mag,
                )

        print()

    _, run_ms, gain, nonzero, mean_error, mean_mag = best

    print("=" * 70)
    print("BEST NON-TRAINING COMBINATION")
    print("=" * 70)
    print(f"run_time_ms: {run_ms}")
    print(f"stimulus_gain: {gain}")
    print(f"nonzero angles: {nonzero}/{len(ANGLES)}")
    print(f"mean error over nonzero angles: {mean_error:.2f} deg")
    print(f"mean vector magnitude: {mean_mag:.6g}")
    print()
    print(
        "Send this BEST block back before running another long training job."
    )


if __name__ == "__main__":
    main()
