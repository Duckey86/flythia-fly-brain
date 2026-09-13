#!/usr/bin/env python3
"""
train_smooth_circular_kc_1deg.py

Stage 4: smooth circular population coding for Flythia.

Why:
The two-population continuous encoder produced large errors near 22.5-degree
handoff points and one severe ~50/50 cancellation near 56 degrees.

This encoder removes the two-population handoff. Every one of the 16 real KC
angle populations receives a smoothly varying circular (von-Mises-like)
sensory strength.

Example around 56 degrees:
    22.5 deg population -> weak
    45.0 deg population -> strong
    67.5 deg population -> strong
    90.0 deg population -> weak
    surrounding groups  -> progressively smaller

The weights change smoothly for every target angle.

Still used:
- real FlyWire KC populations
- full 138k-neuron CUDA connectome
- real action MBONs
- existing KC->MBON edges
- local multiplicative teaching updates
- no backpropagation

Starts from:
    data/fly_memory_CONTINUOUS_2deg.json

Saves:
    data/fly_memory_SMOOTH_CIRCULAR_1deg.json

Previous memories are not overwritten.
"""

import argparse
import copy
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch

from train_continuous_angle import (
    print_metrics,
    save_json_atomic,
)

from train_continuous_kc_stimulus_1deg_8ms import (
    ContinuousKCStimulusTrainer,
    load_checkpoint_into_brain,
)


BASE = Path(__file__).resolve().parent

DEFAULT_SOURCE = (
    BASE / "data" / "fly_memory_CONTINUOUS_2deg.json"
)

DEFAULT_OUTPUT = (
    BASE / "data" / "fly_memory_SMOOTH_CIRCULAR_1deg.json"
)


class SmoothCircularKCTrainer(ContinuousKCStimulusTrainer):
    """
    Replace the two-neighbour linear blend with a circular tuning curve
    across all 16 real KC populations.

    The raw population tuning is:

        exp(kappa * (cos(delta) - 1))

    which is numerically stable and has a peak raw value of 1.

    The 16 raw values are then normalized to sum to 1, so the total
    population-level sensory mass stays approximately constant as the
    target rotates.
    """

    def __init__(
        self,
        *args,
        kappa=8.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kappa = float(kappa)

    def set_smooth_encoding(
        self,
        kappa,
        stimulus_gain,
    ):
        self.kappa = float(kappa)
        self.stimulus_gain = float(stimulus_gain)
        self.clear_continuous_caches()

    @staticmethod
    def _signed_circular_delta_deg(
        target,
        preferred,
    ):
        return (
            (float(target) - float(preferred) + 180.0)
            % 360.0
            - 180.0
        )

    def smooth_population_mix(
        self,
        angle_deg,
    ):
        """
        Return all 16 (state, normalized strength) pairs.
        """
        raw = []

        for bin_index in range(16):
            state_key = self.angle_state[bin_index]
            preferred = bin_index * 22.5

            delta_deg = (
                self._signed_circular_delta_deg(
                    angle_deg,
                    preferred,
                )
            )

            delta_rad = math.radians(
                delta_deg
            )

            value = math.exp(
                self.kappa
                * (
                    math.cos(delta_rad)
                    - 1.0
                )
            )

            raw.append(
                (
                    state_key,
                    value,
                )
            )

        total = sum(
            value
            for _, value in raw
        )

        if total <= 0.0:
            raise RuntimeError(
                "Smooth circular population weights summed to zero."
            )

        return tuple(
            (
                state_key,
                value / total,
            )
            for state_key, value in raw
        )

    def _continuous_stimulus(
        self,
        angle_deg,
    ):
        """
        Build smooth circular sensory drive over all 16 KC populations.
        """
        key = (
            round(
                float(angle_deg) % 360.0,
                6,
            ),
            round(self.kappa, 6),
            round(self.stimulus_gain, 6),
        )

        if key in self.continuous_stimulus_cache:
            return (
                self.continuous_stimulus_cache[
                    key
                ]
            )

        mix = self.smooth_population_mix(
            angle_deg
        )

        kc_indices = []
        strengths = []

        for state_key, strength in mix:
            strength = float(strength)

            # Keep all meaningful populations. Tiny far-side values are
            # skipped only when they are numerically irrelevant.
            if strength < 1e-8:
                continue

            for kc_idx in self.population_kcs[
                state_key
            ]:
                kc_indices.append(
                    int(kc_idx)
                )
                strengths.append(
                    strength
                )

        if len(kc_indices) != len(
            set(kc_indices)
        ):
            raise RuntimeError(
                "Expected disjoint rich-state KC populations."
            )

        idx_t = torch.tensor(
            kc_indices,
            dtype=torch.long,
            device=self.device,
        )

        strength_t = torch.tensor(
            strengths,
            dtype=torch.float32,
            device=self.device,
        )

        result = (
            idx_t,
            strength_t,
            mix,
        )

        self.continuous_stimulus_cache[
            key
        ] = result

        return result

    def continuous_strength_map(
        self,
        angle_deg,
    ):
        """
        KC -> smooth circular sensory strength.

        The local synaptic teaching update is weighted by the same sensory
        strength that drove that KC population.
        """
        result = {}

        for (
            state_key,
            strength,
        ) in self.smooth_population_mix(
            angle_deg
        ):
            strength = float(
                strength
            )

            if strength < 1e-8:
                continue

            for kc_idx in self.population_kcs[
                state_key
            ]:
                result[int(kc_idx)] = (
                    strength
                )

        return result


def percentile(values, q):
    values = sorted(
        float(v)
        for v in values
    )

    if not values:
        return float("nan")

    rank = max(
        0,
        min(
            len(values) - 1,
            math.ceil(
                q * len(values)
            )
            - 1,
        ),
    )

    return values[rank]


def evaluate_smooth(
    brain,
    angles,
):
    errors = []
    rows = []

    for angle in angles:
        scores, _, kc_count = (
            brain.get_continuous_scores(
                angle
            )
        )

        vx, vy = brain.scores_to_vector(
            scores
        )

        predicted = brain.vector_angle(
            vx,
            vy,
        )

        error = brain.angular_error(
            angle,
            predicted,
        )

        errors.append(
            float(error)
        )

        rows.append(
            {
                "target": float(angle),
                "predicted": (
                    None
                    if predicted is None
                    else float(predicted)
                ),
                "error": float(error),
                "vx": float(vx),
                "vy": float(vy),
                "kc_count": int(
                    kc_count
                ),
            }
        )

    return {
        "mean": statistics.mean(
            errors
        ),
        "median": statistics.median(
            errors
        ),
        "max": max(errors),
        "p95": percentile(
            errors,
            0.95,
        ),
        "over_15": sum(
            e > 15.0
            for e in errors
        ),
        "within_5": sum(
            e <= 5.0
            for e in errors
        ),
        "within_10": sum(
            e <= 10.0
            for e in errors
        ),
        "within_15": sum(
            e <= 15.0
            for e in errors
        ),
        "count": len(errors),
        "rows": rows,
    }


def print_smooth_metrics(
    label,
    metrics,
):
    print_metrics(
        label,
        metrics,
    )

    print(
        f"95th percentile: {metrics['p95']:.2f} deg"
    )

    print(
        f">15 deg errors:  {metrics['over_15']}/{metrics['count']}"
    )

    print()


def robust_key(metrics):
    """
    Prefer eliminating severe outliers first, then lower p95, then mean,
    then maximum error.

    This prevents a checkpoint with one catastrophic angle from winning
    merely because its average improved slightly.
    """
    return (
        int(metrics["over_15"]),
        float(metrics["p95"]),
        float(metrics["mean"]),
        float(metrics["max"]),
    )


def probe_encoding_combo(
    brain,
    angles,
    kappa,
    gain,
):
    brain.set_smooth_encoding(
        kappa=kappa,
        stimulus_gain=gain,
    )

    errors = []
    zero_vectors = 0

    for angle in angles:
        learned, _, _ = (
            brain._run_continuous_stimulus(
                angle,
                learned=True,
            )
        )

        baseline, _, _ = (
            brain._run_continuous_stimulus(
                angle,
                learned=False,
            )
        )

        scores = {
            action:
                learned[action]
                - baseline[action]
            for action
            in brain.action_order
        }

        vx, vy = brain.scores_to_vector(
            scores
        )

        predicted = brain.vector_angle(
            vx,
            vy,
        )

        if predicted is None:
            zero_vectors += 1

        errors.append(
            brain.angular_error(
                angle,
                predicted,
            )
        )

    return {
        "kappa": float(kappa),
        "gain": float(gain),
        "mean": statistics.mean(
            errors
        ),
        "median": statistics.median(
            errors
        ),
        "max": max(errors),
        "p95": percentile(
            errors,
            0.95,
        ),
        "over_15": sum(
            e > 15.0
            for e in errors
        ),
        "zero_vectors": int(
            zero_vectors
        ),
    }


def auto_select_encoding(
    brain,
):
    """
    Probe anchors AND their halfway regions before starting the expensive
    training run.

    11.25-degree spacing alternates:
        exact 22.5-degree anchors
        exact midpoints between anchors
    so it specifically checks the failure modes found in Stage 3.
    """
    probe_angles = [
        i * 11.25
        for i in range(32)
    ]

    kappas = [
        4.0,
        8.0,
        12.0,
        16.0,
    ]

    gains = [
        700.0,
        1000.0,
        1400.0,
        1800.0,
    ]

    print(
        "Preflight: testing smooth circular encodings "
        "at anchors + midpoints..."
    )
    print()

    print(
        "kappa   gain   zero   >15   mean    p95     max"
    )
    print(
        "-" * 60
    )

    results = []

    for kappa in kappas:
        for gain in gains:
            result = (
                probe_encoding_combo(
                    brain,
                    probe_angles,
                    kappa,
                    gain,
                )
            )

            results.append(
                result
            )

            print(
                f"{kappa:5.1f}  "
                f"{gain:5.0f}   "
                f"{result['zero_vectors']:2d}    "
                f"{result['over_15']:2d}   "
                f"{result['mean']:6.2f}  "
                f"{result['p95']:6.2f}  "
                f"{result['max']:6.2f}"
            )

    viable = [
        r
        for r in results
        if r["zero_vectors"] == 0
    ]

    if not viable:
        viable = results

    # Robust selection: catastrophic / >15-degree failures matter more than
    # a tiny reduction in mean error.
    best = min(
        viable,
        key=lambda r: (
            r["over_15"],
            r["p95"],
            r["mean"],
            r["max"],
        ),
    )

    brain.set_smooth_encoding(
        kappa=best["kappa"],
        stimulus_gain=best["gain"],
    )

    print()
    print(
        "Selected smooth encoder:"
    )
    print(
        f"  kappa:         {best['kappa']}"
    )
    print(
        f"  stimulus_gain: {best['gain']}"
    )
    print(
        f"  coarse mean:   {best['mean']:.2f} deg"
    )
    print(
        f"  coarse p95:    {best['p95']:.2f} deg"
    )
    print(
        f"  coarse max:    {best['max']:.2f} deg"
    )
    print(
        f"  coarse >15:    {best['over_15']}/{len(probe_angles)}"
    )
    print()

    return results, best


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--step",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.002,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=86,
    )

    parser.add_argument(
        "--kappa",
        type=float,
        default=None,
        help=(
            "Skip preflight kappa selection "
            "and use this value."
        ),
    )

    parser.add_argument(
        "--stimulus-gain",
        type=float,
        default=None,
        help=(
            "Skip preflight gain selection "
            "and use this value."
        ),
    )

    args = parser.parse_args()

    random.seed(
        args.seed
    )
    torch.manual_seed(
        args.seed
    )

    angles = []
    angle = 0.0

    while angle < 360.0 - 1e-9:
        angles.append(
            round(
                angle,
                6,
            )
        )
        angle += args.step

    print()
    print(
        "=" * 72
    )
    print(
        "FLYTHIA STAGE 4 — SMOOTH CIRCULAR KC POPULATION CODE"
    )
    print(
        "=" * 72
    )
    print(
        f"Source:   {args.source}"
    )
    print(
        f"Output:   {args.output}"
    )
    print(
        f"Targets:  {len(angles)} directions"
    )
    print(
        f"Step:     {args.step:g} deg"
    )
    print(
        f"Episodes: {args.episodes}"
    )
    print(
        f"LR:       {args.lr}"
    )
    print(
        "Neural window: 8 ms"
    )
    print()

    brain = SmoothCircularKCTrainer(
        run_time_ms=8,
        device="cuda",
        learning_rate=args.lr,
        min_weight=0.25,
        max_weight=4.0,
        stimulus_gain=1000.0,
        kappa=8.0,
    )

    print(
        "Loading best 2-degree checkpoint..."
    )

    load_checkpoint_into_brain(
        brain,
        args.source,
    )

    print(
        "Checkpoint loaded."
    )
    print()

    if (
        args.kappa is None
        and args.stimulus_gain is None
    ):
        probe_results, selected = (
            auto_select_encoding(
                brain
            )
        )
    elif (
        args.kappa is not None
        and args.stimulus_gain is not None
    ):
        brain.set_smooth_encoding(
            kappa=args.kappa,
            stimulus_gain=args.stimulus_gain,
        )

        probe_results = []
        selected = {
            "kappa": float(
                args.kappa
            ),
            "gain": float(
                args.stimulus_gain
            ),
        }

        print(
            "Using requested smooth encoder:"
        )
        print(
            f"  kappa={brain.kappa}"
        )
        print(
            f"  stimulus_gain={brain.stimulus_gain}"
        )
        print()
    else:
        raise ValueError(
            "Provide both --kappa and --stimulus-gain, "
            "or provide neither and allow auto-selection."
        )

    print(
        f"Caching {len(angles)} smooth-code unlearned baselines..."
    )

    brain.cache_continuous_baselines(
        angles
    )

    initial = evaluate_smooth(
        brain,
        angles,
    )

    print_smooth_metrics(
        "INITIAL SMOOTH-CIRCULAR 1-DEGREE PERFORMANCE",
        initial,
    )

    # Safety gate: don't waste ~15 minutes if the new sensory code is
    # completely broken before training.
    if (
        initial["mean"] > 15.0
        or initial["over_15"] > 90
    ):
        print(
            "ABORTING BEFORE TRAINING:"
        )
        print(
            "The smooth encoder is too poor in its initial state."
        )
        print(
            "No trained checkpoint was written."
        )
        return

    best_metrics = initial
    best_memory = copy.deepcopy(
        brain.memory.memory
    )
    best_episode = 0

    start = time.perf_counter()

    for episode in range(
        1,
        args.episodes + 1,
    ):
        target_angle = random.choice(
            angles
        )

        scores, _, _ = (
            brain.get_continuous_scores(
                target_angle
            )
        )

        vx, vy = brain.scores_to_vector(
            scores
        )

        desired_x, desired_y = (
            brain.desired_vector(
                target_angle
            )
        )

        error_x = (
            desired_x - vx
        )
        error_y = (
            desired_y - vy
        )

        changed, _ = (
            brain.apply_continuous_stimulus_teaching(
                target_angle,
                error_x,
                error_y,
            )
        )

        if (
            episode == 1
            or episode % 25 == 0
        ):
            predicted = (
                brain.vector_angle(
                    vx,
                    vy,
                )
            )

            angle_error = (
                brain.angular_error(
                    target_angle,
                    predicted,
                )
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                "\r"
                f"ep={episode:4d}/{args.episodes} | "
                f"target={target_angle:6.1f} | "
                f"err={angle_error:6.2f} deg | "
                f"dxy=({error_x:+.3f},{error_y:+.3f}) | "
                f"syn={changed:3d} | "
                f"{elapsed:6.1f}s",
                end="",
                flush=True,
            )

        if (
            episode % args.eval_every == 0
            or episode == args.episodes
        ):
            print()

            metrics = evaluate_smooth(
                brain,
                angles,
            )

            print_smooth_metrics(
                (
                    "SMOOTH-CIRCULAR EVALUATION "
                    f"@ EPISODE {episode}"
                ),
                metrics,
            )

            if (
                robust_key(metrics)
                < robust_key(best_metrics)
            ):
                best_metrics = metrics
                best_memory = copy.deepcopy(
                    brain.memory.memory
                )
                best_episode = episode

                print(
                    "New robust best at episode "
                    f"{best_episode}: "
                    f">15={best_metrics['over_15']}, "
                    f"p95={best_metrics['p95']:.2f} deg, "
                    f"mean={best_metrics['mean']:.2f} deg, "
                    f"max={best_metrics['max']:.2f} deg"
                )
                print()

            if (
                metrics["mean"] < 4.5
                and metrics["p95"] < 10.0
                and metrics["over_15"] == 0
            ):
                print(
                    "Stage-4 target reached: "
                    "mean < 4.5 deg, p95 < 10 deg, "
                    "and no >15 deg outliers."
                )
                print()

    elapsed = (
        time.perf_counter()
        - start
    )

    save_json_atomic(
        args.output,
        best_memory,
    )

    metrics_path = (
        args.output.with_name(
            args.output.stem
            + "_metrics.json"
        )
    )

    save_json_atomic(
        metrics_path,
        {
            "source": str(
                args.source
            ),
            "encoding": (
                "16-population smooth circular von-Mises-like KC sensory code"
            ),
            "kappa": float(
                brain.kappa
            ),
            "stimulus_gain": float(
                brain.stimulus_gain
            ),
            "run_time_ms": 8.0,
            "best_episode": int(
                best_episode
            ),
            "episodes_requested": int(
                args.episodes
            ),
            "angle_step_deg": float(
                args.step
            ),
            "learning_rate": float(
                args.lr
            ),
            "seed": int(
                args.seed
            ),
            "elapsed_seconds": float(
                elapsed
            ),
            "preflight": (
                probe_results
            ),
            "selected_preflight": (
                selected
            ),
            "initial": {
                k: v
                for k, v
                in initial.items()
                if k != "rows"
            },
            "best": (
                best_metrics
            ),
        },
    )

    print_smooth_metrics(
        "BEST SMOOTH-CIRCULAR 1-DEGREE RESULT",
        best_metrics,
    )

    print(
        f"Best episode: {best_episode}"
    )

    print(
        f"Kappa: {brain.kappa}"
    )

    print(
        f"Stimulus gain: {brain.stimulus_gain}"
    )

    print(
        f"Training time: {elapsed:.1f}s"
    )

    print()
    print(
        "Saved trained memory:"
    )
    print(
        f"  {args.output}"
    )

    print()
    print(
        "Saved metrics:"
    )
    print(
        f"  {metrics_path}"
    )

    print()
    print(
        "Your 2-degree checkpoint and all earlier memories were not overwritten."
    )


if __name__ == "__main__":
    main()
