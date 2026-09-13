#!/usr/bin/env python3
"""
train_continuous_kc_stimulus_1deg.py

Stage 3 for Flythia continuous aiming — corrected 8 ms neural window.

Key change from the previous population encoder:
------------------------------------------------
OLD:
    angle strength -> round(strength * KC_count)
    selected KCs are forced fully ON at v=-44 mV

NEW:
    ALL KCs in the two neighbouring angle populations receive a
    CONTINUOUS sensory drive proportional to their population weight.

So a target such as 37 degrees can be represented as:
    22.5-degree population -> 0.356 sensory strength
    45.0-degree population -> 0.644 sensory strength

No integer KC-count rounding is used to represent the angle.

The sensory drive is an engineered external current into the same real
FlyWire KCs. The recurrent network, real connectome, real action MBONs,
and learned direct KC->MBON corrections remain the same.

Training remains local/multiplicative. No backpropagation is used.

Starts from:
    data/fly_memory_CONTINUOUS_2deg.json

Best checkpoint:
    data/fly_memory_CONTINUOUS_STIM_1deg.json

The graded KC signal is simulated for 8 ms so it has time to cross the model's synaptic delay and reach the action MBONs.\n\nNothing from the previous stages is overwritten.
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

from fly_mb_policy import ACTIONS
from train_continuous_angle import (
    ContinuousAngleTrainer,
    print_metrics,
    save_json_atomic,
)


BASE = Path(__file__).resolve().parent

DEFAULT_SOURCE = (
    BASE / "data" / "fly_memory_CONTINUOUS_2deg.json"
)

DEFAULT_OUTPUT = (
    BASE / "data" / "fly_memory_CONTINUOUS_STIM_1deg.json"
)


class ContinuousKCStimulusTrainer(ContinuousAngleTrainer):
    """
    Continuous sensory-strength version of the population-coded fly.

    Instead of encoding population strength by recruiting an integer number
    of KCs, every real KC in the two neighbouring populations receives an
    external drive scaled by its continuous population strength.

    `stimulus_gain` is applied before the model's existing wScale.
    """

    def __init__(
        self,
        *args,
        stimulus_gain=700.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.stimulus_gain = float(stimulus_gain)

        # angle -> (KC index tensor, strength tensor, mix)
        self.continuous_stimulus_cache = {}

        # angle -> fixed unlearned response
        self.continuous_baseline_cache = {}

    def clear_continuous_caches(self):
        self.continuous_stimulus_cache.clear()
        self.continuous_baseline_cache.clear()

    def _continuous_stimulus(self, angle_deg):
        """
        Build the continuously weighted KC sensory representation.

        Returns:
            kc_indices: CUDA LongTensor
            strengths:  CUDA FloatTensor, values in (0,1]
            mix:         two neighbouring state populations + weights
        """
        key = round(
            float(angle_deg) % 360.0,
            6,
        )

        if key in self.continuous_stimulus_cache:
            return self.continuous_stimulus_cache[key]

        mix = self._population_mix(angle_deg)

        kc_indices = []
        strengths = []

        for state_key, strength in mix:
            strength = float(strength)

            if strength <= 1e-12:
                continue

            for kc_idx in self.population_kcs[state_key]:
                kc_indices.append(int(kc_idx))
                strengths.append(strength)

        # The original 16 KC groups are disjoint, so an index should not be
        # duplicated. This check catches accidental future pattern changes.
        if len(kc_indices) != len(set(kc_indices)):
            merged = {}

            for kc_idx, strength in zip(
                kc_indices,
                strengths,
            ):
                merged[kc_idx] = (
                    merged.get(kc_idx, 0.0)
                    + float(strength)
                )

            kc_indices = list(merged.keys())
            strengths = [
                min(1.0, merged[k])
                for k in kc_indices
            ]

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

        self.continuous_stimulus_cache[key] = result

        return result

    @torch.no_grad()
    def _run_continuous_stimulus(
        self,
        angle_deg,
        learned,
    ):
        """
        Full 138k-neuron connectome simulation with continuous KC drive.

        Unlike the old encoder, we DO NOT set a rounded subset of KCs to
        v=-44 mV.

        Instead, at every neural timestep the chosen real KCs receive a
        graded external sensory drive. Stronger angle-population membership
        therefore changes neural dynamics continuously rather than by an
        integer KC-count jump.
        """
        kc_idx, strengths, mix = (
            self._continuous_stimulus(
                angle_deg
            )
        )

        (
            conductance,
            delay_buffer,
            spikes,
            v,
            refrac,
        ) = self.model.state_init()

        peak = torch.full(
            (len(self.action_order),),
            -float("inf"),
            device=self.device,
        )

        plan = (
            self.population_learned_plan
            if learned
            else self.population_baseline_plan
        )

        for _ in range(self.num_steps):
            weighted = torch.matmul(
                spikes,
                self.weights.transpose(0, 1),
            )

            # Same learned real KC->MBON correction mechanism as before.
            self._apply_correction(
                weighted,
                spikes,
                plan,
            )

            # Engineered sensory input only.
            #
            # This is added BEFORE the model's existing `wScale`, so it
            # passes through the normal LIF dynamics rather than directly
            # assigning an MBON score or cursor vector.
            if kc_idx.numel() > 0:
                weighted[:, kc_idx] = (
                    weighted[:, kc_idx]
                    + strengths.unsqueeze(0)
                    * self.stimulus_gain
                )

            (
                conductance,
                delay_buffer,
                spikes,
                v,
                refrac,
            ) = self.model.neurons(
                self.model.scale * weighted,
                conductance,
                delay_buffer,
                spikes,
                v,
                refrac,
            )

            peak = torch.maximum(
                peak,
                conductance[
                    0,
                    self.action_indices,
                ],
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        values = peak.cpu().tolist()

        scores = {
            action: float(values[i])
            for i, action
            in enumerate(self.action_order)
        }

        return (
            scores,
            mix,
            int(kc_idx.numel()),
        )

    def cache_continuous_baselines(
        self,
        angles,
    ):
        self.continuous_baseline_cache = {}

        print(
            f"Caching {len(angles)} continuous-stimulus "
            f"unlearned baselines..."
        )

        t0 = time.perf_counter()

        for i, angle in enumerate(
            angles,
            1,
        ):
            key = round(
                float(angle) % 360.0,
                6,
            )

            baseline, _, _ = (
                self._run_continuous_stimulus(
                    angle,
                    learned=False,
                )
            )

            self.continuous_baseline_cache[
                key
            ] = baseline

            if (
                i % 30 == 0
                or i == len(angles)
            ):
                print(
                    f"\r  {i:3d}/{len(angles)}",
                    end="",
                    flush=True,
                )

        print()
        print(
            "Continuous baseline cache ready in "
            f"{time.perf_counter() - t0:.2f}s"
        )
        print()

    def get_continuous_scores(
        self,
        angle_deg,
    ):
        key = round(
            float(angle_deg) % 360.0,
            6,
        )

        if (
            key
            not in self.continuous_baseline_cache
        ):
            baseline, _, _ = (
                self._run_continuous_stimulus(
                    angle_deg,
                    learned=False,
                )
            )

            self.continuous_baseline_cache[
                key
            ] = baseline

        learned, mix, kc_count = (
            self._run_continuous_stimulus(
                angle_deg,
                learned=True,
            )
        )

        baseline = (
            self.continuous_baseline_cache[
                key
            ]
        )

        scores = {
            action:
                learned[action]
                - baseline[action]
            for action in self.action_order
        }

        return (
            scores,
            mix,
            kc_count,
        )

    def continuous_strength_map(
        self,
        angle_deg,
    ):
        """
        CPU dictionary used by local plasticity:
            real KC index -> continuous sensory strength.
        """
        mix = self._population_mix(
            angle_deg
        )

        result = {}

        for state_key, strength in mix:
            strength = float(strength)

            if strength <= 1e-12:
                continue

            for kc_idx in self.population_kcs[
                state_key
            ]:
                result[int(kc_idx)] = (
                    result.get(
                        int(kc_idx),
                        0.0,
                    )
                    + strength
                )

        return result

    def apply_continuous_stimulus_teaching(
        self,
        angle_deg,
        error_x,
        error_y,
    ):
        """
        Local opponent-axis teaching rule.

        The previous stage updated each recruited KC equally.

        Here, each KC's multiplicative plasticity is ALSO weighted by its
        continuous sensory strength. This is what lets neighbouring angles
        make slightly different synaptic updates without integer recruitment.
        """
        kc_strengths = (
            self.continuous_strength_map(
                angle_deg
            )
        )

        signals = {
            "RIGHT": float(error_x),
            "LEFT": -float(error_x),
            "DOWN": float(error_y),
            "UP": -float(error_y),
        }

        mods = self.memory.memory[
            "weight_modifications"
        ]

        changed_synapses = 0
        changed_gpu_positions = 0

        for action, raw_signal in signals.items():
            signal = max(
                -1.0,
                min(1.0, raw_signal),
            )

            if abs(signal) < 1e-7:
                continue

            post_idx = int(
                self.brain.action_mbon_indices[
                    action
                ]
            )

            for (
                kc_idx,
                sensory_strength,
            ) in kc_strengths.items():

                if (
                    kc_idx
                    not in self.edge_positions[
                        action
                    ]
                ):
                    continue

                # The teaching signal is local AND graded:
                #
                # synaptic change
                #   ~ motor error
                #   * this KC's sensory activation strength
                local_signal = (
                    signal
                    * float(sensory_strength)
                )

                factor = math.exp(
                    self.learning_rate
                    * local_signal
                )

                memory_key = (
                    f"{kc_idx}:{post_idx}"
                )

                old = float(
                    mods.get(
                        memory_key,
                        1.0,
                    )
                )

                new = old * factor

                new = max(
                    self.min_weight,
                    min(
                        self.max_weight,
                        new,
                    ),
                )

                if abs(new - old) < 1e-12:
                    continue

                new = round(new, 6)

                mods[memory_key] = new

                changed_synapses += 1

                changed_gpu_positions += (
                    self._set_gpu_multiplier(
                        action,
                        kc_idx,
                        new,
                    )
                )

        return (
            changed_synapses,
            changed_gpu_positions,
        )


def load_checkpoint_into_brain(
    brain,
    source_path,
):
    source_path = Path(source_path)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing source checkpoint: "
            f"{source_path}"
        )

    with open(
        source_path,
        "r",
        encoding="utf-8",
    ) as f:
        memory = json.load(f)

    brain.memory.memory = memory

    # Rebuild state-specific learned plans from the newly loaded memory.
    for rich_state in brain.rich_states:
        for action in ACTIONS:
            brain._rebuild_rich_action(
                rich_state,
                action,
                rebuild_baseline=False,
            )

    # Rebuild the combined population plan used by this experiment.
    brain.population_learned_plan = (
        brain._combine_plans(
            brain.rich_learned_plan
        )
    )

    if brain.device.type == "cuda":
        torch.cuda.synchronize()


def evaluate_continuous(
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

        errors.append(float(error))

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


def coarse_gain_score(
    brain,
    gain,
    angles,
):
    """
    Quick probe used to choose a sensible sensory-current gain before the
    expensive 1-degree baseline cache is built.
    """
    brain.stimulus_gain = float(gain)
    brain.clear_continuous_caches()

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
        "gain": float(gain),
        "mean": statistics.mean(
            errors
        ),
        "median": statistics.median(
            errors
        ),
        "zero_vectors": zero_vectors,
    }


def select_stimulus_gain(
    brain,
    candidates,
):
    # 10-degree coarse sweep: enough to reject bad/dead sensory gains
    # without spending too long before real training starts.
    probe_angles = [
        float(x)
        for x in range(
            0,
            360,
            10,
        )
    ]

    print(
        "Auto-tuning continuous KC sensory gain "
        "on a coarse 10-degree sweep..."
    )

    results = []

    for gain in candidates:
        result = coarse_gain_score(
            brain,
            gain,
            probe_angles,
        )

        results.append(result)

        print(
            f"  gain={gain:6.1f} | "
            f"mean={result['mean']:6.2f} deg | "
            f"median={result['median']:6.2f} deg | "
            f"zero={result['zero_vectors']:2d}"
        )

    # First avoid dead/zero-vector encodings, then choose lowest mean error.
    viable = [
        r
        for r in results
        if r["zero_vectors"] == 0
    ]

    if not viable:
        viable = results

    best = min(
        viable,
        key=lambda r: (
            r["mean"],
            r["median"],
            r["zero_vectors"],
        ),
    )

    brain.stimulus_gain = float(
        best["gain"]
    )

    brain.clear_continuous_caches()

    print()
    print(
        "Selected sensory gain: "
        f"{brain.stimulus_gain:g}"
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
        default=0.003,
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
        "--stimulus-gain",
        type=float,
        default=None,
        help=(
            "skip auto-tuning and use this "
            "continuous sensory gain"
        ),
    )

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if (
        args.step <= 0.0
        or args.step > 22.5
    ):
        raise ValueError(
            "--step must be > 0 and <= 22.5"
        )

    angles = []
    angle = 0.0

    while angle < 360.0 - 1e-9:
        angles.append(
            round(angle, 6)
        )
        angle += args.step

    print()
    print("=" * 68)
    print(
        "FLYTHIA CONTINUOUS KC STIMULUS — STAGE 3"
    )
    print("=" * 68)
    print(f"Source:   {args.source}")
    print(f"Output:   {args.output}")
    print(
        f"Targets:  {len(angles)} directions"
    )
    print(f"Step:     {args.step:g} deg")
    print(f"Episodes: {args.episodes}")
    print(f"LR:       {args.lr}")
    print()
    print(
        "Angle encoding no longer rounds "
        "population strength into an integer KC count."
    )
    print()

    brain = ContinuousKCStimulusTrainer(
        run_time_ms=8,
        device="cuda",
        learning_rate=args.lr,
        min_weight=0.25,
        max_weight=4.0,
        stimulus_gain=700.0,
    )

    print(
        "Loading 2-degree checkpoint..."
    )

    load_checkpoint_into_brain(
        brain,
        args.source,
    )

    print(
        "Checkpoint loaded into CUDA learned plans."
    )
    print()

    if args.stimulus_gain is None:
        gain_results, best_gain = (
            select_stimulus_gain(
                brain,
                candidates=[
                    400.0,
                    550.0,
                    700.0,
                    850.0,
                ],
            )
        )
    else:
        brain.stimulus_gain = float(
            args.stimulus_gain
        )
        brain.clear_continuous_caches()

        gain_results = []
        best_gain = {
            "gain": brain.stimulus_gain
        }

        print(
            "Using requested sensory gain: "
            f"{brain.stimulus_gain:g}"
        )
        print()

    brain.cache_continuous_baselines(
        angles
    )

    initial = evaluate_continuous(
        brain,
        angles,
    )

    print_metrics(
        "INITIAL CONTINUOUS-STIMULUS 1-DEGREE PERFORMANCE",
        initial,
    )

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

        error_x = desired_x - vx
        error_y = desired_y - vy

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

            metrics = evaluate_continuous(
                brain,
                angles,
            )

            print_metrics(
                (
                    "CONTINUOUS-STIMULUS "
                    f"EVALUATION @ EPISODE {episode}"
                ),
                metrics,
            )

            improved = (
                metrics["mean"]
                < best_metrics["mean"]
                - 1e-9
                or (
                    abs(
                        metrics["mean"]
                        - best_metrics["mean"]
                    )
                    < 1e-9
                    and metrics["median"]
                    < best_metrics["median"]
                )
            )

            if improved:
                best_metrics = metrics

                best_memory = copy.deepcopy(
                    brain.memory.memory
                )

                best_episode = episode

                print(
                    f"New best at episode "
                    f"{best_episode}: "
                    f"mean={best_metrics['mean']:.2f} deg, "
                    f"median={best_metrics['median']:.2f} deg"
                )
                print()

            if (
                metrics["mean"] < 3.5
                and metrics["median"] < 3.0
            ):
                print(
                    "Stage-3 target reached: "
                    "mean < 3.5 deg and "
                    "median < 3 deg."
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
            "best_episode": (
                best_episode
            ),
            "episodes_requested": (
                args.episodes
            ),
            "angle_step_deg": (
                args.step
            ),
            "learning_rate": (
                args.lr
            ),
            "seed": args.seed,
            "stimulus_gain": (
                brain.stimulus_gain
            ),
            "gain_probe": (
                gain_results
            ),
            "elapsed_seconds": (
                elapsed
            ),
            "encoding": (
                "continuous external KC sensory "
                "drive on two neighbouring real "
                "KC populations"
            ),
            "initial": {
                k: v
                for k, v in initial.items()
                if k != "rows"
            },
            "best": best_metrics,
        },
    )

    print_metrics(
        "BEST CONTINUOUS-STIMULUS 1-DEGREE RESULT",
        best_metrics,
    )

    print(
        f"Best episode: {best_episode}"
    )

    print(
        "Selected sensory gain: "
        f"{brain.stimulus_gain:g}"
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
        "Your 2-degree checkpoint, "
        "5-degree checkpoint, and original "
        "memories were not overwritten."
    )


if __name__ == "__main__":
    main()
