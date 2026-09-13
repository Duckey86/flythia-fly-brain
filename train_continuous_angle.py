#!/usr/bin/env python3
"""
train_continuous_angle.py

Experimental continuous-angle KC->MBON trainer for Flythia.

What stays biological / connectome-grounded:
- the same real FlyWire KC indices
- the same real action MBONs
- the same full 138k-neuron CUDA connectome simulation
- only existing direct KC->MBON edges are modified
- multiplicative local synaptic changes (no backpropagation)

What is engineered:
- target angle is encoded by overlapping neighbouring KC populations
- a signed X/Y aiming error is used as the teaching signal

IMPORTANT:
This script does NOT overwrite data/fly_memory.json or
data/fly_memory_FINAL_300bpm.json.

It saves the best result to:
    data/fly_memory_CONTINUOUS_5deg.json
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
from population_angle_probe import PopulationAngleProbe


BASE = Path(__file__).resolve().parent

DEFAULT_OUTPUT = BASE / "data" / "fly_memory_CONTINUOUS_5deg.json"
DEFAULT_METRICS = BASE / "data" / "fly_memory_CONTINUOUS_5deg_metrics.json"


class ContinuousAngleTrainer(PopulationAngleProbe):
    def __init__(
        self,
        *args,
        learning_rate=0.015,
        min_weight=0.25,
        max_weight=4.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.learning_rate = float(learning_rate)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

        # Cache fixed baseline responses per sampled angle.
        self.angle_baseline_cache = {}

        # Map:
        #   action -> KC index -> [(combined_plan_position, rich_state), ...]
        #
        # PopulationAngleProbe concatenates the 16 rich-state action plans
        # in self.rich_states order. Reconstruct that exact position mapping
        # so we can update only the affected GPU correction entries in-place.
        self.edge_positions = {
            action: {}
            for action in ACTIONS
        }

        for action in ACTIONS:
            offset = 0

            for state_key in self.rich_states:
                pre_t, _, _ = self.rich_learned_plan[state_key][action]
                pre_list = pre_t.detach().cpu().tolist()

                for local_pos, kc_idx in enumerate(pre_list):
                    kc_idx = int(kc_idx)

                    self.edge_positions[action].setdefault(
                        kc_idx,
                        [],
                    ).append(
                        (
                            offset + local_pos,
                            state_key,
                        )
                    )

                offset += len(pre_list)

        print("Continuous trainer edge map ready.")
        for action in ACTIONS:
            edge_count = sum(
                len(v)
                for v in self.edge_positions[action].values()
            )
            print(
                f"  {action:5s}: "
                f"{len(self.edge_positions[action]):3d} KCs / "
                f"{edge_count:3d} direct KC->MBON edges"
            )
        print()

    def active_population_ids(self, angle_deg):
        """
        Return the exact deterministic KC set used by the population encoder.

        Recruitment strength is represented by recruiting a proportion of each
        neighbouring real KC population, matching population_angle_probe.py.
        """
        active = []
        mix = self._population_mix(angle_deg)

        for state_key, strength in mix:
            ids = self.population_kcs[state_key]

            count = int(round(float(strength) * len(ids)))
            count = max(0, min(len(ids), count))

            active.extend(ids[:count])

        return sorted(set(int(x) for x in active)), mix

    def cache_baselines(self, angles):
        print(
            f"Caching {len(angles)} fixed unlearned angle baselines..."
        )

        t0 = time.perf_counter()

        for i, angle in enumerate(angles, 1):
            key = round(float(angle) % 360.0, 6)

            baseline, _, _ = self._run_population(
                angle,
                learned=False,
            )

            self.angle_baseline_cache[key] = baseline

            if i % 12 == 0 or i == len(angles):
                print(
                    f"\r  {i:3d}/{len(angles)}",
                    end="",
                    flush=True,
                )

        print()
        print(
            "Angle baseline cache ready in "
            f"{time.perf_counter() - t0:.2f}s"
        )
        print()

    def get_training_scores(self, angle_deg):
        key = round(float(angle_deg) % 360.0, 6)

        if key not in self.angle_baseline_cache:
            baseline, _, _ = self._run_population(
                angle_deg,
                learned=False,
            )
            self.angle_baseline_cache[key] = baseline

        learned, mix, kc_count = self._run_population(
            angle_deg,
            learned=True,
        )

        baseline = self.angle_baseline_cache[key]

        scores = {
            action: learned[action] - baseline[action]
            for action in self.action_order
        }

        return scores, mix, kc_count

    @staticmethod
    def desired_vector(angle_deg):
        theta = math.radians(float(angle_deg))
        return math.cos(theta), math.sin(theta)

    def _set_gpu_multiplier(
        self,
        action,
        kc_idx,
        new_multiplier,
    ):
        """
        Update every occurrence of this real KC->action-MBON edge in the
        already-built combined learned GPU correction plan.
        """
        positions = self.edge_positions[action].get(
            int(kc_idx),
            [],
        )

        if not positions:
            return 0

        post_idx = int(
            self.brain.action_mbon_indices[action]
        )

        biological_weight = float(
            self.pair_weight.get(
                (int(kc_idx), post_idx),
                0.0,
            )
        )

        if biological_weight == 0.0:
            return 0

        delta_tensor = self.population_learned_plan[action][1]

        changed_positions = 0

        for position, state_key in positions:
            info = self.rich_patterns[state_key]
            direction_key = self._direction_key(
                info["direction"]
            )

            calibration = float(
                self.calibration[direction_key][action]["factor"]
            )

            delta = biological_weight * (
                calibration * float(new_multiplier) - 1.0
            )

            delta_tensor[position] = float(delta)
            changed_positions += 1

        return changed_positions

    def apply_continuous_teaching_signal(
        self,
        angle_deg,
        error_x,
        error_y,
    ):
        """
        Local multiplicative KC->MBON update.

        If X drive is too small:
            strengthen RIGHT / weaken LEFT.
        If X drive is too large:
            weaken RIGHT / strengthen LEFT.

        Same idea for DOWN / UP on the screen Y axis.

        Only currently active real KCs and their existing direct action-MBON
        synapses are eligible.
        """
        active_kcs, _ = self.active_population_ids(angle_deg)

        signals = {
            "RIGHT": float(error_x),
            "LEFT": -float(error_x),
            "DOWN": float(error_y),
            "UP": -float(error_y),
        }

        mods = self.memory.memory["weight_modifications"]

        changed_synapses = 0
        changed_gpu_positions = 0

        for action, raw_signal in signals.items():
            signal = max(-1.0, min(1.0, raw_signal))

            if abs(signal) < 1e-6:
                continue

            # Multiplicative dopamine-style local update.
            factor = math.exp(
                self.learning_rate * signal
            )

            post_idx = int(
                self.brain.action_mbon_indices[action]
            )

            for kc_idx in active_kcs:
                if kc_idx not in self.edge_positions[action]:
                    continue

                memory_key = f"{kc_idx}:{post_idx}"

                old = float(
                    mods.get(memory_key, 1.0)
                )

                new = old * factor
                new = max(
                    self.min_weight,
                    min(self.max_weight, new),
                )

                if abs(new - old) < 1e-12:
                    continue

                # Keep enough precision for many tiny continuous updates.
                new = round(new, 6)
                mods[memory_key] = new

                changed_synapses += 1
                changed_gpu_positions += self._set_gpu_multiplier(
                    action,
                    kc_idx,
                    new,
                )

        return changed_synapses, changed_gpu_positions


def evaluate(brain, angles):
    errors = []
    rows = []

    for angle in angles:
        scores, _, kc_count = brain.get_training_scores(
            angle
        )

        vx, vy = brain.scores_to_vector(scores)
        predicted = brain.vector_angle(vx, vy)
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
                "kc_count": int(kc_count),
            }
        )

    return {
        "mean": statistics.mean(errors),
        "median": statistics.median(errors),
        "max": max(errors),
        "within_5": sum(e <= 5.0 for e in errors),
        "within_10": sum(e <= 10.0 for e in errors),
        "within_15": sum(e <= 15.0 for e in errors),
        "count": len(errors),
        "rows": rows,
    }


def print_metrics(label, metrics):
    n = metrics["count"]

    print()
    print("=" * 64)
    print(label)
    print("=" * 64)
    print(
        f"Mean error:   {metrics['mean']:.2f} deg"
    )
    print(
        f"Median error: {metrics['median']:.2f} deg"
    )
    print(
        f"Max error:    {metrics['max']:.2f} deg"
    )
    print()
    print(
        f"Within  5 deg: {metrics['within_5']}/{n}"
    )
    print(
        f"Within 10 deg: {metrics['within_10']}/{n}"
    )
    print(
        f"Within 15 deg: {metrics['within_15']}/{n}"
    )
    print()


def save_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(temp, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
        )

    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=int,
        default=4500,
    )

    parser.add_argument(
        "--step",
        type=float,
        default=5.0,
        help="training/evaluation angle spacing in degrees",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.015,
        help="multiplicative continuous teaching rate",
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=86,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.step <= 0.0 or args.step > 45.0:
        raise ValueError(
            "--step must be > 0 and <= 45 degrees"
        )

    angles = []
    angle = 0.0

    while angle < 360.0 - 1e-9:
        angles.append(round(angle, 6))
        angle += args.step

    print()
    print("Loading continuous-angle CUDA fly...")
    print(
        "Source memory is read from data/fly_memory.json."
    )
    print(
        "The source file will NOT be overwritten."
    )
    print(
        "Best trained copy will be written to:"
    )
    print(f"  {args.output}")
    print()

    brain = ContinuousAngleTrainer(
        run_time_ms=3,
        device="cuda",
        learning_rate=args.lr,
        min_weight=0.25,
        max_weight=4.0,
    )

    brain.cache_baselines(angles)

    initial = evaluate(
        brain,
        angles,
    )

    print_metrics(
        "INITIAL CONTINUOUS-ANGLE PERFORMANCE",
        initial,
    )

    best_metrics = initial
    best_memory = copy.deepcopy(
        brain.memory.memory
    )
    best_episode = 0

    start = time.perf_counter()

    for episode in range(1, args.episodes + 1):
        target_angle = random.choice(angles)

        scores, _, _ = brain.get_training_scores(
            target_angle
        )

        vx, vy = brain.scores_to_vector(scores)
        desired_x, desired_y = brain.desired_vector(
            target_angle
        )

        # When the brain emits zero drive, vx=vy=0, which naturally creates
        # a strong teaching signal toward the desired vector.
        error_x = desired_x - vx
        error_y = desired_y - vy

        changed, gpu_changes = (
            brain.apply_continuous_teaching_signal(
                target_angle,
                error_x,
                error_y,
            )
        )

        if (
            episode == 1
            or episode % 25 == 0
        ):
            predicted = brain.vector_angle(vx, vy)
            angle_error = brain.angular_error(
                target_angle,
                predicted,
            )

            elapsed = time.perf_counter() - start

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

            metrics = evaluate(
                brain,
                angles,
            )

            print_metrics(
                f"EVALUATION @ EPISODE {episode}",
                metrics,
            )

            # Mean error is the primary target; median breaks close ties.
            improved = (
                metrics["mean"] < best_metrics["mean"] - 1e-9
                or (
                    abs(
                        metrics["mean"]
                        - best_metrics["mean"]
                    ) < 1e-9
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
                    f"New best at episode {best_episode}: "
                    f"mean={best_metrics['mean']:.2f} deg"
                )
                print()

            # First milestone from the current ~17.5 deg baseline.
            if (
                metrics["mean"] < 10.0
                and metrics["median"] < 8.0
            ):
                print(
                    "Stage-1 target reached: "
                    "mean < 10 deg and median < 8 deg."
                )
                print(
                    "Continuing to the requested episode count "
                    "so we can keep the best checkpoint."
                )
                print()

    elapsed = time.perf_counter() - start

    # Save only the best in-memory checkpoint.
    save_json_atomic(
        args.output,
        best_memory,
    )

    metrics_path = args.output.with_name(
        args.output.stem + "_metrics.json"
    )

    save_json_atomic(
        metrics_path,
        {
            "best_episode": best_episode,
            "episodes_requested": args.episodes,
            "angle_step_deg": args.step,
            "learning_rate": args.lr,
            "seed": args.seed,
            "elapsed_seconds": elapsed,
            "initial": {
                k: v
                for k, v in initial.items()
                if k != "rows"
            },
            "best": best_metrics,
        },
    )

    print_metrics(
        "BEST CONTINUOUS-ANGLE RESULT",
        best_metrics,
    )

    print(
        f"Best episode: {best_episode}"
    )
    print(
        f"Training time: {elapsed:.1f}s"
    )
    print()
    print("Saved trained memory:")
    print(f"  {args.output}")
    print()
    print("Saved metrics:")
    print(f"  {metrics_path}")
    print()
    print(
        "Your original data/fly_memory.json and "
        "data/fly_memory_FINAL_300bpm.json were not overwritten."
    )


if __name__ == "__main__":
    main()
