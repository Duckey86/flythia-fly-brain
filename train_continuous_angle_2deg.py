#!/usr/bin/env python3
"""
train_continuous_angle_2deg.py

Stage-2 continuous-angle fine-tuning for Flythia.

Starts from:
    data/fly_memory_CONTINUOUS_5deg.json

Trains on:
    2-degree targets (180 directions)

Uses:
    smaller multiplicative learning rate (default 0.005)

Saves best checkpoint to:
    data/fly_memory_CONTINUOUS_2deg.json

Does NOT overwrite your 5-degree checkpoint or original memories.
"""

import argparse
import copy
import json
import random
import statistics
import time
from pathlib import Path

import torch

from fly_mb_policy import ACTIONS
from train_continuous_angle import (
    ContinuousAngleTrainer,
    evaluate,
    print_metrics,
    save_json_atomic,
)


BASE = Path(__file__).resolve().parent
DEFAULT_SOURCE = BASE / "data" / "fly_memory_CONTINUOUS_5deg.json"
DEFAULT_OUTPUT = BASE / "data" / "fly_memory_CONTINUOUS_2deg.json"


def load_memory_into_brain(brain, source_path):
    source_path = Path(source_path)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing source memory: {source_path}"
        )

    with open(source_path, "r", encoding="utf-8") as f:
        trained_memory = json.load(f)

    brain.memory.memory = trained_memory

    # Rebuild the learned correction plans so the full CUDA connectome
    # actually uses the 5-degree checkpoint we just loaded.
    for rich_state in brain.rich_states:
        for action in ACTIONS:
            brain._rebuild_rich_action(
                rich_state,
                action,
                rebuild_baseline=False,
            )

    # Recombine the 16 state-specific plans into the population-coded plan.
    brain.population_learned_plan = brain._combine_plans(
        brain.rich_learned_plan
    )

    if brain.device.type == "cuda":
        torch.cuda.synchronize()


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
        default=4500,
    )

    parser.add_argument(
        "--step",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
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

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.step <= 0.0 or args.step > 45.0:
        raise ValueError("--step must be > 0 and <= 45")

    angles = []
    angle = 0.0

    while angle < 360.0 - 1e-9:
        angles.append(round(angle, 6))
        angle += args.step

    print()
    print("============================================================")
    print("FLYTHIA CONTINUOUS ANGLE — STAGE 2")
    print("============================================================")
    print(f"Source:   {args.source}")
    print(f"Output:   {args.output}")
    print(f"Targets:  {len(angles)} directions")
    print(f"Step:     {args.step:g} deg")
    print(f"Episodes: {args.episodes}")
    print(f"LR:       {args.lr}")
    print()

    brain = ContinuousAngleTrainer(
        run_time_ms=3,
        device="cuda",
        learning_rate=args.lr,
        min_weight=0.25,
        max_weight=4.0,
    )

    print("Loading 5-degree continuous checkpoint...")
    load_memory_into_brain(
        brain,
        args.source,
    )
    print("Checkpoint loaded into CUDA learned plans.")
    print()

    brain.cache_baselines(angles)

    initial = evaluate(
        brain,
        angles,
    )

    print_metrics(
        "INITIAL 2-DEGREE PERFORMANCE",
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

        error_x = desired_x - vx
        error_y = desired_y - vy

        changed, _ = brain.apply_continuous_teaching_signal(
            target_angle,
            error_x,
            error_y,
        )

        if episode == 1 or episode % 25 == 0:
            predicted = brain.vector_angle(
                vx,
                vy,
            )

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
                    f"mean={best_metrics['mean']:.2f} deg, "
                    f"median={best_metrics['median']:.2f} deg"
                )
                print()

            if (
                metrics["mean"] < 4.0
                and metrics["median"] < 3.0
            ):
                print(
                    "Stage-2 target reached: "
                    "mean < 4 deg and median < 3 deg."
                )
                print()

    elapsed = time.perf_counter() - start

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
            "source": str(args.source),
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
        "BEST 2-DEGREE CONTINUOUS RESULT",
        best_metrics,
    )

    print(f"Best episode: {best_episode}")
    print(f"Training time: {elapsed:.1f}s")
    print()
    print("Saved trained memory:")
    print(f"  {args.output}")
    print()
    print("Saved metrics:")
    print(f"  {metrics_path}")
    print()
    print(
        "Your 5-degree checkpoint and original memories were not overwritten."
    )


if __name__ == "__main__":
    main()
