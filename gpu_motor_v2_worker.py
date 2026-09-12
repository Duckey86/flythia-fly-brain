import argparse
import json
from statistics import mean

from live_torch_motor_v2 import LiveTorchMotorV2, motor_success

ACTIONS = ["LEFT", "RIGHT", "UP", "DOWN"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration-ms", type=float, required=True)
    p.add_argument("--stim-hz", type=float, required=True)
    p.add_argument("--trials", type=int, default=30)
    args = p.parse_args()

    motor = LiveTorchMotorV2(
        run_time_ms=args.duration_ms,
        stim_hz=args.stim_hz,
        device="cuda",
    )

    # One untimed warm-up per action.
    for action in ACTIONS:
        motor.run(action)

    results = {}

    for action in ACTIONS:
        success_count = 0
        times = []

        for _ in range(args.trials):
            r = motor.run(action)
            times.append(r["wall_time_sec"] * 1000.0)

            if motor_success(action, r["output_neuron_activity"]):
                success_count += 1

        results[action] = {
            "successes": success_count,
            "pct": 100.0 * success_count / args.trials,
            "avg_ms": mean(times),
        }

    result = {
        "duration_ms": args.duration_ms,
        "stim_hz": args.stim_hz,
        "trials": args.trials,
        "actions": results,
        "avg_motor_ms": mean(results[a]["avg_ms"] for a in ACTIONS),
    }

    print("JSON_RESULT=" + json.dumps(result))


if __name__ == "__main__":
    main()
