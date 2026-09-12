import time
from statistics import mean

import torch

from live_torch_motor import LiveTorchMotor, motor_success


TRIALS = 20
ACTIONS = ["LEFT", "RIGHT", "UP", "DOWN"]


def main():
    motor = LiveTorchMotor(
        run_time_ms=40,
        stim_hz=250,
        device="cuda",
    )

    results = {}

    for action in ACTIONS:
        successes = 0
        times_ms = []

        print()
        print("=" * 72)
        print(action)
        print("=" * 72)

        for trial in range(1, TRIALS + 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            result = motor.run(action)

            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            out = result["output_neuron_activity"]
            ok = motor_success(action, out)

            if ok:
                successes += 1

            times_ms.append(elapsed_ms)

            compact = {
                name: data["spikes"]
                for name, data in out.items()
                if data["spikes"] > 0
            }

            print(
                f"{trial:02d}/{TRIALS} "
                f"{'PASS' if ok else 'FAIL'} | "
                f"{elapsed_ms:.1f} ms | "
                f"spikes={compact}"
            )

        results[action] = {
            "successes": successes,
            "pct": successes / TRIALS * 100.0,
            "avg_ms": mean(times_ms),
            "min_ms": min(times_ms),
            "max_ms": max(times_ms),
        }

    print()
    print("=" * 84)
    print("FINAL GPU MOTOR RESULT")
    print("=" * 84)

    for action in ACTIONS:
        r = results[action]
        print(
            f"{action:5s}: "
            f"{r['successes']:2d}/{TRIALS} "
            f"({r['pct']:5.1f}%) | "
            f"avg {r['avg_ms']:.1f} ms | "
            f"range {r['min_ms']:.1f}-{r['max_ms']:.1f} ms"
        )

    overall_avg = mean(results[a]["avg_ms"] for a in ACTIONS)

    print()
    print(f"Average GPU motor time: {overall_avg:.1f} ms")
    print("Brian2 40 ms @ 250 Hz reference: ~354.9 ms")

    all_pass = all(
        results[a]["successes"] == TRIALS
        for a in ACTIONS
    )

    print(
        "Reliability:",
        "PASS (20/20 each)" if all_pass else "NOT YET RELIABLE"
    )


if __name__ == "__main__":
    main()
