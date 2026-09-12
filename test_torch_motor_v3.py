from statistics import mean

from live_torch_motor_v3 import LiveTorchMotorV3, motor_success


TRIALS = 50
ACTIONS = ["LEFT", "RIGHT", "UP", "DOWN"]


def main():
    motor = LiveTorchMotorV3(
        run_time_ms=40,
        stim_hz=250,
        device="cuda",
    )

    # Warm each pathway once; do not count warm-up.
    for action in ACTIONS:
        motor.run(action)

    results = {}

    for action in ACTIONS:
        successes = 0
        times = []

        print()
        print("=" * 76)
        print(action)
        print("=" * 76)

        for trial in range(1, TRIALS + 1):
            r = motor.run(action)
            out = r["output_neuron_activity"]

            ok = motor_success(action, out)
            successes += int(ok)
            times.append(r["wall_time_sec"] * 1000.0)

            if trial <= 5 or not ok:
                active = {
                    name: data["spikes"]
                    for name, data in out.items()
                    if data["spikes"] > 0
                }

                print(
                    f"{trial:02d}/{TRIALS} "
                    f"{'PASS' if ok else 'FAIL'} | "
                    f"{times[-1]:.1f} ms | "
                    f"spikes={active}"
                )

        results[action] = {
            "successes": successes,
            "pct": successes / TRIALS * 100.0,
            "avg_ms": mean(times),
            "min_ms": min(times),
            "max_ms": max(times),
        }

    print()
    print("=" * 90)
    print("FINAL GPU V3 RESULT")
    print("=" * 90)

    for action in ACTIONS:
        r = results[action]

        print(
            f"{action:5s}: "
            f"{r['successes']:2d}/{TRIALS} "
            f"({r['pct']:5.1f}%) | "
            f"avg {r['avg_ms']:.1f} ms | "
            f"range {r['min_ms']:.1f}-{r['max_ms']:.1f} ms"
        )

    overall = mean(results[a]["avg_ms"] for a in ACTIONS)

    print()
    print(f"Average GPU v3 motor time: {overall:.1f} ms")
    print("Brian2 reference: ~354.9 ms at 40 ms @ 250 Hz")

    reliable = all(
        results[a]["successes"] == TRIALS
        for a in ACTIONS
    )

    print(
        "Reliability:",
        "PASS (50/50 each)" if reliable else "NOT YET RELIABLE"
    )


if __name__ == "__main__":
    main()
