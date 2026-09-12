import time
from statistics import mean

import fly_actions


DURATIONS = [0.05, 0.04, 0.03]   # 50 ms, 40 ms, 30 ms
TRIALS_PER_ACTION = 20
ACTIONS = ["LEFT", "RIGHT", "UP", "DOWN"]


def rate(out, name):
    return out.get(name, {}).get("rate_hz", 0.0)


def spikes(out, name):
    return out.get(name, {}).get("spikes", 0)


def action_success(action, out):
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


def pathway_spikes(action, out):
    if action == "LEFT":
        return (
            spikes(out, "DNa02_left"),
            spikes(out, "DNa02_right"),
        )

    if action == "RIGHT":
        return (
            spikes(out, "DNa02_right"),
            spikes(out, "DNa02_left"),
        )

    if action == "UP":
        return (
            spikes(out, "P9_oDN1_left")
            + spikes(out, "P9_oDN1_right")
        )

    if action == "DOWN":
        return sum(
            spikes(out, f"MDN_{i}")
            for i in range(1, 5)
        )

    return 0


def benchmark_brain(brain, duration):
    print()
    print("=" * 92)
    print(
        f"MOTOR WINDOW: {duration * 1000:.0f} ms | "
        f"{TRIALS_PER_ACTION} trials per action"
    )
    print("=" * 92)

    duration_results = {}

    for action in ACTIONS:
        successes = 0
        timings = []
        pathway_counts = []

        for trial in range(1, TRIALS_PER_ACTION + 1):
            t0 = time.perf_counter()
            result = brain.run(action)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            out = result["output_neuron_activity"]
            ok = action_success(action, out)

            successes += int(ok)
            timings.append(elapsed_ms)
            pathway_counts.append(pathway_spikes(action, out))

            status = "PASS" if ok else "FAIL"

            print(
                f"{action:5s} "
                f"{trial:02d}/{TRIALS_PER_ACTION} "
                f"{status} | "
                f"{elapsed_ms:7.1f} ms | "
                f"path spikes={pathway_counts[-1]}"
            )

        success_pct = successes / TRIALS_PER_ACTION * 100.0

        duration_results[action] = {
            "successes": successes,
            "success_pct": success_pct,
            "avg_ms": mean(timings),
            "min_ms": min(timings),
            "max_ms": max(timings),
            "min_pathway_spikes": min(pathway_counts),
        }

        print(
            f"\n{action} SUMMARY: "
            f"{successes}/{TRIALS_PER_ACTION} "
            f"({success_pct:.1f}%) | "
            f"avg {mean(timings):.1f} ms | "
            f"range {min(timings):.1f}-{max(timings):.1f} ms | "
            f"min pathway spikes {min(pathway_counts)}\n"
        )

    return duration_results


def main():
    all_results = {}

    # Reuse the already-built 50 ms brain from fly_actions.py.
    brains = {
        0.05: fly_actions._motor_brain
    }

    for duration in DURATIONS:
        if duration not in brains:
            brains[duration] = fly_actions.PersistentMotorBrain(
                freq_hz=200.0,
                duration_sec=duration,
            )

        all_results[duration] = benchmark_brain(
            brains[duration],
            duration,
        )

    print()
    print("=" * 92)
    print("FINAL SUMMARY")
    print("=" * 92)
    print(
        f"{'Window':>8} | "
        f"{'LEFT':>9} | "
        f"{'RIGHT':>9} | "
        f"{'UP':>9} | "
        f"{'DOWN':>9} | "
        f"{'Avg motor':>10} | "
        f"Verdict"
    )
    print("-" * 92)

    best_safe = None

    for duration in DURATIONS:
        r = all_results[duration]

        percentages = [
            r[action]["success_pct"]
            for action in ACTIONS
        ]

        avg_motor = mean(
            r[action]["avg_ms"]
            for action in ACTIONS
        )

        # For now, only call a window "safe" if every action
        # passed all 20 stochastic trials.
        safe = all(p == 100.0 for p in percentages)

        if safe:
            best_safe = duration

        verdict = "SAFE 20/20 EACH" if safe else "HAS FAILURES"

        print(
            f"{duration * 1000:7.0f}ms | "
            f"{r['LEFT']['success_pct']:8.1f}% | "
            f"{r['RIGHT']['success_pct']:8.1f}% | "
            f"{r['UP']['success_pct']:8.1f}% | "
            f"{r['DOWN']['success_pct']:8.1f}% | "
            f"{avg_motor:9.1f}ms | "
            f"{verdict}"
        )

    print()

    if best_safe is None:
        print(
            "No tested window achieved 20/20 for every action. "
            "Keep 50 ms for now and increase reliability."
        )
    else:
        print(
            "Shortest tested window with 20/20 success for every action:",
            f"{best_safe * 1000:.0f} ms"
        )

    print()
    print(
        "Do not change fly_actions.py yet. "
        "Use this benchmark result first."
    )


if __name__ == "__main__":
    main()
