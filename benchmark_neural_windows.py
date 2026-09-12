import time

from live_neural_mbon_policy_cached import LiveNeuralMBONPolicyCached


VALID = {
    (-1, -1): {"LEFT", "UP"},
    (0, -1): {"UP"},
    (1, -1): {"RIGHT", "UP"},
    (-1, 0): {"LEFT"},
    (1, 0): {"RIGHT"},
    (-1, 1): {"LEFT", "DOWN"},
    (0, 1): {"DOWN"},
    (1, 1): {"RIGHT", "DOWN"},
}

DURATIONS_MS = [50, 30, 20, 10]

summary = []

for duration in DURATIONS_MS:
    print()
    print("=" * 90)
    print(f"TESTING {duration} ms")
    print("=" * 90)

    startup_start = time.perf_counter()

    brain = LiveNeuralMBONPolicyCached(
        run_time_ms=duration
    )

    startup_time = time.perf_counter() - startup_start

    correct = 0
    decision_times = []

    for state, valid in VALID.items():
        start = time.perf_counter()

        action, scores = brain.choose_action(
            state,
            epsilon=0
        )

        elapsed = time.perf_counter() - start
        decision_times.append(elapsed)

        ok = action in valid
        correct += int(ok)

        print(
            f"{state} -> {action:5s} "
            f"{'PASS' if ok else 'FAIL'} | "
            f"{elapsed:.3f}s"
        )

    avg_time = sum(decision_times) / len(decision_times)

    summary.append({
        "duration": duration,
        "correct": correct,
        "startup": startup_time,
        "avg": avg_time,
        "fastest": min(decision_times),
        "slowest": max(decision_times),
    })

    print()
    print(
        f"{duration} ms result: "
        f"{correct}/8 correct, "
        f"avg {avg_time:.3f}s/decision"
    )

print()
print("=" * 90)
print("FINAL DURATION BENCHMARK")
print("=" * 90)

for row in summary:
    print(
        f"{row['duration']:>2} ms | "
        f"{row['correct']}/8 correct | "
        f"avg={row['avg']:.3f}s | "
        f"fastest={row['fastest']:.3f}s | "
        f"slowest={row['slowest']:.3f}s | "
        f"startup={row['startup']:.2f}s"
    )

perfect = [
    row for row in summary
    if row["correct"] == 8
]

if perfect:
    best = min(
        perfect,
        key=lambda row: row["duration"]
    )

    print()
    print(
        "SHORTEST 8/8 WINDOW:",
        f"{best['duration']} ms"
    )

    print(
        "Average decision time there:",
        f"{best['avg']:.3f}s"
    )
else:
    print()
    print(
        "None of the tested shorter windows preserved 8/8."
    )
