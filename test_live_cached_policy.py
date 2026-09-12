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


print()
print("Creating cached-baseline live neural policy...")

brain = LiveNeuralMBONPolicyCached(
    run_time_ms=10
)

print()
print("=" * 80)
print("TESTING ALL 8 LIVE DECISIONS")
print("=" * 80)

correct = 0
times = []

for state, valid in VALID.items():

    start = time.perf_counter()

    action, scores = brain.choose_action(
        state,
        epsilon=0,
    )

    elapsed = time.perf_counter() - start
    times.append(elapsed)

    ok = action in valid
    correct += int(ok)

    print(
        state,
        "->",
        action,
        "PASS" if ok else "FAIL",
    )
    print()

print("=" * 80)
print("FINAL")
print("=" * 80)
print(
    f"Correct: {correct}/{len(VALID)}"
)

print(
    f"Average live decision time: "
    f"{sum(times)/len(times):.2f}s"
)

print(
    f"Fastest: {min(times):.2f}s"
)

print(
    f"Slowest: {max(times):.2f}s"
)
