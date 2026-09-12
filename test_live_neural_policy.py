import time

from live_neural_mbon_policy import LiveNeuralMBONPolicy

print()
print("Creating live persistent neural policy...")
brain = LiveNeuralMBONPolicy(run_time_ms=50)

print()
print("Testing ONE live decision for state (-1, 0)...")

start = time.perf_counter()

action, scores = brain.choose_action(
    (-1, 0),
    epsilon=0,
)

elapsed = time.perf_counter() - start

print()
print("RESULT")
print("Action:", action)
print("Scores:", scores)
print(f"Decision time: {elapsed:.2f} seconds")

if action == "LEFT":
    print("PASS: live Brian2 chose LEFT")
else:
    print("FAIL: expected LEFT")
