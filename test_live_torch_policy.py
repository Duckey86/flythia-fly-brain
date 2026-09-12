import time
import torch
from live_torch_mbon_policy import LiveTorchMBONPolicy

VALID = {
    (-1,-1): {"LEFT","UP"},
    (0,-1): {"UP"},
    (1,-1): {"RIGHT","UP"},
    (-1,0): {"LEFT"},
    (1,0): {"RIGHT"},
    (-1,1): {"LEFT","DOWN"},
    (0,1): {"DOWN"},
    (1,1): {"RIGHT","DOWN"},
}

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print()

brain = LiveTorchMBONPolicy(run_time_ms=3, device="cuda")

# Warm one decision before timing.
brain.choose_action((-1,0), epsilon=0)

correct = 0
times = []

print()
print("="*80)
print("TESTING ALL 8 GPU DECISIONS")
print("="*80)

for state, valid in VALID.items():
    torch.cuda.synchronize()
    start = time.perf_counter()

    action, scores = brain.choose_action(state, epsilon=0)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    times.append(elapsed)

    ok = action in valid
    correct += int(ok)

    print(
        state,
        "->",
        action,
        "PASS" if ok else "FAIL",
        f"| {elapsed:.4f}s",
    )
    print()

print("="*80)
print("FINAL GPU RESULT")
print("="*80)
print(f"Correct: {correct}/8")
print(f"Average GPU decision time: {sum(times)/len(times):.4f}s")
print(f"Fastest: {min(times):.4f}s")
print(f"Slowest: {max(times):.4f}s")
print("Brian2 CPU reference: ~0.168s/decision at 3 ms.")
