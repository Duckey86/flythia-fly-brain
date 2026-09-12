import json
import subprocess
import sys

# First see whether v2 itself fixes latency.
# Then trade duration against stronger stimulation for UP reliability.
CONFIGS = [
    (40, 250),
    (40, 300),
    (40, 350),
    (35, 300),
    (35, 350),
    (35, 400),
    (30, 350),
    (30, 400),
    (30, 450),
]

TRIALS = 30
results = []

for duration_ms, stim_hz in CONFIGS:
    print()
    print("=" * 88)
    print(
        f"GPU V2 TEST: {duration_ms} ms @ {stim_hz} Hz "
        f"| {TRIALS} trials/action"
    )
    print("=" * 88)

    proc = subprocess.run(
        [
            sys.executable,
            "gpu_motor_v2_worker.py",
            "--duration-ms", str(duration_ms),
            "--stim-hz", str(stim_hz),
            "--trials", str(TRIALS),
        ],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        print("FAILED")
        continue

    marker = None
    for line in proc.stdout.splitlines():
        if line.startswith("JSON_RESULT="):
            marker = line[len("JSON_RESULT="):]

    if marker is None:
        print(proc.stdout)
        print("No JSON_RESULT found.")
        continue

    r = json.loads(marker)
    results.append(r)

    a = r["actions"]

    print(
        f"LEFT {a['LEFT']['pct']:5.1f}% | "
        f"RIGHT {a['RIGHT']['pct']:5.1f}% | "
        f"UP {a['UP']['pct']:5.1f}% | "
        f"DOWN {a['DOWN']['pct']:5.1f}% | "
        f"avg {r['avg_motor_ms']:.1f} ms"
    )

print()
print("=" * 104)
print("FINAL GPU V2 SUMMARY")
print("=" * 104)
print(
    f"{'Window':>8} | {'Stim':>7} | {'LEFT':>7} | {'RIGHT':>7} | "
    f"{'UP':>7} | {'DOWN':>7} | {'Avg motor':>10} | Verdict"
)
print("-" * 104)

best = None

for r in results:
    a = r["actions"]
    reliable = all(a[x]["pct"] == 100.0 for x in a)

    print(
        f"{r['duration_ms']:7.0f}ms | "
        f"{r['stim_hz']:6.0f}Hz | "
        f"{a['LEFT']['pct']:6.1f}% | "
        f"{a['RIGHT']['pct']:6.1f}% | "
        f"{a['UP']['pct']:6.1f}% | "
        f"{a['DOWN']['pct']:6.1f}% | "
        f"{r['avg_motor_ms']:9.1f}ms | "
        f"{'30/30 EACH' if reliable else 'HAS FAILURES'}"
    )

    if reliable:
        if best is None or r["avg_motor_ms"] < best["avg_motor_ms"]:
            best = r

print()
print("Brian2 reference: 40 ms @ 250 Hz = ~354.9 ms average")

if best is None:
    print("No GPU v2 configuration achieved 30/30 on every action.")
else:
    print(
        "Best reliable GPU v2:",
        f"{best['duration_ms']:.0f} ms @ {best['stim_hz']:.0f} Hz",
        f"= {best['avg_motor_ms']:.1f} ms average"
    )
