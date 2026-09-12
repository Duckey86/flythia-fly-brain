import json
import subprocess
import sys

CONFIGS = [
    # Establish a reliable baseline first.
    (0.100, 200),
    (0.075, 200),
    (0.060, 200),

    # Current fast window at stronger stimulation.
    (0.050, 200),
    (0.050, 250),
    (0.050, 300),

    # Only test 40 ms with stronger drive.
    (0.040, 250),
    (0.040, 300),
]

TRIALS = 50

results = []

for duration, freq in CONFIGS:
    print()
    print("=" * 88)
    print(f"TEST: {duration*1000:.0f} ms @ {freq} Hz, {TRIALS} trials/action")
    print("=" * 88)

    proc = subprocess.run(
        [
            sys.executable,
            "motor_benchmark_worker.py",
            "--duration", str(duration),
            "--freq", str(freq),
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
        print("No result marker found.")
        continue

    r = json.loads(marker)
    results.append(r)

    a = r["actions"]
    print(
        f"LEFT {a['LEFT']['success_pct']:5.1f}% | "
        f"RIGHT {a['RIGHT']['success_pct']:5.1f}% | "
        f"UP {a['UP']['success_pct']:5.1f}% | "
        f"DOWN {a['DOWN']['success_pct']:5.1f}% | "
        f"avg {r['avg_motor_ms']:.1f} ms"
    )

print()
print("=" * 100)
print("FINAL CLEAN SUMMARY")
print("=" * 100)
print(
    f"{'Window':>8} | {'Stim':>7} | {'LEFT':>7} | {'RIGHT':>7} | "
    f"{'UP':>7} | {'DOWN':>7} | {'Avg motor':>10} | Verdict"
)
print("-" * 100)

best = None

for r in results:
    a = r["actions"]
    all_100 = all(a[x]["success_pct"] == 100.0 for x in a)

    verdict = "50/50 EACH" if all_100 else "HAS FAILURES"

    print(
        f"{r['duration_ms']:7.0f}ms | "
        f"{r['freq_hz']:6.0f}Hz | "
        f"{a['LEFT']['success_pct']:6.1f}% | "
        f"{a['RIGHT']['success_pct']:6.1f}% | "
        f"{a['UP']['success_pct']:6.1f}% | "
        f"{a['DOWN']['success_pct']:6.1f}% | "
        f"{r['avg_motor_ms']:9.1f}ms | "
        f"{verdict}"
    )

    if all_100:
        if best is None or r["avg_motor_ms"] < best["avg_motor_ms"]:
            best = r

print()

if best:
    print(
        "Best fully reliable tested config:",
        f"{best['duration_ms']:.0f} ms @ {best['freq_hz']:.0f} Hz",
        f"({best['avg_motor_ms']:.1f} ms average motor time)"
    )
else:
    print("No config achieved 50/50 for every action.")
