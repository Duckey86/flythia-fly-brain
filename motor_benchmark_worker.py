import argparse
import json
import time
from statistics import mean

from fly_actions import PersistentMotorBrain


OUTPUTS = [
    "DNa02_left",
    "DNa02_right",
    "P9_oDN1_left",
    "P9_oDN1_right",
    "MDN_1",
    "MDN_2",
    "MDN_3",
    "MDN_4",
]


def rate(out, name):
    return out.get(name, {}).get("rate_hz", 0.0)


def success(action, out):
    dl = rate(out, "DNa02_left")
    dr = rate(out, "DNa02_right")
    pl = rate(out, "P9_oDN1_left")
    pr = rate(out, "P9_oDN1_right")
    mdns = [rate(out, f"MDN_{i}") for i in range(1, 5)]

    if action == "LEFT":
        return dl > dr and dl >= 20
    if action == "RIGHT":
        return dr > dl and dr >= 20
    if action == "UP":
        return (pl + pr) / 2 >= 10
    if action == "DOWN":
        return sum(mdns) / 4 >= 20
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--freq", type=float, required=True)
    p.add_argument("--trials", type=int, default=50)
    args = p.parse_args()

    brain = PersistentMotorBrain(
        freq_hz=args.freq,
        duration_sec=args.duration,
    )

    actions = ["LEFT", "RIGHT", "UP", "DOWN"]
    result = {
        "duration_ms": args.duration * 1000,
        "freq_hz": args.freq,
        "trials": args.trials,
        "actions": {},
    }

    # One warm-up per action; not included in timing/reliability totals.
    for action in actions:
        brain.run(action)

    for action in actions:
        ok = 0
        times = []

        for _ in range(args.trials):
            t0 = time.perf_counter()
            r = brain.run(action)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            times.append(elapsed_ms)

            if success(action, r["output_neuron_activity"]):
                ok += 1

        result["actions"][action] = {
            "successes": ok,
            "success_pct": 100.0 * ok / args.trials,
            "avg_ms": mean(times),
            "min_ms": min(times),
            "max_ms": max(times),
        }

    result["avg_motor_ms"] = mean(
        result["actions"][a]["avg_ms"] for a in actions
    )

    print("JSON_RESULT=" + json.dumps(result))


if __name__ == "__main__":
    main()
