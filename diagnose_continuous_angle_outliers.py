#!/usr/bin/env python3
"""
diagnose_continuous_angle_outliers.py

Reads the saved Stage-3 metrics JSON only.
No neural simulation, no training, no memory changes.

It reports:
- all directions with >15 deg error
- worst 20 directions
- whether failures cluster near the 22.5 deg population boundaries
- error by distance to the nearest 22.5 deg anchor
"""

import json
import math
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent
METRICS = BASE / "data" / "fly_memory_CONTINUOUS_STIM_1deg_metrics.json"

BOUNDARY_STEP = 22.5


def circular_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def distance_to_nearest_anchor(angle):
    anchors = [i * BOUNDARY_STEP for i in range(16)]
    return min(circular_diff(angle, a) for a in anchors)


def nearest_anchor(angle):
    anchors = [i * BOUNDARY_STEP for i in range(16)]
    return min(anchors, key=lambda a: circular_diff(angle, a))


def main():
    if not METRICS.exists():
        raise FileNotFoundError(
            f"Could not find:\n  {METRICS}\n\n"
            "Make sure this script is inside your fly-brain folder."
        )

    with open(METRICS, "r", encoding="utf-8") as f:
        data = json.load(f)

    best = data.get("best", {})
    rows = best.get("rows", [])

    if not rows:
        raise RuntimeError(
            "The metrics file has no per-angle rows under best.rows."
        )

    rows = sorted(rows, key=lambda r: float(r["target"]))

    bad15 = [r for r in rows if float(r["error"]) > 15.0]
    worst = sorted(rows, key=lambda r: float(r["error"]), reverse=True)[:20]

    print()
    print("=" * 78)
    print("STAGE-3 CONTINUOUS ANGLE OUTLIER DIAGNOSTIC")
    print("=" * 78)
    print(f"Metrics file: {METRICS}")
    print(f"Total angles: {len(rows)}")
    print(f">15 deg error: {len(bad15)}")
    print()

    print("WORST 20 DIRECTIONS")
    print("-" * 78)
    print("target   predicted   error    nearest 22.5° anchor   distance")
    print("-" * 78)

    for r in worst:
        target = float(r["target"])
        pred = r.get("predicted")
        err = float(r["error"])
        anchor = nearest_anchor(target)
        dist = distance_to_nearest_anchor(target)

        pred_text = "None" if pred is None else f"{float(pred):8.2f}"

        print(
            f"{target:6.1f}   "
            f"{pred_text:>9}   "
            f"{err:6.2f}°      "
            f"{anchor:6.1f}°             "
            f"{dist:5.1f}°"
        )

    print()
    print("ALL DIRECTIONS ABOVE 15° ERROR")
    print("-" * 78)

    if not bad15:
        print("None.")
    else:
        for r in bad15:
            target = float(r["target"])
            pred = r.get("predicted")
            err = float(r["error"])
            anchor = nearest_anchor(target)
            dist = distance_to_nearest_anchor(target)

            pred_text = "None" if pred is None else f"{float(pred):.2f}°"

            print(
                f"target={target:6.1f}° | "
                f"pred={pred_text:>8} | "
                f"err={err:6.2f}° | "
                f"nearest_anchor={anchor:6.1f}° | "
                f"anchor_dist={dist:4.1f}°"
            )

    # Bin by distance from the nearest 22.5° anchor.
    # For 1° targets the maximum possible distance is 11.25°.
    bins = [
        (0.0, 2.0),
        (2.0, 4.0),
        (4.0, 6.0),
        (6.0, 8.0),
        (8.0, 10.0),
        (10.0, 11.25 + 1e-9),
    ]

    print()
    print("ERROR VS DISTANCE FROM NEAREST 22.5° ANCHOR")
    print("-" * 78)
    print("distance bin     n    mean err   median err   max err   >15°")
    print("-" * 78)

    for lo, hi in bins:
        selected = []

        for r in rows:
            d = distance_to_nearest_anchor(float(r["target"]))
            include = (
                lo <= d < hi
                if hi < 11.25
                else lo <= d <= hi
            )

            if include:
                selected.append(float(r["error"]))

        if not selected:
            continue

        print(
            f"{lo:4.1f}-{min(hi,11.25):4.2f}°   "
            f"{len(selected):3d}   "
            f"{statistics.mean(selected):8.2f}°   "
            f"{statistics.median(selected):10.2f}°   "
            f"{max(selected):7.2f}°   "
            f"{sum(e > 15.0 for e in selected):3d}"
        )

    # Count severe errors around each anchor.
    anchor_groups = {}

    for r in bad15:
        target = float(r["target"])
        anchor = nearest_anchor(target)
        anchor_groups.setdefault(anchor, []).append(r)

    print()
    print(">15° FAILURES GROUPED BY NEAREST 22.5° ANCHOR")
    print("-" * 78)

    if not anchor_groups:
        print("None.")
    else:
        for anchor in sorted(anchor_groups):
            group = anchor_groups[anchor]
            mean_err = statistics.mean(float(r["error"]) for r in group)
            targets = ", ".join(f"{float(r['target']):.0f}°" for r in group)

            print(
                f"{anchor:6.1f}° anchor | "
                f"{len(group):2d} failures | "
                f"mean={mean_err:6.2f}° | "
                f"targets: {targets}"
            )

    print()
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print(
        "If the large-error rows bunch tightly near certain 22.5° anchors, "
        "the problem is likely a population-transition / threshold effect."
    )
    print(
        "If they bunch around only certain compass regions regardless of "
        "anchor distance, the specific KC→MBON connectivity for those regions "
        "is the more likely bottleneck."
    )
    print()


if __name__ == "__main__":
    main()
