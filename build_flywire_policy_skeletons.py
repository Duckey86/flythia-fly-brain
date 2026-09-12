import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ACTION_MBONS = {
    "LEFT": "720575940617552340",
    "RIGHT": "720575940624185095",
    "UP": "720575940623201833",
    "DOWN": "720575940629422086",
}

BASE = Path(__file__).resolve().parent
DEFAULT_PATTERN = BASE / "priority_kc_subpatterns_16.json"
DEFAULT_COMPLETENESS = BASE / "data" / "2025_Completeness_783.csv"
DEFAULT_OUT = BASE / "data" / "flybrain_visual" / "flybrain_policy_skeletons.json"


def require_flywire_packages():
    try:
        import navis  # noqa: F401
        from fafbseg import flywire  # noqa: F401
    except ImportError as exc:
        print()
        print("Missing FlyWire visualization packages.")
        print("Run this once inside your brain-fly environment:")
        print()
        print("    python -m pip install -U fafbseg navis")
        print()
        raise SystemExit(2) from exc


def read_inputs(pattern_file: Path, completeness_file: Path):
    with open(pattern_file, "r", encoding="utf-8") as f:
        patterns = json.load(f)

    comp = pd.read_csv(completeness_file, index_col=0)
    root_ids = [str(int(x)) for x in comp.index]

    state_roots = {}
    membership = {}

    for state_key, info in patterns.items():
        state_ids = []
        for idx in info.get("kc_indices", []):
            idx = int(idx)
            if idx < 0 or idx >= len(root_ids):
                raise IndexError(
                    f"KC index {idx} from state {state_key} is outside "
                    f"completeness table size {len(root_ids)}"
                )
            rid = root_ids[idx]
            state_ids.append(rid)
            membership.setdefault(rid, []).append(state_key)
        state_roots[state_key] = state_ids

    policy_ids = set()
    for ids in state_roots.values():
        policy_ids.update(ids)
    policy_ids.update(ACTION_MBONS.values())

    return patterns, root_ids, state_roots, membership, sorted(policy_ids)


def neuron_to_segments(neuron, downsample_factor: int, max_segments: int):
    import navis

    # Preserve branch points while reducing long straight chains.
    factor = max(1, int(downsample_factor))

    try:
        n_nodes = int(neuron.n_nodes)
    except Exception:
        n_nodes = len(neuron.nodes)

    if max_segments > 0 and n_nodes > max_segments:
        factor = max(factor, int(math.ceil(n_nodes / max_segments)))

    if factor > 1:
        try:
            neuron = navis.downsample_neuron(
                neuron,
                downsampling_factor=factor,
                inplace=False,
            )
        except Exception as exc:
            print(f"  warning: downsample failed for {neuron.id}: {exc}")

    nodes = neuron.nodes
    required = {"node_id", "parent_id", "x", "y", "z"}
    missing = required.difference(nodes.columns)
    if missing:
        raise ValueError(
            f"Neuron {neuron.id} is missing skeleton columns: {sorted(missing)}"
        )

    coords = {}
    for row in nodes[["node_id", "x", "y", "z"]].itertuples(index=False):
        coords[int(row.node_id)] = (
            float(row.x),
            float(row.y),
            float(row.z),
        )

    flat = []
    min_xyz = [float("inf"), float("inf"), float("inf")]
    max_xyz = [float("-inf"), float("-inf"), float("-inf")]

    for row in nodes[["node_id", "parent_id", "x", "y", "z"]].itertuples(index=False):
        pid = row.parent_id
        if pd.isna(pid):
            continue
        pid = int(pid)
        if pid < 0 or pid not in coords:
            continue

        p = (float(row.x), float(row.y), float(row.z))
        q = coords[pid]

        flat.extend((p[0], p[1], p[2], q[0], q[1], q[2]))

        for xyz in (p, q):
            for j in range(3):
                min_xyz[j] = min(min_xyz[j], xyz[j])
                max_xyz[j] = max(max_xyz[j], xyz[j])

    return flat, min_xyz, max_xyz, factor


def normalize_vertices(vertices, center, scale):
    out = []

    # FAFB skeleton coordinates are nanometres. Keep the real relative 3D
    # anatomy, then map to a compact Godot coordinate system.
    # Godot Y is flipped so the default dorsal view is visually upright.
    cx, cy, cz = center

    for i in range(0, len(vertices), 3):
        x = (vertices[i] - cx) * scale
        y = -(vertices[i + 1] - cy) * scale
        z = (vertices[i + 2] - cz) * scale

        out.extend((round(x, 5), round(y, 5), round(z, 5)))

    return out


def fetch_skeletons(root_ids, batch_size, threads, downsample_factor, max_segments):
    """
    Fetch v783 precomputed skeletons WITHOUT triggering a CAVE/chunkedgraph
    existence check.

    Important fafbseg detail:
      - get_skeletons([list, ...], dataset=783) first calls is_latest_root(),
        which hits the authenticated FlyWire CAVE/chunkedgraph service.
      - get_skeletons(single_root_id, dataset=783) directly reads the public
        precomputed skeleton URL.

    Our IDs already come from the v783 completeness table, so we deliberately
    fetch them one-by-one (in a small local thread pool). This avoids the
    unnecessary 401-prone CAVE check while still retrieving the same real
    public v783 neuron morphologies.
    """
    from concurrent.futures import ThreadPoolExecutor
    from fafbseg import flywire

    raw = {}
    failed = []
    total = len(root_ids)

    def fetch_one(rid):
        try:
            result = flywire.get_skeletons(
                int(rid),
                dataset=783,
                omit_failures=True,
                progress=False,
            )

            # Scalar calls normally return TreeNeuron. Be tolerant of an
            # empty/single-item NeuronList as well.
            if hasattr(result, "nodes"):
                neuron = result
            else:
                seq = list(result)
                neuron = seq[0] if seq else None

            if neuron is None:
                return rid, None, "no precomputed skeleton"

            return rid, neuron, None
        except Exception as exc:
            return rid, None, str(exc)

    for start in range(0, total, batch_size):
        batch = root_ids[start:start + batch_size]

        print()
        print(
            f"Fetching public FlyWire v783 skeletons "
            f"{start + 1}-{min(start + len(batch), total)} / {total}..."
        )

        with ThreadPoolExecutor(max_workers=max(1, int(threads))) as pool:
            results = list(pool.map(fetch_one, batch))

        for rid, neuron, error in results:
            if neuron is None:
                print(f"  failed {rid}: {error}")
                failed.append(rid)
                continue

            try:
                flat, mn, mx, used_factor = neuron_to_segments(
                    neuron,
                    downsample_factor=downsample_factor,
                    max_segments=max_segments,
                )
            except Exception as exc:
                print(f"  failed to convert {rid}: {exc}")
                failed.append(rid)
                continue

            if not flat:
                print(f"  no drawable segments for {rid}")
                failed.append(rid)
                continue

            raw[rid] = {
                "vertices_raw": flat,
                "min": mn,
                "max": mx,
                "downsample_factor": used_factor,
                "segments": len(flat) // 6,
            }

            print(
                f"  {rid}: {len(flat) // 6:,} segments "
                f"(downsample x{used_factor})"
            )

    return raw, sorted(set(failed))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download the real FlyWire v783 morphologies used by the trained "
            "16-state Rhythia fly and convert them into a Godot-friendly cache."
        )
    )
    parser.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    parser.add_argument("--completeness", type=Path, default=DEFAULT_COMPLETENESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--godot-project",
        type=Path,
        default=None,
        help=(
            "Optional Rhythia Godot project root. If supplied, the cache is "
            "written directly to data/flybrain_visual/ inside that project."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--downsample",
        type=int,
        default=8,
        help="NAVis skeleton downsample factor. 8 is a good live-view default.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=1800,
        help="Soft cap per neuron; higher keeps more morphological detail.",
    )
    parser.add_argument(
        "--fit-size",
        type=float,
        default=10.0,
        help="Longest brain dimension in Godot units.",
    )
    args = parser.parse_args()

    require_flywire_packages()

    if not args.pattern.exists():
        raise FileNotFoundError(f"Pattern file not found: {args.pattern}")
    if not args.completeness.exists():
        raise FileNotFoundError(f"Completeness file not found: {args.completeness}")

    _, _, state_roots, membership, root_ids = read_inputs(
        args.pattern,
        args.completeness,
    )

    unique_kcs = set()
    for values in state_roots.values():
        unique_kcs.update(values)

    print("FlyWire morphology cache builder")
    print("--------------------------------")
    print(f"16-state sensory patterns: {len(state_roots)}")
    print(f"Unique policy KCs:         {len(unique_kcs)}")
    print(f"Action MBONs:              {len(ACTION_MBONS)}")
    print(f"Total real neurons:        {len(root_ids)}")
    print()
    print("These are real FAFB/FlyWire v783 neuron skeletons.")
    print("First run needs internet access and can take a few minutes.")

    raw, failed = fetch_skeletons(
        root_ids,
        batch_size=max(1, args.batch_size),
        threads=max(1, args.threads),
        downsample_factor=max(1, args.downsample),
        max_segments=max(0, args.max_segments),
    )

    if not raw:
        raise RuntimeError(
            "No FlyWire skeletons were downloaded. Check internet access and "
            "that fafbseg can access the public v783 skeleton service."
        )

    global_min = [float("inf"), float("inf"), float("inf")]
    global_max = [float("-inf"), float("-inf"), float("-inf")]

    for info in raw.values():
        for j in range(3):
            global_min[j] = min(global_min[j], info["min"][j])
            global_max[j] = max(global_max[j], info["max"][j])

    center = [
        (global_min[j] + global_max[j]) * 0.5
        for j in range(3)
    ]
    spans = [global_max[j] - global_min[j] for j in range(3)]
    longest = max(spans)
    scale = float(args.fit_size) / longest if longest > 0 else 1.0

    neurons = []
    total_segments = 0

    mbon_to_action = {rid: action for action, rid in ACTION_MBONS.items()}

    for rid in sorted(raw.keys(), key=int):
        info = raw[rid]
        role = "MBON" if rid in mbon_to_action else "KC"
        action = mbon_to_action.get(rid, "")

        vertices = normalize_vertices(
            info["vertices_raw"],
            center=center,
            scale=scale,
        )

        total_segments += info["segments"]

        neurons.append(
            {
                "id": rid,
                "role": role,
                "action": action,
                "states": membership.get(rid, []),
                "segments": info["segments"],
                "downsample_factor": info["downsample_factor"],
                "vertices": vertices,
            }
        )

    cache = {
        "format": "flythia_flywire_policy_skeletons_v1",
        "dataset": "FlyWire FAFB v783",
        "coordinate_units_source": "nanometer",
        "coordinate_transform": "Godot=(x,-y,z), globally centered and scaled",
        "source_bounds_nm": {
            "min": global_min,
            "max": global_max,
            "center": center,
        },
        "godot_fit_size": float(args.fit_size),
        "scale_from_nm": scale,
        "action_mbons": ACTION_MBONS,
        "states": state_roots,
        "neuron_count": len(neurons),
        "segment_count": total_segments,
        "failed_root_ids": failed,
        "neurons": neurons,
    }

    if args.godot_project is not None:
        output = (
            args.godot_project
            / "data"
            / "flybrain_visual"
            / "flybrain_policy_skeletons.json"
        )
    else:
        output = args.output

    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(cache, f, separators=(",", ":"))

    size_mb = output.stat().st_size / (1024 * 1024)

    print()
    print("========================================")
    print(" FLYWIRE POLICY MORPHOLOGY CACHE READY")
    print("========================================")
    print(f"Neurons:  {len(neurons):,}")
    print(f"Segments: {total_segments:,}")
    print(f"Size:     {size_mb:.1f} MB")
    print(f"Saved:    {output}")

    if failed:
        print()
        print(f"Warning: {len(failed)} neuron(s) had no precomputed skeleton:")
        for rid in failed[:20]:
            print("  ", rid)
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
        print("The visualizer will simply omit those neurons.")


if __name__ == "__main__":
    main()
