"""EvidenceScene kernel performance suite.

The suite measures deterministic, output-preserving library workloads.  It is
not a recorded-session endpoint benchmark and does not compare pose accuracy.
Every workload records a digest of its output; a speed comparison is only
meaningful when the baseline and candidate digests match.

Examples
--------
Run the current package::

    python3 benchmarks/performance/bench_suite.py --out results/current.json

Run the same suite against a past git worktree::

    python3 benchmarks/performance/bench_suite.py \
        --package-root /path/to/old/worktree --label v2.0.0 --out results/v2.json

Compare two runs::

    python3 benchmarks/performance/compare.py results/v2.json results/current.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "evidencescene.performance-benchmark.v1"
BENCHMARK_ID = "kernel-v1"


def _timed_ms(function, *, repeat: int, warmup: int = 2) -> dict[str, float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(samples)
    return {
        "median_ms": round(statistics.median(samples), 4),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4),
        "min_ms": round(min(samples), 4),
        "repeat": repeat,
    }


def _digest(*arrays: object) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def _scalar_digest(*values: object) -> str:
    payload = json.dumps([repr(v) for v in values], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _load_api(package_root: Path) -> SimpleNamespace:
    root = str(package_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import evidencescene

    from evidencescene.primitives.image_mask import encode_mask_rle
    from evidencescene.runtime.request_evidence_deduplication import (
        rle_mask_box_and_centroid,
    )
    from evidencescene.surface.object_surface_fusion import (
        RevisableSurfelMap,
        observation_voxel_centroids,
    )
    from evidencescene.surface.shadow_surface_snapshot_selection import (
        select_existing_points_shadow,
    )
    from evidencescene.teleop.task_features import (
        compare_circle_consensus_3d,
        fit_circular_aperture_3d,
    )

    return SimpleNamespace(
        package=evidencescene,
        RevisableSurfelMap=RevisableSurfelMap,
        observation_voxel_centroids=observation_voxel_centroids,
        encode_mask_rle=encode_mask_rle,
        rle_mask_box_and_centroid=rle_mask_box_and_centroid,
        select_existing_points_shadow=select_existing_points_shadow,
        compare_circle_consensus_3d=compare_circle_consensus_3d,
        fit_circular_aperture_3d=fit_circular_aperture_3d,
    )


def _grid_points(count: int = 60_000, voxel_size_m: float = 0.004) -> np.ndarray:
    grid = np.indices((40, 40, 40)).reshape(3, -1).T[:count]
    return grid.astype(np.float64) * voxel_size_m


def _snapshot(api: SimpleNamespace, repeat: int = 20) -> dict:
    points = _grid_points()
    surfel_map = api.RevisableSurfelMap(
        voxel_size_m=0.004,
        maximum_voxels=100_000,
        maximum_live_bytes=512 * 1024 * 1024,
    )
    surfel_map.update({"chest": ("chest", points)})
    surfel_map.update({"head": ("head", points)})
    result = _timed_ms(
        lambda: surfel_map.snapshot(
            minimum_observation_support=2,
            maximum_point_count=1_500,
        ),
        repeat=repeat,
    )
    xyz, support, cameras = surfel_map.snapshot(
        minimum_observation_support=2,
        maximum_point_count=1_500,
    )
    return {
        "case": "surface_snapshot_60k_budget_1500",
        "unit": "ms",
        "direction": "lower_is_better",
        "output_digest": _digest(xyz, support, cameras),
        **result,
    }


def _update(api: SimpleNamespace, repeat: int = 8) -> dict:
    points = _grid_points()
    samples = []
    last_map = None
    for _ in range(max(2, repeat)):
        surfel_map = api.RevisableSurfelMap(
            voxel_size_m=0.004,
            maximum_voxels=100_000,
            maximum_live_bytes=512 * 1024 * 1024,
        )
        start = time.perf_counter()
        surfel_map.update({"chest": ("chest", points)})
        samples.append((time.perf_counter() - start) * 1000.0)
        last_map = surfel_map
    ordered = sorted(samples)
    xyz, support, cameras = last_map.snapshot(
        minimum_observation_support=1,
        maximum_point_count=1_500,
    )
    return {
        "case": "surface_update_60k_points",
        "unit": "ms",
        "direction": "lower_is_better",
        "output_digest": _digest(xyz, support, cameras),
        "median_ms": round(statistics.median(samples), 4),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4),
        "min_ms": round(min(samples), 4),
        "repeat": len(samples),
    }


def _voxel_centroids(api: SimpleNamespace, repeat: int = 20) -> dict:
    rng = np.random.default_rng(5)
    points = rng.integers(0, 40, size=(60_000, 3)).astype(np.float64) * 0.004
    points += rng.normal(0.0, 1e-6, points.shape)
    result = _timed_ms(
        lambda: api.observation_voxel_centroids(points, 0.004),
        repeat=repeat,
    )
    unique, centroids = api.observation_voxel_centroids(points, 0.004)
    return {
        "case": "voxel_centroids_60k_points",
        "unit": "ms",
        "direction": "lower_is_better",
        "output_digest": _digest(unique, centroids),
        **result,
    }


def _rle(api: SimpleNamespace, repeat: int = 30) -> dict:
    mask = np.zeros((480, 640), dtype=bool)
    mask[80:400, 100:540] = True
    mask[150:180, 220:260] = False
    mask[300:330, 400:460] = False
    payload = api.encode_mask_rle(mask)
    result = _timed_ms(
        lambda: api.rle_mask_box_and_centroid(payload, width=640, height=480),
        repeat=repeat,
    )
    box, centroid = api.rle_mask_box_and_centroid(payload, width=640, height=480)
    return {
        "case": "rle_box_centroid_480x640",
        "unit": "ms",
        "direction": "lower_is_better",
        "output_digest": _scalar_digest(box, centroid),
        **result,
    }


def _selection(api: SimpleNamespace, repeat: int = 20) -> dict:
    points = _grid_points()
    surfel_map = api.RevisableSurfelMap(
        voxel_size_m=0.004,
        maximum_voxels=100_000,
        maximum_live_bytes=512 * 1024 * 1024,
    )
    surfel_map.update({"chest": ("chest", points)})
    surfel_map.update({"head": ("head", points)})
    xyz, support, cameras, keys = surfel_map.snapshot(
        minimum_observation_support=2,
        maximum_point_count=100_000,
        include_voxel_keys=True,
    )
    result = _timed_ms(
        lambda: api.select_existing_points_shadow(
            xyz, support, cameras, keys, budget=1_500
        ),
        repeat=repeat,
    )
    selection = api.select_existing_points_shadow(
        xyz, support, cameras, keys, budget=1_500
    )
    return {
        "case": "surface_selection_60k_budget_1500",
        "unit": "ms",
        "direction": "lower_is_better",
        "output_digest": _digest(selection["indices"]),
        **result,
    }


def _consensus(api: SimpleNamespace, repeat: int = 50) -> dict:
    rng = np.random.default_rng(3)
    angles = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    rim = np.column_stack([0.04 * np.cos(angles), 0.04 * np.sin(angles)])
    rim += rng.normal(0.0, 0.0005, rim.shape)
    result = _timed_ms(
        lambda: api.compare_circle_consensus_3d(
            rim,
            fitted_center_2d=np.zeros(2),
            fitted_radius_m=0.04,
        ),
        repeat=repeat,
    )
    agreed, center, radius = api.compare_circle_consensus_3d(
        rim,
        fitted_center_2d=np.zeros(2),
        fitted_radius_m=0.04,
    )
    return {
        "case": "circle_consensus_400_points_256_hypotheses",
        "unit": "ms",
        "direction": "lower_is_better",
        "output_digest": _scalar_digest(agreed, center, radius),
        **result,
    }


def _aperture(api: SimpleNamespace, repeat: int = 30) -> dict:
    rng = np.random.default_rng(3)
    angles = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    rim = np.column_stack(
        [0.04 * np.cos(angles), 0.04 * np.sin(angles), np.zeros_like(angles)]
    )
    rim += rng.normal(0.0, 0.0005, rim.shape)
    result = _timed_ms(lambda: api.fit_circular_aperture_3d(rim), repeat=repeat)
    aperture = api.fit_circular_aperture_3d(rim)
    return {
        "case": "circular_aperture_fit_400_points",
        "unit": "ms",
        "direction": "lower_is_better",
        "output_digest": _scalar_digest(
            aperture.center_m,
            aperture.normal,
            aperture.radius_m,
        ),
        **result,
    }


WORKLOADS = (
    ("surface_snapshot_60k_budget_1500", _snapshot, 20),
    ("surface_update_60k_points", _update, 8),
    ("voxel_centroids_60k_points", _voxel_centroids, 20),
    ("rle_box_centroid_480x640", _rle, 30),
    ("surface_selection_60k_budget_1500", _selection, 20),
    ("circle_consensus_400_points_256_hypotheses", _consensus, 50),
    ("circular_aperture_fit_400_points", _aperture, 30),
)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _git_sha(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_suite(
    *,
    package_root: Path,
    label: str,
    repeat_scale: float = 1.0,
    only: set[str] | None = None,
) -> dict:
    api = _load_api(package_root)
    results = []
    for case, function, default_repeat in WORKLOADS:
        if only and case not in only:
            continue
        repeat = max(2, int(round(default_repeat * repeat_scale)))
        try:
            result = function(api, repeat=repeat)
            result["status"] = "ok"
        except Exception as error:  # noqa: BLE001 - report unsupported old APIs
            result = {
                "case": case,
                "unit": "ms",
                "direction": "lower_is_better",
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "repeat": repeat,
            }
        results.append(result)
    return {
        "schema": SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "label": label,
        "scope": (
            "synthetic kernel microbenchmark; not recorded-session endpoint "
            "latency and not a pose/surface accuracy evaluation"
        ),
        "package": {
            "name": "evidencescene",
            "version": api.package.__version__,
            "root": str(package_root.resolve()),
            "git_sha": _git_sha(package_root),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu": _cpu_model(),
        },
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeat-scale", type=float, default=1.0)
    parser.add_argument("--only", action="append", default=None)
    args = parser.parse_args(argv)
    if args.repeat_scale <= 0:
        parser.error("--repeat-scale must be positive")
    scale = args.repeat_scale * (0.25 if args.quick else 1.0)
    label = args.label or _git_sha(args.package_root) or "uncommitted"
    payload = run_suite(
        package_root=args.package_root,
        label=label,
        repeat_scale=scale,
        only=set(args.only) if args.only else None,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
