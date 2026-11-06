"""Synthetic microbenchmarks for EvidenceScene hot paths.

These numbers are deliberately separate from the recorded-session endpoint
latency receipts. They isolate two output-preserving Python/NumPy paths:

* ``RevisableSurfelMap.snapshot`` over a 60k-voxel map at a 1,500-point budget.
* ``compare_circle_consensus_3d`` on a 400-point synthetic rim.

Run from the repository root:

    python3 benchmarks/performance/bench_hot_paths.py

The benchmark does not assert timings; use ``benchmarks/performance/README.md``
for the recorded before/after development measurements.
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evidencescene.surface.object_surface_fusion import (
    RevisableSurfelMap,
    observation_voxel_centroids,
)
from evidencescene.teleop.task_features import (
    compare_circle_consensus_3d,
    fit_circular_aperture_3d,
)


def _timed_ms(function, *, repeat: int, warmup: int = 3) -> dict[str, float]:
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


def _snapshot_benchmark() -> dict:
    grid = np.indices((40, 40, 40)).reshape(3, -1).T[:60_000]
    points = grid.astype(np.float64) * 0.004
    surfel_map = RevisableSurfelMap(
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
        repeat=20,
    )
    return {"case": "snapshot_60k_voxels_budget_1500", **result}


def _voxel_centroid_benchmark() -> dict:
    rng = np.random.default_rng(5)
    points = rng.integers(0, 40, size=(60_000, 3)).astype(np.float64) * 0.004
    points += rng.normal(0.0, 1e-6, points.shape)
    result = _timed_ms(
        lambda: observation_voxel_centroids(points, 0.004),
        repeat=20,
    )
    return {"case": "voxel_centroids_60k_points", **result}


def _update_benchmark() -> dict:
    grid = np.indices((40, 40, 40)).reshape(3, -1).T[:60_000]
    points = grid.astype(np.float64) * 0.004
    samples = []
    for _ in range(8):
        surfel_map = RevisableSurfelMap(
            voxel_size_m=0.004,
            maximum_voxels=100_000,
            maximum_live_bytes=512 * 1024 * 1024,
        )
        start = time.perf_counter()
        surfel_map.update({"chest": ("chest", points)})
        samples.append((time.perf_counter() - start) * 1000.0)
        surfel_map.close()
    ordered = sorted(samples)
    return {
        "case": "surface_update_60k_points",
        "median_ms": round(statistics.median(samples), 4),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4),
        "min_ms": round(min(samples), 4),
        "repeat": len(samples),
    }


def _consensus_benchmark() -> dict:
    rng = np.random.default_rng(3)
    angles = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    rim = np.column_stack([0.04 * np.cos(angles), 0.04 * np.sin(angles)])
    rim += rng.normal(0.0, 0.0005, rim.shape)
    result = _timed_ms(
        lambda: compare_circle_consensus_3d(
            rim,
            fitted_center_2d=np.zeros(2),
            fitted_radius_m=0.04,
        ),
        repeat=50,
    )
    return {"case": "circle_consensus_400_points_256_hypotheses", **result}


def _aperture_benchmark() -> dict:
    rng = np.random.default_rng(3)
    angles = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    rim = np.column_stack(
        [
            0.04 * np.cos(angles),
            0.04 * np.sin(angles),
            np.zeros_like(angles),
        ]
    )
    rim += rng.normal(0.0, 0.0005, rim.shape)
    result = _timed_ms(
        lambda: fit_circular_aperture_3d(rim),
        repeat=30,
    )
    return {"case": "circular_aperture_fit_400_points", **result}


def main() -> None:
    payload = {
        "scope": "synthetic microbenchmark; not recorded-session endpoint latency",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "results": [
            _snapshot_benchmark(),
            _update_benchmark(),
            _voxel_centroid_benchmark(),
            _consensus_benchmark(),
            _aperture_benchmark(),
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
