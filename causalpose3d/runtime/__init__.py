"""Zero-loss GPU/CPU overlap, O(runs) integer RLE mask arithmetic, and vectorized AprilTag RANSAC."""

from __future__ import annotations

__all__ = [
    "coplanar_rgbd_tag_observations",
    "request_evidence_deduplication",
    "sam2_cpu_preparation",
    "sam2_gpu_window_overlap",
]
