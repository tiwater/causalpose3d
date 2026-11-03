"""Smoke and contract verification tests for standalone causalpose3d package."""

from __future__ import annotations

import numpy as np

import causalpose3d
from causalpose3d.observability.point2pose_adapter import (
    adapt_shadow_pose,
    adapt_shadow_poses,
)
from causalpose3d.runtime.request_evidence_deduplication import (
    RequestDeduplicationStats,
    rle_mask_box_and_centroid,
)
from causalpose3d.surface.causal_object_surface import CausalObjectSurface
from causalpose3d.surface.object_surface_fusion import RevisableSurfelMap
from causalpose3d.tracking.object_centric_observation_session import (
    ObjectCentricObservationSession,
)
from causalpose3d.tracking.pinned_bundletrack_runner import PinnedBundleTrackSession


def test_package_exports_and_version() -> None:
    assert causalpose3d.__version__ == "1.0.0"
    assert CausalObjectSurface is not None
    assert RevisableSurfelMap is not None
    assert ObjectCentricObservationSession is not None
    assert PinnedBundleTrackSession is not None
    assert RequestDeduplicationStats is not None


def test_point2pose_adapter_fail_closed_and_valid_composition() -> None:
    eye = np.eye(4, dtype=np.float64)
    blocked = adapt_shadow_pose(
        camera_current_from_camera_initial=eye,
        camera_initial_from_object=eye,
        lost=True,
    )
    assert blocked["status"] == "not_observable"
    assert blocked["camera_from_object"] is None
    assert blocked["wire_authorized"] is False

    candidate = adapt_shadow_pose(
        camera_current_from_camera_initial=eye,
        camera_initial_from_object=eye,
        lost=False,
    )
    assert candidate["status"] == "candidate"
    assert candidate["wire_authorized"] is False
    assert np.allclose(np.asarray(candidate["camera_from_object"]), eye)

    batch = adapt_shadow_poses(
        object_keys=["carton"],
        camera_current_from_camera_initial=[eye],
        camera_initial_from_object={"carton": eye},
        lost=[False],
    )
    assert len(batch) == 1
    assert batch[0]["status"] == "candidate"


def test_rle_run_interval_geometry() -> None:
    # 10x10 image with two runs on rows 2..3, cols 4..6 (start=24, len=3; start=34, len=3)
    width, height = 10, 10
    rle = {"size": [height, width], "counts": [24, 3, 34, 3]}

    geom = rle_mask_box_and_centroid(rle, width=width, height=height)
    assert geom is not None
    bbox, centroid = geom
    assert bbox == (4, 2, 6, 3)
    assert np.isclose(centroid[0], 5.0)
    assert np.isclose(centroid[1], 2.5)
