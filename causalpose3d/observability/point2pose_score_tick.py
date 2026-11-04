"""Adapt an existing SAM2 observation and Point2Pose result to a shadow tick.

The registered RGB-D/mask is owned by SAM2ObservationSession and the live
pose/surface by OriginalPersistentWorldRegistry / BundleTrack. This stateless
adapter neither runs an estimator nor publishes. Receipts and matching clocks
do not independently establish actual inference, identity or freshness.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from causalpose3d.primitives.bundletrack_live_input import RegisteredBundleTrackInput
from causalpose3d.observability.point2pose_adapter import adapt_shadow_pose
from causalpose3d.primitives.sam2_observation_session import _mask_evidence_sha256
from causalpose3d.primitives.image_mask import decode_mask_rle, encode_mask_rle

import numpy as np


def make_point2pose_shadow_score_tick(
    *,
    frame_index: int,
    registered_input: RegisteredBundleTrackInput,
    sam_visual_mask_evidence: Mapping[str, Any],
    point_result: Mapping[str, Any] | None,
    camera_initial_from_object: object,
    max_sync_skew_ms: float = 20.0,
) -> dict[str, Any]:
    """Serialize one *available* existing SAM2 observation for scoring only.

    A supplied Point2Pose result must name the current frame and registered
    input digest, contain one native entry, and explicitly agree with its loss
    status. A missing result produces a not-observable tick. This API cannot
    authenticate the caller, prove consumption/freshness or handle unavailable
    SAM observations; the scoreboard retains those independent blockers.
    """
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("frame_index must be a nonnegative integer")
    if isinstance(max_sync_skew_ms, bool):
        raise ValueError("max_sync_skew_ms must be finite and positive")
    try:
        skew_ms = float(max_sync_skew_ms)
        max_skew_ns = int(skew_ms * 1_000_000)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("max_sync_skew_ms must be finite and positive") from None
    if not math.isfinite(skew_ms) or skew_ms <= 0 or max_skew_ns <= 0:
        raise ValueError("max_sync_skew_ms must be finite and positive")
    if not isinstance(registered_input, RegisteredBundleTrackInput):
        raise ValueError("existing registered input required")
    frame = registered_input
    if (not isinstance(frame.mask, np.ndarray) or frame.mask.dtype != np.bool_
            or not isinstance(frame.depth_mm, np.ndarray) or frame.depth_mm.dtype != np.uint16
            or frame.mask.ndim != 2 or frame.depth_mm.shape != frame.mask.shape
            or not np.any(frame.mask & (frame.depth_mm > 0))):
        raise ValueError("registered mask-depth evidence is unavailable")
    key = frame.stream_key
    if (not isinstance(key, tuple) or len(key) != 8
            or any(not ((type(v) is str and bool(v)) or type(v) is int) for v in key)):
        raise ValueError("invalid registered stream key")
    if (any(type(t) is not int or not 0 < t <= 2**64 - 1
            for t in (frame.color_timestamp_ns, frame.depth_timestamp_ns))
            or abs(frame.color_timestamp_ns - frame.depth_timestamp_ns) > max_skew_ns):
        raise ValueError("invalid or unsynchronized registered clocks")
    if not isinstance(sam_visual_mask_evidence, Mapping):
        raise ValueError("same-tick SAM2 visual evidence required")
    evidence = sam_visual_mask_evidence
    if (evidence.get("source") != "original_sam2_video_memory"
            or evidence.get("stream_key") != list(key)
            or evidence.get("color_timestamp_ns") != str(frame.color_timestamp_ns)
            or evidence.get("depth_timestamp_ns") != str(frame.depth_timestamp_ns)
            or evidence.get("coordinate_frame") != "sensor_color_image"
            or not isinstance(evidence.get("mask_rle"), dict)):
        raise ValueError("SAM2 evidence identity or mask is invalid")
    sensor_rgb = frame.sam_color_rgb if frame.sam_color_rgb is not None else frame.color_rgb
    if not isinstance(sensor_rgb, np.ndarray) or sensor_rgb.ndim != 3 or sensor_rgb.shape[2] != 3:
        raise ValueError("registered sensor image is invalid")
    height, width = sensor_rgb.shape[:2]
    rle = evidence["mask_rle"]
    size, counts = rle.get("size"), rle.get("counts")
    if (not isinstance(size, list) or len(size) != 2
            or any(type(value) is not int for value in size)
            or size != [height, width] or not isinstance(counts, list)
            or any(type(value) is not int for value in counts)):
        raise ValueError("SAM2 evidence mask is invalid")
    decoded = decode_mask_rle(rle, width=width, height=height)
    if decoded is None or not np.any(decoded) or encode_mask_rle(decoded) != rle:
        raise ValueError("SAM2 evidence mask is invalid")
    # Without sensor rectification the existing SAM2 owner returns the same
    # mask grid. With rectification, the owner's sensor mask and registered
    # mask differ by design; no second warp is inferred here.
    if frame.sensor_calibration is None and not np.array_equal(decoded, frame.mask):
        raise ValueError("SAM2 sensor and registered masks differ")
    try:
        expected = _mask_evidence_sha256(
            key, frame.color_timestamp_ns, frame.depth_timestamp_ns, evidence["mask_rle"]
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("SAM2 evidence mask is invalid") from exc
    if evidence.get("mask_evidence_sha256") != expected:
        raise ValueError("SAM2 evidence digest mismatch")
    if point_result is None:
        adapted = {"status": "not_observable", "reason": "point2pose_not_evaluated",
                   "camera_from_object": None, "wire_authorized": False}
    else:
        if (not isinstance(point_result, Mapping)
                or type(point_result.get("index")) is not int
                or point_result["index"] != frame_index
                or point_result.get("input_evidence_hash") != frame.evidence_hash
                or type(point_result.get("native_entry_calls")) is not int
                or point_result["native_entry_calls"] != 1
                or point_result.get("wire_authorized") is not False):
            raise ValueError("Point2Pose result is not bound to this shadow input")
        lost = point_result.get("lost")
        if (type(lost) is not bool or point_result.get("status") != ("lost" if lost else "candidate")):
            raise ValueError("Point2Pose status and loss evidence disagree")
        adapted = adapt_shadow_pose(
            camera_current_from_camera_initial=point_result.get("pose"),
            camera_initial_from_object=camera_initial_from_object,
            lost=lost,
        )
    return {
        "frame_index": frame_index,
        "object_key": key[6],
        "stream_key": list(key),
        "color_timestamp_ns": frame.color_timestamp_ns,
        "depth_timestamp_ns": frame.depth_timestamp_ns,
        "input_evidence_hash": frame.evidence_hash,
        "source_mask_evidence_sha256": expected,
        "mask_available": True,
        **adapted,
    }
