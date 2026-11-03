"""Same-thread CPU preparation overlap inside SAM2's GPU execution window.

Preserves existing perception owners (OriginalPersistentWorldRegistry,
ObjectCentricObservationSession, SAM2ObservationSession,
SAM2VideoMemoryTracker, PinnedBundleTrackRuntime, CausalObjectSurface),
exact state transition order, fail-closed validation, and wire authority.
No background thread pool or second estimator is introduced.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
import hashlib
import json
import math
import time
from typing import Any

import cv2
import numpy as np


_ACTIVE_REGISTRY_CALL: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ticos_sam2_gpu_window_active_registry",
    default=False,
)
_PENDING_PREPARATION: contextvars.ContextVar["PendingObservationCpuPreparation | None"] = (
    contextvars.ContextVar(
        "ticos_sam2_gpu_window_pending_preparation",
        default=None,
    )
)


@dataclass
class PendingObservationCpuPreparation:
    """Validated same-observation CPU work scheduled for SAM2's GPU window."""

    observation: Any
    registered: Any
    color: np.ndarray
    selected_mask: np.ndarray
    color_info: Any
    depth_info: Any
    registration: Any
    stream_key: tuple[object, ...]
    color_timestamp_ns: int
    depth_timestamp_ns: int
    calibration: dict[str, Any]
    object_id: str
    sensor_calibration: tuple[ Any, ...]
    frame: Any
    tracker_id: int
    state_obj_id: int
    materialized: bool = False
    executed_in_gpu_window: bool = False
    gpu_launch_to_yield_ms: float | None = None
    gpu_window_cpu_prepare_ms: float | None = None
    post_cpu_gpu_sync_wait_ms: float | None = None
    evidence_hash_tail_ms: float | None = None
    rle_reused: bool = False
    native_buffers_reused: bool = False
    evidence_hash_color_prefix: Any = field(default=None, repr=False)
    evidence_hash_prefix: Any = field(default=None, repr=False)
    color_bgr: np.ndarray | None = field(default=None, repr=False)
    depth_m: np.ndarray | None = field(default=None, repr=False)
    visual_mask_rle_cache: tuple[np.ndarray, dict[str, list[int]], str] | None = field(
        default=None, repr=False
    )

    def materialize(self, *, in_gpu_window: bool) -> None:
        """Execute rectification, digest prefix, and native buffer conversion once."""
        if self.materialized:
            return
        started_ns = time.perf_counter_ns()
        from ticos_replay.bundletrack_live_input import (
            PreparedFeatureSourceBinding,
            _immutable,
            _update_array_digest,
        )
        from ticos_replay.registered_rgbd import rectify_registered_rgbd
        from ticos_vision.perception.image_mask import encode_mask_rle
        from ticos_vision.perception.measured_task_features import (
            mask_evidence_sha256,
            registered_feature_stream_key,
        )
        from ticos_vision.runtime.observation_computation_cache import (
            observation_digest,
        )

        rgb, depth, mask, intrinsics, _diagnostics = rectify_registered_rgbd(
            color=self.color,
            color_mask=self.selected_mask,
            color_info=self.color_info,
            registered=self.registered,
        )
        metadata = dict(
            stream_key=self.stream_key,
            color_timestamp_ns=self.color_timestamp_ns,
            depth_timestamp_ns=self.depth_timestamp_ns,
            calibration=self.calibration,
        )
        digest = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        )
        for array in (self.color, self.selected_mask):
            _update_array_digest(digest, np.ascontiguousarray(array))
        source_evidence_hash = digest.hexdigest()

        feature_key = registered_feature_stream_key(
            self.observation, object_id=self.object_id
        )
        if np.any(self.selected_mask):
            selected_rle = encode_mask_rle(self.selected_mask)
        else:
            selected_rle = {
                "size": [
                    int(self.selected_mask.shape[0]),
                    int(self.selected_mask.shape[1]),
                ],
                "counts": [],
            }
        feature_binding = (
            PreparedFeatureSourceBinding(
                observation_digest(self.observation),
                tuple(feature_key),
                mask_evidence_sha256(selected_rle),
                hashlib.sha256(mask.tobytes()).hexdigest(),
                self.stream_key,
                self.color_timestamp_ns,
                self.depth_timestamp_ns,
            )
            if feature_key is not None
            else None
        )

        for name, value in (
            ("color_rgb", rgb),
            ("depth_mm", depth),
            ("mask", mask),
            ("intrinsics", intrinsics),
        ):
            if (
                isinstance(value, np.ndarray)
                and value.flags.owndata
                and value.flags.c_contiguous
            ):
                value.setflags(write=False)
                frozen = value
            else:
                frozen = _immutable(value)
            object.__setattr__(self.frame, name, frozen)

        object.__setattr__(self.frame, "source_evidence_hash", source_evidence_hash)
        object.__setattr__(self.frame, "feature_source_binding", feature_binding)

        color_prefix = hashlib.sha256(
            json.dumps(
                dict(
                    source=source_evidence_hash,
                    stream_key=self.stream_key,
                    color_timestamp_ns=self.color_timestamp_ns,
                    depth_timestamp_ns=self.depth_timestamp_ns,
                ),
                sort_keys=True,
            ).encode()
        )
        color_arr = self.frame.color_rgb
        color_prefix.update(
            json.dumps((color_arr.dtype.str, color_arr.shape)).encode()
        )
        _update_array_digest(color_prefix, color_arr)
        self.evidence_hash_color_prefix = color_prefix

        prefix = color_prefix.copy()
        depth_arr = self.frame.depth_mm
        prefix.update(
            json.dumps((depth_arr.dtype.str, depth_arr.shape)).encode()
        )
        _update_array_digest(prefix, depth_arr)
        self.evidence_hash_prefix = prefix

        self.color_bgr = cv2.cvtColor(self.frame.color_rgb, cv2.COLOR_RGB2BGR)
        depth_m = np.ascontiguousarray(self.frame.depth_mm, dtype=np.float32)
        np.divide(depth_m, np.float32(1000.0), out=depth_m)
        self.depth_m = depth_m

        self.materialized = True
        self.executed_in_gpu_window = bool(in_gpu_window)
        self.gpu_window_cpu_prepare_ms = round(
            (time.perf_counter_ns() - started_ns) / 1e6, 3
        )


def validate_and_create_pending_preparation(
    observation: Any,
    *,
    registered: Any,
    object_id: str,
    color_rgb: np.ndarray | None,
    max_sync_skew_ms: float,
    active_session: Any,
) -> PendingObservationCpuPreparation | None:
    """Validate all input/session invariants before deferring CPU rectification.

    Raises the exact upstream ValueError/RuntimeError if any input invariant
    is violated, before touching SAM2 state. Returns None when the frame is
    not eligible for distorted warm-frame GPU-window overlap.
    """
    from ticos_replay.bundletrack_live_input import (
        RegisteredBundleTrackInput,
        _decode_color_rgb,
        _registration_transform_matrix,
    )
    from ticos_replay.sam2_observation_session import (
        SAM2ObservationSession,
        _same_tracker_new_sensor_epoch,
    )

    masks_session = getattr(active_session, "masks", None)
    tracker = getattr(masks_session, "tracker", None)
    if (
        not isinstance(masks_session, SAM2ObservationSession)
        or tracker is None
        or getattr(masks_session, "_closed", True)
        or getattr(masks_session, "_failed", True)
        or getattr(masks_session, "frame_count", 0) <= 0
        or getattr(tracker, "_state", None) is None
    ):
        return None

    stream = observation.perception_stream
    color_info = observation.color_camera_info
    depth_info = observation.depth_camera_info
    registration = observation.rgbd_registration
    if (
        stream is None
        or not observation.camera_id
        or not isinstance(object_id, str)
        or not object_id.strip()
    ):
        raise ValueError("explicit stream, camera and selected object identity required")
    if (
        registration.status != "validated"
        or registration.color_from_depth is None
        or color_info is None
        or depth_info is None
        or not registered.ready
    ):
        raise ValueError("current validated RGB-D registration required")
    color_ns, depth_ns = (
        observation.color_timestamp_ns,
        observation.depth_timestamp_ns,
    )
    try:
        maximum_skew_ns = int(float(max_sync_skew_ms) * 1_000_000.0)
    except (TypeError, ValueError, OverflowError):
        maximum_skew_ns = 0
    if (
        not math.isfinite(float(max_sync_skew_ms))
        or float(max_sync_skew_ms) <= 0.0
        or maximum_skew_ns <= 0
    ):
        raise ValueError("max_sync_skew_ms must be finite and positive")
    if (
        type(color_ns) is not int
        or type(depth_ns) is not int
        or min(color_ns, depth_ns) <= 0
        or abs(color_ns - depth_ns) > maximum_skew_ns
    ):
        raise ValueError(
            f"tracking requires positive sensor timestamps within {float(max_sync_skew_ms):g} ms"
        )
    if (
        registered.color_timestamp_ns != color_ns
        or registered.depth_timestamp_ns != depth_ns
        or registered.color_frame_id != color_info.frame_id
        or registered.depth_frame_id != depth_info.frame_id
    ):
        raise ValueError("registered frame identity does not match observation")
    color = (
        _decode_color_rgb(observation.color_image_base64)
        if color_rgb is None
        else np.asarray(color_rgb)
    )
    shape = (observation.color_height, observation.color_width)
    selected_mask = np.zeros(shape, dtype=bool)
    if (
        color.shape[:2] != shape
        or shape != (color_info.height, color_info.width)
        or registered.color_image_size != (shape[1], shape[0])
    ):
        raise ValueError("current dense mask and color camera grid must match")

    distortion = tuple(float(value) for value in (getattr(color_info, "d", ()) or ()))
    if not any(abs(value) > 1e-12 for value in distortion):
        return None

    calibration = dict(
        registration=registration.model_dump(mode="json"),
        color_camera_info=color_info.model_dump(mode="json"),
        depth_camera_info=depth_info.model_dump(mode="json"),
    )
    calibration_hash = hashlib.sha256(
        json.dumps(calibration, sort_keys=True).encode()
    ).hexdigest()
    key = (
        stream.stream_id,
        stream.epoch,
        observation.camera_id,
        observation.camera_topic,
        color_info.frame_id,
        depth_info.frame_id,
        object_id,
        calibration_hash,
    )

    new_epoch = masks_session._key is not None and _same_tracker_new_sensor_epoch(
        masks_session._key, key
    )
    if masks_session._key is not None and key != masks_session._key and not new_epoch:
        return None
    if (
        masks_session._color_ns is not None
        and (
            color_ns <= masks_session._color_ns
            or depth_ns <= masks_session._depth_ns
        )
        and not new_epoch
    ):
        return None
    if (
        masks_session.maximum_frames is not None
        and masks_session.frame_count >= masks_session.maximum_frames
    ):
        return None

    sensor_calibration = (
        int(color_info.width),
        int(color_info.height),
        float(color_info.fx),
        float(color_info.fy),
        float(color_info.cx),
        float(color_info.cy),
        distortion,
    )
    placeholder_rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    placeholder_depth = np.zeros((1, 1), dtype=np.uint16)
    placeholder_mask = np.zeros((1, 1), dtype=bool)
    placeholder_k = np.eye(3, dtype=np.float64)
    frame = RegisteredBundleTrackInput(
        key,
        color_ns,
        depth_ns,
        placeholder_rgb,
        placeholder_depth,
        placeholder_mask,
        placeholder_k,
        "",
        _take_ownership=True,
        sam_color_rgb=color,
        sam_anchor_mask=None,
        sensor_calibration=sensor_calibration,
        color_frame_id=color_info.frame_id,
        depth_frame_id=depth_info.frame_id,
        color_from_depth=_registration_transform_matrix(
            registration.color_from_depth
        ),
        feature_source_binding=None,
    )
    return PendingObservationCpuPreparation(
        observation=observation,
        registered=registered,
        color=color,
        selected_mask=selected_mask,
        color_info=color_info,
        depth_info=depth_info,
        registration=registration,
        stream_key=key,
        color_timestamp_ns=color_ns,
        depth_timestamp_ns=depth_ns,
        calibration=calibration,
        object_id=object_id,
        sensor_calibration=sensor_calibration,
        frame=frame,
        tracker_id=id(tracker),
        state_obj_id=id(getattr(tracker, "_state", None)),
    )


def complete_evidence_hash_from_prefix(
    current: Any,
    pending: PendingObservationCpuPreparation | None,
) -> bool:
    """Finish RegisteredBundleTrackInput.evidence_hash from the GPU-window prefix."""
    if (
        pending is None
        or not pending.materialized
        or pending.evidence_hash_prefix is None
        or pending.evidence_hash_color_prefix is None
        or current.stream_key != pending.stream_key
        or current.color_timestamp_ns != pending.color_timestamp_ns
        or current.depth_timestamp_ns != pending.depth_timestamp_ns
        or current.source_evidence_hash != pending.frame.source_evidence_hash
        or not np.shares_memory(current.color_rgb, pending.frame.color_rgb)
    ):
        return False
    from ticos_replay.bundletrack_live_input import _update_array_digest

    started_ns = time.perf_counter_ns()
    if np.shares_memory(current.depth_mm, pending.frame.depth_mm):
        digest = pending.evidence_hash_prefix.copy()
        tail_names: tuple[str, ...] = ("mask", "intrinsics")
    else:
        digest = pending.evidence_hash_color_prefix.copy()
        tail_names = ("depth_mm", "mask", "intrinsics")
    for name in tail_names:
        array = getattr(current, name)
        digest.update(json.dumps((array.dtype.str, array.shape)).encode())
        _update_array_digest(digest, array)
    with current._evidence_hash_lock:
        object.__setattr__(current, "_evidence_hash", digest.hexdigest())
    pending.evidence_hash_tail_ms = round(
        (time.perf_counter_ns() - started_ns) / 1e6, 3
    )
    return True


def consume_prepared_native_buffers(
    frame: Any,
    pending: PendingObservationCpuPreparation | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return one-shot precomputed (color_bgr, depth_m) when frame evidence matches."""
    if (
        pending is None
        or not pending.materialized
        or pending.color_bgr is None
        or pending.depth_m is None
        or frame.stream_key != pending.stream_key
        or frame.color_timestamp_ns != pending.color_timestamp_ns
        or frame.depth_timestamp_ns != pending.depth_timestamp_ns
        or not np.shares_memory(frame.color_rgb, pending.frame.color_rgb)
    ):
        return None
    color_bgr = pending.color_bgr
    if np.shares_memory(frame.depth_mm, pending.frame.depth_mm):
        depth_m = pending.depth_m
    else:
        depth_m = np.ascontiguousarray(frame.depth_mm, dtype=np.float32)
        np.divide(depth_m, np.float32(1000.0), out=depth_m)
    pending.color_bgr = None
    pending.depth_m = None
    pending.native_buffers_reused = True
    return color_bgr, depth_m


def advance_registered_with_prepared_buffers(
    session: Any,
    frame: Any,
    pending: PendingObservationCpuPreparation | None,
    fallback_advance_registered: Any,
) -> Any:
    """Advance PinnedBundleTrackSession using one-shot GPU-window buffers when valid."""
    from ticos_replay.bundletrack_live_input import RegisteredBundleTrackInput
    from ticos_replay.sam2_observation_session import (
        _same_tracker_new_sensor_epoch,
    )

    if not isinstance(frame, RegisteredBundleTrackInput):
        return fallback_advance_registered(session, frame)
    buffers = consume_prepared_native_buffers(frame, pending)
    if buffers is None:
        return fallback_advance_registered(session, frame)
    if session._registered_stream_key is None and session.frame_count:
        raise ValueError("cannot mix unbound and identity-bound tracker input")
    if (
        session._registered_stream_key is not None
        and frame.stream_key != session._registered_stream_key
    ):
        if not _same_tracker_new_sensor_epoch(
            session._registered_stream_key, frame.stream_key
        ):
            raise ValueError(
                "camera, object, calibration or source epoch changed"
            )
        session.last_timestamp_ns = None
    elif (
        session._registered_color_timestamp_ns is not None
        and frame.color_timestamp_ns <= session._registered_color_timestamp_ns
    ):
        raise ValueError(
            "registered tracker cannot reuse or reorder color evidence"
        )
    color_bgr, depth_m = buffers
    mask = np.array(frame.mask, dtype=np.uint8, copy=True, order="C")
    if session._mask_kernel is not None:
        cv2.erode(mask, session._mask_kernel, dst=mask)
    mask *= np.uint8(255)
    result = session._advance_arrays(
        color_bgr=color_bgr,
        depth_m=depth_m,
        depth_owned=frame.depth_mm.dtype == np.uint16,
        private_input_buffers=True,
        mask_u8=mask,
        intrinsics=frame.intrinsics,
        timestamp_ns=frame.depth_timestamp_ns,
    )
    session._registered_stream_key = frame.stream_key
    session._registered_color_timestamp_ns = frame.color_timestamp_ns
    return result
