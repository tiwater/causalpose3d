"""Bind causal multi-camera masks to camera-local BundleSDF providers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Iterable, Mapping

import numpy as np

from causalpose3d.primitives.geometry import (
    BundleSDFFrameInput,
    BundleSDFProvider,
    BundleSDFProviderResult,
    CausalMulticameraMaskRuntime,
    CausalRuntimeBatchResult,
    CausalRuntimeFrame,
)
from causalpose3d.primitives.content_hash import content_sha256


@dataclass(frozen=True)
class RegisteredRgbdTrackerFrame:
    camera_id: str
    frame_index: int
    timestamp_ns: int
    camera_frame_id: str
    source_sha256: str
    registration_bundle_sha256: str
    raw_color_rgb: np.ndarray
    registered_color_rgb: np.ndarray
    aligned_depth_m: np.ndarray
    intrinsics_camera: np.ndarray


@dataclass(frozen=True)
class CausalRgbdTrackerFrameResult:
    camera_id: str
    frame_index: int
    timestamp_ns: int
    status: str
    reason: str
    tracker_called: bool
    tracker_input: BundleSDFFrameInput | None
    tracker_result: BundleSDFProviderResult | None
    input_build_latency_ms: float
    tracker_call_latency_ms: float


@dataclass(frozen=True)
class CausalRgbdTrackerBatchResult:
    mask_runtime: CausalRuntimeBatchResult
    frames: tuple[CausalRgbdTrackerFrameResult, ...]
    batch_latency_ms: float
    world_publication_authorized: bool = False


class CausalRgbdTrackerRuntime:
    """Produce tracker calls only from ready, current P5bd masks.

    Every provider remains camera-local. This runtime intentionally performs
    no cross-camera pose fusion and exposes no world-pose output.
    """

    def __init__(
        self,
        *,
        mask_runtime: CausalMulticameraMaskRuntime,
        tracker_by_camera: Mapping[str, BundleSDFProvider],
        maximum_surface_points: int = 12000,
    ) -> None:
        if not isinstance(mask_runtime, CausalMulticameraMaskRuntime):
            raise ValueError("mask_runtime must be CausalMulticameraMaskRuntime")
        trackers = dict(tracker_by_camera)
        if set(trackers) != set(mask_runtime.camera_ids):
            raise ValueError("tracker providers must exactly match mask camera_ids")
        if not all(isinstance(value, BundleSDFProvider) for value in trackers.values()):
            raise ValueError("every tracker provider must be BundleSDFProvider")
        if type(maximum_surface_points) is not int or maximum_surface_points <= 0:
            raise ValueError("maximum_surface_points must be a positive integer")
        self.mask_runtime = mask_runtime
        self.tracker_by_camera = trackers
        self.maximum_surface_points = maximum_surface_points
        self._closed = False

    def process_batch(
        self,
        frames: Iterable[RegisteredRgbdTrackerFrame],
    ) -> CausalRgbdTrackerBatchResult:
        if self._closed:
            raise RuntimeError("causal RGB-D tracker runtime is closed")
        started = time.perf_counter()
        batch = list(frames)
        if not batch:
            raise ValueError("registered RGB-D tracker batch must not be empty")
        self._validate_mask_inputs(batch)
        mask_batch = self.mask_runtime.process_batch(
            CausalRuntimeFrame(
                camera_id=frame.camera_id,
                frame_index=frame.frame_index,
                decoded_rgb=frame.raw_color_rgb,
            )
            for frame in batch
        )
        if len(mask_batch.frames) != len(batch):
            raise RuntimeError("mask runtime result count changed")

        results: list[CausalRgbdTrackerFrameResult] = []
        for frame, mask_frame in zip(batch, mask_batch.frames, strict=True):
            foreground = mask_frame.foreground
            if (
                foreground.camera_id != frame.camera_id
                or foreground.frame_index != frame.frame_index
            ):
                results.append(
                    self._skipped(frame, "mask_identity_mismatch")
                )
                continue
            if not foreground.ready or foreground.mask is None:
                results.append(
                    self._skipped(
                        frame,
                        f"foreground_not_ready:{foreground.reason}",
                    )
                )
                continue
            if (
                foreground.refresh_age_frames is None
                or foreground.refresh_age_frames
                > self.mask_runtime.coordinator.maximum_refresh_age_frames
            ):
                results.append(self._skipped(frame, "foreground_mask_stale"))
                continue

            build_started = time.perf_counter()
            try:
                tracker_input = self._tracker_input(
                    frame,
                    mask=foreground.mask,
                    maximum_surface_points=self.maximum_surface_points,
                )
            except (TypeError, ValueError):
                results.append(
                    self._skipped(
                        frame,
                        "registered_rgbd_tracker_input_invalid",
                        input_build_latency_ms=(
                            time.perf_counter() - build_started
                        )
                        * 1000.0,
                    )
                )
                continue
            build_latency_ms = (time.perf_counter() - build_started) * 1000.0

            tracker_started = time.perf_counter()
            try:
                tracker_result = self.tracker_by_camera[frame.camera_id].track(
                    tracker_input
                )
            except Exception as error:  # noqa: BLE001 - isolate camera-local tracker failure.
                results.append(
                    CausalRgbdTrackerFrameResult(
                        camera_id=frame.camera_id,
                        frame_index=frame.frame_index,
                        timestamp_ns=frame.timestamp_ns,
                        status="not_observable",
                        reason=f"tracker_provider_exception:{type(error).__name__}",
                        tracker_called=True,
                        tracker_input=tracker_input,
                        tracker_result=None,
                        input_build_latency_ms=build_latency_ms,
                        tracker_call_latency_ms=(
                            time.perf_counter() - tracker_started
                        )
                        * 1000.0,
                    )
                )
                continue
            tracker_latency_ms = (
                time.perf_counter() - tracker_started
            ) * 1000.0
            if tracker_result.timestamp_ns != frame.timestamp_ns:
                results.append(
                    CausalRgbdTrackerFrameResult(
                        camera_id=frame.camera_id,
                        frame_index=frame.frame_index,
                        timestamp_ns=frame.timestamp_ns,
                        status="not_observable",
                        reason="tracker_timestamp_mismatch",
                        tracker_called=True,
                        tracker_input=tracker_input,
                        tracker_result=None,
                        input_build_latency_ms=build_latency_ms,
                        tracker_call_latency_ms=tracker_latency_ms,
                    )
                )
                continue
            results.append(
                CausalRgbdTrackerFrameResult(
                    camera_id=frame.camera_id,
                    frame_index=frame.frame_index,
                    timestamp_ns=frame.timestamp_ns,
                    status=tracker_result.status,
                    reason=tracker_result.reason,
                    tracker_called=True,
                    tracker_input=tracker_input,
                    tracker_result=tracker_result,
                    input_build_latency_ms=build_latency_ms,
                    tracker_call_latency_ms=tracker_latency_ms,
                )
            )

        return CausalRgbdTrackerBatchResult(
            mask_runtime=mask_batch,
            frames=tuple(results),
            batch_latency_ms=(time.perf_counter() - started) * 1000.0,
            world_publication_authorized=False,
        )

    def wait_for_refresh(self, *, timeout_s: float):
        return self.mask_runtime.wait_for_refresh(timeout_s=timeout_s)

    def close(self, *, wait: bool = True):
        if self._closed:
            return ()
        self._closed = True
        return self.mask_runtime.close(wait=wait)

    @staticmethod
    def _validate_mask_inputs(batch: list[RegisteredRgbdTrackerFrame]) -> None:
        camera_ids: list[str] = []
        for frame in batch:
            if not isinstance(frame, RegisteredRgbdTrackerFrame):
                raise ValueError("tracker batch entries must be RegisteredRgbdTrackerFrame")
            camera = str(frame.camera_id).strip()
            if not camera:
                raise ValueError("camera_id must not be empty")
            if type(frame.frame_index) is not int or frame.frame_index < 0:
                raise ValueError("frame_index must be a nonnegative integer")
            if type(frame.timestamp_ns) is not int or frame.timestamp_ns <= 0:
                raise ValueError("timestamp_ns must be a positive integer")
            rgb = np.asarray(frame.raw_color_rgb)
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("raw_color_rgb must be uint8 HxWx3")
            camera_ids.append(camera)
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("one tracker batch may contain one frame per camera")

    @staticmethod
    def _tracker_input(
        frame: RegisteredRgbdTrackerFrame,
        *,
        mask: np.ndarray,
        maximum_surface_points: int,
    ) -> BundleSDFFrameInput:
        camera = str(frame.camera_id).strip()
        camera_frame_id = str(frame.camera_frame_id).strip()
        if not camera_frame_id:
            raise ValueError("camera_frame_id is required")
        for name, value in (
            ("source_sha256", frame.source_sha256),
            ("registration_bundle_sha256", frame.registration_bundle_sha256),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be lowercase hexadecimal SHA-256")

        raw = np.asarray(frame.raw_color_rgb)
        registered = np.asarray(frame.registered_color_rgb)
        depth = np.asarray(frame.aligned_depth_m)
        intrinsics = np.asarray(frame.intrinsics_camera)
        foreground = np.asarray(mask)
        if raw.dtype != np.uint8 or registered.dtype != np.uint8:
            raise ValueError("color inputs must be uint8")
        if depth.dtype != np.float32:
            raise ValueError("aligned_depth_m must be float32")
        if intrinsics.dtype != np.float64:
            raise ValueError("intrinsics_camera must be float64")
        if foreground.dtype != np.bool_:
            raise ValueError("foreground mask must be bool")
        if (
            raw.ndim != 3
            or raw.shape[2] != 3
            or registered.shape != raw.shape
            or depth.shape != raw.shape[:2]
            or foreground.shape != depth.shape
            or intrinsics.shape != (3, 3)
        ):
            raise ValueError("registered color, depth, and mask grids must match exactly")
        if (
            not np.isfinite(depth).all()
            or np.any(depth < 0.0)
            or not np.isfinite(intrinsics).all()
            or intrinsics[0, 0] <= 0.0
            or intrinsics[1, 1] <= 0.0
            or not np.allclose(
                intrinsics[2],
                (0.0, 0.0, 1.0),
                atol=1e-12,
                rtol=0.0,
            )
        ):
            raise ValueError("metric depth or calibrated intrinsics are invalid")
        valid = foreground & (depth >= 0.1)
        ys, xs = np.nonzero(valid)
        if not len(xs):
            raise ValueError("foreground mask has no valid metric depth")
        z = depth[ys, xs].astype(np.float64)
        points = np.column_stack(
            (
                (xs.astype(np.float64) - intrinsics[0, 2])
                * z
                / intrinsics[0, 0],
                (ys.astype(np.float64) - intrinsics[1, 2])
                * z
                / intrinsics[1, 1],
                z,
            )
        )
        if len(points) > maximum_surface_points:
            selected = (
                np.arange(maximum_surface_points, dtype=np.int64)
                * len(points)
                // maximum_surface_points
            )
            points = points[selected]
        array_hashes = {
            "raw_color_rgb": _array_sha256(raw),
            "registered_color_rgb": _array_sha256(registered),
            "aligned_depth_m": _array_sha256(depth),
            "mask": _array_sha256(foreground),
            "intrinsics_camera": _array_sha256(intrinsics),
            "points_camera_m": _array_sha256(points),
        }
        observation_hash = content_sha256(
            {
                "schema": "ticos.causal-registered-rgbd-tracker-input.v1",
                "camera_id": camera,
                "frame_index": frame.frame_index,
                "timestamp_ns": frame.timestamp_ns,
                "camera_frame_id": camera_frame_id,
                "source_sha256": frame.source_sha256,
                "registration_bundle_sha256": frame.registration_bundle_sha256,
                "arrays": array_hashes,
            }
        )
        return BundleSDFFrameInput(
            observation_hash=observation_hash,
            source_sha256=frame.source_sha256,
            g1_registered_rgbd_authorization="registered_rgbd",
            g1_registered_rgbd_bundle_hash=frame.registration_bundle_sha256,
            raw_color_rgb=raw,
            registered_color_rgb=registered,
            aligned_depth_m=depth,
            mask=foreground,
            intrinsics_camera=intrinsics,
            raw_color_rgb_sha256=array_hashes["raw_color_rgb"],
            registered_color_rgb_sha256=array_hashes["registered_color_rgb"],
            aligned_depth_m_sha256=array_hashes["aligned_depth_m"],
            mask_sha256=array_hashes["mask"],
            intrinsics_camera_sha256=array_hashes["intrinsics_camera"],
            timestamp_ns=frame.timestamp_ns,
            camera_frame_id=camera_frame_id,
            points_camera_m=points,
        )

    @staticmethod
    def _skipped(
        frame: RegisteredRgbdTrackerFrame,
        reason: str,
        *,
        input_build_latency_ms: float = 0.0,
    ) -> CausalRgbdTrackerFrameResult:
        return CausalRgbdTrackerFrameResult(
            camera_id=str(frame.camera_id),
            frame_index=int(frame.frame_index),
            timestamp_ns=int(frame.timestamp_ns),
            status="not_observable",
            reason=reason,
            tracker_called=False,
            tracker_input=None,
            tracker_result=None,
            input_build_latency_ms=input_build_latency_ms,
            tracker_call_latency_ms=0.0,
        )


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
