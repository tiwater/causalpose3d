"""Run the pinned BundleTrack build without resizing registered RGB-D input."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import pdb
import subprocess
import sys
import time
from contextlib import nullcontext
from typing import Any

import cv2
import numpy as np


SEQUENCE_SCHEMA = "ticos.bundlesdf-registered-sequence.v3"
RUNNER_SCHEMA = "ticos.pinned-bundletrack-run.v2"
UPSTREAM_REVISION = "ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1"
LOFTR_CHECKPOINT_SHA256 = (
    "21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec"
)
RUNTIME_OUTPUT_MODES = ("full_audit", "online_pose")
LATENCY_WARMUP_FRAME_COUNT = 2


def _yaml_backend() -> Any:
    """Return the original YAML API without adding a runtime-only dependency.

    The research checkout historically used ``ruamel.yaml``.  The production
    scene-runtime lock intentionally carries PyYAML instead, so the pinned
    runtime can still write its generated configs without importing the
    replay/training dependency set.  Tests and full research environments may
    continue to provide ruamel unchanged.
    """

    try:
        from ruamel.yaml import YAML
    except ModuleNotFoundError:
        import yaml

        class _PyYamlCompat:
            def load(self, stream: Any) -> Any:
                return yaml.safe_load(stream)

            def dump(self, value: Any, stream: Any) -> None:
                yaml.safe_dump(value, stream, sort_keys=False)

        return _PyYamlCompat()
    return YAML()


class NonInteractivePdbGuard:
    """Turn upstream pdb.set_trace calls into counted, non-blocking events."""

    def __init__(self) -> None:
        self.break_count = 0
        self._original: Any | None = None

    def _bypass(self, *_args: object, **_kwargs: object) -> None:
        self.break_count += 1

    def install(self) -> None:
        if self._original is not None:
            raise RuntimeError("pdb guard is already installed")
        if isinstance(getattr(pdb.set_trace, "__self__", None), NonInteractivePdbGuard):
            raise RuntimeError("another BundleTrack runtime owns this process debugger hook")
        self._original = pdb.set_trace
        pdb.set_trace = self._bypass

    def restore(self) -> None:
        if self._original is None:
            return
        pdb.set_trace = self._original
        self._original = None


def _native_inference_context() -> Any:
    """Use inference-mode bookkeeping around the unchanged native tracker.

    BundleTrack remains the sole estimator; this only removes autograd version
    counter overhead from its existing LoFTR forward.  Keep a no-op fallback
    so adapter/unit-test environments without torch retain the same behavior.
    """
    try:
        import torch
    except ModuleNotFoundError:
        return nullcontext()
    return torch.inference_mode()


def _preload_torchvision_for_active_torch() -> None:
    """Bind torchvision to the process' Torch before adding native roots.

    The scene runtime and the pinned BundleTrack environment intentionally
    share native tracking code but may carry different Python dependency
    roots.  If the scene runtime has already imported its Torch, allowing the
    later BundleTrack root to resolve a second torchvision wheel can make its
    fake ``torchvision::nms`` registration run against the wrong Torch ABI.
    Importing the already-selected pair first keeps the native adapter and
    estimator unchanged while making the process-level dependency boundary
    deterministic.  Environments without Torch/torchvision retain the
    existing import behavior for adapter-only tests.
    """
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return
    try:
        import torchvision  # noqa: F401
    except ModuleNotFoundError:
        return


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sequence_for_tracking(sequence_root: Path) -> dict[str, Any]:
    manifest_path = sequence_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SEQUENCE_SCHEMA:
        raise ValueError(f"sequence schema must be {SEQUENCE_SCHEMA}")
    if manifest.get("color_frame_reuse_forbidden") is not True:
        raise ValueError("sequence must forbid stale color reuse")
    sampling_contract = (
        manifest.get("sample_every_basis"),
        manifest.get("rgbd_pairing_algorithm"),
    )
    capture_order_contract = (
        "optimal_monotonic_unique_rgbd_pairs",
        "maximum_cardinality_minimum_total_capture_skew_within_20ms_v1",
    )
    log_order_contract = (
        "unique_rgbd_pairs_after_mcap_log_time_ordering",
        "maximum_cardinality_minimum_total_capture_skew_then_log_chronology_v2",
    )
    if sampling_contract not in (capture_order_contract, log_order_contract):
        raise ValueError("sequence must use the authorized RGB-D pairing algorithm")
    if manifest.get("pairing_clock") != "ros_image_header_capture_time":
        raise ValueError("sequence must pair RGB-D on image capture timestamps")
    if sampling_contract == log_order_contract:
        if manifest.get("sequence_clock") != "mcap_log_time_pair_midpoint":
            raise ValueError("sequence must declare MCAP log-time chronology")
        if manifest.get("sequence_log_time_regression_count") != 0:
            raise ValueError("sequence declares a log-time regression")
    maximum_pair_skew_ms = manifest.get("maximum_pair_skew_ms")
    if (
        isinstance(maximum_pair_skew_ms, bool)
        or not isinstance(maximum_pair_skew_ms, (int, float))
        or float(maximum_pair_skew_ms) > 20.0
    ):
        raise ValueError("sequence RGB-D capture skew gate must be at most 20 ms")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("sequence frames must be a non-empty list")
    if manifest.get("frame_count") != len(frames):
        raise ValueError("sequence frame_count does not match frames")
    timestamps: list[int] = []
    sequence_timestamps: list[int] = []
    for index, row in enumerate(frames):
        if not isinstance(row, dict) or row.get("output_index") != index:
            raise ValueError("sequence output indices must be contiguous")
        timestamp = row.get("color_timestamp_ns")
        if type(timestamp) is not int or timestamp <= 0:
            raise ValueError("sequence color timestamps must be positive integers")
        timestamps.append(timestamp)
        if sampling_contract == log_order_contract:
            color_log = row.get("color_log_timestamp_ns")
            depth_log = row.get("depth_log_timestamp_ns")
            if type(color_log) is not int or type(depth_log) is not int:
                raise ValueError("sequence log timestamps must be integers")
            sequence_timestamps.append((color_log + depth_log) // 2)
        for key in ("rgb_path", "depth_path", "mask_path"):
            path = sequence_root / str(row.get(key, ""))
            if not path.is_file():
                raise ValueError(f"sequence frame {index} is missing {key}")
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("sequence reuses stale RGB evidence")
    chronology = (
        sequence_timestamps
        if sampling_contract == log_order_contract
        else timestamps
    )
    if any(
        current <= previous
        for previous, current in zip(chronology, chronology[1:])
    ):
        raise ValueError("sequence chronology must be strictly increasing")
    intrinsics = np.asarray(
        np.loadtxt(sequence_root / "cam_K.txt"),
        dtype=np.float64,
    ).reshape(3, 3)
    if (
        not np.isfinite(intrinsics).all()
        or intrinsics[0, 0] <= 0.0
        or intrinsics[1, 1] <= 0.0
        or not np.allclose(intrinsics[2], (0.0, 0.0, 1.0))
    ):
        raise ValueError("sequence intrinsics are invalid")
    return manifest


def _verify_runtime(
    *,
    source_root: Path,
    build_root: Path,
    dependency_prefix: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    revision = subprocess.run(
        ("git", "-C", str(source_root), "rev-parse", "HEAD"),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("BundleSDF source revision is not pinned")
    checkpoint_hash = _sha256_path(checkpoint_path)
    if checkpoint_hash != LOFTR_CHECKPOINT_SHA256:
        raise ValueError("LoFTR checkpoint hash mismatch")
    binaries = (
        build_root / "my_cpp.cpython-312-x86_64-linux-gnu.so",
        build_root / "libBundleTrack.so",
        build_root / "libMY_CUDA_LIB.so",
    )
    if any(not path.is_file() for path in binaries):
        raise ValueError("pinned BundleTrack build is incomplete")
    dependency_manifest = dependency_prefix / "bundlesdf-dependencies-v1.json"
    if not dependency_manifest.is_file():
        raise ValueError("versioned dependency prefix is missing")
    return {
        "source_revision": revision,
        "source_diff_sha256": hashlib.sha256(
            subprocess.run(
                ("git", "-C", str(source_root), "diff", "--binary"),
                check=True,
                stdout=subprocess.PIPE,
            ).stdout
        ).hexdigest(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "dependency_manifest_path": str(dependency_manifest),
        "dependency_manifest_sha256": _sha256_path(dependency_manifest),
        "build_binaries": {
            path.name: _sha256_path(path)
            for path in binaries
        },
    }


def _tracking_configs(
    *,
    source_root: Path,
    output_root: Path,
    maximum_ba_frames: int,
    feature_resize: int,
    maximum_neighbor_translation_m: float,
    runtime_output_mode: str,
) -> tuple[Any, Any]:
    yaml = _yaml_backend()
    with (source_root / "BundleTrack" / "config_ho3d.yml").open(
        "r", encoding="utf-8"
    ) as stream:
        tracking = yaml.load(stream)
    tracking["SPDLOG"] = 0 if runtime_output_mode == "online_pose" else 1
    tracking["debug_dir"] = f"{output_root}/"
    tracking["depth_processing"]["percentile"] = 95
    tracking["erode_mask"] = 3
    tracking["bundle"]["max_BA_frames"] = maximum_ba_frames
    if runtime_output_mode == "online_pose":
        # Keep the original BundleTrack owner and its existing sliding-window
        # policy, but retain only three non-keyframes in the realtime route.
        # Full-audit/reconstruction runs keep the upstream window unchanged.
        tracking["bundle"]["window_size"] = 3
        # Keep the existing keyframe selector as the source of pose evidence,
        # but avoid allowing realtime fallback matching to grow a keyframe
        # search set faster than the live route can service. This is a
        # scheduling/configuration bound, not a second tracker or eviction
        # policy; full-audit retains the source configuration.
        tracking["keyframe"]["min_interval"] = 4
    tracking["bundle"]["max_optimized_feature_loss"] = 0.03
    # HO3D's 2 cm neighbor gate is for a slow in-hand object. On the pour, the
    # carton face moves several centimeters between frames, every
    # correspondence is rejected, and the next frame can only see the stale
    # keyframe. 10 cm is the upstream BEHAVE handheld gate and matches the
    # contract's maximum camera-local jump, so a correspondence cannot outrun
    # a pose the wire is already forbidden to publish.
    tracking["feature_corres"]["max_dist_neighbor"] = 0.10
    tracking["feature_corres"]["max_normal_neighbor"] = 30
    tracking["feature_corres"]["max_dist_no_neighbor"] = 0.01
    tracking["feature_corres"]["max_normal_no_neighbor"] = 20
    tracking["feature_corres"]["map_points"] = True
    tracking["feature_corres"]["resize"] = feature_resize
    tracking["feature_corres"]["rematch_after_nerf"] = False
    # Realtime pose uses the existing BundleTrack keyframe selector with the
    # upstream angular separation.  The interval bound above already limits
    # reference-bank growth; widening this angular gate to 10 degrees caused
    # the recorded right-arm stream to retain only one keyframe and reject
    # 115/119 warm observations.  Full audit/reconstruction keeps the source
    # configuration unchanged as well.
    tracking["keyframe"]["min_rot"] = 5
    tracking["ransac"]["inlier_dist"] = 0.01
    tracking["ransac"]["inlier_normal_angle"] = 20
    tracking["ransac"]["max_trans_neighbor"] = (
        maximum_neighbor_translation_m
    )
    tracking["ransac"]["max_rot_deg_neighbor"] = 30
    tracking["ransac"]["max_trans_no_neighbor"] = 0.01
    tracking["ransac"]["max_rot_no_neighbor"] = 10
    tracking["p2p"]["max_dist"] = 0.02
    tracking["p2p"]["max_normal_angle"] = 45

    with (source_root / "config.yml").open("r", encoding="utf-8") as stream:
        nerf = yaml.load(stream)
    nerf["continual"] = True
    nerf["trunc_start"] = 0.01
    nerf["trunc"] = 0.01
    nerf["mesh_resolution"] = 0.005
    nerf["down_scale_ratio"] = 1
    nerf["fs_sdf"] = 0.1
    nerf["far"] = tracking["depth_processing"]["zfar"]
    nerf["datadir"] = f"{output_root}/nerf_disabled"
    nerf["save_dir"] = nerf["datadir"]
    nerf["notes"] = "tracking_only_no_nerf"
    return tracking, nerf


def _object_in_camera_pose(tracked_frame: Any) -> np.ndarray:
    pose_in_model = np.asarray(
        tracked_frame._pose_in_model,
        dtype=np.float64,
    )
    if pose_in_model.shape != (4, 4) or not np.isfinite(pose_in_model).all():
        raise ValueError("BundleTrack returned an invalid in-memory pose")
    return np.linalg.inv(pose_in_model)


def _validated_object_in_camera_pose(tracked_frame: Any) -> np.ndarray:
    matrix = _object_in_camera_pose(tracked_frame)
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], (0., 0., 0., 1.), atol=1e-6, rtol=0.)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.)
            or not math.isclose(float(np.linalg.det(rotation)), 1., abs_tol=1e-5)):
        raise ValueError("nonrigid backend pose")
    return matrix


def _same_tracker_new_sensor_epoch(old: tuple, new: tuple) -> bool:
    """True when only the sensor epoch changed. The camera and object stay."""
    return (
        isinstance(old, tuple) and isinstance(new, tuple)
        and len(old) == len(new) and len(old) > 2
        and old[0] == new[0] and old[2:] == new[2:] and old[1] != new[1]
    )


def _reject_unmeasured_native_depth_fill(_frame: Any) -> bool:
    """Keep native failures on measured RGB-D; never synthesize missing depth."""
    return False


class PinnedBundleTrackSession:
    """One ordered camera/object stream using an already verified backend.

    Accepts registered, preprocessed BGR/metric-depth/mask arrays directly.
    Runtime verification and registration authorization remain the owner's
    responsibility. This does not convert tracking output into world authority.
    A source restart requires a new session/backend, not timestamp rewriting.
    Calls and close must be serialized by the owner.
    """

    def __init__(self, tracker: Any, *, erode_mask_size: int = 3) -> None:
        if type(erode_mask_size) is not int or erode_mask_size < 0:
            raise ValueError("mask erosion size must be a nonnegative integer")
        self.tracker = tracker
        if callable(getattr(tracker, "_fill_frame_depth_holes", None)):
            # The historical fallback copies neighboring depth into holes and
            # rebuilds geometry. It is not the allowed session-level clipped
            # translation recovery and must not run on the existing owner path.
            tracker._fill_frame_depth_holes = _reject_unmeasured_native_depth_fill
        self.erode_mask_size = erode_mask_size
        # The mask shape is fixed for a registered tracking stream. Keep the
        # small erosion kernel out of the per-frame hot path; this is an
        # adapter optimization only and does not change the selected evidence.
        self._mask_kernel = (
            np.ones((erode_mask_size, erode_mask_size), dtype=np.uint8)
            if erode_mask_size else None
        )
        self._registered_stream_key: tuple[object, ...] | None = None
        self._registered_color_timestamp_ns: int | None = None
        self.frame_count = 0
        self.last_timestamp_ns: int | None = None
        self._grid: tuple[int, ...] | None = None
        self._intrinsics: np.ndarray | None = None
        self._closed = False
        self._failed = False

    def advance_registered(self, frame: Any) -> Any:
        """Consume the current-observation adapter without disk materialization."""
        from ticos_replay.bundletrack_live_input import RegisteredBundleTrackInput
        if not isinstance(frame, RegisteredBundleTrackInput):
            raise ValueError("registered BundleTrack input required")
        if self._registered_stream_key is None and self.frame_count:
            raise ValueError("cannot mix unbound and identity-bound tracker input")
        if self._registered_stream_key is not None and frame.stream_key != self._registered_stream_key:
            if not _same_tracker_new_sensor_epoch(self._registered_stream_key, frame.stream_key):
                raise ValueError("camera, object, calibration or source epoch changed")
            self.last_timestamp_ns = None
        elif (self._registered_color_timestamp_ns is not None
                and frame.color_timestamp_ns <= self._registered_color_timestamp_ns):
            raise ValueError("registered tracker cannot reuse or reorder color evidence")
        # Registered evidence is immutable. Own one writable 0/1 raster, then
        # erode and scale it in place so the native boundary still receives
        # the exact historical 0/255 mask without a second full-frame mask.
        mask = np.array(frame.mask, dtype=np.uint8, copy=True, order="C")
        if self._mask_kernel is not None:
            cv2.erode(mask, self._mask_kernel, dst=mask)
        mask *= np.uint8(255)
        # The uint16 evidence is immutable, so this dtype conversion creates
        # a private writable buffer owned by this call.  Divide in-place and
        # transfer that buffer to the native boundary; the generic ``advance``
        # path still copies caller-owned arrays before BundleTrack can mutate
        # its percentile-filtered depth.
        depth_m = np.ascontiguousarray(frame.depth_mm, dtype=np.float32)
        np.divide(depth_m, np.float32(1000.0), out=depth_m)
        result = self._advance_arrays(
            color_bgr=cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR),
            depth_m=depth_m,
            # ``rectify_registered_rgbd`` emits clipped uint16 millimetre
            # evidence.  Only that exact source may skip the duplicate
            # finite/negative scan after conversion to private float32.
            depth_owned=frame.depth_mm.dtype == np.uint16,
            private_input_buffers=True,
            mask_u8=mask, intrinsics=frame.intrinsics, timestamp_ns=frame.depth_timestamp_ns,
        )
        self._registered_stream_key = frame.stream_key
        self._registered_color_timestamp_ns = frame.color_timestamp_ns
        return result

    def advance(self, **kwargs: Any) -> Any:
        if self._registered_stream_key is not None:
            raise ValueError("cannot bypass identity-bound tracker input")
        return self._advance_arrays(**kwargs)

    def _advance_arrays(self, *, color_bgr: np.ndarray, depth_m: np.ndarray,
                mask_u8: np.ndarray, intrinsics: np.ndarray, timestamp_ns: int,
                depth_owned: bool = False,
                private_input_buffers: bool = False) -> Any:
        if self._closed or self._failed:
            raise RuntimeError("tracking session is closed or failed")
        if type(timestamp_ns) is not int or timestamp_ns <= 0:
            raise ValueError("tracking timestamp must be a positive integer")
        if self.last_timestamp_ns is not None and timestamp_ns <= self.last_timestamp_ns:
            raise ValueError("tracking chronology must increase strictly within one source epoch")
        if (color_bgr.dtype != np.uint8 or color_bgr.ndim != 3 or color_bgr.shape[2] != 3
            or depth_m.dtype != np.float32 or depth_m.shape != color_bgr.shape[:2]
            or mask_u8.dtype != np.uint8 or mask_u8.shape != depth_m.shape
            or not depth_m.size):
            raise ValueError("tracking requires one registered BGR/metric-depth/mask grid")
        if not depth_owned and (not np.isfinite(depth_m).all() or np.any(depth_m < 0)):
            raise ValueError("tracking requires one registered BGR/metric-depth/mask grid")
        if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
            or not np.allclose(intrinsics[2], (0., 0., 1.))):
            raise ValueError("tracking intrinsics are invalid")
        if self._grid is not None and (depth_m.shape != self._grid
                or not np.array_equal(intrinsics, self._intrinsics)):
            raise ValueError("camera grid or intrinsics changed within tracking session")
        # The pinned Frame constructor clones color/depth/mask before retaining
        # them. Keep the caller-owned depth copy because the original runtime
        # applies its percentile filter in place, but avoid redundant Python
        # copies for inputs that the native boundary already owns.
        if private_input_buffers:
            # ``advance_registered`` created these writable arrays from the
            # immutable RegisteredBundleTrackInput immediately above. The
            # native Frame constructor clones them synchronously, so retaining
            # the private buffers here avoids a second full-frame copy without
            # exposing caller-owned evidence to BundleTrack mutation.
            color = color_bgr
        else:
            color = np.array(color_bgr, dtype=np.uint8, copy=True, order="C")
        owns_depth = (
            depth_owned
            and depth_m.flags.owndata
            and depth_m.flags.writeable
            and depth_m.flags.c_contiguous
        )
        depth = depth_m if owns_depth else np.array(depth_m, copy=True, order="C")
        mask = (
            mask_u8
            if private_input_buffers
            else np.array(mask_u8, dtype=np.uint8, copy=True, order="C")
        )
        camera_matrix = np.ascontiguousarray(intrinsics, dtype=np.float64)
        try:
            with _native_inference_context():
                self.tracker.run(color, depth, camera_matrix, f"{self.frame_count:06d}",
                    mask=mask, occ_mask=None, pose_in_model=np.eye(4))
            tracked_frame = self.tracker.bundler._newframe
        except Exception:
            # Native tracking may have partially advanced. Never continue from
            # an unknown state merely because the caller retries the same input.
            self._failed = True
            raise
        self._grid = depth_m.shape
        self._intrinsics = np.array(intrinsics, copy=True)
        self.last_timestamp_ns = timestamp_ns
        self.frame_count += 1
        return tracked_frame

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            tracker = self.tracker
            try:
                tracker.on_finish()
            finally:
                try:
                    manager = getattr(tracker, "manager", None)
                    if manager is not None:
                        manager.shutdown()
                finally:
                    # Native frame/keyframe buffers belong to this session, not
                    # the next source epoch. Do not retain them after shutdown.
                    self.tracker = None


def _latency_distribution(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("latency distribution requires at least one sample")
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("latency samples must be finite and non-negative")
    return {
        "p50": round(float(np.percentile(values, 50)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
        "maximum": round(max(values), 3),
    }


def _steady_state_latency_distribution(
    values: list[float],
) -> dict[str, float] | None:
    steady_state = values[LATENCY_WARMUP_FRAME_COUNT:]
    return _latency_distribution(steady_state) if steady_state else None


@dataclass
class PinnedBundleTrackRuntime:
    """One isolated worker's backend, configuration, provenance and cleanup."""

    session: PinnedBundleTrackSession
    tracking: Any
    evidence: dict[str, Any]
    frame_status: Any
    debugger_guard: NonInteractivePdbGuard
    _closed: bool = False
    _retained_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    _native_environment_key: str | None = None
    _native_environment_previous: str | None = None
    _previous_root_log_level: int | None = None
    # Cache only the serialized evidence derived from each currently retained
    # native frame.  BundleTrack remains the sole pose owner; a changed native
    # pose/status invalidates the entry and is validated again.
    _pose_snapshot_cache: dict[str, tuple[np.ndarray, Any, list[list[float]] | None, str | None]] = field(
        default_factory=dict
    )

    def _cache_pose_snapshot(
        self,
        native: Any,
        identity: str,
        pose: list[list[float]] | None,
        reason: str | None,
    ) -> None:
        """Remember one validated native pose without retaining native state."""
        try:
            source_pose = np.asarray(native._pose_in_model, dtype=np.float64)
        except (AttributeError, TypeError, ValueError):
            return
        if source_pose.shape != (4, 4) or not np.isfinite(source_pose).all():
            return
        self._pose_snapshot_cache[identity] = (
            np.array(source_pose, dtype=np.float64, copy=True),
            native._status,
            pose,
            reason,
        )

    def _pose_snapshot_for_native(
        self,
        native: Any,
        identity: str,
    ) -> tuple[list[list[float]] | None, str | None]:
        """Return current native evidence, revalidating only changed poses."""
        try:
            source_pose = np.asarray(native._pose_in_model, dtype=np.float64)
        except (AttributeError, TypeError, ValueError):
            source_pose = None
        status = native._status
        cached = self._pose_snapshot_cache.get(identity)
        if (
            source_pose is not None
            and source_pose.shape == (4, 4)
            and cached is not None
            and cached[1] == status
            and np.array_equal(source_pose, cached[0])
        ):
            return cached[2], cached[3]

        if status == self.frame_status.Frame.FAIL:
            pose, reason = None, "tracker_failed_observation"
        elif status == self.frame_status.Frame.NO_BA:
            pose, reason = None, "tracker_reference_retained"
        else:
            try:
                pose = _validated_object_in_camera_pose(native).tolist()
                reason = None
            except (ValueError, np.linalg.LinAlgError):
                pose, reason = None, "backend_pose_invalid"
        if source_pose is not None and source_pose.shape == (4, 4) and np.isfinite(source_pose).all():
            self._pose_snapshot_cache[identity] = (
                np.array(source_pose, dtype=np.float64, copy=True),
                status,
                pose,
                reason,
            )
        return pose, reason

    def _retained_pose_snapshot(self, frame: Any, expected_id: str) -> dict[str, Any]:
        """Copy revised native poses with the evidence that originally made them.

        BA can change older frames. The current input's timestamp/hash must not
        be assigned to those older observations. Only metadata is retained here;
        native retirement removes it, without claiming a complete trajectory or
        deleting any downstream reconstructed object.
        """
        known = {**self._retained_inputs, expected_id: {
            "backend_frame_id": expected_id,
            "color_timestamp_ns": str(frame.color_timestamp_ns),
            "depth_timestamp_ns": str(frame.depth_timestamp_ns),
            "input_evidence_hash": frame.evidence_hash,
        }}
        retained = {}
        rows = []
        current_ids: set[str] = set()
        for native in self.session.tracker.bundler._frames.values():
            identity = getattr(native, "_id_str", None)
            if identity not in known or identity in retained:
                self.session._failed = True
                raise ValueError("retained backend frame identity is unknown or duplicated")
            current_ids.add(identity)
            retained[identity] = known[identity]
            pose, reason = self._pose_snapshot_for_native(native, identity)
            rows.append(dict(known[identity], camera_from_object=pose,
                status="candidate" if pose is not None else "not_observable", reason=reason))
        retired = sorted(set(self._retained_inputs) - set(retained))
        self._retained_inputs = retained
        self._pose_snapshot_cache = {
            identity: cached
            for identity, cached in self._pose_snapshot_cache.items()
            if identity in current_ids
        }
        return {
            "scope": "currently_retained_backend_observations_not_complete_trajectory",
            "as_of_color_timestamp_ns": str(frame.color_timestamp_ns),
            "as_of_depth_timestamp_ns": str(frame.depth_timestamp_ns),
            "stream_key": list(frame.stream_key),
            "observations": sorted(rows, key=lambda row: int(row["backend_frame_id"])),
            "retired_backend_frame_ids": retired,
            "world_pose_authorized": False,
        }

    def resource_usage(self) -> dict[str, Any]:
        """Actual backend retention, not an inferred bound from BA window size."""
        if self._closed or self.session._closed:
            return {"closed": True, "bounded_total_memory_verified": False}
        tracker = self.session.tracker
        return {
            "retained_native_frames": len(tracker.bundler._frames),
            "retained_keyframes": len(tracker.bundler._keyframes),
            "pending_neural_reconstruction_frames": len(tracker.kf_to_nerf_list),
            "neural_reconstruction_worker_alive": tracker.p_nerf is not None and tracker.p_nerf.is_alive(),
            "bounded_total_memory_verified": False,
        }

    def track_registered(self, frame: Any) -> dict[str, Any]:
        """Snapshot a camera-local candidate; never return a mutable native frame."""
        started = time.perf_counter()
        tracked = self.session.advance_registered(frame)
        tracker_run_ms = (time.perf_counter() - started) * 1000.0
        expected_id = f"{self.session.frame_count - 1:06d}"
        reason = None
        pose = None
        if getattr(tracked, "_id_str", None) != expected_id:
            self.session._failed = True
            reason = "backend_frame_identity_mismatch"
        elif tracked._status == self.frame_status.Frame.FAIL:
            reason = "tracker_failed_current_observation"
        elif tracked._status == self.frame_status.Frame.NO_BA:
            # Retained only as the next frame's match reference. The pose is
            # the previous frame's pose, so publishing it would freeze a hold
            # onto the wire.
            reason = "tracker_reference_retained"
        else:
            try:
                matrix = _validated_object_in_camera_pose(tracked)
                pose = matrix.tolist()
            except (ValueError, np.linalg.LinAlgError):
                reason = "backend_pose_invalid"
        if reason != "backend_frame_identity_mismatch":
            self._cache_pose_snapshot(tracked, expected_id, pose, reason)
        history = None
        history_started = time.perf_counter()
        if reason != "backend_frame_identity_mismatch":
            history = self._retained_pose_snapshot(frame, expected_id)
        history_snapshot_ms = (time.perf_counter() - history_started) * 1000.0
        runtime_timing_ms = {
            "tracker_run_ms": round(tracker_run_ms, 3),
            "history_snapshot_ms": round(history_snapshot_ms, 3),
            "track_registered_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        return {
            "schema": "ticos.bundletrack-camera-local-candidate.v1",
            "status": "candidate" if pose is not None else "not_observable",
            "reason": reason,
            "tracking_status": "no_ba" if tracked._status == self.frame_status.Frame.NO_BA else
                "fail" if tracked._status == self.frame_status.Frame.FAIL else "candidate",
            "output_index": self.session.frame_count - 1,
            "stream_key": list(frame.stream_key),
            "camera_frame_id": frame.stream_key[4],
            "selected_object_id": frame.stream_key[6],
            "object_frame_definition": "backend_tracking_gauge_not_semantic_body_frame",
            "color_timestamp_ns": str(frame.color_timestamp_ns),
            "depth_timestamp_ns": str(frame.depth_timestamp_ns),
            "pose_time_basis": "depth_sensor_timestamp",
            "camera_from_object": pose,
            "input_evidence_hash": frame.evidence_hash,
            "world_pose_authorized": False,
            "runtime_resources": self.resource_usage(),
            "retained_pose_snapshot": history,
            "runtime_timing_ms": runtime_timing_ms,
        }

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self.session.close()
            finally:
                self._retained_inputs.clear()
                self._pose_snapshot_cache.clear()
                self.debugger_guard.restore()
                if self._previous_root_log_level is not None:
                    logging.getLogger().setLevel(self._previous_root_log_level)
                if self._native_environment_key is not None:
                    if self._native_environment_previous is None:
                        os.environ.pop(self._native_environment_key, None)
                    else:
                        os.environ[self._native_environment_key] = (
                            self._native_environment_previous
                        )

    def __enter__(self):
        if self._closed:
            raise RuntimeError("tracking runtime is closed")
        return self

    def __exit__(self, *_exc):
        self.close()


def open_pinned_bundletrack_runtime(
    *,
    output_root: Path,
    source_root: Path,
    build_root: Path,
    dependency_prefix: Path,
    checkpoint_path: Path,
    maximum_ba_frames: int = 4,
    feature_resize: int = 320,
    maximum_neighbor_translation_m: float = 0.02,
    runtime_output_mode: str = "online_pose",
    tracking_only: bool = False,
) -> PinnedBundleTrackRuntime:
    """Create the original verified backend without requiring a sequence.

    Use one runtime per isolated worker: upstream imports, RNG and debugger
    hooks are process-global. Configuration/debug output remains exclusive;
    no input image or pose files are required to advance its session.
    """
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite BundleTrack output: {output_root}")
    if type(maximum_ba_frames) is not int or maximum_ba_frames <= 0:
        raise ValueError("maximum_ba_frames must be a positive integer")
    if type(feature_resize) is not int or feature_resize < 128:
        raise ValueError("feature_resize must be an integer of at least 128")
    if (
        isinstance(maximum_neighbor_translation_m, bool)
        or not isinstance(maximum_neighbor_translation_m, (int, float))
        or not math.isfinite(float(maximum_neighbor_translation_m))
        or not 0.005 <= float(maximum_neighbor_translation_m) <= 0.10
    ):
        raise ValueError(
            "maximum_neighbor_translation_m must be within [0.005, 0.10]"
        )
    if runtime_output_mode not in RUNTIME_OUTPUT_MODES:
        raise ValueError(
            f"runtime_output_mode must be one of {RUNTIME_OUTPUT_MODES}"
        )
    runtime = _verify_runtime(
        source_root=source_root,
        build_root=build_root,
        dependency_prefix=dependency_prefix,
        checkpoint_path=checkpoint_path,
    )

    _preload_torchvision_for_active_torch()
    sys.path.insert(0, str(build_root))
    sys.path.insert(0, str(source_root))
    import my_cpp
    import loftr_wrapper

    checkpoint_runtime_root = checkpoint_path.parents[3]
    loftr_wrapper.code_dir = str(checkpoint_runtime_root)
    from bundlesdf import BundleSdf, set_seed
    if type(tracking_only) is not bool:
        raise ValueError("tracking_only must be boolean")
    options = {}
    native_environment_key = None
    native_environment_previous = None
    native_raw_frame_patch_present = False
    if tracking_only:
        import inspect
        if "tracking_only" not in inspect.signature(BundleSdf).parameters:
            raise ValueError("tracking-only runtime requires tools/patches/bundlesdf-tracking-only.patch")
        options["tracking_only"] = True
        # Raw RGB/depth buffers are only consumed by the disabled
        # reconstruction/audit path. Enable the native allocation elision
        # only when the pinned source carries the matching opt-in patch;
        # older builds retain their original behavior.
        raw_frame_source = source_root / "BundleTrack" / "src" / "Frame.cpp"
        try:
            source_marker_present = (
                "TICOS_BUNDLETRACK_DROP_RAW"
                in raw_frame_source.read_text(encoding="utf-8")
            )
        except OSError:
            source_marker_present = False
        native_binary = Path(str(getattr(my_cpp, "__file__", "")))
        try:
            with native_binary.open("rb") as stream:
                binary_marker_present = b"TICOS_BUNDLETRACK_DROP_RAW" in stream.read()
        except OSError:
            binary_marker_present = False
        if source_marker_present and binary_marker_present:
            native_raw_frame_patch_present = True
            runtime["tracking_only_raw_frame_copy_optimization"] = "enabled"
        else:
            runtime["tracking_only_raw_frame_copy_optimization"] = "unavailable"
    runtime["tracking_only"] = tracking_only

    # The pinned research checkout configures the process root logger at INFO
    # and emits a line for every LoFTR/RANSAC operation.  The live online-pose
    # contract already carries structured timing/evidence, so suppress those
    # debug lines while the tracking-only owner is active.  Restore the caller
    # level when the runtime closes; this does not change estimator state.
    previous_root_log_level = logging.getLogger().level

    output_root.mkdir(parents=True, exist_ok=False)
    tracking, nerf = _tracking_configs(
        source_root=source_root,
        output_root=output_root,
        maximum_ba_frames=maximum_ba_frames,
        feature_resize=feature_resize,
        maximum_neighbor_translation_m=float(
            maximum_neighbor_translation_m
        ),
        runtime_output_mode=runtime_output_mode,
    )
    yaml = _yaml_backend()
    track_config_path = output_root / "config_bundletrack.yml"
    nerf_config_path = output_root / "config_nerf.yml"
    with track_config_path.open("w", encoding="utf-8") as stream:
        yaml.dump(tracking, stream)
    with nerf_config_path.open("w", encoding="utf-8") as stream:
        yaml.dump(nerf, stream)

    if native_raw_frame_patch_present:
        native_environment_key = "TICOS_BUNDLETRACK_DROP_RAW"
        native_environment_previous = os.environ.get(native_environment_key)
        os.environ[native_environment_key] = "1"
    debugger_guard = NonInteractivePdbGuard()
    if tracking_only and runtime_output_mode == "online_pose":
        logging.getLogger().setLevel(logging.WARNING)
    try:
        debugger_guard.install()
        set_seed(0)
        tracker = BundleSdf(
            cfg_track_dir=str(track_config_path),
            cfg_nerf_dir=str(nerf_config_path),
            start_nerf_keyframes=1_000_000,
            use_gui=False,
            **options,
        )
    except BaseException:
        debugger_guard.restore()
        if tracking_only and runtime_output_mode == "online_pose":
            logging.getLogger().setLevel(previous_root_log_level)
        if native_environment_key is not None:
            if native_environment_previous is None:
                os.environ.pop(native_environment_key, None)
            else:
                os.environ[native_environment_key] = native_environment_previous
        raise
    session = PinnedBundleTrackSession(tracker, erode_mask_size=int(tracking["erode_mask"]))
    runtime["native_depth_hole_fill_policy"] = "disabled_unmeasured_depth"
    return PinnedBundleTrackRuntime(
        session, tracking, runtime, my_cpp, debugger_guard,
        _native_environment_key=native_environment_key,
        _native_environment_previous=native_environment_previous,
        _previous_root_log_level=(
            previous_root_log_level
            if tracking_only and runtime_output_mode == "online_pose"
            else None
        ),
    )


def run_pinned_bundletrack(
    *,
    sequence_root: Path,
    output_root: Path,
    source_root: Path,
    build_root: Path,
    dependency_prefix: Path,
    checkpoint_path: Path,
    limit: int | None = None,
    maximum_ba_frames: int = 4,
    feature_resize: int = 320,
    maximum_neighbor_translation_m: float = 0.02,
    runtime_output_mode: str = "full_audit",
    tracking_only: bool = False,
) -> dict[str, Any]:
    if type(tracking_only) is not bool:
        raise ValueError("tracking_only must be boolean")
    if tracking_only and runtime_output_mode != "online_pose":
        raise ValueError("tracking_only requires online_pose output")
    manifest = validate_sequence_for_tracking(sequence_root)
    frame_count = len(manifest["frames"])
    use_log_sequence_clock = manifest.get("sequence_clock") == "mcap_log_time_pair_midpoint"
    if limit is not None:
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer or None")
        frame_count = min(frame_count, limit)
    opened = open_pinned_bundletrack_runtime(output_root=output_root, source_root=source_root,
        build_root=build_root, dependency_prefix=dependency_prefix, checkpoint_path=checkpoint_path,
        maximum_ba_frames=maximum_ba_frames, feature_resize=feature_resize,
        maximum_neighbor_translation_m=maximum_neighbor_translation_m, runtime_output_mode=runtime_output_mode,
        tracking_only=tracking_only)
    session, tracking, runtime = opened.session, opened.tracking, opened.evidence
    my_cpp, debugger_guard = opened.frame_status, opened.debugger_guard
    input_decode_and_preprocess_latencies_ms: list[float] = []
    latencies_ms: list[float] = []
    online_pose_ready_latencies_ms: list[float] = []
    pose_persistence_latencies_ms: list[float] = []
    frame_results: list[dict[str, Any]] = []
    online_pose_dir = output_root / "ob_in_cam"
    try:
        if runtime_output_mode == "online_pose":
            online_pose_dir.mkdir(parents=True, exist_ok=True)
        intrinsics = np.asarray(
            np.loadtxt(sequence_root / "cam_K.txt"),
            dtype=np.float64,
        ).reshape(3, 3)
        for index in range(frame_count):
            stem = f"{index:06d}"
            manifest_frame = manifest["frames"][index]
            color_timestamp_ns = int(manifest_frame["color_timestamp_ns"])
            timestamp_ns = (
                (int(manifest_frame["color_log_timestamp_ns"])
                 + int(manifest_frame["depth_log_timestamp_ns"])) // 2
                if use_log_sequence_clock else color_timestamp_ns
            )
            frame_started = time.perf_counter()
            color = cv2.imread(str(sequence_root / "rgb" / f"{stem}.png"))
            depth = cv2.imread(
                str(sequence_root / "depth" / f"{stem}.png"),
                cv2.IMREAD_UNCHANGED,
            )
            mask = cv2.imread(
                str(sequence_root / "masks" / f"{stem}.png"),
                cv2.IMREAD_UNCHANGED,
            )
            if color is None or depth is None or mask is None:
                raise ValueError(f"failed to decode registered frame {index}")
            if color.shape[:2] != depth.shape or depth.shape != mask.shape:
                raise ValueError("registered RGB-D grid changed or would require resize")
            depth_m = np.ascontiguousarray(depth, dtype=np.float32) / 1000.0
            mask_u8 = np.ascontiguousarray(mask > 0, dtype=np.uint8) * 255
            erode_size = int(tracking["erode_mask"])
            if erode_size > 0:
                mask_u8 = cv2.erode(
                    mask_u8,
                    np.ones((erode_size, erode_size), dtype=np.uint8),
                )
            tracker_started = time.perf_counter()
            input_decode_and_preprocess_latency_ms = (
                tracker_started - frame_started
            ) * 1000.0
            input_decode_and_preprocess_latencies_ms.append(
                input_decode_and_preprocess_latency_ms
            )
            tracked_frame = session.advance(
                color_bgr=color,
                depth_m=depth_m,
                intrinsics=intrinsics,
                mask_u8=mask_u8,
                timestamp_ns=timestamp_ns,
            )
            latency_ms = (time.perf_counter() - tracker_started) * 1000.0
            latencies_ms.append(latency_ms)
            online_pose_ready_latency_ms: float | None = None
            pose_persistence_latency_ms: float | None = None
            if runtime_output_mode == "online_pose":
                object_in_camera = _object_in_camera_pose(tracked_frame)
                online_pose_ready_latency_ms = (
                    time.perf_counter() - frame_started
                ) * 1000.0
                online_pose_ready_latencies_ms.append(
                    online_pose_ready_latency_ms
                )
                persistence_started = time.perf_counter()
                np.savetxt(
                    online_pose_dir / f"{stem}.txt",
                    object_in_camera,
                    fmt="%.10g",
                )
                pose_persistence_latency_ms = (
                    time.perf_counter() - persistence_started
                ) * 1000.0
                pose_persistence_latencies_ms.append(
                    pose_persistence_latency_ms
                )
            if tracked_frame._status == my_cpp.Frame.FAIL:
                tracking_status = "fail"
            elif tracked_frame._status == my_cpp.Frame.NO_BA:
                tracking_status = "no_ba"
            else:
                tracking_status = "candidate"
            previous_timestamp_ns = (
                (
                    (
                        int(
                            manifest["frames"][index - 1][
                                "color_log_timestamp_ns"
                            ]
                        )
                        + int(
                            manifest["frames"][index - 1][
                                "depth_log_timestamp_ns"
                            ]
                        )
                    )
                    // 2
                    if use_log_sequence_clock
                    else int(
                        manifest["frames"][index - 1][
                            "color_timestamp_ns"
                        ]
                    )
                )
                if index > 0
                else None
            )
            frame_results.append(
                {
                    "output_index": index,
                    "color_timestamp_ns": color_timestamp_ns,
                    "sequence_timestamp_ns": timestamp_ns,
                    "input_gap_ms": (
                        round((timestamp_ns - previous_timestamp_ns) / 1_000_000.0, 6)
                        if previous_timestamp_ns is not None
                        else None
                    ),
                    "tracking_status": tracking_status,
                    "input_decode_and_preprocess_latency_ms": round(
                        input_decode_and_preprocess_latency_ms,
                        3,
                    ),
                    "tracker_inference_latency_ms": round(latency_ms, 3),
                    "latency_ms": round(latency_ms, 3),
                    "online_pose_ready_latency_ms": (
                        round(online_pose_ready_latency_ms, 3)
                        if online_pose_ready_latency_ms is not None
                        else None
                    ),
                    "pose_persistence_latency_ms": (
                        round(pose_persistence_latency_ms, 3)
                        if pose_persistence_latency_ms is not None
                        else None
                    ),
                }
            )
    finally:
        opened.close()

    pose_dir = output_root / "ob_in_cam"
    pose_count = len(list(pose_dir.glob("*.txt"))) if pose_dir.is_dir() else 0
    tracking_failure_count = sum(
        row["tracking_status"] == "fail" for row in frame_results
    )
    input_gaps_ms = [
        float(row["input_gap_ms"])
        for row in frame_results
        if row["input_gap_ms"] is not None
    ]
    summary = {
        "schema": RUNNER_SCHEMA,
        "status": "complete" if pose_count == frame_count else "incomplete",
        "sequence_manifest_path": str(sequence_root / "manifest.json"),
        "sequence_manifest_sha256": _sha256_path(sequence_root / "manifest.json"),
        "sequence_input_content_sha256": manifest["input_content_sha256"],
        "frame_count": frame_count,
        "pose_count": pose_count,
        "valid_pose_candidate_count": pose_count - tracking_failure_count,
        "tracking_failure_count": tracking_failure_count,
        "tracking_result": (
            "candidate" if tracking_failure_count == 0 else "fail"
        ),
        "upstream_invalid_pose_files_retained_for_audit": True,
        "upstream_debug_break_bypass_count": debugger_guard.break_count,
        "input_resize_forbidden": True,
        "input_resize_performed": False,
        "maximum_ba_frames": maximum_ba_frames,
        "feature_resize": feature_resize,
        "maximum_neighbor_translation_m": round(
            float(maximum_neighbor_translation_m), 6
        ),
        "runtime_output_mode": runtime_output_mode,
        "debug_artifact_persistence_in_inference_latency": (
            runtime_output_mode == "full_audit"
        ),
        "pose_persistence_after_inference_timing": (
            runtime_output_mode == "online_pose"
        ),
        "nerf_enabled": False,
        "latency_measurement_contract": {
            "schema": "ticos.bundletrack-latency-measurement.v1",
            "clock": "time.perf_counter",
            "legacy_latency_ms_alias": "tracker_inference_latency_ms",
            "online_pose_ready_starts_before": "cv2_image_decode",
            "online_pose_ready_ends_after": "in_memory_object_in_camera_pose",
            "pose_persistence_excluded_from_online_pose_ready": True,
            "steady_state_warmup_frame_count": LATENCY_WARMUP_FRAME_COUNT,
        },
        "input_decode_and_preprocess_latency_ms": _latency_distribution(
            input_decode_and_preprocess_latencies_ms
        ),
        "tracker_inference_latency_ms": _latency_distribution(latencies_ms),
        "latency_ms": _latency_distribution(latencies_ms),
        "online_pose_ready_latency_ms": (
            _latency_distribution(online_pose_ready_latencies_ms)
            if online_pose_ready_latencies_ms
            else None
        ),
        "steady_state_online_pose_ready_latency_ms": (
            _steady_state_latency_distribution(
                online_pose_ready_latencies_ms
            )
            if online_pose_ready_latencies_ms
            else None
        ),
        "pose_persistence_latency_ms": (
            _latency_distribution(pose_persistence_latencies_ms)
            if pose_persistence_latencies_ms
            else None
        ),
        "input_timing": {
            "sequence_clock": (
                "mcap_log_time_pair_midpoint"
                if use_log_sequence_clock
                else "ros_image_header_capture_time"
            ),
            "gap_p50_ms": round(float(np.percentile(input_gaps_ms, 50)), 6),
            "gap_p95_ms": round(float(np.percentile(input_gaps_ms, 95)), 6),
            "gap_maximum_ms": round(max(input_gaps_ms), 6),
            "gap_above_100ms_count": sum(gap > 100.0 for gap in input_gaps_ms),
            "gap_above_250ms_count": sum(gap > 250.0 for gap in input_gaps_ms),
        },
        "frames": frame_results,
        "runtime": runtime,
    }
    summary["content_hash"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (output_root / "ticos_runner_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--dependency-prefix", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tracking-only", action="store_true",
                        help="Use the existing live tracking-only owner in online_pose mode.")
    parser.add_argument("--maximum-ba-frames", type=int, default=4)
    parser.add_argument("--feature-resize", type=int, default=320)
    parser.add_argument(
        "--maximum-neighbor-translation-m",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--runtime-output-mode",
        choices=RUNTIME_OUTPUT_MODES,
        default="full_audit",
    )
    args = parser.parse_args()
    report = run_pinned_bundletrack(
        sequence_root=args.sequence_root,
        output_root=args.output_root,
        source_root=args.source_root,
        build_root=args.build_root,
        dependency_prefix=args.dependency_prefix,
        checkpoint_path=args.checkpoint,
        limit=args.limit,
        maximum_ba_frames=args.maximum_ba_frames,
        feature_resize=args.feature_resize,
        maximum_neighbor_translation_m=(
            args.maximum_neighbor_translation_m
        ),
        runtime_output_mode=args.runtime_output_mode,
        tracking_only=args.tracking_only,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "complete" else 2)


if __name__ == "__main__":
    main()
