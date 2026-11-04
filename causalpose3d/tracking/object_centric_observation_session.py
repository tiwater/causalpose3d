"""Compose original persistent SAM memory and verified BundleTrack runtime.

``OriginalPersistentWorldRegistry`` owns one serialized object session. Its
``ObjectCentricObservationSession.advance`` consumes the current SAM2 mask and
active BundleTrack result; outputs remain camera-local proposals, with no table
transform, body-gauge claim or publication authority.

When BundleTrack returns not_observable but the current SAM2 mask still has
registered depth support, an optional mask-depth translation recovery may keep
the last BundleTrack rotation and publish a clipped camera-local candidate.
That recovery is an adapter inside this owner, not a second world model.
"""
from contextlib import nullcontext
from copy import deepcopy
import inspect
import time

import numpy as np

from causalpose3d.primitives.mask_depth_pca_pose import (
    DEFAULT_MAXIMUM_TRANSLATION_STEP_M,
    DEFAULT_MASK_DEPTH_RECOVERY_CENTROID_ESTIMATOR,
    DEFAULT_MASK_DEPTH_RECOVERY_METHOD,
    DEFAULT_MINIMUM_DEPTH_SUPPORT_PIXELS,
    DEFAULT_RECOVERY_MINIMUM_DEPTH_SUPPORT_PIXELS,
    MASK_DEPTH_RECOVERY_CENTROID_ESTIMATORS,
    MASK_DEPTH_RECOVERY_METHODS,
    estimate_mask_depth_centroid_camera_with_diagnostics,
    recover_camera_from_object_with_prior,
    recover_camera_from_object_with_surface_prior,
)


class ObjectCentricObservationSession:
    def __init__(
        self,
        *,
        masks,
        poses,
        pose_context=nullcontext,
        shape=None,
        shape_factory=None,
        feature_observer=None,
        surface=None,
        enable_mask_depth_recovery: bool = False,
        maximum_recovery_translation_step_m: float = DEFAULT_MAXIMUM_TRANSLATION_STEP_M,
        minimum_recovery_depth_support_pixels: int = DEFAULT_RECOVERY_MINIMUM_DEPTH_SUPPORT_PIXELS,
        mask_depth_recovery_centroid_estimator: str = DEFAULT_MASK_DEPTH_RECOVERY_CENTROID_ESTIMATOR,
        mask_depth_recovery_method: str = DEFAULT_MASK_DEPTH_RECOVERY_METHOD,
        deferred_metric_initialization: bool = False,
    ):
        if shape is not None and shape_factory is not None:
            raise ValueError("provide one shape owner or its initialization factory")
        if feature_observer is not None and not callable(feature_observer):
            raise ValueError("feature_observer must be callable")
        if type(enable_mask_depth_recovery) is not bool:
            raise ValueError("enable_mask_depth_recovery must be boolean")
        if (isinstance(maximum_recovery_translation_step_m, (bool, np.bool_))
                or not np.isfinite(maximum_recovery_translation_step_m)
                or float(maximum_recovery_translation_step_m) <= 0.0):
            raise ValueError("maximum_recovery_translation_step_m must be finite and positive")
        if type(minimum_recovery_depth_support_pixels) is not int or minimum_recovery_depth_support_pixels <= 0:
            raise ValueError("minimum_recovery_depth_support_pixels must be a positive integer")
        if mask_depth_recovery_centroid_estimator not in MASK_DEPTH_RECOVERY_CENTROID_ESTIMATORS:
            raise ValueError("unsupported mask-depth recovery centroid estimator")
        if mask_depth_recovery_method not in MASK_DEPTH_RECOVERY_METHODS:
            raise ValueError("unsupported mask-depth recovery method")
        self.masks = masks
        self.poses = poses
        self.metric_initialized = not deferred_metric_initialization
        self._closed = False
        self._failed = False
        self._pose_context = pose_context
        self.shape = shape
        self._shape_factory = shape_factory
        self._feature_observer = feature_observer
        self.surface = surface
        self.enable_mask_depth_recovery = enable_mask_depth_recovery
        self._last_recovery_failure = None
        self.maximum_recovery_translation_step_m = float(
            maximum_recovery_translation_step_m
        )
        self.minimum_recovery_depth_support_pixels = int(
            minimum_recovery_depth_support_pixels
        )
        self.mask_depth_recovery_centroid_estimator = (
            mask_depth_recovery_centroid_estimator
        )
        self.mask_depth_recovery_method = mask_depth_recovery_method
        self._last_camera_from_object: np.ndarray | None = None
        self._last_camera_mask_centroid: np.ndarray | None = None
        self._recovery_prior_active = False

    def _try_mask_depth_recovery(self, current, *, sam2_latency_ms: float, diagnostic: dict):
        prior = self._last_camera_from_object
        if prior is None:
            self._last_recovery_failure = {
                "reason": "recovery_prior_unavailable",
                "depth_supported_pixels": 0,
            }
            return None
        prior_centroid = self._last_camera_mask_centroid
        if (
            prior_centroid is None
            and self.mask_depth_recovery_method == "mask_centroid"
        ):
            self._last_recovery_failure = {
                "reason": "recovery_prior_mask_centroid_unavailable",
                "depth_supported_pixels": 0,
            }
            return None
        recovery_started = time.perf_counter()
        depth_m = np.asarray(current.depth_mm, dtype=np.float64) / 1000.0
        recovery_kwargs = dict(
            depth_m=depth_m,
            mask=np.asarray(current.mask, dtype=bool),
            camera_matrix=np.asarray(current.intrinsics, dtype=np.float64),
            prior_camera_from_object=prior,
            prior_mask_centroid_camera=prior_centroid,
            maximum_translation_step_m=self.maximum_recovery_translation_step_m,
            minimum_depth_support_pixels=self.minimum_recovery_depth_support_pixels,
        )
        if self.mask_depth_recovery_method == "causal_surface_translation_icp":
            reason, surface_points, surface_diagnostics = self._causal_surface_prior(
                current
            )
            if reason is not None:
                self._last_recovery_failure = {
                    "reason": reason,
                    **surface_diagnostics,
                }
                return None
            pose, reason, recovery_diag = recover_camera_from_object_with_surface_prior(
                **recovery_kwargs,
                surface_points_object_m=surface_points,
            )
            recovery_diag = {
                **surface_diagnostics,
                **dict(recovery_diag or {}),
                "recovery_method": self.mask_depth_recovery_method,
            }
        else:
            pose, reason, recovery_diag = recover_camera_from_object_with_prior(
                **recovery_kwargs,
                centroid_estimator=self.mask_depth_recovery_centroid_estimator,
            )
            recovery_diag = {
                **dict(recovery_diag or {}),
                "recovery_method": self.mask_depth_recovery_method,
            }
        recovery_ms = round((time.perf_counter() - recovery_started) * 1000.0, 3)
        if pose is None:
            self._last_recovery_failure = {
                "reason": reason,
                **dict(recovery_diag or {}),
                "recovery_ms": recovery_ms,
            }
            return None
        self._last_recovery_failure = None
        self._recovery_prior_active = True
        self._last_camera_from_object = np.array(pose, dtype=np.float64, copy=True)
        self._last_camera_mask_centroid = np.asarray(
            recovery_diag["mask_centroid_camera_m"], dtype=np.float64
        )
        return dict(
            status="candidate",
            reason=None,
            color_timestamp_ns=str(current.color_timestamp_ns),
            depth_timestamp_ns=str(current.depth_timestamp_ns),
            stream_key=list(current.stream_key),
            selected_object_id=current.stream_key[6],
            camera_frame_id=current.stream_key[4],
            object_frame_definition="backend_tracking_gauge_not_semantic_body_frame",
            camera_from_object=pose.tolist(),
            input_evidence_hash=current.evidence_hash,
            source_mask_evidence_sha256=self._current_mask_evidence_sha256(current),
            pose_time_basis="depth_sensor_timestamp",
            world_pose_authorized=False,
            pose_source="mask_depth_recovery",
            mask_depth_recovery=dict(recovery_diag),
            mask_diagnostic=diagnostic,
            runtime_timing_ms={
                "sam2_ms": round(sam2_latency_ms, 3),
                "pose_ms": recovery_ms,
                "recovery_ms": recovery_ms,
                "total_ms": round(sam2_latency_ms + recovery_ms, 3),
            },
        )

    def _causal_surface_prior(self, current):
        """Read only the previous CausalObjectSurface state for this stream."""
        if self.surface is None:
            return "causal_surface_prior_unavailable", None, {
                "surface_prior_point_count": 0
            }
        snapshot = self.surface.snapshot
        if "maximum_point_count" in inspect.signature(snapshot).parameters:
            evidence = snapshot(maximum_point_count=1500)
        else:
            evidence = snapshot()
        if not isinstance(evidence, dict):
            return "causal_surface_prior_invalid", None, {
                "surface_prior_point_count": 0
            }
        point_count = len(evidence.get("points_object_m") or [])
        diagnostics = {
            "surface_prior_source_owner": "CausalObjectSurface",
            "surface_prior_gauge_id": evidence.get("gauge_id"),
            "surface_prior_as_of_depth_timestamp_ns": evidence.get(
                "as_of_depth_timestamp_ns"
            ),
            "surface_prior_point_count": point_count,
        }
        if evidence.get("status") != "observed_surface_candidate":
            return "causal_surface_prior_support_insufficient", None, diagnostics
        if evidence.get("stream_key") != list(current.stream_key):
            return "causal_surface_prior_stream_mismatch", None, diagnostics
        try:
            prior_timestamp_ns = int(evidence["as_of_depth_timestamp_ns"])
            current_timestamp_ns = int(current.depth_timestamp_ns)
        except (KeyError, TypeError, ValueError):
            return "causal_surface_prior_timestamp_invalid", None, diagnostics
        if prior_timestamp_ns >= current_timestamp_ns:
            return "causal_surface_prior_not_causal", None, diagnostics
        points = np.asarray(evidence.get("points_object_m"), dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,):
            return "causal_surface_prior_invalid", None, diagnostics
        return None, points, diagnostics

    def _current_mask_evidence_sha256(self, current) -> str | None:
        """Return SAM2's exact mask identity only for this RGB-D tick."""
        evidence = getattr(self.masks, "current_visual_mask_evidence", None)
        if not isinstance(evidence, dict):
            return None
        digest = evidence.get("mask_evidence_sha256")
        if (
            evidence.get("source") != "original_sam2_video_memory"
            or evidence.get("stream_key") != list(current.stream_key)
            or evidence.get("color_timestamp_ns") != str(current.color_timestamp_ns)
            or evidence.get("depth_timestamp_ns") != str(current.depth_timestamp_ns)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return None
        return digest

    def _attach_measured_task_features(
        self,
        current,
        candidate,
        declared_features=None,
    ):
        if (
            self.surface is None
            or candidate.get("status") != "candidate"
        ):
            return candidate
        if declared_features is not None:
            records = declared_features
        elif self._feature_observer is not None:
            records = self._feature_observer(current, candidate)
        else:
            return candidate
        if records is None:
            return candidate
        if not isinstance(records, (list, tuple)):
            raise ValueError("feature observer must return a sequence or None")
        return dict(candidate, measured_task_features=deepcopy(list(records)))

    def advance(
        self,
        frame,
        *,
        initialize=False,
        metric_anchor_gate=None,
        measured_task_features=None,
    ):
        if self._closed or self._failed:
            raise RuntimeError("object-centric observation session closed or failed")
        started = time.perf_counter()
        current = self.masks.advance(frame, initialize=initialize)
        sam2_latency_ms = (time.perf_counter() - started) * 1000.0
        diagnostic = dict(self.masks.last_diagnostic)
        if current is not None and not self.metric_initialized:
            reason = (metric_anchor_gate(self.masks.current_visual_mask)
                      if metric_anchor_gate is not None else "metric_anchor_gate_unavailable")
            if reason is not None:
                current = None
                diagnostic["reason"] = reason
        if current is None:
            result = dict(status="not_observable", reason=diagnostic["reason"],
                color_timestamp_ns=str(frame.color_timestamp_ns),
                depth_timestamp_ns=str(frame.depth_timestamp_ns),
                selected_object_id=frame.stream_key[6], camera_from_object=None,
                mask_diagnostic=diagnostic, world_pose_authorized=False)
            result["metric_initialized"] = self.metric_initialized
            result["current_visual_mask_evidence"] = getattr(self.masks, "current_visual_mask_evidence", None)
            if self.surface is not None and self.metric_initialized:
                snapshot = self.surface.snapshot
                if "maximum_point_count" in inspect.signature(snapshot).parameters:
                    result["surface_evidence"] = snapshot(maximum_point_count=1500)
                else:
                    result["surface_evidence"] = snapshot()
            result["runtime_timing_ms"] = {
                "sam2_ms": round(sam2_latency_ms, 3),
                "total_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
            return result
        pose_started = time.perf_counter()
        feature_observer_ms = 0.0
        reentry_rejected: dict[str, float] | None = None
        try:
            with self._pose_context():
                candidate = self.poses.track_registered(current)
                self.metric_initialized = True
            pose_latency_ms = (time.perf_counter() - pose_started) * 1000.0
            mask_evidence_sha256 = self._current_mask_evidence_sha256(current)
            if mask_evidence_sha256 is not None:
                candidate = dict(
                    candidate,
                    source_mask_evidence_sha256=mask_evidence_sha256,
                )
            if (
                self.enable_mask_depth_recovery
                and self._recovery_prior_active
                and candidate.get("status") == "candidate"
            ):
                raw_pose = candidate.get("camera_from_object")
                try:
                    reentered_pose = np.asarray(raw_pose, dtype=np.float64)
                except (TypeError, ValueError):
                    reentered_pose = np.empty((0, 0), dtype=np.float64)
                if (
                    self._last_camera_from_object is not None
                    and reentered_pose.shape == (4, 4)
                    and np.isfinite(reentered_pose).all()
                ):
                    displacement = float(np.linalg.norm(
                        reentered_pose[:3, 3] - self._last_camera_from_object[:3, 3]
                    ))
                    if displacement > self.maximum_recovery_translation_step_m:
                        reentry_rejected = {"bundletrack_reentry_translation_step_m": displacement}
                        candidate = dict(
                            candidate,
                            status="not_observable",
                            reason="bundletrack_reentry_exceeds_recovery_jump_gate",
                            camera_from_object=None,
                        )
                else:
                    reentry_rejected = {"bundletrack_reentry_pose_invalid": 1.0}
                    candidate = dict(
                        candidate,
                        status="not_observable",
                        reason="bundletrack_reentry_pose_invalid",
                        camera_from_object=None,
                    )
            # A rejected native candidate remains a candidate in the native
            # retained history. Passing its rewritten status to the surface
            # violates the history contract; passing the native candidate
            # would fuse the pose we just rejected. Retire this session before
            # either action so the owner can fail closed with a clear reason.
            if reentry_rejected is not None and self.surface is not None:
                error = RuntimeError(
                    "bundletrack_reentry_rejected_surface_requires_session_reset"
                )
                error.reentry_failure_evidence = {
                    "reason": candidate["reason"],
                    "translation_step_m": reentry_rejected.get("bundletrack_reentry_translation_step_m"),
                    "maximum_translation_step_m": self.maximum_recovery_translation_step_m,
                    "color_timestamp_ns": str(current.color_timestamp_ns),
                    "depth_timestamp_ns": str(current.depth_timestamp_ns),
                    "input_evidence_hash": current.evidence_hash,
                    "physical_pose_error_verified": False,
                }
                raise error
            if (
                self.enable_mask_depth_recovery
                and candidate.get("status") != "candidate"
            ):
                recovered = self._try_mask_depth_recovery(
                    current,
                    sam2_latency_ms=sam2_latency_ms,
                    diagnostic=diagnostic,
                )
                if recovered is not None and reentry_rejected is not None:
                    recovered = dict(
                        recovered,
                        bundletrack_reentry=reentry_rejected,
                    )
                if recovered is None and self._last_recovery_failure is not None:
                    candidate = dict(
                        candidate,
                        mask_depth_recovery_failure=dict(self._last_recovery_failure),
                    )
                if recovered is not None:
                    surface_started = time.perf_counter()
                    surface_evidence = None
                    if self.surface is not None:
                        # Advance the gauge with the native BundleTrack tick
                        # (often not_observable) so output_index stays
                        # consecutive. Recovery pose is wire-only and must not
                        # be fed into CausalObjectSurface.update().
                        surface_evidence = self.surface.update(current, candidate)
                    surface_latency_ms = (
                        time.perf_counter() - surface_started
                    ) * 1000.0
                    if surface_evidence is not None:
                        recovered = dict(
                            recovered, surface_evidence=surface_evidence
                        )
                    recovered["runtime_timing_ms"] = dict(
                        recovered.get("runtime_timing_ms") or {},
                        bundletrack_pose_ms=round(pose_latency_ms, 3),
                        feature_observer_ms=0.0,
                        surface_ms=round(surface_latency_ms, 3),
                        total_ms=round(
                            (time.perf_counter() - started) * 1000.0, 3
                        ),
                    )
                    recovered["current_sensor_mask_evidence"] = getattr(self.masks, "current_sensor_mask_evidence", None)
                    recovered["current_visual_mask_evidence"] = getattr(self.masks, "current_visual_mask_evidence", None)
                    recovered["metric_initialized"] = self.metric_initialized
                    return recovered
            if self.shape is None and self._shape_factory is not None and candidate.get("status") == "candidate":
                self.shape = self._shape_factory(current, candidate)
            if self.shape is not None and candidate.get("status") == "candidate":
                candidate = dict(candidate, shape_evidence=self.shape.update(current, candidate))
            feature_started = time.perf_counter()
            candidate = self._attach_measured_task_features(
                current,
                candidate,
                measured_task_features,
            )
            feature_observer_ms = (time.perf_counter() - feature_started) * 1000.0
            surface_started = time.perf_counter()
            if self.surface is not None:
                surface_evidence = self.surface.update(current, candidate)
                candidate = dict(candidate, surface_evidence=surface_evidence)
                if isinstance(surface_evidence, dict) and "measured_task_features" in surface_evidence:
                    candidate["measured_task_features"] = deepcopy(
                        surface_evidence["measured_task_features"]
                    )
            surface_latency_ms = (time.perf_counter() - surface_started) * 1000.0
        except Exception:
            # SAM has already consumed this observation. Retrying the same
            # input would mismatch the provider histories.
            self._failed = True
            raise
        timing = dict(
            candidate.get("runtime_timing_ms")
            if isinstance(candidate.get("runtime_timing_ms"), dict)
            else {}
        )
        timing.update(
            diagnostic.get("runtime_timing_ms")
            if isinstance(diagnostic.get("runtime_timing_ms"), dict)
            else {}
        )
        timing.update(
            sam2_ms=round(sam2_latency_ms, 3),
            pose_ms=round(pose_latency_ms, 3),
            feature_observer_ms=round(feature_observer_ms, 3),
            surface_ms=round(surface_latency_ms, 3),
            total_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
        result = dict(candidate, mask_diagnostic=diagnostic, runtime_timing_ms=timing)
        result["current_sensor_mask_evidence"] = getattr(self.masks, "current_sensor_mask_evidence", None)
        result["current_visual_mask_evidence"] = getattr(self.masks, "current_visual_mask_evidence", None)
        result["metric_initialized"] = self.metric_initialized
        if result.get("status") == "candidate":
            pose = result.get("camera_from_object")
            if isinstance(pose, list):
                matrix = np.asarray(pose, dtype=np.float64)
                if matrix.shape == (4, 4) and np.isfinite(matrix).all():
                    # Native history and surface fusion already consumed this
                    # exact pose. Post-fusion clipping would make the wire pose
                    # disagree with its geometry, including when no recovery
                    # happened. Keep native re-entry measurable on the scoreboard.
                    result = dict(
                        result,
                        pose_source=result.get("pose_source") or "bundletrack",
                    )
                    self._last_camera_from_object = np.array(
                        matrix, dtype=np.float64, copy=True
                    )
                    self._recovery_prior_active = False
                    if self.enable_mask_depth_recovery:
                        anchor_started = time.perf_counter()
                        depth_mm = getattr(current, "depth_mm", None)
                        mask = getattr(current, "mask", None)
                        intrinsics = getattr(current, "intrinsics", None)
                        if (
                            depth_mm is not None
                            and mask is not None
                            and intrinsics is not None
                        ):
                            centroid_estimator = (
                                self.mask_depth_recovery_centroid_estimator
                                if self.mask_depth_recovery_method == "mask_centroid"
                                else DEFAULT_MASK_DEPTH_RECOVERY_CENTROID_ESTIMATOR
                            )
                            centroid, _support, _centroid_diagnostics = (
                                estimate_mask_depth_centroid_camera_with_diagnostics(
                                    depth_m=np.asarray(depth_mm, dtype=np.float64)
                                    / 1000.0,
                                    mask=np.asarray(mask, dtype=bool),
                                    camera_matrix=np.asarray(
                                        intrinsics, dtype=np.float64
                                    ),
                                    minimum_depth_support_pixels=(
                                        self.minimum_recovery_depth_support_pixels
                                    ),
                                    estimator=centroid_estimator,
                                )
                            )
                            self._last_camera_mask_centroid = centroid
                        else:
                            self._last_camera_mask_centroid = None
                        timing = dict(result.get("runtime_timing_ms") or {})
                        timing["recovery_anchor_ms"] = round(
                            (time.perf_counter() - anchor_started) * 1000.0, 3
                        )
                        timing["total_ms"] = round(
                            (time.perf_counter() - started) * 1000.0, 3
                        )
                        result = dict(result, runtime_timing_ms=timing)
        return result

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self.poses.close()
            finally:
                try:
                    self.masks.close()
                finally:
                    try:
                        if self.shape is not None:
                            self.shape.close()
                    finally:
                        if self.surface is not None:
                            self.surface.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
