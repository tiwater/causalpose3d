"""Stateful causal runtime for M1d chest/head fast poses and Top XY updates.

Chest and head metric observations advance the absolute shared-SE(3) fast
runtime.  A lower-rate Top message may refine only the latest already-published
fast pose, may republish only that same frame/timestamp, and never seeds the
next fast frame.  Every physical Top submission remains in accounting even
when it is duplicated, late, malformed, or rejected by a metric camera.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from threading import RLock
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from causalpose3d.surface.causal_surface_track_memory import (
    CausalSurfaceTrackMemory,
    CausalSurfaceTrackMemoryConfig,
    CausalSurfaceTrackMemoryOutput,
)
from causalpose3d.tracking.causal_metric_point_pose import (
    BearingPointObservation,
    CausalAbsoluteMetricPointPoseRuntime,
    PinholeCameraModel,
    SharedSe3PoseReport,
    TableMetricPointObservation,
)
from causalpose3d.tracking.causal_point_tracker import CausalPointTracks
from causalpose3d.tracking.constrained_top_xy_refinement import (
    ConstrainedTopXyRefinementReport,
    constrained_top_xy_refinement,
)


@dataclass(frozen=True)
class FastPosePublication:
    event_sequence: int
    frame_index: int
    timestamp_ns: int
    arrival_timestamp_ns: int
    ready: bool
    reason: str
    pose_table_from_object: np.ndarray
    pose_version: int
    published: bool
    degraded: bool
    evidence_camera_ids: tuple[str, ...]
    factor_report: SharedSe3PoseReport


@dataclass(frozen=True)
class TopPhysicalMessage:
    event_sequence: int
    physical_message_index: int
    physical_timestamp_ns: int
    arrival_timestamp_ns: int
    message_sha256: str
    associated_frame_index: int
    associated_pose_timestamp_ns: int
    feature_anchor_sha256: str
    temporally_compensated_to_associated_pose: bool
    observations: tuple[BearingPointObservation, ...]
    upstream_rejection_reason: str | None = None
    surface_memory_authorized: bool = False


@dataclass(frozen=True)
class TopPoseRefinementResult:
    event_sequence: int
    physical_message_index: int
    physical_timestamp_ns: int
    associated_frame_index: int
    associated_pose_timestamp_ns: int
    accepted: bool
    republished: bool
    reason: str
    pose_table_from_object: np.ndarray
    prior_pose_version: int | None
    result_pose_version: int | None
    association_lag_ms: float | None
    top_correspondence_count: int
    refinement_report: ConstrainedTopXyRefinementReport | None
    total_latency_ms: float


@dataclass(frozen=True)
class TopSurfaceTrackRefinementResult:
    """One surface-memory decision bound to its attempted pose refinement."""

    surface_memory: CausalSurfaceTrackMemoryOutput
    pose_refinement: TopPoseRefinementResult


@dataclass
class _StoredPublication:
    event_sequence: int
    frame_index: int
    timestamp_ns: int
    arrival_timestamp_ns: int
    fast_pose: np.ndarray
    settled_pose: np.ndarray
    pose_version: int
    metric_observations: tuple[TableMetricPointObservation, ...]
    degraded: bool
    evidence_camera_ids: tuple[str, ...]


def _positive_integer(value: object, *, label: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return digest


def _readonly_pose(value: object, *, label: str = "pose") -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{label} must be one finite 4x4 transform")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"{label} must be homogeneous")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6
    ):
        raise ValueError(f"{label} rotation must be proper orthonormal")
    result = pose.copy()
    result.flags.writeable = False
    return result


def _owned_landmarks(value: Mapping[int, np.ndarray]) -> Mapping[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for landmark_id, point in value.items():
        if type(landmark_id) is not int or landmark_id < 0:
            raise ValueError("landmark IDs must be nonnegative integers")
        owned = np.asarray(point, dtype=np.float64)
        if owned.shape != (3,) or not np.isfinite(owned).all():
            raise ValueError("landmarks must be finite 3D points")
        owned = owned.copy()
        owned.flags.writeable = False
        output[landmark_id] = owned
    return MappingProxyType(output)


def _owned_camera_model(value: PinholeCameraModel) -> PinholeCameraModel:
    if not isinstance(value, PinholeCameraModel) or not value.camera_id:
        raise ValueError("camera models require a stable physical camera_id")
    camera_from_table = _readonly_pose(
        value.camera_from_table, label=f"{value.camera_id} camera_from_table"
    )
    camera_matrix = np.asarray(value.camera_matrix, dtype=np.float64)
    if (
        camera_matrix.shape != (3, 3)
        or not np.isfinite(camera_matrix).all()
        or camera_matrix[0, 0] <= 0.0
        or camera_matrix[1, 1] <= 0.0
        or not np.allclose(camera_matrix[2], (0.0, 0.0, 1.0), atol=1e-12)
        or abs(float(np.linalg.det(camera_matrix))) < 1e-12
    ):
        raise ValueError("camera intrinsics must be a nondegenerate pinhole matrix")
    camera_matrix = camera_matrix.copy()
    camera_matrix.flags.writeable = False
    return PinholeCameraModel(
        camera_id=value.camera_id,
        camera_from_table=camera_from_table,
        camera_matrix=camera_matrix,
    )


def _owned_metric_rows(
    rows: Sequence[TableMetricPointObservation],
) -> tuple[TableMetricPointObservation, ...]:
    output = []
    for row in rows:
        if not isinstance(row, TableMetricPointObservation):
            raise TypeError("fast path accepts TableMetricPointObservation rows only")
        point = np.asarray(row.point_table_frame_m, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("metric observation points must be finite 3D values")
        point = point.copy()
        point.flags.writeable = False
        output.append(
            TableMetricPointObservation(
                frame_index=row.frame_index,
                timestamp_ns=row.timestamp_ns,
                camera_id=row.camera_id,
                landmark_id=row.landmark_id,
                point_table_frame_m=point,
                uncertainty_m=row.uncertainty_m,
            )
        )
    return tuple(output)


def _owned_bearing_rows(
    rows: Sequence[BearingPointObservation],
) -> tuple[BearingPointObservation, ...]:
    output = []
    for row in rows:
        if not isinstance(row, BearingPointObservation):
            raise TypeError("Top path accepts BearingPointObservation rows only")
        pixel = np.asarray(row.pixel_xy, dtype=np.float64)
        if pixel.shape != (2,) or not np.isfinite(pixel).all():
            raise ValueError("Top observation pixels must be finite 2D values")
        pixel = pixel.copy()
        pixel.flags.writeable = False
        output.append(
            BearingPointObservation(
                frame_index=row.frame_index,
                timestamp_ns=row.timestamp_ns,
                camera_id=row.camera_id,
                landmark_id=row.landmark_id,
                pixel_xy=pixel,
                uncertainty_px=row.uncertainty_px,
            )
        )
    return tuple(output)


class CausalThreeCameraPoseRuntime:
    """Own one causal chest/head fast state and asynchronous Top refinements."""

    def __init__(
        self,
        *,
        landmarks_object_frame_m: Mapping[int, np.ndarray],
        camera_models: Sequence[PinholeCameraModel],
        initial_table_from_object: np.ndarray,
        metric_camera_ids: Sequence[str] = ("chest", "head"),
        top_camera_id: str = "top",
        history_capacity: int = 8,
        maximum_top_association_lag_ms: float = 120.0,
        minimum_correspondence_count_per_camera: int = 6,
        fast_pose_gain: float = 0.45,
        fast_maximum_translation_innovation_m: float = 0.01,
        fast_maximum_rotation_innovation_deg: float = 5.0,
        fast_maximum_normal_condition_number: float = 1e10,
        fast_maximum_iterations: int = 20,
        minimum_top_residual_improvement_px: float = 0.004,
        maximum_metric_residual_regression_m: float = 0.0,
        maximum_top_translation_m: float = 0.01,
        maximum_top_normal_condition_number: float = 1e8,
        top_huber_delta_px: float = 3.0,
        top_damping: float = 1e-6,
        top_line_search_scales: Sequence[float] = (
            1.0,
            0.5,
            0.25,
            0.125,
            0.0625,
        ),
        surface_track_memory_config: CausalSurfaceTrackMemoryConfig | None = None,
    ) -> None:
        metric_ids = tuple(metric_camera_ids)
        if (
            len(metric_ids) != 2
            or len(set(metric_ids)) != 2
            or any(not isinstance(value, str) or not value for value in metric_ids)
        ):
            raise ValueError("M1d requires exactly two distinct metric camera IDs")
        if not isinstance(top_camera_id, str) or not top_camera_id:
            raise ValueError("Top camera ID must be nonempty")
        if top_camera_id in metric_ids:
            raise ValueError("Top and metric camera identities must be distinct")
        if type(history_capacity) is not int or history_capacity < 3:
            raise ValueError("publication history capacity must be at least three")
        if (
            not math.isfinite(maximum_top_association_lag_ms)
            or maximum_top_association_lag_ms <= 0.0
        ):
            raise ValueError("maximum Top association lag must be positive")

        landmarks = _owned_landmarks(landmarks_object_frame_m)
        models = tuple(_owned_camera_model(model) for model in camera_models)
        model_by_id = {model.camera_id: model for model in models}
        required_ids = set(metric_ids) | {top_camera_id}
        if len(model_by_id) != len(models) or set(model_by_id) != required_ids:
            raise ValueError("runtime requires exactly one model for chest, head, and Top")
        self._landmarks = landmarks
        self._metric_camera_ids = metric_ids
        self._top_camera_id = top_camera_id
        self._top_camera_model = model_by_id[top_camera_id]
        self._minimum_correspondence_count = minimum_correspondence_count_per_camera
        self._history_capacity = history_capacity
        self._maximum_top_association_lag_ns = int(
            maximum_top_association_lag_ms * 1_000_000.0
        )
        self._top_options = {
            "required_metric_camera_ids": metric_ids,
            "minimum_correspondence_count_per_camera": (
                minimum_correspondence_count_per_camera
            ),
            "minimum_top_residual_improvement_px": (
                minimum_top_residual_improvement_px
            ),
            "maximum_metric_residual_regression_m": (
                maximum_metric_residual_regression_m
            ),
            "maximum_translation_m": maximum_top_translation_m,
            "maximum_normal_condition_number": (
                maximum_top_normal_condition_number
            ),
            "huber_delta_px": top_huber_delta_px,
            "damping": top_damping,
            "line_search_scales": tuple(top_line_search_scales),
        }
        self._surface_track_memory = CausalSurfaceTrackMemory(
            surface_track_memory_config or CausalSurfaceTrackMemoryConfig()
        )
        self._fast_runtime = CausalAbsoluteMetricPointPoseRuntime(
            landmarks_object_frame_m=landmarks,
            camera_models=models,
            initial_table_from_object=_readonly_pose(
                initial_table_from_object, label="initial_table_from_object"
            ),
            minimum_physical_camera_count=1,
            minimum_correspondence_count_per_camera=(
                minimum_correspondence_count_per_camera
            ),
            pose_gain=fast_pose_gain,
            maximum_translation_innovation_m=(
                fast_maximum_translation_innovation_m
            ),
            maximum_rotation_innovation_deg=(
                fast_maximum_rotation_innovation_deg
            ),
            maximum_normal_condition_number=(
                fast_maximum_normal_condition_number
            ),
            maximum_iterations=fast_maximum_iterations,
        )
        constrained_top_xy_refinement(
            frame_index=0,
            timestamp_ns=0,
            prior_table_from_object=initial_table_from_object,
            landmarks_object_frame_m=landmarks,
            top_camera_model=self._top_camera_model,
            top_observations=(),
            metric_observations=(),
            **self._top_options,
        )
        self._history: OrderedDict[
            tuple[int, int], _StoredPublication
        ] = OrderedDict()
        self._lock = RLock()
        self._last_event_sequence: int | None = None
        self._last_top_message_index: int | None = None
        self._last_top_physical_timestamp_ns: int | None = None
        self._seen_top_identities: set[tuple[int, int, str]] = set()
        self._feature_anchor_sha256: str | None = None
        self._counts: Counter[str] = Counter()
        self._rejections: Counter[str] = Counter()

    @property
    def fast_state(self) -> np.ndarray:
        return _readonly_pose(self._fast_runtime.state, label="fast_state")

    def _accept_event_sequence(self, value: int) -> None:
        sequence = _positive_integer(value, label="event_sequence", allow_zero=True)
        if self._last_event_sequence is not None and sequence <= self._last_event_sequence:
            raise ValueError("event_sequence must be globally strictly increasing")
        self._last_event_sequence = sequence

    def publish_fast(
        self,
        *,
        event_sequence: int,
        frame_index: int,
        timestamp_ns: int,
        arrival_timestamp_ns: int,
        metric_observations: Sequence[TableMetricPointObservation],
    ) -> FastPosePublication:
        with self._lock:
            self._accept_event_sequence(event_sequence)
            _positive_integer(frame_index, label="frame_index", allow_zero=True)
            _positive_integer(timestamp_ns, label="timestamp_ns")
            _positive_integer(arrival_timestamp_ns, label="arrival_timestamp_ns")
            rows = _owned_metric_rows(metric_observations)
            if any(row.camera_id not in self._metric_camera_ids for row in rows):
                raise ValueError("fast path accepts only the two metric cameras")
            self._counts["fast_input_count"] += 1
            pose, factor_report = self._fast_runtime.step(
                frame_index=frame_index,
                timestamp_ns=timestamp_ns,
                metric_observations=rows,
                bearing_observations=(),
            )
            readonly_pose = _readonly_pose(pose)
            published = bool(factor_report.ready)
            evidence_camera_ids = tuple(factor_report.physical_camera_ids)
            degraded = published and set(evidence_camera_ids) != set(
                self._metric_camera_ids
            )
            publication_reason = (
                "degraded_single_metric_camera_update"
                if degraded
                else factor_report.reason
            )
            if published:
                self._counts["fast_publication_count"] += 1
                if degraded:
                    self._counts["degraded_fast_publication_count"] += 1
                key = (frame_index, timestamp_ns)
                self._history[key] = _StoredPublication(
                    event_sequence=event_sequence,
                    frame_index=frame_index,
                    timestamp_ns=timestamp_ns,
                    arrival_timestamp_ns=arrival_timestamp_ns,
                    fast_pose=readonly_pose,
                    settled_pose=readonly_pose,
                    pose_version=0,
                    metric_observations=rows,
                    degraded=degraded,
                    evidence_camera_ids=evidence_camera_ids,
                )
                self._history.move_to_end(key)
                while len(self._history) > self._history_capacity:
                    self._history.popitem(last=False)
            else:
                self._rejections[f"fast:{factor_report.reason}"] += 1
            return FastPosePublication(
                event_sequence=event_sequence,
                frame_index=frame_index,
                timestamp_ns=timestamp_ns,
                arrival_timestamp_ns=arrival_timestamp_ns,
                ready=factor_report.ready,
                reason=publication_reason,
                pose_table_from_object=readonly_pose,
                pose_version=0,
                published=published,
                degraded=degraded,
                evidence_camera_ids=evidence_camera_ids,
                factor_report=factor_report,
            )

    def _fallback_pose(self, publication: _StoredPublication | None) -> np.ndarray:
        if publication is not None:
            return publication.settled_pose
        if self._history:
            return next(reversed(self._history.values())).settled_pose
        return self.fast_state

    def _reject_top(
        self,
        *,
        message: TopPhysicalMessage,
        reason: str,
        started_ns: int,
        publication: _StoredPublication | None = None,
        lag_ms: float | None = None,
        correspondence_count: int = 0,
        refinement_report: ConstrainedTopXyRefinementReport | None = None,
    ) -> TopPoseRefinementResult:
        self._rejections[f"top:{reason}"] += 1
        pose = self._fallback_pose(publication)
        version = publication.pose_version if publication is not None else None
        return TopPoseRefinementResult(
            event_sequence=message.event_sequence,
            physical_message_index=message.physical_message_index,
            physical_timestamp_ns=message.physical_timestamp_ns,
            associated_frame_index=message.associated_frame_index,
            associated_pose_timestamp_ns=message.associated_pose_timestamp_ns,
            accepted=False,
            republished=False,
            reason=reason,
            pose_table_from_object=_readonly_pose(pose),
            prior_pose_version=version,
            result_pose_version=version,
            association_lag_ms=lag_ms,
            top_correspondence_count=correspondence_count,
            refinement_report=refinement_report,
            total_latency_ms=(time.perf_counter_ns() - started_ns) / 1_000_000.0,
        )

    def submit_top(self, message: TopPhysicalMessage) -> TopPoseRefinementResult:
        started_ns = time.perf_counter_ns()
        if not isinstance(message, TopPhysicalMessage):
            raise TypeError("submit_top requires TopPhysicalMessage")
        with self._lock:
            self._counts["physical_top_message_count"] += 1
            try:
                self._accept_event_sequence(message.event_sequence)
                _positive_integer(
                    message.physical_message_index,
                    label="physical_message_index",
                    allow_zero=True,
                )
                _positive_integer(
                    message.physical_timestamp_ns,
                    label="physical_timestamp_ns",
                )
                _positive_integer(
                    message.arrival_timestamp_ns,
                    label="arrival_timestamp_ns",
                )
                _positive_integer(
                    message.associated_frame_index,
                    label="associated_frame_index",
                    allow_zero=True,
                )
                _positive_integer(
                    message.associated_pose_timestamp_ns,
                    label="associated_pose_timestamp_ns",
                )
                message_hash = _sha256(
                    message.message_sha256, label="Top physical message hash"
                )
                anchor_hash = _sha256(
                    message.feature_anchor_sha256, label="feature anchor hash"
                )
                if type(message.surface_memory_authorized) is not bool:
                    raise ValueError(
                        "surface_memory_authorized must be boolean"
                    )
            except ValueError as error:
                return self._reject_top(
                    message=message,
                    reason=f"invalid_top_envelope:{error}",
                    started_ns=started_ns,
                )

            identity = (
                message.physical_message_index,
                message.physical_timestamp_ns,
                message_hash,
            )
            if (
                identity in self._seen_top_identities
                or (
                    self._last_top_message_index is not None
                    and message.physical_message_index <= self._last_top_message_index
                )
                or (
                    self._last_top_physical_timestamp_ns is not None
                    and message.physical_timestamp_ns
                    <= self._last_top_physical_timestamp_ns
                )
            ):
                return self._reject_top(
                    message=message,
                    reason="duplicate_or_out_of_order_physical_top_message",
                    started_ns=started_ns,
                    publication=self._history.get(
                        (
                            message.associated_frame_index,
                            message.associated_pose_timestamp_ns,
                        )
                    ),
                )
            self._seen_top_identities.add(identity)
            self._last_top_message_index = message.physical_message_index
            self._last_top_physical_timestamp_ns = message.physical_timestamp_ns

            if (
                self._feature_anchor_sha256 is not None
                and anchor_hash != self._feature_anchor_sha256
            ):
                return self._reject_top(
                    message=message,
                    reason="feature_anchor_hash_changed",
                    started_ns=started_ns,
                )
            key = (
                message.associated_frame_index,
                message.associated_pose_timestamp_ns,
            )
            publication = self._history.get(key)
            if publication is None:
                return self._reject_top(
                    message=message,
                    reason="associated_fast_publication_missing_or_evicted",
                    started_ns=started_ns,
                )
            latest = next(reversed(self._history.values()))
            if publication is not latest:
                return self._reject_top(
                    message=message,
                    reason="association_is_not_latest_published_fast_pose",
                    started_ns=started_ns,
                    publication=publication,
                )
            lag_ns = message.arrival_timestamp_ns - publication.arrival_timestamp_ns
            lag_ms = lag_ns / 1_000_000.0
            if lag_ns < 0:
                return self._reject_top(
                    message=message,
                    reason="top_association_reads_future_fast_publication",
                    started_ns=started_ns,
                    publication=publication,
                    lag_ms=lag_ms,
                )
            if lag_ns > self._maximum_top_association_lag_ns:
                return self._reject_top(
                    message=message,
                    reason="top_association_lag_exceeded",
                    started_ns=started_ns,
                    publication=publication,
                    lag_ms=lag_ms,
                )
            if message.upstream_rejection_reason is not None:
                reason = str(message.upstream_rejection_reason).strip()
                if not reason or len(reason) > 160:
                    reason = "invalid_upstream_top_admission_reason"
                return self._reject_top(
                    message=message,
                    reason=reason,
                    started_ns=started_ns,
                    publication=publication,
                    lag_ms=lag_ms,
                )
            if not message.temporally_compensated_to_associated_pose:
                return self._reject_top(
                    message=message,
                    reason="top_observations_not_compensated_to_associated_pose",
                    started_ns=started_ns,
                    publication=publication,
                    lag_ms=lag_ms,
                )
            try:
                rows = _owned_bearing_rows(message.observations)
            except (TypeError, ValueError) as error:
                return self._reject_top(
                    message=message,
                    reason=f"invalid_top_observations:{error}",
                    started_ns=started_ns,
                    publication=publication,
                    lag_ms=lag_ms,
                )
            if any(
                row.frame_index != publication.frame_index
                or row.timestamp_ns != publication.timestamp_ns
                or row.camera_id != self._top_camera_id
                for row in rows
            ):
                return self._reject_top(
                    message=message,
                    reason="top_observation_identity_differs_from_associated_pose",
                    started_ns=started_ns,
                    publication=publication,
                    lag_ms=lag_ms,
                    correspondence_count=len(rows),
                )
            if len(rows) < self._minimum_correspondence_count:
                return self._reject_top(
                    message=message,
                    reason="insufficient_top_correspondences",
                    started_ns=started_ns,
                    publication=publication,
                    lag_ms=lag_ms,
                    correspondence_count=len(rows),
                )

            self._counts["top_attempted_message_count"] += 1
            try:
                top_options = dict(self._top_options)
                if publication.degraded and message.surface_memory_authorized:
                    present_metric_camera_ids = tuple(
                        camera_id
                        for camera_id in self._metric_camera_ids
                        if sum(
                            row.camera_id == camera_id
                            for row in publication.metric_observations
                        )
                        >= self._minimum_correspondence_count
                    )
                    if len(present_metric_camera_ids) == 1:
                        top_options["required_metric_camera_ids"] = (
                            present_metric_camera_ids
                        )
                candidate, refinement = constrained_top_xy_refinement(
                    frame_index=publication.frame_index,
                    timestamp_ns=publication.timestamp_ns,
                    prior_table_from_object=publication.fast_pose,
                    landmarks_object_frame_m=self._landmarks,
                    top_camera_model=self._top_camera_model,
                    top_observations=rows,
                    metric_observations=publication.metric_observations,
                    **top_options,
                )
            except (TypeError, ValueError, FloatingPointError) as error:
                return self._reject_top(
                    message=message,
                    reason=f"top_factor_input_invalid:{type(error).__name__}",
                    started_ns=started_ns,
                    publication=publication,
                    lag_ms=lag_ms,
                    correspondence_count=len(rows),
                )
            self._counts["top_evidence_message_count"] += 1
            if self._feature_anchor_sha256 is None:
                self._feature_anchor_sha256 = anchor_hash
            if refinement.jacobian_rank == 2 and refinement.raw_delta_table_xy_m is not None:
                self._counts["top_solver_ready_message_count"] += 1
            if not refinement.ready:
                return self._reject_top(
                    message=message,
                    reason=refinement.reason,
                    started_ns=started_ns,
                    publication=publication,
                    lag_ms=lag_ms,
                    correspondence_count=len(rows),
                    refinement_report=refinement,
                )

            prior_version = publication.pose_version
            publication.settled_pose = _readonly_pose(candidate)
            publication.pose_version += 1
            self._counts["top_accepted_update_count"] += 1
            self._counts["top_same_timestamp_republication_count"] += 1
            return TopPoseRefinementResult(
                event_sequence=message.event_sequence,
                physical_message_index=message.physical_message_index,
                physical_timestamp_ns=message.physical_timestamp_ns,
                associated_frame_index=publication.frame_index,
                associated_pose_timestamp_ns=publication.timestamp_ns,
                accepted=True,
                republished=True,
                reason=refinement.reason,
                pose_table_from_object=publication.settled_pose,
                prior_pose_version=prior_version,
                result_pose_version=publication.pose_version,
                association_lag_ms=lag_ms,
                top_correspondence_count=len(rows),
                refinement_report=refinement,
                total_latency_ms=(time.perf_counter_ns() - started_ns)
                / 1_000_000.0,
            )

    def submit_top_surface_tracks(
        self,
        *,
        event_sequence: int,
        physical_message_index: int,
        physical_timestamp_ns: int,
        arrival_timestamp_ns: int,
        message_sha256: str,
        associated_frame_index: int,
        associated_pose_timestamp_ns: int,
        persistent_tracks: CausalPointTracks,
        verifier_tracks: CausalPointTracks | None,
        temporally_compensated_to_associated_pose: bool,
    ) -> TopSurfaceTrackRefinementResult:
        with self._lock:
            memory = self._surface_track_memory.step(
                persistent_tracks, verifier_tracks
            )
            self._counts[f"top_surface_memory:{memory.status}"] += 1
            upstream_reason: str | None = None
            if persistent_tracks.camera_id != self._top_camera_id:
                upstream_reason = "surface_memory_camera_is_not_top"
            elif persistent_tracks.timestamp_ns != associated_pose_timestamp_ns:
                upstream_reason = "surface_memory_not_expressed_at_associated_pose_time"
            elif not temporally_compensated_to_associated_pose:
                upstream_reason = "surface_memory_not_temporally_compensated"
            elif not memory.correction_authorized:
                upstream_reason = f"surface_memory_{memory.status}"

            anchor_payload = {
                "schema": "ticos.m1d-top-surface-anchor-lineage.v1",
                "camera_id": persistent_tracks.camera_id,
                "model_sha256": persistent_tracks.model_sha256,
                "query_ids": persistent_tracks.query_ids.tolist(),
            }
            anchor_hash = hashlib.sha256(
                json.dumps(
                    anchor_payload, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            observations: tuple[BearingPointObservation, ...] = ()
            if upstream_reason is None:
                observations = tuple(
                    BearingPointObservation(
                        frame_index=associated_frame_index,
                        timestamp_ns=associated_pose_timestamp_ns,
                        camera_id=self._top_camera_id,
                        landmark_id=int(landmark_id),
                        pixel_xy=position,
                        uncertainty_px=max(0.5, 1.0 / float(confidence)),
                    )
                    for landmark_id, position, confidence in zip(
                        memory.landmark_ids,
                        memory.positions_xy,
                        memory.confidence,
                        strict=True,
                    )
                )
            result = self.submit_top(
                TopPhysicalMessage(
                    event_sequence=event_sequence,
                    physical_message_index=physical_message_index,
                    physical_timestamp_ns=physical_timestamp_ns,
                    arrival_timestamp_ns=arrival_timestamp_ns,
                    message_sha256=message_sha256,
                    associated_frame_index=associated_frame_index,
                    associated_pose_timestamp_ns=associated_pose_timestamp_ns,
                    feature_anchor_sha256=anchor_hash,
                    temporally_compensated_to_associated_pose=(
                        temporally_compensated_to_associated_pose
                    ),
                    observations=observations,
                    upstream_rejection_reason=upstream_reason,
                    surface_memory_authorized=memory.correction_authorized,
                )
            )
            return TopSurfaceTrackRefinementResult(
                surface_memory=memory,
                pose_refinement=result,
            )

    def latest_publication(self) -> Mapping[str, Any] | None:
        with self._lock:
            if not self._history:
                return None
            publication = next(reversed(self._history.values()))
            return MappingProxyType(
                {
                    "event_sequence": publication.event_sequence,
                    "frame_index": publication.frame_index,
                    "timestamp_ns": publication.timestamp_ns,
                    "arrival_timestamp_ns": publication.arrival_timestamp_ns,
                    "pose_version": publication.pose_version,
                    "degraded": publication.degraded,
                    "evidence_camera_ids": publication.evidence_camera_ids,
                    "fast_pose_table_from_object": _readonly_pose(
                        publication.fast_pose
                    ),
                    "settled_pose_table_from_object": _readonly_pose(
                        publication.settled_pose
                    ),
                }
            )

    def diagnostics(self) -> Mapping[str, Any]:
        with self._lock:
            counts = dict(self._counts)
            physical = counts.get("physical_top_message_count", 0)
            evidence = counts.get("top_evidence_message_count", 0)
            accepted = counts.get("top_accepted_update_count", 0)
            republications = counts.get(
                "top_same_timestamp_republication_count", 0
            )
            if accepted != republications:
                raise RuntimeError("accepted Top updates and republications diverged")
            lineage = {
                "feature_anchor_sha256": self._feature_anchor_sha256,
                "fast_path_uses_top_refined_state": False,
                "top_update_seeds_next_fast_frame": False,
                "metric_camera_ids": list(self._metric_camera_ids),
                "top_camera_id": self._top_camera_id,
                "surface_track_memory_guard_available": True,
                "short_track_can_supply_pose_pixels": False,
            }
            lineage_hash = hashlib.sha256(
                json.dumps(
                    lineage,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            return MappingProxyType(
                {
                    "schema": "ticos.m1d-causal-three-camera-pose-runtime-diagnostics.v1",
                    "counts": MappingProxyType(counts),
                    "rejection_counts": MappingProxyType(dict(self._rejections)),
                    "top_evidence_physical_message_ratio": (
                        evidence / physical if physical else 0.0
                    ),
                    "top_accepted_physical_message_ratio": (
                        accepted / physical if physical else 0.0
                    ),
                    "accepted_equals_republication": accepted == republications,
                    "history_size": len(self._history),
                    "history_capacity": self._history_capacity,
                    "last_event_sequence": self._last_event_sequence,
                    "last_top_message_index": self._last_top_message_index,
                    "last_top_physical_timestamp_ns": (
                        self._last_top_physical_timestamp_ns
                    ),
                    "lineage": MappingProxyType(lineage),
                    "lineage_sha256": lineage_hash,
                }
            )
