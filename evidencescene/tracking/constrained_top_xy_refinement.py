"""Causal Top-camera XY refinement constrained by two metric cameras.

The low-rate Top camera is allowed to refine only table X/Y at an already
published timestamp.  Its update direction is computed directly from the Top
reprojection Jacobian; no teacher trajectory or fitted sign/calibration matrix
is used.  Chest and head RGB-D residuals are independent feasibility
constraints, so a dense camera cannot hide regression in the other one.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from evidencescene.tracking.causal_metric_point_pose import (
    BearingPointObservation,
    PinholeCameraModel,
    TableMetricPointObservation,
)


@dataclass(frozen=True)
class ConstrainedTopXyRefinementReport:
    ready: bool
    reason: str
    frame_index: int
    timestamp_ns: int
    top_camera_id: str
    metric_camera_ids: tuple[str, ...]
    top_correspondence_count: int
    metric_counts: Mapping[str, int]
    jacobian_rank: int | None
    normal_condition_number: float | None
    raw_delta_table_xy_m: tuple[float, float] | None
    bounded_delta_table_xy_m: tuple[float, float] | None
    accepted_delta_table_xy_m: tuple[float, float] | None
    selected_line_search_scale: float | None
    line_search_candidate_count: int
    top_residual_before_p50_px: float | None
    top_residual_after_p50_px: float | None
    top_residual_improvement_p50_px: float | None
    metric_residual_before_p50_m: Mapping[str, float]
    metric_residual_after_p50_m: Mapping[str, float]
    factor_latency_ms: float


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {shape} array")
    return result


def _validated_pose(value: object) -> np.ndarray:
    pose = _finite_array(value, (4, 4), "prior_table_from_object")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError("prior_table_from_object must be homogeneous")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("prior_table_from_object rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError("prior_table_from_object rotation must be proper")
    return pose


def _validated_landmarks(
    landmarks: Mapping[int, np.ndarray],
) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for landmark_id, point in landmarks.items():
        if type(landmark_id) is not int or landmark_id < 0:
            raise ValueError("landmark IDs must be nonnegative integers")
        output[landmark_id] = _finite_array(point, (3,), "landmark")
    return output


def _validated_camera_model(value: PinholeCameraModel) -> PinholeCameraModel:
    if not isinstance(value, PinholeCameraModel) or not value.camera_id:
        raise ValueError("Top camera needs a stable nonempty physical identity")
    camera_from_table = _finite_array(
        value.camera_from_table, (4, 4), "top camera_from_table"
    ).copy()
    if not np.allclose(camera_from_table[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError("top camera_from_table must be homogeneous")
    rotation = camera_from_table[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6)
    ):
        raise ValueError("top camera_from_table rotation must be proper orthonormal")
    camera_matrix = _finite_array(
        value.camera_matrix, (3, 3), "top camera_matrix"
    ).copy()
    if (
        camera_matrix[0, 0] <= 0.0
        or camera_matrix[1, 1] <= 0.0
        or not np.allclose(camera_matrix[2], (0.0, 0.0, 1.0), atol=1e-12)
        or abs(float(np.linalg.det(camera_matrix))) < 1e-12
    ):
        raise ValueError("top camera intrinsics must be a nondegenerate pinhole matrix")
    camera_from_table.flags.writeable = False
    camera_matrix.flags.writeable = False
    return PinholeCameraModel(value.camera_id, camera_from_table, camera_matrix)


def _table_point(
    landmark: np.ndarray,
    table_from_object: np.ndarray,
) -> np.ndarray:
    return landmark @ table_from_object[:3, :3].T + table_from_object[:3, 3]


def _project_with_xy_jacobian(
    point_table: np.ndarray,
    camera: PinholeCameraModel,
) -> tuple[np.ndarray, np.ndarray]:
    camera_from_table = _finite_array(
        camera.camera_from_table, (4, 4), "top camera_from_table"
    )
    camera_matrix = _finite_array(
        camera.camera_matrix, (3, 3), "top camera_matrix"
    )
    point_camera = (
        point_table @ camera_from_table[:3, :3].T
        + camera_from_table[:3, 3]
    )
    x, y, z = point_camera
    if z <= 1e-6:
        raise FloatingPointError("Top landmark projects behind the camera")
    normalized = np.asarray((x / z, y / z), dtype=np.float64)
    pixel = camera_matrix[:2, :2] @ normalized + camera_matrix[:2, 2]
    normalized_jacobian = np.asarray(
        ((1.0 / z, 0.0, -x / (z * z)), (0.0, 1.0 / z, -y / (z * z))),
        dtype=np.float64,
    )
    table_xy_to_camera = camera_from_table[:3, :3][:, :2]
    jacobian = (
        camera_matrix[:2, :2]
        @ normalized_jacobian
        @ table_xy_to_camera
    )
    return pixel, jacobian


def _metric_residuals_by_camera(
    *,
    pose: np.ndarray,
    landmarks: Mapping[int, np.ndarray],
    observations: Sequence[TableMetricPointObservation],
    camera_ids: tuple[str, ...],
) -> dict[str, float]:
    values = {camera_id: [] for camera_id in camera_ids}
    for row in observations:
        residual = float(
            np.linalg.norm(
                _table_point(landmarks[row.landmark_id], pose)
                - _finite_array(
                    row.point_table_frame_m,
                    (3,),
                    "metric point_table_frame_m",
                )
            )
        )
        values[row.camera_id].append(residual)
    return {
        camera_id: float(np.quantile(camera_values, 0.50))
        for camera_id, camera_values in values.items()
    }


def constrained_top_xy_refinement(
    *,
    frame_index: int,
    timestamp_ns: int,
    prior_table_from_object: np.ndarray,
    landmarks_object_frame_m: Mapping[int, np.ndarray],
    top_camera_model: PinholeCameraModel,
    top_observations: Sequence[BearingPointObservation],
    metric_observations: Sequence[TableMetricPointObservation],
    required_metric_camera_ids: Sequence[str] = ("chest", "head"),
    minimum_correspondence_count_per_camera: int = 6,
    minimum_top_residual_improvement_px: float = 0.004,
    maximum_metric_residual_regression_m: float = 0.0,
    maximum_translation_m: float = 0.01,
    maximum_normal_condition_number: float = 1e8,
    huber_delta_px: float = 3.0,
    damping: float = 1e-6,
    line_search_scales: Sequence[float] = (1.0, 0.5, 0.25, 0.125, 0.0625),
) -> tuple[np.ndarray, ConstrainedTopXyRefinementReport]:
    """Return a same-timestamp XY refinement or the unchanged prior.

    Every accepted candidate must improve Top p50 by the frozen margin and
    must not regress either required metric camera's p50.  The largest frozen
    line-search scale satisfying both constraints wins.
    """

    started = time.perf_counter_ns()
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("frame_index must be a nonnegative integer")
    if type(timestamp_ns) is not int or timestamp_ns < 0:
        raise ValueError("timestamp_ns must be a nonnegative integer")
    if minimum_correspondence_count_per_camera < 3:
        raise ValueError("each camera requires at least three correspondences")
    positive = (
        minimum_top_residual_improvement_px,
        maximum_translation_m,
        maximum_normal_condition_number,
        huber_delta_px,
        damping,
    )
    if not all(math.isfinite(float(value)) and value > 0.0 for value in positive):
        raise ValueError("refinement limits must be finite and positive")
    if (
        not math.isfinite(float(maximum_metric_residual_regression_m))
        or maximum_metric_residual_regression_m < 0.0
    ):
        raise ValueError("metric residual regression limit must be nonnegative")
    scales = tuple(float(value) for value in line_search_scales)
    if (
        not scales
        or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in scales)
        or any(left <= right for left, right in zip(scales, scales[1:]))
    ):
        raise ValueError("line search scales must be strictly decreasing in (0,1]")
    metric_camera_ids = tuple(required_metric_camera_ids)
    if (
        len(metric_camera_ids) < 1
        or len(metric_camera_ids) != len(set(metric_camera_ids))
        or any(not isinstance(value, str) or not value for value in metric_camera_ids)
    ):
        raise ValueError("at least one unique metric camera ID is required")
    top_camera = _validated_camera_model(top_camera_model)
    if top_camera.camera_id in metric_camera_ids:
        raise ValueError("Top and metric camera identities must be distinct")

    prior = _validated_pose(prior_table_from_object).copy()
    landmarks = _validated_landmarks(landmarks_object_frame_m)
    top_rows = list(top_observations)
    metric_rows = list(metric_observations)
    all_rows = [*top_rows, *metric_rows]
    if any(
        row.frame_index != frame_index or row.timestamp_ns != timestamp_ns
        for row in all_rows
    ):
        raise ValueError("all observations must belong to the same published frame")
    if any(row.camera_id != top_camera.camera_id for row in top_rows):
        raise ValueError("bearing observations must belong to the Top camera")
    if any(row.camera_id not in metric_camera_ids for row in metric_rows):
        raise ValueError("metric observation has an unregistered physical camera")
    if any(row.landmark_id not in landmarks for row in all_rows):
        raise ValueError("observation references an unknown landmark")
    identities = [(row.camera_id, row.landmark_id) for row in all_rows]
    if len(identities) != len(set(identities)):
        raise ValueError("one physical scoring identity may appear only once")
    for row in top_rows:
        if (
            not math.isfinite(float(row.uncertainty_px))
            or row.uncertainty_px <= 0.0
        ):
            raise ValueError("Top uncertainty must be positive")
        _finite_array(row.pixel_xy, (2,), "Top pixel_xy")
    for row in metric_rows:
        if (
            not math.isfinite(float(row.uncertainty_m))
            or row.uncertainty_m <= 0.0
        ):
            raise ValueError("metric uncertainty must be positive")

    metric_counts = {
        camera_id: sum(row.camera_id == camera_id for row in metric_rows)
        for camera_id in metric_camera_ids
    }

    def report(
        *,
        ready: bool,
        reason: str,
        jacobian_rank: int | None = None,
        condition: float | None = None,
        raw_delta: np.ndarray | None = None,
        bounded_delta: np.ndarray | None = None,
        accepted_delta: np.ndarray | None = None,
        selected_scale: float | None = None,
        top_before: float | None = None,
        top_after: float | None = None,
        metric_before: Mapping[str, float] | None = None,
        metric_after: Mapping[str, float] | None = None,
    ) -> ConstrainedTopXyRefinementReport:
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        return ConstrainedTopXyRefinementReport(
            ready=ready,
            reason=reason,
            frame_index=frame_index,
            timestamp_ns=timestamp_ns,
            top_camera_id=top_camera.camera_id,
            metric_camera_ids=metric_camera_ids,
            top_correspondence_count=len(top_rows),
            metric_counts=MappingProxyType(dict(metric_counts)),
            jacobian_rank=jacobian_rank,
            normal_condition_number=condition,
            raw_delta_table_xy_m=(
                tuple(float(value) for value in raw_delta) if raw_delta is not None else None
            ),
            bounded_delta_table_xy_m=(
                tuple(float(value) for value in bounded_delta)
                if bounded_delta is not None
                else None
            ),
            accepted_delta_table_xy_m=(
                tuple(float(value) for value in accepted_delta)
                if accepted_delta is not None
                else None
            ),
            selected_line_search_scale=selected_scale,
            line_search_candidate_count=len(scales),
            top_residual_before_p50_px=top_before,
            top_residual_after_p50_px=top_after,
            top_residual_improvement_p50_px=(
                top_before - top_after
                if top_before is not None and top_after is not None
                else None
            ),
            metric_residual_before_p50_m=MappingProxyType(dict(metric_before or {})),
            metric_residual_after_p50_m=MappingProxyType(dict(metric_after or {})),
            factor_latency_ms=latency_ms,
        )

    if len(top_rows) < minimum_correspondence_count_per_camera:
        return prior, report(ready=False, reason="insufficient_top_correspondences")
    if any(
        count < minimum_correspondence_count_per_camera
        for count in metric_counts.values()
    ):
        return prior, report(ready=False, reason="insufficient_metric_camera_correspondences")

    residual_rows: list[np.ndarray] = []
    jacobian_rows: list[np.ndarray] = []
    top_norms: list[float] = []
    for row in top_rows:
        pixel, jacobian = _project_with_xy_jacobian(
            _table_point(landmarks[row.landmark_id], prior),
            top_camera,
        )
        residual = pixel - _finite_array(row.pixel_xy, (2,), "Top pixel_xy")
        norm = float(np.linalg.norm(residual))
        robust_weight = min(1.0, huber_delta_px / max(norm, 1e-12))
        uncertainty_weight = 1.0 / float(row.uncertainty_px)
        weight = math.sqrt(robust_weight) * uncertainty_weight
        residual_rows.append(residual * weight)
        jacobian_rows.append(jacobian * weight)
        top_norms.append(norm)
    residual_vector = np.concatenate(residual_rows)
    jacobian_matrix = np.concatenate(jacobian_rows, axis=0)
    normal = jacobian_matrix.T @ jacobian_matrix
    rank = int(np.linalg.matrix_rank(normal))
    condition = float(np.linalg.cond(normal)) if rank == 2 else math.inf
    top_before = float(np.quantile(top_norms, 0.50))
    metric_before = _metric_residuals_by_camera(
        pose=prior,
        landmarks=landmarks,
        observations=metric_rows,
        camera_ids=metric_camera_ids,
    )
    if rank < 2 or not math.isfinite(condition) or condition > maximum_normal_condition_number:
        return prior, report(
            ready=False,
            reason="ill_conditioned_top_xy_jacobian",
            jacobian_rank=rank,
            condition=condition,
            top_before=top_before,
            metric_before=metric_before,
        )
    raw_delta = -np.linalg.solve(
        normal + np.eye(2, dtype=np.float64) * damping,
        jacobian_matrix.T @ residual_vector,
    )
    raw_norm = float(np.linalg.norm(raw_delta))
    bounded_delta = raw_delta.copy()
    if raw_norm > maximum_translation_m:
        bounded_delta *= maximum_translation_m / raw_norm

    for scale in scales:
        accepted_delta = bounded_delta * scale
        candidate = prior.copy()
        candidate[:2, 3] += accepted_delta
        candidate_top_norms = [
            float(
                np.linalg.norm(
                    _project_with_xy_jacobian(
                        _table_point(landmarks[row.landmark_id], candidate),
                        top_camera,
                    )[0]
                    - _finite_array(row.pixel_xy, (2,), "Top pixel_xy")
                )
            )
            for row in top_rows
        ]
        top_after = float(np.quantile(candidate_top_norms, 0.50))
        metric_after = _metric_residuals_by_camera(
            pose=candidate,
            landmarks=landmarks,
            observations=metric_rows,
            camera_ids=metric_camera_ids,
        )
        top_pass = (
            top_before - top_after >= minimum_top_residual_improvement_px - 1e-12
        )
        metric_pass = all(
            metric_after[camera_id]
            <= metric_before[camera_id]
            + maximum_metric_residual_regression_m
            + 1e-12
            for camera_id in metric_camera_ids
        )
        if top_pass and metric_pass:
            return candidate, report(
                ready=True,
                reason="accepted_constrained_top_xy_update",
                jacobian_rank=rank,
                condition=condition,
                raw_delta=raw_delta,
                bounded_delta=bounded_delta,
                accepted_delta=accepted_delta,
                selected_scale=scale,
                top_before=top_before,
                top_after=top_after,
                metric_before=metric_before,
                metric_after=metric_after,
            )

    return prior, report(
        ready=False,
        reason="no_jointly_feasible_top_descent_step",
        jacobian_rank=rank,
        condition=condition,
        raw_delta=raw_delta,
        bounded_delta=bounded_delta,
        top_before=top_before,
        metric_before=metric_before,
    )
