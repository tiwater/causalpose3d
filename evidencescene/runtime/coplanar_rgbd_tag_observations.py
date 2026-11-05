"""Metric AprilTag corners from one shared, depth-native support plane."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import cv2
import numpy as np

from evidencescene.primitives.geometry import (
    CameraInfoRecord,
    DepthBoundaryCandidate,
    RegisteredRgbdFrame,
    intersect_rectified_tag_rays_with_depth_plane,
)


_TAG_REGION_SCALE = 2.5
_MINIMUM_PLANE_POINTS = 80
_PLANE_DISTANCE_THRESHOLD_M = 0.004
_RANSAC_ITERATIONS = 240
_MINIMUM_LOCAL_DEPTH_POINTS = 20
_MAXIMUM_LOCAL_DEPTH_P95_M = 0.010
_MAXIMUM_TAG_SIDE_RELATIVE_ERROR = 0.10
_LOCAL_TAG_REGION_SCALE = 1.5


def _frozen(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {shape} array")
    result = np.ndarray(array.shape, dtype=array.dtype, buffer=np.ascontiguousarray(array).tobytes())
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class CoplanarRgbdTagCorners:
    tag_id: int
    corners_depth_m: np.ndarray

    def __post_init__(self) -> None:
        if type(self.tag_id) is not int or self.tag_id < 0:
            raise ValueError("tag_id must be a nonnegative integer")
        object.__setattr__(
            self,
            "corners_depth_m",
            _frozen(self.corners_depth_m, (4, 3), "corners_depth_m"),
        )


@dataclass(frozen=True)
class CoplanarRgbdTagResult:
    ready: bool
    reason: str
    detected_tag_count: int
    plane_sample_count: int
    plane_inlier_count: int
    plane_rmse_m: float | None
    plane_normal_depth: np.ndarray
    plane_offset_m: float | None
    local_depth_p95_m: float | None
    tags: tuple[CoplanarRgbdTagCorners, ...]
    broad_refinement_applied: bool = False
    shadow_seed_plane_sample_count: int | None = None
    shadow_seed_plane_inlier_count: int | None = None
    shadow_seed_plane_rmse_m: float | None = None
    shadow_seed_tag_corners: tuple[CoplanarRgbdTagCorners, ...] = ()
    shadow_seed_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.ready) is not bool or not self.reason:
            raise ValueError("result readiness and reason are required")
        if any(
            type(value) is not int or value < 0
            for value in (
                self.detected_tag_count,
                self.plane_sample_count,
                self.plane_inlier_count,
            )
        ):
            raise ValueError("result counts must be nonnegative integers")
        normal_shape = (3,) if self.ready else (0,)
        object.__setattr__(
            self,
            "plane_normal_depth",
            _frozen(self.plane_normal_depth, normal_shape, "plane_normal_depth"),
        )
        tags = tuple(sorted(self.tags, key=lambda item: item.tag_id))
        if len({item.tag_id for item in tags}) != len(tags):
            raise ValueError("result tags must have distinct IDs")
        if type(self.broad_refinement_applied) is not bool:
            raise ValueError("broad_refinement_applied must be boolean")
        if (self.shadow_seed_plane_sample_count is None) != (
            self.shadow_seed_plane_inlier_count is None
        ):
            raise ValueError("shadow seed sample and inlier counts must be paired")
        if self.shadow_seed_plane_sample_count is not None:
            if (
                type(self.shadow_seed_plane_sample_count) is not int
                or type(self.shadow_seed_plane_inlier_count) is not int
                or self.shadow_seed_plane_sample_count < 0
                or self.shadow_seed_plane_inlier_count < 0
                or self.shadow_seed_plane_inlier_count > self.shadow_seed_plane_sample_count
            ):
                raise ValueError("shadow seed plane counts are invalid")
        if self.shadow_seed_plane_rmse_m is not None and (
            not math.isfinite(self.shadow_seed_plane_rmse_m)
            or self.shadow_seed_plane_rmse_m < 0.0
        ):
            raise ValueError("shadow seed plane RMSE must be finite and nonnegative")
        shadow_seed_tags = tuple(sorted(self.shadow_seed_tag_corners, key=lambda item: item.tag_id))
        if len({item.tag_id for item in shadow_seed_tags}) != len(shadow_seed_tags):
            raise ValueError("shadow seed tag corners must have distinct IDs")
        if shadow_seed_tags and self.shadow_seed_reason != "ready":
            raise ValueError("shadow seed corners require a ready seed reason")
        if self.shadow_seed_reason == "ready" and not shadow_seed_tags:
            raise ValueError("ready shadow seed requires tag corners")
        if self.shadow_seed_plane_sample_count is None and (
            self.shadow_seed_plane_rmse_m is not None
            or shadow_seed_tags
            or self.shadow_seed_reason is not None
        ):
            raise ValueError("shadow seed diagnostics require seed plane counts")
        if self.ready != bool(tags) or self.ready != (self.reason == "ready"):
            raise ValueError("ready state, reason, and tags are inconsistent")
        if self.ready:
            if (
                self.plane_rmse_m is None
                or self.plane_offset_m is None
                or not math.isfinite(self.plane_rmse_m)
                or self.plane_rmse_m < 0.0
                or not math.isfinite(self.plane_offset_m)
            ):
                raise ValueError("ready result requires finite plane metrics")
            if self.local_depth_p95_m is not None and (
                not math.isfinite(self.local_depth_p95_m)
                or self.local_depth_p95_m < 0.0
            ):
                raise ValueError("local depth p95 must be finite and nonnegative")
        elif self.plane_rmse_m is not None or self.plane_offset_m is not None:
            raise ValueError("rejected result cannot publish a plane")
        elif self.local_depth_p95_m is not None:
            raise ValueError("rejected result cannot publish local depth metrics")
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "shadow_seed_tag_corners", shadow_seed_tags)


def observe_coplanar_rgbd_tag_corners(
    markers: Iterable[tuple[np.ndarray, int]],
    *,
    registered_frame: RegisteredRgbdFrame,
    color_info: CameraInfoRecord,
    color_from_depth: object,
    minimum_distinct_tags: int = 3,
    include_shadow_seed_corners: bool = False,
) -> CoplanarRgbdTagResult:
    """Intersect all tag rays with one plane fitted from their pooled depth.

    AprilTags in this observation must be fixed to one physical support plane.
    No tag layout, table dimensions, frame index, or object position is used.
    """

    if type(minimum_distinct_tags) is not int or minimum_distinct_tags < 1:
        raise ValueError("minimum_distinct_tags must be an integer of at least 1")
    if type(include_shadow_seed_corners) is not bool:
        raise ValueError("include_shadow_seed_corners must be boolean")
    if not isinstance(registered_frame, RegisteredRgbdFrame):
        raise ValueError("registered_frame must be RegisteredRgbdFrame")
    try:
        rows = tuple(markers)
    except TypeError as error:
        raise ValueError("markers must be iterable") from error
    parsed: list[tuple[np.ndarray, int]] = []
    for marker in rows:
        if not isinstance(marker, tuple) or len(marker) != 2:
            return _rejected("invalid_marker", len(rows))
        corners, tag_id = marker
        if type(tag_id) is not int or tag_id < 0:
            return _rejected("invalid_marker", len(rows))
        try:
            parsed.append((_frozen(corners, (4, 2), "marker corners"), tag_id))
        except ValueError:
            return _rejected("invalid_marker", len(rows))
    tag_ids = [tag_id for _, tag_id in parsed]
    if len(set(tag_ids)) != len(tag_ids):
        return _rejected("duplicate_tag_id", len(parsed))
    if len(parsed) < minimum_distinct_tags:
        return _rejected("insufficient_distinct_tags", len(parsed))
    if not registered_frame.ready:
        return _rejected(registered_frame.reason, len(parsed))
    if not isinstance(color_info, CameraInfoRecord):
        return _rejected("invalid_color_camera_info", len(parsed))
    if registered_frame.color_image_size != (color_info.width, color_info.height):
        return _rejected("color_image_geometry_mismatch", len(parsed))
    transform = np.asarray(color_from_depth, dtype=np.float64)
    if not _is_rigid_transform(transform):
        return _rejected("invalid_color_from_depth", len(parsed))

    color_pixels = np.asarray(registered_frame.color_pixels_uv, dtype=np.int64)
    points_depth = np.asarray(registered_frame.points_depth_m, dtype=np.float64)
    if (
        color_pixels.shape != (len(points_depth), 2)
        or points_depth.ndim != 2
        or points_depth.shape[1:] != (3,)
        or not np.isfinite(points_depth).all()
    ):
        return _rejected("registered_points_invalid", len(parsed))

    mask = np.zeros((color_info.height, color_info.width), dtype=np.uint8)
    for corners, _ in parsed:
        center = np.mean(corners, axis=0)
        polygon = center + (corners - center) * _TAG_REGION_SCALE
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
    inside = (
        (color_pixels[:, 0] >= 0)
        & (color_pixels[:, 0] < color_info.width)
        & (color_pixels[:, 1] >= 0)
        & (color_pixels[:, 1] < color_info.height)
    )
    selected = np.zeros(len(points_depth), dtype=bool)
    selected[inside] = mask[color_pixels[inside, 1], color_pixels[inside, 0]].astype(bool)
    plane_samples = points_depth[selected]
    if len(plane_samples) < _MINIMUM_PLANE_POINTS:
        return _rejected(
            "insufficient_shared_plane_points",
            len(parsed),
            plane_sample_count=len(plane_samples),
        )
    plane = _fit_shared_plane(plane_samples)
    if plane is None:
        return _rejected(
            "shared_plane_not_observable",
            len(parsed),
            plane_sample_count=len(plane_samples),
        )
    normal, offset, inlier_count, rmse = plane
    seed_normal = np.asarray(normal, dtype=np.float64).copy()
    seed_offset = float(offset)
    seed_inlier_count = int(inlier_count)
    seed_rmse = float(rmse)
    broad = _refine_seeded_plane(points_depth, normal, offset)
    broad_refinement_applied = broad is not None
    plane_sample_count = len(plane_samples)
    if broad is not None:
        normal, offset, inlier_count, rmse = broad
        # The accepted fit now uses the complete registered-depth population,
        # so its inlier count must be paired with that population's size.
        plane_sample_count = len(points_depth)

    intrinsic = np.asarray(color_info.k, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(color_info.d, dtype=np.float64)
    accepted: list[CoplanarRgbdTagCorners] = []
    rectified_rows: list[tuple[np.ndarray, int]] = []
    try:
        for corners, tag_id in parsed:
            rectified = cv2.undistortPoints(
                np.asarray(corners).reshape(-1, 1, 2),
                intrinsic,
                distortion,
                P=intrinsic,
            ).reshape(4, 2)
            rectified_rows.append((rectified, tag_id))
            metric = intersect_rectified_tag_rays_with_depth_plane(
                rectified_corners_px=rectified,
                color_k=intrinsic,
                color_from_depth=transform,
                plane_normal_depth=normal,
                plane_offset_m=offset,
            )
            accepted.append(CoplanarRgbdTagCorners(tag_id, metric))
    except (ValueError, cv2.error):
        return _rejected(
            "tag_ray_plane_intersection_failed",
            len(parsed),
            plane_sample_count=plane_sample_count,
            plane_inlier_count=inlier_count,
        )
    shadow_seed_tags: list[CoplanarRgbdTagCorners] = []
    shadow_seed_reason = None
    if include_shadow_seed_corners:
        shadow_seed_reason = "ready"
        if broad_refinement_applied:
            try:
                for rectified, tag_id in rectified_rows:
                    metric = intersect_rectified_tag_rays_with_depth_plane(
                        rectified_corners_px=rectified,
                        color_k=intrinsic,
                        color_from_depth=transform,
                        plane_normal_depth=seed_normal,
                        plane_offset_m=seed_offset,
                    )
                    shadow_seed_tags.append(CoplanarRgbdTagCorners(tag_id, metric))
            except (ValueError, cv2.error):
                shadow_seed_tags.clear()
                shadow_seed_reason = "tag_local_seed_ray_plane_intersection_failed"
        else:
            # When broad refinement did not replace the seed, both arms of
            # the ablation consume the same exact metric corners.
            shadow_seed_tags.extend(accepted)
    return CoplanarRgbdTagResult(
        ready=True,
        reason="ready",
        detected_tag_count=len(parsed),
        plane_sample_count=plane_sample_count,
        plane_inlier_count=inlier_count,
        plane_rmse_m=rmse,
        plane_normal_depth=normal,
        plane_offset_m=offset,
        local_depth_p95_m=None,
        tags=tuple(accepted),
        broad_refinement_applied=broad_refinement_applied,
        shadow_seed_plane_sample_count=(len(plane_samples) if include_shadow_seed_corners else None),
        shadow_seed_plane_inlier_count=(seed_inlier_count if include_shadow_seed_corners else None),
        shadow_seed_plane_rmse_m=(seed_rmse if include_shadow_seed_corners else None),
        shadow_seed_tag_corners=tuple(shadow_seed_tags),
        shadow_seed_reason=shadow_seed_reason,
    )


def observe_candidate_plane_rgbd_tag_corners(
    markers: Iterable[tuple[np.ndarray, int]],
    *,
    depth_candidates: Iterable[DepthBoundaryCandidate],
    registered_frame: RegisteredRgbdFrame,
    color_info: CameraInfoRecord,
    color_from_depth: object,
    tag_side_m: float,
) -> CoplanarRgbdTagResult:
    """Select a depth-native plane using tag shape and local depth evidence."""

    if not isinstance(registered_frame, RegisteredRgbdFrame):
        raise ValueError("registered_frame must be RegisteredRgbdFrame")
    try:
        rows = tuple(markers)
        candidates = tuple(depth_candidates)
    except TypeError as error:
        raise ValueError("markers and depth_candidates must be iterable") from error
    parsed: list[tuple[np.ndarray, int]] = []
    for marker in rows:
        if not isinstance(marker, tuple) or len(marker) != 2:
            return _rejected("invalid_marker", len(rows))
        corners, tag_id = marker
        if type(tag_id) is not int or tag_id < 0:
            return _rejected("invalid_marker", len(rows))
        try:
            parsed.append((_frozen(corners, (4, 2), "marker corners"), tag_id))
        except ValueError:
            return _rejected("invalid_marker", len(rows))
    ids = [tag_id for _, tag_id in parsed]
    if len(set(ids)) != len(ids):
        return _rejected("duplicate_tag_id", len(parsed))
    if not parsed:
        return _rejected("tags_missing", 0)
    if not registered_frame.ready:
        return _rejected(registered_frame.reason, len(parsed))
    if not isinstance(color_info, CameraInfoRecord):
        return _rejected("invalid_color_camera_info", len(parsed))
    if registered_frame.color_image_size != (color_info.width, color_info.height):
        return _rejected("color_image_geometry_mismatch", len(parsed))
    transform = np.asarray(color_from_depth, dtype=np.float64)
    if not _is_rigid_transform(transform):
        return _rejected("invalid_color_from_depth", len(parsed))
    side = float(tag_side_m)
    if not math.isfinite(side) or side <= 0.0:
        raise ValueError("tag_side_m must be finite and positive")
    if any(not isinstance(candidate, DepthBoundaryCandidate) for candidate in candidates):
        raise ValueError("depth_candidates must contain DepthBoundaryCandidate values")
    if not candidates:
        return _rejected("depth_plane_candidates_missing", len(parsed))

    color_pixels = np.asarray(registered_frame.color_pixels_uv, dtype=np.int64)
    points_depth = np.asarray(registered_frame.points_depth_m, dtype=np.float64)
    if (
        color_pixels.shape != (len(points_depth), 2)
        or points_depth.ndim != 2
        or points_depth.shape[1:] != (3,)
        or not np.isfinite(points_depth).all()
    ):
        return _rejected("registered_points_invalid", len(parsed))
    intrinsic = np.asarray(color_info.k, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(color_info.d, dtype=np.float64)
    local_points: list[np.ndarray] = []
    rectified_rows: list[tuple[np.ndarray, int]] = []
    for corners, tag_id in parsed:
        center = np.mean(corners, axis=0)
        polygon = center + (corners - center) * _LOCAL_TAG_REGION_SCALE
        mask = np.zeros((color_info.height, color_info.width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
        inside = (
            (color_pixels[:, 0] >= 0)
            & (color_pixels[:, 0] < color_info.width)
            & (color_pixels[:, 1] >= 0)
            & (color_pixels[:, 1] < color_info.height)
        )
        selected = np.zeros(len(points_depth), dtype=bool)
        selected[inside] = mask[color_pixels[inside, 1], color_pixels[inside, 0]].astype(bool)
        if np.any(selected):
            local_points.append(points_depth[selected])
        try:
            rectified = cv2.undistortPoints(
                np.asarray(corners).reshape(-1, 1, 2),
                intrinsic,
                distortion,
                P=intrinsic,
            ).reshape(4, 2)
        except cv2.error:
            return _rejected("tag_rectification_failed", len(parsed))
        rectified_rows.append((rectified, tag_id))
    local = np.concatenate(local_points) if local_points else np.empty((0, 3))
    if len(local) < _MINIMUM_LOCAL_DEPTH_POINTS:
        return _rejected(
            "insufficient_local_registered_depth",
            len(parsed),
            plane_sample_count=len(local),
        )

    accepted_candidates: list[
        tuple[float, float, str, DepthBoundaryCandidate, tuple[CoplanarRgbdTagCorners, ...], np.ndarray]
    ] = []
    for candidate in candidates:
        if (
            candidate.frame_id != registered_frame.depth_frame_id
            or candidate.timestamp_ns != registered_frame.depth_timestamp_ns
        ):
            continue
        normal = np.asarray(candidate.plane_normal_camera, dtype=np.float64)
        offset = float(candidate.plane_offset_m)
        residuals = np.abs(local @ normal + offset)
        local_p95 = float(np.percentile(residuals, 95.0))
        if local_p95 > _MAXIMUM_LOCAL_DEPTH_P95_M:
            continue
        tag_rows: list[CoplanarRgbdTagCorners] = []
        maximum_side_error = 0.0
        try:
            for rectified, tag_id in rectified_rows:
                metric = intersect_rectified_tag_rays_with_depth_plane(
                    rectified_corners_px=rectified,
                    color_k=intrinsic,
                    color_from_depth=transform,
                    plane_normal_depth=normal,
                    plane_offset_m=offset,
                )
                edges = np.linalg.norm(np.roll(metric, -1, axis=0) - metric, axis=1)
                side_error = float(np.max(np.abs(edges - side)) / side)
                maximum_side_error = max(maximum_side_error, side_error)
                tag_rows.append(CoplanarRgbdTagCorners(tag_id, metric))
        except ValueError:
            continue
        if maximum_side_error > _MAXIMUM_TAG_SIDE_RELATIVE_ERROR:
            continue
        accepted_candidates.append((
            local_p95,
            maximum_side_error,
            candidate.content_hash,
            candidate,
            tuple(tag_rows),
            residuals,
        ))
    if not accepted_candidates:
        return _rejected(
            "no_candidate_passed_tag_and_depth_gates",
            len(parsed),
            plane_sample_count=len(local),
        )
    local_p95, _, _, selected_candidate, tags, residuals = min(
        accepted_candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    normal = np.asarray(selected_candidate.plane_normal_camera, dtype=np.float64)
    offset = float(selected_candidate.plane_offset_m)
    return CoplanarRgbdTagResult(
        ready=True,
        reason="ready",
        detected_tag_count=len(parsed),
        plane_sample_count=len(local),
        plane_inlier_count=int(np.count_nonzero(residuals <= _MAXIMUM_LOCAL_DEPTH_P95_M)),
        plane_rmse_m=float(np.sqrt(np.mean(np.square(residuals)))),
        plane_normal_depth=normal,
        plane_offset_m=offset,
        local_depth_p95_m=local_p95,
        tags=tags,
    )


def _fit_shared_plane(points: np.ndarray) -> tuple[np.ndarray, float, int, float] | None:
    rng = np.random.default_rng(0)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(_RANSAC_ITERATIONS):
        indices = rng.choice(len(points), size=3, replace=False)
        first, second, third = points[indices]
        normal = np.cross(second - first, third - first)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-10:
            continue
        normal /= norm
        inliers = np.abs((points - first) @ normal) <= _PLANE_DISTANCE_THRESHOLD_M
        if int(np.count_nonzero(inliers)) > int(np.count_nonzero(best)):
            best = inliers
    count = int(np.count_nonzero(best))
    if count < _MINIMUM_PLANE_POINTS:
        return None
    inliers = points[best]
    center = np.mean(inliers, axis=0)
    _, _, right_transpose = np.linalg.svd(inliers - center, full_matrices=False)
    normal = right_transpose[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ center)
    residuals = np.abs(inliers @ normal + offset)
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    if not math.isfinite(rmse) or rmse > _PLANE_DISTANCE_THRESHOLD_M:
        return None
    return normal, offset, count, rmse


def _refine_seeded_plane(
    points: np.ndarray,
    seed_normal: np.ndarray,
    seed_offset: float,
    *,
    diagnostic_iterations: list[dict] | None = None,
) -> tuple[np.ndarray, float, int, float] | None:
    """Refine a tag-local plane using all globally consistent depth support."""
    values = np.asarray(points, dtype=np.float64)
    normal = np.asarray(seed_normal, dtype=np.float64).reshape(3)
    offset = float(seed_offset)
    if (
        values.ndim != 2
        or values.shape[1:] != (3,)
        or not np.isfinite(values).all()
        or not np.isfinite(normal).all()
        or not math.isfinite(offset)
    ):
        raise ValueError("seeded plane refinement inputs are invalid")
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError("seeded plane normal is degenerate")
    normal = normal / norm
    offset /= norm
    seed = normal.copy()
    selected = np.zeros(len(values), dtype=bool)
    for _ in range(2):
        selected = np.abs(values @ normal + offset) <= _PLANE_DISTANCE_THRESHOLD_M
        if int(np.count_nonzero(selected)) < _MINIMUM_PLANE_POINTS:
            return None
        inliers = values[selected]
        center = np.mean(inliers, axis=0)
        _, _, right_transpose = np.linalg.svd(inliers - center, full_matrices=False)
        normal = right_transpose[-1]
        normal /= np.linalg.norm(normal)
        if float(normal @ seed) < 0.0:
            normal = -normal
        offset = -float(normal @ center)
        if diagnostic_iterations is not None:
            diagnostic_iterations.append({
                "selected_input_indices": np.flatnonzero(selected).tolist(),
                "plane_normal": normal.tolist(),
                "plane_offset_m": offset,
                "selected_centroid_m": center.tolist(),
            })
    residuals = np.abs(values[selected] @ normal + offset)
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    if not math.isfinite(rmse) or rmse > _PLANE_DISTANCE_THRESHOLD_M:
        return None
    return normal, offset, int(np.count_nonzero(selected)), rmse


def _is_rigid_transform(value: np.ndarray) -> bool:
    return bool(
        value.shape == (4, 4)
        and np.isfinite(value).all()
        and np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12, rtol=0.0)
        and np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-9, rtol=0.0)
        and math.isclose(float(np.linalg.det(value[:3, :3])), 1.0, abs_tol=1e-9)
    )


def _rejected(
    reason: str,
    detected_tag_count: int,
    *,
    plane_sample_count: int = 0,
    plane_inlier_count: int = 0,
) -> CoplanarRgbdTagResult:
    return CoplanarRgbdTagResult(
        ready=False,
        reason=reason,
        detected_tag_count=detected_tag_count,
        plane_sample_count=plane_sample_count,
        plane_inlier_count=plane_inlier_count,
        plane_rmse_m=None,
        plane_normal_depth=np.empty(0),
        plane_offset_m=None,
        local_depth_p95_m=None,
        tags=(),
    )
