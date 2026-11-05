"""Camera-agnostic current-surface projection with target-depth validation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from evidencescene.primitives.geometry import CameraInfoRecord


@dataclass(frozen=True)
class SurfaceProjection:
    ready: bool
    reason: str
    source_camera_id: str
    target_camera_id: str
    target_image_size: tuple[int, int]
    pixels_uv: np.ndarray
    depth_m: np.ndarray
    row_spans: tuple[tuple[int, int, int], ...]
    uncertainty_m: float
    source_timestamp_ns: int | None
    target_timestamp_ns: int | None
    sync_skew_ms: float | None
    target_depth_supported_ratio: float
    median_target_depth_residual_m: float | None


@dataclass(frozen=True)
class ProjectedSurfaceSamples:
    """A z-buffered source surface in one target camera before depth gating."""

    ready: bool
    reason: str
    source_camera_id: str
    target_camera_id: str
    target_image_size: tuple[int, int]
    pixels_uv: np.ndarray
    depth_m: np.ndarray
    uncertainty_m: float
    source_timestamp_ns: int | None
    target_timestamp_ns: int | None
    sync_skew_ms: float | None
    source_point_indices: np.ndarray | None = None


@dataclass(frozen=True)
class NativeDepthProjectionShadow:
    """Separate measured-depth evidence and color samples; never wire authority."""

    native_depth_validation: SurfaceProjection
    color_samples: ProjectedSurfaceSamples | None
    native_supported_source_indices: np.ndarray
    publication_authorized: bool = field(default=False, init=False)

    @property
    def ready(self) -> bool:
        return bool(self.native_depth_validation.ready and self.color_samples is not None and self.color_samples.ready)


def _validated_projection_transform(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f'{name} must be a finite 4x4 matrix')
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-9, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0)):
        raise ValueError(f'{name} must be a rigid right-handed transform')
    return matrix


def _validate_projection_info(info: CameraInfoRecord) -> None:
    if (not isinstance(info.frame_id, str) or not info.frame_id.strip()
            or type(info.width) is not int or type(info.height) is not int
            or info.width <= 0 or info.height <= 0):
        raise ValueError('Explicit optical frame and positive camera grid required')
    matrix = np.asarray(info.k, dtype=np.float64)
    coefficients = np.asarray(info.d, dtype=np.float64)
    if (matrix.shape != (9,) or not np.isfinite(matrix).all()
            or matrix[0] <= 0 or matrix[4] <= 0
            or not np.allclose(matrix[6:], (0, 0, 1), atol=1e-9, rtol=0)
            or coefficients.ndim != 1 or len(coefficients) not in (0, 4, 5, 8, 12, 14)
            or not np.isfinite(coefficients).all()
            or info.distortion_model not in ('', 'none', 'plumb_bob', 'rational_polynomial')
            or (info.distortion_model in ('', 'none') and np.any(coefficients != 0))):
        raise ValueError('Supported calibrated OpenCV pinhole camera required')


def project_surface_via_native_depth_shadow(
    *, points_source_m: np.ndarray, target_color_from_source: np.ndarray,
    target_color_from_depth: np.ndarray, target_depth_info: CameraInfoRecord,
    target_color_info: CameraInfoRecord, target_native_depth_m: np.ndarray | None,
    source_camera_id: str, target_camera_id: str, source_timestamp_ns: int | None,
    target_depth_timestamp_ns: int | None, target_color_timestamp_ns: int | None,
    uncertainty_m: float, max_sync_skew_ms: float = 80.,
    max_target_depth_residual_m: float = .03, min_target_depth_supported_ratio: float = .5,
    min_supported_pixels: int = 50,
) -> NativeDepthProjectionShadow:
    """Original projection/validation composed on native target depth, shadow only.

    No points are lifted from projected/predicted depth. Both z-buffers retain
    indices into the original source array. No default caller or live cutover.
    Invalid calibration raises; missing/skewed/occluded evidence stays rejected.
    """
    for info in (target_depth_info, target_color_info):
        _validate_projection_info(info)
    color_from_source = _validated_projection_transform(target_color_from_source, 'target_color_from_source')
    color_from_depth = _validated_projection_transform(target_color_from_depth, 'target_color_from_depth')
    for stamp in (source_timestamp_ns, target_depth_timestamp_ns, target_color_timestamp_ns):
        if stamp is not None and (type(stamp) is not int or stamp <= 0):
            raise ValueError('Sensor timestamps must be positive integers or missing')
    if any(not isinstance(camera, str) or not camera.strip() for camera in (source_camera_id, target_camera_id)):
        raise ValueError('Explicit camera identities required')
    for value in (uncertainty_m, max_sync_skew_ms, max_target_depth_residual_m, min_target_depth_supported_ratio):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)):
            raise ValueError('Projection limits must be finite numeric values')
    if (uncertainty_m < 0 or max_sync_skew_ms < 0 or max_target_depth_residual_m <= 0
            or not 0 < min_target_depth_supported_ratio <= 1
            or type(min_supported_pixels) is not int or min_supported_pixels < 1):
        raise ValueError('Invalid projection limits')
    empty = np.empty(0, dtype=np.int64)
    empty.setflags(write=False)
    if (target_depth_timestamp_ns is not None and target_color_timestamp_ns is not None
            and abs(target_depth_timestamp_ns-target_color_timestamp_ns)/1e6 > max_sync_skew_ms):
        rejected = _rejected_projection('target_rgbd_timestamp_skew_exceeded',
            source_camera_id=source_camera_id, target_camera_id=target_camera_id,
            target_info=target_depth_info, uncertainty_m=uncertainty_m,
            source_timestamp_ns=source_timestamp_ns, target_timestamp_ns=target_depth_timestamp_ns)
        return NativeDepthProjectionShadow(rejected, None, empty)
    depth_from_color = np.eye(4)
    depth_from_color[:3, :3] = color_from_depth[:3, :3].T
    depth_from_color[:3, 3] = -depth_from_color[:3, :3] @ color_from_depth[:3, 3]
    samples = project_surface_samples_to_camera(points_source_m=points_source_m,
        target_from_source=depth_from_color @ color_from_source, target_info=target_depth_info,
        source_camera_id=source_camera_id, target_camera_id=target_camera_id,
        source_timestamp_ns=source_timestamp_ns, target_timestamp_ns=target_depth_timestamp_ns,
        uncertainty_m=uncertainty_m, max_sync_skew_ms=max_sync_skew_ms, retain_source_point_indices=True)
    validated = validate_projected_surface_depth(samples, target_info=target_depth_info,
        target_depth_m=target_native_depth_m, max_target_depth_residual_m=max_target_depth_residual_m,
        min_target_depth_supported_ratio=min_target_depth_supported_ratio, min_supported_pixels=min_supported_pixels)
    if not validated.ready:
        return NativeDepthProjectionShadow(validated, None, empty)
    if samples.source_point_indices is None or len(samples.source_point_indices) != len(samples.pixels_uv):
        raise RuntimeError('Original projection lost source indices')
    lookup = {tuple(pixel): int(index) for pixel, index in zip(samples.pixels_uv, samples.source_point_indices, strict=True)}
    if len(lookup) != len(samples.pixels_uv):
        raise RuntimeError('Original native z-buffer contains duplicate pixels')
    try:
        indices = np.asarray([lookup[tuple(pixel)] for pixel in validated.pixels_uv], dtype=np.int64)
    except KeyError as error:
        raise RuntimeError('Validated pixel is not an original projected sample') from error
    indices.setflags(write=False)
    color = project_surface_samples_to_camera(points_source_m=np.asarray(points_source_m)[indices],
        target_from_source=color_from_source, target_info=target_color_info,
        source_camera_id=source_camera_id, target_camera_id=target_camera_id,
        source_timestamp_ns=source_timestamp_ns, target_timestamp_ns=target_color_timestamp_ns,
        uncertainty_m=uncertainty_m, max_sync_skew_ms=max_sync_skew_ms, retain_source_point_indices=True)
    if color.ready:
        if color.source_point_indices is None:
            raise RuntimeError('Original color z-buffer lost source indices')
        global_indices = indices[color.source_point_indices].copy()
        global_indices.setflags(write=False)
        color = replace(color, source_point_indices=global_indices)
    return NativeDepthProjectionShadow(validated, color, indices)


@dataclass(frozen=True)
class FusedSurfaceProjection:
    ready: bool
    reason: str
    target_camera_id: str | None
    pixels_uv: np.ndarray
    depth_m: np.ndarray
    row_spans: tuple[tuple[int, int, int], ...]
    uncertainty_m: float | None
    source_camera_ids: tuple[str, ...]
    rejected_camera_reasons: dict[str, str]


def project_surface_to_camera(
    *,
    points_source_m: np.ndarray,
    target_from_source: np.ndarray,
    target_info: CameraInfoRecord,
    target_depth_m: np.ndarray | None,
    source_camera_id: str,
    target_camera_id: str,
    source_timestamp_ns: int | None,
    target_timestamp_ns: int | None,
    uncertainty_m: float,
    max_sync_skew_ms: float = 80.0,
    max_target_depth_residual_m: float = 0.03,
    min_target_depth_supported_ratio: float = 0.5,
    min_supported_pixels: int = 50,
) -> SurfaceProjection:
    samples = project_surface_samples_to_camera(
        points_source_m=points_source_m,
        target_from_source=target_from_source,
        target_info=target_info,
        source_camera_id=source_camera_id,
        target_camera_id=target_camera_id,
        source_timestamp_ns=source_timestamp_ns,
        target_timestamp_ns=target_timestamp_ns,
        uncertainty_m=uncertainty_m,
        max_sync_skew_ms=max_sync_skew_ms,
    )
    return validate_projected_surface_depth(
        samples,
        target_info=target_info,
        target_depth_m=target_depth_m,
        max_target_depth_residual_m=max_target_depth_residual_m,
        min_target_depth_supported_ratio=min_target_depth_supported_ratio,
        min_supported_pixels=min_supported_pixels,
    )


def project_surface_samples_to_camera(
    *,
    points_source_m: np.ndarray,
    target_from_source: np.ndarray,
    target_info: CameraInfoRecord,
    source_camera_id: str,
    target_camera_id: str,
    source_timestamp_ns: int | None,
    target_timestamp_ns: int | None,
    uncertainty_m: float,
    max_sync_skew_ms: float = 80.0,
    retain_source_point_indices: bool = False,
) -> ProjectedSurfaceSamples:
    if type(retain_source_point_indices) is not bool:
        raise ValueError('source point provenance flag must be boolean')
    rejected = lambda reason, **values: _rejected_projected_samples(
        reason,
        source_camera_id=source_camera_id,
        target_camera_id=target_camera_id,
        target_info=target_info,
        uncertainty_m=uncertainty_m,
        source_timestamp_ns=source_timestamp_ns,
        target_timestamp_ns=target_timestamp_ns,
        **values,
    )
    if source_timestamp_ns is None or target_timestamp_ns is None:
        return rejected("projection_timestamp_missing")
    skew_ms = abs(int(source_timestamp_ns) - int(target_timestamp_ns)) / 1_000_000.0
    if skew_ms > float(max_sync_skew_ms):
        return rejected("projection_timestamp_skew_exceeded", sync_skew_ms=skew_ms)
    points = np.asarray(points_source_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return rejected("source_surface_points_missing", sync_skew_ms=skew_ms)

    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    points_target = (
        np.asarray(target_from_source, dtype=np.float64).reshape(4, 4) @ homogeneous.T
    ).T[:, :3]
    in_front = np.isfinite(points_target).all(axis=1) & (points_target[:, 2] > 0.0)
    if not bool(np.any(in_front)):
        return rejected("source_surface_behind_target_camera", sync_skew_ms=skew_ms)

    import cv2

    source_indices = np.flatnonzero(in_front) if retain_source_point_indices else None
    points_target = points_target[in_front]
    projected, _ = cv2.projectPoints(
        points_target,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.asarray(target_info.k, dtype=np.float64).reshape(3, 3),
        np.asarray(target_info.d, dtype=np.float64),
    )
    pixels = np.rint(projected.reshape(-1, 2)).astype(np.int64)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < target_info.width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < target_info.height)
    )
    pixels = pixels[inside]
    if source_indices is not None:
        source_indices = source_indices[inside]
    projected_depth = points_target[inside, 2]
    if len(pixels) == 0:
        return rejected("source_surface_outside_target_image", sync_skew_ms=skew_ms)
    selected = _zbuffer_pixel_indices(pixels, projected_depth, image_width=target_info.width)
    pixels, projected_depth = pixels[selected].astype(np.int64), projected_depth[selected].astype(np.float64)
    return ProjectedSurfaceSamples(
        ready=True,
        reason="ready",
        source_camera_id=str(source_camera_id),
        target_camera_id=str(target_camera_id),
        target_image_size=(target_info.width, target_info.height),
        pixels_uv=pixels,
        depth_m=projected_depth,
        uncertainty_m=float(uncertainty_m),
        source_timestamp_ns=int(source_timestamp_ns),
        target_timestamp_ns=int(target_timestamp_ns),
        sync_skew_ms=float(skew_ms),
        source_point_indices=source_indices[selected] if source_indices is not None else None,
    )


def validate_projected_surface_depth(
    samples: ProjectedSurfaceSamples,
    *,
    target_info: CameraInfoRecord,
    target_depth_m: np.ndarray | None,
    target_depth_pixels_uv: np.ndarray | None = None,
    target_depth_values_m: np.ndarray | None = None,
    max_target_depth_residual_m: float = 0.03,
    min_target_depth_supported_ratio: float = 0.5,
    min_supported_pixels: int = 50,
) -> SurfaceProjection:
    rejected = lambda reason, **values: _rejected_projection(
        reason,
        source_camera_id=samples.source_camera_id,
        target_camera_id=samples.target_camera_id,
        target_info=target_info,
        uncertainty_m=samples.uncertainty_m,
        source_timestamp_ns=samples.source_timestamp_ns,
        target_timestamp_ns=samples.target_timestamp_ns,
        sync_skew_ms=samples.sync_skew_ms,
        **values,
    )
    if samples.reason in {
        "projection_timestamp_missing",
        "projection_timestamp_skew_exceeded",
        "source_surface_points_missing",
    }:
        return rejected(samples.reason)
    has_sparse_depth = (
        target_depth_pixels_uv is not None
        and target_depth_values_m is not None
    )
    if target_depth_m is None and not has_sparse_depth:
        return rejected("target_depth_missing")
    target_depth = None
    if target_depth_m is not None:
        target_depth = np.asarray(target_depth_m, dtype=np.float64)
        if target_depth.shape != (target_info.height, target_info.width):
            return rejected("target_depth_geometry_mismatch")
    if not samples.ready:
        return rejected(samples.reason)
    pixels = samples.pixels_uv
    projected_depth = samples.depth_m
    if target_depth is not None:
        observed_depth = target_depth[pixels[:, 1], pixels[:, 0]]
    else:
        observed_depth = _sparse_depth_at_pixels(
            pixels,
            target_depth_pixels_uv,
            target_depth_values_m,
            image_width=target_info.width,
        )
    observed_valid = np.isfinite(observed_depth) & (observed_depth > 0.0)
    residual = np.abs(projected_depth - observed_depth)
    supported = observed_valid & (residual <= float(max_target_depth_residual_m))
    supported_ratio = float(np.count_nonzero(supported) / len(pixels))
    valid_residual = residual[observed_valid]
    median_residual = float(np.median(valid_residual)) if len(valid_residual) else None
    if (
        int(np.count_nonzero(supported)) < int(min_supported_pixels)
        or supported_ratio < float(min_target_depth_supported_ratio)
    ):
        return rejected(
            "target_depth_support_insufficient",
            target_depth_supported_ratio=supported_ratio,
            median_target_depth_residual_m=median_residual,
        )
    pixels = pixels[supported]
    projected_depth = projected_depth[supported]
    return SurfaceProjection(
        ready=True,
        reason="ready",
        source_camera_id=str(samples.source_camera_id),
        target_camera_id=str(samples.target_camera_id),
        target_image_size=(target_info.width, target_info.height),
        pixels_uv=pixels,
        depth_m=projected_depth,
        row_spans=surface_row_spans(pixels),
        uncertainty_m=float(samples.uncertainty_m),
        source_timestamp_ns=samples.source_timestamp_ns,
        target_timestamp_ns=samples.target_timestamp_ns,
        sync_skew_ms=samples.sync_skew_ms,
        target_depth_supported_ratio=supported_ratio,
        median_target_depth_residual_m=median_residual,
    )


def _sparse_depth_at_pixels(
    query_pixels_uv: np.ndarray,
    depth_pixels_uv: np.ndarray | None,
    depth_values_m: np.ndarray | None,
    *,
    image_width: int,
) -> np.ndarray:
    query = np.asarray(query_pixels_uv, dtype=np.int64)
    depth_pixels = np.asarray(depth_pixels_uv, dtype=np.int64)
    depth_values = np.asarray(depth_values_m, dtype=np.float64)
    if (
        depth_pixels.ndim != 2
        or depth_pixels.shape[1] != 2
        or depth_values.shape != (len(depth_pixels),)
    ):
        return np.full(len(query), np.nan, dtype=np.float64)
    target_linear = (
        depth_pixels[:, 1] * int(image_width) + depth_pixels[:, 0]
    )
    order = np.argsort(target_linear, kind="stable")
    target_linear = target_linear[order]
    depth_values = depth_values[order]
    query_linear = query[:, 1] * int(image_width) + query[:, 0]
    indices = np.searchsorted(target_linear, query_linear)
    matched = indices < len(target_linear)
    matched[matched] &= target_linear[indices[matched]] == query_linear[matched]
    observed = np.full(len(query), np.nan, dtype=np.float64)
    observed[matched] = depth_values[indices[matched]]
    return observed


def fuse_surface_projections(
    projections: list[SurfaceProjection],
) -> FusedSurfaceProjection:
    accepted = [projection for projection in projections if projection.ready]
    rejected_reasons = {
        projection.source_camera_id: projection.reason
        for projection in projections
        if not projection.ready
    }
    if not accepted:
        return FusedSurfaceProjection(
            ready=False,
            reason="no_accepted_surface_projections",
            target_camera_id=None,
            pixels_uv=np.empty((0, 2), dtype=np.int64),
            depth_m=np.empty(0, dtype=np.float64),
            row_spans=(),
            uncertainty_m=None,
            source_camera_ids=(),
            rejected_camera_reasons=rejected_reasons,
        )
    target_camera_id = accepted[0].target_camera_id
    image_width = accepted[0].target_image_size[0]
    compatible: list[SurfaceProjection] = []
    for projection in accepted:
        if projection.target_camera_id != target_camera_id:
            rejected_reasons[projection.source_camera_id] = "fusion_target_camera_mismatch"
            continue
        compatible.append(projection)
    pixels = np.concatenate([projection.pixels_uv for projection in compatible], axis=0)
    depths = np.concatenate([projection.depth_m for projection in compatible], axis=0)
    pixels, depths = _zbuffer_pixels(pixels, depths, image_width=image_width)
    inverse_variance = sum(
        1.0 / max(float(projection.uncertainty_m), 1e-6) ** 2
        for projection in compatible
    )
    return FusedSurfaceProjection(
        ready=True,
        reason="ready",
        target_camera_id=target_camera_id,
        pixels_uv=pixels,
        depth_m=depths,
        row_spans=surface_row_spans(pixels),
        uncertainty_m=float(1.0 / np.sqrt(inverse_variance)),
        source_camera_ids=tuple(sorted({projection.source_camera_id for projection in compatible})),
        rejected_camera_reasons=rejected_reasons,
    )


def _zbuffer_pixels(
    pixels: np.ndarray,
    depth_m: np.ndarray,
    *,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = _zbuffer_pixel_indices(pixels, depth_m, image_width=image_width)
    return pixels[selected].astype(np.int64), depth_m[selected].astype(np.float64)


def _zbuffer_pixel_indices(pixels: np.ndarray, depth_m: np.ndarray, *, image_width: int) -> np.ndarray:
    linear = pixels[:, 1] * int(image_width) + pixels[:, 0]
    order = np.lexsort((depth_m, linear))
    ordered_linear = linear[order]
    nearest = np.concatenate(([True], ordered_linear[1:] != ordered_linear[:-1]))
    selected = order[nearest]
    return selected


def surface_row_spans(pixels: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    values = np.asarray(pixels, dtype=np.int64)
    if len(values) == 0:
        return ()
    order = np.lexsort((values[:, 0], values[:, 1]))
    ordered = values[order]
    starts = np.concatenate(
        (
            np.array([True]),
            (ordered[1:, 1] != ordered[:-1, 1])
            | (ordered[1:, 0] > ordered[:-1, 0] + 1),
        )
    )
    start_indices = np.flatnonzero(starts)
    end_indices = np.concatenate((start_indices[1:] - 1, [len(ordered) - 1]))
    return tuple(
        (
            int(ordered[start, 1]),
            int(ordered[start, 0]),
            int(ordered[end, 0]),
        )
        for start, end in zip(start_indices, end_indices, strict=True)
    )


def _rejected_projection(
    reason: str,
    *,
    source_camera_id: str,
    target_camera_id: str,
    target_info: CameraInfoRecord,
    uncertainty_m: float,
    source_timestamp_ns: int | None,
    target_timestamp_ns: int | None,
    sync_skew_ms: float | None = None,
    target_depth_supported_ratio: float = 0.0,
    median_target_depth_residual_m: float | None = None,
) -> SurfaceProjection:
    return SurfaceProjection(
        ready=False,
        reason=reason,
        source_camera_id=str(source_camera_id),
        target_camera_id=str(target_camera_id),
        target_image_size=(target_info.width, target_info.height),
        pixels_uv=np.empty((0, 2), dtype=np.int64),
        depth_m=np.empty(0, dtype=np.float64),
        row_spans=(),
        uncertainty_m=float(uncertainty_m),
        source_timestamp_ns=source_timestamp_ns,
        target_timestamp_ns=target_timestamp_ns,
        sync_skew_ms=sync_skew_ms,
        target_depth_supported_ratio=float(target_depth_supported_ratio),
        median_target_depth_residual_m=median_target_depth_residual_m,
    )


def _rejected_projected_samples(
    reason: str,
    *,
    source_camera_id: str,
    target_camera_id: str,
    target_info: CameraInfoRecord,
    uncertainty_m: float,
    source_timestamp_ns: int | None,
    target_timestamp_ns: int | None,
    sync_skew_ms: float | None = None,
) -> ProjectedSurfaceSamples:
    return ProjectedSurfaceSamples(
        ready=False,
        reason=reason,
        source_camera_id=str(source_camera_id),
        target_camera_id=str(target_camera_id),
        target_image_size=(target_info.width, target_info.height),
        pixels_uv=np.empty((0, 2), dtype=np.int64),
        depth_m=np.empty(0, dtype=np.float64),
        uncertainty_m=float(uncertainty_m),
        source_timestamp_ns=source_timestamp_ns,
        target_timestamp_ns=target_timestamp_ns,
        sync_skew_ms=sync_skew_ms,
    )
