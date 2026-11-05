"""Fuse current object surfaces across cameras through a shared table frame."""

from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass
from functools import partial

import numpy as np

from evidencescene.primitives.geometry import CameraInfoRecord
from evidencescene.surface.realtime_surface_projection import (
    FusedSurfaceProjection,
    ProjectedSurfaceSamples,
    fuse_surface_projections,
    project_surface_to_camera,
    validate_projected_surface_depth,
)


@dataclass(frozen=True)
class SurfaceEvidence:
    label: str
    score: float
    source_camera_id: str
    points_source_m: np.ndarray
    table_from_source: np.ndarray
    source_timestamp_ns: int | None
    uncertainty_m: float


@dataclass(frozen=True)
class ProjectionTarget:
    camera_id: str
    target_color_from_table: np.ndarray
    color_info: CameraInfoRecord
    color_depth_m: np.ndarray | None
    target_timestamp_ns: int | None
    projected_samples_by_evidence: dict[int, ProjectedSurfaceSamples] | None = None
    color_depth_pixels_uv: np.ndarray | None = None
    color_depth_values_m: np.ndarray | None = None


@dataclass(frozen=True)
class TableAlignedSurfaceProjection:
    ready: bool
    label: str
    score: float
    target_camera_id: str
    source_camera_ids: tuple[str, ...]
    fused_projection: FusedSurfaceProjection
    target_depth_supported_ratio: float
    median_target_depth_residual_m: float | None


def project_table_aligned_surfaces(
    evidences: list[SurfaceEvidence],
    targets: list[ProjectionTarget],
    *,
    executor: Executor | None = None,
) -> list[TableAlignedSurfaceProjection]:
    labels = sorted({evidence.label for evidence in evidences})
    project_target = partial(
        _project_table_aligned_surface_target,
        evidences,
        labels,
    )
    target_results = (
        list(executor.map(project_target, targets))
        if executor is not None and len(targets) > 1
        else [project_target(target) for target in targets]
    )
    return [result for rows in target_results for result in rows]


def _project_table_aligned_surface_target(
    evidences: list[SurfaceEvidence],
    labels: list[str],
    target: ProjectionTarget,
) -> list[TableAlignedSurfaceProjection]:
    results: list[TableAlignedSurfaceProjection] = []
    for label in labels:
        label_evidence = [
            (evidence_index, evidence)
            for evidence_index, evidence in enumerate(evidences)
            if evidence.label == label
            and evidence.source_camera_id != target.camera_id
        ]
        projections = []
        scores: list[float] = []
        supported_ratios: list[float] = []
        residuals: list[float] = []
        for evidence_index, evidence in label_evidence:
            cached = (target.projected_samples_by_evidence or {}).get(
                evidence_index
            )
            if cached is not None:
                projection = validate_projected_surface_depth(
                    cached,
                    target_info=target.color_info,
                    target_depth_m=target.color_depth_m,
                    target_depth_pixels_uv=target.color_depth_pixels_uv,
                    target_depth_values_m=target.color_depth_values_m,
                )
            else:
                projection = project_surface_to_camera(
                    points_source_m=evidence.points_source_m,
                    target_from_source=(
                        np.asarray(
                            target.target_color_from_table,
                            dtype=np.float64,
                        ).reshape(4, 4)
                        @ np.asarray(
                            evidence.table_from_source,
                            dtype=np.float64,
                        ).reshape(4, 4)
                    ),
                    target_info=target.color_info,
                    target_depth_m=target.color_depth_m,
                    source_camera_id=evidence.source_camera_id,
                    target_camera_id=target.camera_id,
                    source_timestamp_ns=evidence.source_timestamp_ns,
                    target_timestamp_ns=target.target_timestamp_ns,
                    uncertainty_m=evidence.uncertainty_m,
                )
            if not projection.ready:
                continue
            projections.append(projection)
            scores.append(float(evidence.score))
            supported_ratios.append(float(projection.target_depth_supported_ratio))
            if projection.median_target_depth_residual_m is not None:
                residuals.append(float(projection.median_target_depth_residual_m))
        if not projections:
            continue
        fused = fuse_surface_projections(projections)
        if not fused.ready:
            continue
        results.append(
            TableAlignedSurfaceProjection(
                ready=True,
                label=label,
                score=max(scores),
                target_camera_id=target.camera_id,
                source_camera_ids=fused.source_camera_ids,
                fused_projection=fused,
                target_depth_supported_ratio=min(supported_ratios),
                median_target_depth_residual_m=(
                    float(np.median(residuals)) if residuals else None
                ),
            )
        )
    return results
