"""Causal object surface reconstruction and RevisableSurfelMap selectors."""

from __future__ import annotations

from causalpose3d.surface.live_surface_fusion import (
    ProjectionTarget,
    SurfaceEvidence,
    TableAlignedSurfaceProjection,
    project_table_aligned_surfaces,
)
from causalpose3d.surface.object_surface_fusion import (
    RevisableSurfelMap,
    measured_free_space_margin,
)
from causalpose3d.surface.realtime_surface_projection import (
    FusedSurfaceProjection,
    NativeDepthProjectionShadow,
    ProjectedSurfaceSamples,
    SurfaceProjection,
    fuse_surface_projections,
    project_surface_samples_to_camera,
    project_surface_to_camera,
    project_surface_via_native_depth_shadow,
    surface_row_spans,
    validate_projected_surface_depth,
)
from causalpose3d.surface.shadow_surface_snapshot_selection import (
    select_existing_points_shadow,
)
from causalpose3d.surface.supplemental_surface_observation import (
    prepare_packet,
)

__all__ = [
    "FusedSurfaceProjection",
    "NativeDepthProjectionShadow",
    "ProjectedSurfaceSamples",
    "ProjectionTarget",
    "RevisableSurfelMap",
    "SurfaceEvidence",
    "SurfaceProjection",
    "TableAlignedSurfaceProjection",
    "causal_metric_point_map",
    "causal_object_surface",
    "causal_surface_track_memory",
    "fuse_surface_projections",
    "live_surface_fusion",
    "measured_free_space_margin",
    "object_surface_fusion",
    "prepare_packet",
    "project_surface_samples_to_camera",
    "project_surface_to_camera",
    "project_surface_via_native_depth_shadow",
    "project_table_aligned_surfaces",
    "realtime_surface_projection",
    "select_existing_points_shadow",
    "shadow_surface_snapshot_selection",
    "supplemental_surface_observation",
    "surface_corroboration",
    "surface_row_spans",
    "validate_projected_surface_depth",
]
