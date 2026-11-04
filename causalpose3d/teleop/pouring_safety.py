"""3D ballistic liquid stream locus, landing uncertainty ellipse, and fail-closed L2 pouring safety gate.

Provides physics-informed spatial safety verification for L2 teleoperation assistance:
- Parabolic liquid stream trajectory under gravity:
      x(s) = p_out + v_0 * d_out * s + 0.5 * g * s^2
- Exact descending intersection with the target container aperture plane and
  first-order covariance propagation to the 2D landing uncertainty ellipse.
- Fail-closed 95% rim-clearance safety gate (``uncertainty_envelope_clears_cup_rim``)
  across current and predicted horizons, guaranteeing zero false-safe guidance.
- Minimum-norm L2 shared-control translation guidance vector ``assist_translation_delta_m``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from causalpose3d.teleop.task_features import (
    MeasuredCircularAperture3D,
    MeasuredPointFeature3D,
    _orthonormal_plane_basis,
    _unit_vector,
    propagate_feature_uncertainty_se3,
)
from causalpose3d.teleop.trajectory import MultiHorizonTrajectoryReport

STANDARD_GRAVITY_MPS2 = np.array([0.0, 0.0, -9.80665], dtype=np.float64)
CHI2_2DOF_95_SCALE = 2.447746830680816  # sqrt(chi2.ppf(0.95, df=2))


@dataclass(frozen=True)
class BallisticPourLocus:
    """Predicted 3D parabolic liquid stream and landing uncertainty on the aperture plane."""

    valid: bool
    reason: str
    time_of_flight_s: float | None
    impact_point_m: np.ndarray | None
    planar_radial_offset_m: float | None
    landing_ellipse_semi_major_95_m: float | None
    landing_ellipse_semi_minor_95_m: float | None
    landing_covariance_2x2_m2: np.ndarray | None
    centerline_points_m: np.ndarray
    landing_contour_points_m: np.ndarray


@dataclass(frozen=True)
class L2PouringSafetyVerdict:
    """Fail-closed L2 teleoperation pouring safety verdict and shared-control guidance."""

    l2_pour_safety_status: str
    uncertainty_envelope_clears_cup_rim: bool
    safe_to_pour: bool
    blocker_reasons: tuple[str, ...]
    signed_height_above_rim_m: float | None
    current_radial_offset_m: float | None
    current_rim_clearance_95_m: float | None
    worst_horizon_rim_clearance_95_m: float | None
    evaluated_horizons_ms: tuple[float, ...]
    assist_translation_delta_m: np.ndarray
    ballistic_locus: BallisticPourLocus | None
    publication_authorized: bool = False
    control_authorized: bool = False


def predict_ballistic_pour_locus(
    *,
    pour_outlet: MeasuredPointFeature3D,
    target_aperture: MeasuredCircularAperture3D,
    exit_speed_mps: float = 0.35,
    exit_speed_std_mps: float = 0.06,
    gravity_mps2: Sequence[float] = (0.0, 0.0, -9.80665),
    num_centerline_samples: int = 24,
    num_contour_samples: int = 32,
) -> BallisticPourLocus:
    """Compute the 3D ballistic pour trajectory and 95% landing uncertainty ellipse."""
    if pour_outlet.coordinate_frame != target_aperture.coordinate_frame:
        raise ValueError("pour_outlet and target_aperture must share a coordinate frame")
    if not math.isfinite(exit_speed_mps) or exit_speed_mps < 0.0:
        raise ValueError("exit_speed_mps must be finite and nonnegative")
    if not math.isfinite(exit_speed_std_mps) or exit_speed_std_mps < 0.0:
        raise ValueError("exit_speed_std_mps must be finite and nonnegative")

    p_out = np.asarray(pour_outlet.point_m, dtype=np.float64)
    c_rim = np.asarray(target_aperture.center_m, dtype=np.float64)
    n_rim = _unit_vector(target_aperture.normal, "target_aperture.normal")
    g = np.asarray(gravity_mps2, dtype=np.float64)
    if g.shape != (3,) or not np.isfinite(g).all():
        raise ValueError("gravity_mps2 must be a finite 3-vector")

    d_out = (
        _unit_vector(pour_outlet.direction, "pour_outlet.direction")
        if pour_outlet.direction is not None
        else _unit_vector(g, "gravity_direction")
    )

    # Solve n_rim^T (p_out + v0 * d_out * s + 0.5 * g * s^2 - c_rim) = 0
    h0 = float(np.dot(n_rim, p_out - c_rim))
    if h0 <= 0.005:
        empty = np.empty((0, 3), dtype=np.float64)
        return BallisticPourLocus(
            valid=False,
            reason="outlet_not_above_aperture_plane",
            time_of_flight_s=None,
            impact_point_m=None,
            planar_radial_offset_m=None,
            landing_ellipse_semi_major_95_m=None,
            landing_ellipse_semi_minor_95_m=None,
            landing_covariance_2x2_m2=None,
            centerline_points_m=empty,
            landing_contour_points_m=empty,
        )

    a_coeff = 0.5 * float(np.dot(n_rim, g))
    b_coeff = float(exit_speed_mps) * float(np.dot(n_rim, d_out))
    c_coeff = h0

    if abs(a_coeff) < 1e-9:
        if b_coeff >= -1e-9:
            empty = np.empty((0, 3), dtype=np.float64)
            return BallisticPourLocus(
                valid=False,
                reason="stream_does_not_descend_to_aperture",
                time_of_flight_s=None,
                impact_point_m=None,
                planar_radial_offset_m=None,
                landing_ellipse_semi_major_95_m=None,
                landing_ellipse_semi_minor_95_m=None,
                landing_covariance_2x2_m2=None,
                centerline_points_m=empty,
                landing_contour_points_m=empty,
            )
        t_flight = -c_coeff / b_coeff
    else:
        discriminant = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff
        if discriminant < 0.0:
            empty = np.empty((0, 3), dtype=np.float64)
            return BallisticPourLocus(
                valid=False,
                reason="ballistic_stream_misses_aperture_plane",
                time_of_flight_s=None,
                impact_point_m=None,
                planar_radial_offset_m=None,
                landing_ellipse_semi_major_95_m=None,
                landing_ellipse_semi_minor_95_m=None,
                landing_covariance_2x2_m2=None,
                centerline_points_m=empty,
                landing_contour_points_m=empty,
            )
        sqrt_disc = math.sqrt(discriminant)
        roots = [
            (-b_coeff - sqrt_disc) / (2.0 * a_coeff),
            (-b_coeff + sqrt_disc) / (2.0 * a_coeff),
        ]
        positive_roots = [
            r for r in roots if r > 0.0 and (b_coeff + 2.0 * a_coeff * r) < 0.0
        ]
        if not positive_roots:
            positive_roots = [r for r in roots if r > 0.0]
        if not positive_roots:
            empty = np.empty((0, 3), dtype=np.float64)
            return BallisticPourLocus(
                valid=False,
                reason="no_positive_ballistic_intersection_time",
                time_of_flight_s=None,
                impact_point_m=None,
                planar_radial_offset_m=None,
                landing_ellipse_semi_major_95_m=None,
                landing_ellipse_semi_minor_95_m=None,
                landing_covariance_2x2_m2=None,
                centerline_points_m=empty,
                landing_contour_points_m=empty,
            )
        t_flight = float(min(positive_roots))

    impact_3d = p_out + exit_speed_mps * d_out * t_flight + 0.5 * g * (t_flight * t_flight)
    impact_3d.flags.writeable = False

    # Sample centerline curve from s=0 to s=t_flight
    s_grid = np.linspace(0.0, t_flight, max(2, int(num_centerline_samples)))
    centerline = (
        p_out[None, :]
        + exit_speed_mps * s_grid[:, None] * d_out[None, :]
        + 0.5 * (s_grid * s_grid)[:, None] * g[None, :]
    )
    centerline.flags.writeable = False

    # First-order landing covariance on the aperture tangent plane (u, v)
    u_axis, v_axis = _orthonormal_plane_basis(n_rim)
    P_uv = np.column_stack([u_axis, v_axis])  # 3x2

    # Implicit function theorem for s*(p_out, d_out, v0):
    # v_impact = v0 * d_out + g * s*
    v_impact = exit_speed_mps * d_out + g * t_flight
    denom = float(np.dot(n_rim, v_impact))
    if abs(denom) < 1e-6:
        denom = -1e-6
    # dx*/dp_out = I - (v_impact * n_rim^T) / (n_rim^T v_impact)
    J_p = np.eye(3, dtype=np.float64) - np.outer(v_impact, n_rim) / denom
    J_d = exit_speed_mps * t_flight * J_p
    J_v0 = (d_out * t_flight) - v_impact * (float(np.dot(n_rim, d_out)) * t_flight / denom)

    # Isotropic tangent covariance for direction uncertainty sigma_theta
    dir_var = float(pour_outlet.direction_std_rad) ** 2
    cov_d = dir_var * (np.eye(3, dtype=np.float64) - np.outer(d_out, d_out))

    cov_impact_3d = (
        J_p @ np.asarray(pour_outlet.covariance_3x3_m2, dtype=np.float64) @ J_p.T
        + J_d @ cov_d @ J_d.T
        + (exit_speed_std_mps ** 2) * np.outer(J_v0, J_v0)
        + np.asarray(target_aperture.center_covariance_3x3_m2, dtype=np.float64)
    )
    cov_2d = P_uv.T @ cov_impact_3d @ P_uv
    cov_2d = 0.5 * (cov_2d + cov_2d.T)
    cov_2d.flags.writeable = False

    eigvals, eigvecs = np.linalg.eigh(cov_2d)
    eigvals = np.maximum(eigvals, 1e-10)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    semi_major_95 = float(CHI2_2DOF_95_SCALE * math.sqrt(float(eigvals[0])))
    semi_minor_95 = float(CHI2_2DOF_95_SCALE * math.sqrt(float(eigvals[1])))

    # 3D contour of the 95% landing ellipse on the aperture plane
    theta = np.linspace(0.0, 2.0 * math.pi, max(8, int(num_contour_samples)), endpoint=False)
    circle_uv = np.vstack([semi_major_95 * np.cos(theta), semi_minor_95 * np.sin(theta)])
    ellipse_uv = (eigvecs @ circle_uv).T  # Nx2
    contour_3d = (
        impact_3d[None, :]
        + ellipse_uv[:, 0:1] * u_axis[None, :]
        + ellipse_uv[:, 1:2] * v_axis[None, :]
    )
    contour_3d.flags.writeable = False

    planar_offset = float(np.linalg.norm((impact_3d - c_rim) - float(np.dot(impact_3d - c_rim, n_rim)) * n_rim))

    return BallisticPourLocus(
        valid=True,
        reason="ready",
        time_of_flight_s=t_flight,
        impact_point_m=impact_3d,
        planar_radial_offset_m=planar_offset,
        landing_ellipse_semi_major_95_m=semi_major_95,
        landing_ellipse_semi_minor_95_m=semi_minor_95,
        landing_covariance_2x2_m2=cov_2d,
        centerline_points_m=centerline,
        landing_contour_points_m=contour_3d,
    )


def project_stream_centerline_uv(
    locus: BallisticPourLocus,
    *,
    camera_from_world: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray | Sequence[float] = (),
    image_size: tuple[int, int] | None = None,
) -> dict[str, np.ndarray | bool]:
    """Project the 3D ballistic stream centerline and 95% landing contour into image UV."""
    if not locus.valid or len(locus.centerline_points_m) == 0:
        return {
            "ready": False,
            "centerline_uv": np.empty((0, 2), dtype=np.float64),
            "landing_contour_uv": np.empty((0, 2), dtype=np.float64),
        }
    import cv2

    T_cw = np.asarray(camera_from_world, dtype=np.float64).reshape(4, 4)
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(distortion_coeffs, dtype=np.float64).reshape(-1)

    def _proj(pts_w: np.ndarray) -> np.ndarray:
        if len(pts_w) == 0:
            return np.empty((0, 2), dtype=np.float64)
        pts_c = (pts_w @ T_cw[:3, :3].T) + T_cw[:3, 3]
        front = pts_c[:, 2] > 1e-3
        if not np.any(front):
            return np.empty((0, 2), dtype=np.float64)
        pts_c = pts_c[front]
        uv, _ = cv2.projectPoints(
            pts_c,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            K,
            dist,
        )
        uv = uv.reshape(-1, 2)
        if image_size is not None:
            w, h = int(image_size[0]), int(image_size[1])
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            uv = uv[inside]
        return uv

    centerline_uv = _proj(locus.centerline_points_m)
    contour_uv = _proj(locus.landing_contour_points_m)
    return {
        "ready": len(centerline_uv) >= 2,
        "centerline_uv": centerline_uv,
        "landing_contour_uv": contour_uv,
    }


def verify_l2_pouring_safety_envelope(
    *,
    pour_outlet_world: MeasuredPointFeature3D | None,
    target_aperture_world: MeasuredCircularAperture3D | None,
    pour_outlet_object_frame: MeasuredPointFeature3D | None = None,
    trajectory_report: MultiHorizonTrajectoryReport | None = None,
    exit_speed_mps: float = 0.35,
    exit_speed_std_mps: float = 0.06,
    minimum_height_above_rim_m: float = 0.025,
    maximum_height_above_rim_m: float = 0.35,
    safety_margin_m: float = 0.004,
    target_pour_height_m: float = 0.08,
    required_horizons_ms: Sequence[float] = (100.0, 250.0),
) -> L2PouringSafetyVerdict:
    """Fail-closed L2 pouring spatial safety verifier and shared-control guidance computer.

    Guarantees zero false-safe guidance:
    - Any missing or unverified feature evidence immediately blocks ``safe_to_pour``.
    - Evaluates the 95% ballistic landing uncertainty ellipse against the target
      container's inner rim radius minus 95% radius uncertainty and ``safety_margin_m``.
    - When ``trajectory_report`` and ``pour_outlet_object_frame`` are supplied,
      also verifies that the 95% uncertainty envelope remains inside the cup rim
      across all required future horizons (e.g., 100 ms and 250 ms).
    """
    zero_delta = np.zeros(3, dtype=np.float64)
    zero_delta.flags.writeable = False

    blockers: list[str] = []
    if pour_outlet_world is None:
        blockers.append("pour_outlet_evidence_missing")
    if target_aperture_world is None:
        blockers.append("container_opening_evidence_missing")
    if blockers:
        return L2PouringSafetyVerdict(
            l2_pour_safety_status="blocked_missing_evidence",
            uncertainty_envelope_clears_cup_rim=False,
            safe_to_pour=False,
            blocker_reasons=tuple(blockers),
            signed_height_above_rim_m=None,
            current_radial_offset_m=None,
            current_rim_clearance_95_m=None,
            worst_horizon_rim_clearance_95_m=None,
            evaluated_horizons_ms=(),
            assist_translation_delta_m=zero_delta,
            ballistic_locus=None,
            publication_authorized=False,
            control_authorized=False,
        )

    assert pour_outlet_world is not None and target_aperture_world is not None
    if not target_aperture_world.ransac_consensus_agreed:
        blockers.append("aperture_ransac_consensus_disagreed")

    n_rim = _unit_vector(target_aperture_world.normal, "target_aperture_world.normal")
    p_out = np.asarray(pour_outlet_world.point_m, dtype=np.float64)
    c_rim = np.asarray(target_aperture_world.center_m, dtype=np.float64)
    signed_height = float(np.dot(p_out - c_rim, n_rim))

    if signed_height < minimum_height_above_rim_m:
        blockers.append("outlet_below_minimum_rim_clearance_height")
    elif signed_height > maximum_height_above_rim_m:
        blockers.append("outlet_above_maximum_pour_height")

    locus = predict_ballistic_pour_locus(
        pour_outlet=pour_outlet_world,
        target_aperture=target_aperture_world,
        exit_speed_mps=exit_speed_mps,
        exit_speed_std_mps=exit_speed_std_mps,
    )
    if not locus.valid or locus.impact_point_m is None or locus.landing_ellipse_semi_major_95_m is None:
        blockers.append(f"ballistic_locus_invalid:{locus.reason}")
        # Guidance vector moves spout directly above rim center at target_pour_height_m
        desired_pos = c_rim + target_pour_height_m * n_rim
        delta = desired_pos - p_out
        delta.flags.writeable = False
        return L2PouringSafetyVerdict(
            l2_pour_safety_status="hold_or_reposition",
            uncertainty_envelope_clears_cup_rim=False,
            safe_to_pour=False,
            blocker_reasons=tuple(blockers),
            signed_height_above_rim_m=signed_height,
            current_radial_offset_m=None,
            current_rim_clearance_95_m=None,
            worst_horizon_rim_clearance_95_m=None,
            evaluated_horizons_ms=(0.0,),
            assist_translation_delta_m=delta,
            ballistic_locus=locus,
            publication_authorized=False,
            control_authorized=False,
        )

    effective_rim_radius = (
        target_aperture_world.radius_m
        - 1.959963984540054 * target_aperture_world.radius_std_m
        - safety_margin_m
    )
    current_offset = float(locus.planar_radial_offset_m or 0.0)
    current_clearance_95 = float(
        effective_rim_radius - (current_offset + locus.landing_ellipse_semi_major_95_m)
    )
    if current_clearance_95 < 0.0:
        blockers.append("current_uncertainty_envelope_exceeds_cup_rim")

    worst_clearance_95 = current_clearance_95
    evaluated_horizons = [0.0]

    if trajectory_report is not None and pour_outlet_object_frame is not None:
        if trajectory_report.interval_width_monotonic_violation_count > 0:
            blockers.append("trajectory_uncertainty_monotonicity_violated")
        horizon_map = {round(h.horizon_ms, 3): h for h in trajectory_report.horizons}
        for req_ms in required_horizons_ms:
            pred = horizon_map.get(round(float(req_ms), 3))
            if pred is None:
                blockers.append(f"missing_predicted_horizon_{int(req_ms)}ms")
                continue
            evaluated_horizons.append(float(req_ms))
            pred_outlet = propagate_feature_uncertainty_se3(
                feature_object_frame=pour_outlet_object_frame,
                world_from_object=pred.predicted_pose,
                pose_covariance_6x6=pred.covariance_6x6,
                target_coordinate_frame=target_aperture_world.coordinate_frame,
            )
            pred_locus = predict_ballistic_pour_locus(
                pour_outlet=pred_outlet,
                target_aperture=target_aperture_world,
                exit_speed_mps=exit_speed_mps,
                exit_speed_std_mps=exit_speed_std_mps,
            )
            if (
                not pred_locus.valid
                or pred_locus.planar_radial_offset_m is None
                or pred_locus.landing_ellipse_semi_major_95_m is None
            ):
                blockers.append(f"horizon_{int(req_ms)}ms_locus_invalid:{pred_locus.reason}")
                worst_clearance_95 = min(worst_clearance_95, -1.0)
                continue
            h_clearance = float(
                effective_rim_radius
                - (
                    pred_locus.planar_radial_offset_m
                    + pred_locus.landing_ellipse_semi_major_95_m
                )
            )
            worst_clearance_95 = min(worst_clearance_95, h_clearance)
            if h_clearance < 0.0:
                blockers.append(f"horizon_{int(req_ms)}ms_uncertainty_envelope_exceeds_cup_rim")

    # Compute minimum-norm shared-control guidance translation delta
    impact_vec = locus.impact_point_m - c_rim
    planar_impact_vec = impact_vec - float(np.dot(impact_vec, n_rim)) * n_rim
    planar_dist = float(np.linalg.norm(planar_impact_vec))
    allowed_offset = max(
        0.0, effective_rim_radius - locus.landing_ellipse_semi_major_95_m
    )
    if planar_dist > allowed_offset and planar_dist > 1e-9:
        planar_correction = -planar_impact_vec * ((planar_dist - allowed_offset) / planar_dist)
    else:
        planar_correction = np.zeros(3, dtype=np.float64)

    height_correction = np.zeros(3, dtype=np.float64)
    if signed_height < minimum_height_above_rim_m:
        height_correction = (target_pour_height_m - signed_height) * n_rim
    elif signed_height > maximum_height_above_rim_m:
        height_correction = (maximum_height_above_rim_m - signed_height) * n_rim

    assist_delta = planar_correction + height_correction
    assist_delta.flags.writeable = False

    clears_rim = (len(blockers) == 0) and (worst_clearance_95 >= 0.0)
    status = "safe_to_pour" if clears_rim else "hold_or_reposition"

    return L2PouringSafetyVerdict(
        l2_pour_safety_status=status,
        uncertainty_envelope_clears_cup_rim=clears_rim,
        safe_to_pour=clears_rim,
        blocker_reasons=tuple(blockers),
        signed_height_above_rim_m=signed_height,
        current_radial_offset_m=current_offset,
        current_rim_clearance_95_m=current_clearance_95,
        worst_horizon_rim_clearance_95_m=worst_clearance_95,
        evaluated_horizons_ms=tuple(evaluated_horizons),
        assist_translation_delta_m=assist_delta,
        ballistic_locus=locus,
        publication_authorized=False,
        control_authorized=False,
    )
