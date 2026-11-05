"""Bounded causal SE(3) prediction for short explicit observation failures."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class CausalSe3Prediction:
    pose: np.ndarray
    horizon_ms: float
    translation_speed_mps: float
    angular_speed_deg_s: float
    translation_clipped: bool
    rotation_clipped: bool


def _pose(value: np.ndarray, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
        or not math.isclose(float(np.linalg.det(pose[:3, :3])), 1.0, abs_tol=1e-6)
    ):
        raise ValueError(f"{name} must be one finite SE(3) matrix")
    return pose


def predict_bounded_constant_velocity_se3(
    *,
    older_pose: np.ndarray,
    older_timestamp_ns: int,
    latest_pose: np.ndarray,
    latest_timestamp_ns: int,
    target_timestamp_ns: int,
    maximum_prediction_horizon_ms: float = 150.0,
    maximum_translation_speed_mps: float = 1.0,
    maximum_angular_speed_deg_s: float = 180.0,
) -> CausalSe3Prediction:
    """Extrapolate from two strictly past poses with physical velocity caps."""

    older = _pose(older_pose, "older_pose")
    latest = _pose(latest_pose, "latest_pose")
    timestamps = (
        int(older_timestamp_ns),
        int(latest_timestamp_ns),
        int(target_timestamp_ns),
    )
    if not timestamps[0] < timestamps[1] < timestamps[2]:
        raise ValueError("prediction timestamps must increase strictly")
    limits = (
        maximum_prediction_horizon_ms,
        maximum_translation_speed_mps,
        maximum_angular_speed_deg_s,
    )
    if not all(math.isfinite(float(value)) and value > 0.0 for value in limits):
        raise ValueError("prediction limits must be finite and positive")

    history_dt_s = (timestamps[1] - timestamps[0]) / 1e9
    requested_horizon_s = (timestamps[2] - timestamps[1]) / 1e9
    horizon_s = min(
        requested_horizon_s, float(maximum_prediction_horizon_ms) / 1000.0
    )
    translation_velocity = (
        latest[:3, 3] - older[:3, 3]
    ) / history_dt_s
    translation_speed = float(np.linalg.norm(translation_velocity))
    translation_clipped = translation_speed > maximum_translation_speed_mps
    if translation_clipped:
        translation_velocity *= maximum_translation_speed_mps / translation_speed
        translation_speed = float(maximum_translation_speed_mps)

    relative_rotation = Rotation.from_matrix(
        older[:3, :3].T @ latest[:3, :3]
    )
    angular_velocity = relative_rotation.as_rotvec() / history_dt_s
    angular_speed_rad_s = float(np.linalg.norm(angular_velocity))
    maximum_angular_speed_rad_s = math.radians(maximum_angular_speed_deg_s)
    rotation_clipped = angular_speed_rad_s > maximum_angular_speed_rad_s
    if rotation_clipped:
        angular_velocity *= maximum_angular_speed_rad_s / angular_speed_rad_s
        angular_speed_rad_s = maximum_angular_speed_rad_s

    predicted = np.eye(4, dtype=np.float64)
    predicted[:3, 3] = (
        latest[:3, 3] + translation_velocity * horizon_s
    )
    predicted[:3, :3] = latest[:3, :3] @ Rotation.from_rotvec(
        angular_velocity * horizon_s
    ).as_matrix()
    predicted.flags.writeable = False
    return CausalSe3Prediction(
        pose=predicted,
        horizon_ms=horizon_s * 1000.0,
        translation_speed_mps=translation_speed,
        angular_speed_deg_s=math.degrees(angular_speed_rad_s),
        translation_clipped=translation_clipped,
        rotation_clipped=rotation_clipped,
    )
