"""Fail-closed camera-local boundary for causal long-range point trackers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping

import numpy as np

from causalpose3d.primitives.content_hash import content_sha256


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase hexadecimal SHA-256")
    return value


def _frozen_array(
    value: object,
    *,
    dtype: np.dtype,
    name: str,
    shape: tuple[int, ...] | None = None,
    last_dimension: int | None = None,
) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != dtype:
        raise ValueError(f"{name} must be a {dtype.str} ndarray")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if last_dimension is not None and (
        value.ndim < 1 or value.shape[-1] != last_dimension
    ):
        raise ValueError(f"{name} has an invalid shape")
    if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    result = np.ndarray(
        value.shape,
        dtype=dtype,
        buffer=np.ascontiguousarray(value).tobytes(),
    )
    result.flags.writeable = False
    return result


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _diagnostics(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("diagnostics must be a mapping")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key
            or any(token in key.lower() for token in ("world", "pose"))
        ):
            raise ValueError("world or pose diagnostics are forbidden")
        if isinstance(item, bool) or item is None or isinstance(item, (str, int)):
            result[key] = item
        elif isinstance(item, float) and math.isfinite(item):
            result[key] = item
        else:
            raise ValueError("diagnostics must contain finite JSON scalars")
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True)
class CausalPointFrame:
    """Immutable current RGB observation; it contains no world-state input."""

    source_sha256: str
    color_rgb_sha256: str
    camera_id: str
    timestamp_ns: int
    color_rgb: np.ndarray
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        source_sha256 = _sha256(self.source_sha256, "source hash")
        color_sha256 = _sha256(self.color_rgb_sha256, "color hash")
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("camera_id is required")
        if type(self.timestamp_ns) is not int or self.timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be a positive integer")
        color = _frozen_array(
            self.color_rgb,
            dtype=np.dtype("u1"),
            name="color_rgb",
            last_dimension=3,
        )
        if color.ndim != 3 or min(color.shape[:2]) <= 0:
            raise ValueError("color_rgb must be a non-empty HxWx3 image")
        if _array_sha256(color) != color_sha256:
            raise ValueError("color hash does not match actual bytes")
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "color_rgb_sha256", color_sha256)
        object.__setattr__(self, "color_rgb", color)
        object.__setattr__(
            self,
            "content_hash",
            content_sha256(
                {
                    "schema": "ticos.causal-point-frame.v1",
                    "source_sha256": source_sha256,
                    "color_rgb_sha256": color_sha256,
                    "camera_id": self.camera_id,
                    "timestamp_ns": self.timestamp_ns,
                    "shape": list(color.shape),
                }
            ),
        )

    @classmethod
    def from_rgb(
        cls,
        *,
        source_sha256: str,
        camera_id: str,
        timestamp_ns: int,
        color_rgb: np.ndarray,
    ) -> "CausalPointFrame":
        return cls(
            source_sha256=source_sha256,
            color_rgb_sha256=_array_sha256(color_rgb),
            camera_id=camera_id,
            timestamp_ns=timestamp_ns,
            color_rgb=color_rgb,
        )


@dataclass(frozen=True)
class CausalPointTracks:
    """Current camera-local point observations with immutable query identity."""

    status: Literal["candidate", "not_observable"]
    reason: str
    frame_content_hash: str
    model_sha256: str
    camera_id: str
    timestamp_ns: int
    query_ids: np.ndarray
    positions_xy: np.ndarray
    visible: np.ndarray
    confidence: np.ndarray
    inference_latency_ms: float
    diagnostics: Mapping[str, Any]
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        frame_hash = _sha256(self.frame_content_hash, "frame content hash")
        model_hash = _sha256(self.model_sha256, "model hash")
        if self.status not in {"candidate", "not_observable"}:
            raise ValueError("invalid point-track status")
        if not isinstance(self.reason, str):
            raise ValueError("point-track reason must be text")
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("camera_id is required")
        if type(self.timestamp_ns) is not int or self.timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be positive")
        query_ids = _frozen_array(
            self.query_ids, dtype=np.dtype("<i8"), name="query_ids"
        )
        positions = _frozen_array(
            self.positions_xy,
            dtype=np.dtype("<f8"),
            name="positions_xy",
            last_dimension=2,
        )
        visible = _frozen_array(
            self.visible, dtype=np.dtype("?"), name="visible"
        )
        confidence = _frozen_array(
            self.confidence, dtype=np.dtype("<f8"), name="confidence"
        )
        if (
            query_ids.ndim != 1
            or positions.ndim != 2
            or visible.ndim != 1
            or confidence.ndim != 1
            or not len(query_ids) == len(positions) == len(visible) == len(confidence)
        ):
            raise ValueError("point-track arrays must share a one-dimensional query axis")
        if np.any(confidence < 0.0) or np.any(confidence > 1.0):
            raise ValueError("point confidence must be in [0,1]")
        if not isinstance(self.inference_latency_ms, (int, float)) or not math.isfinite(
            float(self.inference_latency_ms)
        ) or float(self.inference_latency_ms) < 0.0:
            raise ValueError("inference latency must be finite and nonnegative")
        diagnostics = _diagnostics(self.diagnostics)
        if self.status == "candidate":
            if self.reason or not len(query_ids) or not np.any(visible):
                raise ValueError("candidate tracks require current visible observations")
        elif not self.reason:
            raise ValueError("not_observable tracks require a reason")
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "positions_xy", positions)
        object.__setattr__(self, "visible", visible)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "inference_latency_ms", float(self.inference_latency_ms))
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(
            self,
            "content_hash",
            content_sha256(
                {
                    "schema": "ticos.causal-point-tracks.v1",
                    "status": self.status,
                    "reason": self.reason,
                    "frame_content_hash": frame_hash,
                    "model_sha256": model_hash,
                    "camera_id": self.camera_id,
                    "timestamp_ns": self.timestamp_ns,
                    "query_ids": query_ids.tolist(),
                    "positions_xy": positions.tolist(),
                    "visible": visible.tolist(),
                    "confidence": confidence.tolist(),
                    "inference_latency_ms": float(self.inference_latency_ms),
                    "diagnostics": dict(diagnostics),
                }
            ),
        )


PointTrackerRunner = Callable[
    [np.ndarray, np.ndarray | None, object | None], Mapping[str, Any]
]


class CausalPointTrackerProvider:
    """Own one camera's recurrent state and reject replay/future/identity changes."""

    def __init__(
        self,
        *,
        model_sha256: str,
        runner: PointTrackerRunner,
        maximum_query_points: int = 1024,
        minimum_visible_points: int = 1,
    ) -> None:
        self._model_sha256 = _sha256(model_sha256, "model hash")
        if maximum_query_points <= 0:
            raise ValueError("maximum_query_points must be positive")
        if minimum_visible_points <= 0 or minimum_visible_points > maximum_query_points:
            raise ValueError("minimum_visible_points is outside the query budget")
        self._runner = runner
        self._maximum_query_points = int(maximum_query_points)
        self._minimum_visible_points = int(minimum_visible_points)
        self._camera_id: str | None = None
        self._query_ids: np.ndarray | None = None
        self._state: object | None = None
        self._last_timestamp_ns: int | None = None

    def start(
        self, frame: CausalPointFrame, query_points_xy: np.ndarray
    ) -> CausalPointTracks:
        if self._query_ids is not None:
            raise ValueError("causal point provider is already initialized")
        points = _frozen_array(
            query_points_xy,
            dtype=np.dtype("<f8"),
            name="query_points_xy",
            last_dimension=2,
        )
        if points.ndim != 2 or not 0 < len(points) <= self._maximum_query_points:
            raise ValueError("query point count is outside the frozen budget")
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("query points must be unique")
        height, width = frame.color_rgb.shape[:2]
        if np.any(points[:, 0] < 0.0) or np.any(points[:, 0] >= width) or np.any(
            points[:, 1] < 0.0
        ) or np.any(points[:, 1] >= height):
            raise ValueError("query points must lie inside the current image")
        self._camera_id = frame.camera_id
        self._query_ids = np.arange(len(points), dtype=np.int64)
        self._query_ids.flags.writeable = False
        return self._run(frame, points)

    def track(self, frame: CausalPointFrame) -> CausalPointTracks:
        if self._query_ids is None:
            raise ValueError("causal point provider must be initialized")
        return self._run(frame, None)

    def _run(
        self, frame: CausalPointFrame, query_points_xy: np.ndarray | None
    ) -> CausalPointTracks:
        if not isinstance(frame, CausalPointFrame):
            raise ValueError("frame must be CausalPointFrame")
        if self._camera_id is not None and frame.camera_id != self._camera_id:
            raise ValueError("one provider cannot mix physical cameras")
        if self._last_timestamp_ns is not None and frame.timestamp_ns <= self._last_timestamp_ns:
            raise ValueError("causal point frames must be strictly increasing")
        self._last_timestamp_ns = frame.timestamp_ns
        assert self._query_ids is not None
        try:
            raw = self._runner(frame.color_rgb, query_points_xy, self._state)
            if not isinstance(raw, Mapping) or set(raw) != {
                "positions_xy",
                "visible",
                "confidence",
                "state",
                "inference_latency_ms",
                "diagnostics",
            }:
                raise ValueError("point runner output has unsupported fields")
            if any(
                token in str(key).lower()
                for key in raw
                for token in ("world", "pose")
            ):
                raise ValueError("point runner cannot emit world or pose state")
            positions = np.asarray(raw["positions_xy"], dtype=np.float64)
            visible = np.asarray(raw["visible"], dtype=np.bool_)
            confidence = np.asarray(raw["confidence"], dtype=np.float64)
            if positions.ndim != 2 or positions.shape[1:] != (2,):
                raise ValueError("point positions must have shape Nx2")
            if visible.ndim != 1 or confidence.ndim != 1:
                raise ValueError("point visibility and confidence must be vectors")
            if not len(positions) == len(visible) == len(confidence) == len(
                self._query_ids
            ):
                raise ValueError("point runner changed query identity cardinality")
            if not np.isfinite(positions).all() or not np.isfinite(confidence).all():
                raise ValueError("point runner output must be finite")
            if np.any(confidence < 0.0) or np.any(confidence > 1.0):
                raise ValueError("point confidence must be in [0,1]")
            latency_ms = float(raw["inference_latency_ms"])
            if not math.isfinite(latency_ms) or latency_ms < 0.0:
                raise ValueError("point latency must be finite and nonnegative")
            diagnostics = _diagnostics(raw["diagnostics"])
            height, width = frame.color_rgb.shape[:2]
            in_bounds = (
                (positions[:, 0] >= 0.0)
                & (positions[:, 0] < width)
                & (positions[:, 1] >= 0.0)
                & (positions[:, 1] < height)
            )
            visible = visible & in_bounds
            if int(np.sum(visible)) < self._minimum_visible_points:
                self._state = raw["state"]
                return self._unavailable(frame, "insufficient_visible_points")
            candidate = CausalPointTracks(
                status="candidate",
                reason="",
                frame_content_hash=frame.content_hash,
                model_sha256=self._model_sha256,
                camera_id=frame.camera_id,
                timestamp_ns=frame.timestamp_ns,
                query_ids=self._query_ids,
                positions_xy=positions,
                visible=visible,
                confidence=confidence,
                inference_latency_ms=latency_ms,
                diagnostics=diagnostics,
            )
            self._state = raw["state"]
            return candidate
        except (IndexError, KeyError, TypeError, ValueError):
            return self._unavailable(frame, "provider_output_invalid")

    def _unavailable(
        self, frame: CausalPointFrame, reason: str
    ) -> CausalPointTracks:
        assert self._query_ids is not None
        count = len(self._query_ids)
        return CausalPointTracks(
            status="not_observable",
            reason=reason,
            frame_content_hash=frame.content_hash,
            model_sha256=self._model_sha256,
            camera_id=frame.camera_id,
            timestamp_ns=frame.timestamp_ns,
            query_ids=self._query_ids,
            positions_xy=np.zeros((count, 2), dtype=np.float64),
            visible=np.zeros(count, dtype=np.bool_),
            confidence=np.zeros(count, dtype=np.float64),
            inference_latency_ms=0.0,
            diagnostics={},
        )
