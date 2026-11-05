"""Compact row-major masks shared by detectors and metric projection."""

from __future__ import annotations

from typing import Any

import numpy as np


def encode_mask_rle(mask: np.ndarray) -> dict[str, list[int]]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    flat = values.reshape(-1)
    if not flat.size:
        counts: list[int] = []
    else:
        # Find both run boundaries in one scan; preserve row-major wire bytes.
        edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        if flat[0]:
            edges = np.concatenate(([0], edges))
        if flat[-1]:
            edges = np.concatenate((edges, [flat.size]))
        runs = edges.reshape(-1, 2)
        counts = np.column_stack((runs[:, 0], runs[:, 1] - runs[:, 0])).reshape(-1).tolist()
    return {"size": [int(values.shape[0]), int(values.shape[1])], "counts": counts}


def decode_mask_rle(payload: Any, *, width: int, height: int) -> np.ndarray | None:
    if not isinstance(payload, dict):
        return None
    size = payload.get("size")
    counts = payload.get("counts")
    if not (
        isinstance(size, list)
        and len(size) == 2
        and int(size[0]) == int(height)
        and int(size[1]) == int(width)
        and isinstance(counts, list)
        and len(counts) % 2 == 0
    ):
        return None
    flat = np.zeros(int(width) * int(height), dtype=bool)
    for raw_start, raw_length in zip(counts[0::2], counts[1::2], strict=False):
        if not isinstance(raw_start, int) or not isinstance(raw_length, int) or raw_length <= 0:
            return None
        end = raw_start + raw_length
        if raw_start < 0 or end > len(flat):
            return None
        flat[raw_start:end] = True
    return flat.reshape((int(height), int(width)))


try:
    from ticos_vision.perception.image_mask import (  # type: ignore[import-not-found]
        decode_mask_rle,
        encode_mask_rle,
    )
except Exception:
    pass

