"""Build shadow pose-owner A/B arms from a registered RGB-D + mask sequence.

Uses an existing BundleTrack/BundleSDF ``ob_in_cam`` pose directory as the
baseline arm and ``mask_depth_pca_pose`` as the shadow arm. Neither arm
authorizes the teleop-assist wire.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from causalpose3d.observability.shadow_pose_owner_ab import evaluate_shadow_pose_owner_ab
from causalpose3d.primitives.mask_depth_pca_pose import (
    DEFAULT_MAXIMUM_TRANSLATION_STEP_M,
    DEFAULT_MINIMUM_DEPTH_SUPPORT_PIXELS,
    advance_mask_depth_pca_pose,
    estimate_mask_depth_centroid_camera,
    recover_camera_from_object_with_prior,
)

SCHEMA_INPUT = "ticos.shadow-pose-owner-ab-registered-sequence.v1"


def _load_camera_matrix(path: Path) -> np.ndarray:
    values = [
        float(token)
        for token in path.read_text(encoding="utf-8").replace(",", " ").split()
        if token
    ]
    if len(values) != 9 or any(not math.isfinite(value) for value in values):
        raise ValueError(f"camera matrix must contain nine finite values: {path}")
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def _load_depth_m(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        if source.format != "PNG":
            raise ValueError(f"depth must be a millimetre PNG: {path}")
        image = np.asarray(source)
    if (image.ndim != 2 or not np.issubdtype(image.dtype, np.integer)
            or np.any(image < 0)):
        raise ValueError(f"depth must be a nonnegative integer single-channel image: {path}")
    # BundleSDF registered exports use millimetres even when a frame contains
    # only zeros or very small depth values; magnitudes do not identify units.
    return image.astype(np.float64) / 1000.0


def _load_mask(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path))
    if image.ndim == 3:
        image = image[..., 0]
    return image > 0


def _load_pose_txt(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    values = [
        float(token)
        for token in path.read_text(encoding="utf-8").replace(",", " ").split()
        if token
    ]
    if len(values) != 16:
        return None
    matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(matrix).all():
        return None
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-5):
        return None
    return matrix


def _frame_entries(sequence_dir: Path) -> list[dict[str, Any]]:
    manifest_path = sequence_dir / "manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"manifest must be an object: {manifest_path}")
        frames = payload.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"manifest frames missing: {manifest_path}")
        if any(not isinstance(frame, Mapping) for frame in frames):
            raise ValueError(f"manifest contains invalid frame entries: {manifest_path}")
        declared_count = payload.get("frame_count")
        if declared_count is not None and (
            type(declared_count) is not int or declared_count != len(frames)
        ):
            raise ValueError(f"manifest frame count mismatch: {manifest_path}")
        indices = [frame.get("output_index") for frame in frames]
        if (any(type(index) is not int or index < 0 for index in indices)
                or any(right <= left for left, right in zip(indices, indices[1:]))):
            raise ValueError(f"manifest frame indices invalid: {manifest_path}")
        return [dict(frame) for frame in frames]
    depth_dir = sequence_dir / "depth"
    mask_dir = sequence_dir / "masks"
    names = sorted(path.stem for path in depth_dir.glob("*.png"))
    if not names:
        raise ValueError(f"no depth frames under {depth_dir}")
    mask_names = {path.stem for path in mask_dir.glob("*.png")}
    if set(names) != mask_names:
        raise ValueError("depth and mask frame filenames do not match")
    return [
        {
            "output_index": index,
            "depth_path": f"depth/{name}.png",
            "mask_path": f"masks/{name}.png",
        }
        for index, name in enumerate(names)
    ]


def _timestamp_ns(value: Any, *, positive: bool) -> int:
    if type(value) is int:
        number = value
    elif (isinstance(value, str) and value.isascii() and value.isdecimal()
          and len(value) <= 20):
        number = int(value)
    else:
        raise ValueError("timestamp must be an unsigned integer nanosecond value")
    if number < (1 if positive else 0) or number > 2**64 - 1:
        raise ValueError("timestamp outside supported range")
    return number


def _frame_clocks(frame: Mapping[str, Any], index: int) -> tuple[int, int, dict[str, Any]]:
    has_color = "color_timestamp_ns" in frame
    has_depth = "depth_timestamp_ns" in frame
    if has_color != has_depth:
        raise ValueError("paired color and depth timestamps are required together")
    if has_color:
        color = _timestamp_ns(frame["color_timestamp_ns"], positive=True)
        depth = _timestamp_ns(frame["depth_timestamp_ns"], positive=True)
        return color, depth, {
            "source_clock_kind": "recorded_color_depth_pair",
            "color_timestamp_ns": str(color), "depth_timestamp_ns": str(depth),
            "source_timestamp_ns": color,
        }
    if "timestamp_ns" in frame:
        legacy = _timestamp_ns(frame["timestamp_ns"], positive=False)
        return legacy, legacy, {"source_clock_kind": "legacy_single_clock_unverified",
                                "source_timestamp_ns": legacy}
    return index, index, {"source_clock_kind": "frame_index_only"}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _baseline_receipt(sequence_dir: Path, path: Path, frames: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Validate a caller-supplied archive binding, not independent provenance."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = sequence_dir / "manifest.json"
    if (not isinstance(payload, dict) or payload.get("schema") != "ticos.archive-baseline-status.v1"
            or not manifest_path.is_file()
            or payload.get("sequence_manifest_sha256") != _sha256_path(manifest_path)
            or payload.get("camera_matrix_sha256") != _sha256_path(sequence_dir / "cam_K.txt")):
        raise ValueError("baseline receipt source binding invalid")
    rows = payload.get("frames")
    if not isinstance(rows, list) or len(rows) != len(frames):
        raise ValueError("baseline receipt frame count mismatch")
    result = {}
    for frame, row in zip(frames, rows):
        if not isinstance(row, dict):
            raise ValueError("baseline receipt frame identity or status invalid")
        _, _, clocks = _frame_clocks(frame, frame["output_index"])
        kind = clocks["source_clock_kind"]
        if kind == "recorded_color_depth_pair":
            clock_matches = (
                row.get("source_clock_kind", kind) == kind
                and _timestamp_ns(row.get("color_timestamp_ns"), positive=True)
                == _timestamp_ns(frame.get("color_timestamp_ns"), positive=True)
                and ("depth_timestamp_ns" not in row
                     or _timestamp_ns(row["depth_timestamp_ns"], positive=True)
                     == _timestamp_ns(frame.get("depth_timestamp_ns"), positive=True))
            )
        elif kind == "legacy_single_clock_unverified":
            clock_matches = (
                row.get("source_clock_kind") == kind
                and _timestamp_ns(row.get("source_timestamp_ns"), positive=True)
                == _timestamp_ns(frame.get("timestamp_ns"), positive=True)
            )
        else:
            clock_matches = False
        if (not isinstance(row, dict) or row.get("frame_index") != frame["output_index"]
                or type(row.get("frame_index")) is not int
                or not clock_matches
                or row.get("tracking_status") not in ("candidate", "fail", "no_ba")):
            raise ValueError("baseline receipt frame identity or status invalid")
        digest = row.get("pose_file_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("baseline receipt pose digest invalid")
        for kind in ("rgb", "depth", "mask"):
            source_path, expected = frame.get(f"{kind}_path"), frame.get(f"{kind}_sha256")
            if (not isinstance(source_path, str) or not source_path
                    or not isinstance(expected, str) or len(expected) != 64
                    or _sha256_path(sequence_dir / source_path) != expected):
                raise ValueError("baseline receipt input image binding invalid")
        result[row["frame_index"]] = row
    return result


def build_registered_sequence_arms(
    *,
    sequence_dir: Path,
    baseline_pose_dir: Path,
    minimum_depth_support_pixels: int = DEFAULT_MINIMUM_DEPTH_SUPPORT_PIXELS,
    baseline_keep_every: int = 1,
    maximum_recovery_translation_step_m: float = DEFAULT_MAXIMUM_TRANSLATION_STEP_M,
    assume_pose_file_candidate: bool = False,
    baseline_status_receipt: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return baseline BundleTrack, sparse BT, recovery, and PCA tick arms.

    ``baseline_keep_every`` keeps every Nth eligible archived pose for the
    sparse and recovery arms (1 keeps all). File existence does not establish
    tracking status. The opt-in assumption is a synthetic diagnostic, never
    evidence of live duty or freshness.
    """
    if type(baseline_keep_every) is not int or baseline_keep_every <= 0:
        raise ValueError("baseline_keep_every must be a positive int")
    if type(assume_pose_file_candidate) is not bool:
        raise ValueError("assume_pose_file_candidate must be a boolean")
    if assume_pose_file_candidate and baseline_status_receipt is not None:
        raise ValueError("choose receipt or unverified file-presence assumption")
    sequence_dir = sequence_dir.resolve()
    baseline_pose_dir = baseline_pose_dir.resolve()
    camera_matrix = _load_camera_matrix(sequence_dir / "cam_K.txt")
    baseline: list[dict[str, Any]] = []
    sparse: list[dict[str, Any]] = []
    recovered: list[dict[str, Any]] = []
    shadow: list[dict[str, Any]] = []
    last_pose: np.ndarray | None = None
    last_mask_centroid: np.ndarray | None = None
    previous_clocks: tuple[int, int] | None = None
    previous_clock_kind: str | None = None
    frames = _frame_entries(sequence_dir)
    receipt = (_baseline_receipt(sequence_dir, baseline_status_receipt, frames)
               if baseline_status_receipt is not None else None)
    for frame in frames:
        index = int(frame.get("output_index", len(baseline)))
        depth_rel = str(frame.get("depth_path") or "")
        mask_rel = str(frame.get("mask_path") or "")
        depth = _load_depth_m(sequence_dir / depth_rel)
        mask = _load_mask(sequence_dir / mask_rel)
        if depth.shape != mask.shape:
            raise ValueError(f"depth and mask grids differ at frame {index}")
        color_ns, depth_ns, clock_evidence = _frame_clocks(frame, index)
        clock_kind = clock_evidence["source_clock_kind"]
        if previous_clock_kind is not None and clock_kind != previous_clock_kind:
            raise ValueError("clock representation changes within a sequence")
        if (clock_kind == "recorded_color_depth_pair" and previous_clocks is not None
                and (color_ns <= previous_clocks[0] or depth_ns <= previous_clocks[1])):
            raise ValueError("paired sensor timestamps must strictly increase")
        previous_clocks, previous_clock_kind = (color_ns, depth_ns), clock_kind
        support = int(np.count_nonzero(mask & (depth > 0)))
        mask_available = bool(np.any(mask))

        pose_path = baseline_pose_dir / f"{index:06d}.txt"
        if not pose_path.is_file():
            pose_path = baseline_pose_dir / f"{index}.txt"
        recorded_status = None
        if receipt is not None:
            entry = receipt[index]
            if not pose_path.is_file() or _sha256_path(pose_path) != entry["pose_file_sha256"]:
                raise ValueError(f"baseline receipt pose digest mismatch at frame {index}")
            recorded_status = entry["tracking_status"]
        started = time.perf_counter()
        pose = _load_pose_txt(pose_path)
        unverified_pose_present = pose is not None and not assume_pose_file_candidate and receipt is None
        if unverified_pose_present:
            pose = None
        if recorded_status in ("fail", "no_ba"):
            pose = None
        baseline_ms = round((time.perf_counter() - started) * 1000.0, 3)
        keep_bt = pose is not None and (index % baseline_keep_every == 0)
        if pose is None:
            baseline_tick = {
                "frame_index": index,
                "status": "not_observable",
                "reason": (f"archive_tracking_{recorded_status}" if recorded_status in ("fail", "no_ba") else
                           "archived_pose_status_unverified" if unverified_pose_present else
                           "backend_pose_invalid" if mask_available else "sam_mask_unavailable"),
                "mask_available": mask_available,
                "depth_supported_pixels": support,
                "camera_from_object": None,
                "runtime_timing_ms": {
                    "archive_pose_read_ms": baseline_ms,
                },
                "wire_authorized": False,
            }
        else:
            baseline_tick = {
                "frame_index": index,
                "status": "candidate",
                "reason": None,
                "mask_available": mask_available,
                "depth_supported_pixels": support,
                "camera_from_object": pose.tolist(),
                "runtime_timing_ms": {
                    "archive_pose_read_ms": baseline_ms,
                },
                "wire_authorized": False,
                "pose_source": "bundletrack",
            }
        baseline.append(baseline_tick)

        if keep_bt:
            sparse_tick = dict(baseline_tick)
            bt_pose = np.asarray(pose, dtype=np.float64)
            if last_pose is not None:
                delta = bt_pose[:3, 3] - last_pose[:3, 3]
                distance = float(np.linalg.norm(delta))
                if distance > float(maximum_recovery_translation_step_m):
                    clipped = np.array(bt_pose, dtype=np.float64, copy=True)
                    clipped[:3, 3] = last_pose[:3, 3] + delta * (
                        float(maximum_recovery_translation_step_m) / distance
                    )
                    bt_pose = clipped
                    recovered.append(
                        {
                            "frame_index": index,
                            "status": "candidate",
                            "reason": None,
                            "mask_available": mask_available,
                            "depth_supported_pixels": support,
                            "camera_from_object": bt_pose.tolist(),
                            "runtime_timing_ms": {
                                "archive_pose_read_ms": baseline_ms,
                            },
                            "wire_authorized": False,
                            "pose_source": "bundletrack_clipped_reentry",
                            "bundletrack_reentry_clip_m": distance,
                        }
                    )
                    last_pose = bt_pose
                else:
                    last_pose = bt_pose
                    recovered.append(dict(sparse_tick, pose_source="bundletrack"))
            else:
                last_pose = bt_pose
                recovered.append(dict(sparse_tick, pose_source="bundletrack"))
            last_mask_centroid, _ = estimate_mask_depth_centroid_camera(
                depth_m=depth,
                mask=mask,
                camera_matrix=camera_matrix,
                minimum_depth_support_pixels=minimum_depth_support_pixels,
            )
        else:
            sparse_tick = {
                "frame_index": index,
                "status": "not_observable",
                "reason": (f"archive_tracking_{recorded_status}" if recorded_status in ("fail", "no_ba") else
                           "archived_pose_status_unverified" if unverified_pose_present else
                           "backend_pose_invalid" if mask_available else "sam_mask_unavailable"),
                "mask_available": mask_available,
                "depth_supported_pixels": support,
                "camera_from_object": None,
                "runtime_timing_ms": {
                    "archive_pose_read_ms": baseline_ms,
                },
                "wire_authorized": False,
            }
            if last_pose is not None and last_mask_centroid is not None and mask_available:
                recovery_started = time.perf_counter()
                recovered_pose, reason, diag = recover_camera_from_object_with_prior(
                    depth_m=depth,
                    mask=mask,
                    camera_matrix=camera_matrix,
                    prior_camera_from_object=last_pose,
                    prior_mask_centroid_camera=last_mask_centroid,
                    maximum_translation_step_m=maximum_recovery_translation_step_m,
                    minimum_depth_support_pixels=minimum_depth_support_pixels,
                )
                recovery_ms = round((time.perf_counter() - recovery_started) * 1000.0, 3)
                if recovered_pose is not None:
                    last_pose = recovered_pose
                    last_mask_centroid = np.asarray(
                        diag["mask_centroid_camera_m"], dtype=np.float64
                    )
                    recovered.append(
                        {
                            "frame_index": index,
                            "status": "candidate",
                            "reason": None,
                            "mask_available": True,
                            "depth_supported_pixels": diag.get("depth_supported_pixels"),
                            "camera_from_object": recovered_pose.tolist(),
                            "runtime_timing_ms": {
                                "pose_ms": recovery_ms,
                            },
                            "wire_authorized": False,
                            "pose_source": "mask_depth_recovery",
                            "mask_depth_recovery": dict(diag),
                        }
                    )
                else:
                    recovered.append(
                        dict(
                            sparse_tick,
                            reason=reason or sparse_tick["reason"],
                        )
                    )
            else:
                recovered.append(dict(sparse_tick))
        sparse.append(sparse_tick)

        pca = advance_mask_depth_pca_pose(
            depth_m=depth,
            mask=mask,
            camera_matrix=camera_matrix,
            color_timestamp_ns=color_ns,
            depth_timestamp_ns=depth_ns,
            minimum_depth_support_pixels=minimum_depth_support_pixels,
            pixel_stride=2,
        )
        shadow.append(
            {
                "frame_index": index,
                "status": pca["status"],
                "reason": pca["reason"],
                "mask_available": bool(pca.get("mask_available")),
                "depth_supported_pixels": pca.get("depth_supported_pixels"),
                "mask_pixel_count": pca.get("mask_pixel_count"),
                "sampled_depth_point_count": pca.get("sampled_depth_point_count"),
                "camera_from_object": pca["camera_from_object"],
                "runtime_timing_ms": {"pose_ms": pca["runtime_timing_ms"]["total_ms"]},
                "wire_authorized": False,
            }
        )
        for ticks in (baseline, sparse, recovered, shadow):
            ticks[-1].update(clock_evidence)
        for ticks in (baseline, sparse, recovered):
            ticks[-1]["baseline_status_evidence"] = (
                "caller_supplied_archive_binding_not_independently_verified" if receipt is not None
                else "pose_file_presence_assumption_unverified" if assume_pose_file_candidate
                else "archive_status_not_supplied"
            )
    return {
        "bundletrack_baseline": baseline,
        "bundletrack_sparse": sparse,
        "bundletrack_with_mask_depth_recovery": recovered,
        "mask_depth_pca": shadow,
    }


def evaluate_registered_sequence_ab(
    *,
    sequence_dir: Path,
    baseline_pose_dir: Path,
    minimum_depth_support_pixels: int = DEFAULT_MINIMUM_DEPTH_SUPPORT_PIXELS,
    baseline_keep_every: int = 1,
    maximum_recovery_translation_step_m: float = DEFAULT_MAXIMUM_TRANSLATION_STEP_M,
    baseline_arm: str = "bundletrack_sparse",
    assume_pose_file_candidate: bool = False,
    baseline_status_receipt: Path | None = None,
    declared_frame_interval: tuple[int, int] | None = None,
) -> dict[str, Any]:
    arms = build_registered_sequence_arms(
        sequence_dir=sequence_dir,
        baseline_pose_dir=baseline_pose_dir,
        minimum_depth_support_pixels=minimum_depth_support_pixels,
        baseline_keep_every=baseline_keep_every,
        maximum_recovery_translation_step_m=maximum_recovery_translation_step_m,
        assume_pose_file_candidate=assume_pose_file_candidate,
        baseline_status_receipt=baseline_status_receipt,
    )
    if baseline_keep_every == 1:
        # Full BT coverage: compare recovery/PCA against the complete dump.
        baseline_arm = "bundletrack_baseline"
    report = evaluate_shadow_pose_owner_ab(arms, baseline_arm=baseline_arm,
                                           declared_frame_interval=declared_frame_interval)
    report["input"] = {
        "schema": SCHEMA_INPUT,
        "sequence_dir": str(sequence_dir),
        "baseline_pose_dir": str(baseline_pose_dir),
        "minimum_depth_support_pixels": int(minimum_depth_support_pixels),
        "baseline_keep_every": int(baseline_keep_every),
        "baseline_status_evidence": (
            "caller_supplied_archive_binding_not_independently_verified" if baseline_status_receipt is not None
            else "pose_file_presence_assumption_unverified" if assume_pose_file_candidate
            else "archive_status_not_supplied"
        ),
        "baseline_status_receipt_sha256": (
            _sha256_path(baseline_status_receipt) if baseline_status_receipt is not None else None
        ),
        "maximum_recovery_translation_step_m": float(
            maximum_recovery_translation_step_m
        ),
        "cutover_authorized": False,
        "notes": [
            "bundletrack_with_mask_depth_recovery uses BT when present and clipped "
            "mask-depth translation recovery on BT misses.",
            "Shadow/recovery arms never set wire_authorized.",
            "A pose file alone does not establish BundleTrack tracking status or freshness; file-presence candidates are an explicit unverified diagnostic assumption.",
            "A caller-supplied archive binding does not independently verify status provenance, freshness or frozen release.",
            "Archived BundleTrack pose-file reads and recovery/PCA components are not complete session timings; total_ms is intentionally absent.",
        ],
    }
    report["controlled_reanchor_policy"] = {
        "event_count": report["controlled_reanchor_policy"]["event_count"],
        "would_request_count": report["controlled_reanchor_policy"][
            "would_request_count"
        ],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score BundleTrack poses vs mask_depth recovery on a registered sequence."
    )
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--baseline-pose-dir", type=Path, required=True)
    parser.add_argument("--baseline-status-receipt", type=Path)
    parser.add_argument("--declared-first-output-index", type=int)
    parser.add_argument("--declared-last-output-index", type=int)
    parser.add_argument(
        "--assume-pose-file-candidate", action="store_true",
        help="Unverified legacy diagnostic: treat present pose files as candidates; never authorizes cutover.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--minimum-depth-support-pixels",
        type=int,
        default=DEFAULT_MINIMUM_DEPTH_SUPPORT_PIXELS,
    )
    parser.add_argument(
        "--baseline-keep-every",
        type=int,
        default=3,
        help="Keep every Nth BT pose for the sparse/recovery comparison (default 3 ≈ 33%% BT).",
    )
    parser.add_argument(
        "--maximum-recovery-translation-step-m",
        type=float,
        default=DEFAULT_MAXIMUM_TRANSLATION_STEP_M,
    )
    args = parser.parse_args(argv)
    if (args.declared_first_output_index is None) != (args.declared_last_output_index is None):
        parser.error("both declared output-index endpoints are required")
    declared = ((args.declared_first_output_index, args.declared_last_output_index)
                if args.declared_first_output_index is not None else None)
    report = evaluate_registered_sequence_ab(
        sequence_dir=args.sequence_dir,
        baseline_pose_dir=args.baseline_pose_dir,
        minimum_depth_support_pixels=args.minimum_depth_support_pixels,
        baseline_keep_every=args.baseline_keep_every,
        maximum_recovery_translation_step_m=args.maximum_recovery_translation_step_m,
        assume_pose_file_candidate=args.assume_pose_file_candidate,
        baseline_status_receipt=args.baseline_status_receipt,
        declared_frame_interval=declared,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sparse = report["arms"]["bundletrack_sparse"]["mask_available_candidate_ratio"]
    recovery = report["arms"]["bundletrack_with_mask_depth_recovery"][
        "mask_available_candidate_ratio"
    ]
    comparison = report["comparisons"].get("bundletrack_with_mask_depth_recovery", {})
    print(
        json.dumps(
            {
                "output": str(args.output),
                "baseline_keep_every": args.baseline_keep_every,
                "sparse_bt_mask_available_candidate_ratio": sparse,
                "recovery_mask_available_candidate_ratio": recovery,
                "duty_cycle_delta": recovery - sparse,
                "cutover_candidate": comparison.get("cutover_candidate"),
                "blockers": comparison.get("blockers"),
                "cutover_authorized": report["cutover_authorized"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
