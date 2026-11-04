"""Score BundleTrack vs shadow pose owners without authorizing the wire.

Primary metric: mask-available candidate duty cycle.
Hard constraints: session latency p50/p95 and camera-local jump p95/max.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from causalpose3d.primitives.mask_depth_pca_pose import tick_is_mask_available
from causalpose3d.primitives.shadow_pose_owner import score_controlled_reanchor_policy

SCHEMA = "ticos.shadow-pose-owner-ab.v1"
DEFAULT_LATENCY_P50_TOLERANCE_MS = 15.0
DEFAULT_LATENCY_P95_TOLERANCE_MS = 25.0
DEFAULT_JUMP_P95_TOLERANCE_M = 0.01
DEFAULT_JUMP_MAX_TOLERANCE_M = 0.02
MINIMUM_MASK_AVAILABLE_CANDIDATE_RATIO = 0.80


def _registered_input_identity(tick: Mapping[str, Any]) -> tuple | None:
    stream=tick.get('stream_key');digest=tick.get('input_evidence_hash')
    if (not isinstance(stream,(list,tuple)) or not stream
            or any(not ((isinstance(v,str) and bool(v)) or type(v)is int) for v in stream)
            or not isinstance(digest,str) or len(digest)!=64
            or any(c not in '0123456789abcdefABCDEF' for c in digest)):
        return None
    clocks=[]
    for key in ('color_timestamp_ns','depth_timestamp_ns'):
        value=tick.get(key)
        if type(value)is int and 0<=value<=2**64-1:clocks.append(value)
        elif (isinstance(value,str) and len(value)<=20 and value.isascii() and value.isdecimal()
                and int(value)<=2**64-1):clocks.append(int(value))
        else:return None
    return (tuple(stream),*clocks,digest.lower())


def _registered_input_summary(ticks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identities=[_registered_input_identity(tick) for tick in ticks]
    valid=[identity for identity in identities if identity is not None]
    previous={};monotonic=True
    for stream,color,depth,_ in valid:
        old=previous.get(stream)
        if old is not None and (color<=old[0] or depth<=old[1]):monotonic=False
        previous[stream]=(color,depth)
    complete=len(valid)==len(ticks)
    return dict(count=len(valid),stream_count=len({identity[0] for identity in valid}),timestamps_monotonic=monotonic,
        ordered_sequence_sha256=(hashlib.sha256(json.dumps(identities,separators=(',',':'),allow_nan=False).encode()).hexdigest() if complete else None))


def _frame_index_coverage(ticks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe gaps in the supplied index sequence, not raw-frame coverage."""
    indices = [tick.get("frame_index") for tick in ticks]
    valid = (bool(indices) and all(type(index) is int and index >= 0 for index in indices)
             and all(right > left for left, right in zip(indices, indices[1:])))
    if not valid:
        return {"valid": False, "first": None, "last": None,
                "span": None, "missing_indices": None, "contiguous": False}
    span = indices[-1] - indices[0] + 1
    missing = span - len(indices)
    return {"valid": True, "first": indices[0], "last": indices[-1],
            "span": span, "missing_indices": missing, "contiguous": missing == 0}


def _object_key_summary(ticks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe caller-supplied slot labels, without validating physical identity."""
    keys = [tick.get("object_key") for tick in ticks]
    valid = [key for key in keys if type(key) is int or (type(key) is str and bool(key))]
    unique = {(type(key), key) for key in valid}
    complete = len(valid) == len(ticks)
    stable = complete and len(unique) == 1
    return {"provided_count": sum("object_key" in tick for tick in ticks),
            "valid_count": len(valid), "complete": complete, "stable": stable,
            "key_type": type(valid[0]).__name__ if stable else None,
            "key": valid[0] if stable else None}


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _mask_availability_known(tick: Mapping[str, Any]) -> bool:
    """Whether the existing denominator predicate has explicit evidence."""
    diagnostic = tick.get("mask_diagnostic")
    unavailable = (tick.get("reason") == "sam_mask_unavailable" or
                   (isinstance(diagnostic, Mapping) and
                    diagnostic.get("reason") == "sam_mask_unavailable"))
    supports = [tick.get("depth_supported_pixels")]
    if isinstance(diagnostic, Mapping):
        supports.append(diagnostic.get("depth_supported_pixels"))
    if unavailable and any(type(value) is int and value > 0 for value in supports):
        return False
    if "mask_available" in tick:
        declared = tick["mask_available"]
        if type(declared) is not bool:
            return False
        # The shared predicate may fall back to positive depth support even
        # when an explicit False was supplied. Such an ambiguous tick must
        # never be used to authorize a pose-owner cutover.
        if tick_is_mask_available(tick) != declared:
            return False
        if declared and (tick.get("reason") == "sam_mask_unavailable" or
                         (isinstance(diagnostic, Mapping) and
                          diagnostic.get("reason") == "sam_mask_unavailable")):
            return False
        return True
    if tick.get("reason") == "sam_mask_unavailable":
        return True
    diagnostic = tick.get("mask_diagnostic")
    if isinstance(diagnostic, Mapping):
        if diagnostic.get("reason") == "sam_mask_unavailable":
            return True
        support = diagnostic.get("depth_supported_pixels")
        if type(support) is int and support >= 0:
            return True
    support = tick.get("depth_supported_pixels")
    return type(support) is int and support >= 0


def _mask_source_summary(ticks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare declared SAM2 mask evidence, without claiming native consumption."""
    rows: list[list[Any]] = []
    missing = 0
    known = True
    for tick in ticks:
        if type(tick.get("frame_index")) is not int or tick["frame_index"] < 0:
            known = False
        if not _mask_availability_known(tick):
            known = False
            continue
        available = tick_is_mask_available(tick)
        digest = tick.get("source_mask_evidence_sha256") if available else None
        if available and (not isinstance(digest, str) or len(digest) != 64
                          or any(c not in "0123456789abcdef" for c in digest)):
            missing += 1
        rows.append([tick.get("frame_index"), available, digest])
    complete = known and missing == 0 and len(rows) == len(ticks)
    return {
        "complete": complete,
        "missing_or_invalid_mask_available_count": missing,
        "ordered_sequence_sha256": (
            hashlib.sha256(json.dumps(rows, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if complete else None
        ),
    }


def _pose_translation(pose: object) -> np.ndarray | None:
    if not isinstance(pose, (list, tuple)):
        return None
    try:
        matrix = np.asarray(pose, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-6, rtol=0)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5, rtol=0)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5, rtol=0)
    ):
        return None
    return matrix[:3, 3].copy()


def _camera_local_jumps_m(ticks: Sequence[Mapping[str, Any]]) -> list[float]:
    jumps: list[float] = []
    previous: np.ndarray | None = None
    for tick in ticks:
        if str(tick.get("status") or "") != "candidate":
            previous = None
            continue
        translation = _pose_translation(tick.get("camera_from_object"))
        if translation is None:
            previous = None
            continue
        if previous is not None:
            jumps.append(float(np.linalg.norm(translation - previous)))
        previous = translation
    return jumps


def _camera_local_rotation_jumps_deg(ticks: Sequence[Mapping[str, Any]]) -> list[float]:
    """Geodesic rotation between valid candidates; descriptive, not a gate."""
    jumps: list[float] = []
    previous: np.ndarray | None = None
    for tick in ticks:
        if str(tick.get("status") or "") != "candidate":
            previous = None
            continue
        pose = tick.get("camera_from_object")
        if _pose_translation(pose) is None:
            previous = None
            continue
        rotation = np.asarray(pose, dtype=np.float64)[:3, :3]
        if previous is not None:
            cosine = (float(np.trace(previous.T @ rotation)) - 1.0) / 2.0
            jumps.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
        previous = rotation
    return jumps


def _jump_non_candidate_gaps(ticks: Sequence[Mapping[str, Any]]) -> list[int]:
    """Count skipped non-candidate ticks for each scored candidate-to-candidate jump.

    The gate intentionally scores displacements between available candidates,
    even when unavailable ticks occur between them. Invalid candidate poses
    break the sequence just as they do in the jump calculation.
    """
    previous = False
    skipped = 0
    gaps: list[int] = []
    for tick in ticks:
        if str(tick.get("status") or "") != "candidate":
            if previous:
                skipped += 1
            continue
        if _pose_translation(tick.get("camera_from_object")) is None:
            previous = False
            skipped = 0
            continue
        if previous:
            gaps.append(skipped)
        previous = True
        skipped = 0
    return gaps


def _timing_values(ticks: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for tick in ticks:
        timing = tick.get("runtime_timing_ms")
        if not isinstance(timing, Mapping):
            continue
        value = timing.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number) and number >= 0:
                values.append(number)
    return values


def summarize_arm(ticks: Sequence[Mapping[str, Any]], *, arm: str) -> dict[str, Any]:
    """Summarize one arm's tick stream for the A/B scorecard."""
    if not ticks:
        raise ValueError(f"arm {arm} has no ticks")
    frame_ids = [tick.get("frame_index") for tick in ticks]
    valid_opportunities = (
        all(type(index) is int for index in frame_ids)
        and all(right > left for left, right in zip(frame_ids, frame_ids[1:]))
        and all(_mask_availability_known(tick) for tick in ticks)
    )
    opportunities = [[index, tick_is_mask_available(tick)]
                     for index, tick in zip(frame_ids, ticks)]
    opportunity_digest = (
        hashlib.sha256(json.dumps(opportunities, separators=(",", ":"))
                       .encode()).hexdigest() if valid_opportunities else None
    )
    mask_available = [tick for tick in ticks if tick_is_mask_available(tick)]
    candidates = [
        tick for tick in ticks if str(tick.get("status") or "") == "candidate"
    ]
    invalid_status_count = sum(
        1 for tick in ticks
        if tick.get("status") not in ("candidate", "not_observable")
    )
    mask_available_candidates = [
        tick
        for tick in mask_available
        if str(tick.get("status") or "") == "candidate"
    ]
    reasons: dict[str, int] = {}
    for tick in ticks:
        if str(tick.get("status") or "") == "candidate":
            continue
        reason = tick.get("reason")
        key = str(reason) if isinstance(reason, str) and reason else "unspecified"
        reasons[key] = reasons.get(key, 0) + 1
    wire_authorized = sum(1 for tick in ticks if tick.get("wire_authorized") is True)
    wire_authority_unknown = sum(
        1 for tick in ticks if type(tick.get("wire_authorized")) is not bool
    )
    totals = _timing_values(ticks, "total_ms")
    poses = _timing_values(ticks, "pose_ms")
    jumps = _camera_local_jumps_m(
        [tick for tick in ticks if str(tick.get("status") or "") == "candidate"]
    )
    rotation_jumps = _camera_local_rotation_jumps_deg(candidates)
    jump_gaps = _jump_non_candidate_gaps(ticks)
    adjacent_jumps = [jump for jump, gap in zip(jumps, jump_gaps) if gap == 0]
    bridged_jumps = [jump for jump, gap in zip(jumps, jump_gaps) if gap > 0]
    return {
        "arm": arm,
        "frame_count": len(ticks),
        "object_key_evidence": _object_key_summary(ticks),
        "frame_index_coverage": _frame_index_coverage(ticks),
        "ordered_mask_opportunities_sha256": opportunity_digest,
        "registered_input_evidence": _registered_input_summary(ticks),
        "mask_source_evidence": _mask_source_summary(ticks),
        "candidate_count": len(candidates),
        "invalid_status_count": invalid_status_count,
        "invalid_candidate_pose_count": sum(
            _pose_translation(tick.get("camera_from_object")) is None
            for tick in candidates
        ),
        "candidate_ratio": _ratio(len(candidates), len(ticks)),
        "mask_available_count": len(mask_available),
        "mask_availability_unknown_count": sum(
            1 for tick in ticks if not _mask_availability_known(tick)
        ),
        "mask_available_candidate_count": len(mask_available_candidates),
        "mask_available_candidate_ratio": _ratio(
            len(mask_available_candidates), len(mask_available)
        ),
        "reason_histogram": dict(sorted(reasons.items())),
        "wire_authorized_count": wire_authorized,
        "wire_authority_unknown_count": wire_authority_unknown,
        "session_total_ms": {
            "count": len(totals),
            "p50": _percentile(totals, 0.50),
            "p95": _percentile(totals, 0.95),
            "max": max(totals) if totals else None,
        },
        "pose_ms": {
            "p50": _percentile(poses, 0.50),
            "p95": _percentile(poses, 0.95),
            "max": max(poses) if poses else None,
        },
        "camera_local_rotation_jump_deg": {
            "count": len(rotation_jumps),
            "p95": _percentile(rotation_jumps, 0.95),
            "max": max(rotation_jumps) if rotation_jumps else None,
            "adjacent_candidate_ticks": {
                "count": sum(gap == 0 for gap in jump_gaps),
                "p95": _percentile([v for v, gap in zip(rotation_jumps, jump_gaps) if gap == 0], 0.95),
                "max": max((v for v, gap in zip(rotation_jumps, jump_gaps) if gap == 0), default=None),
            },
            "bridged_non_candidate_ticks": {
                "count": sum(gap > 0 for gap in jump_gaps),
                "p95": _percentile([v for v, gap in zip(rotation_jumps, jump_gaps) if gap > 0], 0.95),
                "max": max((v for v, gap in zip(rotation_jumps, jump_gaps) if gap > 0), default=None),
            },
        },
        "camera_local_jump_m": {
            "p95": _percentile(jumps, 0.95),
            "max": max(jumps) if jumps else None,
            "count": len(jumps),
            "jumps_bridging_non_candidate_ticks": sum(gap > 0 for gap in jump_gaps),
            "max_intervening_non_candidate_ticks": max(jump_gaps) if jump_gaps else None,
            "adjacent_candidate_ticks": {
                "count": len(adjacent_jumps), "p95": _percentile(adjacent_jumps, 0.95),
                "max": max(adjacent_jumps) if adjacent_jumps else None,
            },
            "bridged_non_candidate_ticks": {
                "count": len(bridged_jumps), "p95": _percentile(bridged_jumps, 0.95),
                "max": max(bridged_jumps) if bridged_jumps else None,
            },
        },
    }


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _validate_tolerances(**values: float) -> None:
    for name, value in values.items():
        number = _finite_or_none(value)
        if number is None or number < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")


def compare_to_baseline(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    latency_p50_tolerance_ms: float = DEFAULT_LATENCY_P50_TOLERANCE_MS,
    latency_p95_tolerance_ms: float = DEFAULT_LATENCY_P95_TOLERANCE_MS,
    jump_p95_tolerance_m: float = DEFAULT_JUMP_P95_TOLERANCE_M,
    jump_max_tolerance_m: float = DEFAULT_JUMP_MAX_TOLERANCE_M,
) -> dict[str, Any]:
    """Return gate results for one shadow arm versus the BundleTrack baseline."""
    _validate_tolerances(
        latency_p50_tolerance_ms=latency_p50_tolerance_ms,
        latency_p95_tolerance_ms=latency_p95_tolerance_ms,
        jump_p95_tolerance_m=jump_p95_tolerance_m,
        jump_max_tolerance_m=jump_max_tolerance_m,
    )
    base_duty = float(baseline["mask_available_candidate_ratio"])
    cand_duty = float(candidate["mask_available_candidate_ratio"])
    duty_improved = cand_duty > base_duty + 1e-12
    blockers: list[str] = []
    # Matching input identities and a candidate status do not establish that the
    # returned pose was updated from that input. No independent freshness audit
    # is currently consumed by this scoreboard, so the policy's zero
    # false-fresh requirement cannot yet be established.
    blockers.append("candidate_freshness_not_independently_verified")
    # Matched ticks do not prove that the recording was reserved before
    # candidate selection or that the baseline came from the frozen release.
    # No validated release/recording registry binding is consumed here.
    blockers.append("heldout_recording_role_not_independently_verified")
    blockers.append("frozen_baseline_release_not_independently_verified")
    # Finite caller-supplied numbers do not prove comparable timing boundaries
    # or bind the numbers to measured execution of the respective sessions.
    # The scoreboard does not yet consume an independently validated trace.
    blockers.append("session_timing_not_independently_verified")
    # Caller-provided keys can reveal contradictions but cannot authenticate
    # a physical object or link the two pose owners' identities.
    blockers.append("object_identity_not_independently_verified")
    # A supplied mask digest does not establish what the native tracker consumed.
    blockers.append("mask_consumption_not_independently_verified")
    duty_target_met = cand_duty >= MINIMUM_MASK_AVAILABLE_CANDIDATE_RATIO
    if not duty_target_met:
        blockers.append("candidate_duty_cycle_below_product_target")
    for role, summary in (("baseline", baseline), ("candidate", candidate)):
        identity = summary.get("object_key_evidence", {})
        if identity.get("provided_count", 0) and identity.get("stable") is not True:
            blockers.append(f"{role}_object_key_evidence_inconsistent")
        status_count = summary.get("invalid_status_count")
        if type(status_count) is not int or status_count != 0:
            blockers.append(f"{role}_status_evidence_invalid")
        if summary.get("invalid_candidate_pose_count") != 0:
            blockers.append(f"{role}_pose_evidence_invalid")
        frame_count = summary.get("frame_count", 0)
        if summary.get("frame_index_coverage", {}).get("contiguous") is not True:
            blockers.append(f"{role}_frame_index_coverage_incomplete")
        source=summary.get('registered_input_evidence',{})
        if source.get('count')!=frame_count or not source.get('ordered_sequence_sha256'):
            blockers.append(f'{role}_registered_input_evidence_incomplete')
        if source.get('stream_count') != 1:
            blockers.append(f'{role}_camera_local_stream_not_unique')
        if source.get('timestamps_monotonic')is not True:
            blockers.append(f'{role}_source_timestamps_not_monotonic')
        if not frame_count or summary.get("session_total_ms", {}).get("count") != frame_count:
            blockers.append(f"{role}_timing_evidence_incomplete")
        if not summary.get("mask_available_count"):
            blockers.append(f"{role}_mask_evidence_missing")
        mask_unknown = summary.get("mask_availability_unknown_count")
        if type(mask_unknown) is not int or mask_unknown != 0:
            blockers.append(f"{role}_mask_availability_evidence_incomplete")
        mask_source = summary.get("mask_source_evidence", {})
        if mask_source.get("complete") is not True or not mask_source.get("ordered_sequence_sha256"):
            blockers.append(f"{role}_mask_source_evidence_incomplete")
        opportunity = summary.get("ordered_mask_opportunities_sha256")
        if (not isinstance(opportunity, str) or len(opportunity) != 64
                or any(c not in "0123456789abcdef" for c in opportunity)):
            blockers.append(f"{role}_mask_opportunity_evidence_incomplete")
        if not summary.get("camera_local_jump_m", {}).get("count"):
            blockers.append(f"{role}_jump_evidence_missing")
        for metric, keys in (("session_total_ms", ("p50", "p95")), ("camera_local_jump_m", ("p95", "max"))):
            if any(_finite_or_none(summary.get(metric, {}).get(key)) is None for key in keys):
                blockers.append(f"{role}_{metric}_missing")
    if baseline.get("frame_count") != candidate.get("frame_count"):
        blockers.append("paired_frame_count_mismatch")
    base_object = baseline.get("object_key_evidence", {})
    candidate_object = candidate.get("object_key_evidence", {})
    if (base_object.get("stable") is True and candidate_object.get("stable") is True
            and (base_object.get("key_type"), base_object.get("key"))
            != (candidate_object.get("key_type"), candidate_object.get("key"))):
        blockers.append("paired_object_key_evidence_mismatch")
    if baseline.get("mask_available_count") != candidate.get("mask_available_count"):
        blockers.append("paired_mask_available_count_mismatch")
    base_opportunities = baseline.get("ordered_mask_opportunities_sha256")
    candidate_opportunities = candidate.get("ordered_mask_opportunities_sha256")
    if base_opportunities and candidate_opportunities and base_opportunities != candidate_opportunities:
        blockers.append("paired_mask_opportunity_evidence_mismatch")
    base_mask = baseline.get("mask_source_evidence", {}).get("ordered_sequence_sha256")
    candidate_mask = candidate.get("mask_source_evidence", {}).get("ordered_sequence_sha256")
    if base_mask and candidate_mask and base_mask != candidate_mask:
        blockers.append("paired_mask_source_evidence_mismatch")
    base_source=baseline.get('registered_input_evidence',{}).get('ordered_sequence_sha256')
    candidate_source=candidate.get('registered_input_evidence',{}).get('ordered_sequence_sha256')
    if base_source and candidate_source and base_source!=candidate_source:
        blockers.append('paired_registered_input_evidence_mismatch')
    wire_count = candidate.get("wire_authorized_count")
    unknown_count = candidate.get("wire_authority_unknown_count")
    if type(wire_count) is int and wire_count > 0:
        blockers.append("shadow_arm_authorized_wire")
    if (type(wire_count) is not int or wire_count < 0
            or type(unknown_count) is not int or unknown_count != 0):
        blockers.append("shadow_wire_authority_evidence_incomplete")

    base_p50 = _finite_or_none(baseline["session_total_ms"]["p50"])
    cand_p50 = _finite_or_none(candidate["session_total_ms"]["p50"])
    if base_p50 is not None and cand_p50 is not None:
        if cand_p50 > base_p50 + float(latency_p50_tolerance_ms):
            blockers.append("session_p50_regression")
    base_p95 = _finite_or_none(baseline["session_total_ms"]["p95"])
    cand_p95 = _finite_or_none(candidate["session_total_ms"]["p95"])
    if base_p95 is not None and cand_p95 is not None:
        if cand_p95 > base_p95 + float(latency_p95_tolerance_ms):
            blockers.append("session_p95_regression")

    base_jump_p95 = _finite_or_none(baseline["camera_local_jump_m"]["p95"])
    cand_jump_p95 = _finite_or_none(candidate["camera_local_jump_m"]["p95"])
    if base_jump_p95 is not None and cand_jump_p95 is not None:
        if cand_jump_p95 > base_jump_p95 + float(jump_p95_tolerance_m):
            blockers.append("jump_p95_regression")
    base_jump_max = _finite_or_none(baseline["camera_local_jump_m"]["max"])
    cand_jump_max = _finite_or_none(candidate["camera_local_jump_m"]["max"])
    if base_jump_max is not None and cand_jump_max is not None:
        if cand_jump_max > base_jump_max + float(jump_max_tolerance_m):
            blockers.append("jump_max_regression")

    return {
        "arm": candidate["arm"],
        "mask_available_candidate_ratio_delta": cand_duty - base_duty,
        "duty_cycle_improved": duty_improved,
        "duty_cycle_product_target": MINIMUM_MASK_AVAILABLE_CANDIDATE_RATIO,
        "duty_cycle_product_target_met": duty_target_met,
        "hard_constraints_passed": not blockers,
        "cutover_candidate": duty_improved and not blockers,
        "blockers": blockers,
    }


def evaluate_shadow_pose_owner_ab(
    arms: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    baseline_arm: str = "bundletrack_baseline",
    latency_p50_tolerance_ms: float = DEFAULT_LATENCY_P50_TOLERANCE_MS,
    latency_p95_tolerance_ms: float = DEFAULT_LATENCY_P95_TOLERANCE_MS,
    jump_p95_tolerance_m: float = DEFAULT_JUMP_P95_TOLERANCE_M,
    jump_max_tolerance_m: float = DEFAULT_JUMP_MAX_TOLERANCE_M,
    declared_frame_interval: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed A/B scorecard across named tick arms."""
    if declared_frame_interval is not None:
        if (not isinstance(declared_frame_interval, tuple)
                or len(declared_frame_interval) != 2
                or any(type(value) is not int or value < 0 for value in declared_frame_interval)
                or declared_frame_interval[0] > declared_frame_interval[1]):
            raise ValueError("declared_frame_interval must be an inclusive nonnegative integer pair")
    _validate_tolerances(
        latency_p50_tolerance_ms=latency_p50_tolerance_ms,
        latency_p95_tolerance_ms=latency_p95_tolerance_ms,
        jump_p95_tolerance_m=jump_p95_tolerance_m,
        jump_max_tolerance_m=jump_max_tolerance_m,
    )
    if baseline_arm not in arms:
        raise ValueError(f"baseline arm missing: {baseline_arm}")
    summaries = {
        name: summarize_arm(ticks, arm=name) for name, ticks in arms.items()
    }
    if declared_frame_interval is not None:
        first, last = declared_frame_interval
        for name, ticks in arms.items():
            coverage = summaries[name]["frame_index_coverage"]
            indices = [tick.get("frame_index") for tick in ticks]
            valid = coverage["valid"] is True
            inside = sum(first <= index <= last for index in indices) if valid else None
            outside = len(indices) - inside if valid else None
            missing = last - first + 1 - inside if valid else None
            summaries[name]["declared_frame_interval_coverage"] = {
                "first": first, "last": last,
                "missing_indices": missing, "outside_indices": outside,
                "exact": valid and missing == 0 and outside == 0,
                "independently_verified": False,
            }
    baseline = summaries[baseline_arm]
    comparisons = {
        name: compare_to_baseline(
            baseline,
            summary,
            latency_p50_tolerance_ms=latency_p50_tolerance_ms,
            latency_p95_tolerance_ms=latency_p95_tolerance_ms,
            jump_p95_tolerance_m=jump_p95_tolerance_m,
            jump_max_tolerance_m=jump_max_tolerance_m,
        )
        for name, summary in summaries.items()
        if name != baseline_arm
    }
    # Ratios alone cannot prove paired evidence. Both arms must name the
    # same ordered recording frames and mask-availability opportunities.
    baseline_ids = [tick.get("frame_index") for tick in arms[baseline_arm]]
    baseline_masks = [tick_is_mask_available(tick) for tick in arms[baseline_arm]]
    for name, comparison in comparisons.items():
        if declared_frame_interval is not None:
            for role, arm_name in (("baseline", baseline_arm), ("candidate", name)):
                if summaries[arm_name]["declared_frame_interval_coverage"]["exact"] is not True:
                    comparison["blockers"].append(f"{role}_declared_frame_interval_incomplete")
                    comparison["hard_constraints_passed"] = False
                    comparison["cutover_candidate"] = False
        ids = [tick.get("frame_index") for tick in arms[name]]
        valid_ids = all(type(value) is int for value in baseline_ids + ids)
        if (
            not valid_ids
            or ids != baseline_ids
            or len(set(ids)) != len(ids)
            or any(right <= left for left, right in zip(ids, ids[1:]))
            or [tick_is_mask_available(tick) for tick in arms[name]] != baseline_masks
        ):
            comparison["blockers"].append("paired_frame_evidence_mismatch")
            comparison["hard_constraints_passed"] = False
            comparison["cutover_candidate"] = False
    reanchor_events = score_controlled_reanchor_policy(
        list(arms[baseline_arm])
    )
    would_reanchor = sum(
        1 for event in reanchor_events if event["would_request_controlled_reanchor"]
    )
    return {
        "schema": SCHEMA,
        "baseline_arm": baseline_arm,
        "arms": summaries,
        "comparisons": comparisons,
        "controlled_reanchor_policy": {
            "event_count": len(reanchor_events),
            "would_request_count": would_reanchor,
            "events": reanchor_events,
        },
        "cutover_authorized": False,
        "notes": [
            "Shadow arms must keep wire_authorized_count at 0.",
            "Cutover requires a separate production change citing a passing scorecard.",
            "Registered input source bindings must be complete and paired; frame numbers alone are not source identity.",
            "Matched declared SAM2 mask digests do not independently prove what each native tracker consumed.",
            "Contiguous supplied indices do not prove full recording coverage or independently verify endpoints.",
            "Camera-local rotation jumps are descriptive only and are not a cutover constraint.",
            "Each arm must contain one camera-local stream; poses in different frames or epochs cannot be compared as consecutive translations.",
        ],
    }


def _load_arms(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "arms" not in payload:
        raise ValueError("input must be an object with an arms map")
    arms = payload["arms"]
    if not isinstance(arms, dict) or not arms:
        raise ValueError("arms must be a nonempty object")
    result: dict[str, list[dict[str, Any]]] = {}
    for name, ticks in arms.items():
        if not isinstance(name, str) or not name:
            raise ValueError("arm names must be nonempty strings")
        if not isinstance(ticks, list) or not ticks:
            raise ValueError(f"arm {name} must be a nonempty tick list")
        result[name] = [dict(tick) for tick in ticks if isinstance(tick, dict)]
        if len(result[name]) != len(ticks):
            raise ValueError(f"arm {name} ticks must be objects")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score BundleTrack vs shadow pose-owner arms."
    )
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--baseline-arm", default="bundletrack_baseline")
    parser.add_argument("--declared-first-frame", type=int, default=None)
    parser.add_argument("--declared-last-frame", type=int, default=None)
    args = parser.parse_args(argv)
    if (args.declared_first_frame is None) != (args.declared_last_frame is None):
        parser.error("both declared frame endpoints are required")
    declared = (args.declared_first_frame, args.declared_last_frame) if args.declared_first_frame is not None else None
    report = evaluate_shadow_pose_owner_ab(
        _load_arms(args.input_json),
        baseline_arm=args.baseline_arm,
        declared_frame_interval=declared,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
