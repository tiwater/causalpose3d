"""Score SAM2 GPU-window same-thread CPU preparation overlap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
R = H.parents[2]
E = R / "artifacts/evaluation/sam-gpu-window-cpu-overlap-20261102"
N = R / "artifacts/evaluation/existing-owner-opt-in-contract-20261101/configured-route-final"
P = R / "artifacts/evaluation/configured-array-snapshot-shadow-20261101/control-1"

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def q(values: list[float]) -> dict[str, float | int]:
    if not values:
        return dict(count=0)
    return dict(
        count=len(values),
        p50_ms=round(float(np.percentile(values, 50)), 3),
        p95_ms=round(float(np.percentile(values, 95)), 3),
        max_ms=round(float(np.max(values)), 3),
    )


def main() -> None:
    execution = read(E / "execution.json")
    assert len(execution["runs"]) == 5
    for rel, digest in execution["source_bindings"].items():
        assert sha(R / rel) == digest, rel

    trace = [
        json.loads(line)
        for line in (N / "native-history.jsonl").read_text().splitlines()
    ]
    baseline_inputs = [
        dict(
            color_timestamp_ns=item["frame"]["color_timestamp_ns"],
            input_evidence_hash=item["frame"]["input_evidence_hash"],
        )
        for item in trace
    ]
    baseline_portal = read(P / "portal-lifetime.json")

    runs = {}
    raw_records = {}
    observed_masks = {}
    warm_rows_by_run = {}

    for job in execution["runs"]:
        assert job["exit_code"] == 0
        assert sha(R / job["log"]) == job["log_sha256"]
        name = f"{job['mode']}-{job['repeat']}"
        root = E / name
        mode = job["mode"]

        overlap = read(root / "gpu-window-overlap.json")
        portal = read(root / "portal-lifetime.json")
        bound = read(root / "runtime-policy-bindings.json")
        summary = read(root / "summary.json")
        replay_rows = [
            json.loads(line)
            for line in (root / "replay.jsonl").read_text().splitlines()
        ]

        assert summary["frames"] == len(replay_rows) == 40
        assert summary["bundletrack_camera_local_candidate_frames"] == 34
        assert summary["original_world_current_registered_surface_frames"] == 32
        assert not summary["control_authorized"]
        assert not overlap["wire_authorized"]
        assert all(not r["control_ready"] for r in replay_rows)
        assert portal["portal_closed"] and not portal["app_lifespan_entered"]
        assert portal["masks"] == baseline_portal["masks"]
        assert len(overlap["masks"]) == 80
        assert sum(m["registry_context"] for m in overlap["masks"]) == 40
        assert overlap["native_inputs"] == baseline_inputs
        assert len(overlap["native_inputs"]) == 38

        assert len(bound["records"]) == 1
        runtime = bound["records"][0]
        assert runtime["legacy_depth_fill_attempts"] == 5
        for lib_info in runtime["loaded_libraries"].values():
            assert sha(lib_info["path"]) == lib_info["sha256"]

        assert len(overlap["native_calls"]) == 38
        assert all(
            c["before"]["torch_threads"]
            == c["before"]["torch_omp"]
            == c["before"]["native_omp"]
            == 1
            and c["after"]["torch_threads"]
            == c["after"]["torch_omp"]
            == c["after"]["native_omp"]
            == 1
            for c in overlap["native_calls"]
        )

        warm = overlap["warm_frames"]
        assert len(warm) == 39
        if mode == "control":
            assert not any(w["eligible"] for w in warm)
            assert not any(w["executed_in_gpu_window"] for w in warm)
            assert sum(w["evidence_hash_thread_spawned"] for w in warm) == 38
        else:
            assert all(w["eligible"] and w["executed_in_gpu_window"] for w in warm)
            assert sum(w["native_buffers_reused"] for w in warm) == 38
            assert sum(w["rle_reused"] for w in warm) == sum(
                w["native_buffers_reused"] and not w.get("depth_far_zeroed")
                for w in warm
            )
            assert sum(w["evidence_hash_thread_spawned"] for w in warm) == 0
            if mode == "validate":
                assert all(w.get("prepared_frame_exact") for w in warm)
                native_warm = [w for w in warm if w["native_buffers_reused"]]
                assert len(native_warm) == 38
                assert all(
                    w.get("evidence_hash_exact")
                    and w.get("native_buffers_exact")
                    and (w.get("rle_exact") if w.get("rle_reused") else True)
                    for w in native_warm
                )

        raw_records[name] = replay_rows
        warm_rows_by_run[name] = warm
        observed_masks[name] = [
            (m["registry_context"], m["mask_sha256"]) for m in overlap["masks"]
        ]

        runs[name] = dict(
            mode=mode,
            source_observations=40,
            warm_observations=39,
            native_inputs=38,
            candidate_frames=summary["bundletrack_camera_local_candidate_frames"],
            current_surface_frames=summary[
                "original_world_current_registered_surface_frames"
            ],
            evidence_hash_threads_spawned=overlap["evidence_hash_threads_spawned"],
            background_worker_threads_created=overlap[
                "background_worker_threads_created"
            ],
            eligible_warm_frames=sum(w["eligible"] for w in warm),
            executed_in_gpu_window_frames=sum(
                w["executed_in_gpu_window"] for w in warm
            ),
            rle_reused_frames=sum(w["rle_reused"] for w in warm),
            native_buffers_reused_frames=sum(
                w["native_buffers_reused"] for w in warm
            ),
            all_source_ms={
                k: q([r[k] for r in replay_rows])
                for k in ("endpoint_ms", "pipeline_ms")
            },
            excluding_source_frame_0_ms=dict(
                endpoint_ms=q([r["endpoint_ms"] for r in replay_rows[1:]]),
                pipeline_ms=q([r["pipeline_ms"] for r in replay_rows[1:]]),
                registry_advance_ms=q([w["registry_advance_ms"] for w in warm]),
                input_prepare_ms=q([w["input_prepare_ms"] for w in warm]),
                sam_session_advance_ms=q(
                    [w["sam_session_advance_ms"] for w in warm]
                ),
                sam_track_ms=q([w["sam_track_ms"] for w in warm]),
                gpu_launch_to_yield_ms=q(
                    [
                        w["gpu_launch_to_yield_ms"]
                        for w in warm
                        if w.get("gpu_launch_to_yield_ms") is not None
                    ]
                ),
                gpu_window_cpu_prepare_ms=q(
                    [
                        w.get("gpu_window_cpu_prepare_ms") or 0.0
                        for w in warm
                    ]
                ),
                post_cpu_gpu_sync_wait_ms=q(
                    [
                        w["post_cpu_gpu_sync_wait_ms"]
                        for w in warm
                        if w.get("post_cpu_gpu_sync_wait_ms") is not None
                    ]
                ),
                evidence_hash_tail_ms=q(
                    [
                        w["evidence_hash_tail_ms"]
                        for w in warm
                        if w.get("evidence_hash_tail_ms") is not None
                    ]
                ),
                native_track_registered_ms=q(
                    [
                        w["native_track_registered_ms"]
                        for w in warm
                        if w.get("native_track_registered_ms") is not None
                    ]
                ),
            ),
            memory_before=overlap["memory_before"],
            memory_after=overlap["memory_after"],
        )

    comparisons = {}
    metrics = (
        "endpoint_ms",
        "pipeline_ms",
        "registry_advance_ms",
        "input_prepare_ms",
        "sam_session_advance_ms",
        "sam_track_ms",
        "post_cpu_gpu_sync_wait_ms",
        "native_track_registered_ms",
    )
    for repeat in (1, 2):
        ctrl = f"control-{repeat}"
        ovlp = f"overlap-{repeat}"
        assert observed_masks[ctrl] == observed_masks[ovlp]
        assert [r["source_owner_evidence"] for r in raw_records[ctrl]] == [
            r["source_owner_evidence"] for r in raw_records[ovlp]
        ]
        comparisons[f"repeat_{repeat}"] = {
            m: {
                stat: round(
                    runs[ovlp]["excluding_source_frame_0_ms"][m][stat]
                    - runs[ctrl]["excluding_source_frame_0_ms"][m][stat],
                    3,
                )
                for stat in ("p50_ms", "p95_ms", "max_ms")
            }
            for m in metrics
        }

    assert (
        observed_masks["validate-1"]
        == observed_masks["control-1"]
        == observed_masks["overlap-1"]
    )

    pooled_control_endpoint = [
        r["endpoint_ms"]
        for name in ("control-1", "control-2")
        for r in raw_records[name][1:]
    ]
    pooled_overlap_endpoint = [
        r["endpoint_ms"]
        for name in ("overlap-1", "overlap-2")
        for r in raw_records[name][1:]
    ]
    pooled_control_pipeline = [
        r["pipeline_ms"]
        for name in ("control-1", "control-2")
        for r in raw_records[name][1:]
    ]
    pooled_overlap_pipeline = [
        r["pipeline_ms"]
        for name in ("overlap-1", "overlap-2")
        for r in raw_records[name][1:]
    ]
    pooled_control_advance = [
        w["registry_advance_ms"]
        for name in ("control-1", "control-2")
        for w in warm_rows_by_run[name]
    ]
    pooled_overlap_advance = [
        w["registry_advance_ms"]
        for name in ("overlap-1", "overlap-2")
        for w in warm_rows_by_run[name]
    ]
    pooled = dict(
        control_warm_78=dict(
            endpoint_ms=q(pooled_control_endpoint),
            pipeline_ms=q(pooled_control_pipeline),
            registry_advance_ms=q(pooled_control_advance),
        ),
        overlap_warm_78=dict(
            endpoint_ms=q(pooled_overlap_endpoint),
            pipeline_ms=q(pooled_overlap_pipeline),
            registry_advance_ms=q(pooled_overlap_advance),
        ),
    )
    pooled["delta_overlap_minus_control"] = {
        m: {
            stat: round(
                pooled["overlap_warm_78"][m][stat]
                - pooled["control_warm_78"][m][stat],
                3,
            )
            for stat in ("p50_ms", "p95_ms", "max_ms")
        }
        for m in ("endpoint_ms", "pipeline_ms", "registry_advance_ms")
    }

    result = dict(
        passed=True,
        study="sam-gpu-window-cpu-overlap-20261102",
        runs=runs,
        comparisons=comparisons,
        pooled_warm_78=pooled,
        strict_warm_frames_byte_identical=39,
        strict_native_warm_frames_byte_identical=38,
        sam_masks_verified_per_run=80,
        native_inputs_verified_per_run=38,
        candidate_frames_per_run=34,
        current_surface_frames_per_run=32,
        background_worker_threads_created=0,
        warm_evidence_hash_threads_eliminated_per_run=38,
        root_production_sources_unchanged=True,
        live_acceptance_gates_passed="0/6",
        gate_passed=False,
        wire_authorized=False,
        scope="Same-thread CPU preparation inside SAM2 propagate_in_video GPU execution window on the 40-frame configured HTTP replay clip (teleop_20260625_173830). Does not claim live 15 Hz or <100 ms site acceptance.",
    )
    (E / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (H / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                comparisons=comparisons,
                pooled=pooled,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
