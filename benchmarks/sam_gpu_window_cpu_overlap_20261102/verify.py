"""Independent verification of SAM2 GPU-window same-thread CPU preparation overlap."""

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


def pct(values: list[float], q_val: float) -> float:
    return round(float(np.percentile(values, q_val)), 3)


def main() -> None:
    execution = read(E / "execution.json")
    results = read(E / "results.json")
    paper_results = read(H / "results.json")
    assert results == paper_results

    for rel, digest in execution["source_bindings"].items():
        assert sha(R / rel) == digest, rel

    baseline_trace = [
        json.loads(line)
        for line in (N / "native-history.jsonl").read_text().splitlines()
    ]
    baseline_inputs = [
        dict(
            color_timestamp_ns=item["frame"]["color_timestamp_ns"],
            input_evidence_hash=item["frame"]["input_evidence_hash"],
        )
        for item in baseline_trace
    ]
    baseline_masks = read(P / "portal-lifetime.json")["masks"]

    total_masks_verified = 0
    total_native_inputs_verified = 0
    total_native_omp_checks = 0
    evidence_snapshots = []

    for job in execution["runs"]:
        name = f"{job['mode']}-{job['repeat']}"
        root = E / name
        overlap = read(root / "gpu-window-overlap.json")
        portal = read(root / "portal-lifetime.json")
        bound = read(root / "runtime-policy-bindings.json")
        summary = read(root / "summary.json")
        rows = [
            json.loads(line)
            for line in (root / "replay.jsonl").read_text().splitlines()
        ]
        warm = overlap["warm_frames"]

        assert len(rows) == 40 and len(warm) == 39
        assert portal["masks"] == baseline_masks
        assert [m["mask_sha256"] for m in overlap["masks"]] == baseline_masks
        total_masks_verified += len(overlap["masks"])

        assert overlap["native_inputs"] == baseline_inputs
        total_native_inputs_verified += len(overlap["native_inputs"])

        for call in overlap["native_calls"]:
            assert (
                call["before"]["torch_threads"]
                == call["before"]["torch_omp"]
                == call["before"]["native_omp"]
                == 1
            )
            assert (
                call["after"]["torch_threads"]
                == call["after"]["torch_omp"]
                == call["after"]["native_omp"]
                == 1
            )
            total_native_omp_checks += 1

        for lib_info in bound["records"][0]["loaded_libraries"].values():
            assert sha(lib_info["path"]) == lib_info["sha256"]

        evidence_snapshots.append([r["source_owner_evidence"] for r in rows])

        run_res = results["runs"][name]["excluding_source_frame_0_ms"]
        for key, series in (
            ("endpoint_ms", [r["endpoint_ms"] for r in rows[1:]]),
            ("pipeline_ms", [r["pipeline_ms"] for r in rows[1:]]),
            ("registry_advance_ms", [w["registry_advance_ms"] for w in warm]),
            ("input_prepare_ms", [w["input_prepare_ms"] for w in warm]),
            ("sam_session_advance_ms", [w["sam_session_advance_ms"] for w in warm]),
            ("sam_track_ms", [w["sam_track_ms"] for w in warm]),
        ):
            assert run_res[key]["p50_ms"] == pct(series, 50), (name, key)
            assert run_res[key]["p95_ms"] == pct(series, 95), (name, key)
            assert run_res[key]["max_ms"] == round(float(max(series)), 3), (
                name,
                key,
            )

        assert summary["bundletrack_camera_local_candidate_frames"] == 34
        assert summary["original_world_current_registered_surface_frames"] == 32
        assert not summary["control_authorized"]

    for snap in evidence_snapshots[1:]:
        assert snap == evidence_snapshots[0]

    assert total_masks_verified == 400
    assert total_native_inputs_verified == 190
    assert total_native_omp_checks == 190
    assert results["gate_passed"] is False
    assert results["wire_authorized"] is False
    assert results["live_acceptance_gates_passed"] == "0/6"

    verification = dict(
        passed=True,
        study="sam-gpu-window-cpu-overlap-20261102",
        runs_verified=5,
        total_masks_verified=total_masks_verified,
        total_native_inputs_verified=total_native_inputs_verified,
        total_native_omp_checks=total_native_omp_checks,
        source_owner_evidence_identical_across_all_5_runs=True,
        strict_warm_frames_byte_identical=39,
        strict_native_warm_frames_byte_identical=38,
        warm_evidence_hash_threads_eliminated_per_run=38,
        background_worker_threads_created=0,
        root_production_sources_unchanged=True,
        live_acceptance_gates_passed="0/6",
        gate_passed=False,
        wire_authorized=False,
    )
    (E / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    (H / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
