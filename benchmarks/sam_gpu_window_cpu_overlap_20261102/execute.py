"""Execute validate + counterbalanced control/overlap paired processes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

H = Path(__file__).resolve().parent
R = H.parents[2]
E = R / "artifacts/evaluation/sam-gpu-window-cpu-overlap-20261102"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    E.mkdir(parents=True, exist_ok=False)
    source = json.loads(
        (
            R
            / "artifacts/evaluation/existing-owner-opt-in-contract-20261101/configured-route-final/summary.json"
        ).read_text()
    )
    base = json.loads(
        (
            R
            / "artifacts/evaluation/scoped-sam-cpu-preparation-20261101/integrated-execution.json"
        ).read_text()
    )["runs"][1]["command"]

    bound_paths = [
        H / "worker.py",
        Path(__file__),
        R / "docs/plans/2026-11-02-sam-gpu-window-cpu-overlap.md",
        R / "apps/scene-runtime/ticos_vision/perception/sam2_gpu_window_overlap.py",
        R / "apps/scene-runtime/ticos_vision/inference/run_live.py",
        R / "apps/scene-runtime/ticos_vision/perception/original_persistent_world.py",
        R / "apps/scene-runtime/ticos_vision/perception/sam2_video_memory.py",
        R / "apps/replay-api/ticos_replay/sam2_observation_session.py",
        R / "apps/replay-api/ticos_replay/bundletrack_live_input.py",
        R / "apps/replay-api/ticos_replay/pinned_bundletrack_runner.py",
        R / "apps/replay-api/ticos_replay/causal_object_surface.py",
    ]
    bindings = {str(p.relative_to(R)): sha(p) for p in bound_paths}
    runs = []

    schedule = [
        ("validate", 1),
        ("control", 1),
        ("overlap", 1),
        ("overlap", 2),
        ("control", 2),
    ]
    for mode, repeat in schedule:
        target = E / f"{mode}-{repeat}"
        assert not target.exists()
        command = list(base)
        command[1] = str(H / "worker.py")
        command[command.index("--output") + 1] = str(target)

        env = dict(
            os.environ,
            PYTHONPATH="apps/replay-api:apps/scene-runtime",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            TICOS_SAM_GPU_WINDOW_OVERLAP_MODE=mode,
            TICOS_VISION_ORIGINAL_WORLD_SAM2_CPU_PREPARE_THREADS="8",
        )
        env.update(source["native_runtime_environment"])
        env["TICOS_VISION_ORIGINAL_WORLD_OUTPUT_ROOT"] = str(target / "native")
        env["TICOS_VISION_ORIGINAL_WORLD_SAM2_CPU_PREPARE_THREADS"] = "8"
        env["LD_LIBRARY_PATH"] = (
            source["native_runtime_environment"][
                "TICOS_VISION_ORIGINAL_WORLD_BUNDLESDF_BUILD_ROOT"
            ]
            + ":/home/hugh/.local/bundlesdf-deps/prefix/lib:"
            + env.get("LD_LIBRARY_PATH", "")
        )

        log = E / f"{mode}-{repeat}.log"
        assert not log.exists()
        started = time.time()
        with log.open("w") as handle:
            code = subprocess.call(
                command,
                cwd=R,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        elapsed = time.time() - started
        assert all(
            sha(R / rel) == digest for rel, digest in bindings.items()
        ), "Bound source files changed during execution"

        runs.append(
            dict(
                mode=mode,
                repeat=repeat,
                command=command,
                exit_code=code,
                seconds=round(elapsed, 3),
                log=str(log.relative_to(R)),
                log_sha256=sha(log),
            )
        )
        (E / "execution.json").write_text(
            json.dumps(
                dict(
                    runs=runs,
                    source_bindings=bindings,
                    source_unchanged=True,
                    wire_authorized=False,
                ),
                indent=2,
            )
            + "\n"
        )
        print(mode, repeat, code, round(elapsed, 2), flush=True)
        if code != 0:
            raise SystemExit(code)
        if mode == "validate":
            proof = json.loads((target / "gpu-window-overlap.json").read_text())
            warm = proof["warm_frames"]
            assert len(warm) == 39, len(warm)
            assert all(
                w["eligible"]
                and w["executed_in_gpu_window"]
                and w.get("prepared_frame_exact")
                for w in warm
            ), "Validate mode failed GPU-window preparation checks"
            native_warm = [w for w in warm if w.get("native_buffers_reused")]
            assert len(native_warm) == 38, len(native_warm)
            assert all(
                w.get("evidence_hash_exact")
                and w.get("native_buffers_exact")
                and (w.get("rle_exact") if w.get("rle_reused") else True)
                for w in native_warm
            ), "Validate mode failed exactness checks on native warm frames"


if __name__ == "__main__":
    main()
