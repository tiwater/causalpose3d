"""Shared provenance helpers for the controlled YCBInEOAT benchmark."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path


def frame_stems(sequence: Path, frames: int) -> list[str]:
    stems = sorted(path.stem for path in (sequence / "rgb").glob("*.png"))
    if frames > 0:
        stems = stems[:frames]
    if not stems:
        raise SystemExit(f"no RGB frames found under {sequence}")
    return stems


def input_sha256(sequence: Path, stems: list[str]) -> str:
    digest = hashlib.sha256()
    for stem in stems:
        for sub, ext in (("rgb", ".png"), ("depth", ".png"), ("gt_mask", ".png")):
            path = sequence / sub / f"{stem}{ext}"
            digest.update(f"{sub}/{stem}{ext}".encode())
            digest.update(str(path.stat().st_size).encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def p50_p95(samples: list[float], *, warmup: int) -> dict:
    warm = samples[warmup:] if len(samples) > warmup else samples
    if not warm:
        raise SystemExit("no warm samples")
    ordered = sorted(warm)
    return {
        "p50_ms": round(ordered[len(ordered) // 2], 4),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4),
        "min_ms": round(min(warm), 4),
        "max_ms": round(max(warm), 4),
        "samples": len(warm),
        "warmup": warmup,
    }


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def gpu_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def base_metadata(*, protocol: str, sequence: Path, stems: list[str], method: str) -> dict:
    import numpy as np
    import torch

    return {
        "schema": "evidencescene.system-performance.v1",
        "system": method,
        "version": "controlled-ycb-v1",
        "protocol": protocol,
        "session": {
            "id": "ycbineoat/bleach0",
            "frames": len(stems),
            "input_sha256": input_sha256(sequence, stems),
        },
        "environment": {
            "cpu": cpu_model(),
            "gpu": gpu_name(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }


def write_result(path: Path | None, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    print(text)
