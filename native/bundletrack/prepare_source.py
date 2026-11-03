"""Fetch exactly the pinned BundleSDF source and apply its allowed patches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
FORBIDDEN = ("BUNDLESDF_SKIP_BA", "FEATURE_RESIZE", "threshold", "DISABLE_NERF")


def _run(*command: str, cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE).stdout


def prepare(destination: Path) -> None:
    if destination.resolve() == Path("/tmp/BundleSDF").resolve():
        raise ValueError("/tmp/BundleSDF ad hoc modifications are forbidden")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("BundleSDF destination must be empty")
    patches = tuple(ROOT / item["path"] for item in MANIFEST["patches"])
    for patch, item in zip(patches, MANIFEST["patches"], strict=True):
        if hashlib.sha256(patch.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"BundleSDF patch hash mismatch: {item['path']}")
    _run("git", "clone", MANIFEST["upstream"]["url"], str(destination))
    _run("git", "checkout", "--detach", MANIFEST["upstream"]["revision"], cwd=destination)
    if _run("git", "rev-parse", "HEAD", cwd=destination).strip() != MANIFEST["upstream"]["revision"]:
        raise ValueError("BundleSDF revision mismatch")
    for patch in patches:
        _run("git", "apply", "--check", str(patch), cwd=destination)
        _run("git", "apply", str(patch), cwd=destination)
    diff = _run("git", "diff", "--no-ext-diff", "--binary", cwd=destination)
    names = _run("git", "diff", "--name-only", cwd=destination).splitlines()
    expected_names = ["BundleTrack/CMakeLists.txt", "Utils.py"]
    if names != expected_names:
        raise ValueError("prepared BundleSDF source differs from the declared patch set")
    if any(token in diff for token in FORBIDDEN):
        raise ValueError("ad hoc BundleSDF modifications are forbidden")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    prepare(parser.parse_args().destination)
