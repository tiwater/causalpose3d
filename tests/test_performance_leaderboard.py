"""Contract checks for the homepage performance dashboard."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_keeps_own_and_external_data_separate() -> None:
    data = json.loads((ROOT / "docs" / "performance_leaderboard.json").read_text())
    assert data["schema"] == "evidencescene.performance-dashboard.v1"
    kernel = data["kernel"]
    assert kernel["protocol"] == "kernel-v1"
    assert len(kernel["rows"]) == 7
    for row in kernel["rows"]:
        assert row["baseline_ms"] > 0 and row["candidate_ms"] > 0
        assert row["speedup"] > 0

    snapshot = data["external_snapshot"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", snapshot["date"])
    controlled = snapshot["controlled"]
    assert controlled["protocol"] == "ycbineoat_bleach0_refmask_40f_v1"
    assert re.fullmatch(r"[0-9a-f]{64}", controlled["input_sha256"])
    assert len(controlled["rows"]) >= 2
    for row in controlled["rows"]:
        assert row["p50_ms"] > 0 and row["p95_ms"] >= row["p50_ms"]
    assert len(snapshot["reference"]["rows"]) >= 4
    assert "not run by CI" in snapshot["note"]


def test_homepage_renders_own_and_dated_external_tables() -> None:
    page = (ROOT / "docs" / "index.html").read_text()
    block = page.split("PERFORMANCE_LEADERBOARD_START", 1)[1].split(
        "PERFORMANCE_LEADERBOARD_END", 1
    )[0]
    assert "EvidenceScene kernel performance" in block
    assert "External competitor snapshot" in block
    assert "Controlled same-input subset" in block
    assert "Reference context" in block
    assert "ycbineoat_bleach0_refmask_40f_v1" in block
    assert "BundleSDF" in block
    assert "FoundationPose" in block
