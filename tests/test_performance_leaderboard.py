"""Contract checks for the homepage performance dashboard."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_tracks_own_metrics_and_external_snapshot_separately() -> None:
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
    assert snapshot["protocol"]
    assert snapshot["url"].startswith("https://")
    assert "not run by CI" in snapshot["note"]


def test_homepage_renders_own_dashboard_without_live_competitor_table() -> None:
    page = (ROOT / "docs" / "index.html").read_text()
    assert "PERFORMANCE_LEADERBOARD_START" in page
    assert "PERFORMANCE_LEADERBOARD_END" in page
    block = page.split("PERFORMANCE_LEADERBOARD_START", 1)[1].split(
        "PERFORMANCE_LEADERBOARD_END", 1
    )[0]
    assert "EvidenceScene kernel performance" in block
    assert "External competitor snapshot" in block
    assert "Controlled same-input comparison" not in block
    for competitor in ("BundleSDF", "FoundationPose"):
        assert competitor not in block
