"""Contract checks for the checked-in homepage performance leaderboard."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_leaderboard_data_is_complete_and_traceable() -> None:
    data = json.loads((ROOT / "docs" / "performance_leaderboard.json").read_text())
    assert data["schema"] == "evidencescene.performance-leaderboard.v1"
    controlled = data["controlled"]
    assert controlled["protocol"] == "ycbineoat_bleach0_refmask_40f_v1"
    assert controlled["frames"] == 40
    assert re.fullmatch(r"[0-9a-f]{64}", controlled["input_sha256"])
    assert len(controlled["rows"]) >= 2
    for row in controlled["rows"]:
        assert row["p50_ms"] > 0 and row["p95_ms"] >= row["p50_ms"]
        assert row["source_url"]
    for row in data["reference"]["rows"]:
        assert row["scope"] and row["source"]


def test_homepage_contains_rendered_leaderboard() -> None:
    page = (ROOT / "docs" / "index.html").read_text()
    assert "PERFORMANCE_LEADERBOARD_START" in page
    assert "PERFORMANCE_LEADERBOARD_END" in page
    block = page.split("PERFORMANCE_LEADERBOARD_START", 1)[1].split(
        "PERFORMANCE_LEADERBOARD_END", 1
    )[0]
    assert "Controlled same-input comparison" in block
    assert "Point2Pose" in block
    assert "Reference context" in block
    assert "ycbineoat_bleach0_refmask_40f_v1" in block
