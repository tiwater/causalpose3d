"""Render EvidenceScene's own performance dashboard from checked-in JSON.

Competitor comparisons are intentionally not rendered as a live table: they
live as dated, isolated snapshots under ``benchmarks/competitors/`` and are
linked, not continuously integrated.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "performance_leaderboard.json"
PAGE = ROOT / "docs" / "index.html"
START = "<!-- PERFORMANCE_LEADERBOARD_START -->"
END = "<!-- PERFORMANCE_LEADERBOARD_END -->"


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _link(text: str, url: str) -> str:
    safe_text = html.escape(text)
    safe_url = html.escape(url, quote=True)
    return f'<a href="{safe_url}" target="_blank" rel="noopener">{safe_text}</a>' if url else safe_text


def _kernel_table(section: dict) -> str:
    rows = []
    for row in section["rows"]:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(row['case'])}</code></td>"
            f"<td>{_fmt(row['baseline_ms'], 3)}</td>"
            f"<td><strong>{_fmt(row['candidate_ms'], 3)}</strong></td>"
            f"<td>{_fmt(row['speedup'], 2)}x</td>"
            "</tr>"
        )
    return f"""
      <h3>EvidenceScene kernel performance (<code>{html.escape(section['protocol'])}</code>)</h3>
      <p class="caption" style="margin-top:0.35rem;">
        Baseline <code>{html.escape(section['baseline'])}</code> → candidate <code>{html.escape(section['candidate_label'])}</code>
        (<code>{html.escape(section['candidate'])}</code>). Median latency in milliseconds; higher speed-up is better.
      </p>
      <div class="table-wrapper">
        <table class="benchmark-table">
          <thead><tr>
            <th>Workload</th><th>Baseline (ms)</th><th>Current (ms)</th><th>Speed-up</th>
          </tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
      <p class="caption">
        Geometric-mean speed-up: <strong>{_fmt(section['geomean_speedup'], 2)}x</strong>;
        output digests: <strong>{html.escape(section['output_parity'])}</strong>.
        {html.escape(section['notes'])}
      </p>
"""


def _external_snapshot(section: dict) -> str:
    return f"""
      <h3 style="margin-top:2rem;">External competitor snapshot</h3>
      <p class="caption" style="margin-top:0.35rem;">
        {_link(f"Snapshot {section['date']}", section['url'])} ·
        protocol <code>{html.escape(section['protocol'])}</code>.
        {html.escape(section['note'])}
      </p>
"""


def render() -> str:
    data = json.loads(DATA.read_text())
    return (
        '<div class="claim-callout">'
        "<strong>Scope:</strong> this dashboard reports EvidenceScene's own regression and version performance. "
        "Competitor comparisons are dated, isolated snapshots — not CI jobs, package dependencies or a live leaderboard. "
        f'Last updated: {html.escape(data["updated"])}.'
        "</div>\n"
        + _kernel_table(data["kernel"])
        + _external_snapshot(data["external_snapshot"])
    )


def main() -> int:
    page = PAGE.read_text()
    block = START + "\n" + render() + END
    if START not in page or END not in page:
        raise SystemExit("performance leaderboard markers are missing")
    before, remainder = page.split(START, 1)
    _, after = remainder.split(END, 1)
    PAGE.write_text(before + block + after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
