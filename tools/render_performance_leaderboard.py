"""Render the homepage performance leaderboard from checked-in JSON.

The generated block lives between two HTML comments in ``docs/index.html``.
Keeping the data separate makes every number traceable and lets a version
comparison be updated without hand-editing table markup.
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


def _controlled_table(section: dict) -> str:
    rows = []
    for row in section["rows"]:
        css = ' class="highlight-row"' if row.get("highlight") else ""
        rows.append(
            f"<tr{css}>"
            f"<td><strong>{html.escape(row['system'])}</strong><br><code>{html.escape(row.get('variant', ''))}</code></td>"
            f"<td>{html.escape(row['scope'])}</td>"
            f"<td><strong>{_fmt(row.get('p50_ms'))}</strong></td>"
            f"<td>{_fmt(row.get('p95_ms'))}</td>"
            f"<td>{_fmt(row.get('throughput_fps'), 1)}</td>"
            f"<td>{_fmt(row.get('peak_rss_mb'), 1)}</td>"
            f"<td>{html.escape(row.get('frames', ''))}</td>"
            f"<td>{_link(row.get('source', 'source'), row.get('source_url', ''))}</td>"
            "</tr>"
        )
    return f"""
      <h3>Controlled same-input comparison</h3>
      <p class="caption" style="margin-top:0.35rem;">
        Protocol <code>{html.escape(section['protocol'])}</code> · {html.escape(section['session'])} · {section['frames']} frames ·
        input SHA-256 <code>{html.escape(section['input_sha256'][:16])}…</code><br>
        {html.escape(section['hardware'])} · {html.escape(section['software'])}
      </p>
      <div class="table-wrapper">
        <table class="benchmark-table">
          <thead><tr>
            <th>System</th><th>Scope</th><th>p50 (ms)</th><th>p95 (ms)</th><th>Throughput (fps)</th><th>Peak RSS (MB)</th><th>Frames</th><th>Source</th>
          </tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
      <p class="caption">{html.escape(section['notes'])}</p>
"""


def _reference_table(section: dict) -> str:
    rows = []
    for row in section["rows"]:
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(row['system'])}</strong></td>"
            f"<td>{html.escape(row['scope'])}</td>"
            f"<td>{html.escape(row.get('latency_text', 'n/a'))}</td>"
            f"<td>{html.escape(row.get('hardware', ''))}</td>"
            f"<td>{_link(row.get('source', 'source'), row.get('source_url', ''))}</td>"
            "</tr>"
        )
    return f"""
      <h3 style="margin-top:2rem;">Reference context (not directly comparable)</h3>
      <p class="caption" style="margin-top:0.35rem;">{html.escape(section['notes'])}</p>
      <div class="table-wrapper">
        <table class="benchmark-table">
          <thead><tr>
            <th>System</th><th>Scope / protocol</th><th>Reported latency</th><th>Hardware</th><th>Source</th>
          </tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
"""


def render() -> str:
    data = json.loads(DATA.read_text())
    return (
        '<div class="claim-callout">'
        "<strong>How to read this table:</strong> the controlled rows share one frozen input, host and GPU. "
        "The reference rows come from different inputs/protocols and are shown only as context — they are not a ranking. "
        f'Last updated: {html.escape(data["updated"])}. '
        'Full protocol and commands: '
        '<a href="https://github.com/tiwater/evidencescene/tree/main/benchmarks/performance" target="_blank" rel="noopener">benchmarks/performance</a>.'
        "</div>\n"
        + _controlled_table(data["controlled"])
        + _reference_table(data["reference"])
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
