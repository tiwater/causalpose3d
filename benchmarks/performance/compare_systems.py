"""Compare recorded-session system performance results fairly.

All inputs must share the same ``protocol`` and frozen input manifest hash.
Output digests may differ because different systems produce different outputs;
quality must be compared separately.

Usage::

    python3 benchmarks/performance/compare_systems.py a.json b.json
    python3 benchmarks/performance/compare_systems.py --out leaderboard.md a.json b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = ("system", "version", "protocol", "session", "environment", "metrics")


def _load(path: Path) -> dict:
    data = json.loads(path.read_text())
    missing = [key for key in REQUIRED if key not in data]
    if missing:
        raise ValueError(f"{path}: missing required keys: {missing}")
    session = data["session"]
    for key in ("id", "frames", "input_sha256"):
        if key not in session:
            raise ValueError(f"{path}: missing session.{key}")
    return data


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return str(value)


def compare(results: list[dict]) -> tuple[str, bool]:
    keys = {(row["protocol"], row["session"]["input_sha256"], row["session"]["frames"], row["session"]["id"]) for row in results}
    comparable = len(keys) == 1
    first = results[0]
    lines = [
        "# System performance comparison",
        "",
        f"- Protocol: `{first['protocol']}`",
        f"- Session: `{first['session']['id']}` ({first['session']['frames']} frames)",
        f"- Input SHA-256: `{first['session']['input_sha256']}`",
        f"- Comparable inputs: **{'yes' if comparable else 'NO'}**",
        "",
        "| System | Version | p50 (ms) | p95 (ms) | Peak RSS (MB) | Non-lost frames | Output digest |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results:
        metrics = row["metrics"]
        lines.append(
            f"| {row['system']} | `{row['version']}` | "
            f"{_fmt(metrics.get('p50_ms'))} | {_fmt(metrics.get('p95_ms'))} | "
            f"{_fmt(metrics.get('peak_rss_mb'))} | {_fmt(metrics.get('nonlost_frames'))} | "
            f"`{metrics.get('output_digest') or 'n/a'}` |"
        )
    if not comparable:
        lines.extend(["", "## Not comparable", "", "The inputs disagree on protocol/session identity:"])
        for key in sorted(keys):
            lines.append(f"- `{key}`")
    lines.append("")
    return "\n".join(lines), comparable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = [_load(path) for path in args.results]
    text, comparable = compare(rows)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0 if comparable else 2


if __name__ == "__main__":
    raise SystemExit(main())
