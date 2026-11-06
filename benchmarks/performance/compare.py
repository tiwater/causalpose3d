"""Compare two EvidenceScene benchmark JSON files.

Usage::

    python3 benchmarks/performance/compare.py baseline.json candidate.json
    python3 benchmarks/performance/compare.py baseline.json candidate.json \
        --out leaderboard.md --require-identical

A comparison is only valid when the workload output digests match; use
``--require-identical`` in CI or release gates.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _geomean(values: list[float]) -> float | None:
    if not values or any(value <= 0 for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def compare(baseline: dict, candidate: dict) -> tuple[str, bool]:
    base_results = {row["case"]: row for row in baseline.get("results", [])}
    cand_results = {row["case"]: row for row in candidate.get("results", [])}
    cases = [case for case in base_results if case in cand_results]
    identical = True
    speedups = []
    lines = [
        f"# {candidate.get('label', 'candidate')} vs {baseline.get('label', 'baseline')}",
        "",
        "| Metadata | Baseline | Candidate |",
        "| --- | --- | --- |",
        f"| Label | `{baseline.get('label', '')}` | `{candidate.get('label', '')}` |",
        f"| Git SHA | `{(baseline.get('package') or {}).get('git_sha') or 'n/a'}` | `{(candidate.get('package') or {}).get('git_sha') or 'n/a'}` |",
        f"| Package | `{(baseline.get('package') or {}).get('version', 'n/a')}` | `{(candidate.get('package') or {}).get('version', 'n/a')}` |",
        f"| CPU | {((baseline.get('environment') or {}).get('cpu') or 'n/a')} | {((candidate.get('environment') or {}).get('cpu') or 'n/a')} |",
        "",
        "| Workload | Baseline median (ms) | Candidate median (ms) | Δ | Speed-up | Output digest |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in cases:
        base = base_results[case]
        cand = cand_results[case]
        if base.get("status") != "ok" or cand.get("status") != "ok":
            lines.append(
                f"| `{case}` | {_fmt(base.get('median_ms'))} | {_fmt(cand.get('median_ms'))} | n/a | n/a | error |"
            )
            identical = False
            continue
        base_median = float(base["median_ms"])
        cand_median = float(cand["median_ms"])
        delta = cand_median - base_median
        speedup = base_median / cand_median if cand_median > 0 else float("inf")
        speedups.append(speedup)
        digest_match = base.get("output_digest") == cand.get("output_digest")
        identical = identical and digest_match
        lines.append(
            f"| `{case}` | {_fmt(base_median)} | {_fmt(cand_median)} | "
            f"{delta:+.3f} ms | {speedup:.2f}x | "
            f"{'identical' if digest_match else '**MISMATCH**'} |"
        )
    geo = _geomean(speedups)
    lines.extend(
        [
            "",
            f"- Workloads compared: **{len(cases)}**",
            f"- Geometric-mean speed-up: **{_fmt(geo, 2)}x**" if geo else "- Geometric-mean speed-up: n/a",
            f"- Output digests: **{'all identical' if identical else 'not all identical'}**",
            f"- Scope: {candidate.get('scope', 'unknown')}",
            "",
        ]
    )
    return "\n".join(lines), identical


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--require-identical", action="store_true")
    args = parser.parse_args(argv)
    text, identical = compare(_load(args.baseline), _load(args.candidate))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    if args.require_identical and not identical:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
