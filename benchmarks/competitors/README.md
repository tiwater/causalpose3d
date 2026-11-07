# Competitor benchmark snapshots

This directory is deliberately **outside the product and CI paths**.

- Competitor frameworks are not dependencies of `evidencescene`.
- Their code is not imported, built, or tested by the package CI.
- Snapshots are produced manually or periodically in an isolated environment.
- Each snapshot is dated and includes its frozen input hash, hardware,
  software, exact commands, and caveats.
- A snapshot is a benchmark comparison, not a maintained leaderboard and not an
  EvidenceScene release gate.

The first snapshot is [`2026-11-06-ycb/`](2026-11-06-ycb/): EvidenceScene's
native pose branch vs Point2Pose on 40 YCBInEOAT `bleach0` frames. It is not
run by `.github/workflows/ci.yml`.

To add a new snapshot, create a dated directory and keep the result schema
`evidencescene.system-performance.v1`. Do not add competitor packages to
`pyproject.toml`, the package tests, or the release workflow.
