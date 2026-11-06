#!/usr/bin/env bash
# Run the same kernel suite against two git refs and print a Markdown diff.
#
#   ./benchmarks/performance/run_versions.sh <baseline-ref> [candidate-ref]
#
# The current suite script is used for both runs; --package-root selects the
# package source under test.  Refs must contain the `evidencescene` package
# (the v2 rename or later).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_REF="${1:-04a1b6b}"
CAND_REF="${2:-HEAD}"
OUT_DIR="${ROOT}/benchmarks/performance/results"
mkdir -p "${OUT_DIR}"
TMP="$(mktemp -d)"
cleanup() {
  git -C "${ROOT}" worktree remove --force "${TMP}/base" >/dev/null 2>&1 || true
  git -C "${ROOT}" worktree remove --force "${TMP}/candidate" >/dev/null 2>&1 || true
  rm -rf "${TMP}"
}
trap cleanup EXIT

git -C "${ROOT}" worktree add --detach "${TMP}/base" "${BASE_REF}" >/dev/null
git -C "${ROOT}" worktree add --detach "${TMP}/candidate" "${CAND_REF}" >/dev/null
for side in base candidate; do
  if [[ ! -d "${TMP}/${side}/evidencescene" ]]; then
    echo "ref ${side} does not contain the evidencescene package (pre-v2 rename?)" >&2
    exit 2
  fi
done

safe_base="$(printf '%s' "${BASE_REF}" | tr '/:' '__')"
safe_cand="$(printf '%s' "${CAND_REF}" | tr '/:' '__')"
BASE_JSON="${OUT_DIR}/${safe_base}.json"
CAND_JSON="${OUT_DIR}/${safe_cand}.json"
REPORT="${OUT_DIR}/${safe_base}_vs_${safe_cand}.md"

python3 "${ROOT}/benchmarks/performance/bench_suite.py" \
  --package-root "${TMP}/base" --label "${BASE_REF}" --out "${BASE_JSON}" >/dev/null
python3 "${ROOT}/benchmarks/performance/bench_suite.py" \
  --package-root "${TMP}/candidate" --label "${CAND_REF}" --out "${CAND_JSON}" >/dev/null
python3 "${ROOT}/benchmarks/performance/compare.py" \
  "${BASE_JSON}" "${CAND_JSON}" --require-identical --out "${REPORT}"

echo
echo "artifact: ${REPORT}"
