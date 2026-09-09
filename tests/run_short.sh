#!/usr/bin/env bash
# Short-time CWSR smoke tests (alloy database + real Matbench task).
#
# These run a *small* CWSR search on a downsampled subset of each database so
# they complete in roughly ~15-30 s each. They verify the whole task-agnostic
# pipeline end-to-end (dataset load via cwsr.datasets -> split -> fit ->
# forward sanity check). They do NOT find an accurate model - use larger
# budgets and the full datasets for real runs.
#
# Requirements: run inside the 'cwsr' conda env.
#   conda activate cwsr
#   ./tests/run_short.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

PY="${PYTHON:-python}"
OUT="${OUT:-tests/_out}"

echo "=============================================="
echo "1/2  Alloy database smoke test (density)"
echo "      data: examples/alloys/data (self-contained copy)"
echo "=============================================="
$PY tests/smoke_alloy.py \
    --property density \
    --max-samples 150 --max-expressions 80 \
    --num-parallel 4 --seed 1 --output-dir "$OUT"

echo
echo "=============================================="
echo "2/2  Matbench smoke test (matbench_glass)"
echo "      (first run prompts + downloads/caches the dataset)"
echo "=============================================="
$PY tests/smoke_matbench.py \
    --task matbench_glass \
    --max-samples 150 --max-expressions 80 \
    --num-parallel 4 --seed 1 --output-dir "$OUT"

echo
echo "All short-time CWSR smoke tests passed."
