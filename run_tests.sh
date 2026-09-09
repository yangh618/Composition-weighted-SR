#!/usr/bin/env bash
# Run all CWSR tests: static compile, fast sanity checks, the alloy and Matbench
# smoke tests, and (if available) the strict docs build.
#
# Usage:
#   ./run_tests.sh
#   MATBENCH_DOWNLOAD=0 ./run_tests.sh     # never download Matbench data (abort if not cached)
#   SMOKE_EXPRESSIONS=40 ./run_tests.sh    # smaller smoke budget for CI/speed
#
# Exit code 0 if everything passes, 1 otherwise.

set -uo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

PY="${PYTHON:-python}"
OUT="${OUT:-tests/_out}"
mkdir -p "$OUT"

ALLOW_PROP="${ALLOW_PROP:-density}"
MATBENCH_TASK="${MATBENCH_TASK:-matbench_glass}"
SMOKE_EXPRESSIONS="${SMOKE_EXPRESSIONS:-80}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-150}"
NUM_PARALLEL="${NUM_PARALLEL:-4}"

if [ "${MATBENCH_DOWNLOAD:-1}" = "1" ]; then
    MATBENCH_DL="--yes"
else
    MATBENCH_DL="--no"
fi

failures=0
passed=0

note()  { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
pass()  { printf '\033[32m[PASS]\033[0m %s\n' "$1"; passed=$((passed + 1)); }
fail()  { printf '\033[31m[FAIL]\033[0m %s\n' "$1"; failures=$((failures + 1)); }

run() { # run <label> cmd...
    local label="$1"; shift
    if "$@"; then pass "$label"; else fail "$label"; fi
}

note "1/6  Compile all sources (syntax check)"
if find cwsr tests -name '*.py' -print0 | xargs -0 -n1 "$PY" -m py_compile \
        && "$PY" -m py_compile eval.py run_cwsr.py setup.py; then
    pass "py_compile all modules"
else
    fail "py_compile all modules"
fi

note "2/6  Fast sanity checks (imports, formula, splits, gradients, query CLI)"
run "fast sanity (tests/check_env.py)" "$PY" tests/check_env.py

note "3/6  Alloy smoke test ($ALLOW_PROP)"
run "alloy smoke ($ALLOW_PROP)" "$PY" tests/smoke_alloy.py \
    --property "$ALLOW_PROP" \
    --max-samples "$SMOKE_SAMPLES" --max-expressions "$SMOKE_EXPRESSIONS" \
    --num-parallel "$NUM_PARALLEL" --seed 1 --output-dir "$OUT"

note "4/6  Matbench smoke test ($MATBENCH_TASK)"
run "matbench smoke ($MATBENCH_TASK)" "$PY" tests/smoke_matbench.py \
    --task "$MATBENCH_TASK" \
    --max-samples "$SMOKE_SAMPLES" --max-expressions "$SMOKE_EXPRESSIONS" \
    --num-parallel "$NUM_PARALLEL" --seed 1 --output-dir "$OUT" \
    "$MATBENCH_DL"

note "5/6  CLI entry points (query help)"
run "cwsr-query --help" "$PY" -m cwsr.predict.query --help >/dev/null

note "6/6  Docs (strict MkDocs build, if available)"
if "$PY" -c "import mkdocs" 2>/dev/null; then
    run "mkdocs build --strict" "$PY" -m mkdocs build --strict -d /tmp/cwsr_test_site
else
    echo "  (mkdocs not installed - skipping docs build)"
fi

echo
echo "=============================================="
echo " RESULTS: $passed passed, $failures failed"
echo "=============================================="
[ "$failures" -eq 0 ]
