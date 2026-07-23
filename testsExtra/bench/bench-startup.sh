#!/usr/bin/env bash
# Benchmark asy startup time by running it N times and reporting total and
# per-run wall-clock time.
#
# Usage (from this directory):
#   ./bench-startup.sh [N]
# N defaults to 50.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# ASY and BASE default to an in-tree build but can be overridden, e.g. to point
# at an out-of-tree CMake build:
#   ASY=~/.local/asy-build/sandbox/asy BASE=~/.local/asy-build/sandbox/base ./bench-startup.sh
ASY="${ASY:-${SCRIPT_DIR}/../../asy}"
BASE="${BASE:-${SCRIPT_DIR}/../../base}"
N="${1:-50}"

echo "Running '$ASY -dir $BASE startup' $N times..."

# Milliseconds from a monotonic high-resolution clock. Uses python3 (a build
# dependency, so always available) to stay portable across Linux and macOS,
# where 'date +%N' is unsupported.
now_ms() {
    python3 -c 'import time; print(round(time.perf_counter() * 1000))'
}

start=$(now_ms)
for ((i = 1; i <= N; i++)); do
    "$ASY" -dir "$BASE" "$SCRIPT_DIR/startup"
done
end=$(now_ms)

total_ms=$(( end - start ))
per_run_ms_x100=$(( total_ms * 100 / N ))

echo "Total:    ${total_ms} ms"
printf 'Per run:  %d.%02d ms  (N=%d)\n' "$(( per_run_ms_x100 / 100 ))" "$(( per_run_ms_x100 % 100 ))" "$N"
