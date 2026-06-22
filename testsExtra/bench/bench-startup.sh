#!/usr/bin/env bash
# Benchmark asy startup time by running it N times and reporting total and
# per-run wall-clock time.
#
# Usage (from this directory):
#   ./bench-startup.sh [N]
# N defaults to 50.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASY="${SCRIPT_DIR}/../../asy"
BASE="${SCRIPT_DIR}/../../base"
N="${1:-50}"

echo "Running '$ASY -dir $BASE startup' $N times..."

start=$(date +%s%N)
for ((i = 1; i <= N; i++)); do
    "$ASY" -dir "$BASE" "$SCRIPT_DIR/startup"
done
end=$(date +%s%N)

total_ms=$(( (end - start) / 1000000 ))
per_ms=$(echo "scale=2; $total_ms / $N" | bc)

echo "Total:    ${total_ms} ms"
echo "Per run:  ${per_ms} ms  (N=$N)"
