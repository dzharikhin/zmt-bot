#!/usr/bin/env bash
# Gates study: outlier method x budget x selection (NSGA-II, lfr@0.8 + dfa@0.775).
# Features must already be extracted as parquet shards under FEATURES_DIR
# (default: ./features/{like,dislike}/*.parquet).
# Usage: ./benchmark/run_gates_study.sh [FEATURES_DIR] [N_ITERATIONS]
set -euo pipefail

FEATURES_DIR="${1:-features}"
N_ITERATIONS="${2:-60}"
if [ -z "${RUNNER:-}" ]; then
    if command -v poetry >/dev/null 2>&1; then
        RUNNER="poetry run python"
    else
        RUNNER="python"
    fi
fi
OUT_DIR="data/benchmark"
mkdir -p "$OUT_DIR"

"$RUNNER" -m benchmark.gates_study \
    --features-dir "$FEATURES_DIR" \
    --n-iterations "$N_ITERATIONS" \
    --output "$OUT_DIR/gates_study.json"
