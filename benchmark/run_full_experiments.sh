#!/usr/bin/env bash
# Phase 1 experiment driver: full+PANNs variant only, features must be cached.
# Usage: ./benchmark/run_full_experiments.sh USER_ID [N_ITERATIONS]
# Results are written to data/benchmark/ (NAS mount in Docker).
set -euo pipefail

USER_ID="${1:?usage: run_full_experiments.sh USER_ID [N_ITERATIONS]}"
N_ITERATIONS="${2:-40}"
if [ -z "${RUNNER:-}" ]; then
    if command -v poetry >/dev/null 2>&1; then
        RUNNER="poetry run python"
    else
        RUNNER="python"
    fi
fi
CFG="benchmark/full_only.yaml"
OUT_DIR="data/benchmark"
mkdir -p "$OUT_DIR"

"$RUNNER" -m benchmark.compare \
    --config "$CFG" \
    --objective-weights 0.5 0.5 \
    --output "$OUT_DIR/full_w5050.json" \
    --user-id "$USER_ID" \
    --n-iterations "$N_ITERATIONS"

"$RUNNER" -m benchmark.compare \
    --config "$CFG" \
    --objective-weights 0.3 0.7 \
    --output "$OUT_DIR/full_w307.json" \
    --user-id "$USER_ID" \
    --n-iterations "$N_ITERATIONS"

"$RUNNER" -m benchmark.compare \
    --config "$CFG" \
    --objective-weights 0.2 0.8 \
    --output "$OUT_DIR/full_w208.json" \
    --user-id "$USER_ID" \
    --n-iterations "$N_ITERATIONS"

"$RUNNER" -m benchmark.compare \
    --config "$CFG" \
    --objective-weights 0.5 0.5 \
    --preprocessor standardize+select_128 \
    --output "$OUT_DIR/full_w5050_sel128.json" \
    --user-id "$USER_ID" \
    --n-iterations "$N_ITERATIONS"

"$RUNNER" -m benchmark.compare \
    --config "$CFG" \
    --objective-weights 0.5 0.5 \
    --preprocessor standardize+select_256 \
    --output "$OUT_DIR/full_w5050_sel256.json" \
    --user-id "$USER_ID" \
    --n-iterations "$N_ITERATIONS"

"$RUNNER" -m benchmark.analyze_pareto \
    --reports "$OUT_DIR/segment_report.json" \
    "$OUT_DIR/full_w5050.json" \
    "$OUT_DIR/full_w307.json" \
    "$OUT_DIR/full_w208.json" \
    "$OUT_DIR/full_w5050_sel128.json" \
    "$OUT_DIR/full_w5050_sel256.json" \
    --variant full \
    --output "$OUT_DIR/full_pooled_analysis.json" \
    --plot "$OUT_DIR/full_pareto.png"
