#!/usr/bin/env bash
# dFA gate study: operating-point sweep (A/B) + liked-only per-model search (C)
# in one session. Features must be cached (or will be extracted on first run).
# Usage: ./benchmark/run_dfa_gate_study.sh USER_ID [N_ITERATIONS]
# Results are written to data/benchmark/ (NAS mount in Docker).
set -euo pipefail

USER_ID="${1:?usage: run_dfa_gate_study.sh USER_ID [N_ITERATIONS]}"
N_ITERATIONS="${2:-40}"
if [ -z "${RUNNER:-}" ]; then
    if command -v poetry >/dev/null 2>&1; then
        RUNNER="poetry run python"
    else
        RUNNER="python"
    fi
fi
OUT_DIR="data/benchmark"
mkdir -p "$OUT_DIR"

"$RUNNER" -m benchmark.dfa_gate_study \
    --config benchmark/full_only.yaml \
    --user-id "$USER_ID" \
    --n-iterations "$N_ITERATIONS" \
    --output "$OUT_DIR/dfa_gate_study.json" \
    --scores-output "$OUT_DIR/dfa_gate_scores.npz"
