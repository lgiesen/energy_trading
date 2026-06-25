#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[WARN] run_rq3_tft_p90_market_benchmark.sh is deprecated; use run_rq3_xgb_p50_market_benchmark.sh." >&2
exec "$SCRIPT_DIR/run_rq3_xgb_p50_market_benchmark.sh" "$@"
