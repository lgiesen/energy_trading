#!/usr/bin/env bash
set -Eeuo pipefail

# RQ3 market benchmark launcher.
#
# Purpose:
#   1) determine the best model/quantile combination from the RQ2 numeric
#      Net Profit heatmap;
#   2) assert that the best combination is TFT p90;
#   3) run that fixed policy for the RQ3 market strategies:
#      multi, DA, BEM, BCM and aFRR.
#
# Time window is intentionally copied from run_final_thesis_multi_3m.sh:
#   2025-05-31T22:00:00Z inclusive -> 2025-07-31T22:00:00Z exclusive
#
# This script starts simulations. It does not modify artifacts/simulation_runs
# unless you execute it.

MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-2}"
START="${START:-2025-05-31T22:00:00Z}"
END="${END:-2025-07-31T22:00:00Z}"
SPLIT="${SPLIT:-test}"
MODEL="${MODEL:-tft}"
QUANTILE_PAIR="${QUANTILE_PAIR:-p90-p90}"
EXPECTED_BEST_MODEL="${EXPECTED_BEST_MODEL:-TFT}"
EXPECTED_BEST_QUANTILE="${EXPECTED_BEST_QUANTILE:-p90}"
FINAL_SOC_MODE="${FINAL_SOC_MODE:-hard_min}"
ID_RECOURSE_MODE="${ID_RECOURSE_MODE:-common}"
OUTPUT_DETAIL="${OUTPUT_DETAIL:-thesis}"
DEBUG_DUMPS="${DEBUG_DUMPS:-accepted_only}"
ALLOW_INVALID_OUTPUT="${ALLOW_INVALID_OUTPUT:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"
PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
RQ2_HEATMAP_CSV="${RQ2_HEATMAP_CSV:-artifacts/benchmark/rq2_simulation_benchmark/backup/csv/1_profit_heatmap.csv}"

read -r -a STRATEGIES <<< "${STRATEGIES:-multi da bem bcm afrr}"

RUN_TS="${RUN_TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-artifacts/simulation_runs/rq3_tft_p90_market_benchmark_${RUN_TS}}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/logs}"
STATUS_CSV="$LOG_ROOT/status.csv"
MANIFEST="$LOG_ROOT/manifest.tsv"
BEST_ASSERTION_CSV="$LOG_ROOT/best_model_quantile_assertion.csv"
BEST_ASSERTION_JSON="$LOG_ROOT/best_model_quantile_assertion.json"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

if [[ ! -f "$STATUS_CSV" ]]; then
  echo "run_id,strategy,model,quantile,status,start_utc,end_utc,exit_code,out_dir,log_dir" > "$STATUS_CSV"
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo -e "run_id\tstrategy\tmodel\tquantile\tout_dir\tlog_dir" > "$MANIFEST"
fi

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

active_job_count() {
  jobs -rp | wc -l | tr -d ' '
}

wait_for_slot() {
  while (( $(active_job_count) >= MAX_PARALLEL_JOBS )); do
    sleep 5
  done
}

mark_status() {
  local run_id="$1"
  local strategy="$2"
  local model="$3"
  local quantile="$4"
  local status="$5"
  local start_utc="$6"
  local end_utc="$7"
  local exit_code="$8"
  local out_dir="$9"
  local log_dir="${10}"
  echo "${run_id},${strategy},${model},${quantile},${status},${start_utc},${end_utc},${exit_code},${out_dir},${log_dir}" >> "$STATUS_CSV"
}

job_complete() {
  local out_dir="$1"
  local log_dir="$2"
  [[ -f "$log_dir/.done" ]] || return 1
  [[ -s "$out_dir/strategy_overview.csv" ]] || return 1
  [[ -s "$out_dir/performance_metrics_all_scenarios.csv" ]] || return 1
  return 0
}

assert_best_model_quantile() {
  if [[ ! -s "$RQ2_HEATMAP_CSV" ]]; then
    echo "[ERROR] Missing RQ2 heatmap CSV: $RQ2_HEATMAP_CSV" >&2
    echo "Run the RQ2 visualization generation first, or set RQ2_HEATMAP_CSV=..." >&2
    exit 1
  fi

  "$PYTHON_BIN" - "$RQ2_HEATMAP_CSV" "$EXPECTED_BEST_MODEL" "$EXPECTED_BEST_QUANTILE" "$BEST_ASSERTION_CSV" "$BEST_ASSERTION_JSON" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

heatmap_csv = Path(sys.argv[1])
expected_model = sys.argv[2]
expected_quantile = sys.argv[3]
out_csv = Path(sys.argv[4])
out_json = Path(sys.argv[5])

df = pd.read_csv(heatmap_csv)
required = {"quantile", "RLQR", "XGB", "TFT"}
missing = required.difference(df.columns)
if missing:
    raise SystemExit(f"Missing required columns in {heatmap_csv}: {sorted(missing)}")

long = df.melt(
    id_vars=["quantile"],
    value_vars=["RLQR", "XGB", "TFT"],
    var_name="model",
    value_name="annualized_net_profit_eur_per_year",
)
long["annualized_net_profit_eur_per_year"] = pd.to_numeric(
    long["annualized_net_profit_eur_per_year"], errors="coerce"
)
long = long.dropna(subset=["annualized_net_profit_eur_per_year"]).copy()
if long.empty:
    raise SystemExit(f"No numeric model/quantile values found in {heatmap_csv}")

long = long.sort_values(
    ["annualized_net_profit_eur_per_year", "model", "quantile"],
    ascending=[False, True, True],
).reset_index(drop=True)
best = long.iloc[0].to_dict()
best["source_csv"] = str(heatmap_csv)
best["expected_model"] = expected_model
best["expected_quantile"] = expected_quantile
best["assertion_passed"] = (
    str(best["model"]) == expected_model
    and str(best["quantile"]) == expected_quantile
)

out_csv.parent.mkdir(parents=True, exist_ok=True)
long.assign(
    source_csv=str(heatmap_csv),
    selected_as_best=False,
).to_csv(out_csv, index=False)
out_json.write_text(json.dumps(best, indent=2), encoding="utf-8")

print(
    "[RQ3_PREFLIGHT] best_model={model} best_quantile={quantile} annualized_net_profit={value:.6f}".format(
        model=best["model"],
        quantile=best["quantile"],
        value=float(best["annualized_net_profit_eur_per_year"]),
    )
)

if not best["assertion_passed"]:
    raise SystemExit(
        "Best model/quantile assertion failed: "
        f"expected {expected_model} {expected_quantile}, got {best['model']} {best['quantile']}. "
        f"See {out_json}"
    )
PY
}

run_strategy_job() {
  local strategy="$1"
  local quantile_label="${QUANTILE_PAIR%-*}"
  local run_id="${MODEL}_${strategy}_${quantile_label}"
  local out_dir="$RUN_ROOT/$run_id"
  local log_dir="$LOG_ROOT/$run_id"

  mkdir -p "$out_dir" "$log_dir"
  echo -e "${run_id}\t${strategy}\t${MODEL}\t${QUANTILE_PAIR}\t${out_dir}\t${log_dir}" >> "$MANIFEST"

  if truthy "$SKIP_COMPLETED" && ! truthy "$FORCE_RERUN" && job_complete "$out_dir" "$log_dir"; then
    local skip_utc
    skip_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[SKIP] $run_id already completed"
    mark_status "$run_id" "$strategy" "$MODEL" "$QUANTILE_PAIR" "skipped_done" "$skip_utc" "$skip_utc" "0" "$out_dir" "$log_dir"
    return 0
  fi

  rm -f "$log_dir/.running" "$log_dir/.done" "$log_dir/.failed"
  local start_utc
  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "$start_utc" > "$log_dir/.running"

  local cmd=(
    "$PYTHON_BIN" -u scripts/run_battery_backtest.py
    --model "$MODEL"
    --split "$SPLIT"
    --trading-strategy "$strategy"
    --quantile-pairs "$QUANTILE_PAIR"
    --start "$START"
    --end "$END"
    --strict-simulation-validity
    --final-soc-mode "$FINAL_SOC_MODE"
    --id-recourse-mode "$ID_RECOURSE_MODE"
    --benchmark-mode model_only
    --disable-naive
    --disable-rhpf
    --no-enable-global-perfect-foresight
    --output-detail "$OUTPUT_DETAIL"
    --debug-dumps "$DEBUG_DUMPS"
    --out-dir "$out_dir"
  )
  if truthy "$ALLOW_INVALID_OUTPUT"; then
    cmd+=(--allow-invalid-output)
  fi
  if truthy "$CLEAN_OUTPUT"; then
    cmd+=(--clean-output)
  fi

  {
    echo "RUN_ID=$run_id"
    echo "STRATEGY=$strategy"
    echo "MODEL=$MODEL"
    echo "QUANTILE_PAIR=$QUANTILE_PAIR"
    echo "START=$START"
    echo "END=$END"
    echo "OUT_DIR=$out_dir"
    echo "LOG_DIR=$log_dir"
    echo "START_UTC=$start_utc"
    printf 'COMMAND=%q ' "${cmd[@]}"
    echo
  } > "$log_dir/run.info"

  echo "[START] $run_id"
  set +e
  PYTHONPYCACHEPREFIX=/tmp/pycache "${cmd[@]}" > "$log_dir/stdout.log" 2> "$log_dir/stderr.log"
  local code=$?
  set -e

  local end_utc
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "$code" -eq 0 ]]; then
    rm -f "$log_dir/.running" "$log_dir/.failed"
    echo "$end_utc" > "$log_dir/.done"
    echo "[DONE] $run_id"
    mark_status "$run_id" "$strategy" "$MODEL" "$QUANTILE_PAIR" "done" "$start_utc" "$end_utc" "$code" "$out_dir" "$log_dir"
  else
    rm -f "$log_dir/.running" "$log_dir/.done"
    echo "$end_utc" > "$log_dir/.failed"
    echo "[FAILED] $run_id exit_code=$code"
    mark_status "$run_id" "$strategy" "$MODEL" "$QUANTILE_PAIR" "failed" "$start_utc" "$end_utc" "$code" "$out_dir" "$log_dir"
    return "$code"
  fi
}

{
  echo "RUN_ROOT=$RUN_ROOT"
  echo "LOG_ROOT=$LOG_ROOT"
  echo "START=$START"
  echo "END=$END"
  echo "SPLIT=$SPLIT"
  echo "MODEL=$MODEL"
  echo "QUANTILE_PAIR=$QUANTILE_PAIR"
  echo "EXPECTED_BEST_MODEL=$EXPECTED_BEST_MODEL"
  echo "EXPECTED_BEST_QUANTILE=$EXPECTED_BEST_QUANTILE"
  echo "RQ2_HEATMAP_CSV=$RQ2_HEATMAP_CSV"
  echo "STRATEGIES=${STRATEGIES[*]}"
  echo "MAX_PARALLEL_JOBS=$MAX_PARALLEL_JOBS"
  echo "FINAL_SOC_MODE=$FINAL_SOC_MODE"
  echo "ID_RECOURSE_MODE=$ID_RECOURSE_MODE"
  echo "ALLOW_INVALID_OUTPUT=$ALLOW_INVALID_OUTPUT"
  echo "SKIP_COMPLETED=$SKIP_COMPLETED"
  echo "FORCE_RERUN=$FORCE_RERUN"
  echo "CLEAN_OUTPUT=$CLEAN_OUTPUT"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "CREATED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$LOG_ROOT/run_meta.env"

assert_best_model_quantile

for strategy in "${STRATEGIES[@]}"; do
  wait_for_slot
  run_strategy_job "$strategy" &
done

while (( $(active_job_count) > 0 )); do
  sleep 5
done

failures=0
if [[ -f "$STATUS_CSV" ]]; then
  failures="$(awk -F, 'NR > 1 && $5 == "failed" {n++} END {print n+0}' "$STATUS_CSV")"
fi

"$PYTHON_BIN" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("\n=== RQ3 summary files ===")
for p in sorted(root.rglob("backtest_summary.json")):
    try:
        s = json.loads(p.read_text())
    except Exception as exc:
        print("BROKEN", p, exc)
        continue
    print(
        p.parent.relative_to(root),
        "strategy=", s.get("strategy"),
        "valid=", s.get("simulation_valid"),
        "reportable=", s.get("thesis_reportable"),
        "model_pnl=", s.get("realized_total_pnl_eur"),
        "invalid_reason=", s.get("invalid_reason"),
    )
PY

echo "[ALL_DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ) failures=$failures"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "STATUS_CSV=$STATUS_CSV"
echo "MANIFEST=$MANIFEST"
echo "BEST_ASSERTION_JSON=$BEST_ASSERTION_JSON"

if [[ "$failures" -gt 0 ]]; then
  echo "[ERROR] $failures RQ3 market run(s) failed. Inspect $STATUS_CSV and per-run stderr.log files." >&2
  exit 1
fi
