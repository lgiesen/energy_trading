#!/usr/bin/env bash
set -Eeuo pipefail

# Final thesis multi-market simulation matrix for the server.
#
# Window:
#   2025-03-01T00:00:00Z inclusive -> 2025-06-01T00:00:00Z exclusive
#
# Matrix:
#   models:    xgb, tft, linear
#   quantiles: p30, p50, p70 first, then p90, p10
#   strategy:  multi
#
# Parallelism:
#   Runs 4 chains in parallel:
#     1 benchmark chain: naive once, RHPF once, never GHPF
#     3 model chains: one sequential quantile chain per model
#
# Output policy:
#   Every run gets a dedicated folder. Do not point multiple runs at the same
#   out-dir because aggregate files inside an out-dir are overwritten by design.
#
# Server usage:
#   chmod +x run_final_thesis_multi_3m.sh
#   nohup ./run_final_thesis_multi_3m.sh > logs/final_thesis_multi_3m_launcher.out 2>&1 &
#   tail -f logs/final_thesis_multi_3m_launcher.out
#
# Resume a specific existing launcher root:
#   RUN_ROOT=artifacts/simulation_runs/thesis_final_multi_3m_YYYYmmddTHHMMSSZ \
#   LOG_ROOT=logs/thesis_final_multi_3m_YYYYmmddTHHMMSSZ \
#   ./run_final_thesis_multi_3m.sh

MAX_PARALLEL_CHAINS=4

START="2025-03-01T00:00:00Z"
END="2025-06-01T00:00:00Z"
SPLIT="test"
STRATEGY="multi"

MODELS=("xgb" "tft" "linear")
QUANTILES=("p30-p30" "p50-p50" "p70-p70" "p90-p90" "p10-p10")

RUN_TS="${RUN_TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-artifacts/simulation_runs/thesis_final_multi_3m_${RUN_TS}}"
LOG_ROOT="${LOG_ROOT:-logs/thesis_final_multi_3m_${RUN_TS}}"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

STATUS_CSV="$LOG_ROOT/status.csv"
MANIFEST="$LOG_ROOT/manifest.tsv"

if [[ ! -f "$STATUS_CSV" ]]; then
  echo "run_id,chain,model,quantile,benchmark_mode,status,start_utc,end_utc,exit_code,out_dir,log_dir" > "$STATUS_CSV"
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo -e "run_id\tchain\tmodel\tquantile\tbenchmark_mode\tout_dir\tlog_dir" > "$MANIFEST"
fi

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
HOST="$(hostname)"

{
  echo "RUN_TS=$RUN_TS"
  echo "RUN_ROOT=$RUN_ROOT"
  echo "LOG_ROOT=$LOG_ROOT"
  echo "START=$START"
  echo "END=$END"
  echo "SPLIT=$SPLIT"
  echo "STRATEGY=$STRATEGY"
  echo "MAX_PARALLEL_CHAINS=$MAX_PARALLEL_CHAINS"
  echo "MODELS=${MODELS[*]}"
  echo "QUANTILES=${QUANTILES[*]}"
  echo "GIT_COMMIT=$GIT_COMMIT"
  echo "GIT_BRANCH=$GIT_BRANCH"
  echo "PYTHON=$PYTHON_BIN"
  echo "HOST=$HOST"
  echo "PWD=$(pwd)"
  echo "CREATED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$LOG_ROOT/run_meta.env"

git status --short > "$LOG_ROOT/git_status_short.txt" 2>/dev/null || true
"$PYTHON_BIN" --version > "$LOG_ROOT/python_version.txt" 2>&1 || true
"$PYTHON_BIN" -m pip freeze > "$LOG_ROOT/pip_freeze.txt" 2>&1 || true

mark_status() {
  local run_id="$1"
  local chain="$2"
  local model="$3"
  local quantile="$4"
  local benchmark_mode="$5"
  local status="$6"
  local start_utc="$7"
  local end_utc="$8"
  local exit_code="$9"
  local out_dir="${10}"
  local log_dir="${11}"

  echo "${run_id},${chain},${model},${quantile},${benchmark_mode},${status},${start_utc},${end_utc},${exit_code},${out_dir},${log_dir}" >> "$STATUS_CSV"
}

run_job() {
  local run_id="$1"
  local chain="$2"
  local model="$3"
  local quantile="$4"
  local benchmark_mode="$5"
  local out_dir="$6"
  local log_dir="$7"
  shift 7
  local extra_args=("$@")

  mkdir -p "$out_dir" "$log_dir"

  local done_marker="$out_dir/.done"
  local failed_marker="$out_dir/.failed"

  if [[ -f "$done_marker" ]]; then
    echo "[SKIP] $run_id already completed: $done_marker"
    return 0
  fi

  rm -f "$failed_marker"

  local start_utc
  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  {
    echo "RUN_ID=$run_id"
    echo "CHAIN=$chain"
    echo "MODEL=$model"
    echo "QUANTILE=$quantile"
    echo "BENCHMARK_MODE=$benchmark_mode"
    echo "START=$START"
    echo "END=$END"
    echo "OUT_DIR=$out_dir"
    echo "LOG_DIR=$log_dir"
    echo "GIT_COMMIT=$GIT_COMMIT"
    echo "GIT_BRANCH=$GIT_BRANCH"
    echo "PYTHON=$PYTHON_BIN"
    echo "HOST=$HOST"
    echo "PWD=$(pwd)"
    echo "START_UTC=$start_utc"
    printf 'COMMAND=%q ' "$PYTHON_BIN" -u scripts/run_battery_backtest.py \
      --model "$model" \
      --split "$SPLIT" \
      --trading-strategy "$STRATEGY" \
      --quantile-pairs "$quantile" \
      --start "$START" \
      --end "$END" \
      --strict-simulation-validity \
      --final-soc-mode hard \
      --id-recourse-mode common \
      --benchmark-mode "$benchmark_mode" \
      --disable-ghpf \
      --no-enable-global-perfect-foresight \
      --output-detail thesis \
      --debug-dumps accepted_only \
      --clean-output \
      --out-dir "$out_dir" \
      "${extra_args[@]}"
    echo
  } > "$log_dir/run.info"

  echo "[START] $run_id at $start_utc"

  set +e
  PYTHONPYCACHEPREFIX=/tmp/pycache "$PYTHON_BIN" -u scripts/run_battery_backtest.py \
    --model "$model" \
    --split "$SPLIT" \
    --trading-strategy "$STRATEGY" \
    --quantile-pairs "$quantile" \
    --start "$START" \
    --end "$END" \
    --strict-simulation-validity \
    --final-soc-mode hard \
    --id-recourse-mode common \
    --benchmark-mode "$benchmark_mode" \
    --disable-ghpf \
    --no-enable-global-perfect-foresight \
    --output-detail thesis \
    --debug-dumps accepted_only \
    --clean-output \
    --out-dir "$out_dir" \
    "${extra_args[@]}" \
    > "$log_dir/stdout.log" \
    2> "$log_dir/stderr.log"
  local code=$?
  set -e

  local end_utc
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  if [[ "$code" -eq 0 ]]; then
    echo "$end_utc" > "$done_marker"
    echo "[DONE] $run_id at $end_utc"
    mark_status "$run_id" "$chain" "$model" "$quantile" "$benchmark_mode" "done" "$start_utc" "$end_utc" "$code" "$out_dir" "$log_dir"
  else
    echo "$end_utc" > "$failed_marker"
    echo "[FAILED] $run_id at $end_utc exit_code=$code"
    mark_status "$run_id" "$chain" "$model" "$quantile" "$benchmark_mode" "failed" "$start_utc" "$end_utc" "$code" "$out_dir" "$log_dir"
    return "$code"
  fi
}

register_manifest_row() {
  local run_id="$1"
  local chain="$2"
  local model="$3"
  local quantile="$4"
  local benchmark_mode="$5"
  local out_dir="$6"
  local log_dir="$7"
  echo -e "${run_id}\t${chain}\t${model}\t${quantile}\t${benchmark_mode}\t${out_dir}\t${log_dir}" >> "$MANIFEST"
}

run_benchmark_chain() {
  local chain="benchmarks"

  # Benchmark carrier is linear+p50. Naive and RHPF should be model-independent
  # for the same strategy/window, and GHPF is explicitly disabled.
  local model="linear"
  local quantile="p50-p50"

  local run_id="benchmarks_naive"
  local out_dir="$RUN_ROOT/$run_id"
  local log_dir="$LOG_ROOT/$run_id"
  register_manifest_row "$run_id" "$chain" "$model" "$quantile" "naive_only" "$out_dir" "$log_dir"
  run_job "$run_id" "$chain" "$model" "$quantile" "naive_only" "$out_dir" "$log_dir"

  run_id="benchmarks_rhpf"
  out_dir="$RUN_ROOT/$run_id"
  log_dir="$LOG_ROOT/$run_id"
  register_manifest_row "$run_id" "$chain" "$model" "$quantile" "rhpf_only" "$out_dir" "$log_dir"
  run_job "$run_id" "$chain" "$model" "$quantile" "rhpf_only" "$out_dir" "$log_dir" \
    --disable-naive \
    --enable-rhpf
}

run_model_chain() {
  local model="$1"
  local chain="model_${model}"

  for quantile_pair in "${QUANTILES[@]}"; do
    local quantile_label="${quantile_pair%-*}"
    local run_id="${model}_${quantile_label}"
    local out_dir="$RUN_ROOT/$run_id"
    local log_dir="$LOG_ROOT/$run_id"

    register_manifest_row "$run_id" "$chain" "$model" "$quantile_pair" "model_only" "$out_dir" "$log_dir"
    run_job "$run_id" "$chain" "$model" "$quantile_pair" "model_only" "$out_dir" "$log_dir" \
      --disable-naive \
      --disable-rhpf
  done
}

wait_for_slot() {
  while (( $(jobs -rp | wc -l | tr -d ' ') >= MAX_PARALLEL_CHAINS )); do
    wait -n || true
  done
}

pids=()

wait_for_slot
run_benchmark_chain &
pids+=("$!")

for model in "${MODELS[@]}"; do
  wait_for_slot
  run_model_chain "$model" &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

echo "[ALL_DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ) failures=$failures"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "STATUS_CSV=$STATUS_CSV"
echo "MANIFEST=$MANIFEST"

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

root = Path("$RUN_ROOT")
print("\n=== Summary files ===")
for p in sorted(root.rglob("backtest_summary.json")):
    try:
        s = json.loads(p.read_text())
    except Exception as e:
        print("BROKEN", p, e)
        continue
    print(
        p.parent.relative_to(root),
        "strategy=", s.get("strategy"),
        "valid=", s.get("simulation_valid"),
        "reportable=", s.get("thesis_reportable"),
        "model_pnl=", s.get("realized_total_pnl_eur"),
        "naive_pnl=", s.get("naive_total_pnl_eur"),
        "rhpf_pnl=", s.get("rolling_perfect_foresight_same_rules_total_pnl_eur"),
        "ghpf_available=", s.get("global_pf_available"),
        "invalid_reason=", s.get("invalid_reason"),
    )
PY

if [[ "$failures" -gt 0 ]]; then
  echo "[ERROR] $failures chain(s) failed. Inspect $STATUS_CSV and per-run stderr.log files." >&2
  exit 1
fi
