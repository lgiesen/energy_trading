#!/usr/bin/env bash
set -Eeuo pipefail

# Final thesis multi-market simulation matrix for the server.
#
# Window:
#   Default: 2025-06-01 00:00 Europe/Berlin inclusive -> 2025-08-01 00:00 Europe/Berlin exclusive
#   UTC:     2025-05-31T22:00:00Z inclusive -> 2025-07-31T22:00:00Z exclusive
#   Override with START=... END=...
#
# Matrix:
#   models:    xgb, tft, linear by default. Override with MODELS="xgb tft linear".
#   quantiles: p30, p50, p70 first, then p90, p10
#   strategy:  multi
#
# Parallelism:
#   Runs up to MAX_PARALLEL_JOBS simulations in parallel.
#   Default: 4 concurrent jobs.
#
# Output policy:
#   Every run gets a dedicated folder. Do not point multiple runs at the same
#   out-dir because aggregate files inside an out-dir are overwritten by design.
#   Resume is launcher-level: completed jobs are skipped, interrupted jobs are
#   rerun from scratch.
#
# Server usage:
#   chmod +x rq2_run_final_thesis_multi_2m.sh
#   nohup ./rq2_run_final_thesis_multi_2m.sh \
#     > artifacts/simulation_runs/final_thesis_multi_2m_launcher.out 2>&1 &
#   tail -f artifacts/simulation_runs/final_thesis_multi_2m_launcher.out
#
# Resume a specific existing launcher root:
#   RUN_ROOT=artifacts/simulation_runs/thesis_final_multi_2m_YYYYmmddTHHMMSSZ \
#   ./rq2_run_final_thesis_multi_2m.sh
#
# Resume controls:
#   SKIP_COMPLETED=1 skips jobs with .done markers and complete aggregate files.
#   FORCE_RERUN=1 ignores .done markers.
#   CLEAN_OUTPUT=1 passes --clean-output to workers; keep it 0 for crash resume.
#
# Check simulation worker process
# pgrep -af "run_battery_backtest.py"
#
# Current logs
# RUN_ROOT="artifacts/simulation_runs/thesis_final_multi_2m_YYYYmmddTHHMMSSZ"
# tail -f "$RUN_ROOT/logs/MODEL_QUANTILE/stdout.log"

MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-4}"

START="${START:-2025-05-31T22:00:00Z}"
END="${END:-2025-07-31T22:00:00Z}"
SPLIT="${SPLIT:-test}"
STRATEGY="${STRATEGY:-multi}"
FINAL_SOC_MODE="${FINAL_SOC_MODE:-hard_min}"
ID_RECOURSE_MODE="${ID_RECOURSE_MODE:-common}"
OUTPUT_DETAIL="${OUTPUT_DETAIL:-thesis}"
DEBUG_DUMPS="${DEBUG_DUMPS:-accepted_only}"
ALLOW_INVALID_OUTPUT="${ALLOW_INVALID_OUTPUT:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"

read -r -a MODELS <<< "${MODELS:-xgb tft linear}"
read -r -a PRIMARY_QUANTILES <<< "${PRIMARY_QUANTILES:-p30-p30 p50-p50 p70-p70}"
read -r -a SECONDARY_QUANTILES <<< "${SECONDARY_QUANTILES:-p90-p90 p10-p10}"
QUANTILES=("${PRIMARY_QUANTILES[@]}" "${SECONDARY_QUANTILES[@]}")

EXTRA_COMMON_ARGS=()
if [[ "$ALLOW_INVALID_OUTPUT" == "1" || "$ALLOW_INVALID_OUTPUT" == "true" || "$ALLOW_INVALID_OUTPUT" == "yes" ]]; then
  EXTRA_COMMON_ARGS+=(--allow-invalid-output)
fi

RUN_TS="${RUN_TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-artifacts/simulation_runs/thesis_final_multi_2m_${RUN_TS}}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/logs}"

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
  echo "FINAL_SOC_MODE=$FINAL_SOC_MODE"
  echo "ID_RECOURSE_MODE=$ID_RECOURSE_MODE"
  echo "OUTPUT_DETAIL=$OUTPUT_DETAIL"
  echo "DEBUG_DUMPS=$DEBUG_DUMPS"
  echo "ALLOW_INVALID_OUTPUT=$ALLOW_INVALID_OUTPUT"
  echo "SKIP_COMPLETED=$SKIP_COMPLETED"
  echo "FORCE_RERUN=$FORCE_RERUN"
  echo "CLEAN_OUTPUT=$CLEAN_OUTPUT"
  echo "MAX_PARALLEL_JOBS=$MAX_PARALLEL_JOBS"
  echo "MODELS=${MODELS[*]}"
  echo "PRIMARY_QUANTILES=${PRIMARY_QUANTILES[*]}"
  echo "SECONDARY_QUANTILES=${SECONDARY_QUANTILES[*]}"
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

active_job_count() {
  jobs -rp | wc -l | tr -d ' '
}

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

job_complete() {
  local out_dir="$1"
  local log_dir="$2"
  local done_marker="$log_dir/.done"

  [[ -f "$done_marker" ]] || return 1
  [[ -s "$out_dir/strategy_overview.csv" ]] || return 1
  [[ -s "$out_dir/quantile_sweep_summary.csv" ]] || return 1
  return 0
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

  local running_marker="$log_dir/.running"
  local done_marker="$log_dir/.done"
  local failed_marker="$log_dir/.failed"

  if truthy "$SKIP_COMPLETED" && ! truthy "$FORCE_RERUN" && job_complete "$out_dir" "$log_dir"; then
    local skip_utc
    skip_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[SKIP] $run_id already completed: $done_marker"
    mark_status "$run_id" "$chain" "$model" "$quantile" "$benchmark_mode" "skipped_done" "$skip_utc" "$skip_utc" "0" "$out_dir" "$log_dir"
    return 0
  fi

  rm -f "$running_marker" "$done_marker" "$failed_marker"

  local start_utc
  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "$start_utc" > "$running_marker"

  local cmd=(
    "$PYTHON_BIN" -u scripts/run_battery_backtest.py
    --model "$model"
    --split "$SPLIT"
    --trading-strategy "$STRATEGY"
    --quantile-pairs "$quantile"
    --start "$START"
    --end "$END"
    --strict-simulation-validity
    --final-soc-mode "$FINAL_SOC_MODE"
    --id-recourse-mode "$ID_RECOURSE_MODE"
    --benchmark-mode "$benchmark_mode"
    --no-enable-global-perfect-foresight
    --output-detail "$OUTPUT_DETAIL"
    --debug-dumps "$DEBUG_DUMPS"
    --out-dir "$out_dir"
  )
  if truthy "$CLEAN_OUTPUT"; then
    cmd+=(--clean-output)
  fi
  if (( ${#EXTRA_COMMON_ARGS[@]} > 0 )); then
    cmd+=("${EXTRA_COMMON_ARGS[@]}")
  fi
  if (( ${#extra_args[@]} > 0 )); then
    cmd+=("${extra_args[@]}")
  fi

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
    printf 'COMMAND=%q ' "${cmd[@]}"
    echo
  } > "$log_dir/run.info"

  echo "[START] $run_id at $start_utc"

  set +e
  PYTHONPYCACHEPREFIX=/tmp/pycache "${cmd[@]}" \
    > "$log_dir/stdout.log" \
    2> "$log_dir/stderr.log"
  local code=$?
  set -e

  local end_utc
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  if [[ "$code" -eq 0 ]]; then
    rm -f "$running_marker" "$failed_marker"
    echo "$end_utc" > "$done_marker"
    echo "[DONE] $run_id at $end_utc"
    mark_status "$run_id" "$chain" "$model" "$quantile" "$benchmark_mode" "done" "$start_utc" "$end_utc" "$code" "$out_dir" "$log_dir"
  else
    rm -f "$running_marker" "$done_marker"
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

run_model_job() {
  local model="$1"
  local quantile_pair="$2"
  local chain="model_${model}"
  local quantile_label="${quantile_pair%-*}"
  local run_id="${model}_${quantile_label}"
  local out_dir="$RUN_ROOT/$run_id"
  local log_dir="$LOG_ROOT/$run_id"

  register_manifest_row "$run_id" "$chain" "$model" "$quantile_pair" "model_only" "$out_dir" "$log_dir"
  run_job "$run_id" "$chain" "$model" "$quantile_pair" "model_only" "$out_dir" "$log_dir" \
    --disable-naive \
    --disable-rhpf
}

wait_for_slot() {
  while (( $(active_job_count) >= MAX_PARALLEL_JOBS )); do
    sleep 5
  done
}

run_model_wave() {
  local quantiles=("$@")
  for quantile_pair in "${quantiles[@]}"; do
    for model in "${MODELS[@]}"; do
      wait_for_slot
      run_model_job "$model" "$quantile_pair" &
    done
  done
}

failures=0

wait_for_slot
run_benchmark_chain &

# Start the thesis priority quantiles first, then queue p90/p10. Jobs are
# independent, so the scheduler keeps up to MAX_PARALLEL_JOBS simulations busy.
run_model_wave "${PRIMARY_QUANTILES[@]}"
run_model_wave "${SECONDARY_QUANTILES[@]}"

while (( $(active_job_count) > 0 )); do
  sleep 5
done

if [[ -f "$STATUS_CSV" ]]; then
  failures="$(awk -F, 'NR > 1 && $6 == "failed" {n++} END {print n+0}' "$STATUS_CSV")"
fi

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
