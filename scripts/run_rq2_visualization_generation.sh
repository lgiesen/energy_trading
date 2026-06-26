#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-artifacts/simulation_runs/thesis_final_multi_2m_20260624T141002Z}"
OUT_ROOT="${OUT_ROOT:-artifacts/benchmark/rq2_simulation_benchmark}"
FORECAST_BENCHMARK_DIR="${FORECAST_BENCHMARK_DIR:-artifacts/benchmark/rq1_ml_model_benchmark}"
PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
EXPORT_ROOT="${EXPORT_ROOT:-/Users/leori/Desktop/ uni/3 Master IS/25 MA/MA___Elevated_Energy_Trading__Forecasting_Expected_Profit_From_Participating_in_the_Day_Ahead_and_Balancing_Markets/figures/4-results}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
FORMATS="${FORMATS:-png}"
RUN_REGRET_DRIVERS="${RUN_REGRET_DRIVERS:-1}"
REGRET_BENCHMARK="${REGRET_BENCHMARK:-rhpf}"
REGRET_MODELS="${REGRET_MODELS:-xgb,tft,linear}"
REGRET_QUANTILES="${REGRET_QUANTILES:-p10,p30,p50,p70,p90}"
REGRET_STRICT="${REGRET_STRICT:-1}"
RUN_INVALIDITY_SEVERITY="${RUN_INVALIDITY_SEVERITY:-1}"
RUN_RISK_DIAGNOSTICS="${RUN_RISK_DIAGNOSTICS:-1}"
RUN_RELATIVE_VALUE_OF_FORECAST="${RUN_RELATIVE_VALUE_OF_FORECAST:-1}"

"${PYTHON_BIN}" scripts/build_rq2_simulation_visualizations.py \
  --run-root "${RUN_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --forecast-benchmark-dir "${FORECAST_BENCHMARK_DIR}" \
  --split test \
  --annualize \
  --strict-validity \
  --formats "${FORMATS}" \
  --overwrite

if [[ "${RUN_REGRET_DRIVERS}" != "0" ]]; then
  REGRET_CMD=(
    "${PYTHON_BIN}" scripts/analyze_regret_drivers.py
    --run-dir "${RUN_ROOT}"
    --out-dir "${OUT_ROOT}/regret_drivers"
    --benchmark "${REGRET_BENCHMARK}"
    --models "${REGRET_MODELS}"
    --quantiles "${REGRET_QUANTILES}"
  )
  if [[ "${REGRET_STRICT}" == "1" ]]; then
    REGRET_CMD+=(--strict)
  fi
  "${REGRET_CMD[@]}"
fi

if [[ "${RUN_INVALIDITY_SEVERITY}" != "0" ]]; then
  "${PYTHON_BIN}" scripts/build_simulation_invalidity_severity.py \
    --run-root "${RUN_ROOT}" \
    --out-root "${OUT_ROOT}" \
    --label rq2
fi

if [[ "${RUN_RISK_DIAGNOSTICS}" != "0" ]]; then
  "${PYTHON_BIN}" scripts/generate_rq2_risk_diagnostics.py \
    --rq2-root "${OUT_ROOT}" \
    --run-root "${RUN_ROOT}"
fi

if [[ "${RUN_RELATIVE_VALUE_OF_FORECAST}" != "0" ]]; then
  "${PYTHON_BIN}" scripts/generate_rq2_relative_value_of_forecast.py \
    --rq2-root "${OUT_ROOT}"
fi

VERIFY_CMD=(
  "${PYTHON_BIN}" scripts/verify_rq2_output_structure.py
  --out-root "${OUT_ROOT}"
)
if [[ "${RUN_INVALIDITY_SEVERITY}" != "0" ]]; then
  VERIFY_CMD+=(--require-invalidity-severity)
fi
"${VERIFY_CMD[@]}"

if [[ "${SKIP_EXPORT}" != "1" ]]; then
  EXPORT_DEST="${EXPORT_ROOT}/$(basename "${OUT_ROOT}")"
  rm -rf "${EXPORT_DEST}"
  mkdir -p "$(dirname "${EXPORT_DEST}")"
  cp -R "${OUT_ROOT}" "${EXPORT_DEST}"
  echo "[OK] Exported RQ2 benchmark: ${EXPORT_DEST}"
fi
