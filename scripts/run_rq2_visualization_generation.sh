#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-artifacts/simulation_runs/thesis_final_multi_2m_20260620T091938Z}"
OUT_ROOT="${OUT_ROOT:-artifacts/benchmark/rq2_simulation_benchmark}"
FORECAST_BENCHMARK_DIR="${FORECAST_BENCHMARK_DIR:-artifacts/benchmark/rq1_ml_model_benchmark}"
PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
EXPORT_ROOT="${EXPORT_ROOT:-/Users/leori/Desktop/ uni/3 Master IS/25 MA/MA___Elevated_Energy_Trading__Forecasting_Expected_Profit_From_Participating_in_the_Day_Ahead_and_Balancing_Markets/figures/4-results}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
FORMATS="${FORMATS:-png}"

"${PYTHON_BIN}" scripts/build_rq2_simulation_visualizations.py \
  --run-root "${RUN_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --forecast-benchmark-dir "${FORECAST_BENCHMARK_DIR}" \
  --split test \
  --annualize \
  --strict-validity \
  --formats "${FORMATS}" \
  --overwrite

"${PYTHON_BIN}" scripts/verify_rq2_output_structure.py \
  --out-root "${OUT_ROOT}"

if [[ "${SKIP_EXPORT}" != "1" ]]; then
  EXPORT_DEST="${EXPORT_ROOT}/$(basename "${OUT_ROOT}")"
  rm -rf "${EXPORT_DEST}"
  mkdir -p "$(dirname "${EXPORT_DEST}")"
  cp -R "${OUT_ROOT}" "${EXPORT_DEST}"
  echo "[OK] Exported RQ2 benchmark: ${EXPORT_DEST}"
fi
