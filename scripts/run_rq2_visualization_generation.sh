#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-artifacts/simulation_runs/thesis_final_multi_2m_20260620T091938Z}"
OUT_ROOT="${OUT_ROOT:-artifacts/benchmark/rq2_simulation_benchmark}"
FORECAST_BENCHMARK_DIR="${FORECAST_BENCHMARK_DIR:-artifacts/benchmark/rq1_ml_model_benchmark}"
PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"

"${PYTHON_BIN}" scripts/build_rq2_simulation_visualizations.py \
  --run-root "${RUN_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --forecast-benchmark-dir "${FORECAST_BENCHMARK_DIR}" \
  --split test \
  --annualize \
  --strict-validity \
  --overwrite

"${PYTHON_BIN}" scripts/verify_rq2_output_structure.py \
  --out-root "${OUT_ROOT}"
