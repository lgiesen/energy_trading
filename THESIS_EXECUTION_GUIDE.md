# THESIS_EXECUTION_GUIDE

## 0. Environment Setup

### 0.1 Prerequisites

This project was tested with:

- macOS / Linux shell environment
- Python 3.10
- Git
- Make
- pip
- A POSIX-compatible shell

### 0.2 Clone Repository

```bash
git clone https://github.com/lgiesen/energy_trading.git
cd energy_trading
```

### 0.3 Create Virtual Environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Check that the virtual environment is active:

```bash
which python
python --version
```

Expected:

```text
.../energy_trading/.venv/bin/python
Python 3.10.20
```

### 0.4 Install Dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

The editable install is required because the repository uses a local package structure. It ensures that imports from the project source tree work consistently from scripts, tests, and Makefile targets.

### 0.5 Verify Installation

```bash
python -m pip check
make doctor
```

If these commands pass, the environment has the required Python dependencies and can see the model input data.

### 0.6 Pipeline Overview & Preflight

This repository uses a DAG-based Makefile architecture (action-model pattern) to guarantee caching, state-awareness, and deterministic rebuilds. Each stage is dependency-aware, so unchanged upstream artifacts are not recomputed, while changed inputs correctly trigger downstream rebuilds.

Run preflight checks before any execution:

```bash
make doctor
```

`make doctor` verifies:

- Required Python dependencies are importable.
- Raw/model input data is present in `data/model_input`.

Main artifact locations:

```text
artifacts/model_runs/
artifacts/benchmark/
artifacts/simulation_runs/
```

## 2. Supervisor Smoke Execution

This section is organized by research question. Each subsection contains:

- A smoke run: short, quick execution to verify that the code path runs.
- A complete run: the thesis run configuration used for the final results.

Run all commands from the repository root with the virtual environment activated.

### 2.1 RQ1: Probabilistic Forecasting Benchmark

RQ1 is the ML forecasting benchmark. It trains and evaluates the probabilistic forecast model families:

- XGBoost
- Linear / regularized linear quantile regression
- TFT

#### 2.1.1 Smoke Run

Use the smoke run to verify the full ML model DAG without launching the full training workload:

```bash
make doctor
make smoke-test DEVICE=cpu
```

Use CUDA instead if the machine has a working NVIDIA GPU:

```bash
make doctor
make smoke-test DEVICE=cuda
```

This smoke target runs the reduced DAG for all three model families:

```text
Tune -> Train -> Simulate -> Audit/package
```

The smoke DAG uses reduced runtime settings, including:

```text
IS_SMOKE_TEST=1
FORECAST_HOURS=24
SIM_HORIZON_HOURS=24
```

After the smoke model artifacts exist, verify the RQ1 benchmark entry point:

```bash
./.venv/bin/python scripts/run_forecast_benchmark.py \
  --out-dir artifacts/benchmark/rq1_ml_model_benchmark_smoke \
  --splits test \
  --no-make-figures \
  --save-joined-predictions
```

Expected smoke outputs:

```text
artifacts/model_runs/latest_xgboost.json
artifacts/model_runs/latest_linear.json
artifacts/model_runs/latest_tft.json
artifacts/benchmark/rq1_ml_model_benchmark_smoke/
```

#### 2.1.2 Complete Run

Run the complete ML model training pipeline:

```bash
make doctor
make all-xgb
make all-linear
make all-tft DEVICE=cuda
```

If CUDA is unavailable, XGBoost and Linear can still be trained on CPU, but full TFT training should be run on a CUDA-capable machine for the final thesis results.

The complete model-family DAG is:

```text
Tune -> Train -> Simulate -> Audit/package
```

Successful full training creates run folders and deliverables under:

```text
artifacts/model_runs/
artifacts/model_runs/<run_id>/manifest.json
artifacts/model_runs/<run_id>_deliverable.zip
artifacts/model_runs/latest_xgboost.json
artifacts/model_runs/latest_linear.json
artifacts/model_runs/latest_tft.json
```

After full training, run the complete RQ1 benchmark:

```bash
./.venv/bin/python scripts/run_rq1_analysis.py \
  --out-dir artifacts/benchmark/rq1_ml_model_benchmark \
  --skip-export
```

Expected complete RQ1 benchmark output:

```text
artifacts/benchmark/rq1_ml_model_benchmark/
```

### 2.2 RQ2: Energy Trading Simulation Backtest

RQ2 is the energy trading simulation backtest for all thesis model families and quantile policies using the multi-market strategy.

It covers:

- Models: `xgb`, `tft`, `linear`
- Strategy: `multi`
- Quantile policies: `p30-p30`, `p50-p50`, `p70-p70`, `p90-p90`, `p10-p10`

#### 2.2.1 Smoke Run

Use this short run to verify the RQ2 launcher and simulation code path without running the full June/July thesis window:

```bash
chmod +x rq2_run_final_thesis_multi_2m.sh
START=2025-06-01T00:00:00Z \
END=2025-06-02T00:00:00Z \
MODELS="xgb tft linear" \
MAX_PARALLEL_JOBS=2 \
RUN_TS=smoke_rq2_multi_24h \
./rq2_run_final_thesis_multi_2m.sh
```

This smoke run uses a 24-hour UTC window and writes to:

```text
artifacts/simulation_runs/thesis_final_multi_2m_smoke_rq2_multi_24h/
```

#### 2.2.2 Complete Run

Run the complete RQ2 thesis simulation for June and July:

```bash
chmod +x rq2_run_final_thesis_multi_2m.sh
nohup ./rq2_run_final_thesis_multi_2m.sh \
  > artifacts/simulation_runs/rq2_final_thesis_multi_2m_launcher.out 2>&1 &
```

Default complete RQ2 window:

```text
2025-05-31T22:00:00Z inclusive -> 2025-07-31T22:00:00Z exclusive
```

This corresponds to:

```text
2025-06-01 00:00 Europe/Berlin -> 2025-08-01 00:00 Europe/Berlin
```

Monitor the launcher:

```bash
tail -f artifacts/simulation_runs/rq2_final_thesis_multi_2m_launcher.out
```

The complete RQ2 launcher writes a timestamped run root:

```text
artifacts/simulation_runs/thesis_final_multi_2m_<UTC_TIMESTAMP>/
```

### 2.3 RQ3: Single and Multi-Market Energy Trading Strategy

RQ3 compares one selected model-quantile combination across single-market and multi-market strategies.

The thesis configuration uses one model-quantile policy and these strategy command values:

- `multi`: multi-market strategy
- `da`: day-ahead-only strategy
- `bcm`: balancing capacity market only
- `bem`: balancing energy market only

The RQ3 thesis comparison excludes `afrr`.

#### 2.3.1 Smoke Run

Use this short run to verify the RQ3 market-strategy launcher:

```bash
chmod +x scripts/run_rq3_xgb_p50_market_benchmark.sh
START=2025-06-01T00:00:00Z \
END=2025-06-02T00:00:00Z \
MODEL=xgb \
QUANTILE_PAIR=p50-p50 \
EXPECTED_BEST_MODEL=XGB \
EXPECTED_BEST_QUANTILE=p50 \
STRATEGIES="multi da bcm bem" \
MAX_PARALLEL_JOBS=2 \
RUN_TS=smoke_rq3_xgb_p50_24h \
./scripts/run_rq3_xgb_p50_market_benchmark.sh
```

This smoke run writes to:

```text
artifacts/simulation_runs/rq3_xgb_p50_market_benchmark_smoke_rq3_xgb_p50_24h/
```

If the RQ3 preflight assertion should use a specific RQ2 run root, add:

```bash
RQ2_SIM_RUN_ROOT=artifacts/simulation_runs/thesis_final_multi_2m_<UTC_TIMESTAMP>
```

#### 2.3.2 Complete Run

Run the complete RQ3 market-strategy comparison over the June/July thesis window:

```bash
chmod +x scripts/run_rq3_xgb_p50_market_benchmark.sh
STRATEGIES="multi da bcm bem" \
nohup ./scripts/run_rq3_xgb_p50_market_benchmark.sh \
  > artifacts/simulation_runs/rq3_xgb_p50_market_benchmark_launcher.out 2>&1 &
```

Default complete RQ3 window:

```text
2025-05-31T22:00:00Z inclusive -> 2025-07-31T22:00:00Z exclusive
```

This corresponds to:

```text
2025-06-01 00:00 Europe/Berlin -> 2025-08-01 00:00 Europe/Berlin
```

Default model-quantile configuration:

```text
MODEL=xgb
QUANTILE_PAIR=p50-p50
EXPECTED_BEST_MODEL=XGB
EXPECTED_BEST_QUANTILE=p50
STRATEGIES=multi da bcm bem
```

To run a different selected model-quantile combination, override the model, quantile, and expected RQ2 assertion:

```bash
MODEL=tft \
QUANTILE_PAIR=p70-p70 \
EXPECTED_BEST_MODEL=TFT \
EXPECTED_BEST_QUANTILE=p70 \
STRATEGIES="multi da bcm bem" \
./scripts/run_rq3_xgb_p50_market_benchmark.sh
```

Monitor the launcher:

```bash
tail -f artifacts/simulation_runs/rq3_xgb_p50_market_benchmark_launcher.out
```

The complete RQ3 launcher writes a timestamped run root:

```text
artifacts/simulation_runs/rq3_xgb_p50_market_benchmark_<UTC_TIMESTAMP>/
```

## 3. Audit & Output Artifacts

For every successful model run, the pipeline automatically produces an immutable deliverable archive:

```text
artifacts/model_runs/<run_id>_deliverable.zip
```

Each deliverable contains, at minimum:

- Trained run `manifest.json`.
- Metrics and evaluation outputs.
- Simulation ledgers and summaries.
- Provenance metadata, including commit, dependency, runtime, and data-hash lineage.

Simulation launchers write their own run metadata, status files, logs, and manifests under:

```text
artifacts/simulation_runs/<run_root>/logs/
```

The most important simulation monitoring files are:

```text
artifacts/simulation_runs/<run_root>/logs/run_meta.env
artifacts/simulation_runs/<run_root>/logs/status.csv
artifacts/simulation_runs/<run_root>/logs/manifest.tsv
```
