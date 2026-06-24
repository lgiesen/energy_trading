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
cd energyTrading
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
.../energyTrading/.venv/bin/python
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

If these commands pass, the environment is ready for full training and simulation runs.

## 1. Pipeline Overview & Preflight

This repository uses a DAG-based Makefile architecture (action-model pattern) to guarantee caching, state-awareness, and strict determinism.  
Each stage is dependency-aware, so unchanged upstream artifacts are not recomputed, while changed inputs correctly trigger downstream rebuilds.

Run preflight checks before any execution:

```bash
make doctor
```

`make doctor` verifies:

- Required Python dependencies are importable.
- Raw/model input data is present in `data/model_input`.

## 2. The Smoke Test (Fast-Fail Validation)

Run the smoke test:

```bash
make smoke-test
```

This executes the full end-to-end pipeline for all three models:

- Tune -> Train -> Simulate -> Audit for XGBoost and Linear.
- Train -> Simulate -> Audit for TFT.

During smoke runs, `IS_SMOKE_TEST=1` is injected and read by Python via `os.environ`, enabling aggressive reductions (e.g., trials/epochs/data rows) to validate codebase integrity in under ~5 minutes without running full multi-hour training loops.

## 3. Step 1: Train The Forecast Models

Train the three model families before running final thesis simulations:

```bash
make all-xgb
make all-linear
make all-tft
```

These commands execute the full reproducible DAG for each model family and enforce:

- Fixed random seed behavior.
- Strictly single-threaded BLAS/OMP settings (from Makefile exports) to maximize exact mathematical reproducibility.

### HPO Artifact Consumption (Implementation Note)

The pipeline uses direct artifact consumption for tuned hyperparameters:

- `make all-xgb` passes `artifacts/hpo/xgb_optuna_da_target_da_price.json` to training.
- `make all-linear` passes `artifacts/hpo/linear_sgd_tuning_da_target_da_price.json` to training.

`scripts/train_and_export_runs.py` reads these files via `--hpo-artifact`, extracts `best_params`, and applies them directly in Python.  
This replaces fragile shell-level JSON parsing and ensures deterministic, auditable parameter handoff from tuning to training.

## 4. Step 2: Run Final Thesis Simulations

For the final thesis multi-market benchmark on the server, use:

```bash
chmod +x rq2_run_final_thesis_multi_2m.sh
nohup ./rq2_run_final_thesis_multi_2m.sh \
  > artifacts/simulation_runs/rq2_final_thesis_multi_2m_launcher.out 2>&1 &
```

This script runs:

- Models: `xgb`, `tft`, `linear`
- Strategy: `multi`
- Horizon: `2025-05-31T22:00:00Z` to `2025-07-31T22:00:00Z`
- Quantiles: `p30-p30`, `p50-p50`, `p70-p70` first, then `p90-p90`, `p10-p10`
- Strict final SoC: `--final-soc-mode hard`
- Strict validity: `--strict-simulation-validity`
- No GHPF: `--disable-ghpf --no-enable-global-perfect-foresight`

The script uses dedicated output folders for every model/quantile run, so aggregate files do not overwrite each other.

Default output layout:

```text
artifacts/simulation_runs/thesis_final_multi_2m_<UTC_TIMESTAMP>/
  xgb_p30/
  xgb_p50/
  ...
  benchmarks_naive/
  benchmarks_rhpf/
  logs/
    status.csv
    manifest.tsv
    xgb_p30/stdout.log
    xgb_p30/stderr.log
    ...
```

Logs are saved inside the simulation run root under:

```text
artifacts/simulation_runs/thesis_final_multi_2m_<UTC_TIMESTAMP>/logs/
```

Monitor the launcher:

```bash
tail -f artifacts/simulation_runs/rq2_final_thesis_multi_2m_launcher.out
```

Monitor a specific simulation:

```bash
RUN_ROOT=artifacts/simulation_runs/thesis_final_multi_2m_<UTC_TIMESTAMP>
tail -f "$RUN_ROOT/logs/xgb_p30/stdout.log"
tail -f "$RUN_ROOT/logs/xgb_p30/stderr.log"
```

The script runs up to 4 simulations concurrently by default. Override this if needed:

```bash
MAX_PARALLEL_JOBS=4 ./rq2_run_final_thesis_multi_2m.sh
```

Naive and RHPF benchmarks are run once each in separate dedicated folders and can be merged into model/quantile analysis later through their ledger and path outputs. GHPF is intentionally never run in this script.

## 5. Quantile Sweep Simulation & Reporting

Run the full quantile sweep simulation:

```bash
make sim-all-quantiles
```

This evaluates BESS trading performance across the following intervals:

- `0.5–0.5`: Baseline median forecast.
- `0.3–0.7`: Standard symmetric 40% prediction interval.
- `0.1–0.9`: Wide 80% confidence interval.
- `0.1–0.3`: Evaluated for varied profit outcomes.
- `0.3–0.5`: Used for stable dual-market strategy analysis.
- `0.5–0.7`: Found to be effective for capturing moderate spikes.
- `0.7–0.9`: Identified as the most effective interval for the Balancing Market (BM) to capture extreme high-reward price values.

Generate the merged thesis benchmark output:

```bash
make thesis-report
```

This aggregates sweep outputs into:

- `artifacts/thesis_benchmark_report.csv`

### 5.1 Full Grid (All Models × All Strategies × All DA Roles × All Quantile Pairs)

Best practice for large benchmark campaigns is to run a timestamped grid to avoid accidental overwrites.

Run the complete grid on the full test horizon:

```bash
make sim-grid-full
```

This executes:

- Models: `xgboost`, `linear`, `tft`
- Strategies: `multi`, `da_only`, `afrr_only`
- DA roles: `low`, `mid`, `high`
- Quantile pairs: `p50-p50,p30-p70,p10-p90,p10-p30,p30-p50,p50-p70,p70-p90`

Outputs are written to unique timestamped roots, e.g.:

- `artifacts/simulation_runs/quantile_grid_mid_<stamp>/xgboost/...`
- `artifacts/simulation_runs/quantile_grid_high_<stamp>/tft/...`

So runs do not overwrite each other unless the same `SIM_GRID_STAMP` is reused.

### 5.2 Smoke Grid (Fast End-to-End Validation)

Run a short-window version of the same full grid:

```bash
make sim-grid-smoke
```

Default smoke horizon is 24 hours and can be changed:

```bash
make sim-grid-smoke GRID_SMOKE_HOURS=12
```

### 5.3 Useful Overrides

You can customize the grid with Make variables:

```bash
make sim-grid-full \
  GRID_DA_ROLES="low mid high" \
  GRID_STRATEGIES="multi da_only afrr_only" \
  GRID_QUANTILE_PAIRS="p50-p50,p30-p70,p10-p90" \
  SIM_GRID_STAMP=$(date +%Y%m%d_%H%M%S)
```

## 6. Audit & Output Artifacts

For every successful run, the pipeline automatically produces an immutable deliverable archive:

- `artifacts/model_runs/<run_id>_deliverable.zip`

Each deliverable contains, at minimum:

- Trained run `manifest.json`.
- Metrics and evaluation outputs.
- Simulation ledgers and summaries.
- Provenance metadata (including commit/dependency/runtime metadata and data-hash lineage) to ensure complete reproducibility and data traceability.
