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

## 2. Supervisor Smoke Execution

The supervisor should run the smoke workflow below. It verifies installation, data availability, model training entry points, prediction export, simulations, audit packaging, and grid backtesting without launching the full thesis workload.

### 2.1 Smoke Workflow

On a CPU-only machine, run:

```bash
make doctor
make smoke-test DEVICE=cpu
make sim-grid-smoke GRID_SMOKE_HOURS=12
```

On a CUDA machine, run:

```bash
make doctor
make smoke-test DEVICE=cuda
make sim-grid-smoke GRID_SMOKE_HOURS=12
```

`make smoke-test` runs the reduced end-to-end DAG for all three model families:

- Tune -> Train -> Simulate -> Audit for XGBoost.
- Tune -> Train -> Simulate -> Audit for Linear.
- Tune -> Train -> Simulate -> Audit for TFT.

`make sim-grid-smoke` then checks the simulation grid against the latest smoke model manifests over a short time window.

### 2.2 What Makes It A Smoke Run

The smoke targets call the normal DAG with reduced runtime settings:

```text
IS_SMOKE_TEST=1
FORECAST_HOURS=24
SIM_HORIZON_HOURS=24
```

The Python training and tuning scripts read `IS_SMOKE_TEST` via `os.environ` and reduce the workload, for example by limiting HPO trials, rows, epochs, and simulation horizon.

The simulation grid smoke test uses:

```text
GRID_SMOKE_HOURS=12
```

This can be increased to the default 24-hour smoke window:

```bash
make sim-grid-smoke
```

### 2.3 Individual Smoke Checks

If a supervisor wants to isolate one model family, run one of:

```bash
make smoke-xgb
make smoke-linear
make smoke-tft DEVICE=cpu
```

Use `DEVICE=cuda` for TFT only when CUDA is available.

### 2.4 Expected Smoke Coverage

The smoke workflow exercises:

- Dependency and data preflight via `make doctor`.
- HPO/tuning entry points.
- Model training for XGBoost, Linear, and TFT.
- Prediction export and latest manifest creation.
- Battery backtest simulation over a short horizon.
- Audit package creation.
- Multi-model, multi-strategy, multi-DA-role grid execution.

The grid smoke test covers:

- Models: `xgboost`, `linear`, `tft`
- Strategies: `multi`, `da`, `afrr`
- DA roles: `low`, `mid`, `high`
- Quantile pairs from `SIM_QUANTILE_SWEEP_DEFAULT`

### 2.5 Success Criteria

A smoke validation is successful if all three commands exit with status code `0`:

```bash
make doctor
make smoke-test DEVICE=cpu
make sim-grid-smoke GRID_SMOKE_HOURS=12
```

Expected artifacts:

- New smoke run folders under `artifacts/model_runs/`.
- Latest model pointers under `artifacts/model_runs/latest_*.json`.
- Deliverable archives matching `artifacts/model_runs/<run_id>_deliverable.zip`.
- Smoke simulation outputs under `artifacts/simulation_runs/default_*`.
- Smoke grid outputs under `artifacts/simulation_runs/quantile_grid_smoke_*`.

## 3. Audit & Output Artifacts

For every successful run, the pipeline automatically produces an immutable deliverable archive:

- `artifacts/model_runs/<run_id>_deliverable.zip`

Each deliverable contains, at minimum:

- Trained run `manifest.json`.
- Metrics and evaluation outputs.
- Simulation ledgers and summaries.
- Provenance metadata (including commit/dependency/runtime metadata and data-hash lineage) to ensure complete reproducibility and data traceability.

## 4. Full Thesis Commands, Not For Smoke Testing

The commands below are intentionally not part of the supervisor smoke workflow. They run the full thesis workload and can take substantially longer.

Full model DAGs:

```bash
make all-xgb
make all-linear
make all-tft
```

Full simulation grid:

```bash
make sim-grid-full
```

Final thesis multi-market benchmark:

```bash
chmod +x rq2_run_final_thesis_multi_2m.sh
nohup ./rq2_run_final_thesis_multi_2m.sh \
  > artifacts/simulation_runs/rq2_final_thesis_multi_2m_launcher.out 2>&1 &
```
