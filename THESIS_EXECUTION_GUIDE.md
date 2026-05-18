# THESIS_EXECUTION_GUIDE

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

## 3. Canonical Full Execution
Run final canonical model pipelines:

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

## 4. Quantile Sweep Simulation & Reporting
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

### 4.1 Full Grid (All Models × All Strategies × All DA Roles × All Quantile Pairs)
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

### 4.2 Smoke Grid (Fast End-to-End Validation)
Run a short-window version of the same full grid:

```bash
make sim-grid-smoke
```

Default smoke horizon is 24 hours and can be changed:

```bash
make sim-grid-smoke GRID_SMOKE_HOURS=12
```

### 4.3 Useful Overrides
You can customize the grid with Make variables:

```bash
make sim-grid-full \
  GRID_DA_ROLES="low mid high" \
  GRID_STRATEGIES="multi da_only afrr_only" \
  GRID_QUANTILE_PAIRS="p50-p50,p30-p70,p10-p90" \
  SIM_GRID_STAMP=$(date +%Y%m%d_%H%M%S)
```

## 5. Audit & Output Artifacts
For every successful run, the pipeline automatically produces an immutable deliverable archive:
- `artifacts/model_runs/<run_id>_deliverable.zip`

Each deliverable contains, at minimum:
- Trained run `manifest.json`.
- Metrics and evaluation outputs.
- Simulation ledgers and summaries.
- Provenance metadata (including commit/dependency/runtime metadata and data-hash lineage) to ensure complete reproducibility and data traceability.
