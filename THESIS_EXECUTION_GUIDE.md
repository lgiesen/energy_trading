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

## 5. Audit & Output Artifacts
For every successful run, the pipeline automatically produces an immutable deliverable archive:
- `artifacts/model_runs/<run_id>_deliverable.zip`

Each deliverable contains, at minimum:
- Trained run `manifest.json`.
- Metrics and evaluation outputs.
- Simulation ledgers and summaries.
- Provenance metadata (including commit/dependency/runtime metadata and data-hash lineage) to ensure complete reproducibility and data traceability.
