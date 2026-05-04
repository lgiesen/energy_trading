# ML Model Documentation

## 1. Documentation Scope
This document covers the implemented machine-learning forecasting stack in this repository as observed from source code, scripts, configs, docs, and local artifacts.

Discovered ML tasks:
- Probabilistic forecasting of Day-Ahead (`da`) and aFRR targets.
- Multi-horizon forecasting with lead-wise outputs.
- Model-driven battery trading simulation input generation.

Discovered model families:
- XGBoost quantile models.
- Temporal Fusion Transformer (TFT).
- Linear quantile baseline (SGD-based with fallback).

Not found in repository:
- Active SSA model implementation/training entrypoint (`src/energy_trading/models/univariate_ssa.py` and `scripts/tune_univariate_ssa.py` are deleted in current workspace state).
- Centralized `AGENTS.md` repository instruction file.

## 2. Executive Summary
The repository implements a three-family probabilistic ensemble (`xgboost`, `tft`, `linear`) with a shared orchestration script and shared evaluation stack. Forecasts are exported in run directories under `artifacts/model_runs/<run_id>/` with manifests, metrics, predictions, and diagnostics.

Core patterns observed:
- Quantile grid is harmonized to `0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99` across model families.
- Conformal post-processing and quantile monotonicity enforcement are implemented for deployment-facing outputs.
- Tail-aware objectives/selection are implemented in tuning scripts (`tail_upper_mae`, `asymmetric_mae` selection).
- TensorBoard logging exists for all three families but via different integration style (Lightning logger for TFT, utility writers for XGBoost/Linear).

Smoke-test path exists through:
- `scripts/tune_xgboost.py`
- `scripts/tune_linear.py`
- `scripts/train_and_export_runs.py`
- `scripts/run_battery_backtest.py`

Major unresolved gaps:
- No active SSA path in codebase (despite historical references in conversation/logs).
- Some reproducibility metadata is present, but no single lockstep “one-command full benchmark” script with guaranteed deterministic environment capture was found.
- Full automated unit/integration tests for all model pipelines were not found in inspected paths (TODO: verify with author where CI test suite lives).

## 3. Repository Map
| Area | Path | Purpose | Notes |
|---|---|---|---|
| Model training orchestration | `scripts/train_and_export_runs.py` | Trains model families per bundle/target, merges outputs, writes manifest/context | `--model-type` choices: `xgboost`, `tft`, `linear` |
| XGBoost model | `src/energy_trading/models/train_xgboost_export.py` | Lead-wise + quantile-wise training/export | Tail weighting, conformal, monotonic sorting |
| TFT model | `src/energy_trading/models/train_tft_export.py` | TFT training/export with quantile outputs | Lightning + checkpointing + interpretation artifacts |
| Linear model | `src/energy_trading/models/train_linear_export.py` | Linear quantile training/export | SGD quantile + fallback to `QuantileRegressor` |
| XGBoost tuning | `scripts/tune_xgboost.py` | Optuna HPO for tail-aware objective | Writes to `artifacts/hpo/*.json`/`*.csv` |
| Linear tuning | `scripts/tune_linear.py` | Grid search HPO for tail-aware objective | Uses shared evaluator metrics |
| Shared evaluator | `src/energy_trading/evaluation/shared_evaluator.py` | Canonical metric computation/export + prediction export | Includes lead-level and tail metrics |
| Conformal calibration | `src/energy_trading/evaluation/conformal_calibration.py` | Split-conformal shifts + application | Also enforces monotonic quantiles |
| TensorBoard utilities | `src/energy_trading/evaluation/tensorboard_utils.py` | Unified TB logdir + scalar logging wrappers | Root default `artifacts/tensorboard_logs` |
| Run context helpers | `src/energy_trading/utils/run_context.py` | Run-id resolution and aliases | Model aliases include xgboost/tft/linear |
| Simulation engine | `scripts/run_battery_backtest.py` | Backtesting using prediction artifacts + manifests | Supports `--run-manifest` / `--run-id` |
| Primary docs | `README.md`, `docs/*.md` | Project and methodology docs | Recent README excludes SSA |
| Dependency spec | `requirements.txt` | Environment package pins | Includes xgboost, torch, lightning, pytorch-forecasting, optuna |
| Artifacts (local) | `artifacts/model_runs/*` | Run outputs (metrics/preds/manifests/checkpoints) | Fulltrain runs for 3 model families found |

## 4. Model Inventory
| Model / family | Implementation path | Main class/function | Task | Inputs | Outputs | Config path(s) | Training entry point | Evaluation entry point | Artifacts |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost quantile ensemble | `src/energy_trading/models/train_xgboost_export.py` | `main()`, `_train_for_lead()`, `_train_for_quantile()` | Probabilistic multi-horizon regression | Tabular engineered features + target, per bundle/target | Quantile predictions (`p01..p99`), lead-wise wide/long frames | argparse in script; optional policy via args | `python -m src.energy_trading.models.train_xgboost_export ...` | Shared evaluator inside training script; downstream summary in orchestration | `artifacts/model_runs/<run_id>/predictions`, `metrics`, `models`, `feature_importance` |
| TFT | `src/energy_trading/models/train_tft_export.py` | `main()`, `TemporalFusionTransformer.from_dataset(...)` | Probabilistic sequence forecasting | Time-indexed features grouped by series id | Long quantile predictions + lead-1 wide outputs | argparse in script | `python -m src.energy_trading.models.train_tft_export ...` | Shared evaluator invoked in script; additional decay reports | `artifacts/model_runs/<run_id>/models`, `predictions`, `metrics`, `plots` |
| Linear quantile baseline | `src/energy_trading/models/train_linear_export.py` | `main()`, `_build_linear_pipeline()`, `_train_target()` | Probabilistic multi-horizon regression baseline | Tabular engineered features + target | Quantile predictions (`p01..p99`), wide + long frames | argparse in script | `python -m src.energy_trading.models.train_linear_export ...` | Shared evaluator invoked in script; per-lead metrics | `artifacts/model_runs/<run_id>/predictions`, `metrics`, `reports`, model pickles |

## 5. Global ML Workflow
Observed end-to-end workflow:
1. Data is loaded from `data/model_input/<bundle>/<split>.parquet` by model training scripts.
2. Scripts prepare features/targets using bundle-specific target mapping and drop/keep logic.
3. Training loops run by model family:
- XGBoost/Linear: explicit loops across forecast lead and quantiles.
- TFT: sequence dataset + joint quantile head.
4. Validation/test predictions are generated and post-processed:
- quantile monotonicity sort.
- conformal calibration shifts (where configured in family scripts).
5. Shared evaluation computes deterministic, directional, tail, spike, and quantile-coverage metrics.
6. Artifacts are written under `artifacts/model_runs/<run_id>/...`.
7. `scripts/train_and_export_runs.py` combines fragment manifests into a run-level `manifest.json`, performs quality checks, and writes run context.
8. `scripts/run_battery_backtest.py` consumes run manifest/predictions for simulation.

Representative commands (from scripts and logs):
```bash
python3 scripts/train_and_export_runs.py --model-type xgboost --run-id smoke_xgb --forecast-horizon-hours 24 --n-estimators 200 --max-depth 4 --learning-rate 0.05 --allow-cpu
python3 scripts/train_and_export_runs.py --model-type linear --run-id smoke_linear --forecast-horizon-hours 24
python3 scripts/train_and_export_runs.py --model-type tft --run-id smoke_tft --forecast-horizon-hours 24 --device cpu --num-workers 0
```

## 6. Configuration System
Configuration mechanism observed:
- Primary: `argparse` in each training/tuning script.
- Optional metadata/policy: run context and policy files handled by orchestration.
- Environment variables used for run context and platform behavior in some flows.

| Config area | Path/key | Default/value | Used by | Meaning | Notes |
|---|---|---|---|---|---|
| Forecast horizon | CLI `--forecast-horizon-hours` | Varies by script (`24`/`48`/`1`) | XGBoost/Linear/TFT/orchestration | Number of lead steps predicted/exported | Verify per-model defaults before benchmark |
| Quantile grid | `QUANTILES` constants in model scripts | `0.01..0.99` fixed set | XGBoost/Linear/TFT | Target probabilistic outputs | Harmonized across families |
| XGB objective | `objective=reg:quantileerror`, `quantile_alpha` | per quantile | XGBoost | Quantile loss per model | One model per lead × quantile |
| XGB tail weighting | `_tail_sample_weights` | multiplier `3.0` | XGBoost + tuner | Emphasize tails in training | Thresholds computed on train target |
| Linear scaler | `RobustScaler(10,90)` | fixed in code | Linear | Improve SGD stability | Before quantile regression |
| Linear reg | `alpha`, `l1_ratio`, `penalty=elasticnet` | script defaults/grid | Linear + tuner | L1/L2 mix and strength | In SGD fallback path too |
| TFT loss | `QuantileLoss(quantiles=...)` | fixed quantile grid | TFT | Joint quantile optimization | Extreme quantiles included |
| TFT optimization | `learning_rate`, `gradient_clip_val` | CLI exposed | TFT | Stability/training speed | Clip default exposed for hardening |
| Early stopping | XGB/TFT args + callbacks | present | XGB/TFT | Stop when no val improvement | Linear has iterative SGD termination |
| Split logic | train/val/test parquet usage | explicit load helpers | all models | dataset partitioning | Detailed split policy not centralized in single config file |
| Seed | `--seed` | often `42` | all models | reproducibility | Determinism still hardware/library dependent |
| Device | `--device` + runtime fallback | e.g. `cpu/cuda/mps` | XGB/TFT | compute backend | Logs show MPS fallback for XGB |
| TB root | `artifacts/tensorboard_logs` | utility default | all models | scalar event output | TFT additionally logs under run dir via Lightning logger |
| Canonical metrics | `canonical_metrics.parquet` | append mode | shared evaluator | comparable model metrics | Path argument-driven in evaluator calls |
| Canonical predictions | `canonical_predictions.parquet` | append mode | shared evaluator | comparable prediction traces | includes `lead_h`, `split` |

## 7. Model Details

### XGBoost Quantile Ensemble

#### Purpose
Probabilistic tabular forecasting per target and forecast lead, with tail-aware fitting and quantile outputs.

#### Source Locations
- Definition/training: `src/energy_trading/models/train_xgboost_export.py`
- Tuning: `scripts/tune_xgboost.py`
- Orchestration: `scripts/train_and_export_runs.py`
- Shared metrics: `src/energy_trading/evaluation/shared_evaluator.py`
- Conformal: `src/energy_trading/evaluation/conformal_calibration.py`
- Artifacts example: `artifacts/model_runs/fulltrain_2026-04-25T17-30-55Z_xgboost/`

#### Observed Implementation
The repository implements separate XGBoost models per lead and per quantile.
Quantiles are represented as fixed probabilities `0.01,0.05,0.10,0.50,0.90,0.95,0.99`.
Leads are represented by horizon step index (1..`forecast_horizon_hours`).
Training loop builds shifted labels per lead and fits models with `reg:quantileerror` and `quantile_alpha=q`.
Outputs are post-processed with monotonic row-wise sorting and optional conformal shifts.

#### Architecture / Algorithm
- Library: `xgboost` Python API (`XGBRegressor` and/or lower-level API depending code path).
- Objective: quantile error.
- Training structure: `for lead in leads` + `for q in quantiles`.
- Tail weighting: `_tail_sample_weights` with train-target percentile thresholds and multiplier.
- Output heads: independent scalar models, later assembled into multi-quantile rows.

#### Important Configuration
| Parameter | Value/default | Source | Meaning | Notes |
|---|---|---|---|---|
| Quantiles | `[0.01,0.05,0.10,0.50,0.90,0.95,0.99]` | `train_xgboost_export.py` | Output quantile levels | Fixed in code |
| Objective | `reg:quantileerror` | `train_xgboost_export.py` | Quantile loss | Per-quantile model |
| Tail multiplier | `3.0` | `train_xgboost_export.py` | Tail weighting strength | Applied to upper/lower tails |
| Horizon | CLI `--forecast-horizon-hours` | script arg | Number of lead models | 24/48 used in logs |
| Early stop | CLI `--early-stopping-rounds` | script arg | Stop criterion | val-set based |

#### Rationale and Trade-offs
| Design choice | Observed implementation | Explicit rationale | Inferred rationale | Trade-offs | Evidence |
|---|---|---|---|---|---|
| Per-lead, per-quantile models | Nested loops by lead and quantile | Not found in repository | Independent optimization per quantile/horizon can capture heteroskedastic errors | Many artifacts, long training time, deployment complexity, crossing risk | `train_xgboost_export.py` loops and output assembly |
| Tail weighting | Percentile-based sample weighting | Not found in repository | Improve spike/crash learning under asymmetric risk | Potential overfitting tails; requires stronger regularization | `_tail_sample_weights`, `tune_xgboost.py` |
| Monotonic sort + conformal | Post-processing before export | Statistical calibration/consistency implied by code comments/util names | Prevent invalid quantile order and improve empirical coverage | Distorts raw model outputs; extra calibration dependency | `conformal_calibration.py`, xgb export script |

Potential alternatives (not necessarily considered in repo):
- Multi-output quantile model per lead.
- Single global-horizon model.
- Neural probabilistic global model only.
- Pure conformal intervals over point models.
- Isotonic/monotone quantile regression constraints.

#### Training
```bash
python -m src.energy_trading.models.train_xgboost_export \
  --base-dir data/model_input \
  --bundle da \
  --target-col target_da_price \
  --run-dir artifacts/model_runs/<run_id> \
  --forecast-horizon-hours 24 \
  --n-estimators 200 \
  --max-depth 4 \
  --learning-rate 0.05 \
  --allow-cpu
```

#### Evaluation
Evaluation is embedded in training/export:
- shared evaluator metrics per split.
- decay/lead metrics CSV and plots.
- summary in JSON/CSV under run `metrics/` and `reports/`.

No dedicated standalone xgboost-only evaluation script found in repository.

#### Inference / Prediction
No separate real-time inference CLI found; prediction generation is integrated in training/export and run orchestration.

#### TensorBoard
```bash
tensorboard --logdir artifacts/tensorboard_logs
```
Observed TB logging via utility functions for scalar metrics.

#### Performance Metrics
Example metrics are stored in:
- `artifacts/model_runs/fulltrain_2026-04-25T17-30-55Z_xgboost/metrics/`
- `artifacts/model_runs/fulltrain_2026-04-25T17-30-55Z_xgboost/reports/`

| Split/dataset | Metric | Value | Direction | Run/checkpoint | Source |
|---|---|---|---|---|---|
| val/test | MAE, RMSE, tail metrics, coverage metrics | See files | lower better for errors | fulltrain run | `metrics/*.json`, `reports/*` |

#### Known Limitations
- Training-time can be large due to lead × quantile decomposition.
- Quantile crossing must be repaired post hoc.
- Device fallback complexity (MPS/CUDA/CPU differences observed in logs).
- TODO: verify model artifact naming guarantees for all lead/quantile combinations in edge-case early stop failures.

#### Reproducibility Notes
- Seed argument exists.
- Determinism beyond seed is not guaranteed across devices/backends.
- Dependency versions pinned in `requirements.txt`.

### Linear Quantile Baseline (SGD + fallback)

#### Purpose
Regularized linear probabilistic baseline for multi-horizon forecasting.

#### Source Locations
- `src/energy_trading/models/train_linear_export.py`
- `scripts/tune_linear.py`
- Shared eval/calibration modules as above.
- Artifacts example: `artifacts/model_runs/fulltrain_2026-04-25T17-30-55Z_linear/`

#### Observed Implementation
The repository implements per-lead and per-quantile linear models with robust preprocessing.
Primary estimator path is `SGDRegressor(loss="quantile", quantile=<q>, penalty="elasticnet", ...)`.
Fallback to `QuantileRegressor` is present on estimator compatibility errors.

#### Architecture / Algorithm
- Pipeline: imputation + robust scaling + quantile regression.
- Quantiles same 7-level grid.
- Independent models per lead/quantile.
- Monotonic sorting and conformal shift application before export.

#### Important Configuration
| Parameter | Value/default | Source | Meaning | Notes |
|---|---|---|---|---|
| Scaler | `RobustScaler(10,90)` | `train_linear_export.py` | Feature scaling for SGD stability | fixed in code |
| Loss | `SGDRegressor(loss="quantile")` | `train_linear_export.py` | Quantile optimization | fallback path exists |
| Regularization | `alpha`, `l1_ratio`, `penalty=elasticnet` | CLI/tuner | L1/L2 shrinkage | tuned by `tune_linear.py` |
| Quantiles | fixed 7 | code constant | probabilistic outputs | harmonized |

#### Rationale and Trade-offs
| Design choice | Observed implementation | Explicit rationale | Inferred rationale | Trade-offs | Evidence |
|---|---|---|---|---|---|
| SGD quantile baseline | `SGDRegressor` quantile loss | Not found in repository | Faster/scalable alternative to simplex quantile regression | May be less stable; sensitive to scaling/eta | `_build_linear_pipeline` + tuner grid |
| Robust scaling | pipeline default | Not found in repository | Reduce outlier impact and stabilize gradients | May compress informative scale differences | pipeline steps |
| Per-lead/per-quantile decomposition | explicit loops | Not found in repository | Horizon-specific error handling | artifact explosion, crossing risk | training loops/output files |

#### Training
```bash
python -m src.energy_trading.models.train_linear_export \
  --base-dir data/model_input \
  --bundle da \
  --target-col target_da_price \
  --run-dir artifacts/model_runs/<run_id> \
  --forecast-horizon-hours 24 \
  --alpha 1.0
```

#### Evaluation
Integrated in training script using shared evaluator; outputs under `metrics/` and `reports/`.

#### Inference / Prediction
No dedicated standalone inference CLI found; prediction export occurs during training/export.

#### TensorBoard
```bash
tensorboard --logdir artifacts/tensorboard_logs
```
Logged scalar metrics through TB utility functions.

#### Performance Metrics
See:
- `artifacts/model_runs/fulltrain_2026-04-25T17-30-55Z_linear/metrics/`
- `artifacts/model_runs/fulltrain_2026-04-25T17-30-55Z_linear/reports/`

#### Known Limitations
- Estimator API differences across sklearn versions can trigger fallback behavior.
- Independent quantile models require crossing fix.
- TODO: verify convergence diagnostics per lead/quantile are persisted consistently.

#### Reproducibility Notes
- Seed control available.
- Requires matching sklearn version for identical behavior.

### Temporal Fusion Transformer (TFT)

#### Purpose
Neural sequence model for joint quantile forecasting over temporal covariates.

#### Source Locations
- `src/energy_trading/models/train_tft_export.py`
- Shared eval/calibration utilities.
- Artifacts example: `artifacts/model_runs/fulltrain_2026-04-25T17-30-55Z_tft/`

#### Observed Implementation
The repository implements TFT via `pytorch_forecasting.TemporalFusionTransformer`.
It trains on time-series windows with encoder/decoder lengths and predicts the same 7 quantiles jointly.
Conformal shifts are applied to test quantiles; metrics exported through shared evaluator and report files.

#### Architecture / Algorithm
- Backbone: TFT with variable selection networks, LSTM encoder/decoder, interpretable attention, gated residual blocks.
- Loss: `QuantileLoss(quantiles=[0.01,0.05,0.1,0.5,0.9,0.95,0.99])`.
- Optimizer/scheduling: configured via Lightning/TFT defaults + CLI-exposed learning rate.
- Stabilization: configurable `gradient_clip_val`.
- Data representation: `TimeSeriesDataSet` style split with known/unknown variable handling.

#### Important Configuration
| Parameter | Value/default | Source | Meaning | Notes |
|---|---|---|---|---|
| Quantiles | 7 quantiles | `train_tft_export.py` | joint output heads | shared with other families |
| `max_encoder_length` | CLI default observed in script | script args | history window | often 72/168 in usage |
| `max_prediction_length` | CLI default observed in script | script args | horizon window | 24/48 usage |
| `learning_rate` | CLI arg | script args | optimizer LR | tunable for stability |
| `gradient_clip_val` | CLI arg | script args | gradient clipping | hardening against extremes |

#### Rationale and Trade-offs
| Design choice | Observed implementation | Explicit rationale | Inferred rationale | Trade-offs | Evidence |
|---|---|---|---|---|---|
| Joint quantile learning | single TFT with quantile loss | Not found in repository | Shared representation for temporal dynamics across quantiles | Heavier compute, harder debugging than per-quantile trees | model/loss construction |
| Gradient clipping exposed | trainer arg | Not found in repository | Stabilize training with extreme quantiles | May slow learning if overly strict | trainer config in script |
| Conformal + monotonic post-process | applied before export | Not found in repository | Improve empirical coverage and consistency | Post-hoc distortions possible | calibration utility usage |

#### Training
```bash
python -m src.energy_trading.models.train_tft_export \
  --base-dir data/model_input \
  --bundle da \
  --target-col target_da_price \
  --run-dir artifacts/model_runs/<run_id> \
  --max-encoder-length 72 \
  --max-prediction-length 24 \
  --learning-rate 5e-4 \
  --gradient-clip-val 0.05 \
  --device cpu \
  --num-workers 0
```

#### Evaluation
Integrated in script (shared evaluator + decay reports). Additional interpretation outputs may be produced.

#### Inference / Prediction
No separate inference-only CLI found; prediction export is integrated into training workflow.

#### TensorBoard
```bash
tensorboard --logdir artifacts/model_runs/<run_id>/tb
```
and/or
```bash
tensorboard --logdir artifacts/tensorboard_logs
```
depending on logger configuration path.

#### Performance Metrics
See:
- `artifacts/model_runs/fulltrain_2026-04-25T17-30-55Z_tft/metrics/`
- `artifacts/model_runs/fulltrain_2026-04-25T17-30-55Z_tft/reports/`

#### Known Limitations
- Slower smoke/full runs on CPU.
- DataLoader worker count can bottleneck performance.
- TODO: verify unified TB tag taxonomy coverage for every logged TFT scalar.

#### Reproducibility Notes
- Seed is set in script.
- Exact reproducibility depends on PyTorch/Lightning/CUDA setup.

## 8. Experiment Tracking, Metrics, and TensorBoard
Tracking mechanisms observed:
- JSON/CSV metric files under each run’s `metrics/` and `reports/`.
- Canonical parquet appends from shared evaluator when configured.
- TensorBoard logs via utility writers and Lightning logger.
- Manifest files capturing output locations.

| Model | Run | Dataset/split | Metric | Value | Checkpoint | Source |
|---|---|---|---|---|---|---|
| xgboost | fulltrain_2026-04-25T17-30-55Z_xgboost | val/test | multiple (MAE/RMSE/tail/coverage etc.) | see files | model files in run dir | `artifacts/model_runs/.../metrics/*.json` |
| linear | fulltrain_2026-04-25T17-30-55Z_linear | val/test | multiple | see files | pickle artifacts | `artifacts/model_runs/.../metrics/*.json` |
| tft | fulltrain_2026-04-25T17-30-55Z_tft | val/test | multiple | see files | `.ckpt` in models | `artifacts/model_runs/.../metrics/*.json` |

What to inspect in TensorBoard:
- training/validation loss trends,
- lead-specific metrics where logged,
- divergence/plateau signs,
- potential overfitting via widening train-vs-val gaps.

## 9. Checkpoints and Artifacts
| Artifact type | Path | Produced by | Consumed by | Naming scheme | Notes |
|---|---|---|---|---|---|
| Run manifest | `artifacts/model_runs/<run_id>/manifest.json` | `train_and_export_runs.py` | `run_battery_backtest.py` and downstream tools | fixed filename | central run index |
| Run context | `artifacts/model_runs/<run_id>/training_run_context.json` | orchestration | auditors/debugging | fixed filename | includes command metadata |
| Prediction parquet | `artifacts/model_runs/<run_id>/predictions/*.parquet` | model scripts | evaluator/simulation | model+bundle+split naming | wide and/or long variants |
| Metrics JSON/CSV | `artifacts/model_runs/<run_id>/metrics/*`, `reports/*` | model scripts | comparison/reporting | per target/split | includes lead decay files |
| XGBoost model files | `artifacts/model_runs/<run_id>/models/*` | xgboost script | inference/export | per lead/quantile patterns | TODO: verify exact extension list per run |
| Linear model files | `artifacts/model_runs/<run_id>/models/*` | linear script | inference/export | per lead/quantile patterns | serialized sklearn objects |
| TFT checkpoints | `artifacts/model_runs/<run_id>/models/*.ckpt` | tft script | inference/export | Lightning checkpoint names | best checkpoint copied to run dir |
| TB event logs | `artifacts/tensorboard_logs/*` and/or run local TB dirs | all families | TensorBoard | framework-generated | layout differs by family |

Best vs latest behavior:
- TFT: best checkpoint selected by monitored validation metric through Lightning checkpoint callback.
- XGBoost/Linear: model artifact selection is training-loop export result; no separate “latest vs best checkpoint” abstraction found.

## 10. Data and Feature Documentation
| Data element | Path/key/code location | Meaning | Required for | Notes |
|---|---|---|---|---|
| Input base directory | `data/model_input/` | prepared model-ready dataset root | all models | expected subfolders by bundle |
| Bundle split files | `data/model_input/<bundle>/train.parquet`, `val.parquet`, `test.parquet` | split datasets | all models/tuners | loaded by scripts |
| Timestamp column | typically `timestamp_utc` | time index | splits/alignment/eval | exact handling differs by script |
| Target mapping | constants/helpers in each model script | forecast target columns | all models | DA and multiple aFRR targets supported |
| Feature columns | script-specific drop/keep logic | model predictors | all models | engineered features expected upstream |
| Missing value handling | model pipelines/loaders | imputation/cleaning | linear/tft/xgb | linear explicitly uses median imputer |

Not found in repository:
- Single canonical feature schema file documenting every feature column and provenance.
- Centralized data contract versioning file for `data/model_input`.

## 11. Training Guide
Environment setup evidence:
- `requirements.txt` exists and pins major libraries.

Minimal setup command:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

| Model | Command | Config | Expected output | Notes |
|---|---|---|---|---|
| XGBoost smoke | `python3 scripts/train_and_export_runs.py --model-type xgboost --run-id smoke_xgb --forecast-horizon-hours 24 --n-estimators 200 --max-depth 4 --learning-rate 0.05 --allow-cpu` | CLI args | run dir with predictions/metrics/manifest | may take significant time by target count |
| Linear smoke | `python3 scripts/train_and_export_runs.py --model-type linear --run-id smoke_linear --forecast-horizon-hours 24` | CLI args | run dir with linear artifacts | ensure sklearn version supports chosen estimator path |
| TFT smoke | `python3 scripts/train_and_export_runs.py --model-type tft --run-id smoke_tft --forecast-horizon-hours 24 --device cpu --num-workers 0` | CLI args | run dir + checkpoints + plots | CPU can be slow |

## 12. Evaluation Guide
| Model | Command | Required checkpoint | Metrics produced | Output path | Notes |
|---|---|---|---|---|---|
| XGBoost | integrated in training commands above | trained model files in run dir | rmse, mae, wmape, mbe, directional, tail, coverage | `artifacts/model_runs/<run_id>/metrics` | no separate eval CLI found |
| Linear | integrated in training | model artifacts | same shared metric suite | same | same |
| TFT | integrated in training | best ckpt | same shared metric suite + TFT reports | same | same |

Comparison rule:
- Use identical split (`val` or `test`), same target, same lead scope, and same metric definitions from shared evaluator.

## 13. Inference / Prediction Guide
No dedicated inference entry point found.

Observed approach:
- Prediction export occurs during train-and-export scripts.
- Simulation consumes exported predictions via run manifest.

Related command:
```bash
python3 scripts/run_battery_backtest.py --run-manifest artifacts/model_runs/<run_id>/manifest.json --split test --model-key xgboost --start 2024-01-01T00:00:00Z --end 2024-01-07T23:00:00Z
```

## 14. Smoke Test Guide for Supervisor
Purpose:
- Validate end-to-end train/export/manifest/simulation plumbing on reduced settings before full runs.

Preconditions:
- `data/model_input` split files available.
- Python env with `requirements.txt` installed.
- write access to `artifacts/model_runs`.

Recommended commands:
```bash
# 1) Tune quick checks
python3 scripts/tune_xgboost.py --bundle da --target-col target_da_price --n-trials 6 --n-estimators 200 --selection-metric tail_upper_mae --allow-cpu
python3 scripts/tune_linear.py --bundle da --target-col target_da_price --selection-metric tail_upper_mae --alpha-grid 1e-4,5e-4 --l1-ratio-grid 0.15,0.30 --learning-rate-grid optimal,adaptive --eta0-grid 0.001,0.01

# 2) Train/export smoke runs
python3 scripts/train_and_export_runs.py --model-type xgboost --run-id smoke_xgb --forecast-horizon-hours 24 --n-estimators 200 --max-depth 4 --learning-rate 0.05 --allow-cpu
python3 scripts/train_and_export_runs.py --model-type linear --run-id smoke_linear --forecast-horizon-hours 24
python3 scripts/train_and_export_runs.py --model-type tft --run-id smoke_tft --forecast-horizon-hours 24 --device cpu --num-workers 0

# 3) Backtest smoke (after manifest exists)
python3 scripts/run_battery_backtest.py --run-manifest artifacts/model_runs/smoke_xgb/manifest.json --split test --model-key xgboost --start 2024-01-01T00:00:00Z --end 2024-01-07T23:00:00Z
```

Expected success criteria:
- `[OK]` completion messages in logs.
- `manifest.json` present in each smoke run dir.
- prediction parquet files present for val/test.
- metrics files generated.

TensorBoard check:
```bash
tensorboard --logdir artifacts/tensorboard_logs
```

Failure diagnosis:
| Symptom | Likely cause | What to check |
|---|---|---|
| `manifest.json` missing | orchestration crashed before finalize | run logs in `<run_id>/logs`, `training_run_context.json` |
| `SGDRegressor unexpected keyword` | sklearn version mismatch | `requirements.txt` version vs active env |
| CUDA/MPS warnings | backend mismatch/fallback | device args (`--device`, `--allow-cpu`) |
| empty metrics dir | training aborted before eval | log tail and exception traceback |
| missing TB logs | logger path mismatch | check both run-local TB and `artifacts/tensorboard_logs` |
| backtest `--manifest` error | wrong flag name/use of old command | use `--run-manifest` (alias handling verify script version) |
| shape mismatch in evaluation | lead/quantile assembly issue | prediction wide/long parquet schema |

## 15. Reproducibility and Auditability
| Reproducibility item | Status | Evidence | Gap |
|---|---|---|---|
| Seed control | Partial | `--seed` args in model scripts | Full determinism not guaranteed |
| Dependency pinning | Partial | `requirements.txt` pinned packages | No lockfile for all transitive deps found |
| Config snapshot | Partial | run context + manifests | Not all CLI overrides guaranteed to be serialized uniformly |
| Metric provenance | Good | shared evaluator + per-run metrics files | TODO: verify all scripts append canonical parquet in every branch |
| Checkpoint naming | Partial | TFT best ckpt callback; model dirs | No universal naming standard doc |
| Data versioning | Partial | split file paths fixed | explicit dataset version hash not found |
| Git commit capture | Not found in repository | N/A | TODO: verify with author if external tracker stores commit SHA |

## 16. Design Rationale and Trade-offs
| Decision | Observed implementation | Explicit rationale | Inferred rationale | Trade-offs | Evidence/source |
|---|---|---|---|---|---|
| Use three-family ensemble | xgboost+tft+linear active | README/docs indicate benchmark architecture | balance of capacity/interpretability/robustness | more maintenance and tuning overhead | `README.md`, model scripts |
| Harmonized 7-quantile grid | fixed in all 3 scripts | Not found in repository | apples-to-apples risk comparison and simulation inputs | increases model count for independent approaches | model constants |
| Post-hoc calibration and monotonicity | conformal + sort utilities | Not found in repository | improve calibration and prevent invalid quantile order | additional complexity and possible distribution shift | `conformal_calibration.py` |
| Lead-wise decomposition (xgb/linear) | explicit loops by lead | Not found in repository | horizon-specific specialization | long runtime/storage blow-up | training loops + artifacts |
| Shared evaluator | one metrics module | explicit in code structure | metric consistency across families | integration drift risk if bypassed | `shared_evaluator.py` usage |

## 17. Risks, Limitations, and Open Questions
Risks:
- Quantile crossing exists before post-process; correctness depends on repair step.
- Tail weighting may overfit if regularization/tuning is weak.
- Hardware backend variability (CPU/CUDA/MPS) affects reproducibility and runtime.
- Missing centralized feature-contract documentation can cause silent schema drift.
- Orchestration failures can leave partial runs without final manifest.

Open questions:
- [ ] TODO: confirm canonical parquet export is enabled in every production training path.
- [ ] TODO: confirm final benchmark uses identical horizon and split policies across all families.
- [ ] TODO: verify whether any external experiment tracker (MLflow/W&B) is used outside repo.
- [ ] TODO: verify expected runtime budgets for full tuning/training on target hardware.
- [ ] TODO: confirm current backtest consumes long quantile predictions consistently for all families.

## 18. Documentation Coverage
| Area | Status | Evidence | Missing items |
|---|---|---|---|
| Model inventory | Covered | model scripts + orchestration | none |
| Configs/params | Covered (argparse) | script args/tuners | no centralized config catalog file |
| Training | Covered | commands and script flows | none |
| Evaluation | Covered | shared evaluator + run metrics | standalone eval CLIs not found |
| Inference | Partial | integrated export + backtest ingestion | no dedicated inference-only entrypoint |
| Metrics | Covered | shared evaluator + run metrics files | ensure canonical parquet usage across all code paths |
| TensorBoard | Covered | utilities + Lightning logger | tag taxonomy completeness TODO |
| Checkpoints/artifacts | Covered | run dirs/manifests/models | some naming details TODO |
| Smoke tests | Covered | practical commands and criteria | exact runtime expectations vary by hardware |
| Rationale/trade-offs | Covered | fact + inference separation | some rationale inferred only |
| Reproducibility | Partial | seed/deps/context files | commit/data version provenance gaps |
| Data documentation | Partial | split paths + script handling | full feature dictionary not found |

