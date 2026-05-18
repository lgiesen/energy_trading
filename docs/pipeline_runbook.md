# Pipeline Runbook

This document explains:

1. what to run,
2. in which order,
3. which files/columns are produced,
4. how to validate that the run is correct.

It is the operational reference for the ingestion + processing pipeline.

---

## End-to-End Reproducible Workflow (Thesis)

### Phase 1: Data Engineering Pipeline

Purpose:
- fetch and merge source data,
- run refinement/pruning/transformation/feature engineering,
- produce the canonical artifact `data/features/all_data_features.parquet`.

Recommended terminal run:

```bash
./.venv/bin/python scripts/run_full_pipeline.py \
  --start 2020-11-30T23:00:00Z \
  --end 2026-03-01T02:00:00Z \
  --verify-lags
```

Phase-1 output:
- `data/features/all_data_features.parquet`

### Phase 2: Machine Learning Preparation (`data_prep.py`)

Purpose:
- construct leakage-safe `X`/`y`,
- apply strict chronological train/test split,
- switch preprocessing by model family (scaled vs unscaled).

Notebook usage example:

```python
import pandas as pd
from energy_trading.models.data_prep import prepare_model_data

df = pd.read_parquet("data/features/all_data_features.parquet")

# Tree-based models (no scaling)
X_train, X_test, y_train, y_test = prepare_model_data(
    df,
    target_col="target_afrr_activation_price_vwap_pos",
    model_type="xgboost",
    test_size=0.2,
)

# Linear / neural models (StandardScaler fit on train only)
X_train_lin, X_test_lin, y_train_lin, y_test_lin = prepare_model_data(
    df,
    target_col="target_afrr_activation_price_vwap_pos",
    model_type="linear",
    test_size=0.2,
)
```

Leakage guarantee in Phase 2:
- unlagged target columns are removed from `X` before model fitting,
- split is chronological (no shuffle), preventing look-ahead bias.

### Phase 2b: Create DA/aFRR ML Bundles

Purpose:
- generate reusable `train/val/test` parquet bundles for DA and aFRR tracks,
- export `feature_config.json` with exact `X`/`y` column mapping,
- fit and store preprocessing scaler on train split only.

```bash
./.venv/bin/python -m src.energy_trading.models.prepare_ml_bundles \
  --input data/features/all_data_features.parquet \
  --output-dir data/model_input \
  --doc-path docs/features_documentation.md \
  --scaler-out models/preprocessing/scaler.joblib
```

Notebook-Hinweis (Pfadstabilitaet):
- Notebooks sollten Artefakte immer relativ zu `REPO_ROOT` aufloesen
  (Ordner mit `src/`), nicht relativ zum aktuellen Arbeitsverzeichnis.
- Dadurch entstehen keine Pfad-Drifts zwischen Starts aus Repo-Root und
  `notebooks/`.

### Phase 3: Train + Export Versioned Model Run

Purpose:
- train DA and aFRR models into one versioned run artifact,
- export predictions (`val`, `test`) in canonical simulation schema,
- write `manifest.json` + `latest_<model>.json` for simulation autoload.

```bash
./.venv/bin/python scripts/train_and_export_runs.py \
  --base-dir data/model_input \
  --run-root artifacts/model_runs \
  --device cuda
```

Run outputs:
- `artifacts/model_runs/<run_id>/manifest.json`
- `artifacts/model_runs/<run_id>/models/*.joblib`
- `artifacts/model_runs/<run_id>/metrics/*.json`
- `artifacts/model_runs/<run_id>/predictions/*.parquet`
- `artifacts/model_runs/latest_<model>.json`

Download challenger artifacts from server to local machine:

```bash
scripts/pull_challenger_artifacts.sh \
  --ssh-host <ssh-host-alias-or-ip> \
  --ssh-user <ssh-user>
```

Notes:
- run this command on your local machine terminal, not in the remote server shell.
- prefer an SSH alias in `~/.ssh/config` (for example `uni-gpu`) to avoid repeatedly typing host/IP.

### Phase 3b: Final Thesis Repro Run (One Scripted Flow)

For the final supervisor handover, run the full sequence once and keep both
model run IDs (`xgboost` and `tft`) in text files.

```bash
set -euo pipefail

# 1) Build canonical features and bundles (includes latest target definitions)
./.venv/bin/python scripts/run_full_pipeline.py \
  --start 2020-11-30T23:00:00Z \
  --end 2026-03-01T02:00:00Z \
  --verify-lags

./.venv/bin/python -m src.energy_trading.models.prepare_ml_bundles \
  --input data/features/all_data_features.parquet \
  --output-dir data/model_input \
  --doc-path docs/features_documentation.md \
  --scaler-out models/preprocessing/scaler.joblib

# 2) Train XGBoost run (DA + aFRR)
./.venv/bin/python scripts/train_and_export_runs.py \
  --model-type xgboost \
  --base-dir data/model_input \
  --device cuda \
  --n-estimators 500 \
  --max-depth 6 \
  --learning-rate 0.05 \
  --early-stopping-rounds 50 \
  --forecast-horizon-hours 48 \
  --seed 42

./.venv/bin/python - <<'PY'
import json, pathlib
latest = json.loads(pathlib.Path("artifacts/model_runs/latest_tft.json").read_text())
run_id = latest["run_id"]
pathlib.Path("artifacts/model_runs/latest_xgboost_run_id.txt").write_text(run_id + "\n")
print("XGBoost run_id:", run_id)
PY

# 3) Train TFT run (DA + aFRR)
# Recommended for constrained server / shm issues: --num-workers 0
./.venv/bin/python scripts/train_and_export_runs.py \
  --model-type tft \
  --base-dir data/model_input \
  --device cuda \
  --forecast-horizon-hours 48 \
  --seed 42 \
  --num-workers 0

./.venv/bin/python - <<'PY'
import json, pathlib
latest = json.loads(pathlib.Path("artifacts/model_runs/latest_xgboost.json").read_text())
run_id = latest["run_id"]
pathlib.Path("artifacts/model_runs/latest_tft_run_id.txt").write_text(run_id + "\n")
print("TFT run_id:", run_id)
PY
```

Why this is the recommended "final" path:
- identical data snapshot and bundle schema for both model families,
- deterministic split policy from `prepare_ml_bundles`,
- explicit run IDs persisted for audit/reproduction.

When an aFRR-only rerun is acceptable:
- only for quick regression checks after aFRR-target changes,
- not as final thesis baseline comparison (for final results, rerun both
  model families from the same data/bundle state).

### Phase 4: Simulation from Run Manifest

Purpose:
- build central `backtest_table` from DA+aFRR prediction files (manifest split),
- optimize dispatch on predictions,
- settle on ground truth,
- report `Actual`, `Oracle`, and `Cost of Forecast Error`.

```bash
./.venv/bin/python scripts/run_battery_backtest.py \
  --run-manifest artifacts/model_runs/latest_xgboost.json \
  --split test \
  --horizon-hours 48 \
  --reopt-step-hours 1 \
  --da-gate-hour-utc 11 \
  --soc-feedback-mode realized
```

Optional explicit merge step (separate utility):

```bash
./.venv/bin/python scripts/merge_backtest_inputs.py \
  --run-manifest artifacts/model_runs/latest_xgboost.json \
  --split test
```

Backtest outputs:
- `artifacts/simulation_runs/<run_id>/<split>/backtest_table_<split>.parquet`
- `artifacts/simulation_runs/<run_id>/<split>/backtest_hourly.parquet`
- `artifacts/simulation_runs/<run_id>/<split>/backtest_plan_history.parquet`
- `artifacts/simulation_runs/<run_id>/<split>/backtest_decision_volatility.csv`
- `artifacts/simulation_runs/<run_id>/<split>/backtest_monthly.csv`
- `artifacts/simulation_runs/<run_id>/<split>/backtest_yearly.csv`
- `artifacts/simulation_runs/<run_id>/<split>/backtest_summary.json`
- `artifacts/backtest_plan_history.parquet` (global mirror)

CLI usage details (`scripts/run_battery_backtest.py`):
- `--run-manifest`: autoload DA+aFRR predictions + ground truth from a run.
- `--split {val,test}`: chooses prediction split from manifest.
- `--horizon-hours`: rolling MIP horizon length (default/recommended `48`).
- `--reopt-step-hours`: re-optimization cadence in hours (typically `1`).
- `--da-gate-hour-utc`: DA bid lock hour (default/recommended `11`).
- `--soc-feedback-mode {realized,predicted}`:
  `realized` is operationally realistic and recommended for backtests.
- `--start/--end`: optional UTC range filters.
- `--disable-rolling-horizon`: fallback full-horizon optimization.

Manual-file mode (without manifest):

```bash
./.venv/bin/python scripts/run_battery_backtest.py \
  --predictions artifacts/simulation_runs/manual/backtest_table_test.parquet \
  --ground-truth data/features/all_data_features.parquet \
  --timestamp-col timestamp_utc \
  --pred-da-col pred_da_price \
  --true-da-col da_price \
  --horizon-hours 48 \
  --reopt-step-hours 1
```

---

## 1) Pipeline Structure

Main scripts:

- `src/energy_trading/ingestion/collect_and_merge_all_data.py`  
  Orchestrator (runs all fetchers + merge).
- `src/energy_trading/ingestion/fetch_regelleistung.py`  
  Regelleistung aFRR CAPACITY/ENERGY + optional anonymous-bid derived prices.
- `scripts/fetch_entsoe_outages.py`  
  ENTSO-E planned/unplanned generation outage events (DE-LU).
- `scripts/transform_entsoe_outages_hourly.py`  
  Converts outage events to hourly outage features.
- `src/energy_trading/ingestion/merge_data.py`  
  Full join of all raw parquet files on hourly UTC timestamps + outage merge.
- `src/energy_trading/processing/refine_market_data.py`  
  Refines merged data, consolidates sources, and applies market logic.

Raw outputs are written to `data/raw/*.parquet`.  
Merged table is written to `data/processed/all_data.parquet`.  
Refined table is written to `data/processed/all_data_refined.parquet`.  
Cleaned table is written to `data/processed/all_data_cleaned.parquet`.  
Pruned table is written to `data/processed/all_data_pruned.parquet`.  
Transformed table is written to `data/processed/all_data_transformed.parquet`.  
Final features are written to `data/features/all_data_features.parquet`.

### Step-by-step I/O map

| Step | Script | Main Input(s) | Main Output |
|---|---|---|---|
| 1 | `fetch_entsoe.py` | ENTSO-E API (`--start`, `--end`) | `data/raw/entsoe.parquet` |
| 2 | `fetch_energy_charts.py` | Energy-Charts API | `data/raw/energy_charts.parquet` |
| 3 | `fetch_netztransparenz.py` | Netztransparenz API | `data/raw/netztransparenz.parquet` (15-min activation volumes) |
| 4 | `fetch_smard.py` | SMARD API (+ optional market-data CSV) | `data/raw/smard.parquet` (+ `data/raw/installed_capacity.csv`) |
| 5 | `fetch_yfinance.py` | Yahoo Finance API | `data/raw/yfinance.parquet` |
| 6 | `fetch_regelleistung.py` | Regelleistung API + bid files in `data/raw/bids/` | `data/raw/regelleistung.parquet` (hourly) + `data/raw/regelleistung_15min/afrr_prices_15min.parquet` + `data/raw/regelleistung_15min/afrr_volumes_15min.parquet` + `data/raw/regelleistung_15min/afrr_price_volume_15min.parquet` |
| 7 | `fetch_entsoe_outages.py` | ENTSO-E API (`A80` + `A53/A54`) | `data/raw/entsoe_outages/planned_generation_outages.parquet` + `data/raw/entsoe_outages/unplanned_generation_outages.parquet` |
| 8 | `transform_entsoe_outages_hourly.py` | `data/raw/entsoe_outages/planned_generation_outages.parquet` + `data/raw/entsoe_outages/unplanned_generation_outages.parquet` | `data/processed/outages_hourly.parquet` |
| 9 | `merge_data.py` | All `data/raw/*.parquet` files + `data/processed/outages_hourly.parquet` | `data/processed/all_data.parquet` |
| 10 | `refine_market_data.py` | `data/processed/all_data.parquet` | `data/processed/all_data_refined.parquet` |
| 11 | `handle_missing_values.py` | `data/processed/all_data_refined.parquet` | `data/processed/all_data_cleaned.parquet` |
| 12 | `drop_redundant_features.py` | `data/processed/all_data_cleaned.parquet` | `data/processed/all_data_pruned.parquet` |
| 13 | `transform_data.py` | `data/processed/all_data_pruned.parquet` | `data/processed/all_data_transformed.parquet` |
| 14 | `build_features.py` | `data/processed/all_data_transformed.parquet` | `data/features/all_data_features.parquet` |

Note: The canonical features output path is `data/features/all_data_features.parquet`.
Do not maintain a duplicate `data/processed/all_data_features.parquet`.

---

## Design Principles

- Separation of concerns:
  Fetchers mirror raw API data and avoid domain-specific transformations.
- Market logic in processing:
  Processing scripts implement source consolidation, structural-break logic,
  and feature engineering in a traceable layer.
- Reproducibility:
  Intermediate artifacts (`all_data.parquet`, `all_data_refined.parquet`,
  `all_data_cleaned.parquet`,
  `all_data_pruned.parquet`, `all_data_transformed.parquet`) make every
  transformation stage auditable.

## 2) Canonical Command (Recommended)

Run everything end-to-end:

```bash
./.venv/bin/python scripts/collect_and_merge_all_data.py \
  --start 2020-11-30T23:00:00Z \
  --end 2026-03-01T02:00:00Z
```

Then fetch outage events, build hourly outages, and re-run merge so outages are
embedded in `all_data.parquet`:

```bash
./.venv/bin/python scripts/fetch_entsoe_outages.py --days-ahead 7
./.venv/bin/python scripts/transform_entsoe_outages_hourly.py --days-ahead 7
./.venv/bin/python -m energy_trading.ingestion.merge_data \
  --data-dir data/raw \
  --out data/processed/all_data.parquet \
  --clip-start 2020-11-30T23:00:00Z \
  --clip-end 2026-03-01T02:00:00Z
```

This corresponds to:

- Start: `2020-12-01 00:00:00 CET`
- End: `2026-03-01 03:00:00 CET`

Important behavior:

- The collector fetches with a one-day lookback (`start - 1 day`) to reduce boundary losses.
- `merge_data.py` clips back to the exact `--start/--end` window.

---

## 3) Useful Variants

Run only post-collection processing (if `all_data.parquet` already exists):

```bash
./.venv/bin/python scripts/post_collection_pipeline.py \
  --input data/processed/all_data.parquet
```

Skip anonymous-bid activation price reconstruction (faster):

```bash
./.venv/bin/python scripts/collect_and_merge_all_data.py \
  --start 2020-11-30T23:00:00Z \
  --end 2026-03-01T02:00:00Z \
  --skip-bid-activation-prices
```

Run Regelleistung only:

```bash
./.venv/bin/python -m energy_trading.ingestion.fetch_regelleistung \
  --start 2020-11-30T23:00:00Z \
  --end 2026-03-01T02:00:00Z \
  --out data/raw/regelleistung.parquet \
  --bids-dir data/raw/bids
```

Run merge only:

```bash
./.venv/bin/python -m energy_trading.ingestion.merge_data \
  --data-dir data/raw \
  --out data/processed/all_data.parquet \
  --clip-start 2020-11-30T23:00:00Z \
  --clip-end 2026-03-01T02:00:00Z
```

Run strict linear processing chain manually:

```bash
./.venv/bin/python -m energy_trading.processing.refine_market_data \
  --in data/processed/all_data.parquet \
  --out data/processed/all_data_refined.parquet

./.venv/bin/python -m energy_trading.processing.handle_missing_values \
  --in data/processed/all_data_refined.parquet \
  --out data/processed/all_data_cleaned.parquet

./.venv/bin/python -m energy_trading.processing.drop_redundant_features \
  --in data/processed/all_data_cleaned.parquet \
  --out data/processed/all_data_pruned.parquet

./.venv/bin/python -m energy_trading.processing.transform_data \
  --in data/processed/all_data_pruned.parquet \
  --out data/processed/all_data_transformed.parquet

./.venv/bin/python -m energy_trading.features.build_features \
  --in data/processed/all_data_transformed.parquet \
  --out data/features/all_data_features.parquet
```

Missingness note:
- `handle_missing_values.py` applies category-based imputation first.
- Commodity prices `co2_price`, `gas_price`, `coal_price` have explicit special
  rules: unlimited `ffill()` plus leading-gap `bfill()` to ensure no startup
  NaNs remain.
- Capacity columns (`*_capacity`) use unlimited `ffill()` due to their
  structural/slow-moving nature.
- `da_price_BE` uses a deterministic fallback from the previous day at the same
  UTC hour (`t-24h`) for missing timestamps.
- Remaining leading NaNs before first quote availability are intentionally kept
  at this stage and later resolved in bundle creation via train-fitted median
  fallback (`prepare_ml_bundles.py`).

---

## 4) Timestamp and Timezone Rules

- Command-line bounds should be passed as explicit UTC ISO strings (`...Z`).
- `merge_data.py` interprets naive clip timestamps as `Europe/Berlin`; avoid that and always pass `Z`.
- Final merged file contains:
  - `timestamp_utc`
  - `timestamp_cet`
- Internal join key is hourly UTC.

---

## 5) Activation Price Columns (What They Mean)

Activation prices (work prices) are sourced exclusively from
`regelleistung.net` (directly and bid-derived VWAP variants). ENTSO-E CBMP is
not part of the production pipeline.

From Regelleistung TSO aggregation:

- `afrr_avg_activation_price_pos`
- `afrr_avg_activation_price_neg`
- `afrr_activation_price_vwap_pos`
- `afrr_activation_price_vwap_neg`

From anonymous bids (derived in `fetch_regelleistung.py`):

- `afrr_bid_vwap_activation_price_pos`
- `afrr_bid_vwap_activation_price_neg`
- `afrr_bid_avg_activation_price_pos`
- `afrr_bid_avg_activation_price_neg`
- `bid_alloc_mw_pos`
- `bid_alloc_mw_neg`

Implementation note:

- `afrr_bid_vwap_activation_price_*` is volume-weighted from accepted anonymous bids.
- `afrr_bid_avg_activation_price_*` is currently derived as an extremum proxy on signed bid prices
  (`POS -> max`, `NEG -> min`) after payment-direction sign handling.
- Primary model work-price VWAP is computed from 15-minute Regelleistung price/volume
  exports:
  `data/raw/regelleistung_15min/afrr_price_volume_15min.parquet`.
- VWAP is calculated in two stages:
  1. `fetch_regelleistung.py` (hourly export convenience),
  2. `refine_market_data.py` (processing-stage recomputation/join for canonical use).

---

## 6) Validation Checklist (After Every Full Run)

### A) Basic row/time checks

Confirm merged output exists and has expected time window:

```bash
./.venv/bin/python - <<'PY'
import polars as pl
df = pl.read_parquet("data/processed/all_data.parquet")
print("rows:", df.height)
print("start_utc:", df["timestamp_utc"].min())
print("end_utc:", df["timestamp_utc"].max())
PY
```

### A2) Shape lineage across all major artifacts (rows x cols)

This should be documented in the operational runbook (this file), because it is
part of reproducibility/audit evidence for every pipeline run.

Recommended export artifact per run:
- `data/reports/pipeline_shape_lineage.csv`

```bash
./.venv/bin/python - <<'PY'
from pathlib import Path
import polars as pl

stages = [
    ("raw_regelleistung", "data/raw/regelleistung.parquet"),
    ("raw_entsoe", "data/raw/entsoe.parquet"),
    ("raw_energy_charts", "data/raw/energy_charts.parquet"),
    ("raw_netztransparenz", "data/raw/netztransparenz.parquet"),
    ("raw_smard", "data/raw/smard.parquet"),
    ("raw_yfinance", "data/raw/yfinance.parquet"),
    ("processed_all_data", "data/processed/all_data.parquet"),
    ("processed_refined", "data/processed/all_data_refined.parquet"),
    ("processed_cleaned", "data/processed/all_data_cleaned.parquet"),
    ("processed_pruned", "data/processed/all_data_pruned.parquet"),
    ("processed_transformed", "data/processed/all_data_transformed.parquet"),
    ("features_all", "data/features/all_data_features.parquet"),
    ("bundle_da_X_train", "data/model_input/da/train_X.parquet"),
    ("bundle_da_y_train", "data/model_input/da/train_y.parquet"),
    ("bundle_afrr_X_train", "data/model_input/afrr/train_X.parquet"),
    ("bundle_afrr_y_train", "data/model_input/afrr/train_y.parquet"),
]

def _ts_bounds(df: pl.DataFrame) -> tuple[str, str]:
    for c in ("timestamp_utc", "snapshot_time", "target_time"):
        if c in df.columns:
            return str(df[c].min()), str(df[c].max())
    return "", ""

rows = []
for stage, p in stages:
    path = Path(p)
    if not path.exists():
        rows.append({
            "stage": stage,
            "path": p,
            "exists": False,
            "rows": None,
            "cols": None,
            "min_time": "",
            "max_time": "",
        })
        continue
    df = pl.read_parquet(path)
    tmin, tmax = _ts_bounds(df)
    rows.append({
        "stage": stage,
        "path": p,
        "exists": True,
        "rows": df.height,
        "cols": len(df.columns),
        "min_time": tmin,
        "max_time": tmax,
    })

out = pl.DataFrame(rows)
out_path = Path("data/reports/pipeline_shape_lineage.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
out.write_csv(out_path)
print(out)
print("\\nwritten:", out_path)
PY
```

Interpretation guideline:
- row drops are expected after clipping, null-handling, and train/val/test
  slicing,
- column drops are expected after pruning and leakage-safe `X` construction,
- any unexpected increase/decrease should be explained in the run notes.

### B) Null count for activation columns

```bash
./.venv/bin/python - <<'PY'
import polars as pl
cols = [
    "afrr_avg_activation_price_pos",
    "afrr_avg_activation_price_neg",
    "afrr_bid_vwap_activation_price_pos",
    "afrr_bid_vwap_activation_price_neg",
    "afrr_bid_avg_activation_price_pos",
    "afrr_bid_avg_activation_price_neg",
]
df = pl.read_parquet("data/processed/all_data.parquet")
for c in cols:
    if c in df.columns:
        print(c, int(df[c].null_count()))
    else:
        print(c, "MISSING")
PY
```

### C) Where nulls occur (time ranges)

```bash
./.venv/bin/python - <<'PY'
import polars as pl
target = "afrr_bid_vwap_activation_price_pos"
df = pl.read_parquet("data/processed/all_data.parquet")
if target in df.columns:
    n = df.filter(pl.col(target).is_null())
    print("null_rows:", n.height)
    if n.height:
        print("first_null:", n["timestamp_utc"].min())
        print("last_null:", n["timestamp_utc"].max())
else:
    print("missing column:", target)
PY
```

---

## 7) Common Failure Signatures

- `Missing columns ... bid_vwap ...`  
  Usually stale notebook logic vs renamed columns. Use the names listed in section 5.

- Very long Regelleistung run  
  First run parses all bid files; subsequent runs should be faster due to
  `data/raw/bids/_afrr_bid_hourly_cache.parquet`.

- `Bid cache schema mismatch` warning  
  Indicates cache invalidation after code/schema changes. One slow rebuild is expected.

- Sparse coverage in specific years/months  
  Check logs for yearly-file fallback to monthly files and missing monthly files.

---

## 8) Documentation Best Practices for Thesis Artifacts

Include:

- exact command used,
- git commit hash,
- exact timeframe (UTC + CET explanation),
- column definitions used in analysis,
- number of rows and nulls for key columns.

Exclude:

- raw debug dumps,
- temporary aliases/legacy column names if not used in final results,
- partial experiments without final interpretation.
