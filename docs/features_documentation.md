# Feature Documentation and Data Dictionary

## Table of Contents

- [1. Scope and Snapshot](#1-scope-and-snapshot)
  - [1.1 Introduction](#11-introduction)
  - [1.2 Reproducibility Block](#12-reproducibility-block)
- [2. Feature Space](#2-feature-space)
  - [2.1 Current Feature Concept](#21-current-feature-concept)
  - [2.2 Considered and Removed Features](#22-considered-and-removed-features)
  <!-- - [2.3 Metadata and Excluded Columns](#23-metadata-and-excluded-columns) -->
- [3. Design Decisions](#3-design-decisions)
  - [3.1 Runtime Dynamics Features During Training](#31-runtime-dynamics-features-during-training)
  - [3.2 No-PCA Default Despite Collinearity](#32-no-pca-default-despite-collinearity)
  - [3.3 No Global Scaling for Tree Models](#33-no-global-scaling-for-tree-models)
  - [3.4 Cyclical Encoding of Time Features](#34-cyclical-encoding-of-time-features)
  - [3.5 Model Separation (DA vs aFRR)](#35-model-separation-da-vs-afrr)
- [4. Data Quality and Causality Controls](#4-data-quality-and-causality-controls)
  - [4.1 Publication-Latency Rules (PiT)](#41-publication-latency-rules-pit)
  - [4.2 Data Imputation (Methodological Rationale)](#42-data-imputation-methodological-rationale)
  - [4.3 Observed Imputations (Latest Bundle Run)](#43-observed-imputations-latest-bundle-run)
- [5. Specific Feature Logic and Rationale](#specific-feature-logic-and-rationale)
  - [5.1 Targeted Additional Intraday Lags (Update)](#targeted-additional-intraday-lags-update)
- [6. Compact Data Dictionary (Feature Families)](#compact-data-dictionary-feature-families)
- [7. Causality Check for Lag Naming](#causality-check-for-lag-naming)
- [8. Validation Methodology](#validierungsmethodik)
- [9. Methodological Governance and Evidence](#methodische-governance-und-evidenz)
- [10. Primary-Source Gaps and April-2025 Forensics](#primary-source-gaps-and-april-2025-forensics)
- [11. Empirical Validity](#empirical-validity)
- [12. Parameter Rationale](#parameter-rationale)

## 1. Scope and Snapshot

### 1.1 Introduction

This document describes the final, causally safe feature space used for aFRR and DA forecasting.

- Final artifact file: `data/features/all_data_features.parquet`
- Current snapshot (2026-05-14): **364 columns** in the feature artifact.
- Training features `X` are built **model-specifically**:
  - **DA bundle:** reduced, auction-causal set (fundamental signals), see [DA train bundle](../data/model_input/da/train.parquet)
  - **aFRR bundle:** extended set with stress/spread signals (short-term signals), see [aFRR train bundle](../data/model_input/afrr/train.parquet)
- `timestamp_utc` is **metadata/time index** and not counted as a training feature.

### 1.2 Reproducibility Block

| Element                           | Value                                     |
| --------------------------------- | ----------------------------------------- |
| **Snapshot date**                 | 2026-05-14                                |
| **Feature artifact**              | `data/features/all_data_features.parquet` |
| **Artifact shape (snapshot)**     | `46,009` rows, `364` columns              |
| **Bundle configuration**          | `data/model_input/feature_config.json`    |
| **DA feature count (snapshot)**   | `121`                                     |
| **aFRR feature count (snapshot)** | `342`                                     |
| **Training data end (UTC)**       | `2024-06-30T23:00:00+00:00`               |
| **Regime cut**                    | `2022-06-22 22:00:00+00:00`               |
| **DA gate**                       | `D-1 13:00 UTC` (`da_price_pit`)          |
| **Finalized at (UTC)**            | `2026-05-14T09:42:51+00:00`               |

**Canonical rebuild commands:**

```bash
./.venv/bin/python -m energy_trading.ingestion.merge_data \
  --data-dir data/raw \
  --out data/processed/all_data.parquet \
  --clip-start 2020-11-30T23:00:00Z \
  --clip-end 2026-03-25T23:00:00Z

./.venv/bin/python -m scripts.post_collection_pipeline \
  --input data/processed/all_data.parquet

./.venv/bin/python -m src.energy_trading.models.prepare_ml_bundles \
  --input data/features/all_data_features.parquet \
  --output-dir data/model_input
```

Shape-lineage note:

- End-to-end `rows x cols` lineage (raw -> processed -> features -> bundles) is tracked in `docs/pipeline_runbook.md`.
- Recommended run artifact: `data/reports/pipeline_shape_lineage.csv`.

## 2. Feature Space

### 2.1 Current Feature Concept

- The feature system is split by decision context (DA vs aFRR).
- The DA bundle focuses on ex-ante fundamental signals available at D-1 gate.
- The aFRR bundle extends the DA base with short-horizon balancing and stress context.
- Static bundle definitions live in `data/model_input/feature_config.json`.
- Additional training-time transformations can be applied later in model scripts.
- `data/model_input/feature_config.json` defines **static** bundle selection.
- Additional runtime transformations (dynamics features and mid-term-lag pruning) happen later in training.

### 2.2 Considered and Removed Features

| Status                        | Scope         | Feature / Pattern                                                                                                                                                      | Rule source                                                           | Rationale                                                                                |
| ----------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **Removed**                   | DA bundle     | `afrr_*`, `mfrr_*`, `nrv_*`, `rz_saldo_*`, `picasso_*`, `mari_*`, `is_activated_*`, `system_stress_*`, `grid_stress_*`, `scarcity_*`, `nrv_zscore_*`, `nrv_quantile_*` | `prepare_ml_bundles.py:get_da_optimized_features`                     | D-1 auction causality: near-real-time balancing/stress signals are not ex-ante DA-valid. |
| **Removed**                   | DA bundle     | short lags `*_lag_(1,2,3,6,12)h`                                                                                                                                       | `prepare_ml_bundles.py:get_da_optimized_features`                     | Avoid near-term non-causal information in DA path.                                       |
| **Removed**                   | DA bundle     | `da_spread_*` without lag and `da_spread_*_lag_<24h`                                                                                                                   | `prepare_ml_bundles.py:get_da_optimized_features`                     | Bilateral spreads in DA path are allowed only as day-seasonal memory (`>=24h`).          |
| **Removed**                   | DA bundle     | `total_wind_solar_id_error*`                                                                                                                                           | `prepare_ml_bundles.py:get_da_optimized_features`                     | Intraday error indicators are excluded from DA feature set.                              |
| **Removed (runtime)**         | Training      | `*_lag_4h`, `*_lag_6h` (when dynamics active)                                                                                                                          | `train_xgboost_export.py:_add_dynamics_features`                      | Reduce redundant mid-term lags when 1h momentum/ramp terms exist.                        |
| **Removed (legacy)**          | Feature layer | columns containing `reconstructed` or `grid_share` (if present)                                                                                                        | project cleanup/audit tooling                                         | Remove deprecated feature families if they appear in upstream datasets.                  |
| **Optional (off by default)** | Bundle build  | PCA on forecast families, optional raw-drop                                                                                                                            | `prepare_ml_bundles.py` (`use_forecast_pca`, `forecast_pca_drop_raw`) | Available but disabled by default until robust CV/PnL gains are proven.                  |
| **Ablation only**             | aFRR modeling | groups like `cross_border`, `hydro_pumped`, `load_error`, `picasso_flow`, `orderbook_depth`                                                                            | `scripts/run_feature_ablation.py`                                     | Group removal tested empirically; not enforced globally by default.                      |

<!-- ### 2.3 Metadata and Excluded Columns

| Type                                       | Columns                                                                                                                                                                                                                                       |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Metadata/index                             | `timestamp_utc`                                                                                                                                                                                                                               |
| Technical metadata (excluded from `X`)     | `data_is_lagged`, `is_local_reconstruction_only`, `pit_lagged_column_count`                                                                                                                                                                   |
| Target/outcome columns (excluded from `X`) | `target_afrr_activation_price_vwap_pos`, `target_afrr_activation_price_vwap_neg`, `target_afrr_activation_rate_pos`, `target_afrr_activation_rate_neg`, `target_afrr_capacity_price_pos`, `target_afrr_capacity_price_neg`, `target_da_price` |

Note: `target_afrr_capacity_price_pos` and `target_afrr_capacity_price_neg` are strictly label columns (`y`) and are excluded from `X`. -->

## 3. Design Decisions

### 3.1 Runtime Dynamics Features During Training

In addition to static bundle features from `data/model_input/feature_config.json`, the training pipeline (`src/energy_trading/models/train_xgboost_export.py`) creates short-term runtime features on-the-fly:

| Feature                                      | Formula / Construction                                                               | Unit | Purpose                                                    |
| -------------------------------------------- | ------------------------------------------------------------------------------------ | ---- | ---------------------------------------------------------- |
| `nrv_velocity_1h`                            | `NRV_balance_lag_2h - NRV_balance_lag_3h`                                            | MW   | Short-term imbalance momentum approximation.               |
| `load_ramp_signed_1h`                        | preferred `load_forecast_da_entsoe_h1 - load_forecast_da_entsoe` (fallback: `h2-h1`) | MW   | Direction and strength of load ramp.                       |
| `load_ramp_abs_1h`                           | `abs(load_ramp_signed_1h)`                                                           | MW   | Ramp intensity independent of direction.                   |
| `res_load_ramp_signed_1h`                    | preferred `residual_load_forecast_h1 - residual_load_forecast` (fallback: `h2-h1`)   | MW   | Short-term residual-load pressure signal.                  |
| `res_load_ramp_x_wind_total_error_da_lag_2h` | `res_load_ramp_signed_1h * wind_total_error_da_lag_2h`                               | MW²  | Interaction feature for ramp stress and recent wind error. |

Notes:

- These are **runtime model features** and do not necessarily appear as physical columns in static bundle parquet files.
- If dynamics features are active, redundant mid-term lags (`*_lag_4h`, `*_lag_6h`) can be dropped during training.

### 3.2 No-PCA Default Despite Collinearity

- Reference: `notebooks/13_forecast_collinearity_pca_audit.ipynb`.
- Audit result: high within-family collinearity (especially solar; e.g., high VIF values).
- **Decision:** no PCA in the default final training path.
- **Why:**
  1. XGBoost is relatively robust to collinear inputs.
  2. Physical interpretability of raw features is preserved.

### 3.3 No Global Scaling for Tree Models

- **Decision:** no global feature scaling in final XGBoost training.
- **Why:** tree split logic is mostly invariant to monotonic scaling.
- **Benefit:** better interpretability in physical units (`MW`, `EUR/MWh`).

### 3.4 Cyclical Encoding of Time Features

- **Features:** `hour_sin/cos`, `dayofweek_sin/cos`, `month_sin/cos`.
- **Why:** preserves circular continuity.
- **Benefit:** avoids artificial discontinuity at period boundaries (e.g., hour 23 -> 00).

### 3.5 Model Separation (DA vs aFRR)

- **DA set:** strict D-1 auction-causal features only (fundamental ex-ante drivers).
- **aFRR set:** DA base plus high-frequency stress/flow/spread signals.
- **Purpose:** stronger causal consistency per decision time and better market-specific fit.

| Feature set               | Content                                                                | Exclusions / Additions                               |
| ------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------- |
| **DA model set**          | Fundamental forecasts, commodity prices, calendar/seasonality features | Excludes balancing/stress signals by D-1 causality   |
| **aFRR model set (full)** | DA set plus high-frequency stress/flow/spread signals                  | Designed to model short-term deviation from DA price |

## 4. Data Quality and Causality Controls

### 4.1 Publication-Latency Rules (PiT)

| Data group                                                      | Causal rule                                                         |
| --------------------------------------------------------------- | ------------------------------------------------------------------- |
| ENTSO-E actuals (generation/load/outages/NRV)                   | at least `lag_2h`                                                   |
| aFRR/mFRR activation and activation prices                      | at least `lag_1h`                                                   |
| Grid stress stats (`system_stress_signal`, `grid_stress_index`) | only `lag_2h`, `lag_3h`, `lag_6h`, `lag_12h`, `lag_24h`             |
| Day-ahead price                                                 | `da_price_pit` with D-1 13:00 UTC release gating                    |
| DA forecast horizons (`*_h1..h24`)                              | publication-gated (D-1 13:00 UTC), else causal fallback `shift(24)` |
| Forecast signals (general)                                      | ex-ante usable; no non-causal future fill                           |

### 4.2 Data Imputation (Methodological Rationale)

- To avoid artificial sparsity, installed capacities (`*_capacity`) are backfilled in feature construction.
- First available reported capacity values (typically from late 2023) are propagated backwards at least to PICASSO start (`2022-06-22 22:00:00+00:00`).
- This is methodologically acceptable because installed capacities change slowly and hourly market dynamics are modeled via actual and activation series.
- For `generation_baseload_total`:
  `generation_baseload_total = biomass_actual_entsoe + generation_nuclear_mw`
  (missing `generation_nuclear_mw` values set to `0`).
- Since German nuclear phase-out, the feature is effectively biomass-dominated in recent periods.

### 4.3 Observed Imputations (Latest Bundle Run)

`prepare_ml_bundles.py` imputes `X` per split via `ffill(limit=12)` and then train-fitted median fallback.
Additional upstream special cases in `handle_missing_values.py`:

- Commodity prices (`co2_price`, `gas_price`, `coal_price`): unlimited `ffill()` plus `bfill()` for leading gaps.
- Structural capacities (`*_capacity`): unlimited `ffill()`.
- `da_price_BE`: fallback to previous-day same UTC hour (`t-24h`).

Affected columns in the table below are **illustrative from a recent run pattern** and may differ in your next run:

| Bundle / Split | Affected columns (excerpt)                                                                                                                               | Method                                               |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| DA / test      | `da_price_BE`, `gas_price`, `co2_price`                                                                                                                  | `ffill(limit=12)` + train median fallback            |
| DA / test      | `wind_onshore_capacity`, `wind_offshore_capacity`, `solar_capacity`, `gas_capacity`, `hard_coal_capacity`, `lignite_capacity`, `pumped_storage_capacity` | mostly `ffill(limit=12)`, little/no median fallback  |
| aFRR / train   | `wind_total_error_da_lag_2h`                                                                                                                             | minor `ffill`, no median fallback                    |
| aFRR / test    | DA commodity/capacity columns plus activation/capacity lag features                                                                                      | mostly short-gap `ffill`, usually no median fallback |

The **authoritative, up-to-date** imputation evidence is always the current report files:

- `data/model_input/da/feature_quality_report.csv`
- `data/model_input/afrr/feature_quality_report.csv`
- `data/reports/feature_quality_report_all.csv`

---

## Specific Feature Logic and Rationale

| **Feature Class**                                                             | **Rationale**                                                                                                                          | **Purpose**                                                                                                                  |
| ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Symmetric Log Transformation** (`da_price_slog1p`)                           | Compression of extreme price spikes while preserving the sign.                                                                           | Stabilizes XGBoost gradient behavior and reduces outlier noise.                                    |
| **Pumped-Hydro Isolation** (`generation_hydro_pumped_storage_mw_lag_2h`)      | Separates price-reactive flexibility (pumped storage) from inflexible run-of-river/aggregate hydro components to avoid signal dilution. | Detects tactical flexibility activation; high output signals boundary-capacity usage and increased aFRR volatility. |
| **Renewable-Share-Forecast** (`renewable_share_forecast`)                      | Measures the relative displacement of conventional flexibility in the merit order; more informative than absolute single-MW values.              | Robust indicator for merit-order shifts and flexibility demand.                                                 |
| **EWMA Derivatives** (`*_ewma24`)                                                 | Heavier weighting of recent price information relative to older observations.                                                       | Faster response to market momentum and regime shifts in intraday conditions.                                                |
| **DA Forecast Curve Features** (`*_h1,_h2,_h3,_h6,_h12,_h24`)                  | Explicit representation of the expected forecast trajectory for the next day instead of only point-based ex-ante values.                     | Improves multi-horizon forecasts by exposing trajectories, ramps, and daily profiles within the forecast window.            |
| **Compressed Forecast Curves** (`*_next24_*`)                                | Compresses the 24h trajectory into level/dispersion/ramp statistics (`mean/min/max/std`, `ramp`) to reduce dimensionality.     | Robust capture of trend and volatility patterns with controlled model complexity (especially for tree models).        |
| **Load-Error-Feature** (`load_error_da_lag_2h`)                                | Unexpected load fluctuations are primary drivers of short-term system imbalances.                                          | Direct predictor for balancing-energy activation (aFRR).                                                            |
| **Cross-Border-Spreads** (`da_spread_de_at/de_fr/de_nl`, incl. Lags)           | Represents import/export pressure and coupling intensity of neighboring day-ahead markets.                                                     | Additional explanatory power for price and spread regimes via cross-border arbitrage/congestion signals.             |
| **PICASSO Regime Flag** (`is_picasso_active`)                                  | Represents the active PICASSO market phase (`since July 2024`) as a structural regime anchor.                                               | Allows the model to distinguish pre-PICASSO from PICASSO-coupled European pricing logic.          |
| **PiT latency rule for ENTSO-E actuals** (`*_lag_2h`)                           | Physical actual values are not stably published immediately at decision time.                                                     | Avoids information leakage through consistent causal lagging.                                                  |
| **DA gate logic** (`da_price_pit`)                                             | Day-ahead auction results are only available after the D-1 gate (`13:00 UTC`).                                                      | Causally correct representation of information availability for DA and aFRR-related features.                                    |
| **Stress signals without `lag_1h`** (`system_stress_signal`, `grid_stress_index`) | These indicators are based on actual-/NRV-near sources with additional publication latency.                                           | Robust PiT integrity by starting at `lag_2h` (instead of short-term leakage-prone `lag_1h`).                           |
| **Strict Target Policy** (`target_*`)                                          | Unshifted outcome series would enforce nowcasting instead of forecasting.                                                            | Separation of `X` and `y` along the time axis; the forecast target remains strictly `t+1`.                                          |

### Targeted Additional Intraday Lags (Update)

| Family / Columns                                                                                         | New or extended lags  | Rationale (short)                                                                                  |
| --------------------------------------------------------------------------------------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------- |
| `NRV_balance`, `nrv_zscore_24h`, `nrv_quantile_5`                                                         | `2h, 3h, 4h, 6h, 12h, 24h` | Short-term stress dynamics (ramps/decay) are captured better; no `1h` due to PiT latency.     |
| `afrr_activation_rate_pos`, `afrr_activation_rate_neg`, `is_activated`, `mfrr_active_lag`                 | `1h, 2h, 3h, 6h, 12h, 24h` | Activation intensity shows pronounced intraday momentum clusters, separated by direction for POS/NEG. |
| `afrr_activated_mw_pos/neg`, `mfrr_activated_mw_pos/neg`, `mfrr_mari_net_mw`, `afrr_activation_offered_*` | `1h, 2h, 3h, 6h, 12h, 24h` | Volume and flow trajectories are short-term persistent and relevant for aFRR price/rate forecasts. |
| `afrr_capacity_awarded_*`, `afrr_capacity_offered_*`, `afrr_capacity_price_*`                             | `1h, 2h, 3h, 6h, 12h, 24h` | Auction outcomes and capacity-market tightness typically persist for multiple hours.     |
| `wind_forecast_update`, `wind_onshore_forecast_update`, `solar_forecast_update`, `*_error_da`             | `1h, 2h, 3h, 6h, 12h, 24h` | Forecast “news” signals affect not only the current hour but also the following day.     |

## Compact Data Dictionary (Feature Families)

Completeness note: The feature families below cover the full final artifact scope of **356 columns** (including target columns, metadata/governance variables, and model-relevant feature groups). The presentation is intentionally grouped, not listed row-by-row per single column.

| Feature (available lags)                                                                                                                                                                                                                                                                   | Unit      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `afrr_activation_price_vwap_pos` (1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h)                                                                                                                                                                                                                      | EUR/MWh      | Positive aFRR activation price (VWAP) as a core signal for short-term price regimes and weekly patterns.                                                                                                                                                                                                                                                                                                                                                                                            |
| `afrr_activation_price_vwap_neg` (1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h)                                                                                                                                                                                                                      | EUR/MWh      | Negative aFRR activation price (VWAP); captures asymmetric balancing costs vs POS prices.                                                                                                                                                                                                                                                                                                                                                                                               |
| `afrr_da_price_spread` (1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h)                                                                                                                                                                                                                                | EUR/MWh      | Difference between aFRR activation price and day-ahead price as a direct opportunity-cost indicator.                                                                                                                                                                                                                                                                                                                                                                                              |
| `da_price_pit` (no lag, 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h)                                                                                                                                                                                                                              | EUR/MWh      | PiT-gated day-ahead price that is only model-available after publication time.                                                                                                                                                                                                                                                                                                                                                                                                   |
| `da_price` (24h, 48h, 168h)                                                                                                                                                                                                                                                                 | EUR/MWh      | Historical Day-Ahead-Preisniveaus zur Erfassung stabiler Tages- und Wochenzyklen.                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `gas_price`, `coal_price`, `co2_price` (no lag)                                                                                                                                                                                                                                           | EUR/MWh      | Brennstoff- und Emissionskosten als exogene Kostentreiber der Merit-Order und Strompreisdynamik.                                                                                                                                                                                                                                                                                                                                                                                                       |
| `da_price_slog1p`, `da_price_diff1`, `da_price_diff24`, `da_price_ewma24` (no lag)                                                                                                                                                                                                        | EUR/MWh      | Transformed price levels and changes for robust representation of jumps and short-term trends.                                                                                                                                                                                                                                                                                                                                                                                             |
| `da_price_mean_24h`, `da_price_std_24h`, `da_price_mean_168h`, `da_price_std_168h`, `da_price_volatility_30d` (no lag)                                                                                                                                                                    | EUR/MWh      | Rolling mean/volatility measures to quantify market regime shifts and uncertainty.                                                                                                                                                                                                                                                                                                                                                                                                     |
| `system_stress_signal` (2h, 3h, 6h, 12h, 24h)                                                                                                                                                                                                                                               | Index        | Compressed stress signal from system imbalances; usable only with 2h+ latency per PiT rule.                                                                                                                                                                                                                                                                                                                                                                                                     |
| `grid_stress_index` (2h, 3h, 6h, 12h, 24h)                                                                                                                                                                                                                                                  | Index        | Composite grid-stress index combining multiple stress components into a robust control indicator.                                                                                                                                                                                                                                                                                                                                                                                         |
| `nrv_zscore_24h` (2h, 3h, 4h, 6h, 12h, 24h)                                                                                                                                                                                                                                                 | Index        | Standardized NRV deviation vs the 24h path to identify unusual balance states.                                                                                                                                                                                                                                                                                                                                                                                       |
| `nrv_zscore_24h_lag_2h`                                                                                                                                                                                                                                                                     | Index        | PiT-compliant short-term indicator for acute grid stress and next-hour balancing probability.                                                                                                                                                                                                                                                                                                                                                                                         |
| `nrv_quantile_5` (2h, 3h, 4h, 6h, 12h, 24h)                                                                                                                                                                                                                                                 | Index        | Quantized NRV state (5 classes) for robust regime coding even with outliers.                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `NRV_balance` (2h, 3h, 6h, 12h, 24h)                                                                                                                                                                                                                                                        | MW           | Net balancing-area saldo as a core physical signal for system surplus or deficit.                                                                                                                                                                                                                                                                                                                                                                                                              |
| `neighbor_spread_avg`, `relative_price_competitiveness`, `price_volatility_short_term`, `scarcity_price_premium` (each 2h)                                                                                                                                                                    | EUR/MWh      | Derived competitiveness, scarcity, and volatility indicators explaining short-term price premiums.                                                                                                                                                                                                                                                                                                                                                                                         |
| `load_error_da_lag_2h`                                                                                                                                                                                                                                                                      | MW           | Load error (actual minus forecast) with PiT-compliant lag as a direct driver of short-term system imbalances.                                                                                                                                                                                                                                                                                                                                                                               |
| `da_spread_de_at_lag_2h/24h/48h/168h`, `da_spread_de_fr_lag_2h/24h/48h/168h`, `da_spread_de_nl_lag_2h/24h/48h/168h`                                                                                                                                                                         | EUR/MWh      | Bilateral DE-neighbor spreads as a proxy for cross-border import/export pressure and arbitrage tension; **no `lag_1h`** due to D-1 auction causality.                                                                                                                                                                                                                                                                                                                              |
| `afrr_activated_mw_pos/neg` (1h, 2h, 3h, 6h, 12h, 24h)                                                                                                                                                                                                                                      | MW           | Actually activated aFRR volume as a direct measure of real balancing-energy demand.                                                                                                                                                                                                                                                                                                                                                                                                     |
| `afrr_activated_mw_pos_lag_1h`                                                                                                                                                                                                                                                              | MW           | Short-term activation impulse as a momentum signal for positive reserve demand.                                                                                                                                                                                                                                                                                                                                                                                                             |
| `afrr_capacity_awarded_mw_pos/neg` (1h, 2h, 3h, 6h, 12h, 24h)                                                                                                                                                                                                                               | MW           | Awarded aFRR capacity volumes as information on expected balancing requirements.                                                                                                                                                                                                                                                                                                                                                                                                               |
| `afrr_activation_offered_mw_pos/neg` (1h, 2h, 3h, 6h, 12h, 24h)                                                                                                                                                                                                                             | MW           | Offered activation volumes as liquidity/tightness indicators of balancing markets.                                                                                                                                                                                                                                                                                                                                                                                                       |
| `afrr_activation_rate_pos`, `afrr_activation_rate_neg` (je 1h, 2h, 3h, 6h, 12h, 24h)                                                                                                                                                                                                        | Share (0-1) | Direction-specific activation rates (POS/NEG) as activated-to-reserved ratio; core intensity signals for short-term balancing usage.                                                                                                                                                                                                                                                                                                                      |
| `is_activated` (1h)                                                                                                                                                                                                                                                                         | Flag (0/1)   | Binary activation status to distinguish activated vs non-activated hours.                                                                                                                                                                                                                                                                                                                                                                                                               |
| `mfrr_activated_mw_pos/neg` (1h, 2h, 3h, 6h, 12h, 24h), `mfrr_mari_net_mw` (1h, 2h, 3h, 6h, 12h, 24h), `mfrr_active_lag` (1h, 2h, 3h, 6h, 12h, 24h)                                                                                                                                         | MW / Index   | mFRR activation and MARI flows as complementary balancing signals for system-wide reserve tightness.                                                                                                                                                                                                                                                                                                                                                                                                   |
| `residual_load_actual`, `residual_load_calc` (each 2h)                                                                                                                                                                                                                                        | MW           | Residual load (real load minus renewables) as a core driver of conventional dispatch.                                                                                                                                                                                                                                                                                                                                                                                                           |
| `generation_fossil_total_mw_lag_2h`, `generation_hydro_pumped_storage_mw_lag_2h`, `generation_hydro_actual_total_lag_2h`, `generation_baseload_total_lag_2h`                                                                                                                                | MW / Share  | Aggregated generation blocks represent supply structure; `generation_hydro_pumped_storage_mw_lag_2h` is kept **explicitly separate** because pumped storage is short-term controllable flexibility (charge/discharge) and is more informative for aFRR regimes than pure run-of-river/total-hydro signals. `generation_baseload_total_lag_2h` is defined as biomass + nuclear (`nuclear` missing -> `0`) and is effectively biomass-dominated since the April 2023 nuclear phase-out. |
| `wind_onshore_actual_entsoe` (2h, 24h, 48h, 168h)                                                                                                                                                                                                                                           | MW           | Actual onshore wind generation with short-to-weekly history for modeling weather-driven regimes.                                                                                                                                                                                                                                                                                                                                                                                             |
| `wind_offshore_actual_entsoe`, `solar_actual_entsoe` (each 2h)                                                                                                                                                                                                                                | MW           | Actual offshore wind and solar output as direct determinants of residual load.                                                                                                                                                                                                                                                                                                                                                                                                         |
| `wind_onshore_error_da/id`, `wind_offshore_error_da/id`, `solar_error_da/id`, `wind_total_error_da`, `total_wind_solar_id_error` (each 2h)                                                                                                                                                    | MW           | Forecast errors versus actuals as a proxy for forecast uncertainty and subsequent balancing-energy needs.                                                                                                                                                                                                                                                                                                                                                                                                |
| `wind_onshore_actual_entsoe_mean_24h/std_24h/mean_168h/std_168h` (each 2h)                                                                                                                                                                                                                    | MW           | Rolling level and dispersion measures of onshore feed-in to stabilize wind-regime detection.                                                                                                                                                                                                                                                                                                                                                                                               |
| `unplanned_outages_mw` (2h), `planned_outages_mw` (no lag)                                                                                                                                                                                                                                | MW           | Unplanned and planned plant outages as supply-constraint signals in short-term dispatch.                                                                                                                                                                                                                                                                                                                                                                                                   |
| `wind_onshore_forecast_id_entsoe`, `wind_offshore_forecast_id_entsoe`, `solar_forecast_id_entsoe` (no lag, 24h, 48h, 168h)                                                                                                                                                                | MW           | Ex-ante renewable feed-in forecasts for early representation of expected volatility.                                                                                                                                                                                                                                                                                                                                                                                                 |
| `renewable_share_forecast`, `residual_load_forecast` (no lag, 24h, 48h, 168h)                                                                                                                                                                                                             | Share / MW  | Forecast renewable share and expected residual load as key variables for day-ahead and balancing conditions.                                                                                                                                                                                                                                                                                                                                                                                           |
| `load_forecast_da_entsoe_h1/h2/h3/h6/h12/h24`, `wind_onshore_forecast_da_entsoe_h1/h2/h3/h6/h12/h24`, `wind_offshore_forecast_da_entsoe_h1/h2/h3/h6/h12/h24`, `solar_forecast_da_entsoe_h1/h2/h3/h6/h12/h24`, `residual_load_forecast_da_h1/.../h24`, `renewable_share_forecast_h1/.../h24` | MW / Share  | Sparse DA-Forecast-Horizonte (1..24h) als kausal gegatete Trajektorienpunkte zur expliziten Multi-Horizon-Signalgebung.                                                                                                                                                                                                                                                                                                                                                                                |
| `*_next24_mean/min/max/std`, `*_next24_ramp` (for DA forecast-based core series)                                                                                                                                                                                                            | MW / Share  | Compressed curve description of expected 24h evolution; reduces feature explosion while preserving shape information (level, dispersion, ramps).                                                                                                                                                                                                                                                                                                                                                       |
| `wind_forecast_update`, `wind_onshore_forecast_update`, `solar_forecast_update` (1h, 2h, 3h, 6h, 12h, 24h)                                                                                                                                                                                  | Index        | Forecast change metrics as early indicators for new weather information and repricing risk.                                                                                                                                                                                                                                                                                                                                                                                                      |
| `wind_onshore_capacity`, `wind_offshore_capacity`, `solar_capacity`, `gas_capacity`, `hard_coal_capacity`, `lignite_capacity`, `pumped_storage_capacity` (no lag)                                                                                                                         | MW           | Available capacities as structural upper bounds of generation and flexibility potential.                                                                                                                                                                                                                                                                                                                                                                                                       |
| `picasso_flow_rate` (lag_1h, lag_24h)                                                                                                                                                                                                                                                       | Share (0-1) | Share of cross-border PICASSO activation as an indicator for European coupling effects.                                                                                                                                                                                                                                                                                                                                                                                                       |
| `TE_hour_regime_activation` (1h)                                                                                                                                                                                                                                                            | Index        | Time-regime activation coding for explicit separation of typical hourly patterns.                                                                                                                                                                                                                                                                                                                                                                                                             |
| `hour_sin`, `hour_cos`, `dayofweek_sin`, `dayofweek_cos`, `weekday_sin`, `weekday_cos`, `month_sin`, `month_cos` (no lag)                                                                                                                                                                 | Index        | Cyclical time encodings with circular continuity (especially 23:00 -> 00:00) for robust modeling of periodic patterns.                                                                                                                                                                                                                                                                                                                                                                             |
| `is_weekend`, `is_afternoon`, `is_evening`, `is_morning`, `is_night`, `is_bridge_day`, `is_payday_period`, `is_christmas_break`, `is_picasso_active` (no lag)                                                                                                                             | Flag (0/1)   | Binary regime indicators for calendar- and market-structure-driven demand patterns and activation probabilities.                                                                                                                                                                                                                                                                                                                                                                                 |
| `holiday_severity` (no lag)                                                                                                                                                                                                                                                               | Index        | Compressed calendar index for robust separation of unusual days and operating phases.                                                                                                                                                                                                                                                                                                                                                                                                            |

## Feature Taxonomy (Information Sources)

| Category           | Typical features                                       | Information function                                                      |
| ------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Fundamental**     | Load, renewables, outages, commodity prices            | Physical and cost-driven base forces of the merit order.           |
| **Market sentiment** | DA prices (historical/PiT), spreads, volatilities      | Price regimes, relative valuation, and expectation dynamics in the market.           |
| **Echtzeit-Stress** | NRV, Aktivierungsraten, Netz-Statistiken, PICASSO-Flows | Kurzfristige Systemanspannung und Balancing-Bedarf.                       |
| **Calendar**     | Holidays, bridge days, cyclical time markers             | Structured seasonal and behavior-driven demand/supply patterns. |

## Feature Taxonomy of Model Bundles

| Bundle          | Taxonomy                 | Content focus                                                                                                                      |
| --------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **DA-Bundle**   | **Fundamental & Ex-Ante** | Load and renewable forecasts, commodity prices, calendar/seasonality features; no balancing/stress signals per D-1 causality.        |
| **aFRR-Bundle** | **Stress & Momentum**     | DA base plus short-term stress/momentum signals (NRV, activation lags, spreads, flows) to forecast deviations from the DA level. |

## Causality Check for Lag Naming

- All lag columns follow the `*_lag_Xh` pattern.
- `X` denotes the absolute delay relative to real time.
- For `system_stress_signal` and `grid_stress_index`, there is **no** `lag_1h`.

## Validation Methodology

| Method                        | Configuration                                                       | Purpose                                                                                                                      |
| ------------------------------ | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Purged Cross-Validation**    | `72h` Gap zwischen Train- und Validierungsfenster                   | Excludes temporal leakage paths caused by autocorrelation in weather, load, and price signals.                             |
| **Ablation-Tests**             | Group-wise feature removal/retention on identical splits | Quantifizierung des inkrementellen Nutzens einzelner Feature-Klassen.                                                      |
| **PnL-Proxy (Spread-Capture)** | Fold-weise Logging im CV-Loop                                       | Checks whether a feature improves not only error metrics but also economically usable directional information. |

## Methodological Governance and Evidence

| Topic                             | Implemented rule                                                                                                              | Traceability benefit                                                  |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Regime-Cut (Post-PICASSO)**     | Training bundles are filtered from `2022-06-22 22:00:00+00:00` onward.                                                            | Prevents regime mixing between historical and current market dynamics.    |
| **Strict Target Policy**          | For training, only `target_*` columns are allowed as targets; unshifted target-near series are for audit/`y_true` only. | Ensures forecasting (h+1) does not collapse into nowcasting.                |
| **DA-vs-aFRR Feature Governance** | The DA bundle is stripped of balancing/stress signals; the aFRR bundle retains these signals as core drivers.                 | Clear causal separation by decision time and model purpose.             |
| **Cross-Border-DA-Spreads (PiT)** | `da_spread_de_at/de_fr/de_nl` basieren auf PIT-gegatterten DA-Preisen (`D-1 13:00 UTC`), nicht auf Rohpreisen.                | Avoids hidden leakage paths in cross-border price relationships.    |
| **Imputation Governance**         | Bundle-seitig: `ffill(limit=12)` auf `X` plus train-fitted Median-Fallback; Logging je Spalte mit Imputation-Counts.          | Transparent, reproducible handling of small source gaps without target leakage. |
| **Ablation-Prozess**              | Feature groups are adopted only when they deliver robust value in Purged CV and holdout.                           | Data-driven feature selection instead of heuristic overloading.              |

## Primary-Source Gaps and April 2025 Forensics

- Central missingness documentation: `docs/api_missingness_report.md` und
  `data/reports/api_missingness_audit.csv`.
- For **2025-04-01 00:00 to 2025-04-02 23:00 UTC**, a targeted re-fetch was performed (`data/reports/april_refetch_comparison.csv`).
- Finding: `wind_onshore_forecast_id_entsoe` remained unchanged with **22 missing values**; classified as a **Hard Source Gap** (primary-source limit), not an ingestion error.
- Lag propagation evidence: `data/reports/april_hard_gap_propagation.csv`.

## Empirical Validity

To empirically justify the final training features, two complementary importance measures are exported after XGBoost training:

- **XGBoost Gain**: information gain per feature during tree growth.
- **SHAP (mean absolute values)**: durchschnittlicher marginaler Beitrag je
  feature contribution to prediction.

Erzeugte Artefakte:

- `data/reports/model_training/importance_report.csv`
- `data/reports/model_training/xgboost_da_top20_feature_importance.png`

These artifacts provide verifiable evidence in the methods/results section that the used `X` features (including lag structure) are not only formally causal but also materially effective in the model.

Note: These artifacts and filenames are snapshot-specific and may vary by training run/export configuration.

## Parameter Rationale

| Parameter / Feature logic          | Configuration                                                    | Methodological rationale                                                                                                                                                  |
| ---------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Daily/weekly seasonality         | Windows `24h` and `168h` (e.g., mean/std/lags)             | Power and balancing markets show stable intraday and weekly rhythms (load profiles, weekend effects). These windows explicitly capture recurring patterns. |
| PiT latency for actuals             | `lag_2h` for ENTSO-E actual values/derived actual features        | Avoids look-ahead bias due to publication latency: actual values are observable only with delay.                                                          |
| PiT latency for aFRR/market reaction  | `lag_1h` for aFRR activation/price-near and capacity signals | Market signals are treated as reliably available only after the hour has elapsed/been published.                                                                             |
| DA information gate                | `da_price_pit` with D-1 `13:00 UTC` release logic                 | Strict representation of real information availability before delivery hour; prevents foresight from not-yet-published DA auction values.                            |
| DA forecast trajectories           | Sparse Horizonte `h1,h2,h3,h6,h12,h24` plus `next24`-Kompression | Combines curve shape (ramps/volatility) with dimensionality control for multi-horizon models without breaking PiT causality.                                          |
| DA derivatives (Diff/EWMA/Stats/Slog) | Berechnung auf `da_price_pit` statt Roh-`da_price`               | Consistent PiT causality for all DA-derived features; prevents indirect leakage paths via derivatives.                                                     |
| Load Error                         | `load_error_da_lag_2h`                                           | Direct imbalance indicator from load actual vs. load forecast; causally lagged to avoid leakage.                                                         |
| Cross-Border-Spreads               | `da_spread_de_at/de_fr/de_nl` (incl. Lags)                       | Captures coupling pressure between DE and neighboring markets (import/export, arbitrage, congestion).                                                                             |
| Pumped-hydro isolation            | `generation_hydro_pumped_storage_mw_lag_2h`                      | Separate flexibility signal for tactical storage operation; avoids signal dilution in aggregated hydro blocks.                                           |
| 30-day volatility                | `da_price_volatility_30d` (rollierend, kausal verschoben)        | Captures medium-term regime shifts and stress phases that affect short-term price and activation dynamics.                                                    |
| Signed-Log-Transformation          | `slog1p(x) = sign(x) * log1p(abs(x))`                            | Compresses price spikes (positive and negative) without losing sign; stabilizes numerical gradients and reduces dominance of extreme outliers.                  |

Modeling note:

- The collinearity identified in `notebooks/13_forecast_collinearity_pca_audit.ipynb` (especially in the solar forecast family) is intentionally retained for XGBoost, since tree models are robust to collinear inputs and raw features remain available for economic interpretation.
