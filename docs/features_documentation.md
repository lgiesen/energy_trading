# Feature Documentation for aFRR Battery Simulation

This document describes engineered features used in
`src/energy_trading/features/build_features.py`.

## Data Latency & Information Set

To prevent Look-Ahead Bias, the feature pipeline applies strict Point-in-Time (PiT)
alignment before downstream feature engineering. Physical actuals are lagged by 2h
and balancing/settlement-style observations are lagged by 1h. This ensures that each
row only contains information that was physically available at prediction time.

| Feature Group | Applied Lag | Notes |
|---|---|---|
| ENTSO-E actuals (`*_actual_entsoe`, `generation_fossil_*`, `generation_nuclear_mw`, `generation_hydro_*`, `residual_load_actual`, `NRV_balance`) | 2 hours | Physical observations with publication delay |
| Balancing/settlement-like streams (`afrr_activated_mw*`, `mfrr_activated_mw*`, `afrr_marginal_activation_price*`, `afrr_vwap*`, `reBAP_shortage_surplus*`, `afrr_picasso_mw*`, `afrr_picasso_net_mw`, `mfrr_mari_net_mw`, `price_intraday_eur`) | 1 hour | Finalized only after interval close / settlement publication |
| Day-ahead market prices (`da_price_eur`, `da_price_*`) | 0 hours | Known for delivery day since D-1 13:00 CET/CEST; strict T-0 overlap in feature layer |
| Capacity/calendar/fuel metadata | 0 hours | Available ex-ante |

### Information Gate (Feature Finalization Time)

The information gate defines the latest timestamp at which each feature class is
considered available for a prediction made at decision time `T`.

| Feature Class | Gate Definition | Interpretation |
|---|---|---|
| ENTSO-E actuals and derived actual-based signals (`residual_load_calc`, `wind_total_error_da`) | finalized at `T-2h` | Values from `(T-2h, T]` are not visible to the model at `T`. |
| Balancing/settlement streams (aFRR/mFRR activation and settlement-price proxies) | finalized at `T-1h` | The most recent hour is withheld to reflect settlement/operational delay. |
| Day-ahead forecasts (`*_forecast_da*`) for sequence settings | publication-constrained by 13:00 Europe/Berlin and forecast horizon | If required horizon exceeds known publication coverage, a persistence fallback (`shift(24)`) is used. |
| Day-ahead market prices (`da_price_eur`, `da_price_*`) | finalized at `T` for delivery-day usage | Treated as strict T-0 in the feature layer (no publication gating). |
| Calendar/metadata/capacity | finalized at `T` | No publication lag applied in the feature layer. |

Additional availability handling:
- Day-ahead forecast columns are aligned with a conservative 13:00 Europe/Berlin publication cutoff.
- For 72h sequence contexts (D+3), DA forecasts that are not yet publishable at decision time use persistence fallback (same hour previous day).
- Day-ahead price columns are not gated and remain strict T-0 for delivery-day feature usage.
- Missing values from lagging and long windows are never backfilled from future rows.
- Ground truth (`y_true_pos`, `y_true_neg`) is not lagged. Multi-output targets are created by future shifting (`target_pos_h1..h72`, `target_neg_h1..h72`).

## Newly Added Features

| Feature Name | Definition | Calculation Logic | Unit | Justification |
|---|---|---|---|---|
| `mfrr_active_lag` | mFRR activity indicator built from PiT-aligned activation columns | Let `active_t = 1` if `mfrr_activated_mwh_pos > 0` OR `mfrr_activated_mwh_neg > 0` (fallback: corresponding MW columns), else `0`. `mfrr_active_lag_t = active_t` on already PiT-lagged base inputs (no extra internal shift). | binary (0/1) | Avoids double lagging while preserving a causally valid scarcity indicator. |
| `nrv_zscore_24h` | Standardized NRV anomaly score over a 24h context window | `z_t = (NRV_balance_t - mean_24h(NRV_balance)) / std_24h(NRV_balance)` with frequency-adjusted rolling window size. Undefined values (early window, zero std) are set to `0.0`. | z-score | Highlights abnormal physical imbalance regimes relative to the recent system state. Useful for detecting stress phases linked to price spikes. |
| `picasso_flow_rate` | PICASSO/cross-border flow proxy | Uses `afrr_optimization_mwh` if available. If unavailable, uses `net_import_export_mw` as proxy. | MWh or MW (source-dependent) | Represents European balancing coupling effects and potential cross-border congestion that can decouple local bids from settlement outcomes. |
| `grid_stress_index` | Composite stress indicator in `[0, 1]` | `grid_stress_index = 0.4 * normalized_abs_nrv + 0.4 * mfrr_active_lag + 0.2 * normalized_abs_picasso_flow`. Normalization is rolling 24h max-based (`x / rolling_max_24h(|x|)`), clipped to `[0,1]`. | index [0,1] | Aggregates physical imbalance, reserve scarcity, and cross-border flow pressure into one robust stress signal for high-volatility balancing periods. |
| `system_stress_signal` | Aggregate weather/forecast stress proxy | Horizontal sum of per-technology errors; prefers ID-based errors when available, otherwise DA-based errors. | MW-equivalent composite | Captures coincident stress from wind/solar forecast misses in one compact feature. |
| `market_regime_picasso` | Structural market-regime flag | `0` for timestamps `< 2022-06-22 00:00:00 UTC`, `1` for timestamps `>= 2022-06-22 00:00:00 UTC`. | binary (0/1) | Encodes the structural break from pre-PICASSO behavior to PICASSO-coupled pricing (pay-as-bid era vs. marginal/coupled regime effects). |
| `afrr_id_price_spread` | Cross-market price spread between balancing activation and intraday/day-ahead reference | `afrr_id_price_spread_t = afrr_activation_price_t - intraday_price_da_t` (column fallback to available intraday/DA proxy). | EUR/MWh | Captures opportunity-cost differentials between balancing and energy markets; relevant for bid/dispatch attractiveness. |
| `relative_price_competitiveness` | Relative current activation price level vs recent local market regime | `relative_price_competitiveness_t = afrr_activation_price_t / mean_{(t-24h, t)}(afrr_activation_price)` with leakage-safe rolling mean based on `shift(1)` and time offset `24h`. | ratio (dimensionless) | Indicates whether current activation price is expensive/cheap relative to recent conditions; supports adaptive bidding and regime-aware forecasting. |
| `price_volatility_short_term` | Short-horizon activation price uncertainty proxy | `price_volatility_short_term_t = std_{(t-4h, t)}(afrr_activation_price)` with leakage-safe window via `shift(1)` and time offset `4h`. | EUR/MWh | Short-term volatility often co-moves with reserve scarcity and uncertainty; useful explanatory signal for both activation-rate and capacity-price models. |
| `scarcity_price_premium` | Stress-weighted price interaction | `scarcity_price_premium_t = afrr_activation_price_t * grid_stress_index_t`. | composite (EUR/MWh-scaled) | Emphasizes prices observed under physical stress, improving signal quality during scarcity-driven balancing phases. |
| `TE_hour_regime_activation` | Leakage-safe target encoding of activation likelihood by hour and market regime | Group by `market_regime_picasso` and `hour`, then compute `shift(1).expanding().mean()` of `is_activated`; fill early NaNs with global mean activation rate. | probability-like ratio [0,1] | Encodes historically observed activation propensity for the same intraday/regime context while preventing look-ahead leakage. |
| `market_state_cluster` | Latent market-state label from stress/flow regime vectors | Build rolling-normalized (24h, past-only) versions of `nrv_zscore_24h`, `grid_stress_index`, `picasso_flow_rate`; apply `StandardScaler` then `KMeans(n_clusters=4, random_state=42)`. | categorical cluster id (0-3) | Captures non-linear market-state regimes (e.g., scarcity/export vs stable/import) that are hard to represent with single linear features. |
| `nrv_quantile_5` | Discrete NRV stress bucket | `pd.qcut(nrv_zscore_24h, q=5, labels=False, duplicates='drop')` with robust fallback for missing bins. | ordinal bucket (0-4, data-dependent if duplicates dropped) | Adds rank-based stress representation robust to scale/outlier effects and useful for tree/sequence models. |
| `load_actual_entsoe_mean_24h`, `load_actual_entsoe_std_24h`, `load_actual_entsoe_mean_168h`, `load_actual_entsoe_std_168h` | Rolling load moments | Rolling mean/std over 24h and 168h windows on PiT-lagged `load_actual_entsoe`. | MW | Captures short and weekly load-state context. |
| `da_price_eur_mean_24h`, `da_price_eur_std_24h`, `da_price_eur_mean_168h`, `da_price_eur_std_168h` | Rolling day-ahead price moments | Rolling mean/std over 24h and 168h windows on `da_price_eur`. | EUR/MWh | Adds multi-scale price regime context. |
| `wind_onshore_actual_entsoe_mean_24h`, `wind_onshore_actual_entsoe_std_24h`, `wind_onshore_actual_entsoe_mean_168h`, `wind_onshore_actual_entsoe_std_168h` | Rolling wind onshore moments | Rolling mean/std over 24h and 168h windows on PiT-lagged `wind_onshore_actual_entsoe`. | MW | Captures renewable supply level and variability. |
| `target_pos_h1..target_pos_h72`, `target_neg_h1..target_neg_h72` | 72-step direct multi-output targets | `target_pos_hk = y_true_pos.shift(-k)`, `target_neg_hk = y_true_neg.shift(-k)` for `k=1..72`. | EUR/MWh | Enables direct horizon-wise training for 72h sequence forecasting. |

### Time-Based Features

| Feature Name | Definition | Calculation Logic | Unit | Justification |
|---|---|---|---|---|
| `hour_sin`, `hour_cos` | Cyclical hour-of-day representation | With `h_t ∈ {0,...,23}` from timestamp column: `hour_sin_t = sin(2π h_t / 24)`, `hour_cos_t = cos(2π h_t / 24)`. | continuous [-1,1] | Preserves circular continuity of intraday time (e.g., 23:00 adjacent to 00:00). |
| `weekday_sin`, `weekday_cos` | Cyclical weekday representation | With `w_t ∈ {0,...,6}`: `weekday_sin_t = sin(2π w_t / 7)`, `weekday_cos_t = cos(2π w_t / 7)`. | continuous [-1,1] | Captures repeating weekly demand and balancing behavior without artificial ordinal jumps. |
| `month_sin`, `month_cos` | Cyclical month-of-year representation | With `m_t ∈ {1,...,12}`: `month_sin_t = sin(2π m_t / 12)`, `month_cos_t = cos(2π m_t / 12)`. | continuous [-1,1] | Models annual seasonality (weather/load) while preserving month-cycle continuity. |
| `is_weekend` | Weekend indicator | `is_weekend_t = 1` if weekday is Saturday or Sunday, else `0`. | binary (0/1) | Weekend operations often show different load patterns, liquidity, and balancing activation behavior. |
| `is_payday_period` | Payday-window indicator | `is_payday_period_t = 1` if day-of-month `>= 27` OR `<= 3`, else `0`. | binary (0/1) | Approximates monthly liquidity and behavior shifts that can propagate into consumption and balancing patterns. |
| `is_morning` | Morning segment flag | `1` if hour in `[06,11]`, else `0`. | binary (0/1) | Captures morning ramp effects and typical market-state transitions. |
| `is_afternoon` | Afternoon segment flag | `1` if hour in `[12,16]`, else `0`. | binary (0/1) | Captures midday balancing dynamics and load/RES transitions. |
| `is_evening` | Evening segment flag | `1` if hour in `[17,22]`, else `0`. | binary (0/1) | Captures evening peak-related stress and price-relevant balancing conditions. |
| `is_night` | Night segment flag | `1` if hour in `[23,05]`, else `0`. | binary (0/1) | Captures low-load base regime and nocturnal balancing behavior. |
| `days_since_last_activation` | User-level recency feature | For each user (grouped by user ID), compute chronological delta in days to previous event: `Δdays_t = (ts_t - ts_{t-1})/86400`. First event per user is set to `-1`. | days | Encodes user/event rhythm and recency effects; useful for activation propensity timing. |
| `holiday_severity` *(existing)* | Holiday intensity score | Already precomputed in upstream pipeline (national/regional severity encoding). | score | Public holiday intensity strongly affects demand profile and balancing dynamics. |
| `is_bridge_day` *(existing)* | Bridge-day indicator | Already precomputed in upstream pipeline. | binary (0/1) | Bridge days can induce non-standard industrial/commercial load patterns. |
| `is_christmas_break` *(existing)* | Christmas-break indicator | Already precomputed in upstream pipeline. | binary (0/1) | End-of-year operations materially shift demand and reserve activation behavior. |

## Notes on Leakage and Robustness

- `mfrr_active_lag` is computed from already PiT-lagged activation columns and therefore does not apply an additional internal lag.
- Rolling features use frequency-aware window sizes (24h in row space).
- Early-window NaNs and divide-by-zero cases are handled with safe defaults.
- The PICASSO regime switch is inclusive from **2022-06-22** onward.
- `days_since_last_activation` uses a deterministic fallback (`-1`) for each user’s first event, ensuring robust model input without row drops.
- Cyclical time features (`*_sin`, `*_cos`) are leakage-safe because they are derived only from the current timestamp, not from future observations.
- `TE_hour_regime_activation` is leakage-protected by construction: `shift(1)` excludes the current target and `expanding().mean()` uses only historical observations.
- `market_state_cluster` is computed from rollingly normalized stress inputs using past-only windows, reducing regime drift sensitivity and preventing future-information bleed.

## Automated Audit Protocol

The notebook
`notebooks/09_lag_information_set_validation.ipynb`
contains a cell-by-cell mathematical audit between:
- Ground Truth (GT): `data/processed/all_data_refined.parquet`
- Feature Set (FE): `data/processed/all_data_features.parquet`

Audit constraints:
- 2H group: `FE[C]_t = GT[C]_{t-2}`
- 1H group: `FE[C]_t = GT[C]_{t-1}`
- 0H group: `FE[C]_t = GT[C]_t`
- Targets: `FE[target_*_h]_t = REF[y_true_*]_{t+h}` for `h in [1,72]`,
  where `REF` is `FE` when `y_true_*` is generated in feature engineering
  (fallback to `GT` when present).

Tolerance:
- `max_abs_error <= 1e-9` for lag-policy groups (`2H`, `1H`, `0H`, `D+3_GATE`)
- Derived-feature verification uses a separate floating-point tolerance (`1e-7`)
  for numerical stability in rolling-standard-deviation recomputation checks.
- Boundary NaNs are ignored through overlap-only comparisons.

Special D+3 forecast availability audit:
- Forecast columns are checked against a publication-aware expected series
  (13:00 Europe/Berlin DA publication gate with persistence fallback `shift(24)`).
- Additional anti-look-ahead check ensures gated rows do not collapse to raw future-truth values when informative differences exist.

Latest audited outcome (executed on March 20, 2026):
- `da_price_eur` strict T-0 check: **PASS** (`max_abs_error = 0.0`)
- Lag-policy failures (`2H`, `1H`, `0H`, `D+3_GATE`): **0** (PASS)
- Informational groups (`TARGET`, `DERIVED`, `UNMAPPED`) are reported
  separately and do not drive the hard policy gate.
