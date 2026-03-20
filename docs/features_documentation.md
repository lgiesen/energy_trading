# Feature Documentation for aFRR Battery Simulation

This document describes engineered features used in
`src/energy_trading/features/build_features.py`.

## Newly Added Features

| Feature Name | Definition | Calculation Logic | Unit | Justification |
|---|---|---|---|---|
| `mfrr_active_lag` | Lagged indicator of mFRR activation activity | Let `active_t = 1` if `mfrr_activated_mwh_pos > 0` OR `mfrr_activated_mwh_neg > 0` (fallback: corresponding MW columns), else `0`. Then `mfrr_active_lag_t = active_{t-k}` with `k=1` for hourly data and `k=4` for 15-min data. | binary (0/1) | Captures scarcity/escalation conditions where mFRR is used, while lagging avoids leakage from contemporaneous balancing actions. |
| `nrv_zscore_24h` | Standardized NRV anomaly score over a 24h context window | `z_t = (NRV_balance_t - mean_24h(NRV_balance)) / std_24h(NRV_balance)` with frequency-adjusted rolling window size. Undefined values (early window, zero std) are set to `0.0`. | z-score | Highlights abnormal physical imbalance regimes relative to the recent system state. Useful for detecting stress phases linked to price spikes. |
| `picasso_flow_rate` | PICASSO/cross-border flow proxy | Uses `afrr_optimization_mwh` if available. If unavailable, uses `net_import_export_mw` as proxy. | MWh or MW (source-dependent) | Represents European balancing coupling effects and potential cross-border congestion that can decouple local bids from settlement outcomes. |
| `grid_stress_index` | Composite stress indicator in `[0, 1]` | `grid_stress_index = 0.4 * normalized_abs_nrv + 0.4 * mfrr_active_lag + 0.2 * normalized_abs_picasso_flow`. Normalization is rolling 24h max-based (`x / rolling_max_24h(|x|)`), clipped to `[0,1]`. | index [0,1] | Aggregates physical imbalance, reserve scarcity, and cross-border flow pressure into one robust stress signal for high-volatility balancing periods. |
| `market_regime_picasso` | Structural market-regime flag | `0` for timestamps `< 2022-06-22 00:00:00 UTC`, `1` for timestamps `>= 2022-06-22 00:00:00 UTC`. | binary (0/1) | Encodes the structural break from pre-PICASSO behavior to PICASSO-coupled pricing (pay-as-bid era vs. marginal/coupled regime effects). |
| `afrr_id_price_spread` | Cross-market price spread between balancing activation and intraday/day-ahead reference | `afrr_id_price_spread_t = afrr_activation_price_t - intraday_price_da_t` (column fallback to available intraday/DA proxy). | EUR/MWh | Captures opportunity-cost differentials between balancing and energy markets; relevant for bid/dispatch attractiveness. |
| `relative_price_competitiveness` | Relative current activation price level vs recent local market regime | `relative_price_competitiveness_t = afrr_activation_price_t / mean_{(t-24h, t)}(afrr_activation_price)` with leakage-safe rolling mean based on `shift(1)` and time offset `24h`. | ratio (dimensionless) | Indicates whether current activation price is expensive/cheap relative to recent conditions; supports adaptive bidding and regime-aware forecasting. |
| `price_volatility_short_term` | Short-horizon activation price uncertainty proxy | `price_volatility_short_term_t = std_{(t-4h, t)}(afrr_activation_price)` with leakage-safe window via `shift(1)` and time offset `4h`. | EUR/MWh | Short-term volatility often co-moves with reserve scarcity and uncertainty; useful explanatory signal for both activation-rate and capacity-price models. |
| `scarcity_price_premium` | Stress-weighted price interaction | `scarcity_price_premium_t = afrr_activation_price_t * grid_stress_index_t`. | composite (EUR/MWh-scaled) | Emphasizes prices observed under physical stress, improving signal quality during scarcity-driven balancing phases. |
| `TE_hour_regime_activation` | Leakage-safe target encoding of activation likelihood by hour and market regime | Group by `market_regime_picasso` and `hour`, then compute `shift(1).expanding().mean()` of `is_activated`; fill early NaNs with global mean activation rate. | probability-like ratio [0,1] | Encodes historically observed activation propensity for the same intraday/regime context while preventing look-ahead leakage. |
| `market_state_cluster` | Latent market-state label from stress/flow regime vectors | Build rolling-normalized (24h, past-only) versions of `nrv_zscore_24h`, `grid_stress_index`, `picasso_flow_rate`; apply `StandardScaler` then `KMeans(n_clusters=4, random_state=42)`. | categorical cluster id (0-3) | Captures non-linear market-state regimes (e.g., scarcity/export vs stable/import) that are hard to represent with single linear features. |
| `nrv_quantile_5` | Discrete NRV stress bucket | `pd.qcut(nrv_zscore_24h, q=5, labels=False, duplicates='drop')` with robust fallback for missing bins. | ordinal bucket (0-4, data-dependent if duplicates dropped) | Adds rank-based stress representation robust to scale/outlier effects and useful for tree/sequence models. |

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

- `mfrr_active_lag` is explicitly lagged to prevent target leakage.
- Rolling features use frequency-aware window sizes (24h in row space).
- Early-window NaNs and divide-by-zero cases are handled with safe defaults.
- The PICASSO regime switch is inclusive from **2022-06-22** onward.
- `days_since_last_activation` uses a deterministic fallback (`-1`) for each user’s first event, ensuring robust model input without row drops.
- Cyclical time features (`*_sin`, `*_cos`) are leakage-safe because they are derived only from the current timestamp, not from future observations.
- `TE_hour_regime_activation` is leakage-protected by construction: `shift(1)` excludes the current target and `expanding().mean()` uses only historical observations.
- `market_state_cluster` is computed from rollingly normalized stress inputs using past-only windows, reducing regime drift sensitivity and preventing future-information bleed.
