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

## Notes on Leakage and Robustness

- `mfrr_active_lag` is explicitly lagged to prevent target leakage.
- Rolling features use frequency-aware window sizes (24h in row space).
- Early-window NaNs and divide-by-zero cases are handled with safe defaults.
- The PICASSO regime switch is inclusive from **2022-06-22** onward.
