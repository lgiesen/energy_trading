# Feature Documentation (Current Pipeline State)

This document is synchronized with the current refinement logic in
`src/energy_trading/processing/drop_redundant_features.py` and the resulting
refined dataset (`data/processed/all_data_refined.parquet`, 57 columns).

## Scope

- Purpose: document the model-ready *refined* feature table before downstream
  training/feature expansion.
- Time standard: UTC only (`timestamp_utc`).
- Design objective: reduce redundancy, prevent multicollinearity, and keep
  economically interpretable features for DA/aFRR forecasting.

## 1) Drop Logic Synchronization

### A. Redundant Sources (SMARD vs ENTSO-E)

SMARD duplicates were removed where ENTSO-E-based signals are preferred for
consistency in the thesis pipeline.

Removed examples:
- `wind_onshore_actual`, `wind_offshore_actual`, `solar_actual`
- `wind_onshore_forecast`, `wind_offshore_forecast`, `solar_forecast`
- legacy error columns (`wind_onshore_error`, `wind_offshore_error`,
  `solar_error`)

Reason:
- avoid parallel source definitions of the same physical quantity,
- keep one canonical lineage for auditability.

### B. Multicollinearity Reduction (VIF-focused)

Kept as canonical price/saldo signals:
- `afrr_vwap_pos`, `afrr_vwap_neg`
- `NRV_balance`

Removed as redundant/competing definitions:
- activation price variants containing:
  - `avg_activation_price`
  - `marginal_activation_price`
  - `bid_avg_activation_price`
- balance alternatives: `rz_saldo_mw`, `reBAP_shortage_surplus`

Reason:
- multiple price and saldo variants were strongly collinear and caused
  instability for linear/regularized models.

### C. Logical Aggregations

Implemented aggregations:
- `neighbor_spread_avg`
- `generation_fossil_total_mw`
- `generation_baseload_total`
- `generation_hydro_actual_total`

and dropped component columns after aggregation.

Reason:
- reduce dimensionality while preserving economic signal.

### D. Technical Cleanup

Removed:
- `timestamp_cet`

Kept:
- `timestamp_utc`

Reason:
- avoid duplicate time axes and enforce one timezone standard across training,
  validation, and backtests.

## 2) Engineered Features (Definitions & Formulas)

### Residual Load Features

- `residual_load_forecast` [MW]
  - Formula:
    `load_forecast_da_entsoe - solar_forecast_id_entsoe - wind_onshore_forecast_id_entsoe - wind_offshore_forecast_id_entsoe`
  - Meaning: expected non-renewable net demand.

- `residual_load_actual` [MW]
  - Formula:
    `load_actual_entsoe - solar_actual_entsoe - wind_onshore_actual_entsoe - wind_offshore_actual_entsoe`
  - Meaning: realized non-renewable net demand.

### Renewable Share Feature

- `renewable_share_forecast` [ratio, 0..1+ depending on forecast/load relation]
  - Formula:
    `(solar_forecast_id_entsoe + wind_onshore_forecast_id_entsoe + wind_offshore_forecast_id_entsoe) / load_forecast_da_entsoe`
  - Meaning: expected renewable penetration in load.

### Cross-Border Price Pressure

- `neighbor_spread_avg` [EUR/MWh]
  - Formula:
    `mean(da_price_AT, BE, CH, CZ, DK1, DK2, FR, NL, PL, SE4) - da_price_eur`
  - Meaning: regional price pressure relative to Germany.

### Forecast Update Features

- `wind_onshore_forecast_update` [MW]
  - Formula:
    `wind_onshore_forecast_id_entsoe - wind_onshore_forecast_da_entsoe`
- `solar_forecast_update` [MW]
  - Formula:
    `solar_forecast_id_entsoe - solar_forecast_da_entsoe`

Interpretation:
- captures new information arriving between DA and intraday forecast stages.

### Capacity Import/Export Directional Logic

From `fetch_regelleistung.py`, capacity products are parsed from block products
(e.g. `POS_00_04`, `NEG_12_16`), expanded to hourly rows, and direction-split.

Produced columns:
- `capacity_import_export_mw_pos` [MW]
- `capacity_import_export_mw_neg` [MW]

Important implementation detail:
- 4h block values are replicated to each hourly row (not divided by 4),
  consistent with `[MW]` and `[(EUR/MW)/h]` semantics.

## 3) Final Refined Column Set (57)

### Time
- `timestamp_utc`

### Load / Residual / Forecast Core
- `load_actual_entsoe` [MW]
- `load_forecast_da_entsoe` [MW]
- `residual_load_actual` [MW]
- `residual_load_forecast` [MW]
- `renewable_share_forecast` [ratio]

### Renewable Actuals and Forecast Inputs
- `wind_onshore_actual_entsoe` [MW]
- `wind_offshore_actual_entsoe` [MW]
- `solar_actual_entsoe` [MW]
- `wind_onshore_forecast_id_entsoe` [MW]
- `wind_offshore_forecast_id_entsoe` [MW]
- `solar_forecast_id_entsoe` [MW]
- `wind_onshore_error_da` [MW]
- `wind_offshore_error_da` [MW]
- `solar_error_da` [MW]
- `wind_onshore_forecast_update` [MW]
- `solar_forecast_update` [MW]

### Balancing Activation / Flows
- `NRV_balance` [MW]
- `afrr_activated_mw_pos` [MW]
- `afrr_activated_mw_neg` [MW]
- `mfrr_activated_mw_pos` [MW]
- `mfrr_activated_mw_neg` [MW]
- `afrr_picasso_mw_pos` [MW]
- `afrr_picasso_mw_neg` [MW]
- `afrr_picasso_net_mw` [MW]
- `mfrr_mari_mw_pos` [MW]
- `mfrr_mari_mw_neg` [MW]
- `mfrr_mari_net_mw` [MW]
- `capacity_import_export_mw_pos` [MW]
- `capacity_import_export_mw_neg` [MW]
- `net_import_export_mw` [MW]
- `system_stress_signal` [composite index/MW-equivalent]

### aFRR Prices and Offered Capacities
- `afrr_vwap_pos` [EUR/MWh]
- `afrr_vwap_neg` [EUR/MWh]
- `afrr_capacity_price_pos` [EUR/MW/h]
- `afrr_capacity_price_neg` [EUR/MW/h]
- `afrr_capacity_offered_mw_pos` [MW]
- `afrr_capacity_offered_mw_neg` [MW]
- `afrr_activation_offered_mw_pos` [MW]
- `afrr_activation_offered_mw_neg` [MW]

### Energy Prices and Exogenous Commodities
- `da_price_eur` [EUR/MWh]
- `price_intraday_eur` [EUR/MWh]
- `gas_price_ttf` [market quote]
- `coal_price_api2` [market quote]
- `co2_price_eua` [market quote]

### Generation and Capacity Structure
- `generation_hydro_pumped_storage_mw` [MW]
- `generation_fossil_total_mw` [MW]
- `generation_baseload_total` [MW]
- `generation_hydro_actual_total` [MW]
- `wind_onshore_capacity` [MW]
- `wind_offshore_capacity` [MW]
- `solar_capacity` [MW]
- `gas_capacity` [MW]
- `hard_coal_capacity` [MW]
- `lignite_capacity` [MW]
- `pumped_storage_capacity` [MW]

### Cross-Border Price Aggregate
- `neighbor_spread_avg` [EUR/MWh]

## 4) Veraltete Dokumentation entfernt / korrigiert

The following are no longer active in the refined 57-column set and must not be
interpreted as current model inputs in this stage:
- foreign raw DA prices (`da_price_AT`, ..., `da_price_SE4`)
- `neighbor_price_avg`
- `rz_saldo_mw`, `reBAP_shortage_surplus`
- activation price families based on `avg_activation_price`,
  `marginal_activation_price`, `bid_avg_activation_price`
- `timestamp_cet`
- `biomass_actual_entsoe`, `generation_nuclear_mw` (replaced by
  `generation_baseload_total`)

## 5) Unit Check (Current)

- Power/flow/capacity channels: MW
- Activation energy price and DA/ID prices: EUR/MWh
- Capacity prices: EUR/MW/h
- Ratio features: dimensionless (`renewable_share_forecast`)
- Composite signals (`system_stress_signal`): documented as composite index
  or MW-equivalent proxy depending on upstream definition.

## To-Do: Missing Info

1. `system_stress_signal`
- Confirm exact upstream formula and normalization bounds in final pipeline
  run used for thesis tables.

2. `net_import_export_mw` vs directional pair
- Confirm sign convention and intended canonical usage when both
  `capacity_import_export_mw_pos/neg` and `net_import_export_mw` exist.

3. Commodity units metadata
- Add canonical unit metadata for `gas_price_ttf`, `coal_price_api2`,
  `co2_price_eua` (provider-specific quotation conventions).

4. Capacity-source lineage table
- Add explicit per-column lineage for capacities (ENTSO-E vs fallback source)
  for audit appendix completeness.
