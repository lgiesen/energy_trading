# Data Collection Guide: Regelleistung + Energy Charts

This document is the single source of truth for the two market-price data sources used in the project pipeline:

- `regelleistung.net` (aFRR balancing market results)
- `energy-charts.info` (day-ahead spot prices across neighboring bidding zones)

It links the academic rationale (why these signals are needed) with the exact engineering implementation (how they are fetched, transformed, and stored).

---

## A) Regelleistung.net API (aFRR, Germany)

### 1. Academic Context (The "What" and "Why")
The Regelleistung.net platform publishes market results for German balancing products and therefore provides the empirical market-clearing signals required for modeling a Battery Energy Storage System (BESS) in the aFRR (PICASSO) market. In this thesis, the dataset is used to represent settlement-relevant balancing market outcomes at hourly resolution, including activation and capacity prices, offered volumes, and cross-border netting effects.

From an energy-economics perspective, the key variable is the **marginal activation price** (`GERMANY_MARGINAL_ENERGY_PRICE_[EUR/MWh]` in source files). Because PICASSO is a **pay-as-cleared** market, activated energy is settled at the clearing (marginal) price, not at an average bid price. Consequently, marginal price must be treated as the economically valid target for downstream ML and backtesting. Average price remains analytically useful but is not a valid settlement proxy for PnL simulation.

### 2. Technical Architecture (The "How")
#### Source Type
- Public HTTPS file API (REST-style file endpoints) from `regelleistung.net`
- Transport format: XLSX (and ZIP/XLSX for anonymous bid lists)
- Authentication: none required in current implementation

#### Ingestion Implementation
- Script: `src/energy_trading/ingestion/fetch_regelleistung.py`
- Libraries: `requests`, `pandas`, `openpyxl`
- Output: `data/raw/regelleistung.parquet`

#### Endpoints Used
Yearly overview files:
```text
https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/files/
RESULT_OVERVIEW_{MARKET_TYPE}_MARKET_aFRR_{YEAR}-01-01_{YEAR}-12-31.xlsx
```
Where `MARKET_TYPE` is `CAPACITY` or `ENERGY`.

Monthly fallback files (if yearly file is missing/sparse):
```text
https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/files/
RESULT_OVERVIEW_{MARKET_TYPE}_MARKET_aFRR_{YEAR}-{MM}-01_{YEAR}-{MM}-{DD}.xlsx
```

#### Query/Time Parameters
```bash
./.venv/bin/python -m energy_trading.ingestion.fetch_regelleistung \
  --start 2020-12-01T00:00:00Z --end 2026-01-01T02:00:00Z \
  --out data/raw/regelleistung.parquet
```
- Internal filtering uses `start`/`end` bounds after timestamp normalization.
- Product timestamps are parsed from source date/product fields, localized to `Europe/Berlin`, then converted to UTC.

### 3. Data Dictionary
Current `regelleistung.parquet` schema:

| Column Name | Data Type | Unit | Technical Description |
|---|---:|---|---|
| `timestamp_utc` | datetime (UTC) | - | Canonical hourly timestamp used for joins. |
| `afrr_capacity_offered_mw_neg` | float | MW | Awarded/accepted standby capacity (negative direction). |
| `afrr_capacity_offered_mw_pos` | float | MW | Awarded/accepted standby capacity (positive direction). |
| `afrr_capacity_price_neg` | float | EUR/(MW·h) | Marginal clearing capacity price, negative direction. |
| `afrr_capacity_price_pos` | float | EUR/(MW·h) | Marginal clearing capacity price, positive direction. |
| `net_import_export_mw` | float | MW | Net cross-border balancing exchange (when published in source files). |
| `afrr_activation_avg_price_neg` | float | EUR/MWh | Volume-weighted average activation price, negative direction. |
| `afrr_activation_avg_price_pos` | float | EUR/MWh | Volume-weighted average activation price, positive direction. |
| `afrr_activation_marginal_price_neg` | float | EUR/MWh | **Marginal activation clearing price**, negative direction (economically relevant settlement signal). |
| `afrr_activation_marginal_price_pos` | float | EUR/MWh | **Marginal activation clearing price**, positive direction (economically relevant settlement signal). |
| `afrr_activation_offered_mw_neg` | float | MW | Activated offered volume proxy (negative direction) from ENERGY result files. |
| `afrr_activation_offered_mw_pos` | float | MW | Activated offered volume proxy (positive direction) from ENERGY result files. |

### 4. Data Preprocessing & Transformations (ETL)
#### Ingestion-level ETL (`fetch_regelleistung.py`)
- Dynamic header normalization supports German/English column variants.
- Mixed product formats (4h blocks, quarter-hour products, explicit time windows) are parsed and expanded to hourly timeline.
- Directional pivoting creates `_pos`/`_neg` columns.
- Hourly resampling (`mean`) harmonizes CAPACITY and ENERGY outputs.
- Isolated single-hour gaps (often DST-boundary artifacts) are closed with constrained interpolation.
- If yearly overview files are unavailable or sparse, monthly fallback is applied.

#### Missing values
- Final null handling is centralized in `src/energy_trading/processing/handle_missing_values.py`.

#### Outliers and ML targeting
Target engineering is performed in `src/energy_trading/features/build_features.py` (`engineer_targets`):
1. Use **marginal** activation prices as pay-as-cleared target basis.
2. Neutralize technical sentinel-like extremes (about `±99,999`) when activation is effectively zero.
3. Maintain two targets:
   - `y_true_pos`, `y_true_neg`: cleaned, unclipped (for economic backtest)
   - `y_train_pos`, `y_train_neg`: clipped to `[-500, 500]` (for robust regression training)

---

## B) Energy Charts API (Day-Ahead Prices)

### 1. Academic Context (The "What" and "Why")
The Energy Charts API provides historical electricity market price time series for European bidding zones. The platform is operated by **Fraunhofer ISE (Energy-Charts.info)** and is used in this thesis as an external market context source for day-ahead power prices. In the modeling framework, these prices are not treated as balancing settlement prices; instead, they provide cross-border market-state information that can improve feature quality for short-term system and imbalance forecasting.

Methodologically, these series add economically interpretable exogenous inputs from neighboring power markets (e.g., AT, FR, NL, BE). In an integrated European electricity market, cross-zonal day-ahead price signals are correlated with congestion, renewable output patterns, and scarcity conditions. Therefore, including these variables is justified for feature engineering and robustness analysis while remaining separate from pay-as-cleared aFRR target construction.

### 2. Technical Architecture (The "How")
#### Source Type
- Public REST API over HTTPS
- No API key required in current implementation
- JSON response format

#### Ingestion Implementation
- Script: `src/energy_trading/ingestion/fetch_energy_charts.py`
- Libraries: `requests`, `polars`, `zoneinfo`
- Output: `data/raw/energy_charts.parquet`

#### Endpoint
```text
https://api.energy-charts.info/price?start={YYYY-MM-DD}&end={YYYY-MM-DD}&bzn={ZONE}
```

Request pattern used:
```python
base_url = f"https://api.energy-charts.info/price?start={start}&end={end}"
url = f"{base_url}&bzn={zone}"
```

#### Query parameters and zone scope
- `start`, `end`: Date boundaries (`YYYY-MM-DD`), derived from CLI UTC timestamps and converted to Europe/Berlin date format before querying.
- `bzn`: Bidding zone.
- Default zones in current pipeline:
  - `AT`, `BE`, `CH`, `CZ`, `DK1`, `DK2`, `FR`, `NL`, `PL`, `SE4`
- `NO2` is excluded in code due to high missingness in project validation.

Run standalone:
```bash
./.venv/bin/python -m energy_trading.ingestion.fetch_energy_charts \
  --start 2020-12-01T00:00:00Z --end 2026-01-01T02:00:00Z \
  --out data/raw/energy_charts.parquet
```

### 3. Data Dictionary
Current default schema (`data/raw/energy_charts.parquet`):

| Column Name | Data Type | Unit | Technical Description |
|---|---:|---|---|
| `timestamp` | datetime (UTC) | - | Canonical hourly UTC timestamp used for downstream merging. |
| `da_price_AT` | float | EUR/MWh | Day-ahead electricity price for Austria bidding zone. |
| `da_price_BE` | float | EUR/MWh | Day-ahead electricity price for Belgium bidding zone. |
| `da_price_CH` | float | EUR/MWh | Day-ahead electricity price for Switzerland bidding zone. |
| `da_price_CZ` | float | EUR/MWh | Day-ahead electricity price for Czech bidding zone. |
| `da_price_DK1` | float | EUR/MWh | Day-ahead electricity price for Denmark West bidding zone. |
| `da_price_DK2` | float | EUR/MWh | Day-ahead electricity price for Denmark East bidding zone. |
| `da_price_FR` | float | EUR/MWh | Day-ahead electricity price for France bidding zone. |
| `da_price_NL` | float | EUR/MWh | Day-ahead electricity price for Netherlands bidding zone. |
| `da_price_PL` | float | EUR/MWh | Day-ahead electricity price for Poland bidding zone. |
| `da_price_SE4` | float | EUR/MWh | Day-ahead electricity price for Sweden SE4 bidding zone. |

### 4. Data Preprocessing & Transformations (ETL)
Implemented ETL in `fetch_energy_charts.py`:
1. Fetch each zone independently via `GET /price`.
2. Parse `unix_seconds` + `price`.
3. Convert epoch seconds to timezone-aware UTC datetimes.
4. Truncate to hourly buckets and aggregate hourly mean.
5. Build unified timestamp index and full-join all zones.
6. Clip final rows to exact CLI `--start`/`--end` UTC bounds.

Missing-value and outlier policies:
- No aggressive imputation/clipping at fetch stage (source-faithful ingestion).
- Final handling is centralized in `src/energy_trading/processing/handle_missing_values.py`.

