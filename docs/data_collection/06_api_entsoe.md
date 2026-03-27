# ENTSO-E Data Source (Wind/Solar Actuals and Forecasts)

## 1. Academic Context (The "What" and "Why")
ENTSO-E Transparency data provide harmonized, pan-European operational electricity indicators and are therefore a core source for physically consistent renewable-generation modeling. In this thesis, ENTSO-E is used to obtain **actual generation** and **forecast trajectories** (day-ahead and intraday/current) for wind onshore, wind offshore, and solar in the German market context.

The methodological motivation is data consistency: forecast-error features (e.g., actual minus forecast, and forecast revisions) are only interpretable if all components originate from the same institutional source and product family. Using ENTSO-E for both actuals and forecasts avoids cross-source definition drift and supports robust error decomposition for BESS trading research, especially for short-horizon imbalance-sensitive features.

## 2. Technical Architecture (The "How")
### Source Type
- ENTSO-E Transparency Platform API (queried via Python client wrapper)
- Access method in project: `entsoe-py` (`EntsoePandasClient`)
- Authentication: API key via environment variable `ENTSOE_API_KEY`

### Ingestion Implementation
- Script: `src/energy_trading/ingestion/fetch_entsoe.py`
- Libraries: `entsoe`, `pandas`, `polars`
- Output: `data/raw/entsoe.parquet`

### Domain/Area Codes Used in Code
- `10Y1001A1001A83F` (**DE physical/control area**) for generation and wind/solar forecasts in current implementation.
- `10Y1001A1001A82H` (**DE-LU bidding zone**) is defined in code for price-domain contexts but not the primary domain for generation extraction in this fetcher.

### API Calls (via `entsoe-py`)
- Actual generation:
```python
client.query_generation(country_code, start=start, end=end)
```
- Wind/solar forecast (day-ahead):
```python
client.query_wind_and_solar_forecast(country_code, start=start, end=end, process_type="A01")
```
- Wind/solar forecast (intraday/current):
```python
client.query_wind_and_solar_forecast(country_code, start=start, end=end, process_type="A18")
```
- Fallback for intraday/current forecast if A18 is empty:
```python
process_type="A40"
```

### Query Window and Chunking
Example run:
```bash
./.venv/bin/python -m energy_trading.ingestion.fetch_entsoe \
  --start 2020-12-01T00:00:00Z --end 2026-03-01T02:00:00Z \
  --out data/raw/entsoe.parquet
```

Ingestion is chunked by time windows (month-based by default, configurable in script) and retried on transient errors.

## 3. Data Dictionary
Current output schema in `data/raw/entsoe.parquet`:

| Column Name | Data Type | Unit | Technical Description |
|---|---:|---|---|
| `timestamp_utc` | datetime (UTC) | - | Canonical hourly timestamp. |
| `wind_onshore_actual_entsoe` | float | MW | Realized wind onshore generation (ENTSO-E). |
| `wind_offshore_actual_entsoe` | float | MW | Realized wind offshore generation (ENTSO-E). |
| `solar_actual_entsoe` | float | MW | Realized solar generation (ENTSO-E). |
| `wind_onshore_forecast_da_entsoe` | float | MW | Day-ahead wind onshore forecast (process type `A01`). |
| `wind_offshore_forecast_da_entsoe` | float | MW | Day-ahead wind offshore forecast (process type `A01`). |
| `solar_forecast_da_entsoe` | float | MW | Day-ahead solar forecast (process type `A01`). |
| `wind_onshore_forecast_id_entsoe` | float | MW | Intraday/current wind onshore forecast (process type `A18`, fallback `A40`). |
| `wind_offshore_forecast_id_entsoe` | float | MW | Intraday/current wind offshore forecast (process type `A18`, fallback `A40`). |
| `solar_forecast_id_entsoe` | float | MW | Intraday/current solar forecast (process type `A18`, fallback `A40`). |

## 4. Data Preprocessing & Transformations (ETL)
### Ingestion-Level ETL (`fetch_entsoe.py`)
- Parse CLI timestamps as timezone-aware UTC.
- Query actuals + DA + ID forecasts in chunked windows.
- Select only wind/solar columns from ENTSO-E responses.
- Standardize naming to explicit suffix-based schema (`*_actual_entsoe`, `*_forecast_da_entsoe`, `*_forecast_id_entsoe`).
- Resample all series to hourly mean and merge with outer joins on UTC timestamp.

### Missing Values and Fallbacks
- Intraday forecast (`A18`) may be unavailable for subsets of history.
- Script-level fallback attempts `A40` for intraday/current forecast.
- Remaining null handling and DA fallback logic (if configured in pipeline) are applied downstream in `src/energy_trading/processing/handle_missing_values.py`.

### Timezone and Frequency Handling
- All output timestamps are normalized to UTC.
- Final output is clipped to requested `--start/--end` bounds and sorted.

### Outliers / Regulatory Limits
- No target clipping or winsorization is applied in this source fetcher.
- Outlier and target-engineering policies are intentionally implemented later in feature-processing layers to preserve source fidelity.
