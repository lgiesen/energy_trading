# SMARD Data Source (Generation, Forecasts, Prices, Installed Capacity)

## 1. Academic Context (The "What" and "Why")
SMARD (operated by the German Federal Network Agency, Bundesnetzagentur) provides official electricity-system time series for Germany and the DE-LU market area. In this thesis, SMARD is used as a high-coverage operational source for realized generation, day-ahead forecasts, market prices, residual load, and installed generation capacity. These variables are required to construct physically meaningful explanatory features for algorithmic BESS trading models, especially for linking renewable forecast uncertainty and conventional fleet availability to balancing-market behavior.

Methodologically, SMARD serves as a foundational context dataset rather than the final balancing-settlement source. Its combination of physical generation signals and market-price references supports feature engineering, structural diagnostics, and robustness analysis. In particular, SMARD installed-capacity and generation signals are useful for normalizing time-varying renewable output regimes and for interpreting forecast error dynamics over multi-year horizons.

## 2. Technical Architecture (The "How")
### Source Type
- Public HTTPS data services (two API interfaces)
- Authentication: no API key/token required in current implementation

### Ingestion Implementation
- Script: `src/energy_trading/ingestion/fetch_smard.py`
- Libraries: `requests`, `pandas`, `polars`, `zoneinfo`
- Output files:
  - `data/raw/smard.parquet`
  - `data/raw/installed_capacity.csv` (downloaded market-data export)

### API Interface A: SMARD `chart_data` (timeseries)
Used for hourly/quarter-hour time series retrieval.

Index endpoint pattern:
```text
https://www.smard.de/app/chart_data/{filter_id}/{region}/index_{resolution}.json
```

Chunk endpoint pattern:
```text
https://www.smard.de/app/chart_data/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{timestamp}.json
```

Implementation details:
- Default region/resolution for most series: `DE-LU`, `hour`
- Intraday price fetch uses dedicated quarter-hour configuration and is aggregated to hourly mean.
- Each module/filter is fetched, standardized to UTC, then merged into one wide table.

### API Interface B: SMARD `nip-download-manager` (market-data CSV export)
Used for installed generation capacity download.

Init page (session/cookies):
```text
https://www.smard.de/en/downloadcenter/download-market-data/
```

POST download endpoint:
```text
https://www.smard.de/nip-download-manager/nip/download/market-data
```

Payload characteristics in code:
- `format: CSV`
- `region: DE-LU`
- module list for installed-capacity categories
- type/language/resolution metadata as defined in `MARKET_DATA_PAYLOAD`

Downloaded CSV is parsed, normalized, converted to UTC, expanded to hourly timeline, and joined into `smard.parquet`.

### Typical CLI Run
```bash
./.venv/bin/python -m energy_trading.ingestion.fetch_smard \
  --start 2020-12-01T00:00:00Z --end 2026-01-01T02:00:00Z \
  --out data/raw/smard.parquet \
  --market-data-out data/raw/installed_capacity.csv
```

Optional mode (timeseries only):
```bash
./.venv/bin/python -m energy_trading.ingestion.fetch_smard \
  --start 2020-12-01T00:00:00Z --end 2026-01-01T02:00:00Z \
  --out data/raw/smard.parquet \
  --skip-market-data-csv
```

## 3. Data Dictionary
Current output schema in `data/raw/smard.parquet`:

| Column Name | Data Type | Unit | Technical Description |
|---|---:|---|---|
| `timestamp_utc` | datetime (UTC) | - | Canonical UTC timestamp for cross-source merging. |
| `timestamp_cet` | datetime (Europe/Berlin) | - | Local market timestamp for diagnostics/audit. |
| `residual_load_actual` | float | MW | Realized residual load. |
| `wind_onshore_actual` | float | MW | Realized wind onshore generation. |
| `wind_offshore_actual` | float | MW | Realized wind offshore generation. |
| `solar_actual` | float | MW | Realized solar generation. |
| `wind_onshore_forecast` | float | MW | Day-ahead wind onshore forecast. |
| `wind_offshore_forecast` | float | MW | Day-ahead wind offshore forecast. |
| `solar_forecast` | float | MW | Day-ahead solar forecast. |
| `generation_fossil_brown_coal_mw` | float | MW | Realized lignite generation. |
| `generation_fossil_hard_coal_mw` | float | MW | Realized hard-coal generation. |
| `generation_fossil_gas_mw` | float | MW | Realized fossil-gas generation. |
| `generation_nuclear_mw` | float | MW | Realized nuclear generation. |
| `generation_hydro_pumped_storage_mw` | float | MW | Realized hydro pumped-storage generation. |
| `da_price` | float | EUR/MWh | Day-ahead price (DE-LU). |
| `wind_forecast_de` | float | MW | Engineered combined wind forecast signal. |
| `wind_onshore_error` | float | MW | Engineered wind onshore forecast error. |
| `wind_offshore_error` | float | MW | Engineered wind offshore forecast error. |
| `solar_error` | float | MW | Engineered solar forecast error. |
| `system_stress_signal` | float | MW | Engineered aggregate forecast-error stress proxy. |
| `wind_onshore_capacity` | float | MW | Installed wind onshore capacity (from SMARD CSV export). |
| `wind_offshore_capacity` | float | MW | Installed wind offshore capacity (from SMARD CSV export). |
| `solar_capacity` | float | MW | Installed solar capacity (from SMARD CSV export). |
| `gas_capacity` | float | MW | Installed gas capacity (from SMARD CSV export). |
| `hard_coal_capacity` | float | MW | Installed hard-coal capacity (from SMARD CSV export). |
| `lignite_capacity` | float | MW | Installed lignite capacity (from SMARD CSV export). |
| `pumped_storage_capacity` | float | MW | Installed hydro pumped-storage capacity (from SMARD CSV export). |

## 4. Data Preprocessing & Transformations (ETL)
### Ingestion-Level ETL (`fetch_smard.py`)
- Fetches multiple SMARD module/filter series and converts timestamps to timezone-aware UTC.
- Handles mixed resolutions by truncating/aggregating to hourly alignment.
- Computes and persists convenience engineered signals (`*_error`, `wind_forecast_de`, `system_stress_signal`) within source output.
- Downloads installed-capacity CSV via session-based POST flow, detects embedded header rows, parses localized timestamps, and joins capacity columns into hourly table.

### Timezone and Frequency Handling
- Source timestamps are normalized to UTC for canonical storage (`timestamp_utc`).
- Local market time is retained as `timestamp_cet` for auditability.
- Quarter-hour series (where applicable) are aggregated to hourly mean before merge.

### Missing Values and Data Quality Handling
- Fetch-stage logic remains largely source-faithful; no heavy imputation policy is applied inside this fetcher.
- Installed-capacity CSV is expanded to hourly frequency via forward fill after parsing.
- Project-wide missing-value policy is centralized downstream in:
  - `src/energy_trading/processing/handle_missing_values.py`

### Outliers / Regulatory Limits
- No target clipping or aggressive outlier capping is applied in `fetch_smard.py`.
- Outlier treatment and target-specific cleaning are handled in feature-engineering/model layers to avoid source-level distortion.
