# Regelleistung.net API Data Source (aFRR, Germany)

## 1. Academic Context (The "What" and "Why")

The Regelleistung.net platform publishes balancing-market result files for German ancillary-service products and is therefore a core empirical source for modeling Battery Energy Storage System (BESS) trading in the aFRR segment. In this thesis, the dataset represents realized balancing outcomes at market time scale, including activation prices, capacity prices, directional volumes, and cross-border balancing exchange proxies. These variables are required to connect predictive modeling with actual balancing-market microstructure rather than relying only on spot-market proxies.

From an energy-economics perspective, the crucial variable is the **marginal activation price** because aFRR settlement in the PICASSO context follows a **pay-as-cleared** principle. Consequently, the marginal activation price is the economically valid target basis for ML and for financial backtesting. Average activation price can be retained for diagnostics and robustness checks, but it is not a settlement-identical substitute for realized BESS cashflow under pay-as-cleared clearing.

## 2. Technical Architecture (The "How")

### Source Type

- Public HTTPS file API (REST-style endpoint pattern)
- File payloads: XLSX overview files (`ENERGY`, `CAPACITY`)
- Authentication: none in current implementation

### Ingestion Implementation

- Script: `src/energy_trading/ingestion/fetch_regelleistung.py`
- Libraries: `requests`, `pandas`, `openpyxl`
- Output: `data/raw/regelleistung.parquet`

### Endpoints and Extraction Strategy

Yearly overview file pattern:

```text
https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/files/
RESULT_OVERVIEW_{MARKET_TYPE}_MARKET_aFRR_{YEAR}-01-01_{YEAR}-12-31.xlsx
```

Where `MARKET_TYPE` is:

- `ENERGY`
- `CAPACITY`

Monthly fallback file pattern (used when yearly files are missing/sparse):

```text
https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/files/
RESULT_OVERVIEW_{MARKET_TYPE}_MARKET_aFRR_{YEAR}-{MM}-01_{YEAR}-{MM}-{DD}.xlsx
```

### Query/Window Parameters

Example run:

```bash
./.venv/bin/python -m energy_trading.ingestion.fetch_regelleistung \
  --start 2020-12-01T00:00:00Z --end 2026-03-01T02:00:00Z \
  --out data/raw/regelleistung.parquet
```

- No pagination token is required; the script iterates annual files and monthly fallbacks.
- Product timestamps are parsed from source date/product fields, localized to `Europe/Berlin`, then converted to UTC.
- Final filtering is constrained to the requested `--start/--end` interval.

## 3. Data Dictionary

Current output schema (`data/raw/regelleistung.parquet`):

| Column Name                          |      Data Type | Unit       | Technical Description                                                                             |
| ------------------------------------ | -------------: | ---------- | ------------------------------------------------------------------------------------------------- |
| `timestamp_utc`                      | datetime (UTC) | -          | Canonical hourly UTC timestamp for joining and modeling.                                          |
| `afrr_capacity_offered_mw_neg`       |          float | MW         | Awarded/accepted standby capacity, negative direction.                                            |
| `afrr_capacity_offered_mw_pos`       |          float | MW         | Awarded/accepted standby capacity, positive direction.                                            |
| `afrr_capacity_price_neg`            |          float | EUR/(MW·h) | Marginal capacity clearing price, negative direction.                                             |
| `afrr_capacity_price_pos`            |          float | EUR/(MW·h) | Marginal capacity clearing price, positive direction.                                             |
| `afrr_activation_marginal_price_neg` |          float | EUR/MWh    | Marginal activation clearing price, negative direction (economically relevant settlement signal). |
| `afrr_activation_marginal_price_pos` |          float | EUR/MWh    | Marginal activation clearing price, positive direction (economically relevant settlement signal). |
| `afrr_activation_avg_price_neg`      |          float | EUR/MWh    | Volume-weighted average activation price, negative direction (diagnostic variable).               |
| `afrr_activation_avg_price_pos`      |          float | EUR/MWh    | Volume-weighted average activation price, positive direction (diagnostic variable).               |
| `afrr_activation_offered_mw_neg`     |          float | MW         | Directional offered/activated MW proxy from ENERGY result files, negative direction.              |
| `afrr_activation_offered_mw_pos`     |          float | MW         | Directional offered/activated MW proxy from ENERGY result files, positive direction.              |
| `net_import_export_mw`               |          float | MW         | Net cross-border balancing exchange indicator (when present in source files).                     |

## 4. Data Preprocessing & Transformations (ETL)

### Ingestion-Level ETL (`fetch_regelleistung.py`)

- Dynamic header normalization aligns German/English naming variants.
- Product strings are parsed for multiple market formats (block products, quarter-hour products, explicit clock ranges).
- Directional pivoting converts long-format data into `_pos`/`_neg` wide columns.
- Hourly harmonization (`resample("1h").mean()`) standardizes output resolution.
- Isolated one-hour gaps (frequent around DST boundaries) are closed with constrained interpolation.
- If yearly files are unavailable or sparse, the script falls back to monthly downloads.

### Missing-Value Handling

- Fetch-stage processing remains source-faithful where possible.
- Final project-wide imputation logic is centralized in `src/energy_trading/processing/handle_missing_values.py`.

### Outlier and Regulatory-Limit Handling

- The fetch script itself does not clip market prices.
- Target-specific cleaning is implemented downstream in feature engineering (`src/energy_trading/features/build_features.py`):
  - Sentinel-like extremes (around `±99,999 €/MWh`) are neutralized when activation is effectively zero.
  - Two targets are maintained for methodological clarity:
    - `y_true_*`: cleaned, **unclipped** marginal target for financial backtesting
    - `y_train_*`: clipped target (e.g., `[-500, 500]`) for robust ML training under MSE-based loss

This separation preserves economic validity for PnL simulation while improving statistical stability in model training.
