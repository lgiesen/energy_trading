# Yahoo Finance Data Source (Commodity Exogenous Features)

## 1. Academic Context (The "What" and "Why")
Yahoo Finance is used in this thesis as an external macro-commodity data source to capture fuel and carbon cost regimes that influence electricity-market behavior and balancing conditions. Specifically, the dataset provides time series for TTF gas, API2 coal, and EUA CO2 prices, which are economically relevant drivers of marginal generation costs and thus of short-run power-system price formation.

From a methodological perspective, these variables are treated as **exogenous explanatory features** (not target variables). Their inclusion improves model robustness by accounting for structural shifts in commodity and emissions markets that affect thermal dispatch incentives, scarcity patterns, and the broader opportunity-cost environment for BESS trading strategies.

## 2. Technical Architecture (The "How")
### Source Type
- Unofficial market-data access via **Yahoo Finance** through the Python `yfinance` client
- Source transport handled internally by `yfinance` (`yf.download`)
- Authentication: none required in current implementation

### Ingestion Implementation
- Script: `src/energy_trading/ingestion/fetch_yfinance.py`
- Libraries: `yfinance`, `pandas`
- Output: `data/raw/yfinance.parquet`

### Extraction Logic
The script downloads adjusted close series with ticker fallback logic:

- `gas_price`: `TTF=F`
- `coal_price`: `MTF=F`
- `co2_price`: `CO2.L`, fallback `CBU2.DE`

Core call pattern:
```python
yf.download(
    ticker,
    start=start,
    end=end,
    interval="1d",
    auto_adjust=False,
    progress=False,
    threads=False,
)
```

### Query Parameters
Example run:
```bash
./.venv/bin/python -m energy_trading.ingestion.fetch_yfinance \
  --start 2020-11-30T00:00:00Z --end 2026-01-01T02:00:00Z \
  --out data/raw/yfinance.parquet
```

- `--start`, `--end`: UTC window used for clipping final hourly output
- `--interval`: Yahoo interval (default `1d`)

### Availability / Failure Behavior
- If a ticker fails (stale/delisted/unavailable), the script logs warnings and tries configured fallbacks.
- If no ticker is usable for a commodity in a run, schema stability is preserved by emitting the column with nulls.

## 3. Data Dictionary
Current output schema in `data/raw/yfinance.parquet`:

| Column Name | Data Type | Unit | Technical Description |
|---|---:|---|---|
| `timestamp` | datetime (UTC) | - | Canonical hourly UTC timestamp after upsampling. |
| `gas_price` | float | (market quote units) | TTF gas adjusted close series from Yahoo ticker mapping. |
| `coal_price` | float | (market quote units) | API2 coal adjusted close series from Yahoo ticker mapping. |
| `co2_price` | float | (market quote units) | EUA CO2 adjusted close series from Yahoo ticker mapping/fallback. |

> Note: The script stores **adjusted close values as provided by Yahoo Finance**. Unit conventions depend on ticker metadata and should be documented in thesis tables as source-reported market quote units.

## 4. Data Preprocessing & Transformations (ETL)
### Ingestion-Level ETL (`fetch_yfinance.py`)
1. Download daily adjusted close (`Adj Close`) for each candidate ticker.
2. Coalesce fallback tickers per commodity (first non-null candidate).
3. Build a unified commodity table with stable column schema.

### Temporal Normalization
- Source daily values are converted to UTC timestamps.
- Data are upsampled from daily to hourly by forward fill:
  - reindex to daily grid and fill gaps
  - expand to hourly grid and forward fill
- Final output is clipped exactly to requested `--start/--end` UTC window.

### Missing Values
- Fetch-level missingness is handled conservatively (fallback tickers + schema preservation).
- Additional imputation policy is applied downstream in `src/energy_trading/processing/handle_missing_values.py` (e.g., weekend carry-forward in the broader pipeline).

### Outliers / Regulatory Limits
- No winsorization/clipping is applied in this source fetcher.
- Commodity-specific outlier treatment is intentionally deferred to downstream processing/modeling layers.

This separation ensures source-level reproducibility while keeping statistical assumptions explicit in the processing stage.
