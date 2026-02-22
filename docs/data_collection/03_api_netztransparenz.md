# Netztransparenz API Data Source (System Balance & Activated Volumes)

## 1. Academic Context (The "What" and "Why")
The Netztransparenz platform (German TSO transparency interface) provides high-frequency operational balancing signals that are directly relevant for short-term system-state modeling. In this thesis, these data are used to characterize real-time imbalance conditions, including NRV balance, RZ saldo, reBAP outcomes, and activated aFRR/mFRR volumes. These variables are essential for modeling battery-relevant market stress because they capture physical and economic balancing behavior beyond day-ahead price information.

From a methodological perspective, Netztransparenz adds explanatory variables that are tightly linked to balancing-system dynamics: activated reserve direction and magnitude, aggregate balancing state, and settlement-side imbalance indicators. This supports a more realistic feature space for BESS trading models in the German aFRR context by connecting forecast errors and market conditions to observed balancing activation patterns.

## 2. Technical Architecture (The "How")
### Source Type
- Authenticated HTTPS REST API
- Payload format: CSV (semicolon-delimited)
- Authentication: Bearer token (`NETZTRANSPARENZ_TOKEN`) or OAuth client credentials

### Ingestion Implementation
- Script: `src/energy_trading/ingestion/fetch_netztransparenz.py`
- Libraries: `requests`, `polars`, `python-dotenv`, `zoneinfo`
- Output: `data/raw/netztransparenz.parquet`

### Authentication Flow
Token endpoint:
```text
https://identity.netztransparenz.de/users/connect/token
```
If no token is provided via `--token`/`NETZTRANSPARENZ_TOKEN`, the script requests one using:
- `NETZTRANSPARENZ_CLIENT_ID`
- `NETZTRANSPARENZ_CLIENT_SECRET`

### API Endpoints Queried
Base:
```text
https://ds.netztransparenz.de/api/v1/data
```

Quality-assured endpoints:
```text
/NrvSaldo/NRVSaldo/Qualitaetsgesichert/{start_local}/{end_local}
/NrvSaldo/reBAP/Qualitaetsgesichert/{start_local}/{end_local}
/NrvSaldo/RZSaldo/Qualitaetsgesichert/{start_local}/{end_local}
/NrvSaldo/AktivierteSRL/Qualitaetsgesichert/{start_local}/{end_local}
/NrvSaldo/AktivierteMRL/Qualitaetsgesichert/{start_local}/{end_local}
```

Operational fallback endpoints:
```text
/NrvSaldo/NRVSaldo/Betrieblich/{start_local}/{end_local}
/NrvSaldo/RZSaldo/Betrieblich/{start_local}/{end_local}
```

### Chunking / Request Strategy
To improve reliability and avoid server-side failures, extraction is chunked by date window:
```bash
./.venv/bin/python -m energy_trading.ingestion.fetch_netztransparenz \
  --start 2020-12-01T00:00:00Z --end 2026-01-01T02:00:00Z \
  --chunk-days 30 --chunk-sleep 3 --resample 1h \
  --out data/raw/netztransparenz.parquet
```
- `--chunk-days`: rolling request window size
- `--chunk-sleep`: delay between chunks
- `--resample 1h`: final hourly harmonization

### Additional Source Fallback
The script includes an optional fallback for `rz_saldo_mw` from SMARD (filter 37) if Netztransparenz leaves residual nulls.

## 3. Data Dictionary
Current schema in `data/raw/netztransparenz.parquet`:

| Column Name | Data Type | Unit | Technical Description |
|---|---:|---|---|
| `timestamp_utc` | datetime (UTC) | - | Canonical hourly UTC timestamp. |
| `NRV_balance_qs` | float | MW | NRV balance from quality-assured source. |
| `NRV_balance_op` | float | MW | NRV balance from operational source (fallback candidate). |
| `NRV_balance` | float | MW | Canonical NRV balance after QS/operational coalescing. |
| `reBAP_shortage_surplus` | float | EUR/MWh | reBAP shortage/surplus signal (economic imbalance indicator). |
| `rz_saldo_mw_qs` | float | MW | RZ saldo from quality-assured source. |
| `rz_saldo_mw_op` | float | MW | RZ saldo from operational source (fallback candidate). |
| `rz_saldo_mw` | float | MW | Canonical RZ saldo after QS/operational coalescing. |
| `afrr_activated_mw_pos` | float | MW | Activated aFRR power, positive direction. |
| `afrr_activated_mw_neg` | float | MW | Activated aFRR power, negative direction. |
| `mfrr_activated_mw_pos` | float | MW | Activated mFRR power, positive direction. |
| `mfrr_activated_mw_neg` | float | MW | Activated mFRR power, negative direction. |
| `afrr_activated_mwh_pos` | float | MWh | Hourly activated aFRR energy, positive direction. |
| `afrr_activated_mwh_neg` | float | MWh | Hourly activated aFRR energy, negative direction. |
| `mfrr_activated_mwh_pos` | float | MWh | Hourly activated mFRR energy, positive direction. |
| `mfrr_activated_mwh_neg` | float | MWh | Hourly activated mFRR energy, negative direction. |

## 4. Data Preprocessing & Transformations (ETL)
### Parsing and Normalization
- CSV responses are parsed with robust delimiter and encoding cleanup.
- Date/time fields are converted to timezone-aware datetimes and normalized to UTC.
- Numeric fields are sanitized from localized decimal/thousands formats.

### Canonical Column Construction
- `NRV_balance` is built by coalescing quality-assured and operational sources.
- `rz_saldo_mw` is built by coalescing quality-assured and operational sources.
- Raw provenance columns (`*_qs`, `*_op`) are preserved for auditability.

### Resolution Harmonization
- Source values are ingested at quarter-hour granularity where applicable.
- Power (`MW`) series are resampled to hourly mean.
- Energy (`MWh`) series are computed from quarter-hour MW (`MW * 0.25`) and then summed to hourly totals.

### Missing Data and Fallback Handling
- Chunked fetching and retry logic reduce transport-level gaps.
- Operational fallback is applied specifically to close known QS gaps in 2022 for NRV/RZ series.
- Optional SMARD fallback is used for residual `rz_saldo_mw` gaps.
- Final project-wide imputation policy is centralized downstream in `src/energy_trading/processing/handle_missing_values.py`.

### Outliers / Regulatory Values
- No aggressive clipping/winsorization is applied in this fetcher.
- Outlier treatment is performed later at feature-target stage, depending on model purpose.
