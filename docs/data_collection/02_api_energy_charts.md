# Energy Charts API Data Source (Day-Ahead Prices)

## 1. Academic Context (The "What" and "Why")
The Energy Charts API (Fraunhofer ISE) provides historical day-ahead electricity prices for European bidding zones. In this thesis, these series are used as exogenous market-state variables to improve short-term forecasting of imbalance-sensitive conditions relevant to BESS trading decisions. They are not treated as balancing settlement prices, but as economically meaningful context features capturing cross-zonal scarcity and coupling effects. [CITATION_EC_1], [CITATION_EC_2]

Including neighboring bidding-zone prices is methodologically justified in integrated electricity markets because cross-border day-ahead price dynamics co-move with congestion, renewable production patterns, and regional supply-demand tightness. This improves feature richness while preserving conceptual separation between spot-price context and balancing-market settlement targets. [CITATION_EC_1], [CITATION_EC_2]

## 2. Technical Architecture (The "How")
### Source Type
- Public REST API over HTTPS
- JSON response payload
- Authentication: none required in current implementation

### Ingestion Implementation
- Script: `src/energy_trading/ingestion/fetch_energy_charts.py`
- Libraries: `requests`, `polars`, `zoneinfo`
- Output: `data/raw/energy_charts.parquet`

### Endpoint
```text
https://api.energy-charts.info/price?start={YYYY-MM-DD}&end={YYYY-MM-DD}&bzn={ZONE}
```

Implementation pattern:
```python
base_url = f"https://api.energy-charts.info/price?start={start}&end={end}"
url = f"{base_url}&bzn={zone}"
```

### Query Parameters and Zone Scope
- `start`, `end`: API date bounds in `YYYY-MM-DD` (derived from UTC CLI inputs).
- `bzn`: bidding-zone code.
- Default zone list in code:
  - `AT`, `BE`, `CH`, `CZ`, `DK1`, `DK2`, `FR`, `NL`, `PL`, `SE4`

Run standalone:
```bash
./.venv/bin/python -m energy_trading.ingestion.fetch_energy_charts \
  --start 2020-12-01T00:00:00Z --end 2026-01-01T02:00:00Z \
  --out data/raw/energy_charts.parquet
```

## 3. Data Dictionary
| Column Name | Data Type | Unit | Technical Description |
|---|---:|---|---|
| `timestamp` | datetime (UTC) | - | Canonical hourly UTC timestamp for joining/merging. |
| `da_price_AT` | float | EUR/MWh | Day-ahead electricity price, Austria. |
| `da_price_BE` | float | EUR/MWh | Day-ahead electricity price, Belgium. |
| `da_price_CH` | float | EUR/MWh | Day-ahead electricity price, Switzerland. |
| `da_price_CZ` | float | EUR/MWh | Day-ahead electricity price, Czech Republic. |
| `da_price_DK1` | float | EUR/MWh | Day-ahead electricity price, Denmark West. |
| `da_price_DK2` | float | EUR/MWh | Day-ahead electricity price, Denmark East. |
| `da_price_FR` | float | EUR/MWh | Day-ahead electricity price, France. |
| `da_price_NL` | float | EUR/MWh | Day-ahead electricity price, Netherlands. |
| `da_price_PL` | float | EUR/MWh | Day-ahead electricity price, Poland. |
| `da_price_SE4` | float | EUR/MWh | Day-ahead electricity price, Sweden SE4. |

## 4. Data Preprocessing & Transformations (ETL)
- Per-zone retrieval via `GET /price`.
- Parse `unix_seconds` and `price` arrays.
- Convert epoch timestamps to timezone-aware UTC datetimes.
- Truncate to hourly bins and aggregate hourly mean where required.
- Build a unified timestamp index and full-join all zones.
- Filter final output strictly to requested UTC `--start/--end` window.
- No aggressive imputation or clipping is applied at fetch stage; downstream handling is centralized in `src/energy_trading/processing/handle_missing_values.py`.

## 5. Reproducibility & Market Constraints
- **Temporal availability:** hourly day-ahead price data are consistently available from approximately **2018 onward** for the zones used in this project.
- **Operational limits:** in current use, no major pagination logic is required because retrieval is date-window based and zone-wise.
- **Zone exclusion decision:** `NO2` is explicitly excluded in the project code due to high historical missingness identified during validation.
- **Reproducibility guidance:** keep zone list and date windows version-controlled in pipeline scripts to ensure deterministic reruns.
