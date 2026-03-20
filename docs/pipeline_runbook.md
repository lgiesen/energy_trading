# Pipeline Runbook

This document explains:

1. what to run,
2. in which order,
3. which files/columns are produced,
4. how to validate that the run is correct.

It is the operational reference for the ingestion + merge pipeline.

---

## 1) Pipeline Structure

Main scripts:

- `src/energy_trading/ingestion/collect_and_merge_all_data.py`  
  Orchestrator (runs all fetchers + merge).
- `src/energy_trading/ingestion/fetch_regelleistung.py`  
  Regelleistung aFRR CAPACITY/ENERGY + optional anonymous-bid derived prices.
- `scripts/fetch_entsoe_outages.py`  
  ENTSO-E planned/unplanned generation outage events (DE-LU).
- `scripts/transform_entsoe_outages_hourly.py`  
  Converts outage events to hourly sidecar features for short-term forecasting.
- `src/energy_trading/ingestion/merge_data.py`  
  Full join of all raw parquet files on hourly UTC timestamps.
- `src/energy_trading/processing/refine_market_data.py`  
  Refines merged data, consolidates sources, and applies market logic.

Raw outputs are written to `data/raw/*.parquet`.  
Merged table is written to `data/processed/all_data.parquet`.  
Refined table is written to `data/processed/all_data_refined.parquet`.  
Cleaned table is written to `data/processed/cleaned_data.parquet`.

### Step-by-step I/O map

| Step | Script | Main Input(s) | Main Output |
|---|---|---|---|
| 1 | `fetch_entsoe.py` | ENTSO-E API (`--start`, `--end`) | `data/raw/entsoe.parquet` |
| 2 | `fetch_energy_charts.py` | Energy-Charts API | `data/raw/energy_charts.parquet` |
| 3 | `fetch_netztransparenz.py` | Netztransparenz API | `data/raw/netztransparenz.parquet` (15-min activation volumes) |
| 4 | `fetch_smard.py` | SMARD API (+ optional market-data CSV) | `data/raw/smard.parquet` (+ `data/raw/installed_capacity.csv`) |
| 5 | `fetch_yfinance.py` | Yahoo Finance API | `data/raw/yfinance.parquet` |
| 6 | `fetch_regelleistung.py` | Regelleistung API + bid files in `data/raw/bids/` | `data/raw/regelleistung.parquet` (hourly) + `data/raw/regelleistung_15min/afrr_prices_15min.parquet` + `data/raw/regelleistung_15min/afrr_volumes_15min.parquet` + `data/raw/regelleistung_15min/afrr_price_volume_15min.parquet` |
| 7 | `fetch_entsoe_outages.py` | ENTSO-E API (`A80` + `A53/A54`) | `data/raw/entsoe_outages/planned_generation_outages.parquet` + `data/raw/entsoe_outages/unplanned_generation_outages.parquet` |
| 8 | `merge_data.py` | All `data/raw/*.parquet` files | `data/processed/all_data.parquet` |
| 9 | `refine_market_data.py` | `data/processed/all_data.parquet` | `data/processed/all_data_refined.parquet` |
| 10 | `handle_missing_values.py` | `data/processed/all_data_refined.parquet` | `data/processed/cleaned_data.parquet` |
| 11 | `transform_data.py` | `data/processed/cleaned_data.parquet` | `data/processed/all_data_transformed.parquet` |
| 12 | `build_features.py` | `data/processed/all_data_transformed.parquet` | `data/features/all_data_features.parquet` |
| 13 | `transform_entsoe_outages_hourly.py` | `data/raw/entsoe_outages/planned_generation_outages.parquet` + `data/raw/entsoe_outages/unplanned_generation_outages.parquet` | `data/processed/outages_hourly.parquet` |

---

## Design Principles

- Separation of concerns:
  Fetchers mirror raw API data and avoid domain-specific transformations.
- Market logic in processing:
  Processing scripts implement source consolidation, structural-break logic,
  and feature engineering in a traceable layer.
- Reproducibility:
  Intermediate artifacts (`all_data.parquet`, `all_data_refined.parquet`,
  `cleaned_data.parquet`) make every transformation stage auditable.

## 2) Canonical Command (Recommended)

Run everything end-to-end:

```bash
./.venv/bin/python scripts/collect_and_merge_all_data.py \
  --start 2020-11-30T23:00:00Z \
  --end 2025-12-31T23:00:00Z
```

Then fetch outage events and build hourly outage sidecar features:

```bash
./.venv/bin/python scripts/fetch_entsoe_outages.py --days-ahead 7
./.venv/bin/python scripts/transform_entsoe_outages_hourly.py --days-ahead 7
```

This corresponds to:

- Start: `2020-12-01 00:00:00 CET`
- End: `2026-01-01 00:00:00 CET`

Important behavior:

- The collector fetches with a one-day lookback (`start - 1 day`) to reduce boundary losses.
- `merge_data.py` clips back to the exact `--start/--end` window.

---

## 3) Useful Variants

Run only post-collection processing (if raw files are already collected):

```bash
./.venv/bin/python scripts/post_collection_pipeline.py \
  --clip-start 2020-11-30T23:00:00Z \
  --clip-end 2025-12-31T23:00:00Z
```

`post_collection_pipeline.py` now also runs outage-event transformation to:

- `data/processed/outages_hourly.parquet`

if both outage inputs exist:

- `data/raw/entsoe_outages/planned_generation_outages.parquet`
- `data/raw/entsoe_outages/unplanned_generation_outages.parquet`

To disable this sidecar step:

```bash
./.venv/bin/python scripts/post_collection_pipeline.py \
  --clip-start 2020-11-30T23:00:00Z \
  --clip-end 2025-12-31T23:00:00Z \
  --skip-outages-hourly
```

Skip anonymous-bid activation price reconstruction (faster):

```bash
./.venv/bin/python scripts/collect_and_merge_all_data.py \
  --start 2020-11-30T23:00:00Z \
  --end 2025-12-31T23:00:00Z \
  --skip-bid-activation-prices
```

Run Regelleistung only:

```bash
./.venv/bin/python -m energy_trading.ingestion.fetch_regelleistung \
  --start 2020-11-30T23:00:00Z \
  --end 2025-12-31T23:00:00Z \
  --out data/raw/regelleistung.parquet \
  --bids-dir data/raw/bids
```

Run merge only:

```bash
./.venv/bin/python -m energy_trading.ingestion.merge_data \
  --data-dir data/raw \
  --out data/processed/all_data.parquet \
  --clip-start 2020-11-30T23:00:00Z \
  --clip-end 2025-12-31T23:00:00Z
```

---

## 4) Timestamp and Timezone Rules

- Command-line bounds should be passed as explicit UTC ISO strings (`...Z`).
- `merge_data.py` interprets naive clip timestamps as `Europe/Berlin`; avoid that and always pass `Z`.
- Final merged file contains:
  - `timestamp_utc`
  - `timestamp_cet`
- Internal join key is hourly UTC.

---

## 5) Activation Price Columns (What They Mean)

Activation prices (work prices) are sourced exclusively from
`regelleistung.net` (directly and bid-derived VWAP variants). ENTSO-E CBMP is
not part of the production pipeline.

From Regelleistung TSO aggregation:

- `afrr_avg_activation_price_pos`
- `afrr_avg_activation_price_neg`
- `afrr_vwap_pos`
- `afrr_vwap_neg`

From anonymous bids (derived in `fetch_regelleistung.py`):

- `afrr_bid_vwap_activation_price_pos`
- `afrr_bid_vwap_activation_price_neg`
- `afrr_bid_avg_activation_price_pos`
- `afrr_bid_avg_activation_price_neg`
- `bid_alloc_mw_pos`
- `bid_alloc_mw_neg`

Implementation note:

- `afrr_bid_vwap_activation_price_*` is volume-weighted from accepted anonymous bids.
- `afrr_bid_avg_activation_price_*` is currently derived as an extremum proxy on signed bid prices
  (`POS -> max`, `NEG -> min`) after payment-direction sign handling.
- Primary model work-price VWAP is computed from 15-minute Regelleistung price/volume
  exports:
  `data/raw/regelleistung_15min/afrr_price_volume_15min.parquet`.
- VWAP is calculated in two stages:
  1. `fetch_regelleistung.py` (hourly export convenience),
  2. `refine_market_data.py` (processing-stage recomputation/join for canonical use).

---

## 6) Validation Checklist (After Every Full Run)

### A) Basic row/time checks

Confirm merged output exists and has expected time window:

```bash
./.venv/bin/python - <<'PY'
import polars as pl
df = pl.read_parquet("data/processed/all_data.parquet")
print("rows:", df.height)
print("start_utc:", df["timestamp_utc"].min())
print("end_utc:", df["timestamp_utc"].max())
PY
```

### B) Null count for activation columns

```bash
./.venv/bin/python - <<'PY'
import polars as pl
cols = [
    "afrr_avg_activation_price_pos",
    "afrr_avg_activation_price_neg",
    "afrr_bid_vwap_activation_price_pos",
    "afrr_bid_vwap_activation_price_neg",
    "afrr_bid_avg_activation_price_pos",
    "afrr_bid_avg_activation_price_neg",
]
df = pl.read_parquet("data/processed/all_data.parquet")
for c in cols:
    if c in df.columns:
        print(c, int(df[c].null_count()))
    else:
        print(c, "MISSING")
PY
```

### C) Where nulls occur (time ranges)

```bash
./.venv/bin/python - <<'PY'
import polars as pl
target = "afrr_bid_vwap_activation_price_pos"
df = pl.read_parquet("data/processed/all_data.parquet")
if target in df.columns:
    n = df.filter(pl.col(target).is_null())
    print("null_rows:", n.height)
    if n.height:
        print("first_null:", n["timestamp_utc"].min())
        print("last_null:", n["timestamp_utc"].max())
else:
    print("missing column:", target)
PY
```

---

## 7) Common Failure Signatures

- `Missing columns ... bid_vwap ...`  
  Usually stale notebook logic vs renamed columns. Use the names listed in section 5.

- Very long Regelleistung run  
  First run parses all bid files; subsequent runs should be faster due to
  `data/raw/bids/_afrr_bid_hourly_cache.parquet`.

- `Bid cache schema mismatch` warning  
  Indicates cache invalidation after code/schema changes. One slow rebuild is expected.

- Sparse coverage in specific years/months  
  Check logs for yearly-file fallback to monthly files and missing monthly files.

---

## 8) Documentation Best Practices for Thesis Artifacts

Include:

- exact command used,
- git commit hash,
- exact timeframe (UTC + CET explanation),
- column definitions used in analysis,
- number of rows and nulls for key columns.

Exclude:

- raw debug dumps,
- temporary aliases/legacy column names if not used in final results,
- partial experiments without final interpretation.
