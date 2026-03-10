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
- `src/energy_trading/ingestion/merge_data.py`  
  Full join of all raw parquet files on hourly UTC timestamps.

Raw outputs are written to `data/raw/*.parquet`.  
Final merged table is written to `data/processed/all_data.parquet`.

---

## 2) Canonical Command (Recommended)

Run everything end-to-end:

```bash
./.venv/bin/python scripts/collect_and_merge_all_data.py \
  --start 2020-11-30T23:00:00Z \
  --end 2025-12-31T23:00:00Z
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

From Regelleistung TSO aggregation:

- `afrr_avg_activation_price_pos`
- `afrr_avg_activation_price_neg`

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
