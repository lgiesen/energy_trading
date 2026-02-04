"""Fetch ENTSO-E load and outage data for Germany (DE_LU) using entsoe-py.

Usage:
    ./.venv/bin/python -m energy_trading.ingestion.fetch_entsoe \
        --start 2020-12-01T00:00:00Z --end 2026-01-01T02:00:00Z \
        --out data/raw/entsoe.parquet

Outputs:
    - entsoe.parquet with hourly UTC timestamps and:
        timestamp_utc, load_actual, load_forecast_da,
        outage_baseload_mw, outage_hard_coal_mw, outage_gas_mw
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl
from entsoe import EntsoePandasClient
from entsoe.parsers import PSRTYPE_MAPPINGS
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

LOGGER = logging.getLogger(__name__)

COUNTRY_CODE = "DE_LU"

PSR_MAP = {
    "gas": "B04",
    "hard_coal": "B02",
    "lignite": "B11",
    "nuclear": "B14",
}


def _parse_utc(ts: str) -> pd.Timestamp:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return pd.Timestamp(dt).tz_convert("UTC")


def _month_ranges(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ranges = []
    cur = start
    while cur < end:
        nxt = (cur + pd.offsets.MonthBegin(1)).normalize()
        if nxt <= cur:
            nxt = cur + pd.DateOffset(months=1)
        ranges.append((cur, min(nxt, end)))
        cur = nxt
    return ranges


def _retry(func, attempts: int = 3, sleep_s: float = 2.0):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # pragma: no cover - network errors
            last_exc = exc
            LOGGER.warning("Attempt %s failed: %s", attempt, exc)
            if attempt < attempts:
                time.sleep(sleep_s)
    raise RuntimeError(f"Failed after {attempts} attempts") from last_exc


def _ensure_utc_index(series: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if series is None or len(series) == 0:
        return series
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    else:
        series.index = series.index.tz_convert("UTC")
    return series


def _resample_hourly(obj: pd.Series | pd.DataFrame) -> pd.Series:
    obj = _ensure_utc_index(obj)
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            obj = obj.iloc[:, 0]
        else:
            obj = obj.sum(axis=1)
    return obj.resample("1h").mean()


def _events_to_hourly(events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    idx = pd.date_range(start=start, end=end, freq="1h", tz="UTC", inclusive="left")
    out = pd.Series(0.0, index=idx)
    if events is None or events.empty:
        return out

    # Expected columns from entsoe-py: start, end, avail_qty, nominal_power
    for _, row in events.iterrows():
        ev_start = pd.Timestamp(row["start"]).tz_convert("UTC")
        ev_end = pd.Timestamp(row["end"]).tz_convert("UTC")
        nominal = float(row.get("nominal_power", 0) or 0)
        avail = float(row.get("avail_qty", 0) or 0)
        missing = max(nominal - avail, 0.0)
        if missing == 0.0:
            continue

        # Align to hourly boundaries.
        h_start = ev_start.floor("1h")
        h_end = ev_end.ceil("1h")
        if h_end <= start or h_start >= end:
            continue
        h_start = max(h_start, start)
        h_end = min(h_end, end)
        if h_start >= h_end:
            continue
        out.loc[h_start:h_end - pd.Timedelta(hours=1)] += missing

    return out


def _psr_name(psr_code: str) -> str:
    return PSRTYPE_MAPPINGS.get(psr_code, psr_code)


def _filter_by_psr(events: pd.DataFrame, psr_code: str) -> pd.DataFrame:
    if events is None or events.empty:
        return events
    name = _psr_name(psr_code)
    for col in ("plant_type", "production_resource_psr_name"):
        if col in events.columns:
            return events[events[col] == name]
    # If no known PSR column, return empty to avoid mixing types.
    return events.iloc[0:0]


def fetch_chunk(client: EntsoePandasClient, start: pd.Timestamp, end: pd.Timestamp) -> pl.DataFrame:
    LOGGER.info("Fetching %s to %s", start, end)

    load_actual = _retry(lambda: client.query_load(COUNTRY_CODE, start=start, end=end))
    load_forecast = _retry(lambda: client.query_load_forecast(COUNTRY_CODE, start=start, end=end))

    load_actual = _resample_hourly(load_actual)
    load_actual = load_actual.rename("load_actual")
    load_forecast = _resample_hourly(load_forecast)
    load_forecast = load_forecast.rename("load_forecast_da")

    # Outages by PSR type (filter locally; entsoe-py does not accept psr_type)
    events_all = _retry(
        lambda: client.query_unavailability_of_generation_units(
            COUNTRY_CODE,
            start=start,
            end=end,
        )
    )
    outages = {}
    for key, psr in PSR_MAP.items():
        events = _filter_by_psr(events_all, psr)
        outages[key] = _events_to_hourly(events, start, end)

    outage_baseload = (outages["lignite"] + outages["nuclear"]).rename("outage_baseload_mw")
    outage_hard_coal = outages["hard_coal"].rename("outage_hard_coal_mw")
    outage_gas = outages["gas"].rename("outage_gas_mw")

    df = pd.concat(
        [load_actual, load_forecast, outage_baseload, outage_hard_coal, outage_gas],
        axis=1,
    )
    df.index = df.index.tz_convert("UTC")
    df = df.sort_index()

    pl_df = pl.from_pandas(df.reset_index()).rename({"index": "timestamp_utc"})
    return pl_df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fetch ENTSO-E load + outage data (DE_LU).")
    parser.add_argument("--start", required=True, help="Start ISO8601 (UTC).")
    parser.add_argument("--end", required=True, help="End ISO8601 (UTC).")
    parser.add_argument("--out", default="data/raw/entsoe.parquet", help="Output parquet path.")
    args = parser.parse_args()

    # Load .env from repo root if available.
    if load_dotenv is not None:
        load_dotenv(dotenv_path=Path(".env"))

    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ENTSOE_API_KEY environment variable")

    start = _parse_utc(args.start)
    end = _parse_utc(args.end)
    if end <= start:
        raise ValueError("--end must be after --start")

    client = EntsoePandasClient(api_key=api_key)

    frames = []
    for s, e in _month_ranges(start, end):
        frames.append(fetch_chunk(client, s, e))

    merged = pl.concat(frames).unique(subset=["timestamp_utc"], keep="last").sort("timestamp_utc")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(out_path, compression="zstd")
    LOGGER.info("Wrote %s rows to %s", len(merged), out_path)


if __name__ == "__main__":
    main()
