"""
Fetch commodity prices (TTF gas, CO2, API2 coal) from Yahoo Finance and store as parquet.

Usage:
    ./.venv/bin/python -m energy_trading.ingestion.fetch_yfinance \
        --start 2020-11-30T23:00:00Z --end 2025-12-31T23:00:00Z \
        --out data/raw/yfinance.parquet

        Outputs:
    - yfinance.parquet with daily prices upsampled to hourly (ffill).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf

# Monkey-patch Timestamp.utcnow to the recommended Timestamp.now('UTC') until yfinance/pandas switch.
pd.Timestamp.utcnow = staticmethod(lambda: pd.Timestamp.now("UTC"))
LOGGER = logging.getLogger(__name__)

# Yahoo tickers (adjusted close) with fallbacks to reduce missingness:
# - TTF gas
# - API2 coal
# - CO2 EUA
CANDIDATE_TICKERS: Dict[str, list[str]] = {
    "gas_price_ttf": ["TTF=F"],
    "coal_price_api2": ["MTF=F"],
    "co2_price_eua": ["CO2.L", "CBU2.DE"],
}


def _configure_third_party_logging() -> None:
    # Keep pipeline logs readable: yfinance emits error logs for stale/delisted symbols.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def _download_adj_close(ticker: str, start: str, end: str, interval: str) -> pd.Series | None:
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )
    except Exception as exc:
        LOGGER.warning("Failed to download %s: %s", ticker, exc)
        return None
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if "Adj Close" not in df.columns:
        return None
    s = df["Adj Close"].copy()
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def fetch_prices(start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Download adjusted close prices using multiple tickers per series and coalesce."""
    series_list = []

    for col_name, tickers in CANDIDATE_TICKERS.items():
        candidates = []
        for t in tickers:
            s = _download_adj_close(t, start, end, interval)
            if s is None or s.empty:
                LOGGER.warning("No data for %s (%s)", col_name, t)
                continue
            candidates.append(s.rename(t))
        if not candidates:
            LOGGER.warning("No usable ticker for %s", col_name)
            continue
        df_cand = pd.concat(candidates, axis=1, sort=False).sort_index()
        # Coalesce across candidate tickers (first non-null).
        df_cand[col_name] = df_cand.bfill(axis=1).iloc[:, 0]
        series_list.append(df_cand[[col_name]])

    if not series_list:
        # Keep schema stable even when Yahoo has no usable symbols in this run.
        idx = pd.date_range(start=pd.to_datetime(start, utc=True), end=pd.to_datetime(end, utc=True), freq="1D")
        empty = pd.DataFrame({"timestamp": idx})
        for col_name in CANDIDATE_TICKERS:
            empty[col_name] = pd.NA
        return empty

    merged = pd.concat(series_list, axis=1, sort=False).sort_index()
    for col_name in CANDIDATE_TICKERS:
        if col_name not in merged.columns:
            merged[col_name] = pd.NA
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.reset_index()
    if "timestamp" not in merged.columns:
        if "index" in merged.columns:
            merged = merged.rename(columns={"index": "timestamp"})
        elif "Date" in merged.columns:
            merged = merged.rename(columns={"Date": "timestamp"})
    return merged


def upsample_daily_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Repeat daily prices across hourly slots (forward-fill) to align with hourly datasets."""
    if df.empty:
        return df
    start = df["timestamp"].min().floor("D")
    end = df["timestamp"].max().ceil("D")
    hourly_index = pd.date_range(start, end, freq="1h", tz="UTC")
    df = df.sort_values("timestamp").set_index("timestamp")
    # Fill daily gaps before hourly expansion.
    daily_index = pd.date_range(start, end, freq="1D", tz="UTC")
    df = df.reindex(daily_index, method="ffill")
    # Backfill leading gaps if series starts after requested window.
    df = df.bfill()
    df = df.reindex(hourly_index, method="ffill").reset_index().rename(columns={"index": "timestamp"})
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _configure_third_party_logging()
    parser = argparse.ArgumentParser(description="Fetch TTF gas, EUA CO2, and API2 coal prices from Yahoo Finance.")
    parser.add_argument("--start", required=True, help="Start ISO8601 (UTC).")
    parser.add_argument("--end", required=True, help="End ISO8601 (UTC).")
    parser.add_argument("--out", default="data/raw/yfinance.parquet", help="Output parquet path.")
    parser.add_argument("--interval", default="1d", help="Yahoo interval (default 1d).")
    args = parser.parse_args()

    start_dt = pd.to_datetime(args.start, utc=True)
    end_dt = pd.to_datetime(args.end, utc=True)
    # yfinance expects naive dates; use UTC dates without timezone.
    df = fetch_prices(start_dt.tz_localize(None), end_dt.tz_localize(None), interval=args.interval)
    df = upsample_daily_to_hourly(df)
    # Enforce UTC, hourly truncation, and clip
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("1h")
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)].sort_values("timestamp")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd", index=False)
    LOGGER.info("Wrote %s rows to %s", len(df), out_path)


if __name__ == "__main__":
    main()
