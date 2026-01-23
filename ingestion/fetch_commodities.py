"""Fetch commodity prices (TTF gas, CO2, API2 coal) from Yahoo Finance and store as parquet."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf
# Monkey-patch Timestamp.utcnow to the recommended Timestamp.now('UTC') until yfinance/pandas switch.
pd.Timestamp.utcnow = staticmethod(lambda: pd.Timestamp.now("UTC"))

# Yahoo tickers (adjusted close):
# - TTF gas: TTF=F
# - CO2: try fallbacks and pick the first that returns data (many symbols get delisted)
# - API2 coal: MTF=F
BASE_TICKERS: Dict[str, str] = {
    "gas_price_ttf": "TTF=F",
    "coal_price_api2": "MTF=F",
}
EUA_CANDIDATES = ["CO2.L", "CHEC.SW", "CBU2.DE", "EUA=F", "C02.F", "CO2.DE", "ECF2.EX"]


def fetch_prices(start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Download adjusted close prices for all tickers."""
    tickers = dict(BASE_TICKERS)  # local copy
    frames = []
    # Resolve EUA ticker by trying candidates in order.
    eua_symbol = None
    for cand in EUA_CANDIDATES:
        test = yf.download(
            cand,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="column",
        )
        if not test.empty and ("Adj Close" in test.columns or ("Adj Close" in test.columns.get_level_values(0) if isinstance(test.columns, pd.MultiIndex) else False)):
            eua_symbol = cand
            break
    if eua_symbol:
        tickers["co2_price_eua"] = eua_symbol
    else:
        print("Warning: no EUA ticker returned data; skipping CO2 series.")

    for col_name, ticker in tickers.items():
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="column",  # single-level columns
        )
        if df.empty:
            print(f"Warning: no data for {ticker}")
            continue
        # Flatten possible multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if "Adj Close" not in df.columns:
            print(f"Warning: missing Adj Close for {ticker}; skipping.")
            continue
        df = df.reset_index().rename(columns={"Date": "timestamp", "Adj Close": col_name})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # Deduplicate per timestamp and set index to simplify concat.
        df = df[["timestamp", col_name]].drop_duplicates(subset=["timestamp"]).set_index("timestamp")
        frames.append(df)

    if not frames:
        raise RuntimeError("No data downloaded for any ticker.")

    merged = pd.concat(frames, axis=1, sort=False).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.reset_index()


def upsample_daily_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Repeat daily prices across hourly slots (forward-fill) to align with hourly datasets."""
    if df.empty:
        return df
    start = df["timestamp"].min().floor("D")
    end = df["timestamp"].max().ceil("D")
    hourly_index = pd.date_range(start, end, freq="1h", tz="UTC")
    df = (
        df.sort_values("timestamp")
        .set_index("timestamp")
        .reindex(hourly_index, method="ffill")
        .reset_index()
        .rename(columns={"index": "timestamp"})
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch TTF gas, EUA CO2, and API2 coal prices from Yahoo Finance.")
    parser.add_argument("--start", default="2022-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD).")
    parser.add_argument("--out", default="data/commodities.parquet", help="Output parquet path.")
    parser.add_argument("--interval", default="1d", help="Yahoo interval (default 1d).")
    args = parser.parse_args()

    df = fetch_prices(args.start, args.end, interval=args.interval)
    df = upsample_daily_to_hourly(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd", index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
