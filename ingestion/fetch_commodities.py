"""Fetch commodity prices (TTF gas, EUA CO2, API2 coal) from Yahoo Finance and store as parquet.

Tickers:
- gas_price_ttf: TTF=F
- co2_price_eua: CFI=F
- coal_price_api2: MTF=F
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf

TICKERS: Dict[str, str] = {
    "gas_price_ttf": "TTF=F",
    "co2_price_eua": "CFI=F",
    "coal_price_api2": "MTF=F",
}


def fetch_prices(start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Download adjusted close prices for all tickers."""
    frames = []
    for col_name, ticker in TICKERS.items():
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            print(f"Warning: no data for {ticker}")
            continue
        df = df.reset_index().rename(columns={"Date": "timestamp", "Adj Close": col_name})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        frames.append(df[["timestamp", col_name]])

    if not frames:
        raise RuntimeError("No data downloaded for any ticker.")

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="timestamp", how="outer")
    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Fetch TTF gas, EUA CO2, and API2 coal prices from Yahoo Finance.")
    parser.add_argument("--start", default="2022-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD).")
    parser.add_argument("--out", default="data/commodities.parquet", help="Output parquet path.")
    parser.add_argument("--interval", default="1d", help="Yahoo interval (default 1d).")
    args = parser.parse_args()

    df = fetch_prices(args.start, args.end, interval=args.interval)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd", index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
