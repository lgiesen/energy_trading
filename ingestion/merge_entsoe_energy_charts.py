"""Merge ENTSO-E parquet data with Energy Charts day-ahead prices on timestamp."""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def merge_parquets(entsoe_path: Path, prices_path: Path) -> pl.DataFrame:
    entsoe = pl.read_parquet(entsoe_path)
    prices = pl.read_parquet(prices_path)

    # Ensure timestamp is treated consistently (Polars will keep timezone if present).
    # Use a full join to preserve any timestamps present in either dataset.
    merged = (
        entsoe.join(prices, on="timestamp", how="full", coalesce=True)
        .sort("timestamp")
    )
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge ENTSO-E and Energy Charts day-ahead price parquet files on timestamp.")
    parser.add_argument("--entsoe", default="data/entsoe.parquet", help="Path to entsoe parquet (default: data/entsoe.parquet).")
    parser.add_argument("--prices", default="data/day_ahead_prices.parquet", help="Path to Energy Charts prices parquet (default: data/day_ahead_prices.parquet).")
    parser.add_argument("--out", default="data/entsoe_with_prices.parquet", help="Output parquet path.")
    args = parser.parse_args()

    entsoe_path = Path(args.entsoe)
    prices_path = Path(args.prices)
    if not entsoe_path.exists():
        raise FileNotFoundError(f"ENTSO-E parquet not found: {entsoe_path}")
    if not prices_path.exists():
        raise FileNotFoundError(f"Price parquet not found: {prices_path}")

    merged = merge_parquets(entsoe_path, prices_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(out_path, compression="zstd")
    print(f"Wrote {len(merged)} rows to {out_path}")


if __name__ == "__main__":
    main()
