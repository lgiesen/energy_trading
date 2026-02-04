"""Compare missingness before/after cleaning.

Usage:
    ./.venv/bin/python -m energy_trading.processing.compare_missingness \
        --before data/processed/all_data.parquet \
        --after data/processed/all_data_clean.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def _null_report(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.select([pl.col(c).null_count().alias(c) for c in df.columns])
        .unpivot(variable_name="column", value_name="nulls")
        .sort("nulls", descending=True)
    )


def _top_gaps(df: pl.DataFrame, col: str, top_n: int = 3) -> pl.DataFrame:
    if col not in df.columns:
        return pl.DataFrame()
    s = df.select(["timestamp_utc", col]).sort("timestamp_utc")
    s = s.with_columns(pl.col(col).is_null().cast(pl.Int8).alias("is_null"))
    s = s.with_columns((pl.col("is_null") != pl.col("is_null").shift(1)).cast(pl.Int64).cum_sum().alias("run_id"))
    runs = (
        s.group_by("run_id")
        .agg([
            pl.col("is_null").first().alias("is_null"),
            pl.col("timestamp_utc").first().alias("start"),
            pl.col("timestamp_utc").last().alias("end"),
            pl.len().alias("len"),
        ])
        .filter(pl.col("is_null") == 1)
        .sort("len", descending=True)
    )
    return runs.head(top_n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare missingness before/after cleaning.")
    parser.add_argument("--before", required=True, help="Input parquet before cleaning.")
    parser.add_argument("--after", required=True, help="Input parquet after cleaning.")
    parser.add_argument("--top-gaps", type=int, default=3, help="Top N null runs per column.")
    args = parser.parse_args()

    before_path = Path(args.before)
    after_path = Path(args.after)
    if not before_path.exists():
        raise FileNotFoundError(before_path)
    if not after_path.exists():
        raise FileNotFoundError(after_path)

    before = pl.read_parquet(before_path)
    after = pl.read_parquet(after_path)

    print("Before nulls (top 25):")
    print(_null_report(before).head(25))
    print("\nAfter nulls (top 25):")
    print(_null_report(after).head(25))

    # Compare deltas
    b = _null_report(before)
    a = _null_report(after)
    delta = b.join(a, on="column", how="full", suffix="_after").fill_null(0)
    delta = delta.with_columns(
        (pl.col("nulls_after").cast(pl.Int64) - pl.col("nulls").cast(pl.Int64)).alias("delta")
    )
    delta = delta.sort("delta")
    print("\nLargest reductions (delta negative):")
    print(delta.head(15))
    print("\nLargest increases (delta positive):")
    print(delta.sort("delta", descending=True).head(15))

    # Optional: show top gaps for selected columns if present
    for col in ["co2_price_eua", "gas_price_ttf", "coal_price_api", "load_forecast_da", "da_price_d_eur_mwh"]:
        if col in before.columns:
            print(f"\nTop null runs (before) for {col}:")
            print(_top_gaps(before, col, top_n=args.top_gaps))
        if col in after.columns:
            print(f"\nTop null runs (after) for {col}:")
            print(_top_gaps(after, col, top_n=args.top_gaps))


if __name__ == "__main__":
    main()
