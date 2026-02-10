#!/usr/bin/env python3
"""Quick audit for ENTSO-E wind/solar forecasts in data/raw/entsoe.parquet."""
from __future__ import annotations

from pathlib import Path

import polars as pl


PATH = Path("data/raw/entsoe.parquet")
COLS = [
    "wind_onshore_forecast_da_entsoe",
    "wind_onshore_forecast_id_entsoe",
    "solar_forecast_da_entsoe",
    "solar_forecast_id_entsoe",
]


def main() -> None:
    if not PATH.exists():
        raise SystemExit(f"Missing file: {PATH}")

    df = pl.read_parquet(PATH)
    missing_cols = [c for c in COLS if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Missing columns: {missing_cols}")

    print(f"Loaded {PATH} with {df.height} rows")
    print("Column stats:")
    stats = (
        df.select(
            [
                pl.col(c).null_count().alias(f"{c}__nulls")
                for c in COLS
            ]
            + [
                pl.col(c).mean().alias(f"{c}__mean")
                for c in COLS
            ]
            + [
                (pl.col(c) == 0.0).sum().alias(f"{c}__zeros")
                for c in COLS
            ]
        )
        .transpose(include_header=True, column_names=["value"])
    )
    print(stats)

    # Delta check: intraday vs day-ahead (onshore)
    delta = df.select(
        (pl.col("wind_onshore_forecast_id_entsoe") - pl.col("wind_onshore_forecast_da_entsoe"))
        .abs()
        .mean()
        .alias("mean_abs_diff_onshore_id_vs_da")
    )
    print("\nDelta check:")
    print(delta)

    # First 5 rows where intraday differs from day-ahead (onshore)
    sample = (
        df.filter(
            pl.col("wind_onshore_forecast_id_entsoe")
            != pl.col("wind_onshore_forecast_da_entsoe")
        )
        .select(
            [
                pl.col("timestamp_utc"),
                pl.col("wind_onshore_forecast_da_entsoe"),
                pl.col("wind_onshore_forecast_id_entsoe"),
                (pl.col("wind_onshore_forecast_id_entsoe") - pl.col("wind_onshore_forecast_da_entsoe")).alias(
                    "delta"
                ),
            ]
        )
        .head(5)
    )
    print("\nSample rows where intraday differs from day-ahead (onshore):")
    print(sample)


if __name__ == "__main__":
    main()
