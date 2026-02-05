"""Verify wind_onshore_forecast_intraday scaling fix.

Usage:
    ./.venv/bin/python scripts/tmp_verify_wind_intraday_fix.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl


def main() -> None:
    cand = [
        Path("data/processed/all_data_transformed.parquet"),
        Path("data/processed/all_data_clean.parquet"),
        Path("data/raw/smard.parquet"),
    ]
    data_path = next((p for p in cand if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError("No input parquet found in data/processed or data/raw.")

    df = pl.read_parquet(data_path)
    for c in ["wind_onshore_forecast_intraday", "wind_onshore_actual"]:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    base = df.select(["wind_onshore_forecast_intraday", "wind_onshore_actual"]).drop_nulls()
    base = base.filter(pl.col("wind_onshore_actual") > 0)

    stats = base.select([
        pl.col("wind_onshore_forecast_intraday").median().alias("intraday_median"),
        pl.col("wind_onshore_forecast_intraday").mean().alias("intraday_mean"),
        pl.col("wind_onshore_actual").median().alias("actual_median"),
        pl.col("wind_onshore_actual").mean().alias("actual_mean"),
    ]).to_dicts()[0]

    ratio = base.select(
        (pl.col("wind_onshore_forecast_intraday") / pl.col("wind_onshore_actual")).median().alias("median_ratio")
    ).item()

    print("Using:", data_path)
    print("Intraday median/mean:", stats["intraday_median"], stats["intraday_mean"])
    print("Actual median/mean:", stats["actual_median"], stats["actual_mean"])
    print("Median ratio (ID/Actual):", ratio)

    if 0.9 <= ratio <= 1.1:
        print("PASS: Median ratio within 0.9–1.1")
    elif ratio > 3.0:
        print("FAIL: Median ratio > 3.0 (scaling still wrong)")
    else:
        print("WARN: Median ratio outside 0.9–1.1")

    print("\nSample rows:")
    print(base.head(10))


if __name__ == "__main__":
    main()
