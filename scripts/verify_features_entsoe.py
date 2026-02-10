#!/usr/bin/env python3
"""Verify ENTSO-E-based forecast errors and deltas in the final dataset.

Usage:
    ./.venv/bin/python scripts/verify_features_entsoe.py \
        --input data/processed/all_data_transformed.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


ENTSOE_COLS = [
    "wind_onshore_actual_entsoe",
    "wind_onshore_forecast_da_entsoe",
    "wind_onshore_forecast_id_entsoe",
]

FEATURE_COLS = [
    "wind_onshore_error_da",
    "wind_onshore_error_id",
    "wind_onshore_forecast_delta",
]


def _require_columns(df: pl.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing {label} columns: {missing}")


def _max_abs_diff(expr: pl.Expr) -> float:
    return (
        pl.select(expr.abs().max().alias("max_abs"))["max_abs"][0]
        if isinstance(expr, pl.Expr)
        else 0.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ENTSO-E forecast error features.")
    parser.add_argument(
        "--input",
        default="data/features/all_data_features.parquet",
        help="Input parquet to verify (default: data/features/all_data_features.parquet).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pl.read_parquet(input_path)
    if "timestamp_utc" in df.columns:
        df = df.sort("timestamp_utc")

    try:
        _require_columns(df, ENTSOE_COLS, "ENTSO-E source")
        _require_columns(df, FEATURE_COLS, "feature")
    except KeyError as exc:
        entsoe_like = [c for c in df.columns if "entsoe" in c.lower()]
        wind_solar_like = [c for c in df.columns if "wind" in c.lower() or "solar" in c.lower()]
        msg = [
            str(exc),
            "",
            "Diagnostics:",
            f"- entsoe-like columns: {entsoe_like or 'none'}",
            f"- wind/solar-like columns: {wind_solar_like or 'none'}",
            "",
            "This usually means you are verifying a pre-feature dataset or a stale pipeline output.",
            "Recommended sequence:",
            "1) Merge raw data -> data/processed/all_data.parquet",
            "2) Clean -> data/processed/all_data_clean.parquet",
            "3) Transform -> data/processed/all_data_transformed.parquet",
            "4) Build features -> data/features/all_data_features.parquet",
            "",
            "If ENTSO-E actuals are present with tuple-like names, fix fetch_entsoe to rename them,",
            "then rerun steps 1-4.",
        ]
        raise KeyError("\n".join(msg)) from None

    # Manual checks.
    manual_error_da = pl.col("wind_onshore_actual_entsoe") - pl.col("wind_onshore_forecast_da_entsoe")
    manual_error_id = pl.col("wind_onshore_actual_entsoe") - pl.col("wind_onshore_forecast_id_entsoe")
    manual_delta = pl.col("wind_onshore_forecast_id_entsoe") - pl.col("wind_onshore_forecast_da_entsoe")

    diff_error_da = pl.col("wind_onshore_error_da") - manual_error_da
    diff_error_id = pl.col("wind_onshore_error_id") - manual_error_id
    diff_delta = pl.col("wind_onshore_forecast_delta") - manual_delta

    max_abs_da = df.select(diff_error_da.abs().max().alias("max_abs"))["max_abs"][0]
    max_abs_id = df.select(diff_error_id.abs().max().alias("max_abs"))["max_abs"][0]
    max_abs_delta = df.select(diff_delta.abs().max().alias("max_abs"))["max_abs"][0]

    # Summary stats
    mae_da = df.select((pl.col("wind_onshore_error_da").abs().mean()).alias("mae"))["mae"][0]
    mae_id = df.select((pl.col("wind_onshore_error_id").abs().mean()).alias("mae"))["mae"][0]

    # Sample table
    sample = (
        df.with_columns(diff_error_da.alias("check_diff"))
        .select(
            [
                "timestamp_utc",
                "wind_onshore_actual_entsoe",
                "wind_onshore_forecast_da_entsoe",
                "wind_onshore_error_da",
                "check_diff",
            ]
        )
        .head(5)
    )

    print("ENTSO-E verification: wind_onshore")
    print(f"Max abs diff (error_da): {max_abs_da:.6g}")
    print(f"Max abs diff (error_id): {max_abs_id:.6g}")
    print(f"Max abs diff (delta):    {max_abs_delta:.6g}")
    print(f"MAE error_da: {mae_da:.2f} MW")
    print(f"MAE error_id: {mae_id:.2f} MW")
    print("")
    print("Sample (first 5 rows):")
    print(sample)

    tol = 1e-6
    ok = (max_abs_da <= tol) and (max_abs_delta <= tol)
    if ok:
        print("\nPASS: feature calculations match ENTSO-E inputs within tolerance.")
        sys.exit(0)

    print("\nFAIL: feature calculations do not match ENTSO-E inputs.")
    sys.exit(1)


if __name__ == "__main__":
    main()
