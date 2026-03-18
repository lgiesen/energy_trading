"""Handle missing values for refined market data.

Usage:
    ./.venv/bin/python -m energy_trading.processing.handle_missing_values \
        --in data/processed/all_data_refined.parquet \
        --out data/processed/cleaned_data.parquet
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

LOGGER = logging.getLogger(__name__)
PICASSO_START_UTC = datetime(2022, 6, 22, 22, 0, tzinfo=timezone.utc)


def _existing(df: pl.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def clean(df: pl.DataFrame) -> pl.DataFrame:
    if "timestamp_utc" not in df.columns:
        raise ValueError("Missing required column: timestamp_utc")

    cutoff = pl.lit(PICASSO_START_UTC).cast(pl.Datetime(time_unit="us", time_zone="UTC"))

    # Structural-break columns: set to zero before platform launch.
    platform_cols = _existing(
        df,
        [
            "afrr_picasso_mw_pos",
            "afrr_picasso_mw_neg",
            "afrr_picasso_net_mw",
            "mfrr_mari_mw_pos",
            "mfrr_mari_mw_neg",
            "mfrr_mari_net_mw",
        ],
    )
    if platform_cols:
        df = df.with_columns(
            [
                pl.when(pl.col("timestamp_utc") < cutoff)
                .then(pl.lit(0.0))
                .otherwise(pl.col(c))
                .cast(pl.Float64, strict=False)
                .alias(c)
                for c in platform_cols
            ]
        )
    LOGGER.info("Applied structural-break zeroing to columns: %s", platform_cols)

    # Binary regime feature.
    df = df.with_columns(
        pl.when(pl.col("timestamp_utc") >= cutoff)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int8)
        .alias("is_picasso_active")
    )

    # Price imputation: forward then backward fill.
    price_cols = _existing(df, ["da_price_eur", "afrr_avg_activation_price_pos"])
    if price_cols:
        df = df.with_columns(
            [
                pl.col(c)
                .fill_null(strategy="forward")
                .fill_null(strategy="backward")
                .cast(pl.Float64, strict=False)
                .alias(c)
                for c in price_cols
            ]
        )
    LOGGER.info("Applied forward/backward fill to price columns: %s", price_cols)

    # Physical flows/errors: conservative zero fill.
    flow_error_cols = [
        c
        for c in df.columns
        if (
            c.endswith("_mw")
            or "_error" in c
            or c in {"wind_forecast_update", "afrr_picasso_churn_mw"}
        )
    ]
    if flow_error_cols:
        df = df.with_columns(
            [pl.col(c).fill_null(0.0).cast(pl.Float64, strict=False).alias(c) for c in flow_error_cols]
        )
    LOGGER.info("Applied zero-fill to physical flow/error columns: %s", flow_error_cols)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Handle missing values in refined market data.")
    parser.add_argument(
        "--in",
        dest="input_path",
        default="data/processed/all_data_refined.parquet",
        help="Input refined parquet path.",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default="data/processed/cleaned_data.parquet",
        help="Output cleaned parquet path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input parquet: {input_path}")

    df = pl.read_parquet(input_path).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)
    ).sort("timestamp_utc")
    LOGGER.info("Loaded %s rows and %s columns from %s", df.height, len(df.columns), input_path)

    cleaned = clean(df)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.write_parquet(output_path, compression="zstd")
    LOGGER.info("Wrote %s rows and %s columns to %s", cleaned.height, len(cleaned.columns), output_path)


if __name__ == "__main__":
    main()
