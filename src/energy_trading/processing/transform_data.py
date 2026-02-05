"""Data transformation step between cleaning and feature engineering.

Usage:
    ./.venv/bin/python -m energy_trading.processing.transform_data \
        --in data/processed/all_data_clean.parquet \
        --out data/processed/all_data_transformed.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def transform_data(input_path: Path, output_path: Path) -> None:
    """Apply deterministic transforms (no imputation)."""
    df = pl.read_parquet(input_path)

    # Example transforms (extend as needed): log1p on strictly positive price columns.
    log_cols = [c for c in ["da_price_d_eur_mwh", "da_price_eur", "price_intraday_eur"] if c in df.columns]
    if log_cols:
        df = df.with_columns([pl.col(c).log1p().alias(f"{c}_log1p") for c in log_cols])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transform cleaned data before feature engineering.")
    parser.add_argument(
        "--in",
        dest="input_path",
        default="data/processed/all_data_clean.parquet",
        help="Input parquet (default: data/processed/all_data_clean.parquet).",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default="data/processed/all_data_transformed.parquet",
        help="Output parquet (default: data/processed/all_data_transformed.parquet).",
    )
    args = parser.parse_args()
    transform_data(Path(args.input_path), Path(args.output_path))


if __name__ == "__main__":
    main()
