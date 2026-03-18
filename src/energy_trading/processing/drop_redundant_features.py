"""Refine merged dataset by dropping SMARD redundancies and adding ENTSO-E errors.

Usage:
    ./.venv/bin/python -m energy_trading.processing.drop_redundant_features \
        --in data/processed/all_data.parquet \
        --out data/processed/all_data_refined.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

LOGGER = logging.getLogger(__name__)

DROP_COLS = [
    "wind_onshore_actual",
    "wind_offshore_actual",
    "solar_actual",
    "wind_onshore_forecast",
    "wind_offshore_forecast",
    "solar_forecast",
    "wind_onshore_error",
    "wind_offshore_error",
    "solar_error",
]

ERROR_FEATURE_SPECS: dict[str, tuple[str, str]] = {
    "wind_onshore_error_da": ("wind_onshore_forecast_da_entsoe", "wind_onshore_actual_entsoe"),
    "wind_offshore_error_da": ("wind_offshore_forecast_da_entsoe", "wind_offshore_actual_entsoe"),
    "solar_error_da": ("solar_forecast_da_entsoe", "solar_actual_entsoe"),
    "wind_onshore_forecast_update": ("wind_onshore_forecast_id_entsoe", "wind_onshore_forecast_da_entsoe"),
    "solar_forecast_update": ("solar_forecast_id_entsoe", "solar_forecast_da_entsoe"),
}

KEEP_FOR_STRUCTURAL_ANALYSIS = [
    "NRV_balance_qs",
    "NRV_balance_op",
    "rz_saldo_mw_qs",
    "rz_saldo_mw_op",
]


def _build_feature_exprs(df: pl.DataFrame) -> tuple[list[pl.Expr], list[str], list[str]]:
    exprs: list[pl.Expr] = []
    added: list[str] = []
    skipped: list[str] = []

    for out_col, (left_col, right_col) in ERROR_FEATURE_SPECS.items():
        if left_col in df.columns and right_col in df.columns:
            exprs.append(
                (pl.col(left_col).cast(pl.Float64, strict=False) - pl.col(right_col).cast(pl.Float64, strict=False)).alias(out_col)
            )
            added.append(out_col)
        else:
            skipped.append(out_col)
            LOGGER.warning(
                "Skipping %s: missing input(s) %s, %s",
                out_col,
                left_col,
                right_col,
            )
    return exprs, added, skipped


def refine_dataset(df: pl.DataFrame) -> pl.DataFrame:
    if "timestamp_utc" not in df.columns:
        raise ValueError("Missing required column: timestamp_utc")

    drop_existing = [c for c in DROP_COLS if c in df.columns]
    df_out = df.drop(drop_existing) if drop_existing else df
    LOGGER.info("Dropped %s redundant SMARD columns.", len(drop_existing))
    if drop_existing:
        LOGGER.info("Dropped columns: %s", drop_existing)

    exprs, added, _ = _build_feature_exprs(df_out)
    if exprs:
        df_out = df_out.with_columns(exprs)
    LOGGER.info("Added %s ENTSO-E derived error/update features.", len(added))
    if added:
        LOGGER.info("Added columns: %s", added)

    preserved = [c for c in KEEP_FOR_STRUCTURAL_ANALYSIS if c in df_out.columns]
    LOGGER.info("Preserved structural-analysis columns: %s", preserved)

    metadata_cols = [c for c in df_out.columns if c.endswith("source") or c.endswith("is_fallback")]
    LOGGER.info("Preserved metadata columns: %s", metadata_cols)

    return df_out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Drop redundant SMARD renewable columns and add ENTSO-E-based errors."
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        default="data/processed/all_data.parquet",
        help="Input merged parquet path.",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default="data/processed/all_data_refined.parquet",
        help="Output refined parquet path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input parquet: {input_path}")

    df = pl.read_parquet(input_path)
    LOGGER.info("Loaded %s rows, %s columns from %s", df.height, len(df.columns), input_path)

    refined = refine_dataset(df)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    refined.write_parquet(output_path, compression="zstd")
    LOGGER.info("Wrote %s rows, %s columns to %s", refined.height, len(refined.columns), output_path)


if __name__ == "__main__":
    main()
