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

# Keep only canonical activation-price signal columns.
# All other historical/alternative activation-price variants are removed.
ACTIVATION_PRICE_DROP_COLS = [
    "afrr_marginal_activation_price_pos",
    "afrr_marginal_activation_price_neg",
    "afrr_activation_marginal_price_pos",
    "afrr_activation_marginal_price_neg",
    "afrr_avg_activation_price_pos",
    "afrr_avg_activation_price_neg",
    "afrr_activation_avg_price_pos",
    "afrr_activation_avg_price_neg",
    "afrr_bid_avg_activation_price_pos",
    "afrr_bid_avg_activation_price_neg",
    "afrr_bid_vwap_activation_price_pos",
    "afrr_bid_vwap_activation_price_neg",
    "afrr_reconstructed_marginal_price_pos",
    "afrr_reconstructed_marginal_price_neg",
    "afrr_vwap_pos_eur_mwh",
    "afrr_vwap_neg_eur_mwh",
]

# For ML training datasets: drop physical activation-rate columns.
# They can be re-joined later for optimizer-specific runs.
ACTIVATION_RATE_DROP_COLS = [
    "activation_rate_phys_pos",
    "activation_rate_phys_neg",
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

    df_out = df

    # Canonicalize VWAP naming if old _eur_mwh columns still exist.
    if "afrr_vwap_pos" not in df_out.columns and "afrr_vwap_pos_eur_mwh" in df_out.columns:
        df_out = df_out.with_columns(pl.col("afrr_vwap_pos_eur_mwh").alias("afrr_vwap_pos"))
        LOGGER.info("Canonicalized afrr_vwap_pos from afrr_vwap_pos_eur_mwh")
    if "afrr_vwap_neg" not in df_out.columns and "afrr_vwap_neg_eur_mwh" in df_out.columns:
        df_out = df_out.with_columns(pl.col("afrr_vwap_neg_eur_mwh").alias("afrr_vwap_neg"))
        LOGGER.info("Canonicalized afrr_vwap_neg from afrr_vwap_neg_eur_mwh")

    drop_existing = [c for c in DROP_COLS if c in df_out.columns]
    df_out = df_out.drop(drop_existing) if drop_existing else df_out
    LOGGER.info("Dropped %s redundant SMARD columns.", len(drop_existing))
    if drop_existing:
        LOGGER.info("Dropped columns: %s", drop_existing)

    activation_drop_existing = [c for c in ACTIVATION_PRICE_DROP_COLS if c in df_out.columns]
    if activation_drop_existing:
        df_out = df_out.drop(activation_drop_existing)
    LOGGER.info(
        "Dropped %s non-canonical activation-price columns (kept afrr_vwap_pos/neg).",
        len(activation_drop_existing),
    )
    if activation_drop_existing:
        LOGGER.info("Dropped activation-price columns: %s", activation_drop_existing)

    rate_drop_existing = [c for c in ACTIVATION_RATE_DROP_COLS if c in df_out.columns]
    if rate_drop_existing:
        df_out = df_out.drop(rate_drop_existing)
    LOGGER.info(
        "Dropped %s activation-rate columns for ML dataset.",
        len(rate_drop_existing),
    )
    if rate_drop_existing:
        LOGGER.info("Dropped activation-rate columns: %s", rate_drop_existing)

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
