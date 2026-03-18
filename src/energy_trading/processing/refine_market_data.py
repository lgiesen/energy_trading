"""Refine merged market data with domain-specific consolidation and features.

Usage:
    ./.venv/bin/python -m energy_trading.processing.refine_market_data \
        --in data/processed/all_data.parquet \
        --out data/processed/all_data_refined.parquet
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

LOGGER = logging.getLogger(__name__)
PICASSO_START_UTC = datetime(2022, 6, 22, 22, 0, tzinfo=timezone.utc)

SMARD_REDUNDANT_COLS = [
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


def _add_if_possible(
    exprs: list[pl.Expr],
    created: list[str],
    cols: list[str],
    out_col: str,
    expr: pl.Expr,
    available_cols: list[str],
) -> None:
    if all(c in available_cols for c in cols):
        exprs.append(expr.alias(out_col))
        created.append(out_col)
    else:
        missing = [c for c in cols if c not in available_cols]
        LOGGER.warning("Skip %s, missing inputs: %s", out_col, missing)


def refine(df: pl.DataFrame) -> pl.DataFrame:
    if "timestamp_utc" not in df.columns:
        raise ValueError("Missing required column: timestamp_utc")

    drop_cols = [c for c in SMARD_REDUNDANT_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)
    LOGGER.info("Dropped %s redundant SMARD columns: %s", len(drop_cols), drop_cols)

    exprs: list[pl.Expr] = []
    created: list[str] = []
    cols = df.columns

    _add_if_possible(
        exprs,
        created,
        [
            "wind_onshore_forecast_da_entsoe",
            "wind_offshore_forecast_da_entsoe",
            "wind_onshore_actual_entsoe",
            "wind_offshore_actual_entsoe",
        ],
        "wind_total_error_da",
        (
            pl.col("wind_onshore_forecast_da_entsoe")
            + pl.col("wind_offshore_forecast_da_entsoe")
            - pl.col("wind_onshore_actual_entsoe")
            - pl.col("wind_offshore_actual_entsoe")
        ),
        cols,
    )
    _add_if_possible(
        exprs,
        created,
        ["solar_forecast_da_entsoe", "solar_actual_entsoe"],
        "solar_error_da",
        pl.col("solar_forecast_da_entsoe") - pl.col("solar_actual_entsoe"),
        cols,
    )
    _add_if_possible(
        exprs,
        created,
        ["wind_onshore_forecast_id_entsoe", "wind_onshore_forecast_da_entsoe"],
        "wind_forecast_update",
        pl.col("wind_onshore_forecast_id_entsoe") - pl.col("wind_onshore_forecast_da_entsoe"),
        cols,
    )
    _add_if_possible(
        exprs,
        created,
        [
            "load_actual_entsoe",
            "wind_onshore_actual_entsoe",
            "wind_offshore_actual_entsoe",
            "solar_actual_entsoe",
        ],
        "residual_load_calc",
        (
            pl.col("load_actual_entsoe")
            - pl.col("wind_onshore_actual_entsoe")
            - pl.col("wind_offshore_actual_entsoe")
            - pl.col("solar_actual_entsoe")
        ),
        cols,
    )
    _add_if_possible(
        exprs,
        created,
        ["afrr_picasso_mw_pos", "afrr_picasso_mw_neg"],
        "afrr_picasso_net_mw",
        pl.col("afrr_picasso_mw_pos") - pl.col("afrr_picasso_mw_neg"),
        cols,
    )
    _add_if_possible(
        exprs,
        created,
        ["afrr_picasso_mw_pos", "afrr_picasso_mw_neg"],
        "afrr_picasso_churn_mw",
        pl.col("afrr_picasso_mw_pos") + pl.col("afrr_picasso_mw_neg"),
        cols,
    )
    _add_if_possible(
        exprs,
        created,
        ["mfrr_mari_mw_pos", "mfrr_mari_mw_neg"],
        "mfrr_mari_net_mw",
        pl.col("mfrr_mari_mw_pos") - pl.col("mfrr_mari_mw_neg"),
        cols,
    )

    if exprs:
        df = df.with_columns(exprs)
    LOGGER.info("Created/updated %s columns: %s", len(created), created)

    # Structural break for platform flows in refined layer.
    cutoff = pl.lit(PICASSO_START_UTC).cast(pl.Datetime(time_unit="us", time_zone="UTC"))
    flow_cols = [
        c
        for c in (
            "afrr_picasso_mw_pos",
            "afrr_picasso_mw_neg",
            "afrr_picasso_net_mw",
            "afrr_picasso_churn_mw",
            "mfrr_mari_mw_pos",
            "mfrr_mari_mw_neg",
            "mfrr_mari_net_mw",
        )
        if c in df.columns
    ]
    if flow_cols:
        df = df.with_columns(
            [
                pl.when(pl.col("timestamp_utc") < cutoff)
                .then(pl.lit(0.0))
                .otherwise(pl.col(c))
                .cast(pl.Float64, strict=False)
                .alias(c)
                for c in flow_cols
            ]
        )

    # Keep only net flow for MARI after deriving net.
    mfrr_drop = [c for c in ("mfrr_mari_mw_pos", "mfrr_mari_mw_neg") if c in df.columns]
    if mfrr_drop:
        df = df.drop(mfrr_drop)
    LOGGER.info("Dropped %s redundant MARI columns: %s", len(mfrr_drop), mfrr_drop)

    # Preserve raw provenance columns by design.
    preserved = [c for c in df.columns if c.endswith("_qs") or c.endswith("_op") or c.endswith("source") or c.endswith("is_fallback")]
    LOGGER.info("Preserved provenance columns: %s", preserved)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Refine merged market data for downstream modeling.")
    parser.add_argument("--in", dest="input_path", default="data/processed/all_data.parquet", help="Input parquet path.")
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

    df = pl.read_parquet(input_path).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)
    ).sort("timestamp_utc")
    LOGGER.info("Loaded %s rows and %s columns from %s", df.height, len(df.columns), input_path)

    refined = refine(df)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    refined.write_parquet(output_path, compression="zstd")
    LOGGER.info("Wrote %s rows and %s columns to %s", refined.height, len(refined.columns), output_path)


if __name__ == "__main__":
    main()
