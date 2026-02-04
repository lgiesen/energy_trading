"""Centralized missing value handling for modeling.

Usage:
    ./.venv/bin/python -m energy_trading.processing.handle_missing_values \
        --in data/processed/all_data.parquet \
        --out data/processed/all_data_clean.parquet

Check results with:
    ./.venv/bin/python -m energy_trading.processing.compare_missingness \
        --before data/processed/all_data.parquet \
        --after data/processed/all_data_clean.parquet

"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import polars as pl

LOGGER = logging.getLogger(__name__)


def _interpolate_small_gaps(df: pl.DataFrame, col: str, max_gap_hours: int) -> pl.DataFrame:
    """Interpolate only null runs <= max_gap_hours; keep longer gaps as null."""
    if col not in df.columns:
        return df
    is_null = pl.col(col).is_null()
    run_id = (is_null != is_null.shift(1)).cast(pl.Int64).cum_sum()
    run_len = pl.len().over(run_id)
    keep_interp = is_null & (run_len <= max_gap_hours)
    interpolated = pl.col(col).interpolate()
    return df.with_columns(
        pl.when(keep_interp)
        .then(interpolated)
        .otherwise(pl.col(col))
        .alias(col)
    )


def _ffill_cols(df: pl.DataFrame, cols: Iterable[str]) -> pl.DataFrame:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return df
    return df.with_columns([pl.col(c).fill_null(strategy="forward") for c in existing])


def _recalc_onshore_error(df: pl.DataFrame) -> pl.DataFrame:
    if {"wind_onshore_forecast_intraday", "wind_onshore_actual", "wind_onshore_intraday_error"}.issubset(df.columns):
        df = df.with_columns(
            (pl.col("wind_onshore_forecast_intraday") - pl.col("wind_onshore_actual")).alias(
                "wind_onshore_intraday_error"
            )
        )
    return df


def _recalc_total_wind_error(df: pl.DataFrame) -> pl.DataFrame:
    if {"total_wind_intraday_forecast", "wind_onshore_actual", "wind_offshore_actual"}.issubset(df.columns):
        df = df.with_columns(
            (
                pl.col("total_wind_intraday_forecast")
                - (pl.col("wind_onshore_actual") + pl.col("wind_offshore_actual"))
            ).alias("total_wind_intraday_error")
        )
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Handle missing values for modeling.")
    parser.add_argument(
        "--in",
        dest="input_path",
        default="data/processed/all_data.parquet",
        help="Input parquet (defaults to all_data.parquet).",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default="data/processed/all_data_clean.parquet",
        help="Output parquet path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        fallback = Path("data/processed/all_data.parquet")
        if fallback.exists():
            input_path = fallback
        else:
            raise FileNotFoundError(f"Missing input parquet: {args.input_path}")

    df = pl.read_parquet(input_path).sort("timestamp_utc")
    LOGGER.info("Loaded %s rows from %s", df.height, input_path)

    # 1) Financial data: forward-fill weekends
    df = _ffill_cols(df, ["co2_price_eua", "gas_price_ttf", "coal_price_api"])

    # 2) Physics/grid small gaps (<= 2 hours)
    for col in ["load_forecast_da", "da_price_d_eur_mwh", "ex_rate_eur_usd"]:
        df = _interpolate_small_gaps(df, col, max_gap_hours=2)

    # 3) Wind intraday proxy
    if "wind_onshore_forecast_intraday" in df.columns and "wind_onshore_forecast" in df.columns:
        df = df.with_columns(
            pl.coalesce([pl.col("wind_onshore_forecast_intraday"), pl.col("wind_onshore_forecast")]).alias(
                "wind_onshore_forecast_intraday"
            )
        )
    if "total_wind_intraday_forecast" in df.columns:
        if "wind_forecast_de" in df.columns:
            df = df.with_columns(
                pl.coalesce([pl.col("total_wind_intraday_forecast"), pl.col("wind_forecast_de")]).alias(
                    "total_wind_intraday_forecast"
                )
            )
        elif {"wind_onshore_forecast", "wind_offshore_forecast"}.issubset(df.columns):
            df = df.with_columns(
                pl.coalesce(
                    [
                        pl.col("total_wind_intraday_forecast"),
                        (pl.col("wind_onshore_forecast") + pl.col("wind_offshore_forecast")),
                    ]
                ).alias("total_wind_intraday_forecast")
            )

    df = _recalc_onshore_error(df)
    df = _recalc_total_wind_error(df)

    # 4) Intraday price: fill with day-ahead if missing
    if "price_intraday_eur" in df.columns and "da_price_d_eur_mwh" in df.columns:
        df = df.with_columns(
            pl.coalesce([pl.col("price_intraday_eur"), pl.col("da_price_d_eur_mwh")]).alias("price_intraday_eur")
        )

    # 5) Regelleistung prices: interpolate if volume != 0 and price is null
    price_specs = [
        ("afrr_activation_avg_price_", "afrr_activated_mw_"),
        ("afrr_activation_price_", "afrr_activated_mw_"),
        ("mfrr_activation_price_", "mfrr_activated_mw_"),
        ("afrr_capacity_price_", "afrr_capacity_offered_mw_"),
    ]
    for prefix, vol_prefix in price_specs:
        for direction in ("pos", "neg"):
            price_col = f"{prefix}{direction}"
            vol_col = f"{vol_prefix}{direction}"
            if price_col not in df.columns or vol_col not in df.columns:
                continue

            # If volume is zero and price is null, mark as 0 (irrelevant)
            df = df.with_columns(
                pl.when(pl.col(vol_col) == 0)
                .then(pl.coalesce([pl.col(price_col), pl.lit(0.0)]))
                .otherwise(pl.col(price_col))
                .alias(price_col)
            )

            # Interpolate remaining nulls (only small gaps)
            df = _interpolate_small_gaps(df, price_col, max_gap_hours=2)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")
    LOGGER.info("Wrote %s rows to %s", df.height, output_path)


if __name__ == "__main__":
    main()
