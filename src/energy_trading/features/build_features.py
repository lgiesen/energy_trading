"""Feature engineering module for building features from transformed data."""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def add_confidence_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add forecast confidence and market stress indicators (no look-ahead bias)."""
    # Enforce ENTSO-E as the exclusive source for wind/solar.
    drop_smard = [
        "wind_onshore_actual",
        "wind_offshore_actual",
        "solar_actual",
        "wind_onshore_forecast",
        "wind_offshore_forecast",
        "solar_forecast",
        "wind_onshore_forecast_intraday",
        "wind_offshore_forecast_intraday",
        "solar_forecast_intraday",
    ]
    drop_smard = [c for c in drop_smard if c in df.columns]
    if drop_smard:
        df = df.drop(drop_smard)

    price_col = None
    for candidate in ("da_price", "da_price_d_eur_mwh", "da_price_eur"):
        if candidate in df.columns:
            price_col = candidate
            break
    if price_col is None:
        raise KeyError("Missing required column: da_price (or da_price_d_eur_mwh/da_price_eur)")

    df = df.with_columns(
        pl.col(price_col)
        .shift(24)
        .rolling_std(window_size=720, min_samples=1)
        .alias("da_price_volatility_30d")
    )

    pairs = {
        "solar": ("solar_forecast_da_entsoe", "solar_actual_entsoe"),
        "wind_onshore": ("wind_onshore_forecast_da_entsoe", "wind_onshore_actual_entsoe"),
        "wind_offshore": ("wind_offshore_forecast_da_entsoe", "wind_offshore_actual_entsoe"),
        "load": ("load_forecast_da", "load_actual"),
    }
    for name, (fc, act) in pairs.items():
        if fc in df.columns and act in df.columns:
            df = df.with_columns(
                (pl.col(fc) - pl.col(act)).abs().alias(f"{name}_abs_error")
            )
            df = df.with_columns(
                pl.col(f"{name}_abs_error")
                .shift(24)
                .rolling_mean(window_size=72, min_samples=1)
                .alias(f"{name}_mae_rolling_72h")
            )

    if "solar_forecast_da_entsoe" in df.columns:
        df = df.with_columns(
            (pl.col("solar_forecast_da_entsoe") - pl.col("solar_forecast_da_entsoe").shift(1))
            .abs()
            .alias("solar_ramp")
        )
    if "wind_onshore_forecast_da_entsoe" in df.columns:
        df = df.with_columns(
            (pl.col("wind_onshore_forecast_da_entsoe") - pl.col("wind_onshore_forecast_da_entsoe").shift(1))
            .abs()
            .alias("wind_ramp")
        )
    if {"load_forecast_da", "wind_onshore_forecast_da_entsoe", "solar_forecast_da_entsoe"}.issubset(df.columns):
        df = df.with_columns(
            (
                pl.col("load_forecast_da")
                - pl.col("wind_onshore_forecast_da_entsoe")
                - pl.col("solar_forecast_da_entsoe")
            )
            .alias("residual_load_forecast")
        )

    techs = ["wind_onshore", "wind_offshore", "solar"]
    for tech in techs:
        act = f"{tech}_actual_entsoe"
        da = f"{tech}_forecast_da_entsoe"
        intraday = f"{tech}_forecast_id_entsoe"
        cap = f"{tech}_capacity_entsoe"

        if act in df.columns and da in df.columns:
            df = df.with_columns((pl.col(act) - pl.col(da)).alias(f"{tech}_error_da"))
        if act in df.columns and intraday in df.columns:
            df = df.with_columns((pl.col(act) - pl.col(intraday)).alias(f"{tech}_error_id"))
        if da in df.columns and intraday in df.columns:
            df = df.with_columns((pl.col(intraday) - pl.col(da)).alias(f"{tech}_forecast_delta"))
        if act in df.columns and cap in df.columns:
            df = df.with_columns((pl.col(act) / pl.col(cap)).alias(f"{tech}_capacity_factor"))

    # Primary physical imbalance signal: sum of intraday errors
    error_id_cols = [f"{tech}_error_id" for tech in techs if f"{tech}_error_id" in df.columns]
    if len(error_id_cols) == 3:
        df = df.with_columns(
            pl.sum_horizontal([pl.col(c) for c in error_id_cols]).alias("total_wind_solar_id_error")
        )

    error_id_cols = [f"{tech}_error_id" for tech in techs if f"{tech}_error_id" in df.columns]
    error_da_cols = [f"{tech}_error_da" for tech in techs if f"{tech}_error_da" in df.columns]
    if len(error_id_cols) == 3:
        df = df.with_columns(pl.sum_horizontal([pl.col(c) for c in error_id_cols]).alias("system_stress_signal"))
    elif len(error_da_cols) == 3:
        df = df.with_columns(pl.sum_horizontal([pl.col(c) for c in error_da_cols]).alias("system_stress_signal"))

    fill_cols = [
        "da_price_volatility_30d",
        "solar_mae_rolling_72h",
        "wind_onshore_mae_rolling_72h",
        "wind_offshore_mae_rolling_72h",
        "load_mae_rolling_72h",
        "solar_ramp",
        "wind_ramp",
        "residual_load_forecast",
    ]
    fill_cols = [c for c in fill_cols if c in df.columns]
    if fill_cols:
        df = df.with_columns([pl.col(c).fill_null(strategy="backward").alias(c) for c in fill_cols])

    return df


def build_features(input_path: Path, output_path: Path) -> None:
    """Build ML features from transformed parquet."""
    df = pl.read_parquet(input_path)
    if "timestamp_utc" in df.columns:
        df = df.sort("timestamp_utc")

    df = add_confidence_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML features from transformed data.")
    parser.add_argument(
        "--in",
        dest="input_path",
        default="data/processed/all_data_transformed.parquet",
        help="Input parquet (default: data/processed/all_data_transformed.parquet).",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default="data/features/all_data_features.parquet",
        help="Output parquet (default: data/features/all_data_features.parquet).",
    )
    args = parser.parse_args()
    build_features(Path(args.input_path), Path(args.output_path))


if __name__ == "__main__":
    main()
