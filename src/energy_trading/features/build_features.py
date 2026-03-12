"""Feature engineering for model-ready training/backtest data.

Target design (aFRR PICASSO):
- `y_true_*`: economically correct pay-as-cleared marginal price (for backtests/PnL).
- `y_train_*`: clipped version of `y_true_*` (for stable ML regression training).

Rationale:
- Average activation prices are not valid settlement targets in pay-as-cleared setup.
- Extreme sentinel values (about +/-99,999) are technical artifacts and are neutralized
  only when activation is effectively zero.
- Real scarcity spikes are preserved in `y_true_*` and clipped only in `y_train_*`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

# Columns kept in cleaned datasets for audit/provenance but excluded from ML feature table.
# The list is intentionally conservative and can be tuned per experiment.
DROP_FOR_MODEL = [
    # Alternate timezone representation; keep only timestamp_utc for modeling.
    "timestamp_cet",
    # Netztransparenz provenance streams (canonical columns are NRV_balance / rz_saldo_mw).
    "NRV_balance_qs",
    "NRV_balance_op",
    "rz_saldo_mw_qs",
    "rz_saldo_mw_op",
    # Legacy SMARD aggregate error features (recomputed ENTSO-E based features are used instead).
    "wind_onshore_error",
    "wind_offshore_error",
    "solar_error",
    # Source-tag helper columns used for QA lineage only.
    "solar_forecast_id_entsoe_source",
    "wind_onshore_forecast_id_entsoe_source",
    "wind_offshore_forecast_id_entsoe_source",
    "wind_onshore_capacity_source",
    "wind_offshore_capacity_source",
    "solar_capacity_source",
]


def engineer_targets(df: pl.DataFrame) -> pl.DataFrame:
    """Create pay-as-cleared targets for both economics and ML.

    Inputs required:
    - `afrr_activation_marginal_price_pos`, `afrr_activation_marginal_price_neg`
    - `afrr_activated_mwh_pos`, `afrr_activated_mwh_neg`

    Outputs created:
    - `y_true_pos`, `y_true_neg`: cleaned, *unclipped* marginal price
    - `y_train_pos`, `y_train_neg`: clipped targets in [-500, 500]
    """
    pos_price_col = None
    neg_price_col = None
    for cand in ("afrr_activation_marginal_price_pos", "afrr_avg_activation_price_pos", "afrr_activation_avg_price_pos"):
        if cand in df.columns:
            pos_price_col = cand
            break
    for cand in ("afrr_activation_marginal_price_neg", "afrr_avg_activation_price_neg", "afrr_activation_avg_price_neg"):
        if cand in df.columns:
            neg_price_col = cand
            break

    required = [c for c in (pos_price_col, neg_price_col, "afrr_activated_mwh_pos", "afrr_activated_mwh_neg") if c is not None]
    missing = [c for c in ("afrr_activated_mwh_pos", "afrr_activated_mwh_neg") if c not in df.columns]
    if pos_price_col is None:
        missing.append("afrr activation price (pos)")
    if neg_price_col is None:
        missing.append("afrr activation price (neg)")
    if missing:
        raise KeyError(f"Missing required target-engineering columns: {missing}")

    # 1) Prevent leakage from economically incorrect target candidates.
    #    Keep only marginal-price-based targets for a pay-as-cleared market.
    drop_avg = [c for c in ("afrr_activation_avg_price_pos", "afrr_activation_avg_price_neg") if c in df.columns]
    if drop_avg:
        df = df.drop(drop_avg)

    sentinel_abs = 90_000.0
    clip_low = -500.0
    clip_high = 500.0

    # 2) y_true: economically valid backtest target (pay-as-cleared marginal price),
    # with only technical sentinel values neutralized when activation is effectively zero.
    df = df.with_columns([
        pl.when(
            (pl.col(pos_price_col).cast(pl.Float64).abs() > sentinel_abs)
            & (pl.col("afrr_activated_mwh_pos").cast(pl.Float64).fill_null(0.0).abs() == 0.0)
        )
        .then(0.0)
        .otherwise(pl.col(pos_price_col).cast(pl.Float64))
        .alias("y_true_pos"),
        pl.when(
            (pl.col(neg_price_col).cast(pl.Float64).abs() > sentinel_abs)
            & (pl.col("afrr_activated_mwh_neg").cast(pl.Float64).fill_null(0.0).abs() == 0.0)
        )
        .then(0.0)
        .otherwise(pl.col(neg_price_col).cast(pl.Float64))
        .alias("y_true_neg"),
    ])

    # 3) y_train: ML-stable target. Clip tails to protect MSE from rare scarcity spikes.
    df = df.with_columns([
        pl.col("y_true_pos").clip(clip_low, clip_high).alias("y_train_pos"),
        pl.col("y_true_neg").clip(clip_low, clip_high).alias("y_train_neg"),
    ])

    # 4) Cleanup raw marginal columns to avoid accidental downstream use.
    drop_target_sources = [
        "afrr_activation_marginal_price_pos",
        "afrr_activation_marginal_price_neg",
        "afrr_avg_activation_price_pos",
        "afrr_avg_activation_price_neg",
        "afrr_activation_avg_price_pos",
        "afrr_activation_avg_price_neg",
    ]
    drop_target_sources = [c for c in drop_target_sources if c in df.columns]
    return df.drop(drop_target_sources) if drop_target_sources else df


def add_confidence_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add forecast confidence and market stress indicators (no look-ahead bias)."""
    # Renewable share features for balancing context.
    # Use legacy renewable actual/forecast columns before they are dropped from model inputs.
    actual_share_cols = {
        "wind_onshore_actual",
        "wind_offshore_actual",
        "solar_actual",
        "residual_load_actual",
    }
    if actual_share_cols.issubset(df.columns):
        df = df.with_columns([
            (pl.col("wind_onshore_actual").cast(pl.Float64).fill_null(0.0)).alias("__wind_onshore_actual_f"),
            (pl.col("wind_offshore_actual").cast(pl.Float64).fill_null(0.0)).alias("__wind_offshore_actual_f"),
            (pl.col("solar_actual").cast(pl.Float64).fill_null(0.0)).alias("__solar_actual_f"),
            pl.col("residual_load_actual").cast(pl.Float64).fill_null(0.0).alias("__residual_load_actual_f"),
        ])
        df = df.with_columns([
            (
                pl.col("__wind_onshore_actual_f")
                + pl.col("__wind_offshore_actual_f")
                + pl.col("__solar_actual_f")
            ).alias("total_renewables_actual"),
        ])
        df = df.with_columns([
            (pl.col("__residual_load_actual_f") + pl.col("total_renewables_actual")).alias("total_load_actual"),
        ])
        df = df.with_columns([
            pl.when(pl.col("total_load_actual") > 0)
            .then(pl.col("total_renewables_actual") / pl.col("total_load_actual"))
            .otherwise(None)
            .alias("renewable_share_actual"),
        ])

    forecast_share_cols = {"wind_onshore_forecast", "wind_offshore_forecast", "solar_forecast"}
    if forecast_share_cols.issubset(df.columns):
        df = df.with_columns([
            (pl.col("wind_onshore_forecast").cast(pl.Float64).fill_null(0.0)).alias("__wind_onshore_forecast_f"),
            (pl.col("wind_offshore_forecast").cast(pl.Float64).fill_null(0.0)).alias("__wind_offshore_forecast_f"),
            (pl.col("solar_forecast").cast(pl.Float64).fill_null(0.0)).alias("__solar_forecast_f"),
        ])
        df = df.with_columns([
            (
                pl.col("__wind_onshore_forecast_f")
                + pl.col("__wind_offshore_forecast_f")
                + pl.col("__solar_forecast_f")
            ).alias("total_renewables_forecast"),
        ])
        # Use total_load_forecast when present, else proxy with total_load_actual.
        denom = (
            pl.coalesce([pl.col("total_load_forecast").cast(pl.Float64), pl.col("total_load_actual")])
            if "total_load_forecast" in df.columns
            else pl.col("total_load_actual")
        )
        df = df.with_columns([
            pl.when(denom > 0)
            .then(pl.col("total_renewables_forecast") / denom)
            .otherwise(None)
            .alias("renewable_share_forecast"),
        ])
        drop_tmp = [c for c in ("__wind_onshore_forecast_f", "__wind_offshore_forecast_f", "__solar_forecast_f") if c in df.columns]
        if drop_tmp:
            df = df.drop(drop_tmp)

    drop_tmp = [c for c in ("__wind_onshore_actual_f", "__wind_offshore_actual_f", "__solar_actual_f", "__residual_load_actual_f") if c in df.columns]
    if drop_tmp:
        df = df.drop(drop_tmp)

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

    # Keep pay-as-cleared marginal prices as market target inputs; drop average activation prices.
    drop_avg_activation_prices = [
        "afrr_activation_avg_price_pos",
        "afrr_activation_avg_price_neg",
        "GERMANY_AVERAGE_ENERGY_PRICE_[EUR/MWh]",
    ]
    drop_avg_activation_prices = [c for c in drop_avg_activation_prices if c in df.columns]
    if drop_avg_activation_prices:
        df = df.drop(drop_avg_activation_prices)

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

    df = engineer_targets(df)
    df = add_confidence_features(df)
    drop_cols = [c for c in DROP_FOR_MODEL if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)

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
