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
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

try:
    import holidays
except ModuleNotFoundError:  # pragma: no cover - runtime fallback for missing optional dep
    holidays = None

# Columns kept in cleaned datasets for audit/provenance but excluded from ML feature table.
# The list is intentionally conservative and can be tuned per experiment.
DROP_FOR_MODEL = [
    # Alternate timezone representation; keep only timestamp_utc for modeling.
    "timestamp_cet",
    # Netztransparenz provenance streams kept for QA, not modeling.
    "NRV_balance_qs",
    "NRV_balance_op",
    "rz_saldo_mw_qs",
    "rz_saldo_mw_op",
    "rz_saldo_mw",
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

DATA_IS_LAGGED = True


def _lag_hours_for_column(col: str) -> int:
    """Return lag (hours) for PiT alignment based on column semantics."""
    if (
        "_actual_entsoe" in col
        or "generation_fossil_" in col
        or col == "generation_nuclear_mw"
        or "generation_hydro_" in col
        or col == "residual_load_actual"
        or col == "NRV_balance"
        or col == "residual_load_calc"
        or col == "wind_total_error_da"
    ):
        return 2
    if (
        "afrr_activated_mw" in col
        or "afrr_activated_mwh" in col
        or "mfrr_activated_mw" in col
        or "mfrr_activated_mwh" in col
        or "afrr_marginal_activation_price" in col
        or "afrr_activation_marginal_price" in col
        or "afrr_vwap" in col
        or "reBAP_shortage_surplus" in col
        or "afrr_picasso_mw" in col
        or "price_intraday_eur" in col
        or "afrr_picasso_net_mw" in col
        or "mfrr_mari_net_mw" in col
    ):
        return 1
    return 0


def apply_point_in_time_lag_layer(df: pl.DataFrame) -> pl.DataFrame:
    """Apply strict PiT publication lags to observation features.

    Ground-truth target columns are protected and never lagged.
    """
    protected = {"timestamp_utc", "y_true_pos", "y_true_neg", "y_train_pos", "y_train_neg"}
    exprs: list[pl.Expr] = []
    lagged_cols = 0
    for col in df.columns:
        if col in protected:
            continue
        lag_h = _lag_hours_for_column(col)
        if lag_h > 0:
            exprs.append(pl.col(col).shift(lag_h).alias(col))
            lagged_cols += 1
    if exprs:
        df = df.with_columns(exprs)
    return df.with_columns([
        pl.lit(DATA_IS_LAGGED).cast(pl.Boolean).alias("data_is_lagged"),
        pl.lit(lagged_cols).cast(pl.Int32).alias("pit_lagged_column_count"),
    ])


def apply_day_ahead_forecast_availability(df: pl.DataFrame) -> pl.DataFrame:
    """Apply conservative DA publication availability handling (13:00 Europe/Berlin).

    For day-ahead forecast columns used in 72h sequence settings, enforce a
    conservative availability rule:
    - If the full horizon cannot be covered by currently available DA publications,
      replace with persistence fallback (same hour previous day, `shift(24)`).

    Note:
    - Day-ahead market prices (e.g., `da_price_eur`, `da_price_*`) are intentionally
      excluded and treated as strict T-0 features for the delivery day.
    """
    if "timestamp_utc" not in df.columns:
        return df
    forecast_cols = [c for c in df.columns if "forecast_da" in c]
    if not forecast_cols:
        return df

    # Compute horizon visibility wrt next DA publication in local time (13:00 CET/CEST).
    out = df.with_columns(
        pl.col("timestamp_utc").dt.convert_time_zone("Europe/Berlin").dt.hour().alias("__local_hour")
    ).with_columns(
        pl.when(pl.col("__local_hour") < 13)
        .then(13 - pl.col("__local_hour"))
        .otherwise(37 - pl.col("__local_hour"))
        .cast(pl.Int32)
        .alias("__hours_to_next_da_pub")
    ).with_columns(
        (pl.col("__hours_to_next_da_pub") + 24).alias("__known_horizon_hours")
    )
    exprs: list[pl.Expr] = []
    for c in forecast_cols:
        # Persistence fallback for same hour previous day (hourly data).
        exprs.append(
            pl.when(pl.col("__known_horizon_hours") < 72)
            .then(pl.col(c).shift(24))
            .otherwise(pl.col(c))
            .forward_fill()
            .alias(c)
        )
    return out.with_columns(exprs).drop("__local_hour", "__hours_to_next_da_pub", "__known_horizon_hours")


def add_multi_output_targets(df: pl.DataFrame, horizon_hours: int = 72) -> pl.DataFrame:
    """Create direct multi-output target columns for +1h..+72h."""
    if "y_true_pos" not in df.columns or "y_true_neg" not in df.columns:
        return df
    exprs: list[pl.Expr] = []
    for h in range(1, horizon_hours + 1):
        exprs.append(pl.col("y_true_pos").shift(-h).alias(f"target_pos_h{h}"))
        exprs.append(pl.col("y_true_neg").shift(-h).alias(f"target_neg_h{h}"))
    return df.with_columns(exprs)


def engineer_targets(df: pl.DataFrame) -> pl.DataFrame:
    """Create pay-as-cleared targets for both economics and ML.

    Inputs required:
    - `afrr_activation_marginal_price_pos`, `afrr_activation_marginal_price_neg`
      (fallback to avg-activation columns if marginal columns are not present)
    - activation magnitude columns:
      - preferred: `afrr_activated_mwh_pos`, `afrr_activated_mwh_neg`
      - fallback: `afrr_activated_mw_pos`, `afrr_activated_mw_neg`

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

    pos_act_col = (
        "afrr_activated_mwh_pos"
        if "afrr_activated_mwh_pos" in df.columns
        else ("afrr_activated_mw_pos" if "afrr_activated_mw_pos" in df.columns else None)
    )
    neg_act_col = (
        "afrr_activated_mwh_neg"
        if "afrr_activated_mwh_neg" in df.columns
        else ("afrr_activated_mw_neg" if "afrr_activated_mw_neg" in df.columns else None)
    )
    missing = []
    if pos_price_col is None:
        missing.append("afrr activation price (pos)")
    if neg_price_col is None:
        missing.append("afrr activation price (neg)")
    if pos_act_col is None:
        missing.append("afrr activation volume/power (pos)")
    if neg_act_col is None:
        missing.append("afrr activation volume/power (neg)")
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
            & (pl.col(pos_act_col).cast(pl.Float64).fill_null(0.0).abs() == 0.0)
        )
        .then(0.0)
        .otherwise(pl.col(pos_price_col).cast(pl.Float64))
        .alias("y_true_pos"),
        pl.when(
            (pl.col(neg_price_col).cast(pl.Float64).abs() > sentinel_abs)
            & (pl.col(neg_act_col).cast(pl.Float64).fill_null(0.0).abs() == 0.0)
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
    # Additional rolling windows requested for key series.
    extra_rolling_specs = [
        ("load_actual_entsoe", "load_actual_entsoe"),
        ("da_price_eur", "da_price_eur"),
        ("wind_onshore_actual_entsoe", "wind_onshore_actual_entsoe"),
    ]
    for col_name, prefix in extra_rolling_specs:
        if col_name in df.columns:
            df = df.with_columns([
                pl.col(col_name).cast(pl.Float64).rolling_mean(window_size=24, min_samples=1).alias(f"{prefix}_mean_24h"),
                pl.col(col_name).cast(pl.Float64).rolling_std(window_size=24, min_samples=2).alias(f"{prefix}_std_24h"),
                pl.col(col_name).cast(pl.Float64).rolling_mean(window_size=168, min_samples=1).alias(f"{prefix}_mean_168h"),
                pl.col(col_name).cast(pl.Float64).rolling_std(window_size=168, min_samples=2).alias(f"{prefix}_std_168h"),
            ])

    fill_cols = [c for c in fill_cols if c in df.columns]
    if fill_cols:
        # Never backfill from the future in a PiT feature set.
        df = df.with_columns([pl.col(c).fill_null(strategy="forward").alias(c) for c in fill_cols])

    return df


def add_price_offering_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add price/opportunity features for activation-price and capacity-price modeling.

    Features:
    - afrr_id_price_spread = afrr_activation_price - intraday_price_da
    - relative_price_competitiveness = afrr_activation_price / rolling_mean_24h(shifted)
    - price_volatility_short_term = rolling_std_4h(shifted afrr_activation_price)
    - scarcity_price_premium = afrr_activation_price * grid_stress_index

    Notes:
    - Uses time-based rolling windows (`24h`, `4h`) in pandas.
    - Leakage-safe rolling stats via `shift(1)` before rolling.
    - NaNs are forward-filled then zero-filled for robustness.
    """
    if "timestamp_utc" not in df.columns:
        return df

    pdf = df.to_pandas()
    pdf["timestamp_utc"] = pd.to_datetime(pdf["timestamp_utc"], utc=True, errors="coerce")
    pdf = pdf.sort_values("timestamp_utc").reset_index(drop=True)

    def _first_existing(candidates: tuple[str, ...]) -> str | None:
        for c in candidates:
            if c in pdf.columns:
                return c
        return None

    afrr_price_col = _first_existing(
        (
            "afrr_activation_price",
            "afrr_vwap_pos",
            "afrr_avg_activation_price_pos",
            "afrr_bid_vwap_activation_price_pos",
        )
    )
    intraday_col = _first_existing(
        (
            "intraday_price_da",
            "price_intraday_eur",
            "da_price_eur",
        )
    )

    if afrr_price_col is None:
        return df

    p = pd.to_numeric(pdf[afrr_price_col], errors="coerce")
    # Use DatetimeIndex for offset-based rolling windows.
    p_indexed = p.copy()
    p_indexed.index = pdf["timestamp_utc"]

    # 1) Cross-market spread.
    if intraday_col is not None:
        p_id = pd.to_numeric(pdf[intraday_col], errors="coerce")
        pdf["afrr_id_price_spread"] = p - p_id
    else:
        pdf["afrr_id_price_spread"] = np.nan

    # 2) Relative price competitiveness with leakage-safe rolling mean.
    roll_mean_24h = p_indexed.shift(1).rolling("24h", min_periods=1).mean()
    denom = roll_mean_24h.reindex(pdf["timestamp_utc"]).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = p.to_numpy() / denom
    pdf["relative_price_competitiveness"] = rel

    # 3) Short-term price volatility (4h), leakage-safe.
    roll_std_4h = p_indexed.shift(1).rolling("4h", min_periods=2).std(ddof=0)
    pdf["price_volatility_short_term"] = roll_std_4h.reindex(pdf["timestamp_utc"]).to_numpy()

    # 4) Scarcity interaction premium.
    if "grid_stress_index" in pdf.columns:
        gsi = pd.to_numeric(pdf["grid_stress_index"], errors="coerce")
        pdf["scarcity_price_premium"] = p * gsi
    else:
        pdf["scarcity_price_premium"] = np.nan

    # Robust missing handling.
    for c in (
        "afrr_id_price_spread",
        "relative_price_competitiveness",
        "price_volatility_short_term",
        "scarcity_price_premium",
    ):
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce").ffill().fillna(0.0)

    return pl.from_pandas(pdf)


def add_aggregated_and_cluster_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add aggregated target-encoding and market-state clustering features.

    Created features:
    - TE_hour_regime_activation
    - market_state_cluster
    - nrv_quantile_5

    Leakage control:
    - Target encoding uses `shift(1)` + `expanding().mean()` inside each
      (`market_regime_picasso`, `hour`) group.
    - Clustering input uses rolling-normalized signals built from past values
      only (`shift(1)` prior to rolling stats).
    """
    if "timestamp_utc" not in df.columns:
        return df

    pdf = df.to_pandas()
    pdf["timestamp_utc"] = pd.to_datetime(pdf["timestamp_utc"], utc=True, errors="coerce")
    pdf = pdf.sort_values("timestamp_utc").reset_index(drop=True)

    # Ensure required grouping columns exist.
    if "hour" not in pdf.columns:
        pdf["hour"] = pdf["timestamp_utc"].dt.hour.astype(np.int8)
    if "market_regime_picasso" not in pdf.columns:
        picasso_start = pd.Timestamp("2022-06-22 00:00:00+00:00")
        pdf["market_regime_picasso"] = (pdf["timestamp_utc"] >= picasso_start).astype(np.int8)

    # Ensure binary target is available.
    if "is_activated" not in pdf.columns:
        if {"afrr_activated_mw_pos", "afrr_activated_mw_neg"}.issubset(pdf.columns):
            ap = pd.to_numeric(pdf["afrr_activated_mw_pos"], errors="coerce").fillna(0.0)
            an = pd.to_numeric(pdf["afrr_activated_mw_neg"], errors="coerce").fillna(0.0)
            pdf["is_activated"] = ((ap.abs() > 0.0) | (an.abs() > 0.0)).astype(np.int8)
        else:
            # Conservative fallback if no activation columns are available.
            pdf["is_activated"] = 0

    # 1) Leakage-safe target encoding by regime/hour.
    global_mean = float(pd.to_numeric(pdf["is_activated"], errors="coerce").fillna(0.0).mean())
    te = (
        pdf.groupby(["market_regime_picasso", "hour"])["is_activated"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    pdf["TE_hour_regime_activation"] = pd.to_numeric(te, errors="coerce").fillna(global_mean).astype(np.float64)

    # 2) Market-state clustering on rolling-normalized inputs.
    need_cols = ["nrv_zscore_24h", "grid_stress_index", "picasso_flow_rate"]
    for c in need_cols:
        if c not in pdf.columns:
            pdf[c] = 0.0
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce")

    ts_index = pdf["timestamp_utc"]
    norm_feats: list[np.ndarray] = []
    for c in need_cols:
        s = pdf[c].copy()
        s.index = ts_index
        mu = s.shift(1).rolling("24h", min_periods=2).mean()
        sd = s.shift(1).rolling("24h", min_periods=2).std(ddof=0).replace(0.0, np.nan)
        z = ((s - mu) / sd).reindex(ts_index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        norm_feats.append(z.to_numpy(dtype=float))

    X = np.column_stack(norm_feats)
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "scikit-learn is required for market_state_cluster. Install `scikit-learn`."
        ) from exc

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    pdf["market_state_cluster"] = km.fit_predict(X_scaled).astype(np.int8)

    # 3) NRV quantile buckets.
    q = pd.qcut(pd.to_numeric(pdf["nrv_zscore_24h"], errors="coerce"), 5, labels=False, duplicates="drop")
    pdf["nrv_quantile_5"] = pd.to_numeric(q, errors="coerce").fillna(0).astype(np.int8)

    return pl.from_pandas(pdf)


def add_market_regime_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add market-regime and grid-stress features without target leakage.

    Features created:
    - `mfrr_active_lag`: lagged mFRR activity flag (1 if activation > 0 else 0)
    - `nrv_zscore_24h`: 24h z-score of NRV balance
    - `picasso_flow_rate`: cross-border balancing flow proxy
    - `grid_stress_index`: composite stress score in [0, 1]
    - `market_regime_picasso`: 0 before 2022-06-22 UTC, 1 from 2022-06-22 UTC onward

    Notes:
    - Inputs may already be globally lagged by the PiT layer.
    - Rolling windows use a 24h horizon in row-count space based on inferred frequency.
    - Early-window NaNs are handled by safe defaults (mostly 0.0) for model robustness.
    """
    if "timestamp_utc" not in df.columns:
        return df

    pdf = df.to_pandas()
    pdf["timestamp_utc"] = pd.to_datetime(pdf["timestamp_utc"], utc=True, errors="coerce")
    pdf = pdf.sort_values("timestamp_utc").reset_index(drop=True)

    # Infer sampling frequency for rolling-window sizing.
    diffs = pdf["timestamp_utc"].dropna().diff().dropna()
    if len(diffs) == 0:
        step_seconds = 3600.0
    else:
        step_seconds = float(diffs.median().total_seconds())
        if not np.isfinite(step_seconds) or step_seconds <= 0:
            step_seconds = 3600.0
    steps_per_hour = max(1, int(round(3600.0 / step_seconds)))
    window_24h = max(1, 24 * steps_per_hour)

    # 1) mFRR activity feature. Do not apply an additional lag here:
    # base activation columns are already shifted by PiT layer where configured.
    mfrr_pos = "mfrr_activated_mwh_pos" if "mfrr_activated_mwh_pos" in pdf.columns else "mfrr_activated_mw_pos"
    mfrr_neg = "mfrr_activated_mwh_neg" if "mfrr_activated_mwh_neg" in pdf.columns else "mfrr_activated_mw_neg"
    if mfrr_pos in pdf.columns and mfrr_neg in pdf.columns:
        active = ((pdf[mfrr_pos].fillna(0.0) > 0.0) | (pdf[mfrr_neg].fillna(0.0) > 0.0)).astype(int)
        pdf["mfrr_active_lag"] = active.fillna(0).astype(int)
    else:
        pdf["mfrr_active_lag"] = 0

    # 2) NRV z-score over last 24h.
    nrv_col = "NRV_balance" if "NRV_balance" in pdf.columns else None
    if nrv_col is not None:
        nrv = pd.to_numeric(pdf[nrv_col], errors="coerce")
        nrv_mean = nrv.rolling(window_24h, min_periods=2).mean()
        nrv_std = nrv.rolling(window_24h, min_periods=2).std(ddof=0)
        z = (nrv - nrv_mean) / nrv_std.replace(0.0, np.nan)
        pdf["nrv_zscore_24h"] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        nrv_abs = nrv.abs().fillna(0.0)
    else:
        pdf["nrv_zscore_24h"] = 0.0
        nrv_abs = pd.Series(0.0, index=pdf.index)

    # 3) PICASSO flow proxy.
    if "afrr_optimization_mwh" in pdf.columns:
        picasso = pd.to_numeric(pdf["afrr_optimization_mwh"], errors="coerce").fillna(0.0)
    elif "net_import_export_mw" in pdf.columns:
        picasso = pd.to_numeric(pdf["net_import_export_mw"], errors="coerce").fillna(0.0)
    else:
        picasso = pd.Series(0.0, index=pdf.index)
    pdf["picasso_flow_rate"] = picasso

    # 4) Composite stress index in [0, 1].
    nrv_norm_den = nrv_abs.rolling(window_24h, min_periods=2).max().replace(0.0, np.nan)
    nrv_norm = (nrv_abs / nrv_norm_den).fillna(0.0).clip(0.0, 1.0)
    pic_abs = picasso.abs()
    pic_norm_den = pic_abs.rolling(window_24h, min_periods=2).max().replace(0.0, np.nan)
    pic_norm = (pic_abs / pic_norm_den).fillna(0.0).clip(0.0, 1.0)
    gsi = 0.4 * nrv_norm + 0.4 * pdf["mfrr_active_lag"].astype(float) + 0.2 * pic_norm
    pdf["grid_stress_index"] = gsi.clip(0.0, 1.0)

    # 5) Market regime flag (PICASSO structural break).
    picasso_start = pd.Timestamp("2022-06-22 00:00:00+00:00")
    pdf["market_regime_picasso"] = (pdf["timestamp_utc"] >= picasso_start).astype(int)

    return pl.from_pandas(pdf)


def add_time_features(
    df: pd.DataFrame,
    datetime_col: str = "timestamp_utc",
    user_col: str = "user_id",
) -> pd.DataFrame:
    """Add vectorized calendar/cyclical/user-recency time features.

    Notes:
    - Uses only pandas/numpy vectorized operations (no row-wise apply).
    - `days_since_last_activation` is grouped by `user_col` and filled with -1
      for each user's first event.
    """
    if datetime_col not in df.columns:
        raise KeyError(f"Missing datetime column: {datetime_col}")
    if user_col not in df.columns:
        raise KeyError(f"Missing user column: {user_col}")

    out = df.copy()
    out[datetime_col] = pd.to_datetime(out[datetime_col], utc=True, errors="coerce")
    if out[datetime_col].isna().any():
        raise ValueError(f"Column '{datetime_col}' contains invalid datetimes.")

    out = out.sort_values([user_col, datetime_col], kind="mergesort")

    dt = out[datetime_col].dt
    hour = dt.hour.to_numpy()
    weekday = dt.weekday.to_numpy()  # 0..6
    month = dt.month.to_numpy()  # 1..12
    day = dt.day.to_numpy()

    # 1) Cyclical features.
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["weekday_sin"] = np.sin(2.0 * np.pi * weekday / 7.0)
    out["weekday_cos"] = np.cos(2.0 * np.pi * weekday / 7.0)
    out["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    # 2) Calendar/economic indicators.
    out["is_weekend"] = (weekday >= 5).astype(np.int8)
    out["is_payday_period"] = ((day >= 27) | (day <= 3)).astype(np.int8)

    # 3) Daypart one-hot features.
    out["is_morning"] = ((hour >= 6) & (hour <= 11)).astype(np.int8)
    out["is_afternoon"] = ((hour >= 12) & (hour <= 16)).astype(np.int8)
    out["is_evening"] = ((hour >= 17) & (hour <= 22)).astype(np.int8)
    out["is_night"] = ((hour >= 23) | (hour <= 5)).astype(np.int8)

    # 4) User-level recency delta (days).
    delta_days = (
        out.groupby(user_col, sort=False)[datetime_col]
        .diff()
        .dt.total_seconds()
        .div(86400.0)
    )
    out["days_since_last_activation"] = delta_days.fillna(-1.0).astype(np.float64)
    return out


def add_german_holiday_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add German holiday/regime features from local (Europe/Berlin) dates.

    Features:
    - holiday_severity: 1.0 national, 0.5 regional (BW/BY/NW), 0.0 otherwise
    - is_bridge_day: 1 for Mon-before-national-Tue-holiday or Fri-after-national-Thu-holiday
    - is_christmas_break: 1 for Dec 24-31 inclusive
    """
    if "timestamp_utc" not in df.columns:
        return df

    # Convert UTC timestamp to local German date for calendar matching.
    df = df.with_columns(
        pl.col("timestamp_utc")
        .dt.convert_time_zone("Europe/Berlin")
        .dt.date()
        .alias("local_date")
    )

    if holidays is None:
        # Degrade gracefully when optional dependency is unavailable.
        return (
            df.with_columns(
                pl.lit(0.0).cast(pl.Float64).alias("holiday_severity"),
                pl.lit(0).cast(pl.Int8).alias("is_bridge_day"),
                pl.when((pl.col("local_date").dt.month() == 12) & (pl.col("local_date").dt.day().is_between(24, 31)))
                .then(1)
                .otherwise(0)
                .cast(pl.Int8)
                .alias("is_christmas_break"),
            )
            .drop("local_date")
        )

    date_bounds = df.select(
        pl.col("local_date").min().alias("min_date"),
        pl.col("local_date").max().alias("max_date"),
    ).row(0)
    min_date, max_date = date_bounds
    if min_date is None or max_date is None:
        return df.with_columns(
            pl.lit(0.0).cast(pl.Float64).alias("holiday_severity"),
            pl.lit(0).cast(pl.Int8).alias("is_bridge_day"),
            pl.lit(0).cast(pl.Int8).alias("is_christmas_break"),
        )

    years = list(range(min_date.year, max_date.year + 1))
    de_national = holidays.country_holidays("DE", years=years)
    de_bw = holidays.country_holidays("DE", subdiv="BW", years=years)
    de_by = holidays.country_holidays("DE", subdiv="BY", years=years)
    de_nw = holidays.country_holidays("DE", subdiv="NW", years=years)

    calendar_rows: list[dict[str, object]] = []
    d = min_date
    while d <= max_date:
        is_national = d in de_national
        is_regional = (d in de_bw) or (d in de_by) or (d in de_nw)
        if is_national:
            severity = 1.0
        elif is_regional:
            severity = 0.5
        else:
            severity = 0.0

        is_bridge = int(
            (d.weekday() == 0 and ((d + timedelta(days=1)) in de_national))
            or (d.weekday() == 4 and ((d - timedelta(days=1)) in de_national))
        )
        is_xmas_break = int(d.month == 12 and 24 <= d.day <= 31)

        calendar_rows.append(
            {
                "local_date": d,
                "holiday_severity": severity,
                "is_bridge_day": is_bridge,
                "is_christmas_break": is_xmas_break,
            }
        )
        d += timedelta(days=1)

    df_calendar = pl.DataFrame(calendar_rows).with_columns(
        pl.col("local_date").cast(pl.Date),
        pl.col("holiday_severity").cast(pl.Float64),
        pl.col("is_bridge_day").cast(pl.Int8),
        pl.col("is_christmas_break").cast(pl.Int8),
    )

    df = df.join(df_calendar, on="local_date", how="left")
    return df.drop("local_date")


def build_features(input_path: Path, output_path: Path) -> None:
    """Build ML features from transformed parquet."""
    df = pl.read_parquet(input_path)
    if "timestamp_utc" in df.columns:
        df = df.sort("timestamp_utc")

    df = engineer_targets(df)
    # Lag-first architecture for all observation-side features.
    df = apply_point_in_time_lag_layer(df)
    df = apply_day_ahead_forecast_availability(df)
    df = add_confidence_features(df)
    df = add_german_holiday_features(df)
    df = add_market_regime_features(df)
    df = add_price_offering_features(df)
    df = add_aggregated_and_cluster_features(df)
    df = add_multi_output_targets(df, horizon_hours=72)

    # Optional: generic time features for user-centric datasets.
    # For this thesis pipeline, many datasets are system-level and may not include user_id.
    if "timestamp_utc" in df.columns and "user_id" in df.columns:
        df = pl.from_pandas(add_time_features(df.to_pandas(), datetime_col="timestamp_utc", user_col="user_id"))

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
        default="data/processed/all_data_refined.parquet",
        help="Input parquet (default: data/processed/all_data_refined.parquet).",
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
