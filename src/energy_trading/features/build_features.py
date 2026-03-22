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
import logging
import re
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
    # Streamline residual load representation: keep calculated variant only.
    "residual_load_actual",
    # Removed due to >98% missing values identified in the feature audit.
    "bid_provider_to_grid_share_neg",
    "bid_provider_to_grid_share_pos",
    "afrr_reconstructed_marginal_price_pos",
    "afrr_reconstructed_marginal_price_neg",
]

# Removed due to >98% missing values identified in the feature audit.
DROP_HIGH_MISSING_BEFORE_FEATURES = [
    "bid_provider_to_grid_share_neg",
    "bid_provider_to_grid_share_pos",
    "afrr_reconstructed_marginal_price_pos",
    "afrr_reconstructed_marginal_price_neg",
]

DATA_IS_LAGGED = True
LOGGER = logging.getLogger(__name__)


def apply_final_feature_aggregations(df: pl.DataFrame) -> pl.DataFrame:
    """Apply final feature aggregation/reduction blocks with economic semantics.

    Blocks:
    1) Foreign DA prices -> `neighbor_price_avg`, `neighbor_spread_avg`
    2) Balance redundancy cleanup -> drop QS/OP components + source/fallback metadata
    3) Fossil generation aggregation -> `generation_fossil_total_mw`
    """
    before_cols = len(df.columns)

    # --- Block 1: Foreign Price Spread aggregation ---
    if "da_price_eur" not in df.columns:
        LOGGER.info("Skip final aggregation block 1 (missing da_price_eur).")
    else:
        foreign_da_cols = [
            c
            for c in (
                "da_price_AT",
                "da_price_BE",
                "da_price_CH",
                "da_price_CZ",
                "da_price_DK1",
                "da_price_DK2",
                "da_price_FR",
                "da_price_NL",
                "da_price_PL",
                "da_price_SE4",
            )
            if c in df.columns
        ]
        if foreign_da_cols:
            df = df.with_columns(
                (
                    pl.sum_horizontal([pl.col(c).cast(pl.Float64, strict=False) for c in foreign_da_cols])
                    / float(len(foreign_da_cols))
                ).alias("neighbor_price_avg")
            )
            df = df.with_columns(
                (
                    pl.col("neighbor_price_avg").cast(pl.Float64, strict=False)
                    - pl.col("da_price_eur").cast(pl.Float64, strict=False)
                ).alias("neighbor_spread_avg")
            )
            df = df.drop(foreign_da_cols)

    # --- Block 2: Balance redundancy cleanup ---
    balance_drop = [
        c
        for c in (
            "NRV_balance_qs",
            "NRV_balance_op",
            "rz_saldo_mw_qs",
            "rz_saldo_mw_op",
            "NRV_balance_source",
            "NRV_balance_source_is_fallback",
            "rz_saldo_mw_source",
            "rz_saldo_mw_source_is_fallback",
        )
        if c in df.columns
    ]
    if balance_drop:
        df = df.drop(balance_drop)

    # --- Block 3: Fossil generation aggregation ---
    fossil_cols = [
        c
        for c in (
            "generation_fossil_brown_coal_mw",
            "generation_fossil_hard_coal_mw",
            "generation_fossil_gas_mw",
        )
        if c in df.columns
    ]
    if fossil_cols:
        df = df.with_columns(
            (
                pl.sum_horizontal([pl.col(c).cast(pl.Float64, strict=False) for c in fossil_cols])
            ).alias("generation_fossil_total_mw")
        )
        df = df.drop(fossil_cols)

    after_cols = len(df.columns)
    LOGGER.info(
        "Applied final feature aggregations: columns %s -> %s (delta=%s)",
        before_cols,
        after_cols,
        after_cols - before_cols,
    )
    return df


def _lag_hours_for_column(col: str) -> int:
    """Return lag (hours) for PiT alignment based on column semantics."""
    if (
        "_actual_entsoe" in col
        # Legacy/raw actuals from non-ENTSO-E paths must also be delayed.
        or col in {"wind_onshore_actual", "wind_offshore_actual", "solar_actual", "load_actual"}
        or "generation_fossil_" in col
        or col == "generation_nuclear_mw"
        or "generation_hydro_" in col
        or col == "residual_load_actual"
        or col == "NRV_balance"
        or col == "residual_load_calc"
        or col == "wind_total_error_da"
        or col == "unplanned_outages_mw"
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
    """Apply DA publication gates only for explicit future-horizon forecast columns.

    Design:
    - DA prices remain strict T-0 and are never gated here.
    - Current-hour DA forecasts (base columns without horizon suffix) remain strict T-0.
    - Future DA forecast columns (e.g. `*_forecast_da_*_h48`) are gated by publication
      time: D-1 13:00 Europe/Berlin of the target delivery day.
    - If unavailable at decision time, fallback is persistence (`shift(24)`).
    """
    if "timestamp_utc" not in df.columns:
        return df

    base_forecast_cols = {
        "load_forecast_da_entsoe",
        "load_forecast_da",
        "wind_onshore_forecast_da_entsoe",
        "wind_offshore_forecast_da_entsoe",
        "solar_forecast_da_entsoe",
    }
    # Match future-horizon forecast columns like:
    # - load_forecast_da_entsoe_h48
    # - wind_onshore_forecast_da_entsoe_tplus24
    horizon_re = re.compile(
        r"^(?P<base>.+forecast_da(?:_entsoe)?)(?:_(?:h|tplus)(?P<h>\d+))$",
        flags=re.IGNORECASE,
    )

    decision_ts = pd.to_datetime(df["timestamp_utc"].to_pandas(), utc=True, errors="coerce")
    if decision_ts.isna().any():
        return df

    pdf = df.to_pandas()
    changed = False
    for col in list(pdf.columns):
        if col not in base_forecast_cols:
            m = horizon_re.match(col)
            if not m:
                continue
            base = m.group("base")
            if base not in base_forecast_cols:
                continue
            horizon_h = int(m.group("h"))
        else:
            # Strict T-0 overlap for current delivery-hour DA forecasts.
            horizon_h = 0

        if horizon_h <= 0:
            continue

        target_ts = decision_ts + pd.to_timedelta(horizon_h, unit="h")
        target_local = target_ts.dt.tz_convert("Europe/Berlin")
        pub_local = (target_local.dt.normalize() - pd.Timedelta(days=1)) + pd.Timedelta(hours=13)
        pub_utc = pub_local.dt.tz_convert("UTC")
        available = decision_ts >= pub_utc

        s = pd.to_numeric(pdf[col], errors="coerce")
        fallback = s.shift(24)
        gated = s.where(available, fallback).ffill()
        pdf[col] = gated
        changed = True

    if not changed:
        return df
    return pl.from_pandas(pdf)


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
    }
    residual_actual_col = "residual_load_calc" if "residual_load_calc" in df.columns else (
        "residual_load_actual" if "residual_load_actual" in df.columns else None
    )
    if actual_share_cols.issubset(df.columns) and residual_actual_col is not None:
        df = df.with_columns([
            (pl.col("wind_onshore_actual").cast(pl.Float64).fill_null(0.0)).alias("__wind_onshore_actual_f"),
            (pl.col("wind_offshore_actual").cast(pl.Float64).fill_null(0.0)).alias("__wind_offshore_actual_f"),
            (pl.col("solar_actual").cast(pl.Float64).fill_null(0.0)).alias("__solar_actual_f"),
            pl.col(residual_actual_col).cast(pl.Float64).fill_null(0.0).alias("__residual_load_actual_f"),
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

    load_fc_col = "load_forecast_da_entsoe" if "load_forecast_da_entsoe" in df.columns else "load_forecast_da"
    load_act_col = "load_actual_entsoe" if "load_actual_entsoe" in df.columns else "load_actual"
    pairs = {
        "solar": ("solar_forecast_da_entsoe", "solar_actual_entsoe"),
        "wind_onshore": ("wind_onshore_forecast_da_entsoe", "wind_onshore_actual_entsoe"),
        "wind_offshore": ("wind_offshore_forecast_da_entsoe", "wind_offshore_actual_entsoe"),
        "load": (load_fc_col, load_act_col),
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
    # Primary T-0 renewable signal should come from intraday forecasts (ID).
    # Capture the forecast "news" between DA and ID as a direct imbalance driver.
    if {
        "wind_onshore_forecast_id_entsoe",
        "wind_offshore_forecast_id_entsoe",
        "wind_onshore_forecast_da_entsoe",
        "wind_offshore_forecast_da_entsoe",
    }.issubset(df.columns):
        df = df.with_columns(
            (
                pl.col("wind_onshore_forecast_id_entsoe")
                + pl.col("wind_offshore_forecast_id_entsoe")
                - pl.col("wind_onshore_forecast_da_entsoe")
                - pl.col("wind_offshore_forecast_da_entsoe")
            ).alias("wind_total_forecast_update")
        )
    if {"solar_forecast_id_entsoe", "solar_forecast_da_entsoe"}.issubset(df.columns):
        df = df.with_columns(
            (
                pl.col("solar_forecast_id_entsoe")
                - pl.col("solar_forecast_da_entsoe")
            ).alias("solar_forecast_update")
        )

    if {
        load_fc_col,
        "wind_onshore_forecast_da_entsoe",
        "wind_offshore_forecast_da_entsoe",
        "solar_forecast_da_entsoe",
    }.issubset(df.columns):
        df = df.with_columns(
            (
                pl.col(load_fc_col)
                - pl.col("wind_onshore_forecast_da_entsoe")
                - pl.col("wind_offshore_forecast_da_entsoe")
                - pl.col("solar_forecast_da_entsoe")
            )
            .alias("residual_load_forecast_da")
        )
    # ENTSO-E based total system demand proxy including pumping load.
    if {"load_actual_entsoe", "hydro_pumped_actual_entsoe"}.issubset(df.columns):
        df = df.with_columns(
            (
                pl.col("load_actual_entsoe").cast(pl.Float64)
                + pl.col("hydro_pumped_actual_entsoe").cast(pl.Float64)
            ).alias("load_total_incl_pumping")
        )

    # Forecast-vs-realized residual imbalance proxy for aFRR stress conditions.
    if {"residual_load_forecast_da", "residual_load_calc"}.issubset(df.columns):
        df = df.with_columns(
            (
                pl.col("residual_load_forecast_da").cast(pl.Float64)
                - pl.col("residual_load_calc").cast(pl.Float64)
            ).alias("residual_forecast_error")
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
        "residual_load_forecast_da",
        "load_total_incl_pumping",
        "residual_forecast_error",
        "wind_total_forecast_update",
        "solar_forecast_update",
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


def add_advanced_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced ML features on already PiT-lagged/gated inputs.

    This function is intentionally pandas-based and vectorized for fast feature
    generation across tree-based, neural, and linear model families.
    """
    out = df.copy()

    if "timestamp_utc" in out.columns:
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
        out = out.sort_values("timestamp_utc").reset_index(drop=True)

    # 1) Point-in-time slices (explicit lags for tree models).
    # Keep the primary T+0 columns unchanged and add lag snapshots for seasonality/memory.
    lag_targets = [
        "load_actual_entsoe",
        "da_price_eur",
        "wind_onshore_actual_entsoe",
        "load_total_incl_pumping",
        "residual_load_forecast_da",
        "residual_forecast_error",
    ]
    for col in lag_targets:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_lag24"] = s.shift(24)
        out[f"{col}_lag48"] = s.shift(48)
        out[f"{col}_lag168"] = s.shift(168)

    # 2) Momentum/trend features (differences for linear models).
    momentum_targets = ["load_actual_entsoe", "da_price_eur"]
    for col in momentum_targets:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_diff1"] = s.diff(1)
        out[f"{col}_diff24"] = s.diff(24)

    # 3) EWMA features (recently weighted context for sequence models).
    ewma_targets = ["da_price_eur", "load_actual_entsoe"]
    for col in ewma_targets:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_ewma24"] = s.ewm(span=24, adjust=False, min_periods=1, ignore_na=False).mean()

    # 4) Cyclical time features.
    if "timestamp_utc" in out.columns:
        dt = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
        if "hour" not in out.columns:
            out["hour"] = dt.dt.hour.astype("float64")
        if "dayofweek" not in out.columns:
            out["dayofweek"] = dt.dt.dayofweek.astype("float64")
        if "day_of_week" not in out.columns:
            out["day_of_week"] = out["dayofweek"].astype("float64")
        if "month" not in out.columns:
            out["month"] = dt.dt.month.astype("float64")

    # Cyclical encoding prevents the "midnight gap": 23:00 and 00:00 are
    # adjacent on a circle but appear far apart in raw integer space.
    cyclical_specs = [("hour", 24.0), ("dayofweek", 7.0), ("month", 12.0)]
    for col, max_val in cyclical_specs:
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        if col == "month":
            # Map month 1..12 to 0..11 for a clean cyclical phase.
            vals = vals - 1.0
        out[f"{col}_sin"] = np.sin(2.0 * np.pi * vals / max_val)
        out[f"{col}_cos"] = np.cos(2.0 * np.pi * vals / max_val)

    # 5) Cross-features / ratios.
    if {"wind_onshore_actual_entsoe", "load_actual_entsoe"}.issubset(out.columns):
        wind = pd.to_numeric(out["wind_onshore_actual_entsoe"], errors="coerce")
        load = pd.to_numeric(out["load_actual_entsoe"], errors="coerce")
        out["renewable_penetration_ratio"] = wind / (load + 1.0)

    return out


def _null_gap_length_expr(value_col: str, out_col: str) -> pl.Expr:
    """Per-row length of the current null run (0 for non-null rows)."""
    is_null = pl.col(value_col).is_null()
    run_id = is_null.ne(is_null.shift(1).fill_null(False)).cum_sum()
    return (
        pl.when(is_null)
        .then(pl.len().over(run_id))
        .otherwise(pl.lit(0))
        .cast(pl.Int32)
        .alias(out_col)
    )


def apply_multi_strategy_imputation(df: pl.DataFrame) -> pl.DataFrame:
    """Apply feature-specific imputation strategies for market data gaps."""
    if "timestamp_utc" not in df.columns or df.height == 0:
        return df

    out = df.with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime(time_zone="UTC"), strict=False)
    ).sort("timestamp_utc")

    filled_coal = 0
    filled_fc_delta = 0
    filled_balancing = 0

    # Coal: weekend ffill, weekday short-gap interpolation, weekday long-gap ffill.
    coal_col = "coal_price_api2"
    if coal_col in out.columns:
        before = int(out.select(pl.col(coal_col).null_count()).item())
        gap_col = "__gap_len_coal"
        out = (
            out.with_columns(
                _null_gap_length_expr(coal_col, gap_col),
                (pl.col("timestamp_utc").dt.weekday() >= 5).alias("__is_weekend"),
            )
            .with_columns(
                pl.when(pl.col(coal_col).is_not_null())
                .then(pl.col(coal_col))
                .when(pl.col("__is_weekend"))
                .then(pl.col(coal_col).fill_null(strategy="forward"))
                .when(pl.col(gap_col) <= 3)
                .then(pl.col(coal_col).interpolate())
                .otherwise(pl.col(coal_col).fill_null(strategy="forward"))
                .alias(coal_col)
            )
            .drop(gap_col, "__is_weekend")
        )
        after = int(out.select(pl.col(coal_col).null_count()).item())
        filled_coal += max(0, before - after)

    # Forecasts + deltas: linear interpolation.
    forecast_cols: list[str] = []
    if "load_forecast_da_entsoe" in out.columns:
        forecast_cols.append("load_forecast_da_entsoe")
    forecast_cols.extend([c for c in out.columns if c.endswith("_forecast_delta")])
    # deduplicate while preserving order
    seen_fc: set[str] = set()
    forecast_cols = [c for c in forecast_cols if not (c in seen_fc or seen_fc.add(c))]

    for col in forecast_cols:
        before = int(out.select(pl.col(col).null_count()).item())
        out = out.with_columns(pl.col(col).interpolate().alias(col))
        after = int(out.select(pl.col(col).null_count()).item())
        filled_fc_delta += max(0, before - after)

    # Balancing offered MW: interpolate only for short gaps up to 6h.
    for col in ("afrr_activation_offered_mw_pos", "afrr_activation_offered_mw_neg"):
        if col not in out.columns:
            continue
        before = int(out.select(pl.col(col).null_count()).item())
        gap_col = f"__gap_len_{col}"
        out = (
            out.with_columns(_null_gap_length_expr(col, gap_col))
            .with_columns(
                pl.when(pl.col(col).is_not_null())
                .then(pl.col(col))
                .when(pl.col(gap_col) <= 6)
                .then(pl.col(col).interpolate())
                .otherwise(pl.col(col))
                .alias(col)
            )
            .drop(gap_col)
        )
        after = int(out.select(pl.col(col).null_count()).item())
        filled_balancing += max(0, before - after)

    print(
        "[imputation] Validation Summary: "
        f"coal_price_api2_filled={filled_coal}, "
        f"forecast_delta_filled={filled_fc_delta}, "
        f"balancing_offered_filled={filled_balancing}"
    )
    return out


def run_feature_sanity_checks(df_features: pd.DataFrame | pl.DataFrame) -> None:
    """Run strict mathematical sanity checks on final feature outputs.

    Prints `[PASS/FAIL]` per test and raises `AssertionError` on failure.
    """
    if isinstance(df_features, pl.DataFrame):
        pdf = df_features.to_pandas()
    else:
        pdf = df_features.copy()

    if "timestamp_utc" in pdf.columns:
        pdf["timestamp_utc"] = pd.to_datetime(pdf["timestamp_utc"], utc=True, errors="coerce")
        pdf = pdf.sort_values("timestamp_utc").reset_index(drop=True)

    def _max_abs_err(a: pd.Series, b: pd.Series) -> float:
        aa, bb = a.align(b, join="inner")
        valid = aa.notna() & bb.notna()
        if valid.sum() == 0:
            return 0.0
        return float((pd.to_numeric(aa[valid], errors="coerce") - pd.to_numeric(bb[valid], errors="coerce")).abs().max())

    # Test A: explicit slices.
    assert {"da_price_eur", "da_price_eur_lag24"}.issubset(pdf.columns), (
        "Missing columns for Test A: da_price_eur / da_price_eur_lag24"
    )
    err_a = _max_abs_err(pdf["da_price_eur_lag24"], pd.to_numeric(pdf["da_price_eur"], errors="coerce").shift(24))
    if err_a == 0.0:
        print("[PASS] Test A (Explicit Slices): da_price_eur_lag24 == da_price_eur.shift(24)")
    else:
        print(f"[FAIL] Test A (Explicit Slices): max_abs_error={err_a:.12g}")
    assert err_a == 0.0, f"Test A failed: max_abs_error={err_a}"

    # Test B: momentum/diff.
    assert {"da_price_eur", "da_price_eur_diff1"}.issubset(pdf.columns), (
        "Missing columns for Test B: da_price_eur / da_price_eur_diff1"
    )
    expected_b = pd.to_numeric(pdf["da_price_eur"], errors="coerce") - pd.to_numeric(pdf["da_price_eur"], errors="coerce").shift(1)
    err_b = _max_abs_err(pdf["da_price_eur_diff1"], expected_b)
    if err_b == 0.0:
        print("[PASS] Test B (Momentum): da_price_eur_diff1 == da_price_eur - shift(1)")
    else:
        print(f"[FAIL] Test B (Momentum): max_abs_error={err_b:.12g}")
    assert err_b == 0.0, f"Test B failed: max_abs_error={err_b}"

    # Test C: cyclic bounds and hour periodicity.
    assert {"hour", "hour_sin", "hour_cos"}.issubset(pdf.columns), (
        "Missing columns for Test C: hour / hour_sin / hour_cos"
    )
    sin_s = pd.to_numeric(pdf["hour_sin"], errors="coerce")
    cos_s = pd.to_numeric(pdf["hour_cos"], errors="coerce")
    min_sin, max_sin = float(sin_s.min(skipna=True)), float(sin_s.max(skipna=True))
    min_cos, max_cos = float(cos_s.min(skipna=True)), float(cos_s.max(skipna=True))
    bounds_ok = (min_sin >= -1.0 - 1e-12) and (max_sin <= 1.0 + 1e-12) and (min_cos >= -1.0 - 1e-12) and (max_cos <= 1.0 + 1e-12)

    hour_vals = pd.to_numeric(pdf["hour"], errors="coerce")
    expected_hour_sin = np.sin(2.0 * np.pi * hour_vals / 24.0)
    expected_hour_cos = np.cos(2.0 * np.pi * hour_vals / 24.0)
    err_c1 = _max_abs_err(sin_s, expected_hour_sin)
    err_c2 = _max_abs_err(cos_s, expected_hour_cos)
    periodic_ok = bool(np.isclose(np.sin(2.0 * np.pi * 24.0 / 24.0), np.sin(0.0), atol=1e-15)) and bool(
        np.isclose(np.cos(2.0 * np.pi * 24.0 / 24.0), np.cos(0.0), atol=1e-15)
    )
    if bounds_ok and err_c1 == 0.0 and err_c2 == 0.0 and periodic_ok:
        print("[PASS] Test C (Cyclic Bounds): hour_sin/hour_cos in [-1,1] and periodicity holds (0 == 24)")
    else:
        print(
            "[FAIL] Test C (Cyclic Bounds): "
            f"bounds_ok={bounds_ok}, err_sin={err_c1:.12g}, err_cos={err_c2:.12g}, periodic_ok={periodic_ok}"
        )
    assert bounds_ok and err_c1 == 0.0 and err_c2 == 0.0 and periodic_ok, (
        "Test C failed: cyclic bounds/periodicity mismatch."
    )

    # Test D: ratio math.
    assert {"renewable_penetration_ratio", "wind_onshore_actual_entsoe", "load_actual_entsoe"}.issubset(pdf.columns), (
        "Missing columns for Test D: renewable_penetration_ratio / wind_onshore_actual_entsoe / load_actual_entsoe"
    )
    expected_d = pd.to_numeric(pdf["wind_onshore_actual_entsoe"], errors="coerce") / (
        pd.to_numeric(pdf["load_actual_entsoe"], errors="coerce") + 1.0
    )
    err_d = _max_abs_err(pd.to_numeric(pdf["renewable_penetration_ratio"], errors="coerce"), expected_d)
    if err_d <= 1e-12:
        print("[PASS] Test D (Ratio): renewable_penetration_ratio math is correct")
    else:
        print(f"[FAIL] Test D (Ratio): max_abs_error={err_d:.12g}")
    assert err_d <= 1e-12, f"Test D failed: max_abs_error={err_d}"

    # Test E: EWMA causality and formula check.
    assert {"da_price_eur", "da_price_eur_ewma24"}.issubset(pdf.columns), (
        "Missing columns for Test E: da_price_eur / da_price_eur_ewma24"
    )
    base = pd.to_numeric(pdf["da_price_eur"], errors="coerce")
    expected_e = base.ewm(span=24, adjust=False, min_periods=1, ignore_na=False).mean()
    err_e = _max_abs_err(pd.to_numeric(pdf["da_price_eur_ewma24"], errors="coerce"), expected_e)
    if err_e <= 1e-12:
        print("[PASS] Test E (EWMA): da_price_eur_ewma24 matches causal ewm(span=24, adjust=False, ignore_na=False)")
    else:
        print(f"[FAIL] Test E (EWMA): max_abs_error={err_e:.12g}")
    assert err_e <= 1e-12, f"Test E failed: max_abs_error={err_e}"


def merge_outages_sidecar(
    df: pl.DataFrame,
    outages_path: Path | None = None,
) -> pl.DataFrame:
    """Left-join outage sidecar onto main table in a causally safe way."""
    if outages_path is None:
        outages_path = Path("data/processed/outages_hourly.parquet")
    if "timestamp_utc" not in df.columns:
        return df
    if not outages_path.exists():
        return df.with_columns([
            pl.lit(0.0).cast(pl.Float64).alias("planned_outages_mw"),
            pl.lit(0.0).cast(pl.Float64).alias("unplanned_outages_mw"),
        ])

    outages = pl.read_parquet(outages_path)
    if "timestamp_utc" not in outages.columns and "timestamp" in outages.columns:
        outages = outages.rename({"timestamp": "timestamp_utc"})
    if "timestamp_utc" not in outages.columns:
        return df.with_columns([
            pl.lit(0.0).cast(pl.Float64).alias("planned_outages_mw"),
            pl.lit(0.0).cast(pl.Float64).alias("unplanned_outages_mw"),
        ])

    keep_cols = ["timestamp_utc", "planned_outages_mw", "unplanned_outages_mw"]
    keep_cols = [c for c in keep_cols if c in outages.columns]
    outages = outages.select(keep_cols)

    merged = df.join(outages, on="timestamp_utc", how="left")
    for c in ("planned_outages_mw", "unplanned_outages_mw"):
        if c in merged.columns:
            merged = merged.with_columns(pl.col(c).cast(pl.Float64).fill_null(0.0).alias(c))
        else:
            merged = merged.with_columns(pl.lit(0.0).cast(pl.Float64).alias(c))
    return merged


def truncate_to_complete_information_core(df: pl.DataFrame) -> pl.DataFrame:
    """Trim leading/trailing NaN zones to keep the complete information core.

    - head gap: warmup NaNs from lags/rolling windows
    - tail gap: horizon NaNs from forward-shifted targets
    """
    if df.height == 0:
        return df

    pdf = df.to_pandas()
    n_rows = len(pdf)
    na_mask = pdf.isna()

    # Exclude fully-null columns from boundary detection.
    # Those columns are data-quality issues and would otherwise collapse the table.
    valid_cols = [c for c in pdf.columns if not bool(na_mask[c].all())]
    if not valid_cols:
        print("[truncate] No valid columns for boundary detection; returning input unchanged.")
        return df

    def _leading_nan_run(mask_col: pd.Series) -> int:
        arr = mask_col.to_numpy(dtype=bool)
        if arr.size == 0:
            return 0
        if arr.all():
            return n_rows
        return int(np.argmax(~arr))

    def _trailing_nan_run(mask_col: pd.Series) -> int:
        arr = mask_col.to_numpy(dtype=bool)
        if arr.size == 0:
            return 0
        if arr.all():
            return n_rows
        return int(np.argmax(~arr[::-1]))

    max_head_gap = max(_leading_nan_run(na_mask[c]) for c in valid_cols)
    max_tail_gap = max(_trailing_nan_run(na_mask[c]) for c in valid_cols)

    start_idx = max_head_gap
    end_exclusive = n_rows - max_tail_gap
    if start_idx >= end_exclusive:
        raise ValueError(
            "Truncation produced empty frame: "
            f"rows={n_rows}, max_head_gap={max_head_gap}, max_tail_gap={max_tail_gap}"
        )

    out = df.slice(start_idx, end_exclusive - start_idx)
    removed_head = start_idx
    removed_tail = n_rows - end_exclusive
    print(
        "[truncate] "
        f"max_head_gap={max_head_gap}, max_tail_gap={max_tail_gap}, "
        f"removed_head={removed_head}, removed_tail={removed_tail}, "
        f"rows_before={n_rows}, rows_after={out.height}"
    )
    if "timestamp_utc" in out.columns:
        ts_min = out.select(pl.col("timestamp_utc").min()).item()
        ts_max = out.select(pl.col("timestamp_utc").max()).item()
        print(f"[truncate] new_start={ts_min}, new_end={ts_max}")

    return out


def build_features(input_path: Path, output_path: Path) -> None:
    """Build ML features from transformed parquet."""
    # --- STEP 1: Load Data ---
    df = pl.read_parquet(input_path)
    if "timestamp_utc" in df.columns:
        df = df.sort("timestamp_utc")
    # Merge outage sidecar before causal lag layer; missing means no outage.
    df = merge_outages_sidecar(df, outages_path=Path("data/processed/outages_hourly.parquet"))
    # Removed due to >98% missing values identified in the feature audit.
    high_missing_drop = [c for c in DROP_HIGH_MISSING_BEFORE_FEATURES if c in df.columns]
    if high_missing_drop:
        df = df.drop(high_missing_drop)

    # Ground-truth target engineering is intentionally performed on raw market
    # observations. `y_true_*` must remain unlagged economic truth.
    df = engineer_targets(df)

    # --- STEP 2: CAUSAL FIREWALL (GLOBAL PiT LAG LAYER) ---
    df = apply_point_in_time_lag_layer(df)

    if "data_is_lagged" not in df.columns:
        raise RuntimeError("Causal firewall failed: missing `data_is_lagged` marker after PiT lag layer.")
    if not bool(df.select(pl.col("data_is_lagged").all()).item()):
        raise RuntimeError("Causal firewall failed: `data_is_lagged` is not true for all rows.")

    # --- STEP 3: HORIZON GATING (PUBLICATION AVAILABILITY) ---
    df = apply_day_ahead_forecast_availability(df)

    # --- STEP 4: DERIVED & HYBRID FEATURES (ON LAGGED/GATED BASE) ---
    df = add_confidence_features(df)
    # Final dimensionality reduction blocks: foreign spreads, balance cleanup, fossil total.
    df = apply_final_feature_aggregations(df)
    df = add_german_holiday_features(df)
    df = add_market_regime_features(df)
    df = add_price_offering_features(df)
    df = add_aggregated_and_cluster_features(df)
    # Advanced ML block (seasonal slices, momentum, EWMA, cyclical terms, ratios).
    # Must run after PiT lagging/gating and before target matrix.
    df = pl.from_pandas(add_advanced_ml_features(df.to_pandas()))
    # Multi-strategy market-aware imputation for internal feature gaps.
    df = apply_multi_strategy_imputation(df)

    # --- STEP 5: TARGET MATRIX (h+1 ... h+72) ---
    df = add_multi_output_targets(df, horizon_hours=72)

    # --- STEP 6: CLEANUP & DROP ---
    # Optional: generic time features for user-centric datasets.
    # For this thesis pipeline, many datasets are system-level and may not include user_id.
    if "timestamp_utc" in df.columns and "user_id" in df.columns:
        df = pl.from_pandas(add_time_features(df.to_pandas(), datetime_col="timestamp_utc", user_col="user_id"))

    drop_cols = [c for c in DROP_FOR_MODEL if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)

    # Trim dynamic warmup/horizon NaN zones before final export.
    df = truncate_to_complete_information_core(df)

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
