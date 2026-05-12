"""Feature engineering for model-ready training/backtest data.

Target design (next-hour shift, canonical names without suffix):
- `target_afrr_activation_price_vwap_pos`: next-hour positive aFRR activation-price VWAP target.
- `target_afrr_activation_price_vwap_neg`: next-hour negative aFRR activation-price VWAP target.
- `target_da_price`: next-hour DA price target.
- `target_afrr_activation_rate_pos`: next-hour positive aFRR activation-rate target.
- `target_afrr_activation_rate_neg`: next-hour negative aFRR activation-rate target.
- `target_afrr_capacity_price_pos`: next-hour positive aFRR capacity-price target.
- `target_afrr_capacity_price_neg`: next-hour negative aFRR capacity-price target.

Rationale:
- Targets are explicitly prefixed with `target_` to avoid ambiguity in ML flows.
- Historical lag columns (`*_lag_*h`) remain feature-only by naming convention.
- Average activation prices are not valid settlement targets in pay-as-cleared setup.
- Extreme sentinel values (about +/-99,999) are technical artifacts and are neutralized
  only when activation is effectively zero.
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

from energy_trading.constants import DA_GATE_HOUR_UTC, PICASSO_RELEASE_UTC

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
    # Raw calendar integers are dropped in favor of cyclical sin/cos encodings.
    "hour",
    "dayofweek",
    "day_of_week",
    "month",
    "dayofyear",
    # Keep only lagged PICASSO flow features in final ML table.
    "picasso_flow_rate",
    # Remove raw real-time PICASSO channels; keep only engineered lagged proxy.
    "afrr_picasso_mw_pos",
    "afrr_picasso_mw_neg",
    # Use publication-gated DA view in X.
    "da_price",
]

# Backward-compatible guard list (legacy columns are removed upstream in
# refine_market_data.py and transform_data.py).
DROP_HIGH_MISSING_BEFORE_FEATURES: list[str] = []

TRUNCATION_CORE_COLUMNS = [
    "da_price_pit",
    "load_actual_entsoe_lag_2h",
    "afrr_activation_price_vwap_pos_lag_1h",
]

DATA_IS_LAGGED = True
LOGGER = logging.getLogger(__name__)


LAG_SUFFIX_RE = re.compile(r"_lag_(\d+)h$")


def _split_effective_lag(col: str) -> tuple[str, int]:
    """Return (root_name, effective_lag_h) from a column name."""
    m = LAG_SUFFIX_RE.search(col)
    if not m:
        return col, 0
    return col[: m.start()], int(m.group(1))


def apply_final_feature_aggregations(df: pl.DataFrame) -> pl.DataFrame:
    """Apply final feature aggregation/reduction blocks with economic semantics.

    Blocks:
    1) Foreign DA prices -> `neighbor_price_avg`, `neighbor_spread_avg`
    2) Balance redundancy cleanup -> drop QS/OP components + source/fallback metadata
    3) Fossil generation aggregation -> `generation_fossil_total_mw`
    """
    before_cols = len(df.columns)

    # --- Block 1: Foreign Price Spread aggregation ---
    if "da_price_pit" not in df.columns:
        LOGGER.info("Skip final aggregation block 1 (missing da_price_pit).")
    else:
        foreign_da_cols = [
            c
            for c in (
                "da_price_AT_pit",
                "da_price_BE_pit",
                "da_price_CH_pit",
                "da_price_CZ_pit",
                "da_price_DK1_pit",
                "da_price_DK2_pit",
                "da_price_FR_pit",
                "da_price_NL_pit",
                "da_price_PL_pit",
                "da_price_SE4_pit",
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
                    - pl.col("da_price_pit").cast(pl.Float64, strict=False)
                ).alias("neighbor_spread_avg")
            )
            # Keep a small set of explicit bilateral spreads as DA alpha probes.
            bilateral_spread_exprs: list[pl.Expr] = []
            for country in ("AT", "NL", "FR"):
                c = f"da_price_{country}_pit"
                if c in df.columns:
                    bilateral_spread_exprs.append(
                        (
                            pl.col("da_price_pit").cast(pl.Float64, strict=False)
                            - pl.col(c).cast(pl.Float64, strict=False)
                        ).alias(f"da_spread_de_{country.lower()}")
                    )
            if bilateral_spread_exprs:
                df = df.with_columns(bilateral_spread_exprs)
            # Keep only DE anchor and spread signal; drop foreign raw prices and
            # intermediate neighbor average to reduce collinearity.
            df = df.drop(foreign_da_cols + ["neighbor_price_avg"])

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
        or "afrr_capacity_" in col
        or "activation_rate" in col
        or "mfrr_activated_mw" in col
        or "mfrr_activated_mwh" in col
        or "afrr_marginal_activation_price" in col
        or "afrr_activation_marginal_price" in col
        or "afrr_activation_price_vwap" in col
        or "afrr_vwap" in col
        or "reBAP_shortage_surplus" in col
        or "afrr_picasso_mw" in col
        or "afrr_picasso_net_mw" in col
        or "mfrr_mari_net_mw" in col
    ):
        return 1
    return 0


def apply_point_in_time_lag_layer(df: pl.DataFrame) -> pl.DataFrame:
    """Apply strict PiT publication lags to observation features.

    Ground-truth target columns are protected and never lagged.
    """
    protected = {
        "timestamp_utc",
        "__target_source_afrr_activation_price_vwap_pos",
        "__target_source_afrr_activation_price_vwap_neg",
        "__target_source_da_price",
        "__target_source_afrr_rate_pos",
        "__target_source_afrr_rate_neg",
        "__target_source_afrr_capacity_price_pos",
        "__target_source_afrr_capacity_price_neg",
        "target_afrr_activation_price_vwap_pos",
        "target_afrr_activation_price_vwap_neg",
        "target_da_price",
        "target_afrr_activation_rate_pos",
        "target_afrr_activation_rate_neg",
        "target_afrr_capacity_price_pos",
        "target_afrr_capacity_price_neg",
    }
    exprs: list[pl.Expr] = []
    lagged_cols = 0
    for col in df.columns:
        if col in protected:
            continue
        if LAG_SUFFIX_RE.search(col):
            continue
        lag_h = _lag_hours_for_column(col)
        if lag_h > 0:
            # Keep shifted canonical column for downstream feature construction and
            # additionally expose an explicit lag-suffixed twin for auditability.
            exprs.append(pl.col(col).shift(lag_h).alias(col))
            exprs.append(pl.col(col).shift(lag_h).alias(f"{col}_lag_{lag_h}h"))
            lagged_cols += 1
    if exprs:
        df = df.with_columns(exprs)
    return df.with_columns([
        pl.lit(DATA_IS_LAGGED).cast(pl.Boolean).alias("data_is_lagged"),
        pl.lit(lagged_cols).cast(pl.Int32).alias("pit_lagged_column_count"),
    ])


def add_day_ahead_publication_feature(df: pl.DataFrame) -> pl.DataFrame:
    """Build day-ahead price views under a publication-time information gate.

    Rule:
    - For a delivery hour T, day-ahead price is considered available from
      D-1 `DA_GATE_HOUR_UTC` in UTC.
    - Before publication, use a causal fallback (`shift(24)`), i.e. prior-day
      same-hour DA price.
    """
    if "timestamp_utc" not in df.columns or "da_price" not in df.columns:
        return df

    pdf = df.to_pandas()
    ts = pd.to_datetime(pdf["timestamp_utc"], utc=True, errors="coerce")
    da = pd.to_numeric(pdf["da_price"], errors="coerce")

    delivery_day_utc = ts.dt.floor("D")
    publication_ts = delivery_day_utc - pd.Timedelta(days=1) + pd.Timedelta(hours=DA_GATE_HOUR_UTC)
    is_available = ts >= publication_ts

    pdf["da_price_pit"] = da.where(is_available, da.shift(24))

    # Apply the same D-1 13:00 UTC gate to foreign DA prices used for cross-border
    # spreads, so spread features are causally aligned to auction availability.
    foreign_da_cols = [c for c in pdf.columns if re.match(r"^da_price_[A-Z0-9]+$", c)]
    for col in foreign_da_cols:
        s = pd.to_numeric(pdf[col], errors="coerce")
        pdf[f"{col}_pit"] = s.where(is_available, s.shift(24))
    return pl.from_pandas(pdf)


def apply_day_ahead_forecast_availability(df: pl.DataFrame) -> pl.DataFrame:
    """Apply DA publication gates only for explicit future-horizon forecast columns.

    Design:
    - DA prices remain strict T-0 and are never gated here.
    - Current-hour DA forecasts (base columns without horizon suffix) remain strict T-0.
    - Future DA forecast columns (e.g. `*_forecast_da_*_h24`) are gated by publication
      time: D-1 `DA_GATE_HOUR_UTC` in UTC of the target delivery day.
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
    # - load_forecast_da_entsoe_h24
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
        target_day_utc = target_ts.dt.floor("D")
        pub_utc = target_day_utc - pd.Timedelta(days=1) + pd.Timedelta(hours=DA_GATE_HOUR_UTC)
        available = decision_ts >= pub_utc

        s = pd.to_numeric(pdf[col], errors="coerce")
        fallback = s.shift(24)
        gated = s.where(available, fallback).ffill()
        pdf[col] = gated
        changed = True

    if not changed:
        return df
    return pl.from_pandas(pdf)


def add_da_forecast_curve_features(
    df: pl.DataFrame,
    *,
    sparse_horizons: tuple[int, ...] = (1, 2, 3, 6, 12, 24),
    max_horizon: int = 24,
) -> pl.DataFrame:
    """Add causal DA-forecast trajectory features (sparse + compressed).

    Goal:
    - expose future forecast shape for the next day in a PiT-safe way
    - keep dimensionality manageable via sparse horizons + summary stats

    Causality:
    - For each horizon h and decision time t, forecast for target t+h is only
      available from D-1 `DA_GATE_HOUR_UTC` of the target delivery day.
    - If unavailable, fallback uses a causal persistence proxy (`shift(24)`).
    """
    if "timestamp_utc" not in df.columns:
        return df

    pdf = df.to_pandas()
    pdf["timestamp_utc"] = pd.to_datetime(pdf["timestamp_utc"], utc=True, errors="coerce")
    pdf = pdf.sort_values("timestamp_utc").reset_index(drop=True)
    ts = pd.to_datetime(pdf["timestamp_utc"], utc=True, errors="coerce")
    if ts.isna().any():
        return df

    # Forecast families that are DA-curve based and meaningful for next-day outlook.
    candidates = [
        "load_forecast_da_entsoe",
        "load_forecast_da",
        "wind_onshore_forecast_da_entsoe",
        "wind_offshore_forecast_da_entsoe",
        "solar_forecast_da_entsoe",
        "residual_load_forecast_da",
        "renewable_share_forecast",
    ]
    base_cols = [c for c in candidates if c in pdf.columns]
    if not base_cols:
        return df

    max_h = int(max_horizon)
    sparse = tuple(sorted({int(h) for h in sparse_horizons if 1 <= int(h) <= max_h}))
    if not sparse:
        return df

    for base in base_cols:
        s = pd.to_numeric(pdf[base], errors="coerce")
        fallback_24h = s.shift(24)

        dense_values: dict[int, pd.Series] = {}
        for h in range(1, max_h + 1):
            target_ts = ts + pd.to_timedelta(h, unit="h")
            target_day_utc = target_ts.dt.floor("D")
            pub_utc = target_day_utc - pd.Timedelta(days=1) + pd.Timedelta(hours=DA_GATE_HOUR_UTC)
            available = ts >= pub_utc

            future = s.shift(-h)
            gated = future.where(available, fallback_24h).ffill()
            dense_values[h] = pd.to_numeric(gated, errors="coerce")

        # Keep sparse horizon points for tree models.
        for h in sparse:
            pdf[f"{base}_h{h}"] = dense_values[h]

        # Compressed trajectory descriptors (24h curve shape).
        arr24 = np.column_stack([dense_values[h].to_numpy(dtype=float) for h in range(1, 25)])
        with np.errstate(invalid="ignore"):
            pdf[f"{base}_next24_mean"] = np.nanmean(arr24, axis=1)
            pdf[f"{base}_next24_min"] = np.nanmin(arr24, axis=1)
            pdf[f"{base}_next24_max"] = np.nanmax(arr24, axis=1)
            pdf[f"{base}_next24_std"] = np.nanstd(arr24, axis=1)

        h1 = dense_values[1]
        h24 = dense_values[24]
        pdf[f"{base}_next24_ramp"] = h24 - h1

    return pl.from_pandas(pdf)


def add_explicit_targets(df: pl.DataFrame) -> pl.DataFrame:
    """Create explicit next-hour targets and drop temporary unlagged target sources."""
    required = {
        "__target_source_afrr_activation_price_vwap_pos",
        "__target_source_afrr_activation_price_vwap_neg",
        "__target_source_da_price",
        "__target_source_afrr_rate_pos",
        "__target_source_afrr_rate_neg",
        "__target_source_afrr_capacity_price_pos",
        "__target_source_afrr_capacity_price_neg",
    }
    if not required.issubset(set(df.columns)):
        missing = sorted(required - set(df.columns))
        raise KeyError(f"Missing required target-source columns: {missing}")

    out = df.with_columns([
        pl.col("__target_source_afrr_activation_price_vwap_pos").shift(-1).alias("target_afrr_activation_price_vwap_pos"),
        pl.col("__target_source_afrr_activation_price_vwap_neg").shift(-1).alias("target_afrr_activation_price_vwap_neg"),
        pl.col("__target_source_da_price").shift(-1).alias("target_da_price"),
        pl.col("__target_source_afrr_rate_pos").shift(-1).alias("target_afrr_activation_rate_pos"),
        pl.col("__target_source_afrr_rate_neg").shift(-1).alias("target_afrr_activation_rate_neg"),
        pl.col("__target_source_afrr_capacity_price_pos").shift(-1).alias("target_afrr_capacity_price_pos"),
        pl.col("__target_source_afrr_capacity_price_neg").shift(-1).alias("target_afrr_capacity_price_neg"),
    ])

    # Preserve raw (non-imputed) activation-price targets for settlement/forensics.
    out = out.with_columns([
        pl.col("target_afrr_activation_price_vwap_pos").alias("target_afrr_activation_price_vwap_pos_raw"),
        pl.col("target_afrr_activation_price_vwap_neg").alias("target_afrr_activation_price_vwap_neg_raw"),
    ])

    # Strict no-activation semantics: price is undefined when rate is zero.
    out = out.with_columns([
        pl.when(pl.col("target_afrr_activation_rate_pos").cast(pl.Float64).fill_null(0.0).abs() <= 1e-12)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("target_afrr_activation_price_vwap_pos").cast(pl.Float64))
        .alias("target_afrr_activation_price_vwap_pos"),
        pl.when(pl.col("target_afrr_activation_rate_neg").cast(pl.Float64).fill_null(0.0).abs() <= 1e-12)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("target_afrr_activation_price_vwap_neg").cast(pl.Float64))
        .alias("target_afrr_activation_price_vwap_neg"),
    ])

    # ML-only fallback imputation: use DA proxy (opportunity-cost anchor).
    da_proxy = pl.coalesce([
        pl.col("target_da_price").cast(pl.Float64),
        pl.col("__target_source_da_price").cast(pl.Float64),
    ])
    out = out.with_columns([
        pl.when(pl.col("target_afrr_activation_price_vwap_pos").is_null())
        .then(da_proxy * 1.1)
        .otherwise(pl.col("target_afrr_activation_price_vwap_pos").cast(pl.Float64))
        .alias("target_afrr_activation_price_vwap_pos"),
        pl.when(pl.col("target_afrr_activation_price_vwap_neg").is_null())
        .then(da_proxy * 0.9)
        .otherwise(pl.col("target_afrr_activation_price_vwap_neg").cast(pl.Float64))
        .alias("target_afrr_activation_price_vwap_neg"),
    ])

    # Guarantee finite ML labels by local directional fallback if DA is unavailable.
    out = out.with_columns([
        pl.col("target_afrr_activation_price_vwap_pos")
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
        .alias("target_afrr_activation_price_vwap_pos"),
        pl.col("target_afrr_activation_price_vwap_neg")
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
        .alias("target_afrr_activation_price_vwap_neg"),
    ])
    return out.drop([c for c in required if c in out.columns])


def engineer_targets(df: pl.DataFrame) -> pl.DataFrame:
    """Create temporary unlagged target-source columns used for explicit next-hour targets.

    Inputs required:
    - primary: `afrr_activation_price_vwap_pos`, `afrr_activation_price_vwap_neg`
    - fallback: marginal/average activation-price columns when VWAP is unavailable
    - activation magnitude columns:
      - preferred: `afrr_activated_mwh_pos`, `afrr_activated_mwh_neg`
      - fallback: `afrr_activated_mw_pos`, `afrr_activated_mw_neg`

    Outputs created:
    - `__target_source_afrr_activation_price_vwap_pos`
    - `__target_source_afrr_activation_price_vwap_neg`
    - `__target_source_da_price`
    - `__target_source_afrr_rate_pos`
    - `__target_source_afrr_rate_neg`
    - `__target_source_afrr_capacity_price_pos`
    - `__target_source_afrr_capacity_price_neg`
    """
    # Activation-price targets are built separately for positive and negative sides.
    pos_price_col = "afrr_activation_price_vwap_pos" if "afrr_activation_price_vwap_pos" in df.columns else None
    neg_price_col = "afrr_activation_price_vwap_neg" if "afrr_activation_price_vwap_neg" in df.columns else None

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
    #    Average activation prices are not used as training/evaluation targets.
    drop_avg = [c for c in ("afrr_activation_avg_price_pos", "afrr_activation_avg_price_neg") if c in df.columns]
    if drop_avg:
        df = df.drop(drop_avg)

    # Technical sentinel threshold (~+/-99,999) from source data artifacts.
    # This is NOT a regulatory/economic market cap (e.g. 15,000 EUR/MWh).
    sentinel_abs = 90_000.0

    # 2) Unlagged target sources for economically valid pay-as-cleared VWAP,
    # with only technical sentinel values neutralized when activation is effectively zero.
    df = df.with_columns([
        pl.when(
            (pl.col(pos_price_col).cast(pl.Float64).abs() > sentinel_abs)
            & (pl.col(pos_act_col).cast(pl.Float64).fill_null(0.0).abs() == 0.0)
        )
        .then(0.0)
        .otherwise(pl.col(pos_price_col).cast(pl.Float64))
        .alias("__target_source_afrr_activation_price_vwap_pos"),
        pl.when(
            (pl.col(neg_price_col).cast(pl.Float64).abs() > sentinel_abs)
            & (pl.col(neg_act_col).cast(pl.Float64).fill_null(0.0).abs() == 0.0)
        )
        .then(0.0)
        .otherwise(pl.col(neg_price_col).cast(pl.Float64))
        .alias("__target_source_afrr_activation_price_vwap_neg"),
    ])

    # 3) Unlagged DA target source.
    if "da_price" not in df.columns:
        raise KeyError("Missing required target source column: da_price")
    df = df.with_columns([
        pl.col("da_price").cast(pl.Float64).alias("__target_source_da_price"),
    ])

    # 4) Unlagged directional activation-rate target sources (ML-stable).
    #    Strictly directional to preserve charging/discharging asymmetry.
    def _activation_rate_expr(side: str) -> pl.Expr:
        ml_rate_col = f"activation_rate_ml_{side}"
        if ml_rate_col in df.columns:
            return pl.col(ml_rate_col).cast(pl.Float64).fill_null(0.0).clip(0.0, 1.0)

        activated_mw_col = f"afrr_activated_mw_{side}"
        activated_mwh_col = f"afrr_activated_mwh_{side}"
        awarded_col = f"afrr_capacity_awarded_mw_{side}"
        offered_col = f"afrr_activation_offered_mw_{side}"

        if activated_mw_col in df.columns:
            numerator = pl.col(activated_mw_col).cast(pl.Float64).abs().fill_null(0.0)
        elif activated_mwh_col in df.columns:
            numerator = pl.col(activated_mwh_col).cast(pl.Float64).abs().fill_null(0.0)
        else:
            raise KeyError(
                f"Missing required activation magnitude for side='{side}': "
                f"`{activated_mw_col}` or `{activated_mwh_col}`."
            )

        if awarded_col in df.columns:
            denominator = pl.col(awarded_col).cast(pl.Float64).abs().fill_null(0.0)
        elif offered_col in df.columns:
            denominator = pl.col(offered_col).cast(pl.Float64).abs().fill_null(0.0)
        else:
            raise KeyError(
                f"Missing required awarded/offered capacity for side='{side}': "
                f"`{awarded_col}` or `{offered_col}`."
            )

        return pl.when(denominator > 0.0).then(numerator / denominator).otherwise(0.0).clip(0.0, 1.0)

    rate_pos = _activation_rate_expr("pos")
    rate_neg = _activation_rate_expr("neg")
    df = df.with_columns([
        rate_pos.alias("__target_source_afrr_rate_pos"),
        rate_neg.alias("__target_source_afrr_rate_neg"),
        rate_pos.alias("afrr_activation_rate_pos"),
        rate_neg.alias("afrr_activation_rate_neg"),
    ])

    # 5) Unlagged capacity-price target sources.
    # Capacity prices are required as supervised labels but must never be part of X.
    if {"afrr_capacity_price_pos", "afrr_capacity_price_neg"}.issubset(set(df.columns)):
        df = df.with_columns([
            pl.col("afrr_capacity_price_pos").cast(pl.Float64).alias("__target_source_afrr_capacity_price_pos"),
            pl.col("afrr_capacity_price_neg").cast(pl.Float64).alias("__target_source_afrr_capacity_price_neg"),
        ])
    else:
        missing = [c for c in ("afrr_capacity_price_pos", "afrr_capacity_price_neg") if c not in df.columns]
        raise KeyError(f"Missing required capacity-price target source columns: {missing}")

    # 6) Cleanup raw marginal helper columns to avoid accidental downstream use.
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

    if "da_price_pit" not in df.columns:
        raise KeyError("Missing required column: da_price_pit")

    df = df.with_columns(
        pl.col("da_price_pit")
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
    # Signed load forecast error (actual - DA forecast) as imbalance driver.
    if load_fc_col in df.columns and load_act_col in df.columns:
        df = df.with_columns(
            (
                pl.col(load_act_col).cast(pl.Float64)
                - pl.col(load_fc_col).cast(pl.Float64)
            ).alias("load_error_da")
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
        ("wind_onshore_actual_entsoe", "wind_onshore_actual_entsoe"),
    ]
    extra_rolling_specs.append(("da_price_pit", "da_price"))
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
    - afrr_da_price_spread = afrr_activation_price - da_price
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

    # Strict definition: aFRR-DA spread uses canonical positive VWAP only.
    afrr_price_col = "afrr_activation_price_vwap_pos"
    if afrr_price_col not in pdf.columns:
        return df

    p = pd.to_numeric(pdf[afrr_price_col], errors="coerce")
    # Use DatetimeIndex for offset-based rolling windows.
    p_indexed = p.copy()
    p_indexed.index = pdf["timestamp_utc"]

    # 1) Cross-market spread: strictly against Day-Ahead price.
    if "da_price_pit" in pdf.columns:
        p_da = pd.to_numeric(pdf["da_price_pit"], errors="coerce")
        pdf["afrr_da_price_spread"] = p - p_da
    else:
        pdf["afrr_da_price_spread"] = np.nan

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
        "afrr_da_price_spread",
        "relative_price_competitiveness",
        "price_volatility_short_term",
        "scarcity_price_premium",
    ):
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce").ffill().fillna(0.0)

    return pl.from_pandas(pdf)


def add_aggregated_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add aggregated features without train/test leakage.

    Created features:
    - TE_hour_regime_activation
    - nrv_quantile_5

    Leakage control:
    - Target encoding uses `shift(1)` + `expanding().mean()` inside each
      (`is_picasso_active`, `hour`) group.
    """
    if "timestamp_utc" not in df.columns:
        return df

    pdf = df.to_pandas()
    pdf["timestamp_utc"] = pd.to_datetime(pdf["timestamp_utc"], utc=True, errors="coerce")
    pdf = pdf.sort_values("timestamp_utc").reset_index(drop=True)

    # Ensure required grouping columns exist.
    if "hour" not in pdf.columns:
        pdf["hour"] = pdf["timestamp_utc"].dt.hour.astype(np.int8)
    if "is_picasso_active" not in pdf.columns:
        picasso_start = pd.Timestamp(PICASSO_RELEASE_UTC, tz="UTC")
        pdf["is_picasso_active"] = (pdf["timestamp_utc"] >= picasso_start).astype(np.int8)

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
    te = pdf.groupby(["is_picasso_active", "hour"])["is_activated"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    pdf["TE_hour_regime_activation"] = pd.to_numeric(te, errors="coerce").fillna(global_mean).astype(np.float64)

    # 2) NRV quantile buckets.
    q = pd.qcut(pd.to_numeric(pdf["nrv_zscore_24h"], errors="coerce"), 5, labels=False, duplicates="drop")
    pdf["nrv_quantile_5"] = pd.to_numeric(q, errors="coerce").fillna(0).astype(np.int8)

    return pl.from_pandas(pdf)


def add_market_regime_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add market-regime and grid-stress features without target leakage.

    Features created:
    - `mfrr_active_lag`: mFRR activity flag (1 if activation > 0 else 0)
    - `nrv_zscore_24h`: 24h z-score of NRV balance
    - `picasso_flow_rate`: cross-border balancing flow proxy
    - `picasso_flow_rate_lag_1h`: 1-hour lag of PICASSO flow
    - `picasso_flow_rate_lag_24h`: 24-hour lag of PICASSO flow
    - `grid_stress_index`: composite stress score in [0, 1]
    - `is_picasso_active`: 0 before PICASSO release, 1 from release onward

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
    # Do not fallback to capacity-market import/export channels:
    # capacity snapshots are not a valid substitute for real-time PICASSO energy flow.
    if "picasso_flow_rate" in pdf.columns:
        picasso = pd.to_numeric(pdf["picasso_flow_rate"], errors="coerce").fillna(0.0)
    elif "afrr_optimization_mwh" in pdf.columns:
        picasso = pd.to_numeric(pdf["afrr_optimization_mwh"], errors="coerce").fillna(0.0)
    else:
        picasso = pd.Series(0.0, index=pdf.index)
    pdf["picasso_flow_rate"] = picasso
    pdf["picasso_flow_rate_lag_1h"] = picasso.shift(steps_per_hour).fillna(0.0)
    pdf["picasso_flow_rate_lag_24h"] = picasso.shift(24 * steps_per_hour).fillna(0.0)
    # 4) Composite stress index in [0, 1].
    nrv_norm_den = nrv_abs.rolling(window_24h, min_periods=2).max().replace(0.0, np.nan)
    nrv_norm = (nrv_abs / nrv_norm_den).fillna(0.0).clip(0.0, 1.0)
    pic_abs = picasso.abs()
    pic_norm_den = pic_abs.rolling(window_24h, min_periods=2).max().replace(0.0, np.nan)
    pic_norm = (pic_abs / pic_norm_den).fillna(0.0).clip(0.0, 1.0)
    gsi = 0.4 * nrv_norm + 0.4 * pdf["mfrr_active_lag"].astype(float) + 0.2 * pic_norm
    pdf["grid_stress_index"] = gsi.clip(0.0, 1.0)

    return pl.from_pandas(pdf)


def add_time_features(df: pl.DataFrame, datetime_col: str = "timestamp_utc") -> pl.DataFrame:
    """Add vectorized market time features for hourly power-market series.

    Scope:
    - cyclical encodings (hour/weekday/month)
    - calendar/daypart flags
    - stable sorting by timestamp
    """
    if datetime_col not in df.columns:
        raise KeyError(f"Missing datetime column: {datetime_col}")

    out = df.sort(datetime_col)

    hour_expr = pl.col(datetime_col).dt.hour().cast(pl.Float64)
    weekday_expr = pl.col(datetime_col).dt.weekday().cast(pl.Float64)  # 0..6
    month_expr = pl.col(datetime_col).dt.month().cast(pl.Float64)  # 1..12
    day_expr = pl.col(datetime_col).dt.day()

    # Cyclical encodings prevent artificial discontinuities (e.g. 23:00 vs 00:00).
    out = out.with_columns(
        [
            (2.0 * np.pi * hour_expr / 24.0).sin().alias("hour_sin"),
            (2.0 * np.pi * hour_expr / 24.0).cos().alias("hour_cos"),
            (2.0 * np.pi * weekday_expr / 7.0).sin().alias("weekday_sin"),
            (2.0 * np.pi * weekday_expr / 7.0).cos().alias("weekday_cos"),
            (2.0 * np.pi * month_expr / 12.0).sin().alias("month_sin"),
            (2.0 * np.pi * month_expr / 12.0).cos().alias("month_cos"),
            (pl.col(datetime_col).dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),
            ((day_expr >= 27) | (day_expr <= 3)).cast(pl.Int8).alias("is_payday_period"),
            ((pl.col(datetime_col).dt.hour() >= 6) & (pl.col(datetime_col).dt.hour() <= 11)).cast(pl.Int8).alias(
                "is_morning"
            ),
            ((pl.col(datetime_col).dt.hour() >= 12) & (pl.col(datetime_col).dt.hour() <= 16))
            .cast(pl.Int8)
            .alias("is_afternoon"),
            ((pl.col(datetime_col).dt.hour() >= 17) & (pl.col(datetime_col).dt.hour() <= 22)).cast(pl.Int8).alias(
                "is_evening"
            ),
            ((pl.col(datetime_col).dt.hour() >= 23) | (pl.col(datetime_col).dt.hour() <= 5)).cast(pl.Int8).alias(
                "is_night"
            ),
        ]
    )
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

    # All DA derivatives must use PiT-gated DA information.
    if "da_price_pit" not in out.columns:
        raise KeyError("Missing required column for DA derivatives: da_price_pit")
    da_ref_col = "da_price_pit"

    # 1) Point-in-time slices (explicit lags for tree models).
    # Keep the primary T+0 columns unchanged and add lag snapshots for seasonality/memory.
    lag_targets = [
        "load_actual_entsoe",
        "wind_onshore_actual_entsoe",
        "load_total_incl_pumping",
        "residual_load_forecast_da",
        "residual_forecast_error",
    ]
    for col in lag_targets:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_lag_24h"] = s.shift(24)
        out[f"{col}_lag_48h"] = s.shift(48)
        out[f"{col}_lag_168h"] = s.shift(168)
    if da_ref_col in out.columns:
        s_da = pd.to_numeric(out[da_ref_col], errors="coerce")
        out["da_price_lag_24h"] = s_da.shift(24)
        out["da_price_lag_48h"] = s_da.shift(48)
        out["da_price_lag_168h"] = s_da.shift(168)

    # Historical target lags as valid autoregressive features (no leakage):
    # only past values (t-1h, t-24h) are exposed to the model.
    target_history_cols = [
        "afrr_activation_price_vwap_pos",
        "afrr_activation_price_vwap_neg",
        "afrr_activation_rate_pos",
        "afrr_activation_rate_neg",
        "afrr_activated_mw_pos",
        "afrr_activated_mw_neg",
        "afrr_activated_mwh_pos",
        "afrr_activated_mwh_neg",
        "afrr_da_price_spread",
        "afrr_neg_da_price_spread",
    ]
    for col in target_history_cols:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_lag_24h"] = s.shift(24)

    # 2) Momentum/trend features (differences for linear models).
    momentum_targets = ["load_actual_entsoe"]
    for col in momentum_targets:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_diff1"] = s.diff(1)
        out[f"{col}_diff24"] = s.diff(24)
    if da_ref_col in out.columns:
        s_da = pd.to_numeric(out[da_ref_col], errors="coerce")
        out["da_price_diff1"] = s_da.diff(1)
        out["da_price_diff24"] = s_da.diff(24)

    # 3) EWMA features (recently weighted context for sequence models).
    ewma_targets = ["load_actual_entsoe"]
    for col in ewma_targets:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_ewma24"] = s.ewm(span=24, adjust=False, min_periods=1, ignore_na=False).mean()
    if da_ref_col in out.columns:
        s_da = pd.to_numeric(out[da_ref_col], errors="coerce")
        out["da_price_ewma24"] = s_da.ewm(span=24, adjust=False, min_periods=1, ignore_na=False).mean()
        out["da_price_slog1p"] = np.sign(s_da) * np.log1p(np.abs(s_da))

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


def apply_pit_audit_corrections(df: pl.DataFrame) -> pl.DataFrame:
    """Apply explicit lag suffix corrections for audit-critical columns.

    Policy from final PiT audit:
    - high-risk -> enforce lag_1h and remove raw column from X-export
    - medium-risk -> enforce lag_2h and remove raw column from X-export
    """
    high_risk_cols = [
        "afrr_activation_offered_mw_pos",
        "afrr_activation_offered_mw_neg",
        "afrr_capacity_awarded_mw_pos",
        "afrr_capacity_awarded_mw_neg",
        "afrr_da_price_spread",
        "is_activated",
        "TE_hour_regime_activation",
    ]
    medium_risk_cols = [
        "system_stress_signal",
        "solar_error_da",
        "wind_onshore_error_da",
        "wind_offshore_error_da",
        "load_error_da",
        "wind_onshore_error_id",
        "wind_offshore_error_id",
        "solar_error_id",
        "hydro_pumped_actual_entsoe",
        "total_wind_solar_id_error",
        "wind_onshore_actual_entsoe_mean_24h",
        "wind_onshore_actual_entsoe_std_24h",
        "wind_onshore_actual_entsoe_mean_168h",
        "wind_onshore_actual_entsoe_std_168h",
        "nrv_zscore_24h",
        "nrv_quantile_5",
        "grid_stress_index",
        "relative_price_competitiveness",
        "price_volatility_short_term",
        "scarcity_price_premium",
        "neighbor_spread_avg",
        "da_spread_de_at",
        "da_spread_de_nl",
        "da_spread_de_fr",
        "generation_baseload_total",
    ]

    exprs: list[pl.Expr] = []
    for c in high_risk_cols:
        if c in df.columns and f"{c}_lag_1h" not in df.columns:
            exprs.append(pl.col(c).shift(1).alias(f"{c}_lag_1h"))
    for c in medium_risk_cols:
        if c in df.columns and f"{c}_lag_2h" not in df.columns:
            exprs.append(pl.col(c).shift(2).alias(f"{c}_lag_2h"))
    if exprs:
        df = df.with_columns(exprs)

    # Keep only lagged variants for risky groups in final exported X table.
    to_drop = [c for c in (high_risk_cols + medium_risk_cols) if c in df.columns]
    return df.drop(to_drop) if to_drop else df


def add_strategic_momentum_lags(df: pl.DataFrame) -> pl.DataFrame:
    """Create LAG_PLAN features with absolute-latency naming.

    Momentum group:
    - aFRR prices/spreads, stress/NRV, activation/capacity and forecast-update
      families with additional intraday lags.
    Seasonal group:
    - weather/DA/load family with additional lags [24, 48, 168]

    Naming:
    - if anchor is already lagged (e.g. `x_lag_1h`) and shifted by 2h,
      output is `x_lag_3h` (absolute effective latency).
    """
    lag_plan = {
        "momentum_pit1h": {
            "lags": [1, 2, 3],
            "anchors": [
                "afrr_activation_price_vwap_pos_lag_1h",
                "afrr_activation_price_vwap_neg_lag_1h",
                "afrr_da_price_spread_lag_1h",
            ],
            "fallback_roots": [
                "afrr_activation_price_vwap_pos",
                "afrr_activation_price_vwap_neg",
                "afrr_da_price_spread",
            ],
        },
        "momentum_pit2h_nrv": {
            "lags": [2, 3, 4, 6, 12, 24],
            "anchors": [
                "NRV_balance_lag_2h",
                "nrv_zscore_24h_lag_2h",
                "nrv_quantile_5_lag_2h",
            ],
            "fallback_roots": [
                "NRV_balance",
            ],
        },
        "momentum_pit2h_stress": {
            "lags": [2, 3, 6, 12, 24],
            "anchors": [
                "system_stress_signal_lag_2h",
                "grid_stress_index_lag_2h",
            ],
            "fallback_roots": [],
        },
        "momentum_trend_memory": {
            "lags": [1, 2, 3, 6, 12, 24, 48, 168],
            "anchors": [
                "afrr_da_price_spread_lag_1h",
                "afrr_activation_price_vwap_pos_lag_1h",
                "afrr_activation_price_vwap_neg_lag_1h",
                "da_price_pit",
            ],
            "fallback_roots": [
                "afrr_da_price_spread",
                "afrr_activation_price_vwap_pos",
                "afrr_activation_price_vwap_neg",
                "da_price_pit",
            ],
        },
        "activation_rate_dynamics": {
            "lags": [1, 2, 3, 6, 12, 24],
            "anchors": [
                "afrr_activation_rate_pos_lag_1h",
                "afrr_activation_rate_neg_lag_1h",
                "is_activated_lag_1h",
                "mfrr_active_lag",
            ],
            "fallback_roots": [
                "afrr_activation_rate_pos",
                "afrr_activation_rate_neg",
                "is_activated",
                "mfrr_active_lag",
            ],
        },
        "activation_volume_dynamics": {
            "lags": [1, 2, 3, 6, 12, 24],
            "anchors": [
                "afrr_activated_mw_pos_lag_1h",
                "afrr_activated_mw_neg_lag_1h",
                "mfrr_activated_mw_pos_lag_1h",
                "mfrr_activated_mw_neg_lag_1h",
                "mfrr_mari_net_mw_lag_1h",
                "afrr_activation_offered_mw_pos_lag_1h",
                "afrr_activation_offered_mw_neg_lag_1h",
            ],
            "fallback_roots": [
                "afrr_activated_mw_pos",
                "afrr_activated_mw_neg",
                "mfrr_activated_mw_pos",
                "mfrr_activated_mw_neg",
                "mfrr_mari_net_mw",
                "afrr_activation_offered_mw_pos",
                "afrr_activation_offered_mw_neg",
            ],
        },
        "capacity_market_dynamics": {
            "lags": [1, 2, 3, 6, 12, 24],
            "anchors": [
                "afrr_capacity_awarded_mw_pos_lag_1h",
                "afrr_capacity_awarded_mw_neg_lag_1h",
                "afrr_capacity_offered_mw_pos_lag_1h",
                "afrr_capacity_offered_mw_neg_lag_1h",
                "afrr_capacity_price_pos_lag_1h",
                "afrr_capacity_price_neg_lag_1h",
            ],
            "fallback_roots": [
                "afrr_capacity_awarded_mw_pos",
                "afrr_capacity_awarded_mw_neg",
                "afrr_capacity_offered_mw_pos",
                "afrr_capacity_offered_mw_neg",
                "afrr_capacity_price_pos",
                "afrr_capacity_price_neg",
            ],
        },
        "forecast_update_dynamics": {
            "lags": [1, 2, 3, 6, 12, 24],
            "anchors": [
                "wind_forecast_update",
                "wind_onshore_forecast_update",
                "solar_forecast_update",
                "wind_total_error_da_lag_2h",
                "solar_error_da_lag_2h",
                "wind_onshore_error_da_lag_2h",
            ],
            "fallback_roots": [
                "wind_forecast_update",
                "wind_onshore_forecast_update",
                "solar_forecast_update",
                "wind_total_error_da",
                "solar_error_da",
                "wind_onshore_error_da",
            ],
        },
        "seasonal": {
            "lags": [24, 48, 168],
            "anchors": [
                "da_price_pit",
                "da_spread_de_at_lag_2h",
                "da_spread_de_nl_lag_2h",
                "da_spread_de_fr_lag_2h",
                "wind_onshore_forecast_id_entsoe",
                "wind_offshore_forecast_id_entsoe",
                "solar_forecast_id_entsoe",
                "residual_load_forecast",
                "renewable_share_forecast",
                "load_total_incl_pumping_lag_2h",
            ],
            "fallback_roots": [
                "da_price_pit",
                "da_spread_de_at",
                "da_spread_de_nl",
                "da_spread_de_fr",
                "wind_onshore_forecast_id_entsoe",
                "wind_offshore_forecast_id_entsoe",
                "solar_forecast_id_entsoe",
                "residual_load_forecast",
                "renewable_share_forecast",
                "load_total_incl_pumping",
            ],
        },
    }

    exprs: list[pl.Expr] = []
    planned: set[str] = set()

    for group in lag_plan.values():
        lags = group["lags"]
        anchors = [c for c in group["anchors"] if c in df.columns]
        fallbacks = [c for c in group["fallback_roots"] if c in df.columns]
        candidates: list[str] = []
        for c in anchors + fallbacks:
            if c not in candidates:
                candidates.append(c)

        for anchor in candidates:
            root, anchor_lag = _split_effective_lag(anchor)
            # Avoid names like `*_lag_lag_1h` when an anchor ends with `_lag`
            # but does not carry a numeric lag suffix (e.g. `mfrr_active_lag`).
            if anchor_lag == 0 and root.endswith("_lag"):
                root = root[:-4]
            for target_h in lags:
                if target_h < anchor_lag:
                    continue
                shift_h = target_h - anchor_lag
                out_col = f"{root}_lag_{target_h}h"
                if out_col in df.columns or out_col in planned:
                    continue
                planned.add(out_col)
                if shift_h == 0:
                    exprs.append(pl.col(anchor).alias(out_col))
                else:
                    exprs.append(pl.col(anchor).shift(shift_h).alias(out_col))

    if exprs:
        df = df.with_columns(exprs)
    return df


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
    coal_col = "coal_price"
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
        f"coal_price_filled={filled_coal}, "
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
    assert {"da_price_pit", "da_price_lag_24h"}.issubset(pdf.columns), (
        "Missing columns for Test A: da_price_pit / da_price_lag_24h"
    )
    err_a = _max_abs_err(
        pdf["da_price_lag_24h"],
        pd.to_numeric(pdf["da_price_pit"], errors="coerce").shift(24),
    )
    if err_a == 0.0:
        print("[PASS] Test A (Explicit Slices): da_price_lag_24h == da_price_pit.shift(24)")
    else:
        print(f"[FAIL] Test A (Explicit Slices): max_abs_error={err_a:.12g}")
    assert err_a == 0.0, f"Test A failed: max_abs_error={err_a}"

    # Test B: momentum/diff.
    assert {"da_price_pit", "da_price_diff1"}.issubset(pdf.columns), (
        "Missing columns for Test B: da_price_pit / da_price_diff1"
    )
    expected_b = pd.to_numeric(pdf["da_price_pit"], errors="coerce") - pd.to_numeric(
        pdf["da_price_pit"], errors="coerce"
    ).shift(1)
    err_b = _max_abs_err(pdf["da_price_diff1"], expected_b)
    if err_b == 0.0:
        print("[PASS] Test B (Momentum): da_price_diff1 == da_price_pit - shift(1)")
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
    assert {"da_price_pit", "da_price_ewma24"}.issubset(pdf.columns), (
        "Missing columns for Test E: da_price_pit / da_price_ewma24"
    )
    base = pd.to_numeric(pdf["da_price_pit"], errors="coerce")
    expected_e = base.ewm(span=24, adjust=False, min_periods=1, ignore_na=False).mean()
    err_e = _max_abs_err(pd.to_numeric(pdf["da_price_ewma24"], errors="coerce"), expected_e)
    if err_e <= 1e-12:
        print("[PASS] Test E (EWMA): da_price_ewma24 matches causal ewm(span=24, adjust=False, ignore_na=False)")
    else:
        print(f"[FAIL] Test E (EWMA): max_abs_error={err_e:.12g}")
    assert err_e <= 1e-12, f"Test E failed: max_abs_error={err_e}"


def truncate_to_complete_information_core(df: pl.DataFrame) -> pl.DataFrame:
    """Trim leading/trailing NaN zones to keep the complete information core.

    - head gap: warmup NaNs from lags/rolling windows
    - tail gap: horizon NaNs from forward-shifted targets
    """
    if df.height == 0:
        return df

    pdf = df.to_pandas()
    n_rows = len(pdf)
    available_core = [c for c in TRUNCATION_CORE_COLUMNS if c in pdf.columns]
    if not available_core:
        print(
            "[truncate] Core columns missing; returning input unchanged. "
            f"Expected one of {TRUNCATION_CORE_COLUMNS}."
        )
        return df

    na_mask = pdf[available_core].isna()

    # Exclude fully-null core columns from boundary detection.
    valid_cols = [c for c in available_core if not bool(na_mask[c].all())]
    if not valid_cols:
        print("[truncate] No valid core columns for boundary detection; returning input unchanged.")
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
        f"core_cols={valid_cols}, "
        f"max_head_gap={max_head_gap}, max_tail_gap={max_tail_gap}, "
        f"removed_head={removed_head}, removed_tail={removed_tail}, "
        f"rows_before={n_rows}, rows_after={out.height}"
    )
    if "timestamp_utc" in out.columns:
        ts_min = out.select(pl.col("timestamp_utc").min()).item()
        ts_max = out.select(pl.col("timestamp_utc").max()).item()
        print(f"[truncate] new_start={ts_min}, new_end={ts_max}")

    return out


def assert_removed_legacy_columns_absent(df: pl.DataFrame) -> None:
    """Final audit guard for removed legacy feature families."""
    forbidden = [
        c for c in df.columns
        if ("reconstructed" in c.lower()) or ("grid_share" in c.lower())
    ]
    if forbidden:
        raise RuntimeError(
            "Forbidden legacy columns detected in final feature table: "
            f"{sorted(forbidden)}"
        )


def build_features(input_path: Path, output_path: Path) -> None:
    """Build ML features from transformed parquet."""
    # --- STEP 1: Load Data ---
    df = pl.read_parquet(input_path)
    if "timestamp_utc" in df.columns:
        df = df.sort("timestamp_utc")
    # Removed due to >98% missing values identified in the feature audit.
    high_missing_drop = [c for c in DROP_HIGH_MISSING_BEFORE_FEATURES if c in df.columns]
    if high_missing_drop:
        df = df.drop(high_missing_drop)

    # Target-source engineering is intentionally performed on raw market
    # observations. Source columns stay unlagged until explicit next-hour targets are built.
    df = engineer_targets(df)

    # --- STEP 2: CAUSAL FIREWALL (GLOBAL PiT LAG LAYER) ---
    df = apply_point_in_time_lag_layer(df)

    if "data_is_lagged" not in df.columns:
        raise RuntimeError("Causal firewall failed: missing `data_is_lagged` marker after PiT lag layer.")
    if not bool(df.select(pl.col("data_is_lagged").all()).item()):
        raise RuntimeError("Causal firewall failed: `data_is_lagged` is not true for all rows.")

    # --- STEP 3: HORIZON GATING (PUBLICATION AVAILABILITY) ---
    df = apply_day_ahead_forecast_availability(df)
    df = add_day_ahead_publication_feature(df)

    # --- STEP 4: DERIVED & HYBRID FEATURES (ON LAGGED/GATED BASE) ---
    df = add_confidence_features(df)
    # Add DA forecast trajectory features for multi-horizon modeling.
    # This provides both sparse horizon points and compressed curve descriptors.
    df = add_da_forecast_curve_features(df)
    # Final dimensionality reduction blocks: foreign spreads, balance cleanup, fossil total.
    df = apply_final_feature_aggregations(df)
    df = add_german_holiday_features(df)
    df = add_market_regime_features(df)
    df = add_price_offering_features(df)
    df = add_aggregated_features(df)
    # Advanced ML block (seasonal slices, momentum, EWMA, cyclical terms, ratios).
    # Must run after PiT lagging/gating and before target matrix.
    df = pl.from_pandas(add_advanced_ml_features(df.to_pandas()))
    # Enforce explicit PiT anchors first (especially 2h for NRV/stress family),
    # then expand strategic lag plans from those absolute anchors.
    df = apply_pit_audit_corrections(df)
    df = add_strategic_momentum_lags(df)
    # Multi-strategy market-aware imputation for internal feature gaps.
    df = apply_multi_strategy_imputation(df)

    # --- STEP 5: EXPLICIT TARGETS (t+1) ---
    df = add_explicit_targets(df)

    # --- STEP 6: CLEANUP & DROP ---
    # Market-time features are always added when timestamp is available.
    if "timestamp_utc" in df.columns:
        df = add_time_features(df, datetime_col="timestamp_utc")

    drop_cols = [c for c in DROP_FOR_MODEL if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)

    # Final raw-outcome cleanup: only lagged/pit-safe variants are allowed in X.
    critical_raw_drop = [
        "afrr_da_price_spread",
        "is_activated",
        "TE_hour_regime_activation",
        "afrr_activation_offered_mw_pos",
        "afrr_activation_offered_mw_neg",
        "afrr_capacity_awarded_mw_pos",
        "afrr_capacity_awarded_mw_neg",
        "system_stress_signal",
        "nrv_zscore_24h",
        "grid_stress_index",
        # Legacy alias cleanup: keep only canonical afrr_activation_price_vwap_*.
        "afrr_activation_price_vwap_pos",
        "afrr_activation_price_vwap_neg",
    ]
    critical_raw_drop = [c for c in critical_raw_drop if c in df.columns]
    if critical_raw_drop:
        df = df.drop(critical_raw_drop)

    # Enforce audit naming contract:
    # hide unsuffixed PiT-shifted base columns and keep only explicit *_lag_Xh names.
    shifted_base_cols = []
    for c in df.columns:
        if c.startswith("target_") or LAG_SUFFIX_RE.search(c):
            continue
        lag_h = _lag_hours_for_column(c)
        if lag_h > 0 and f"{c}_lag_{lag_h}h" in df.columns:
            shifted_base_cols.append(c)
    if shifted_base_cols:
        df = df.drop(shifted_base_cols)

    # Installed capacities are slow-moving structural fundamentals (not hourly
    # market outcomes). Backfilling is methodically acceptable to avoid
    # artificial data scarcity from late publication starts; intraday dynamics
    # are captured by actual generation/activation features.
    capacity_cols = [c for c in df.columns if "capacity" in c.lower()]
    if capacity_cols:
        cap_exprs = []
        for c in capacity_cols:
            cap_exprs.append(
                pl.col(c).cast(pl.Float64, strict=False).backward_fill().alias(c)
            )
        df = df.with_columns(cap_exprs)

    # Trim dynamic warmup/horizon NaN zones before final export.
    df = truncate_to_complete_information_core(df)
    assert_removed_legacy_columns_absent(df)

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
