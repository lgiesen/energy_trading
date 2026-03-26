"""Refine merged dataset by dropping SMARD redundancies and adding ENTSO-E errors.

Usage:
    ./.venv/bin/python -m energy_trading.processing.drop_redundant_features \
        --in data/processed/all_data_refined.parquet \
        --out data/processed/all_data_pruned.parquet
"""
from __future__ import annotations

import argparse
import logging
import re
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
    # Use UTC only in model-ready datasets.
    "timestamp_cet",
]

# User-requested drops from latest feature-audit cycle.
# Reason: duplicates or variables reserved for optimizer/ex-post analysis.
USER_REQUESTED_DROPS = [
    "afrr_bid_vwap_activation_price_pos",
    "afrr_bid_vwap_activation_price_neg",
    "bid_signed_vwap_eur_mwh_pos",
    "bid_signed_vwap_eur_mwh_neg",
    # activation_rate_phys/ml are used for optimizer/target engineering,
    # not intended as direct ML input features.
    "activation_rate_phys_pos",
    "activation_rate_phys_neg",
    "activation_rate_ml_pos",
    "activation_rate_ml_neg",
    # Final-prune request for compact ML table.
    "capacity_import_export_mw_pos",
    "capacity_import_export_mw_neg",
    "load_actual_entsoe",
    "load_forecast_da_entsoe",
    # Remove raw real-time PICASSO base channels (leakage risk as direct inputs).
    "afrr_picasso_mw_pos",
    "afrr_picasso_mw_neg",
]

# Exact duplicates identified in audit (Pearson r=1.000 against canonical columns).
EXACT_DUPLICATES_AUDIT = [
    "afrr_activated_mw_pos_regelleistung",
    "afrr_activated_mw_neg_regelleistung",
    "afrr_activation_rate_pos",
    "afrr_activation_rate_neg",
    "awarded_capacity_mw_pos",
    "awarded_capacity_mw_neg",
    "bid_alloc_mw_pos",
    "bid_alloc_mw_neg",
    # Remove linear combinations of POS/NEG channels to reduce perfect collinearity.
    "afrr_picasso_net_mw",
    "afrr_picasso_churn_mw",
    "picasso_flow_rate",
]

# Extend this list as audit iterations discover additional non-ML/redundant fields.
ADDITIONAL_AUDIT_DROPS: list[str] = []

# Protection clause: keep causal lag features and regime flags even if a future
# wildcard/regex drop rule becomes too broad.
PROTECTED_KEEP = {
    "market_regime_picasso",
    "is_picasso_regime",
}

# Statistical overload pruning (rolling-window redundancy).
# This list can host explicit manual additions; dynamic pattern-based drops are
# merged in refine_dataset via `_derive_statistical_overload_drops(...)`.
STATISTICAL_OVERLOAD_DROPS: list[str] = []

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

FOREIGN_DA_PRICE_COLS = [
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
]

BALANCE_COMPONENT_DROPS = [
    "NRV_balance_qs",
    "NRV_balance_op",
    "rz_saldo_mw_qs",
    "rz_saldo_mw_op",
]

FOSSIL_COMPONENT_DROPS = [
    "generation_fossil_brown_coal_mw",
    "generation_fossil_hard_coal_mw",
    "generation_fossil_gas_mw",
]

RENEWABLE_DA_FORECAST_DROPS = [
    "wind_onshore_forecast_da_entsoe",
    "wind_offshore_forecast_da_entsoe",
    "solar_forecast_da_entsoe",
]

# Keep only afrr_vwap_pos/neg as canonical activation price pair.
ACTIVATION_PRICE_REDUNDANCY_PATTERNS = [
    "avg_activation_price",
    "marginal_activation_price",
    "bid_avg_activation_price",
]

BALANCE_SIGNAL_DROPS = [
    "rz_saldo_mw",
    "reBAP_shortage_surplus",
]

SPREAD_FOCUS_DROPS = [
    "neighbor_price_avg",
]

HYDRO_COMPONENT_COLS = [
    "hydro_reservoir_actual_entsoe",
    "hydro_pumped_actual_entsoe",
    "hydro_ror_actual_entsoe",
]

AUDIT_REQUESTED_DROPS = [
    "wind_forecast_de",
]

BASELOAD_COMPONENT_DROPS = [
    "biomass_actual_entsoe",
    "generation_nuclear_mw",
]


def _exclude_columns(df: pl.DataFrame, cols: list[str]) -> tuple[pl.DataFrame, list[str]]:
    """Safely exclude columns using select(...exclude(...)) and report removed names."""
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return df, []
    return df.select(pl.all().exclude(existing)), existing


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


def _derive_statistical_overload_drops(columns: list[str]) -> list[str]:
    """Identify rolling-window columns to drop for collinearity control.

    Thesis rationale:
    - Linear/regularized models (Lasso/ElasticNet/Ridge) become unstable under
      highly collinear rolling families (inflated variance / weak selection).
    - Tree models also benefit from reduced redundant windows through better
      generalization and lower risk of fitting noise aliases.
    """
    drops: set[str] = set()
    keep_std24_re = re.compile(r"(price|residual)", flags=re.IGNORECASE)

    for c in columns:
        lc = c.lower()

        # 1) EWMA family is highly collinear with mean_24h family.
        if lc.endswith("_ewma24"):
            drops.add(c)
            continue

        # 2) Mid windows are removed to keep clean 24h/168h seasonality slices.
        if lc.endswith("_mean_48h") or lc.endswith("_std_48h") or lc.endswith("_mean_12h") or lc.endswith("_std_12h"):
            drops.add(c)
            continue

        # 3) Keep std_24h only for price/residual signals; drop other std windows.
        if lc.endswith("_std_168h"):
            drops.add(c)
            continue
        if lc.endswith("_std_24h") and not keep_std24_re.search(lc):
            drops.add(c)
            continue

        # 4) Low-variance rolling std for sluggish families.
        if (lc.endswith("_std_24h") or lc.endswith("_std_168h")) and (
            "biomass" in lc or "capacity" in lc
        ):
            drops.add(c)

    return sorted(drops)


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

    # ---------------------------------------------------------------------
    # Transformation-first refinement:
    # 1) Create compact aggregate signals
    # 2) Drop redundant/raw helper families in one final exclude pass
    # ---------------------------------------------------------------------
    before_cols = len(df_out.columns)

    # Residual load signals (forecast and actual) for DA/aFRR modeling.
    residual_forecast_inputs = [
        "load_forecast_da_entsoe",
        "solar_forecast_id_entsoe",
        "wind_onshore_forecast_id_entsoe",
        "wind_offshore_forecast_id_entsoe",
    ]
    if all(c in df_out.columns for c in residual_forecast_inputs):
        df_out = df_out.with_columns(
            (
                pl.col("load_forecast_da_entsoe").cast(pl.Float64, strict=False)
                - pl.col("solar_forecast_id_entsoe").cast(pl.Float64, strict=False)
                - pl.col("wind_onshore_forecast_id_entsoe").cast(pl.Float64, strict=False)
                - pl.col("wind_offshore_forecast_id_entsoe").cast(pl.Float64, strict=False)
            ).alias("residual_load_forecast")
        )

        # Forecasted renewable share (wind+solar) / load forecast.
        ren_num = (
            pl.col("solar_forecast_id_entsoe").cast(pl.Float64, strict=False)
            + pl.col("wind_onshore_forecast_id_entsoe").cast(pl.Float64, strict=False)
            + pl.col("wind_offshore_forecast_id_entsoe").cast(pl.Float64, strict=False)
        )
        ren_den = pl.col("load_forecast_da_entsoe").cast(pl.Float64, strict=False)
        df_out = df_out.with_columns(
            pl.when(ren_den.is_null() | (ren_den == 0.0))
            .then(None)
            .otherwise(ren_num / ren_den)
            .alias("renewable_share_forecast")
        )

    residual_actual_inputs = [
        "load_actual_entsoe",
        "solar_actual_entsoe",
        "wind_onshore_actual_entsoe",
        "wind_offshore_actual_entsoe",
    ]
    if all(c in df_out.columns for c in residual_actual_inputs):
        df_out = df_out.with_columns(
            (
                pl.col("load_actual_entsoe").cast(pl.Float64, strict=False)
                - pl.col("solar_actual_entsoe").cast(pl.Float64, strict=False)
                - pl.col("wind_onshore_actual_entsoe").cast(pl.Float64, strict=False)
                - pl.col("wind_offshore_actual_entsoe").cast(pl.Float64, strict=False)
            ).alias("residual_load_actual")
        )

    # Neighbor price spread: cross-border pressure vs Germany.
    foreign_existing = [c for c in FOREIGN_DA_PRICE_COLS if c in df_out.columns]
    if "da_price_eur" in df_out.columns and foreign_existing:
        df_out = df_out.with_columns(
            (
                pl.sum_horizontal([pl.col(c).cast(pl.Float64, strict=False) for c in foreign_existing])
                / float(len(foreign_existing))
            ).alias("neighbor_price_avg")
        ).with_columns(
            (
                pl.col("neighbor_price_avg").cast(pl.Float64, strict=False)
                - pl.col("da_price_eur").cast(pl.Float64, strict=False)
            ).alias("neighbor_spread_avg")
        )

    # Fossil total: compact conventional generation signal.
    fossil_existing = [c for c in FOSSIL_COMPONENT_DROPS if c in df_out.columns]
    if fossil_existing:
        df_out = df_out.with_columns(
            pl.sum_horizontal([pl.col(c).cast(pl.Float64, strict=False) for c in fossil_existing]).alias(
                "generation_fossil_total_mw"
            )
        )

    # Stable Baseload consolidation for thesis consistency:
    # Always aggregate biomass + nuclear; if nuclear is zero this remains biomass.
    if all(c in df_out.columns for c in BASELOAD_COMPONENT_DROPS):
        df_out = df_out.with_columns(
            (
                pl.col("biomass_actual_entsoe").cast(pl.Float64, strict=False)
                + pl.col("generation_nuclear_mw").cast(pl.Float64, strict=False)
            ).alias("generation_baseload_total")
        )
    else:
        LOGGER.warning(
            "Skipping generation_baseload_total: missing inputs %s",
            [c for c in BASELOAD_COMPONENT_DROPS if c not in df_out.columns],
        )

    # Hydro total: aggregate hydro actual components into one robust signal.
    hydro_existing = [c for c in HYDRO_COMPONENT_COLS if c in df_out.columns]
    if hydro_existing:
        df_out = df_out.with_columns(
            pl.sum_horizontal([pl.col(c).cast(pl.Float64, strict=False) for c in hydro_existing]).alias(
                "generation_hydro_actual_total"
            )
        )

    exprs, added, _ = _build_feature_exprs(df_out)
    if exprs:
        df_out = df_out.with_columns(exprs)
    LOGGER.info("Added %s ENTSO-E derived error/update features.", len(added))
    if added:
        LOGGER.info("Added columns: %s", added)

    # Dynamically drop metadata lineage helpers.
    metadata_drops = [c for c in df_out.columns if c.endswith("_source") or c.endswith("_is_fallback")]
    activation_price_pattern_drops = [
        c
        for c in df_out.columns
        if any(pat in c.lower() for pat in ACTIVATION_PRICE_REDUNDANCY_PATTERNS)
    ]

    # Compose one final drop set and apply via exclude.
    # Using `exclude` keeps this step idempotent: rerunning on already-pruned
    # datasets won't fail if some columns are already absent.
    dynamic_stat_drops = _derive_statistical_overload_drops(df_out.columns)
    all_to_exclude = sorted(
        set(DROP_COLS)
        .union(USER_REQUESTED_DROPS)
        .union(EXACT_DUPLICATES_AUDIT)
        .union(STATISTICAL_OVERLOAD_DROPS)
        .union(dynamic_stat_drops)
        .union(ADDITIONAL_AUDIT_DROPS)
        .union(FOREIGN_DA_PRICE_COLS)
        .union(BALANCE_COMPONENT_DROPS)
        .union(BALANCE_SIGNAL_DROPS)
        .union(FOSSIL_COMPONENT_DROPS)
        .union(HYDRO_COMPONENT_COLS)
        .union(BASELOAD_COMPONENT_DROPS)
        .union(RENEWABLE_DA_FORECAST_DROPS)
        .union(SPREAD_FOCUS_DROPS)
        .union(AUDIT_REQUESTED_DROPS)
        .union(activation_price_pattern_drops)
        .union(metadata_drops)
    )
    # Never drop lag features or protected regime indicators.
    removed_existing = [
        c
        for c in all_to_exclude
        if c in df_out.columns and ("lag" not in c.lower()) and (c not in PROTECTED_KEEP)
    ]
    if removed_existing:
        df_out = df_out.select(pl.all().exclude(removed_existing))

    LOGGER.info("Total columns dropped during refinement: %s", len(removed_existing))
    LOGGER.info("Columns before refinement=%s, after refinement=%s", before_cols, len(df_out.columns))
    if removed_existing:
        LOGGER.info("Dropped columns: %s", removed_existing)
    created_now = [
        c
        for c in (
            "neighbor_spread_avg",
            "generation_fossil_total_mw",
            "generation_hydro_actual_total",
        )
        if c in df_out.columns
    ]
    LOGGER.info("Created aggregate columns: %s", created_now)

    return df_out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Drop redundant SMARD renewable columns and add ENTSO-E-based errors."
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        default="data/processed/all_data_refined.parquet",
        help="Input refined parquet path.",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default="data/processed/all_data_pruned.parquet",
        help="Output pruned parquet path.",
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
