"""Refine merged dataset by dropping SMARD redundancies and adding ENTSO-E errors.

Usage:
    ./.venv/bin/python -m energy_trading.processing.drop_redundant_features \
        --in data/processed/all_data.parquet \
        --out data/processed/all_data_refined.parquet
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
    # Streamlined in final feature set: keep internally calculated residual-load variant.
    "residual_load_actual",
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
    "picasso_flow_rate",
]

# Extend this list as audit iterations discover additional non-ML/redundant fields.
ADDITIONAL_AUDIT_DROPS: list[str] = []

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

    removed_total: set[str] = set()

    df_out, drop_existing = _exclude_columns(df_out, DROP_COLS)
    removed_total.update(drop_existing)
    LOGGER.info("Dropped %s redundant SMARD columns.", len(drop_existing))
    if drop_existing:
        LOGGER.info("Dropped columns: %s", drop_existing)

    df_out, user_drop_existing = _exclude_columns(df_out, USER_REQUESTED_DROPS)
    removed_total.update(user_drop_existing)
    LOGGER.info("Dropped %s user-requested redundancy/optimizer columns.", len(user_drop_existing))
    if user_drop_existing:
        LOGGER.info("Dropped user-requested columns: %s", user_drop_existing)

    df_out, exact_dup_existing = _exclude_columns(df_out, EXACT_DUPLICATES_AUDIT)
    removed_total.update(exact_dup_existing)
    LOGGER.info("Dropped %s exact-duplicate audit columns.", len(exact_dup_existing))
    if exact_dup_existing:
        LOGGER.info("Dropped exact-duplicate columns: %s", exact_dup_existing)

    dynamic_stat_drops = _derive_statistical_overload_drops(df_out.columns)
    stat_overload_candidates = sorted(set(STATISTICAL_OVERLOAD_DROPS).union(dynamic_stat_drops))
    df_out, stat_overload_existing = _exclude_columns(df_out, stat_overload_candidates)
    removed_total.update(stat_overload_existing)
    LOGGER.info("Dropped %s statistical-overload columns.", len(stat_overload_existing))
    if stat_overload_existing:
        LOGGER.info("Dropped statistical-overload columns: %s", stat_overload_existing)

    df_out, additional_drop_existing = _exclude_columns(df_out, ADDITIONAL_AUDIT_DROPS)
    removed_total.update(additional_drop_existing)
    LOGGER.info("Dropped %s additional audit columns.", len(additional_drop_existing))
    if additional_drop_existing:
        LOGGER.info("Dropped additional-audit columns: %s", additional_drop_existing)

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
    LOGGER.info("Total columns dropped during refinement: %s", len(removed_total))

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
