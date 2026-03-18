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
DEFAULT_REGELLEISTUNG_15M_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "raw"
    / "regelleistung_15min"
    / "afrr_price_volume_15min.parquet"
)

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


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def compute_afrr_vwap_from_15min(path: Path) -> pl.DataFrame:
    """Compute hourly aFRR VWAP from 15-minute Regelleistung price+volume data.

    Method:
    - Use activated work/volume as weight.
    - Pre-PICASSO (< 2022-06-22 22:00 UTC): forward-fill hourly posted prices
      inside each hour so :15/:30/:45 can be weighted correctly.
    - Aggregate with group_by_dynamic(every="1h", closed="left").
    - If hourly volume sum is 0 or null, return NaN VWAP.
    """
    if not path.exists():
        LOGGER.info("No 15-minute Regelleistung file found for VWAP: %s", path)
        return pl.DataFrame()

    df = pl.read_parquet(path)
    if "timestamp_utc" not in df.columns:
        LOGGER.warning("Skip VWAP build: missing timestamp_utc in %s", path)
        return pl.DataFrame()

    price_pos_col = _first_existing(
        df.columns,
        [
            "afrr_activation_price_pos_ffill",
            "afrr_avg_activation_price_pos",
            "afrr_activation_avg_price_pos",
            "afrr_marginal_activation_price_pos",
            "afrr_activation_marginal_price_pos",
            "arbeitspreis_pos",  # legacy alias in older exports
        ],
    )
    price_neg_col = _first_existing(
        df.columns,
        [
            "afrr_activation_price_neg_ffill",
            "afrr_avg_activation_price_neg",
            "afrr_activation_avg_price_neg",
            "afrr_marginal_activation_price_neg",
            "afrr_activation_marginal_price_neg",
            "arbeitspreis_neg",  # legacy alias in older exports
        ],
    )
    vol_pos_col = _first_existing(
        df.columns,
        ["afrr_activated_mw_pos", "activated_volume_pos_mw", "abgerufene_arbeit_pos"],
    )
    vol_neg_col = _first_existing(
        df.columns,
        ["afrr_activated_mw_neg", "activated_volume_neg_mw", "abgerufene_arbeit_neg"],
    )

    has_pos = price_pos_col is not None and vol_pos_col is not None
    has_neg = price_neg_col is not None and vol_neg_col is not None
    if not has_pos and not has_neg:
        LOGGER.warning("Skip VWAP build: no usable 15-minute aFRR price/volume columns in %s", path)
        return pl.DataFrame()

    cast_exprs = [
        pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False),
    ]
    for c in (price_pos_col, price_neg_col, vol_pos_col, vol_neg_col):
        if c is not None:
            cast_exprs.append(pl.col(c).cast(pl.Float64, strict=False))
    df = (
        df.with_columns(cast_exprs)
        .drop_nulls(subset=["timestamp_utc"])
        .sort("timestamp_utc")
    )

    cutoff = pl.lit(PICASSO_START_UTC).cast(pl.Datetime(time_unit="us", time_zone="UTC"))
    ff_exprs: list[pl.Expr] = []
    if has_pos:
        if price_pos_col == "afrr_activation_price_pos_ffill":
            ff_exprs.append(pl.col(price_pos_col).alias("__price_pos_ff"))
        else:
            ff_exprs.append(
                pl.when(pl.col("timestamp_utc") < cutoff)
                .then(pl.col(price_pos_col).forward_fill().over(pl.col("timestamp_utc").dt.truncate("1h")))
                .otherwise(pl.col(price_pos_col))
                .alias("__price_pos_ff")
            )
    if has_neg:
        if price_neg_col == "afrr_activation_price_neg_ffill":
            ff_exprs.append(pl.col(price_neg_col).alias("__price_neg_ff"))
        else:
            ff_exprs.append(
                pl.when(pl.col("timestamp_utc") < cutoff)
                .then(pl.col(price_neg_col).forward_fill().over(pl.col("timestamp_utc").dt.truncate("1h")))
                .otherwise(pl.col(price_neg_col))
                .alias("__price_neg_ff")
            )
    df = df.with_columns(ff_exprs)

    calc_exprs: list[pl.Expr] = []
    agg_exprs: list[pl.Expr] = []
    if has_pos:
        calc_exprs.append((pl.col("__price_pos_ff") * pl.col(vol_pos_col).abs()).alias("__weighted_cost_pos"))
        agg_exprs.extend(
            [
                pl.col("__weighted_cost_pos").sum().alias("__sum_weighted_pos"),
                pl.col(vol_pos_col).abs().sum().alias("__sum_vol_pos"),
            ]
        )
    if has_neg:
        calc_exprs.append((pl.col("__price_neg_ff") * pl.col(vol_neg_col).abs()).alias("__weighted_cost_neg"))
        agg_exprs.extend(
            [
                pl.col("__weighted_cost_neg").sum().alias("__sum_weighted_neg"),
                pl.col(vol_neg_col).abs().sum().alias("__sum_vol_neg"),
            ]
        )

    hourly = (
        df.with_columns(calc_exprs)
        .group_by_dynamic(
            index_column="timestamp_utc",
            every="1h",
            period="1h",
            closed="left",
            label="left",
        )
        .agg(agg_exprs)
        .sort("timestamp_utc")
    )

    vwap_exprs: list[pl.Expr] = []
    out_cols: list[str] = []
    if has_pos:
        out_cols.append("afrr_vwap_pos_eur_mwh")
        vwap_exprs.append(
            pl.when(pl.col("__sum_vol_pos").is_null() | (pl.col("__sum_vol_pos") == 0.0))
            .then(pl.lit(float("nan")))
            .otherwise(pl.col("__sum_weighted_pos") / pl.col("__sum_vol_pos"))
            .alias("afrr_vwap_pos_eur_mwh")
        )
    if has_neg:
        out_cols.append("afrr_vwap_neg_eur_mwh")
        vwap_exprs.append(
            pl.when(pl.col("__sum_vol_neg").is_null() | (pl.col("__sum_vol_neg") == 0.0))
            .then(pl.lit(float("nan")))
            .otherwise(pl.col("__sum_weighted_neg") / pl.col("__sum_vol_neg"))
            .alias("afrr_vwap_neg_eur_mwh")
        )

    return hourly.with_columns(vwap_exprs).select(["timestamp_utc"] + out_cols)


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

    # Build and join hourly VWAP from raw 15-minute Regelleistung exports.
    vwap_hourly = compute_afrr_vwap_from_15min(DEFAULT_REGELLEISTUNG_15M_PATH)
    if not vwap_hourly.is_empty():
        df = df.join(vwap_hourly, on="timestamp_utc", how="left", suffix="_from15m")
        for col in ("afrr_vwap_pos_eur_mwh", "afrr_vwap_neg_eur_mwh"):
            from15 = f"{col}_from15m"
            if from15 in df.columns and col in df.columns:
                df = df.with_columns(pl.coalesce([pl.col(from15), pl.col(col)]).alias(col)).drop(from15)
            elif from15 in df.columns:
                df = df.rename({from15: col})
        LOGGER.info("Joined hourly aFRR VWAP from 15-minute source: %s rows", vwap_hourly.height)
    else:
        LOGGER.info("No 15-minute VWAP source joined; continuing with existing hourly columns.")

    # Keep backwards-compatible aliases if legacy names exist.
    if "afrr_vwap_pos_eur_mwh" not in df.columns and "afrr_vwap_pos" in df.columns:
        df = df.with_columns(pl.col("afrr_vwap_pos").cast(pl.Float64, strict=False).alias("afrr_vwap_pos_eur_mwh"))
    if "afrr_vwap_neg_eur_mwh" not in df.columns and "afrr_vwap_neg" in df.columns:
        df = df.with_columns(pl.col("afrr_vwap_neg").cast(pl.Float64, strict=False).alias("afrr_vwap_neg_eur_mwh"))

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
    # Prefer ENTSO-E load, fallback to generic load_actual when present.
    load_col = "load_actual_entsoe" if "load_actual_entsoe" in cols else ("load_actual" if "load_actual" in cols else None)
    if load_col is not None:
        _add_if_possible(
            exprs,
            created,
            [
                load_col,
                "wind_onshore_actual_entsoe",
                "wind_offshore_actual_entsoe",
                "solar_actual_entsoe",
            ],
            "residual_load_calc",
            (
                pl.col(load_col)
                - pl.col("wind_onshore_actual_entsoe")
                - pl.col("wind_offshore_actual_entsoe")
                - pl.col("solar_actual_entsoe")
            ),
            cols,
        )
    else:
        LOGGER.warning("Skip residual_load_calc, missing inputs: ['load_actual_entsoe' or 'load_actual']")
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
