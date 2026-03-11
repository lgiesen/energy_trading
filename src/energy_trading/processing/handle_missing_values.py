"""Centralized missing value handling for modeling.

Usage:
    ./.venv/bin/python -m energy_trading.processing.handle_missing_values \
        --in data/processed/all_data.parquet \
        --out data/processed/all_data_clean.parquet

Check results with:
    ./.venv/bin/python -m energy_trading.processing.compare_missingness \
        --before data/processed/all_data.parquet \
        --after data/processed/all_data_clean.parquet

Policy notes:
- Raw auditability is preserved by writing a new cleaned file; source-level raw files remain untouched.
- Commodity prices are forward-filled (weekend carry) instead of interpolated to avoid leakage:
  the Saturday/Sunday value is treated as Friday close, not as a look-ahead to Monday open.
- Day-ahead Belgium price (`da_price_BE`) uses linear interpolation only for short gaps (<24h),
  where hourly prices are strongly auto-correlated.
- German nuclear generation is set to 0 only when values are missing on/after 2023-04-15
  (post phase-out), never overwriting non-null observations.
"""
from __future__ import annotations

import argparse
from datetime import date
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


def _clean_and_anchor_smard_capacity(df: pl.DataFrame) -> pl.DataFrame:
    """Clean SMARD capacity columns and apply forward+anchor fill without backward fill."""
    cap_cols = [
        "wind_onshore_capacity",
        "wind_offshore_capacity",
        "solar_capacity",
        "gas_capacity",
        "hard_coal_capacity",
        "lignite_capacity",
        "pumped_storage_capacity",
    ]
    existing = [c for c in cap_cols if c in df.columns]
    if not existing:
        return df

    # Normalize textual placeholders (e.g. "-") and cast to numeric.
    df = df.with_columns(
        [
            pl.when(pl.col(c).cast(pl.Utf8, strict=False).str.strip_chars() == pl.lit("-"))
            .then(None)
            .otherwise(pl.col(c))
            .cast(pl.Float64, strict=False)
            .alias(c)
            for c in existing
        ]
    )

    # Source for Jan 2021 baselines: SMARD.de (Electricity generation - Installed generation capacity, Resolution: Year).
    # Link: https://www.smard.de/en/marktdaten?marketDataAttributes=%7B%22resolution%22:%22year%22,%22from%22:1579042800000,%22to%22:1769900399999,%22moduleIds%22:%5B3000189,3003792,3000188,3000194,3004072,3004075,3000198,3000207,3004076,3004073,3004074,3000186%5D,%22selectedCategory%22:3,%22activeChart%22:false,%22style%22:%22color%22,%22categoriesModuleOrder%22:%7B%7D,%22region%22:%22DE-LU%22%7D
    # These anchors prevent data leakage that would occur from backward-filling future capacity additions.
    exprs = []
    if "wind_offshore_capacity" in df.columns:
        exprs.append(
            pl.col("wind_offshore_capacity").fill_null(strategy="forward").fill_null(value=7774.0).alias("wind_offshore_capacity")
        )
    if "lignite_capacity" in df.columns:
        exprs.append(
            pl.col("lignite_capacity").fill_null(strategy="forward").fill_null(value=20487.0).alias("lignite_capacity")
        )
    if "hard_coal_capacity" in df.columns:
        exprs.append(
            pl.col("hard_coal_capacity").fill_null(strategy="forward").fill_null(value=23499.0).alias("hard_coal_capacity")
        )
    if "pumped_storage_capacity" in df.columns:
        exprs.append(
            pl.col("pumped_storage_capacity").fill_null(strategy="forward").fill_null(value=9422.0).alias("pumped_storage_capacity")
        )
    if "wind_onshore_capacity" in df.columns:
        exprs.append(
            pl.col("wind_onshore_capacity").fill_null(strategy="forward").fill_null(value=54666.0).alias("wind_onshore_capacity")
        )
    if "solar_capacity" in df.columns:
        exprs.append(
            pl.col("solar_capacity").fill_null(strategy="forward").fill_null(value=53538.0).alias("solar_capacity")
        )
    if "gas_capacity" in df.columns:
        exprs.append(
            pl.col("gas_capacity").fill_null(strategy="forward").fill_null(value=32038.0).alias("gas_capacity")
        )
    if exprs:
        df = df.with_columns(exprs)
    return df


def _log_null_counts(df: pl.DataFrame, cols: Iterable[str], label: str) -> None:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return
    counts = {c: df.select(pl.col(c).null_count()).item() for c in existing}
    LOGGER.info("%s null counts: %s", label, counts)


def _null_count_dict(df: pl.DataFrame) -> dict[str, int]:
    return {c: df.select(pl.col(c).null_count()).item() for c in df.columns}


def _build_treatment_map(df: pl.DataFrame) -> dict[str, str]:
    treatment: dict[str, str] = {}

    for c in ("co2_price_eua", "gas_price_ttf", "coal_price_api2", "coal_price_api"):
        if c in df.columns:
            treatment[c] = "forward_fill (weekend carry, no interpolation)"

    cap_anchor = {
        "wind_offshore_capacity": "forward_fill + anchor(7774.0)",
        "lignite_capacity": "forward_fill + anchor(20487.0)",
        "hard_coal_capacity": "forward_fill + anchor(23499.0)",
        "pumped_storage_capacity": "forward_fill + anchor(9422.0)",
        "wind_onshore_capacity": "forward_fill + anchor(54666.0)",
        "solar_capacity": "forward_fill + anchor(53538.0)",
        "gas_capacity": "forward_fill + anchor(32038.0)",
    }
    for c, desc in cap_anchor.items():
        if c in df.columns:
            treatment[c] = desc

    for c in ("load_forecast_da", "da_price_d_eur_mwh", "ex_rate_eur_usd"):
        if c in df.columns:
            treatment[c] = "interpolate_small_gaps(max_gap_hours=2)"
    for c in ("wind_onshore_forecast", "solar_forecast"):
        if c in df.columns:
            treatment[c] = "interpolate_small_gaps(max_gap_hours=1) [DST single-hour gaps]"
    if "da_price_BE" in df.columns:
        treatment["da_price_BE"] = "interpolate_small_gaps(max_gap_hours=24)"

    if "generation_nuclear_mw" in df.columns:
        treatment["generation_nuclear_mw"] = "set 0.0 if null and timestamp>=2023-04-15"

    entsoe_id_cols = [
        "wind_onshore_forecast_id_entsoe",
        "wind_offshore_forecast_id_entsoe",
        "solar_forecast_id_entsoe",
    ]
    for c in entsoe_id_cols:
        if c in df.columns:
            treatment[c] = "coalesce(ID, DA) + source tag"

    if "price_intraday_eur" in df.columns and "da_price_d_eur_mwh" in df.columns:
        treatment["price_intraday_eur"] = "coalesce(intraday, day_ahead)"

    for prefix in (
        "afrr_avg_activation_price_",
        "afrr_activation_avg_price_",
        "afrr_activation_price_",
        "mfrr_activation_price_",
        "afrr_capacity_price_",
    ):
        for direction in ("pos", "neg"):
            c = f"{prefix}{direction}"
            if c in df.columns:
                treatment[c] = "if volume==0 -> 0 else interpolate_small_gaps(max_gap_hours=2)"

    for c in (
        "net_import_export_mw",
        "afrr_capacity_offered_mw_pos",
        "afrr_capacity_offered_mw_neg",
        "afrr_activation_offered_mw_pos",
        "afrr_activation_offered_mw_neg",
        "afrr_activated_mw_pos",
        "afrr_activated_mw_neg",
        "mfrr_activated_mw_pos",
        "mfrr_activated_mw_neg",
    ):
        if c in df.columns:
            treatment[c] = "interpolate_small_gaps(max_gap_hours=8) + forward_fill"

    for c in ("afrr_capacity_offered_mw_pos", "afrr_capacity_offered_mw_neg", "afrr_activation_offered_mw_pos", "afrr_activation_offered_mw_neg"):
        if c in df.columns:
            treatment[c] = "fill_null(0.0) + interpolate_small_gaps(max_gap_hours=8) + forward_fill"

    for c in ("afrr_activated_mwh_pos", "afrr_activated_mwh_neg", "mfrr_activated_mwh_pos", "mfrr_activated_mwh_neg"):
        if c in df.columns:
            treatment[c] = "coalesce(MWh, corresponding MW)"

    for c in (
        "wind_forecast_de",
        "wind_onshore_error",
        "wind_offshore_error",
        "solar_error",
        "wind_onshore_error_da",
        "wind_onshore_error_id",
        "wind_onshore_forecast_delta",
        "wind_offshore_error_da",
        "wind_offshore_error_id",
        "wind_offshore_forecast_delta",
        "solar_error_da",
        "solar_error_id",
        "solar_forecast_delta",
        "system_stress_signal",
    ):
        if c in df.columns:
            treatment[c] = "recalculated/coalesced from base columns"

    for c in ("wind_onshore_capacity_entsoe", "wind_offshore_capacity_entsoe", "solar_capacity_entsoe"):
        if c in df.columns:
            treatment[c] = "coalesce(ENTSO-E, SMARD fallback candidates) + source tag"

    for c in ("rz_saldo_mw_qs", "NRV_balance_qs", "rz_saldo_mw_op", "NRV_balance_op"):
        if c in df.columns:
            treatment[c] = "raw provenance column kept (no direct imputation)"

    return treatment


def _log_missing_report(before: dict[str, int], after: dict[str, int], treatment: dict[str, str]) -> None:
    cols = sorted({*before.keys(), *after.keys()})
    report = []
    for c in cols:
        b = before.get(c, 0)
        a = after.get(c, 0)
        is_new_col = c not in before and c in after
        if b > 0 or a > 0 or is_new_col:
            report.append((c, b, a, b - a, treatment.get(c, "no explicit handling rule"), is_new_col))

    if not report:
        LOGGER.info("Missing-value report: no columns with nulls before/after.")
        return

    ordered = sorted(report, key=lambda x: (-x[1], x[2], x[0]))
    col_w = max(len("column"), max(len(c) for c, *_ in ordered))
    num_w = max(
        len("before"),
        len("after"),
        len("reduced"),
        max(len(str(v)) for _, b, a, d, _, _ in ordered for v in (b, a, d)),
    )

    LOGGER.info("Missing-value report:")
    LOGGER.info(
        "  %-*s | %*s | %*s | %*s | %-38s | %s",
        col_w, "column",
        num_w, "before",
        num_w, "after",
        num_w, "reduced",
        "status",
        "treatment",
    )
    LOGGER.info(
        "  %s-+-%s-+-%s-+-%s-+-%s-+-%s",
        "-" * col_w,
        "-" * num_w,
        "-" * num_w,
        "-" * num_w,
        "-" * 38,
        "-" * 40,
    )
    for c, b, a, d, t, is_new_col in ordered:
        if is_new_col:
            status = "new column introduced (not an error)"
        else:
            status = "ok"
        LOGGER.info(
            "  %-*s | %*d | %*d | %*d | %-38s | %s",
            col_w, c,
            num_w, b,
            num_w, a,
            num_w, d,
            status,
            t,
        )


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


def _apply_capacity_fallback(df: pl.DataFrame) -> pl.DataFrame:
    """Fill ENTSO-E capacity gaps from SMARD capacity columns when available."""
    fallback_candidates = {
        "wind_onshore": [
            "wind_onshore_capacity_smard",
            "wind_onshore_capacity",
            "wind_onshore_installed_capacity_mw",
        ],
        "wind_offshore": [
            "wind_offshore_capacity_smard",
            "wind_offshore_capacity",
            "wind_offshore_installed_capacity_mw",
        ],
        "solar": [
            "solar_capacity_smard",
            "solar_capacity",
            "solar_installed_capacity_mw",
        ],
    }

    for tech, candidates in fallback_candidates.items():
        entsoe_col = f"{tech}_capacity_entsoe"
        if entsoe_col not in df.columns:
            continue

        smard_cols = [c for c in candidates if c in df.columns]
        orig_col = f"__{tech}_capacity_entsoe_orig"
        df = df.with_columns(pl.col(entsoe_col).alias(orig_col))
        if smard_cols:
            df = df.with_columns(
                pl.coalesce([pl.col(entsoe_col)] + [pl.col(c) for c in smard_cols]).alias(entsoe_col)
            )
            smard_any = pl.coalesce([pl.col(c) for c in smard_cols]).is_not_null()
            df = df.with_columns(
                pl.when(pl.col(orig_col).is_not_null())
                .then(pl.lit("entsoe"))
                .when(smard_any)
                .then(pl.lit("smard_fallback"))
                .otherwise(pl.lit("missing"))
                .alias(f"{tech}_capacity_source")
            )
        else:
            df = df.with_columns(
                pl.when(pl.col(orig_col).is_null())
                .then(pl.lit("missing"))
                .otherwise(pl.lit("entsoe"))
                .alias(f"{tech}_capacity_source")
            )
        df = df.drop(orig_col)
    return df


def _apply_entsoe_intraday_fallback(df: pl.DataFrame) -> pl.DataFrame:
    """Fallback ENTSO-E ID forecasts to DA where ID is missing."""
    techs = ("wind_onshore", "wind_offshore", "solar")
    for tech in techs:
        id_col = f"{tech}_forecast_id_entsoe"
        da_col = f"{tech}_forecast_da_entsoe"
        source_col = f"{tech}_forecast_id_entsoe_source"
        if id_col in df.columns and da_col in df.columns:
            original = f"__{id_col}_orig"
            df = df.with_columns(pl.col(id_col).alias(original))
            df = df.with_columns(pl.coalesce([pl.col(id_col), pl.col(da_col)]).alias(id_col))
            df = df.with_columns(
                pl.when(pl.col(original).is_not_null())
                .then(pl.lit("id"))
                .when(pl.col(da_col).is_not_null())
                .then(pl.lit("da_fallback"))
                .otherwise(pl.lit("missing"))
                .alias(source_col)
            ).drop(original)
    return df


def _fill_nuclear_after_phaseout(df: pl.DataFrame) -> pl.DataFrame:
    if "generation_nuclear_mw" not in df.columns or "timestamp_utc" not in df.columns:
        return df
    return df.with_columns(
        pl.when(
            pl.col("generation_nuclear_mw").is_null()
            & (pl.col("timestamp_utc").dt.date() >= pl.lit(date(2023, 4, 15)))
        )
        .then(pl.lit(0.0))
        .otherwise(pl.col("generation_nuclear_mw"))
        .alias("generation_nuclear_mw")
    )


def _fill_activation_energy_from_power(df: pl.DataFrame) -> pl.DataFrame:
    pairs = [
        ("afrr_activated_mwh_pos", "afrr_activated_mw_pos"),
        ("afrr_activated_mwh_neg", "afrr_activated_mw_neg"),
        ("mfrr_activated_mwh_pos", "mfrr_activated_mw_pos"),
        ("mfrr_activated_mwh_neg", "mfrr_activated_mw_neg"),
    ]
    exprs = []
    for mwh_col, mw_col in pairs:
        if mwh_col in df.columns and mw_col in df.columns:
            exprs.append(pl.coalesce([pl.col(mwh_col), pl.col(mw_col)]).alias(mwh_col))
    if exprs:
        df = df.with_columns(exprs)
    return df


def _fix_smard_dst_gaps(df: pl.DataFrame) -> pl.DataFrame:
    """Fix known single-hour DST gaps in legacy SMARD forecasts."""
    for col in ("wind_onshore_forecast", "solar_forecast"):
        df = _interpolate_small_gaps(df, col, max_gap_hours=1)
    return df


def _recalculate_error_columns(df: pl.DataFrame) -> pl.DataFrame:
    # Legacy aggregate forecast: fill/create from components.
    if {"wind_onshore_forecast", "wind_offshore_forecast"}.issubset(df.columns):
        computed = pl.col("wind_onshore_forecast") + pl.col("wind_offshore_forecast")
        if "wind_forecast_de" in df.columns:
            df = df.with_columns(pl.coalesce([pl.col("wind_forecast_de"), computed]).alias("wind_forecast_de"))
        else:
            df = df.with_columns(computed.alias("wind_forecast_de"))

    # Legacy SMARD-style errors: fill only where missing.
    legacy = [
        ("wind_onshore_error", "wind_onshore_actual", "wind_onshore_forecast"),
        ("wind_offshore_error", "wind_offshore_actual", "wind_offshore_forecast"),
        ("solar_error", "solar_actual", "solar_forecast"),
    ]
    for err_col, actual_col, fc_col in legacy:
        if {actual_col, fc_col}.issubset(df.columns):
            computed = pl.col(actual_col) - pl.col(fc_col)
            if err_col in df.columns:
                df = df.with_columns(pl.coalesce([pl.col(err_col), computed]).alias(err_col))
            else:
                df = df.with_columns(computed.alias(err_col))

    # ENTSO-E errors and deltas.
    for tech in ("wind_onshore", "wind_offshore", "solar"):
        actual = f"{tech}_actual_entsoe"
        da = f"{tech}_forecast_da_entsoe"
        intraday = f"{tech}_forecast_id_entsoe"
        err_da = f"{tech}_error_da"
        err_id = f"{tech}_error_id"
        delta = f"{tech}_forecast_delta"
        if {actual, da}.issubset(df.columns):
            computed = pl.col(actual) - pl.col(da)
            if err_da in df.columns:
                df = df.with_columns(pl.coalesce([pl.col(err_da), computed]).alias(err_da))
            else:
                df = df.with_columns(computed.alias(err_da))
        if {actual, intraday}.issubset(df.columns):
            computed = pl.col(actual) - pl.col(intraday)
            if err_id in df.columns:
                df = df.with_columns(pl.coalesce([pl.col(err_id), computed]).alias(err_id))
            else:
                df = df.with_columns(computed.alias(err_id))
        if {intraday, da}.issubset(df.columns):
            computed = pl.col(intraday) - pl.col(da)
            if delta in df.columns:
                df = df.with_columns(pl.coalesce([pl.col(delta), computed]).alias(delta))
            else:
                df = df.with_columns(computed.alias(delta))

    id_error_cols = [c for c in ("wind_onshore_error_id", "wind_offshore_error_id", "solar_error_id") if c in df.columns]
    if id_error_cols and "system_stress_signal" in df.columns:
        sum_expr = pl.sum_horizontal([pl.col(c) for c in id_error_cols])
        any_expr = pl.any_horizontal([pl.col(c).is_not_null() for c in id_error_cols])
        df = df.with_columns(
            pl.coalesce(
                [pl.col("system_stress_signal"), pl.when(any_expr).then(sum_expr).otherwise(None)]
            ).alias("system_stress_signal")
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
    missing_before = _null_count_dict(df)
    treatment_map = _build_treatment_map(df)

    # 1) Commodity prices: weekend carry-forward (no interpolation to avoid leakage).
    commodity_cols = ["co2_price_eua", "gas_price_ttf", "coal_price_api2", "coal_price_api"]
    _log_null_counts(df, commodity_cols, "Before commodity ffill")
    df = _ffill_cols(df, commodity_cols)
    _log_null_counts(df, commodity_cols, "After commodity ffill")

    # 1b) Installed generation capacities from SMARD CSV (safe anchor fill, no backward fill).
    df = _clean_and_anchor_smard_capacity(df)

    # 2) Physics/grid small gaps (<= 2 hours)
    for col in ["load_forecast_da", "da_price_d_eur_mwh", "ex_rate_eur_usd"]:
        df = _interpolate_small_gaps(df, col, max_gap_hours=2)

    # 2b) Belgian day-ahead price: allow short linear interpolation (<24h).
    df = _interpolate_small_gaps(df, "da_price_BE", max_gap_hours=24)

    # 2c) Nuclear post phase-out: missing means no generation.
    df = _fill_nuclear_after_phaseout(df)
    # 2d) Legacy SMARD forecast single-hour DST gaps.
    df = _fix_smard_dst_gaps(df)

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
    df = _apply_capacity_fallback(df)
    df = _apply_entsoe_intraday_fallback(df)

    # 4) Intraday price: fill with day-ahead if missing
    if "price_intraday_eur" in df.columns and "da_price_d_eur_mwh" in df.columns:
        df = df.with_columns(
            pl.coalesce([pl.col("price_intraday_eur"), pl.col("da_price_d_eur_mwh")]).alias("price_intraday_eur")
        )

    # 5) Regelleistung offered MW: missing means zero offer in published aggregates.
    for col in ("afrr_capacity_offered_mw_pos", "afrr_capacity_offered_mw_neg", "afrr_activation_offered_mw_pos", "afrr_activation_offered_mw_neg"):
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null(0.0).alias(col))

    # 5b) Regelleistung prices: close isolated 1h holes (moved from fetch_regelleistung.py).
    for col in (
        "afrr_avg_activation_price_pos",
        "afrr_avg_activation_price_neg",
        "afrr_activation_avg_price_pos",
        "afrr_activation_avg_price_neg",
        "afrr_activation_price_pos",
        "afrr_activation_price_neg",
        "mfrr_activation_price_pos",
        "mfrr_activation_price_neg",
        "afrr_capacity_price_pos",
        "afrr_capacity_price_neg",
    ):
        df = _interpolate_small_gaps(df, col, max_gap_hours=1)

    # 5c) Regelleistung prices: interpolate if volume != 0 and price is null
    price_specs = [
        ("afrr_avg_activation_price_", "afrr_activated_mw_"),
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

    # 5d) Regelleistung/activation volume-like signals: short forward-fill.
    volume_ffill_cols = [
        "net_import_export_mw",
        "afrr_capacity_offered_mw_pos",
        "afrr_capacity_offered_mw_neg",
        "afrr_activation_offered_mw_pos",
        "afrr_activation_offered_mw_neg",
        "afrr_activated_mw_pos",
        "afrr_activated_mw_neg",
        "mfrr_activated_mw_pos",
        "mfrr_activated_mw_neg",
    ]
    for col in volume_ffill_cols:
        if col in df.columns:
            df = _interpolate_small_gaps(df, col, max_gap_hours=8)
            df = df.with_columns(pl.col(col).fill_null(strategy="forward").alias(col))

    df = _fill_activation_energy_from_power(df)

    # 6) Recalculate derived error/signal columns after filling base signals.
    df = _recalculate_error_columns(df)
    missing_after = _null_count_dict(df)
    _log_missing_report(missing_before, missing_after, treatment_map)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")
    LOGGER.info("Wrote %s rows to %s", df.height, output_path)


if __name__ == "__main__":
    main()
