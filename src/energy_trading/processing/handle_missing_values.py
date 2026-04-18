"""Handle missing values for refined market data.

Usage:
    ./.venv/bin/python -m energy_trading.processing.handle_missing_values \
        --in data/processed/all_data_refined.parquet \
        --out data/processed/cleaned_data.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from energy_trading.constants import PICASSO_RELEASE_UTC

LOGGER = logging.getLogger(__name__)


def _existing(df: pl.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _nan_runs(mask: pd.Series) -> list[tuple[int, int]]:
    arr = mask.to_numpy(dtype=bool)
    idx = np.flatnonzero(arr)
    if idx.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i == prev + 1:
            prev = i
        else:
            runs.append((start, prev))
            start = i
            prev = i
    runs.append((start, prev))
    return runs


def _impute_physical_actuals(series: pd.Series) -> tuple[pd.Series, int]:
    """Category A: physical actuals with short-gap interpolation and seasonal fill."""
    s = pd.to_numeric(series, errors="coerce").copy()
    before = int(s.isna().sum())
    runs = _nan_runs(s.isna())
    if not runs:
        return s, 0

    interp_short = s.interpolate(method="linear", limit=3, limit_direction="both")
    lag24 = s.shift(24)
    lag168 = s.shift(168)

    for start, end in runs:
        gap_len = end - start + 1
        sl = slice(start, end + 1)
        if gap_len <= 3:
            s.iloc[sl] = interp_short.iloc[sl]
        else:
            fill_vals = lag24.iloc[sl].copy()
            missing = fill_vals.isna()
            if missing.any():
                fill_vals.loc[missing] = lag168.iloc[sl][missing]
            s.iloc[sl] = fill_vals

    after = int(s.isna().sum())
    return s, max(0, before - after)


def _impute_market_prices(series: pd.Series) -> tuple[pd.Series, int]:
    """Category B: market prices and continuous indicators.

    - ffill with limit=4h
    """
    s = pd.to_numeric(series, errors="coerce").copy()
    before = int(s.isna().sum())
    s_out = s.ffill(limit=4)
    after = int(s_out.isna().sum())
    return s_out, max(0, before - after)


def _impute_forecasts(series: pd.Series) -> tuple[pd.Series, int]:
    """Category C: forecasts stay valid until next publication -> pure ffill."""
    s = pd.to_numeric(series, errors="coerce").copy()
    before = int(s.isna().sum())
    s_out = s.ffill()
    after = int(s_out.isna().sum())
    return s_out, max(0, before - after)


def _impute_co2_price(series: pd.Series) -> tuple[pd.Series, int]:
    """Special policy for CO2 commodity price.

    Use unlimited forward-fill to keep the latest known traded quote available.
    Then backward-fill only the remaining leading gap to remove startup NaNs.
    """
    s = pd.to_numeric(series, errors="coerce").copy()
    before = int(s.isna().sum())
    s_out = s.ffill()
    if s_out.isna().any():
        # After unlimited ffill, any remaining NaNs are at the series start.
        # Fill that leading block from the first observed quote.
        s_out = s_out.bfill()
    after = int(s_out.isna().sum())
    return s_out, max(0, before - after)


def _impute_unlimited_ffill(series: pd.Series) -> tuple[pd.Series, int]:
    """Generic causal imputation via unlimited forward fill."""
    s = pd.to_numeric(series, errors="coerce").copy()
    before = int(s.isna().sum())
    s_out = s.ffill()
    after = int(s_out.isna().sum())
    return s_out, max(0, before - after)


def _impute_prev_day_same_hour(series: pd.Series) -> tuple[pd.Series, int]:
    """Fill NaNs from the same UTC hour on the previous day (t-24h)."""
    s = pd.to_numeric(series, errors="coerce").copy()
    before = int(s.isna().sum())
    s_out = s.copy()
    mask = s_out.isna()
    if mask.any():
        s_out.loc[mask] = s_out.shift(24).loc[mask]
    after = int(s_out.isna().sum())
    return s_out, max(0, before - after)


def _strip_warmup_rows(pdf: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Strip leading warmup rows caused by lag/rolling feature construction."""
    warmup_patterns = (
        "lag",
        "rolling",
        "_mean_",
        "_std_",
        "ewma",
        "zscore",
        "_diff",
        "target_pos_h",
        "target_neg_h",
    )
    warmup_cols = [c for c in pdf.columns if any(p in c for p in warmup_patterns)]
    if not warmup_cols:
        return pdf, 0

    starts: list[int] = []
    for c in warmup_cols:
        nonnull = pdf[c].notna().to_numpy()
        if nonnull.any():
            starts.append(int(np.argmax(nonnull)))
    if not starts:
        return pdf, 0

    cut = max(starts)
    if cut <= 0:
        return pdf, 0
    return pdf.iloc[cut:].reset_index(drop=True), cut


def _categorize_columns(df: pl.DataFrame) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    cols = list(df.columns)

    target_cols = [c for c in cols if c.startswith("y_true") or c.startswith("y_train") or c.startswith("target_")]

    forecast_cols = [
        c for c in cols
        if ("forecast" in c.lower()) and c not in target_cols
    ]

    physical_cols = [
        c
        for c in cols
        if (
            ("_actual_entsoe" in c)
            or c in {"load_actual", "load_actual_entsoe", "residual_load_calc", "residual_load_actual"}
            or c.startswith("generation_")
        )
        and c not in target_cols
    ]

    rare_event_cols = [
        c
        for c in cols
        if (
            ("outage" in c.lower())
            or ("holiday" in c.lower())
            or c.startswith("is_")
        )
        and c not in target_cols
    ]

    market_cols = [
        c
        for c in cols
        if (
            ("price" in c.lower())
            or ("vwap" in c.lower())
            or ("spread" in c.lower())
            or ("volatility" in c.lower())
            or ("competitiveness" in c.lower())
        )
        and c not in target_cols
        and c not in forecast_cols
    ]

    # Remove overlaps by precedence: targets > forecasts > physical > rare > market.
    used = set(target_cols)
    forecast_cols = [c for c in forecast_cols if c not in used]
    used.update(forecast_cols)
    physical_cols = [c for c in physical_cols if c not in used]
    used.update(physical_cols)
    rare_event_cols = [c for c in rare_event_cols if c not in used]
    used.update(rare_event_cols)
    market_cols = [c for c in market_cols if c not in used]

    return physical_cols, market_cols, forecast_cols, rare_event_cols, target_cols


def clean(df: pl.DataFrame) -> pl.DataFrame:
    if "timestamp_utc" not in df.columns:
        raise ValueError("Missing required column: timestamp_utc")

    cutoff = pl.lit(PICASSO_RELEASE_UTC).str.to_datetime(time_zone="UTC", strict=False)

    # Structural-break columns: set to zero before platform launch.
    platform_cols = _existing(
        df,
        [
            "afrr_picasso_mw_pos",
            "afrr_picasso_mw_neg",
            "afrr_picasso_net_mw",
            "mfrr_mari_mw_pos",
            "mfrr_mari_mw_neg",
            "mfrr_mari_net_mw",
        ],
    )
    if platform_cols:
        df = df.with_columns(
            [
                pl.when(pl.col("timestamp_utc") < cutoff)
                .then(pl.lit(0.0))
                .otherwise(pl.col(c))
                .cast(pl.Float64, strict=False)
                .alias(c)
                for c in platform_cols
            ]
        )
    LOGGER.info("Applied structural-break zeroing to columns: %s", platform_cols)

    # Binary regime feature.
    df = df.with_columns(
        pl.when(pl.col("timestamp_utc") >= cutoff)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int8)
        .alias("is_picasso_active")
    )

    # Category-aware imputation policy (A-E).
    physical_cols, market_cols, forecast_cols, rare_event_cols, target_cols = _categorize_columns(df)
    LOGGER.info(
        "Imputation categories: A(physical)=%s, B(market)=%s, C(forecast)=%s, D(rare/binary)=%s, E(targets)=%s",
        len(physical_cols),
        len(market_cols),
        len(forecast_cols),
        len(rare_event_cols),
        len(target_cols),
    )

    pdf = df.to_pandas()
    pdf["timestamp_utc"] = pd.to_datetime(pdf["timestamp_utc"], utc=True, errors="coerce")
    pdf = pdf.sort_values("timestamp_utc").reset_index(drop=True)

    # Warmup handling: strip leading lag/rolling-induced NaN rows instead of imputing.
    pdf, warmup_rows_removed = _strip_warmup_rows(pdf)
    if warmup_rows_removed > 0:
        LOGGER.info("Warmup handling removed %s leading rows.", warmup_rows_removed)

    imputation_counts: dict[str, int] = {}

    # Category A
    for c in physical_cols:
        if c not in pdf.columns:
            continue
        s, n = _impute_physical_actuals(pdf[c])
        pdf[c] = s
        imputation_counts[c] = n

    # Category B
    for c in market_cols:
        if c not in pdf.columns:
            continue
        s, n = _impute_market_prices(pdf[c])
        pdf[c] = s
        imputation_counts[c] = n

    # Category C
    for c in forecast_cols:
        if c not in pdf.columns:
            continue
        s, n = _impute_forecasts(pdf[c])
        pdf[c] = s
        imputation_counts[c] = n

    # Commodity exceptions: use unlimited carry-forward and then close any
    # remaining leading startup gaps via backward fill.
    for commodity_col in ("co2_price", "gas_price", "coal_price"):
        if commodity_col not in pdf.columns:
            continue
        s, n = _impute_co2_price(pdf[commodity_col])
        pdf[commodity_col] = s
        # Merge with prior category-B count if present.
        imputation_counts[commodity_col] = int(imputation_counts.get(commodity_col, 0) + n)
        remaining = int(pd.to_numeric(pdf[commodity_col], errors="coerce").isna().sum())
        LOGGER.info(
            "[impute][special] %s: ffill_plus_leading_bfill_added=%s remaining_after_special=%s",
            commodity_col,
            n,
            remaining,
        )

    # Capacity structural features: unlimited carry-forward is causal and stable.
    capacity_cols = [c for c in pdf.columns if c.lower().endswith("_capacity")]
    for c in capacity_cols:
        s, n = _impute_unlimited_ffill(pdf[c])
        pdf[c] = s
        imputation_counts[c] = int(imputation_counts.get(c, 0) + n)
        remaining = int(pd.to_numeric(pdf[c], errors="coerce").isna().sum())
        if n > 0:
            LOGGER.info(
                "[impute][special] %s: unlimited_ffill_added=%s remaining_after_special=%s",
                c,
                n,
                remaining,
            )

    # Cross-border DA fallback: use previous-day same-hour value for small/isolated gaps.
    if "da_price_BE" in pdf.columns:
        s, n = _impute_prev_day_same_hour(pdf["da_price_BE"])
        pdf["da_price_BE"] = s
        imputation_counts["da_price_BE"] = int(imputation_counts.get("da_price_BE", 0) + n)
        remaining = int(pd.to_numeric(pdf["da_price_BE"], errors="coerce").isna().sum())
        LOGGER.info(
            "[impute][special] da_price_BE: prev_day_same_hour_added=%s remaining_after_special=%s",
            n,
            remaining,
        )

    # Category D
    for c in rare_event_cols:
        if c not in pdf.columns:
            continue
        before = int(pd.to_numeric(pdf[c], errors="coerce").isna().sum())
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce").fillna(0.0)
        after = int(pd.to_numeric(pdf[c], errors="coerce").isna().sum())
        imputation_counts[c] = max(0, before - after)

    # Category E: never impute targets; drop rows with missing targets.
    if target_cols:
        before_rows = len(pdf)
        missing_target_mask = pd.Series(False, index=pdf.index)
        for c in target_cols:
            missing_target_mask |= pd.to_numeric(pdf[c], errors="coerce").isna()
        dropped = int(missing_target_mask.sum())
        if dropped > 0:
            pdf = pdf.loc[~missing_target_mask].reset_index(drop=True)
        LOGGER.info("Dropped %s rows with missing target values (Category E policy).", dropped)
        LOGGER.info("Rows before target-drop=%s, after=%s", before_rows, len(pdf))

    # Log per-column imputation counts.
    for col, n in sorted(imputation_counts.items(), key=lambda x: (-x[1], x[0])):
        if n > 0:
            LOGGER.info("Imputed %s values in column `%s`.", n, col)

    # Final warmup guard (idempotent): if upstream changes introduce additional
    # leading lag-window NaNs, strip them instead of filling with synthetic values.
    pdf, final_warmup_rows_removed = _strip_warmup_rows(pdf)
    if final_warmup_rows_removed > 0:
        LOGGER.info(
            "Final warmup handling removed %s additional leading rows.",
            final_warmup_rows_removed,
        )

    out = pl.from_pandas(pdf).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)
    ).sort("timestamp_utc")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Handle missing values in refined market data.")
    parser.add_argument(
        "--in",
        dest="input_path",
        default="data/processed/all_data_refined.parquet",
        help="Input refined parquet path.",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default="data/processed/cleaned_data.parquet",
        help="Output cleaned parquet path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input parquet: {input_path}")

    df = pl.read_parquet(input_path).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)
    ).sort("timestamp_utc")
    LOGGER.info("Loaded %s rows and %s columns from %s", df.height, len(df.columns), input_path)

    cleaned = clean(df)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.write_parquet(output_path, compression="zstd")
    LOGGER.info("Wrote %s rows and %s columns to %s", cleaned.height, len(cleaned.columns), output_path)


if __name__ == "__main__":
    main()
