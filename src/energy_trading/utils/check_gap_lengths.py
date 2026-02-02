"""Analyze missing-value gap lengths and propose smart imputation for time-series columns."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl
import matplotlib.pyplot as plt


def _gap_lengths(series: pl.Series) -> list[int]:
    """Return lengths of consecutive null runs (in rows) for a series."""
    is_null = series.is_null()
    if is_null.sum() == 0:
        return []

    df = pl.DataFrame({"is_null": is_null})
    # Identify start of each null run and assign a run id.
    df = df.with_columns(
        (
            (pl.col("is_null") & ~pl.col("is_null").shift(1).fill_null(False))
            .cast(pl.Int64)
            .cum_sum()
            .alias("run_id")
        )
    )
    runs = (
        df.filter(pl.col("is_null"))
        .group_by("run_id")
        .len()
        .select(pl.col("len").alias("gap_len"))
        .sort("gap_len")
    )
    return runs["gap_len"].to_list()


def analyze_gaps(df: pl.DataFrame, columns: Iterable[str]) -> None:
    """Print gap summaries for selected columns and alert on long gaps."""
    for col in columns:
        if col not in df.columns:
            print(f"[WARN] Column not found: {col}")
            continue
        gaps = _gap_lengths(df[col])
        total_missing = df[col].null_count()
        num_gaps = len(gaps)
        max_gap = max(gaps) if gaps else 0
        gt_2 = sum(1 for g in gaps if g > 2)
        gt_6 = sum(1 for g in gaps if g > 6)
        gt_24 = sum(1 for g in gaps if g > 24)
        print(
            f"{col}: missing={total_missing}, gaps={num_gaps}, "
            f"max_gap={max_gap}h, >2h={gt_2}, >6h={gt_6}, >24h={gt_24}"
        )

        if "timestamp_utc" not in df.columns:
            print(f"[WARN] timestamp_utc not found; cannot locate gap timestamps for {col}.")
            continue

        # Build gap metadata to locate long gaps in time.
        is_null = df[col].is_null()
        run_id = (
            (is_null & ~is_null.shift(1).fill_null(False))
            .cast(pl.Int64)
            .cum_sum()
        )
        gaps_df = (
            df.select(
                [
                    pl.col("timestamp_utc"),
                    is_null.alias("is_null"),
                    run_id.alias("run_id"),
                ]
            )
            .filter(pl.col("is_null"))
            .group_by("run_id")
            .agg(
                [
                    pl.len().alias("gap_len"),
                    pl.col("timestamp_utc").min().alias("gap_start_null"),
                    pl.col("timestamp_utc").max().alias("gap_end_null"),
                ]
            )
            .sort("gap_start_null")
        )

        if gaps_df.is_empty():
            continue

        # Estimate gap bounds as last valid before null and first valid after null.
        # If not available, fall back to the null range endpoints.
        for row in gaps_df.iter_rows(named=True):
            if row["gap_len"] <= 12:
                continue
            start_null = row["gap_start_null"]
            end_null = row["gap_end_null"]
            prev_valid = (
                df.filter(pl.col("timestamp_utc") < start_null)
                .filter(pl.col(col).is_not_null())
                .select(pl.col("timestamp_utc").max())
                .item()
            )
            next_valid = (
                df.filter(pl.col("timestamp_utc") > end_null)
                .filter(pl.col(col).is_not_null())
                .select(pl.col("timestamp_utc").min())
                .item()
            )
            gap_start = prev_valid if prev_valid is not None else start_null
            gap_end = next_valid if next_valid is not None else end_null
            print(
                f"[ALERT] Column '{col}': Gap found from {gap_start} to {gap_end} "
                f"({row['gap_len']} hours)"
            )
            if row["gap_len"] > 24:
                print("Gap > 24h: Do NOT interpolate. Check raw data source.")


def _plot_gap_histograms(df: pl.DataFrame, columns: Iterable[str]) -> None:
    """Plot histogram of gap lengths for selected columns."""
    for col in columns:
        if col not in df.columns:
            continue
        gaps = _gap_lengths(df[col])
        if not gaps:
            print(f"[INFO] No gaps for {col}, skipping plot.")
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(gaps, bins=30)
        plt.title(f"Gap length histogram: {col}")
        plt.xlabel("Gap length (hours)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


def smart_impute(
    df: pl.DataFrame,
    col_name: str,
    fallback_col_name: str | None = None,
    limit: int = 3,
) -> pl.DataFrame:
    """
    Impute missing values using a limited linear interpolation and optional fallback.

    Step 1: linearly interpolate only within gaps <= limit.
    Step 2: if fallback_col_name is provided, fill remaining nulls from fallback.
    """
    if col_name not in df.columns:
        raise KeyError(f"Missing column: {col_name}")
    if fallback_col_name is not None and fallback_col_name not in df.columns:
        raise KeyError(f"Missing fallback column: {fallback_col_name}")

    s = df[col_name]
    is_null = s.is_null()
    # Build run ids for null stretches.
    run_id = (
        (is_null & ~is_null.shift(1).fill_null(False))
        .cast(pl.Int64)
        .cum_sum()
    )
    # Compute gap length per run_id.
    gap_len = (
        pl.when(is_null)
        .then(run_id)
        .otherwise(None)
        .alias("run_id")
    )
    tmp = df.select([gap_len]).with_columns(pl.col("run_id").is_not_null().alias("is_null"))
    gap_sizes = (
        tmp.filter(pl.col("is_null"))
        .group_by("run_id")
        .len()
        .rename({"len": "gap_len"})
    )
    # Map each row to its gap length (nulls only).
    gap_len_map = tmp.join(gap_sizes, on="run_id", how="left")["gap_len"]

    # Polars interpolate fills all gaps; mask out long gaps afterward.
    interpolated = s.interpolate()
    keep_interp = is_null & (gap_len_map <= limit)
    imputed = pl.when(keep_interp).then(interpolated).otherwise(s)

    if fallback_col_name:
        imputed = pl.when(imputed.is_null()).then(pl.col(fallback_col_name)).otherwise(imputed)

    return df.with_columns(imputed.alias(col_name))


def main() -> None:
    data_path = Path("data/processed/all_data.parquet")
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    df = pl.read_parquet(data_path)

    wind_intraday_cols = [
        "total_wind_intraday_error",
        "wind_onshore_intraday_error",
    ]

    analyze_gaps(df, wind_intraday_cols)
    _plot_gap_histograms(df, wind_intraday_cols)

    # Demonstrate smart imputation on a forecast column.
    if "total_wind_intraday_forecast" in df.columns:
        fallback = "wind_forecast_de" if "wind_forecast_de" in df.columns else None
        df = smart_impute(
            df,
            "total_wind_intraday_forecast",
            fallback_col_name=fallback,
            limit=3,
        )
        print("Applied smart_impute to total_wind_intraday_forecast")
    else:
        print("[WARN] total_wind_intraday_forecast not found; skipping imputation demo.")


if __name__ == "__main__":
    main()
