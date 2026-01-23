"""
Run the script with: python3 -m energy_trading.features.clean_data --data-dir data --out data/all_merged_clean.parquet
Prerequisite: data/all_merged_clean.parquet has been generated with energy_trading.ingestion.merge_data.py
"""

"""Merge and clean all parquet datasets.

Steps:
- merge all parquets in a data directory on `timestamp` (or load a pre-merged file)
- standardize timestamp columns
- drop fully-null columns
- deduplicate timestamps (keep last)
- forward/back fill tiny gaps (<=1% nulls) in numeric columns
- clip numeric outliers at 1st/99th percentile
- write cleaned parquet and print a short audit log
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import polars as pl

from energy_trading.ingestion.merge_data import merge_all


def _coerce_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure a unified timestamp column."""
    if "timestamp" in df.columns:
        ts_col = "timestamp"
    elif "timestamp_utc" in df.columns:
        df = df.rename({"timestamp_utc": "timestamp"})
        ts_col = "timestamp"
    else:
        raise ValueError("No timestamp column found.")

    if df[ts_col].dtype == pl.Int64:
        df = df.with_columns(pl.from_epoch(ts_col, time_unit="ms").alias(ts_col))
    return df


def _clip_outliers(df: pl.DataFrame, q_low: float = 0.01, q_high: float = 0.99) -> Tuple[pl.DataFrame, Dict[str, Tuple[float, float]]]:
    numeric_cols = [c for c, t in zip(df.columns, df.dtypes) if t.is_numeric()]
    bounds: Dict[str, Tuple[float, float]] = {}
    if not numeric_cols:
        return df, bounds

    # Compute quantiles once.
    quantile_exprs = []
    for c in numeric_cols:
        quantile_exprs.append(pl.col(c).quantile(q_low).alias(f"{c}_q_low"))
        quantile_exprs.append(pl.col(c).quantile(q_high).alias(f"{c}_q_high"))
    qs = df.select(quantile_exprs).to_dicts()[0]

    clips = []
    for c in numeric_cols:
        low = qs.get(f"{c}_q_low")
        high = qs.get(f"{c}_q_high")
        if low is None or high is None:
            continue
        bounds[c] = (low, high)
        clips.append(pl.col(c).clip(lower=low, upper=high).alias(c))

    if clips:
        df = df.with_columns(clips)
    return df, bounds


def clean_dataframe(df: pl.DataFrame, gap_threshold: float = 0.01, clip_outliers: bool = True) -> Tuple[pl.DataFrame, Dict[str, object]]:
    """Apply cleaning rules and return dataframe plus an audit dict."""
    audit: Dict[str, object] = {}
    df = _coerce_timestamp(df)

    audit["rows_before"] = df.height
    audit["cols_before"] = len(df.columns)

    # Drop fully-null columns.
    drop_cols = [c for c in df.columns if df[c].null_count() == df.height]
    if drop_cols:
        df = df.drop(drop_cols)
    audit["dropped_null_columns"] = drop_cols

    # Deduplicate timestamps.
    if "timestamp" in df.columns:
        rows_before = df.height
        df = df.unique(subset=["timestamp"], keep="last")
        audit["duplicates_removed"] = rows_before - df.height

    # Fill small gaps.
    fills: List[Tuple[str, int]] = []
    numeric_cols = [c for c, t in zip(df.columns, df.dtypes) if t.is_numeric()]
    for c in numeric_cols:
        nulls = df[c].null_count()
        if nulls == 0:
            continue
        if df.height == 0:
            continue
        if (nulls / df.height) <= gap_threshold and df[c].drop_nulls().height > 1:
            df = df.with_columns(
                pl.col(c)
                .forward_fill()
                .backward_fill()
                .alias(c)
            )
            fills.append((c, nulls))
    audit["filled_columns"] = fills

    # Clip outliers.
    bounds = {}
    if clip_outliers:
        df, bounds = _clip_outliers(df)
    audit["clip_bounds"] = bounds

    audit["rows_after"] = df.height
    audit["cols_after"] = len(df.columns)
    return df.sort("timestamp"), audit


def load_or_merge(data_dir: Path, merged_path: Path | None) -> pl.DataFrame:
    if merged_path:
        return pl.read_parquet(merged_path)
    parquet_paths = sorted(data_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    return merge_all(parquet_paths)


def main():
    parser = argparse.ArgumentParser(description="Merge and clean all parquet files on timestamp.")
    parser.add_argument("--data-dir", default="data", help="Directory containing parquet files (default: data).")
    parser.add_argument("--merged", default=None, help="Optional existing merged parquet to clean instead of merging.")
    parser.add_argument("--out", default="data/all_merged_clean.parquet", help="Output parquet path.")
    parser.add_argument("--gap-threshold", type=float, default=0.01, help="Max null ratio to auto-fill numeric gaps (default 1%).")
    parser.add_argument("--no-clip", action="store_true", help="Disable outlier clipping.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    merged_path = Path(args.merged) if args.merged else None
    df = load_or_merge(data_dir, merged_path)

    cleaned, audit = clean_dataframe(df, gap_threshold=args.gap_threshold, clip_outliers=not args.no_clip)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.write_parquet(out_path, compression="zstd")

    print(f"Wrote {cleaned.height} rows to {out_path}")
    print("Audit:")
    print(f" - rows before/after: {audit.get('rows_before')} -> {audit.get('rows_after')}")
    print(f" - cols before/after: {audit.get('cols_before')} -> {audit.get('cols_after')}")
    if audit.get("dropped_null_columns"):
        print(f\" - dropped null columns: {', '.join(audit['dropped_null_columns'])}\")
    if audit.get("duplicates_removed"):
        print(f\" - duplicates removed: {audit['duplicates_removed']}\")
    if audit.get("filled_columns"):
        fills = ', '.join(f\"{c} ({n} gaps)\" for c, n in audit['filled_columns'])
        print(f\" - filled small gaps: {fills}\")
    if audit.get("clip_bounds"):
        clipped = ', '.join(audit["clip_bounds"].keys())
        if clipped:
            print(f\" - clipped outliers for: {clipped}\")


if __name__ == "__main__":
    main()
