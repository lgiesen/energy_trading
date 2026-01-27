"""Merge all parquet files in the data directory on timestamp_utc into one table.

Usage:
    python -m ingestion.merge_data --data-dir data --out data/all_merged.parquet

Notes:
    - Prefers timestamp_utc if available.
    - Drops other timestamp columns to avoid duplicate/suffixed fields.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import polars as pl


def _load_and_normalize(path: Path, resample_freq: str | None) -> pl.DataFrame | None:
    """Load a parquet and normalize its timestamp column name and type."""
    df = pl.read_parquet(path)
    if "timestamp_utc" in df.columns:
        ts_col = "timestamp_utc"
    elif "timestamp" in df.columns:
        ts_col = "timestamp"
    elif "timestamp_cet" in df.columns:
        ts_col = "timestamp_cet"
    else:
        return None  # Skip files without a timestamp column.

    # Cast timestamp to datetime[us, UTC] to avoid join mismatches.
    ts_dtype = df[ts_col].dtype
    if ts_dtype == pl.Int64:
        df = df.with_columns(pl.from_epoch(ts_col, time_unit="ms").alias(ts_col))
    elif isinstance(ts_dtype, pl.datatypes.Datetime):
        # Align units to microseconds for consistency.
        df = df.with_columns(pl.col(ts_col).dt.cast_time_unit("us").alias(ts_col))
    # Normalize timezone to UTC.
    if df[ts_col].dtype == pl.Datetime:  # type: ignore[attr-defined]
        tz = df[ts_col].dtype.time_zone  # type: ignore[attr-defined]
        if tz is None:
            # If timestamp_cet is naive, assume Europe/Berlin.
            if ts_col == "timestamp_cet":
                df = df.with_columns(pl.col(ts_col).dt.replace_time_zone("Europe/Berlin").alias(ts_col))
                df = df.with_columns(pl.col(ts_col).dt.convert_time_zone("UTC").alias(ts_col))
            else:
                df = df.with_columns(pl.col(ts_col).dt.replace_time_zone("UTC").alias(ts_col))
        elif tz != "UTC":
            df = df.with_columns(pl.col(ts_col).dt.convert_time_zone("UTC").alias(ts_col))

    # Rename to canonical and drop other timestamp columns.
    if ts_col != "timestamp":
        if "timestamp" in df.columns:
            df = df.drop("timestamp")
        df = df.rename({ts_col: "timestamp"})
    drop_cols = [c for c in ("timestamp_utc", "timestamp_cet") if c in df.columns and c != "timestamp"]
    if drop_cols:
        df = df.drop(drop_cols)
    if resample_freq:
        df = _resample_to_freq(df, resample_freq)
    return df


def _resample_to_freq(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    """Downsample to a fixed frequency using mean for numeric columns and last for others."""
    if "timestamp" not in df.columns:
        return df
    df = df.with_columns(pl.col("timestamp").dt.truncate(freq).alias("timestamp"))
    num_cols = [c for c in df.columns if c != "timestamp" and df[c].dtype.is_numeric()]
    other_cols = [c for c in df.columns if c not in num_cols and c != "timestamp"]
    aggs = []
    if num_cols:
        aggs.append(pl.col(num_cols).mean())
    if other_cols:
        aggs.append(pl.col(other_cols).last())
    if not aggs:
        return df.unique(subset=["timestamp"]).sort("timestamp")
    return df.group_by("timestamp").agg(aggs).sort("timestamp")


def merge_all(parquet_paths: Iterable[Path], resample_freq: str | None) -> pl.DataFrame:
    """Full-join all provided parquet files on timestamp."""
    merged: pl.DataFrame | None = None
    for path in parquet_paths:
        df = _load_and_normalize(path, resample_freq)
        if df is None:
            print(f"Skipping {path.name}: no timestamp column.")
            continue

        if merged is None:
            merged = df
            print(f"Seeded merge with {path.name} ({len(df)} rows).")
            continue

        # Avoid column name collisions (other than timestamp) by suffixing with file stem.
        overlap = (set(df.columns) & set(merged.columns)) - {"timestamp"}
        if overlap:
            df = df.rename({col: f"{col}_{path.stem}" for col in overlap})

        merged = merged.join(df, on="timestamp", how="full", suffix=f"_{path.stem}")
        # Older Polars may emit a timestamp_right column; drop it if present.
        if "timestamp_right" in merged.columns:
            merged = merged.drop("timestamp_right")
        print(f"Merged {path.name} -> {len(merged)} rows.")

    if merged is None:
        raise RuntimeError("No parquet files with a timestamp column were merged.")

    return merged.sort("timestamp")


def main():
    parser = argparse.ArgumentParser(description="Merge all parquet files in the data directory on timestamp.")
    parser.add_argument("--data-dir", default="data", help="Directory containing parquet files to merge (default: data).")
    parser.add_argument("--out", default="data/all_merged.parquet", help="Output parquet path.")
    parser.add_argument(
        "--resample-freq",
        default="1h",
        help="Optional resample frequency before merge (default: 1h). Use '' to disable.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    parquet_paths: List[Path] = sorted(
        p for p in data_dir.glob("*.parquet")
        if p.name not in {"all_merged.parquet", "all_data.parquet"}
    )
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    resample_freq = args.resample_freq.strip() if args.resample_freq else ""
    merged = merge_all(parquet_paths, resample_freq or None)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(out_path, compression="zstd")
    print(f"Wrote {len(merged)} rows to {out_path}")


if __name__ == "__main__":
    main()
