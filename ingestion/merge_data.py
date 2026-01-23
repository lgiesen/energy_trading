"""Merge all parquet files in the data directory on timestamp into one table.

Usage: python3 -m ingestion.merge_data --data-dir data --out data/all_merged.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import polars as pl


def _load_and_normalize(path: Path) -> pl.DataFrame | None:
    """Load a parquet and normalize its timestamp column name and type."""
    df = pl.read_parquet(path)
    if "timestamp" in df.columns:
        ts_col = "timestamp"
    elif "timestamp_utc" in df.columns:
        df = df.rename({"timestamp_utc": "timestamp"})
        ts_col = "timestamp"
    else:
        return None  # Skip files without a timestamp column.

    # Cast timestamp to datetime[us, UTC] to avoid join mismatches.
    ts_dtype = df[ts_col].dtype
    if ts_dtype == pl.Int64:
        df = df.with_columns(pl.from_epoch(ts_col, time_unit="ms").alias(ts_col))
    elif isinstance(ts_dtype, pl.datatypes.Datetime):
        # Align units to microseconds for consistency.
        df = df.with_columns(pl.col(ts_col).dt.cast_time_unit("us").alias(ts_col))
    # Add UTC tz if missing.
    if df[ts_col].dtype == pl.Datetime and df[ts_col].dtype.time_zone is None:  # type: ignore[attr-defined]
        df = df.with_columns(pl.col(ts_col).dt.replace_time_zone("UTC").alias(ts_col))
    return df


def merge_all(parquet_paths: Iterable[Path]) -> pl.DataFrame:
    """Full-join all provided parquet files on timestamp."""
    merged: pl.DataFrame | None = None
    for path in parquet_paths:
        df = _load_and_normalize(path)
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

        merged = merged.join(df, on="timestamp", how="full")
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
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    parquet_paths: List[Path] = sorted(data_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    merged = merge_all(parquet_paths)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(out_path, compression="zstd")
    print(f"Wrote {len(merged)} rows to {out_path}")


if __name__ == "__main__":
    main()
