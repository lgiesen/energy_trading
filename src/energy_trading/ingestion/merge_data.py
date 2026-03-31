"""Merge all parquet files in the data directory on timestamp_utc into one table.

Usage:
    ./.venv/bin/python -m energy_trading.ingestion.merge_data \
        --data-dir data/raw \
        --out data/processed/all_data.parquet \
        --clip-start 2020-11-30T23:00:00Z \
        --clip-end 2026-03-01T02:00:00Z


Notes:
    - Prefers timestamp_utc if available.
    - Drops other timestamp columns to avoid duplicate/suffixed fields.
    - Merges all columns produced by fetchers (including anonymous-bid price columns
      from regelleistung.parquet if present).
    - Also left-joins hourly outages sidecar (`data/processed/outages_hourly.parquet`)
      on UTC timestamp when available.
    - The example above clips to CET boundaries:
      2020-12-01 00:00:00 CET -> 2026-03-01 03:00:00 CET,
      passed as UTC (`Z`) values.
    - `--clip-start/--clip-end` accept timezone-aware ISO values (recommended: UTC with `Z`).
      If no timezone is provided, values are interpreted as Europe/Berlin local time.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List
from zoneinfo import ZoneInfo

import polars as pl

LOGGER = logging.getLogger(__name__)
EXCLUDED_TIMESTAMPS_UTC = {
    # Known DST transition artifact hour not relevant for this project window semantics.
    datetime.fromisoformat("2021-10-31T22:00:00+00:00"),
}


def _parse_clip_to_utc(value: str):
    """Parse clip boundary; keep explicit timezone, default naive to Europe/Berlin."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("Europe/Berlin"))
    return dt.astimezone(ZoneInfo("UTC"))

def _normalize_timestamp(df: pl.DataFrame, ts_col: str) -> pl.DataFrame:
    """Normalize timestamp column to datetime[us, UTC] without changing resolution."""
    ts_dtype = df[ts_col].dtype
    if ts_dtype == pl.Int64:
        df = df.with_columns(pl.from_epoch(ts_col, time_unit="ms").alias(ts_col))
    elif ts_dtype == pl.Utf8:
        df = df.with_columns(pl.col(ts_col).str.strptime(pl.Datetime, strict=False).alias(ts_col))
    elif isinstance(ts_dtype, pl.datatypes.Datetime):
        df = df.with_columns(pl.col(ts_col).dt.cast_time_unit("us").alias(ts_col))

    if df[ts_col].dtype == pl.Datetime:  # type: ignore[attr-defined]
        tz = df[ts_col].dtype.time_zone  # type: ignore[attr-defined]
        if tz is None:
            df = df.with_columns(pl.col(ts_col).dt.replace_time_zone("UTC").alias(ts_col))
        elif tz != "UTC":
            df = df.with_columns(pl.col(ts_col).dt.convert_time_zone("UTC").alias(ts_col))
    return df


def _drop_nulls_and_dedup(df: pl.DataFrame, ts_col: str, label: str) -> pl.DataFrame:
    """Drop null timestamps and de-duplicate by timestamp, logging any removals."""
    null_count = df.filter(pl.col(ts_col).is_null()).height
    if null_count:
        LOGGER.warning("Dropping %s rows with null %s in %s.", null_count, ts_col, label)
        df = df.filter(pl.col(ts_col).is_not_null())

    dupes = df.select(pl.col(ts_col).is_duplicated().sum()).item()
    if dupes:
        LOGGER.warning("Dropping %s duplicate %s values in %s.", dupes, ts_col, label)
        df = df.unique(subset=[ts_col], keep="last")
    return df


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
    if ts_col == "timestamp_cet":
        tz = df[ts_col].dtype.time_zone if isinstance(df[ts_col].dtype, pl.datatypes.Datetime) else None
        if tz is None:
            df = df.with_columns(pl.col(ts_col).dt.replace_time_zone("Europe/Berlin").alias(ts_col))
        df = df.with_columns(pl.col(ts_col).dt.convert_time_zone("UTC").alias(ts_col))
    df = _normalize_timestamp(df, ts_col)
    df = _drop_nulls_and_dedup(df, ts_col, path.name)

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
        df = _drop_nulls_and_dedup(df, "timestamp", f"{path.name} (resampled)")
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


def _merge_outages_sidecar(merged: pl.DataFrame, outages_path: Path) -> pl.DataFrame:
    """Left-join hourly outage sidecar on canonical timestamp key.

    Missing sidecar or missing rows are interpreted as no outage and filled with 0.0.
    """
    if "timestamp" not in merged.columns:
        return merged

    if not outages_path.exists():
        LOGGER.warning("Outage sidecar not found (%s). Filling outages with 0.0.", outages_path)
        return merged.with_columns([
            pl.lit(0.0).cast(pl.Float64).alias("planned_outages_mw"),
            pl.lit(0.0).cast(pl.Float64).alias("unplanned_outages_mw"),
        ])

    outages = pl.read_parquet(outages_path)

    if "timestamp_utc" in outages.columns:
        ts_col = "timestamp_utc"
    elif "timestamp" in outages.columns:
        ts_col = "timestamp"
    elif "__index_level_0__" in outages.columns:
        ts_col = "__index_level_0__"
    else:
        LOGGER.warning("Outage sidecar has no timestamp column. Filling outages with 0.0.")
        return merged.with_columns([
            pl.lit(0.0).cast(pl.Float64).alias("planned_outages_mw"),
            pl.lit(0.0).cast(pl.Float64).alias("unplanned_outages_mw"),
        ])

    outages = _normalize_timestamp(outages, ts_col)
    if ts_col != "timestamp":
        outages = outages.rename({ts_col: "timestamp"})

    keep = [c for c in ("timestamp", "planned_outages_mw", "unplanned_outages_mw") if c in outages.columns]
    outages = outages.select(keep)
    outages = _drop_nulls_and_dedup(outages, "timestamp", "outages sidecar")

    joined = merged.join(outages, on="timestamp", how="left")
    for c in ("planned_outages_mw", "unplanned_outages_mw"):
        if c in joined.columns:
            joined = joined.with_columns(pl.col(c).cast(pl.Float64).fill_null(0.0).alias(c))
        else:
            joined = joined.with_columns(pl.lit(0.0).cast(pl.Float64).alias(c))
    return joined


def merge_all(
    parquet_paths: Iterable[Path],
    resample_freq: str | None,
    clip_start: str | None,
    clip_end: str | None,
) -> pl.DataFrame:
    """Full-join all provided parquet files on timestamp."""
    merged: pl.DataFrame | None = None
    for path in parquet_paths:
        df = _load_and_normalize(path, resample_freq)
        if df is None:
            LOGGER.warning("Skipping %s: no timestamp column.", path.name)
            continue

        if merged is None:
            merged = df
            LOGGER.info("Seeded merge with %s (%s rows).", path.name, len(df))
            continue

        # Avoid column name collisions (other than timestamp) by suffixing with file stem.
        overlap = (set(df.columns) & set(merged.columns)) - {"timestamp"}
        if overlap:
            df = df.rename({col: f"{col}_{path.stem}" for col in overlap})

        merged = merged.join(df, on="timestamp", how="full", suffix=f"_{path.stem}")
        # Coalesce any timestamp_right into timestamp, then drop.
        if "timestamp_right" in merged.columns:
            merged = merged.with_columns(
                pl.coalesce(["timestamp", "timestamp_right"]).alias("timestamp")
            ).drop("timestamp_right")
        LOGGER.info("Merged %s -> %s rows.", path.name, len(merged))

    if merged is None:
        raise RuntimeError("No parquet files with a timestamp column were merged.")

    merged = merged.sort("timestamp")
    merged = _drop_nulls_and_dedup(merged, "timestamp", "merged output")
    if EXCLUDED_TIMESTAMPS_UTC:
        before = merged.height
        merged = merged.filter(~pl.col("timestamp").is_in(list(EXCLUDED_TIMESTAMPS_UTC)))
        dropped = before - merged.height
        if dropped:
            LOGGER.info("Dropped %s known excluded timestamp rows.", dropped)
    if clip_start:
        start_dt = _parse_clip_to_utc(clip_start)
        merged = merged.filter(pl.col("timestamp") >= pl.lit(start_dt))
    if clip_end:
        end_dt = _parse_clip_to_utc(clip_end)
        merged = merged.filter(pl.col("timestamp") <= pl.lit(end_dt))

    # Merge outages before final timestamp projection, so outages participate in
    # all downstream processing/audits from all_data.parquet onward.
    outages_path = Path(__file__).resolve().parents[3] / "data" / "processed" / "outages_hourly.parquet"
    merged = _merge_outages_sidecar(merged, outages_path)

    # Keep only canonical UTC and CET timestamps in final output.
    merged = merged.with_columns(
        pl.col("timestamp").alias("timestamp_utc"),
        pl.col("timestamp").dt.convert_time_zone("Europe/Berlin").alias("timestamp_cet"),
    )
    # Drop all other timestamp-like columns and the raw join key.
    drop_ts = [c for c in merged.columns if c.startswith("timestamp_") and c not in {"timestamp_utc", "timestamp_cet"}]
    if drop_ts:
        merged = merged.drop(drop_ts)
    merged = merged.drop("timestamp")
    return merged


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Merge all parquet files in the data directory on timestamp.")
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[3] / "data" / "raw"),
        help="Directory containing parquet files to merge (default: data/raw).",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[3] / "data" / "processed" / "all_data.parquet"),
        help="Output parquet path.",
    )
    parser.add_argument(
        "--resample-freq",
        default="1h",
        help="Optional resample frequency before merge (default: 1h). Use '' to disable.",
    )
    parser.add_argument(
        "--clip-start",
        default="",
        help="Optional clip start (timezone-aware recommended, e.g. 2020-11-30T23:00:00Z). "
             "Naive values are interpreted as Europe/Berlin.",
    )
    parser.add_argument(
        "--clip-end",
        default="",
        help="Optional clip end (timezone-aware recommended, e.g. 2026-03-01T02:00:00Z). "
             "Naive values are interpreted as Europe/Berlin.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    parquet_paths: List[Path] = sorted(
        p for p in data_dir.glob("*.parquet")
        if p.name not in {"all_merged.parquet", "all_data.parquet"}
        and not p.name.endswith("_temp.parquet")
    )
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    resample_freq = args.resample_freq.strip() if args.resample_freq else ""
    clip_start = args.clip_start.strip() or None
    clip_end = args.clip_end.strip() or None
    merged = merge_all(parquet_paths, resample_freq or None, clip_start, clip_end)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(out_path, compression="zstd")
    LOGGER.info("Wrote %s rows to %s", len(merged), out_path)


if __name__ == "__main__":
    main()
