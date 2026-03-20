#!/usr/bin/env python3
"""Transform ENTSO-E outage events to hourly outage features.

Input files:
- data/raw/entsoe_outages/planned_generation_outages.parquet
- data/raw/entsoe_outages/unplanned_generation_outages.parquet

Expected columns in each input:
- mrid
- unit_name
- unavailable_power
- start
- end

Output:
- data/processed/outages_hourly.parquet
  index: timestamp (UTC, hourly)
  columns:
    - planned_outages_mw
    - unplanned_outages_mw
    - total_outages_mw
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)


def _ensure_utc(ts: pd.Series) -> pd.Series:
    out = pd.to_datetime(ts, errors="coerce", utc=True)
    return out


def _load_outages(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing outage file: {path}")

    df = pd.read_parquet(path)
    required = {"unavailable_power", "start", "end"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path} missing required columns: {sorted(missing)}")

    df = df.copy()
    df["unavailable_power"] = pd.to_numeric(df["unavailable_power"], errors="coerce")
    df["start"] = _ensure_utc(df["start"])
    df["end"] = _ensure_utc(df["end"])
    df = df.dropna(subset=["unavailable_power", "start", "end"])
    return df


def _build_hourly_outage_series(
    events: pd.DataFrame,
    range_start: pd.Timestamp,
    range_end: pd.Timestamp,
    hourly_index: pd.DatetimeIndex,
) -> pd.Series:
    """Convert outage intervals [start, end) to hourly summed unavailable power.

    Uses a vectorized delta/cumsum approach:
    - +power at ceil(start)
    - -power at ceil(end)
    Then cumulative-sum over hourly grid.
    """
    if events.empty:
        return pd.Series(0.0, index=hourly_index, dtype="float64")

    overlaps = events[(events["end"] > range_start) & (events["start"] < range_end)].copy()
    if overlaps.empty:
        return pd.Series(0.0, index=hourly_index, dtype="float64")

    overlaps["start_clip"] = overlaps["start"].clip(lower=range_start, upper=range_end)
    overlaps["end_clip"] = overlaps["end"].clip(lower=range_start, upper=range_end)

    # Hour t is active iff start <= t < end, so use ceil-hour boundaries.
    overlaps["delta_start"] = overlaps["start_clip"].dt.ceil("h")
    overlaps["delta_end"] = overlaps["end_clip"].dt.ceil("h")

    overlaps = overlaps[overlaps["delta_start"] < overlaps["delta_end"]]
    if overlaps.empty:
        return pd.Series(0.0, index=hourly_index, dtype="float64")

    add = overlaps.groupby("delta_start", as_index=True)["unavailable_power"].sum()
    sub = overlaps.groupby("delta_end", as_index=True)["unavailable_power"].sum()
    deltas = add.sub(sub, fill_value=0.0).sort_index()

    # Include a baseline delta at range start so cumsum starts at 0.
    if range_start not in deltas.index:
        deltas.loc[range_start] = 0.0
    deltas = deltas.sort_index()

    # Project onto hourly grid and cumulative sum.
    projected = deltas.reindex(hourly_index, fill_value=0.0).cumsum()
    projected.name = "outages_mw"
    return projected.astype("float64")


def transform_outages_hourly(
    planned_path: Path,
    unplanned_path: Path,
    out_path: Path,
    days_ahead: int = 7,
) -> pd.DataFrame:
    now_utc = pd.Timestamp.now("UTC")
    range_start = now_utc.floor("h")
    range_end = range_start + pd.Timedelta(days=days_ahead)
    hourly_index = pd.date_range(range_start, range_end, freq="1h", inclusive="left", tz="UTC")

    planned = _load_outages(planned_path)
    unplanned = _load_outages(unplanned_path)

    planned_series = _build_hourly_outage_series(planned, range_start, range_end, hourly_index)
    unplanned_series = _build_hourly_outage_series(unplanned, range_start, range_end, hourly_index)

    df_outages_hourly = pd.DataFrame(
        {
            "planned_outages_mw": planned_series,
            "unplanned_outages_mw": unplanned_series,
        },
        index=hourly_index,
    )
    df_outages_hourly["total_outages_mw"] = (
        df_outages_hourly["planned_outages_mw"] + df_outages_hourly["unplanned_outages_mw"]
    )
    df_outages_hourly.index.name = "timestamp"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_outages_hourly.to_parquet(out_path, index=True)
    return df_outages_hourly


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Transform ENTSO-E outage events to hourly outage features.")
    parser.add_argument(
        "--planned",
        default="data/raw/entsoe_outages/planned_generation_outages.parquet",
        help="Input parquet for planned outages.",
    )
    parser.add_argument(
        "--unplanned",
        default="data/raw/entsoe_outages/unplanned_generation_outages.parquet",
        help="Input parquet for unplanned outages.",
    )
    parser.add_argument(
        "--out",
        default="data/processed/outages_hourly.parquet",
        help="Output hourly outage parquet.",
    )
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=7,
        help="Forecast horizon in days (default: 7).",
    )
    args = parser.parse_args()

    try:
        out_df = transform_outages_hourly(
            planned_path=Path(args.planned),
            unplanned_path=Path(args.unplanned),
            out_path=Path(args.out),
            days_ahead=args.days_ahead,
        )
    except Exception as exc:
        LOGGER.error("Failed to transform outage events: %s", exc)
        raise

    LOGGER.info("Wrote %s rows to %s", len(out_df), args.out)


if __name__ == "__main__":
    main()
