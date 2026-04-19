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


def _parse_utc_opt(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _load_outages(path: Path, *, require_end: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing outage file: {path}")

    df = pd.read_parquet(path)
    required = {"unavailable_power", "start"} | ({"end"} if require_end else set())
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path} missing required columns: {sorted(missing)}")

    df = df.copy()
    df["unavailable_power"] = pd.to_numeric(df["unavailable_power"], errors="coerce")
    df["start"] = _ensure_utc(df["start"])
    if "end" in df.columns:
        df["end"] = _ensure_utc(df["end"])
    drop_subset = ["unavailable_power", "start"] + (["end"] if require_end else [])
    df = df.dropna(subset=drop_subset)
    return df


def _apply_unplanned_persistence_window(events: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    """Replace ex-post unplanned `end` with causal persistence window.

    For unplanned outages, the historical `end` is often revised retroactively.
    Using it directly leaks future repair information. We therefore define:
      synthetic_end = start + hours
    """
    if events.empty:
        return events
    out = events.copy()
    out["end"] = out["start"] + pd.Timedelta(hours=hours)
    return out


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
    range_start: pd.Timestamp | None = None,
    range_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    planned = _load_outages(planned_path, require_end=True)
    unplanned_raw = _load_outages(unplanned_path, require_end=False)
    # Causal safety: ignore retroactively known repair time for unplanned outages.
    unplanned = _apply_unplanned_persistence_window(unplanned_raw, hours=24)

    if range_start is None or range_end is None:
        starts = [
            s for s in (
                planned["start"].min() if not planned.empty else pd.NaT,
                unplanned["start"].min() if not unplanned.empty else pd.NaT,
            )
            if pd.notna(s)
        ]
        ends = [
            e for e in (
                planned["end"].max() if not planned.empty else pd.NaT,
                unplanned["end"].max() if not unplanned.empty else pd.NaT,
            )
            if pd.notna(e)
        ]
        if not starts or not ends:
            now_utc = pd.Timestamp.now("UTC").floor("h")
            range_start = now_utc
            range_end = now_utc + pd.Timedelta(days=days_ahead)
        else:
            range_start = min(starts).floor("h")
            range_end = max(ends).ceil("h")
    else:
        range_start = pd.Timestamp(range_start).tz_convert("UTC") if pd.Timestamp(range_start).tzinfo else pd.Timestamp(range_start).tz_localize("UTC")
        range_end = pd.Timestamp(range_end).tz_convert("UTC") if pd.Timestamp(range_end).tzinfo else pd.Timestamp(range_end).tz_localize("UTC")
    if range_end <= range_start:
        raise ValueError("range_end must be after range_start")

    hourly_index = pd.date_range(range_start, range_end, freq="1h", inclusive="left", tz="UTC")

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
        help="Fallback horizon in days if no usable event bounds exist (default: 7).",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional UTC start override, e.g. 2020-11-29T23:00:00Z.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Optional UTC end override, e.g. 2026-03-01T02:00:00Z.",
    )
    args = parser.parse_args()

    try:
        out_df = transform_outages_hourly(
            planned_path=Path(args.planned),
            unplanned_path=Path(args.unplanned),
            out_path=Path(args.out),
            days_ahead=args.days_ahead,
            range_start=_parse_utc_opt(args.start),
            range_end=_parse_utc_opt(args.end),
        )
    except Exception as exc:
        LOGGER.error("Failed to transform outage events: %s", exc)
        raise

    LOGGER.info("Wrote %s rows to %s", len(out_df), args.out)


if __name__ == "__main__":
    main()
