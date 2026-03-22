#!/usr/bin/env python3
"""Diagnose and fix missingness in price log features.

Rules:
- If all gaps are 1-hour and occur only on DST spring-transition dates, fill via
  strict-hour interpolation on those timestamps.
- Otherwise treat as structural/source outages and apply seasonal persistence:
  T <- T-1w, fallback T-2w.

Usage:
    ./.venv/bin/python scripts/diagnose_and_fix_price_missingness.py
    ./.venv/bin/python scripts/diagnose_and_fix_price_missingness.py \
        --in data/features/all_data_features.parquet \
        --out data/features/all_data_features.parquet
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl


DEFAULT_INPUT_CANDIDATES = [
    Path("data/features/all_data_features.parquet"),
]
TARGET_COLS = ["da_price_eur_log1p", "price_intraday_eur_log1p"]


@dataclass
class Gap:
    start: datetime
    end: datetime
    length_h: int
    timestamps: list[datetime]


def _find_input_path(cli_path: str | None) -> Path:
    if cli_path:
        p = Path(cli_path)
        if not p.exists():
            raise FileNotFoundError(f"Input parquet not found: {p}")
        return p
    p = next((x for x in DEFAULT_INPUT_CANDIDATES if x.exists()), None)
    if p is None:
        raise FileNotFoundError(f"No input parquet found in: {DEFAULT_INPUT_CANDIDATES}")
    return p


def _ensure_hourly_timeline(df: pl.DataFrame) -> pl.DataFrame:
    if "timestamp_utc" not in df.columns:
        raise KeyError("Missing required column: timestamp_utc")
    out = df.with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime(time_zone="UTC"), strict=False)
    ).sort("timestamp_utc")
    ts_min = out.select(pl.col("timestamp_utc").min()).item()
    ts_max = out.select(pl.col("timestamp_utc").max()).item()
    if ts_min is None or ts_max is None:
        return out
    full = pl.DataFrame(
        {
            "timestamp_utc": pl.datetime_range(
                start=ts_min,
                end=ts_max,
                interval="1h",
                time_zone="UTC",
                eager=True,
            )
        }
    )
    return full.join(out, on="timestamp_utc", how="left").sort("timestamp_utc")


def _extract_gaps(df: pl.DataFrame, col: str) -> list[Gap]:
    miss_ts = (
        df.filter(pl.col(col).is_null())
        .select("timestamp_utc")
        .to_series()
        .to_list()
    )
    if not miss_ts:
        return []
    gaps: list[Gap] = []
    cur = [miss_ts[0]]
    for ts in miss_ts[1:]:
        prev = cur[-1]
        if ts - prev == timedelta(hours=1):
            cur.append(ts)
        else:
            gaps.append(Gap(start=cur[0], end=cur[-1], length_h=len(cur), timestamps=cur.copy()))
            cur = [ts]
    gaps.append(Gap(start=cur[0], end=cur[-1], length_h=len(cur), timestamps=cur.copy()))
    return gaps


def _last_sunday_of_march(y: int) -> date:
    d = date(y, 3, 31)
    while d.weekday() != 6:
        d -= timedelta(days=1)
    return d


def _is_dst_spring_gap_only(gaps: list[Gap]) -> tuple[bool, list[datetime]]:
    if not gaps:
        return False, []
    berlin = ZoneInfo("Europe/Berlin")
    years = {g.start.year for g in gaps}
    dst_dates = {_last_sunday_of_march(y) for y in years}
    eligible: list[datetime] = []

    for g in gaps:
        if g.length_h != 1:
            return False, []
        ts = g.start
        local_date = ts.astimezone(berlin).date()
        if local_date not in dst_dates:
            return False, []
        eligible.append(ts)
    return True, eligible


def _fix_dst_single_hour_interpolation(df: pl.DataFrame, col: str, ts_list: list[datetime]) -> tuple[pl.DataFrame, int]:
    if not ts_list:
        return df, 0
    before = df.filter(pl.col("timestamp_utc").is_in(ts_list) & pl.col(col).is_null()).height
    out = df.with_columns(
        pl.when(pl.col("timestamp_utc").is_in(ts_list) & pl.col(col).is_null())
        .then(pl.col(col).interpolate())
        .otherwise(pl.col(col))
        .alias(col)
    )
    after = out.filter(pl.col("timestamp_utc").is_in(ts_list) & pl.col(col).is_null()).height
    return out, max(0, before - after)


def _fix_outage_with_seasonal_persistence(df: pl.DataFrame, col: str) -> tuple[pl.DataFrame, int]:
    before = df.select(pl.col(col).null_count()).item()
    # Seasonal lookback keying by exact weekly offsets.
    out = (
        df.with_columns(
            pl.col("timestamp_utc").dt.offset_by("-1w").alias("__t_lag_1w"),
            pl.col("timestamp_utc").dt.offset_by("-2w").alias("__t_lag_2w"),
        )
        .join(
            df.select(
                pl.col("timestamp_utc").alias("__k1"),
                pl.col(col).alias("__v_lag_1w"),
            ),
            left_on="__t_lag_1w",
            right_on="__k1",
            how="left",
        )
        .join(
            df.select(
                pl.col("timestamp_utc").alias("__k2"),
                pl.col(col).alias("__v_lag_2w"),
            ),
            left_on="__t_lag_2w",
            right_on="__k2",
            how="left",
        )
        .with_columns(
            pl.coalesce(
                [
                    pl.col(col),
                    pl.col("__v_lag_1w"),
                    pl.col("__v_lag_2w"),
                ]
            ).alias(col)
        )
        .drop("__t_lag_1w", "__t_lag_2w", "__v_lag_1w", "__v_lag_2w")
    )
    after = out.select(pl.col(col).null_count()).item()
    return out, max(0, int(before - after))


def _fix_remaining_with_edge_continuity(df: pl.DataFrame, col: str) -> tuple[pl.DataFrame, int]:
    """Final safety fill for residual nulls after seasonal persistence.

    Applies forward-fill then backward-fill to close any remaining edge gaps.
    """
    before = int(df.select(pl.col(col).null_count()).item())
    out = df.with_columns(
        pl.col(col)
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
        .alias(col)
    )
    after = int(out.select(pl.col(col).null_count()).item())
    return out, max(0, before - after)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose DST vs outage missingness and fix price log columns.")
    parser.add_argument("--in", dest="input_path", default=None, help="Input parquet path.")
    parser.add_argument("--out", dest="output_path", default=None, help="Output parquet path. Defaults to in-place.")
    args = parser.parse_args()

    in_path = _find_input_path(args.input_path)
    out_path = Path(args.output_path) if args.output_path else in_path

    df = pl.read_parquet(in_path)
    df = _ensure_hourly_timeline(df)

    missing_cols = [c for c in TARGET_COLS if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing target column(s): {missing_cols}")

    total_dst_fixed = 0
    total_outage_fixed = 0
    total_edge_fixed = 0

    for col in TARGET_COLS:
        gaps = _extract_gaps(df, col)
        dst_only, dst_ts = _is_dst_spring_gap_only(gaps)

        if dst_only:
            df, n_fixed = _fix_dst_single_hour_interpolation(df, col, dst_ts)
            total_dst_fixed += n_fixed
            print(f"[{col}] mode=DST_INTERPOLATION gaps={len(gaps)} fixed={n_fixed}")
        else:
            df, n_fixed = _fix_outage_with_seasonal_persistence(df, col)
            total_outage_fixed += n_fixed
            max_gap = max((g.length_h for g in gaps), default=0)
            print(f"[{col}] mode=SEASONAL_PERSISTENCE gaps={len(gaps)} max_gap_h={max_gap} fixed={n_fixed}")

            # Residual edge gaps (if any) after T-1w/T-2w lookback.
            if int(df.select(pl.col(col).null_count()).item()) > 0:
                df, n_edge = _fix_remaining_with_edge_continuity(df, col)
                total_edge_fixed += n_edge
                if n_edge > 0:
                    print(f"[{col}] mode=EDGE_CONTINUITY_FALLBACK fixed={n_edge}")

    final_nulls = {
        c: int(df.select(pl.col(c).null_count()).item())
        for c in TARGET_COLS
    }

    print(
        "Summary: "
        f"Total DST Gaps Fixed={total_dst_fixed}, "
        f"Total Outage Gaps Fixed via Seasonal Persistence={total_outage_fixed}, "
        f"Total Edge Gaps Fixed={total_edge_fixed}"
    )
    print("Final null counts:", final_nulls)

    if any(v != 0 for v in final_nulls.values()):
        raise ValueError(f"Validation failed, remaining NaNs in target columns: {final_nulls}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    print(f"Wrote fixed parquet: {out_path}")


if __name__ == "__main__":
    main()
