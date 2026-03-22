#!/usr/bin/env python3
"""Forensic gap mapping and tiered imputation for energy-market features.

Imputation tiers:
1) Micro-gaps (1-3h): linear interpolation.
2) 24h gaps: daily persistence (T-24), fallback weekly persistence (T-168).
3) Long blocks (e.g. ~625h): grouped seasonal mean by (year, month, weekday, hour).

Special handling:
- For `wind_onshore_error_id` and `load_abs_error`, if a 24h gap coincides with
  missing underlying Actual/Forecast inputs, impute underlying series first and
  recompute the error rather than imputing the error directly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl


DEFAULT_INPUT_CANDIDATES = [
    Path("data/features/all_data_features.parquet"),
]
TARGET_PRICE_COLS = ["da_price_eur_log1p", "price_intraday_eur_log1p", "da_price_BE"]
ERROR_COLS = ["wind_onshore_error_id", "load_abs_error"]


@dataclass
class GapBlock:
    start: pd.Timestamp
    end: pd.Timestamp
    duration_h: int
    idx: np.ndarray


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
    full = pl.DataFrame(
        {
            "timestamp_utc": pl.datetime_range(
                start=ts_min,
                end=ts_max,
                interval="1h",
                eager=True,
                time_zone="UTC",
            )
        }
    )
    return full.join(out, on="timestamp_utc", how="left").sort("timestamp_utc")


def _gap_blocks(series: pd.Series, ts: pd.Series) -> list[GapBlock]:
    mask = series.isna().to_numpy(dtype=bool)
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    blocks: list[GapBlock] = []
    s = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            block_idx = np.arange(s, prev + 1, dtype=int)
            blocks.append(
                GapBlock(
                    start=ts.iloc[s],
                    end=ts.iloc[prev],
                    duration_h=len(block_idx),
                    idx=block_idx,
                )
            )
            s = i
            prev = i
    block_idx = np.arange(s, prev + 1, dtype=int)
    blocks.append(
        GapBlock(
            start=ts.iloc[s],
            end=ts.iloc[prev],
            duration_h=len(block_idx),
            idx=block_idx,
        )
    )
    return blocks


def _print_gap_summary(col: str, blocks: Iterable[GapBlock]) -> None:
    for b in blocks:
        print(f"{col} | {b.start} | {b.end} | {b.duration_h}h")


def _seasonal_group_means(series: pd.Series, ts: pd.Series) -> pd.Series:
    tmp = pd.DataFrame(
        {
            "v": pd.to_numeric(series, errors="coerce"),
            "year": ts.dt.year,
            "month": ts.dt.month,
            "weekday": ts.dt.weekday,
            "hour": ts.dt.hour,
        }
    )
    grp_y = (
        tmp.dropna(subset=["v"])
        .groupby(["year", "month", "weekday", "hour"], as_index=False)["v"]
        .mean()
        .rename(columns={"v": "seasonal_mean_y"})
    )
    grp_m = (
        tmp.dropna(subset=["v"])
        .groupby(["month", "weekday", "hour"], as_index=False)["v"]
        .mean()
        .rename(columns={"v": "seasonal_mean_m"})
    )
    grp_w = (
        tmp.dropna(subset=["v"])
        .groupby(["weekday", "hour"], as_index=False)["v"]
        .mean()
        .rename(columns={"v": "seasonal_mean_w"})
    )
    out = tmp.merge(grp_y, on=["year", "month", "weekday", "hour"], how="left")
    out = out.merge(grp_m, on=["month", "weekday", "hour"], how="left")
    out = out.merge(grp_w, on=["weekday", "hour"], how="left")
    return out["seasonal_mean_y"].combine_first(out["seasonal_mean_m"]).combine_first(out["seasonal_mean_w"])


def _apply_tiered_imputation(series: pd.Series, ts: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").copy()
    blocks = _gap_blocks(s, ts)
    if not blocks:
        return s

    interp = s.interpolate(method="linear", limit_direction="both")
    lag24 = s.shift(24)
    lag168 = s.shift(168)
    seasonal_mean = _seasonal_group_means(s, ts)

    for b in blocks:
        if b.duration_h <= 3:
            s.iloc[b.idx] = interp.iloc[b.idx]
        elif b.duration_h == 24:
            fill_vals = lag24.iloc[b.idx].copy()
            missing_fill = fill_vals.isna()
            if missing_fill.any():
                fill_vals.loc[missing_fill] = lag168.iloc[b.idx][missing_fill]
            s.iloc[b.idx] = fill_vals
        else:
            # Long structural outages: grouped seasonal mean, no ffill/bfill.
            s.iloc[b.idx] = seasonal_mean.iloc[b.idx]
    return s


def _check_24h_alignment(blocks_a: list[GapBlock], blocks_b: list[GapBlock]) -> bool:
    a = {(b.start, b.end) for b in blocks_a if b.duration_h == 24}
    b = {(b.start, b.end) for b in blocks_b if b.duration_h == 24}
    return len(a) > 0 and a == b


def _recompute_error_columns(pdf: pd.DataFrame) -> pd.DataFrame:
    # wind_onshore_error_id := actual - id-forecast
    if {"wind_onshore_actual_entsoe", "wind_onshore_forecast_id_entsoe"}.issubset(pdf.columns):
        pdf["wind_onshore_error_id"] = (
            pd.to_numeric(pdf["wind_onshore_actual_entsoe"], errors="coerce")
            - pd.to_numeric(pdf["wind_onshore_forecast_id_entsoe"], errors="coerce")
        )

    # load_abs_error := abs(da-forecast - actual)
    if {"load_forecast_da_entsoe", "load_actual_entsoe"}.issubset(pdf.columns):
        pdf["load_abs_error"] = (
            pd.to_numeric(pdf["load_forecast_da_entsoe"], errors="coerce")
            - pd.to_numeric(pdf["load_actual_entsoe"], errors="coerce")
        ).abs()
    return pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Forensic gap mapping + tiered imputation")
    parser.add_argument("--in", dest="input_path", default=None, help="Input parquet path.")
    parser.add_argument("--out", dest="output_path", default=None, help="Output parquet path.")
    args = parser.parse_args()

    in_path = _find_input_path(args.input_path)
    out_path = Path(args.output_path) if args.output_path else in_path

    df = pl.read_parquet(in_path)
    df = _ensure_hourly_timeline(df)
    pdf = df.to_pandas()
    pdf["timestamp_utc"] = pd.to_datetime(pdf["timestamp_utc"], utc=True, errors="coerce")
    pdf = pdf.sort_values("timestamp_utc").reset_index(drop=True)
    ts = pdf["timestamp_utc"]

    # 1) Gap mapping for requested columns.
    print("=== Gap Mapping ===")
    gap_map: dict[str, list[GapBlock]] = {}
    for col in TARGET_PRICE_COLS:
        if col not in pdf.columns:
            print(f"{col} | MISSING COLUMN")
            gap_map[col] = []
            continue
        blocks = _gap_blocks(pdf[col], ts)
        gap_map[col] = blocks
        _print_gap_summary(col, blocks)

    # Hypothesis check (example da_price_BE vs wind_onshore_error_id)
    if "wind_onshore_error_id" in pdf.columns and "da_price_BE" in gap_map:
        err_blocks = _gap_blocks(pdf["wind_onshore_error_id"], ts)
        same_24h = _check_24h_alignment(gap_map["da_price_BE"], err_blocks)
        print(f"Hypothesis 24h alignment (da_price_BE vs wind_onshore_error_id): {same_24h}")

    # 2) Tiered imputation for price columns.
    print("\n=== Tiered Imputation (Prices) ===")
    for col in TARGET_PRICE_COLS:
        if col not in pdf.columns:
            continue
        before = int(pd.to_numeric(pdf[col], errors="coerce").isna().sum())
        pdf[col] = _apply_tiered_imputation(pdf[col], ts)
        after = int(pd.to_numeric(pdf[col], errors="coerce").isna().sum())
        print(f"{col}: filled={before - after}, remaining_nulls={after}")

    # 3) Refined error logic for 24h blocks.
    print("\n=== Refined Error Logic ===")
    for err_col in ERROR_COLS:
        if err_col not in pdf.columns:
            continue
        blocks = _gap_blocks(pdf[err_col], ts)
        blocks_24h = [b for b in blocks if b.duration_h == 24]
        if not blocks_24h:
            continue

        if err_col == "wind_onshore_error_id":
            deps = ("wind_onshore_actual_entsoe", "wind_onshore_forecast_id_entsoe")
        else:
            deps = ("load_actual_entsoe", "load_forecast_da_entsoe")

        if not set(deps).issubset(pdf.columns):
            # Fallback: direct tiered imputation on error col if deps unavailable.
            pdf[err_col] = _apply_tiered_imputation(pdf[err_col], ts)
            print(f"{err_col}: deps missing -> imputed directly")
            continue

        # If underlying deps are missing over 24h block, impute deps + recompute error.
        dep_a, dep_b = deps
        needs_recompute = False
        for b in blocks_24h:
            dep_missing = pdf.loc[b.idx, dep_a].isna() & pdf.loc[b.idx, dep_b].isna()
            if dep_missing.any():
                needs_recompute = True
                break

        if needs_recompute:
            pdf[dep_a] = _apply_tiered_imputation(pdf[dep_a], ts)
            pdf[dep_b] = _apply_tiered_imputation(pdf[dep_b], ts)
            pdf = _recompute_error_columns(pdf)
            print(f"{err_col}: recomputed from imputed underlying deps ({dep_a}, {dep_b})")
        else:
            pdf[err_col] = _apply_tiered_imputation(pdf[err_col], ts)
            print(f"{err_col}: imputed directly (deps present in 24h blocks)")

    # Final validation for requested forensic columns.
    print("\n=== Final Null Counts ===")
    for col in TARGET_PRICE_COLS + ERROR_COLS:
        if col in pdf.columns:
            n = int(pd.to_numeric(pdf[col], errors="coerce").isna().sum())
            print(f"{col}: {n}")

    out = pl.from_pandas(pdf).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime(time_zone="UTC"), strict=False)
    ).sort("timestamp_utc")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path, compression="zstd")
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
