"""Validate hourly aFRR VWAP integrity against 15-minute source data.

Usage:
    ./.venv/bin/python scripts/validate_afrr_vwap_integrity.py \
        --raw-15m data/raw/regelleistung_15min/afrr_price_volume_15min.parquet \
        --hourly data/processed/all_data_refined.parquet
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

PICASSO_START_UTC = datetime(2022, 6, 22, 22, 0, tzinfo=timezone.utc)


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


@dataclass
class ColMap:
    price_pos_raw: str | None
    price_neg_raw: str | None
    price_pos_ffill: str | None
    price_neg_ffill: str | None
    vol_pos: str | None
    vol_neg: str | None
    vwap_pos: str | None
    vwap_neg: str | None


def _resolve_columns(raw_cols: list[str], hourly_cols: list[str]) -> ColMap:
    return ColMap(
        price_pos_raw=_first_existing(
            raw_cols,
            [
                "afrr_avg_activation_price_pos",
                "afrr_activation_avg_price_pos",
                "arbeitspreis_pos",
            ],
        ),
        price_neg_raw=_first_existing(
            raw_cols,
            [
                "afrr_avg_activation_price_neg",
                "afrr_activation_avg_price_neg",
                "arbeitspreis_neg",
            ],
        ),
        price_pos_ffill=_first_existing(raw_cols, ["afrr_activation_price_pos_ffill"]),
        price_neg_ffill=_first_existing(raw_cols, ["afrr_activation_price_neg_ffill"]),
        vol_pos=_first_existing(raw_cols, ["afrr_activated_mw_pos", "activated_volume_pos_mw", "abgerufene_arbeit_pos"]),
        vol_neg=_first_existing(raw_cols, ["afrr_activated_mw_neg", "activated_volume_neg_mw", "abgerufene_arbeit_neg"]),
        vwap_pos=_first_existing(hourly_cols, ["afrr_vwap_pos_eur_mwh", "afrr_vwap_pos"]),
        vwap_neg=_first_existing(hourly_cols, ["afrr_vwap_neg_eur_mwh", "afrr_vwap_neg"]),
    )


def _load_parquet(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if "timestamp_utc" not in df.columns:
        raise ValueError(f"{path} is missing timestamp_utc")
    return df.with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)
    ).drop_nulls(["timestamp_utc"]).sort("timestamp_utc")


def _manual_hourly_vwap(raw_15m: pl.DataFrame, c: ColMap) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    cutoff = pl.lit(PICASSO_START_UTC).cast(pl.Datetime(time_unit="us", time_zone="UTC"))

    if c.price_pos_ffill and c.price_pos_ffill in raw_15m.columns:
        exprs.append(pl.col(c.price_pos_ffill).cast(pl.Float64, strict=False).alias("__price_pos_work"))
    elif c.price_pos_raw:
        exprs.append(
            pl.when(pl.col("timestamp_utc") < cutoff)
            .then(pl.col(c.price_pos_raw).cast(pl.Float64, strict=False).forward_fill().over(pl.col("timestamp_utc").dt.truncate("1h")))
            .otherwise(pl.col(c.price_pos_raw).cast(pl.Float64, strict=False))
            .alias("__price_pos_work")
        )

    if c.price_neg_ffill and c.price_neg_ffill in raw_15m.columns:
        exprs.append(pl.col(c.price_neg_ffill).cast(pl.Float64, strict=False).alias("__price_neg_work"))
    elif c.price_neg_raw:
        exprs.append(
            pl.when(pl.col("timestamp_utc") < cutoff)
            .then(pl.col(c.price_neg_raw).cast(pl.Float64, strict=False).forward_fill().over(pl.col("timestamp_utc").dt.truncate("1h")))
            .otherwise(pl.col(c.price_neg_raw).cast(pl.Float64, strict=False))
            .alias("__price_neg_work")
        )

    if c.vol_pos:
        exprs.append(pl.col(c.vol_pos).cast(pl.Float64, strict=False).alias("__vol_pos"))
    if c.vol_neg:
        exprs.append(pl.col(c.vol_neg).cast(pl.Float64, strict=False).alias("__vol_neg"))

    df = raw_15m.with_columns(exprs)

    calc_exprs: list[pl.Expr] = []
    agg_exprs: list[pl.Expr] = []
    if "__price_pos_work" in df.columns and "__vol_pos" in df.columns:
        calc_exprs.append((pl.col("__price_pos_work") * pl.col("__vol_pos").abs()).alias("__wc_pos"))
        agg_exprs.extend(
            [
                pl.col("__wc_pos").sum().alias("__sum_wc_pos"),
                pl.col("__vol_pos").abs().sum().alias("__sum_vol_pos"),
                pl.col("__price_pos_work").mean().alias("__mean_price_pos"),
            ]
        )
    if "__price_neg_work" in df.columns and "__vol_neg" in df.columns:
        calc_exprs.append((pl.col("__price_neg_work") * pl.col("__vol_neg").abs()).alias("__wc_neg"))
        agg_exprs.extend(
            [
                pl.col("__wc_neg").sum().alias("__sum_wc_neg"),
                pl.col("__vol_neg").abs().sum().alias("__sum_vol_neg"),
                pl.col("__price_neg_work").mean().alias("__mean_price_neg"),
            ]
        )

    out = (
        df.with_columns(calc_exprs)
        .group_by_dynamic("timestamp_utc", every="1h", period="1h", closed="left", label="left")
        .agg(agg_exprs)
        .sort("timestamp_utc")
    )

    vwap_exprs: list[pl.Expr] = []
    if "__sum_vol_pos" in out.columns:
        vwap_exprs.append(
            pl.when(pl.col("__sum_vol_pos").is_null() | (pl.col("__sum_vol_pos") == 0.0))
            .then(pl.col("__mean_price_pos"))
            .otherwise(pl.col("__sum_wc_pos") / pl.col("__sum_vol_pos"))
            .alias("manual_vwap_pos_eur_mwh")
        )
    if "__sum_vol_neg" in out.columns:
        vwap_exprs.append(
            pl.when(pl.col("__sum_vol_neg").is_null() | (pl.col("__sum_vol_neg") == 0.0))
            .then(pl.col("__mean_price_neg"))
            .otherwise(pl.col("__sum_wc_neg") / pl.col("__sum_vol_neg"))
            .alias("manual_vwap_neg_eur_mwh")
        )
    return out.with_columns(vwap_exprs)


def _sample_hours(df: pl.DataFrame, sample_size: int, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    pre = df.filter(pl.col("timestamp_utc") < pl.lit(PICASSO_START_UTC).cast(pl.Datetime("us", "UTC")))
    post = df.filter(pl.col("timestamp_utc") >= pl.lit(PICASSO_START_UTC).cast(pl.Datetime("us", "UTC")))
    n_pre = min(sample_size, pre.height)
    n_post = min(sample_size, post.height)
    return (
        pre.sample(n=n_pre, with_replacement=False, shuffle=True, seed=seed).sort("timestamp_utc") if n_pre else pre,
        post.sample(n=n_post, with_replacement=False, shuffle=True, seed=seed + 1).sort("timestamp_utc") if n_post else post,
    )


def _dst_transition_report(hourly: pl.DataFrame, vwap_col: str) -> pd.DataFrame:
    pdf = hourly.select(["timestamp_utc", vwap_col]).to_pandas()
    pdf["timestamp_utc"] = pd.to_datetime(pdf["timestamp_utc"], utc=True)
    pdf = pdf.sort_values("timestamp_utc").reset_index(drop=True)
    local = pdf["timestamp_utc"].dt.tz_convert("Europe/Berlin")
    offset = local.map(lambda ts: ts.utcoffset().total_seconds() / 3600.0)
    transition_idx = np.where(offset.ne(offset.shift(1)).fillna(False))[0]
    transition_idx = [i for i in transition_idx if i > 0]
    rows = []
    for i in transition_idx:
        prev = pdf.iloc[i - 1]
        cur = pdf.iloc[i]
        prev_val = prev[vwap_col]
        cur_val = cur[vwap_col]
        pct = np.nan
        if pd.notna(prev_val) and prev_val != 0 and pd.notna(cur_val):
            pct = abs((cur_val - prev_val) / prev_val)
        rows.append(
            {
                "timestamp_utc": cur["timestamp_utc"],
                "timestamp_local": local.iloc[i],
                "utc_offset_h_prev": offset.iloc[i - 1],
                "utc_offset_h_cur": offset.iloc[i],
                "vwap_prev": prev_val,
                "vwap_cur": cur_val,
                "abs_jump": (cur_val - prev_val) if pd.notna(prev_val) and pd.notna(cur_val) else np.nan,
                "pct_jump": pct,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate aFRR VWAP integrity against 15-minute source")
    parser.add_argument(
        "--raw-15m",
        default="data/raw/regelleistung_15min/afrr_price_volume_15min.parquet",
        help="Path to 15-minute joined price/volume parquet",
    )
    parser.add_argument(
        "--hourly",
        default="data/processed/all_data_refined.parquet",
        help="Path to hourly refined parquet containing afrr_vwap_* columns",
    )
    parser.add_argument("--sample-size", type=int, default=5, help="Random sample size for pre/post windows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tol", type=float, default=1e-6, help="Absolute tolerance for manual vs pipeline VWAP")
    args = parser.parse_args()

    raw_path = Path(args.raw_15m)
    hourly_path = Path(args.hourly)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if not hourly_path.exists():
        raise FileNotFoundError(hourly_path)

    raw_15m = _load_parquet(raw_path)
    hourly = _load_parquet(hourly_path)
    cols = _resolve_columns(raw_15m.columns, hourly.columns)
    if cols.vol_pos is None and cols.vol_neg is None:
        raise ValueError("No aFRR volume columns found in 15-minute file.")
    if cols.vwap_pos is None and cols.vwap_neg is None:
        raise ValueError("No hourly afrr_vwap columns found in hourly file.")

    manual = _manual_hourly_vwap(raw_15m, cols)
    join_cols = ["timestamp_utc"]
    if cols.vwap_pos:
        join_cols.append(cols.vwap_pos)
    if cols.vwap_neg:
        join_cols.append(cols.vwap_neg)
    compare = manual.join(hourly.select(join_cols), on="timestamp_utc", how="inner")

    print("=== Column Mapping ===")
    print(cols)
    print()

    pre_sample, post_sample = _sample_hours(compare, sample_size=args.sample_size, seed=args.seed)
    print("=== Random Re-calculation Check (Pre-PICASSO) ===")
    print(pre_sample.select([c for c in ["timestamp_utc", "manual_vwap_pos_eur_mwh", "manual_vwap_neg_eur_mwh", cols.vwap_pos, cols.vwap_neg] if c in pre_sample.columns]))
    print()
    print("=== Random Re-calculation Check (Post-PICASSO) ===")
    print(post_sample.select([c for c in ["timestamp_utc", "manual_vwap_pos_eur_mwh", "manual_vwap_neg_eur_mwh", cols.vwap_pos, cols.vwap_neg] if c in post_sample.columns]))
    print()

    checks = {}
    if cols.vwap_pos:
        pos_diff = compare.with_columns(
            [
                (pl.col("manual_vwap_pos_eur_mwh") - pl.col(cols.vwap_pos)).abs().alias("__abs_diff_pos"),
                pl.col("manual_vwap_pos_eur_mwh").is_finite().alias("__manual_pos_finite"),
                pl.col(cols.vwap_pos).is_finite().alias("__pipe_pos_finite"),
            ]
        )
        checks["manual_match_pos_fail_rows"] = pos_diff.filter(
            pl.col("__manual_pos_finite") & pl.col("__pipe_pos_finite") & (pl.col("__abs_diff_pos") > args.tol)
        ).height
    if cols.vwap_neg:
        neg_diff = compare.with_columns(
            [
                (pl.col("manual_vwap_neg_eur_mwh") - pl.col(cols.vwap_neg)).abs().alias("__abs_diff_neg"),
                pl.col("manual_vwap_neg_eur_mwh").is_finite().alias("__manual_neg_finite"),
                pl.col(cols.vwap_neg).is_finite().alias("__pipe_neg_finite"),
            ]
        )
        checks["manual_match_neg_fail_rows"] = neg_diff.filter(
            pl.col("__manual_neg_finite") & pl.col("__pipe_neg_finite") & (pl.col("__abs_diff_neg") > args.tol)
        ).height

    # Edge-case A: volume > 0 but VWAP is NaN/null.
    if cols.vwap_pos and "__sum_vol_pos" in manual.columns:
        edge_a_pos = compare.filter(
            (pl.col("__sum_vol_pos") > 0.0)
            & (pl.col(cols.vwap_pos).is_null() | pl.col(cols.vwap_pos).is_nan())
        )
        checks["volume_positive_vwap_nan_pos"] = edge_a_pos.height
    else:
        edge_a_pos = pl.DataFrame()

    if cols.vwap_neg and "__sum_vol_neg" in manual.columns:
        edge_a_neg = compare.filter(
            (pl.col("__sum_vol_neg") > 0.0)
            & (pl.col(cols.vwap_neg).is_null() | pl.col(cols.vwap_neg).is_nan())
        )
        checks["volume_positive_vwap_nan_neg"] = edge_a_neg.height
    else:
        edge_a_neg = pl.DataFrame()

    # Edge-case B: VWAP differs from hourly mean of 15-min prices by >500%.
    edge_b_frames: list[pl.DataFrame] = []
    if cols.vwap_pos and "__mean_price_pos" in compare.columns:
        edge_b_pos = compare.with_columns(
            [
                (
                    (pl.col(cols.vwap_pos) - pl.col("__mean_price_pos")).abs()
                    / pl.max_horizontal(pl.col("__mean_price_pos").abs(), pl.lit(1e-9))
                ).alias("__rel_diff_pos"),
                pl.col(cols.vwap_pos).is_finite().alias("__vwap_pos_finite"),
                pl.col("__mean_price_pos").is_finite().alias("__mean_pos_finite"),
            ]
        ).filter(pl.col("__vwap_pos_finite") & pl.col("__mean_pos_finite") & (pl.col("__rel_diff_pos") > 5.0))
        checks["vwap_vs_price_gt_500pct_pos"] = edge_b_pos.height
        if edge_b_pos.height:
            edge_b_frames.append(edge_b_pos.select(["timestamp_utc", cols.vwap_pos, "__mean_price_pos", "__rel_diff_pos"]))

    if cols.vwap_neg and "__mean_price_neg" in compare.columns:
        edge_b_neg = compare.with_columns(
            [
                (
                    (pl.col(cols.vwap_neg) - pl.col("__mean_price_neg")).abs()
                    / pl.max_horizontal(pl.col("__mean_price_neg").abs(), pl.lit(1e-9))
                ).alias("__rel_diff_neg"),
                pl.col(cols.vwap_neg).is_finite().alias("__vwap_neg_finite"),
                pl.col("__mean_price_neg").is_finite().alias("__mean_neg_finite"),
            ]
        ).filter(pl.col("__vwap_neg_finite") & pl.col("__mean_neg_finite") & (pl.col("__rel_diff_neg") > 5.0))
        checks["vwap_vs_price_gt_500pct_neg"] = edge_b_neg.height
        if edge_b_neg.height:
            edge_b_frames.append(edge_b_neg.select(["timestamp_utc", cols.vwap_neg, "__mean_price_neg", "__rel_diff_neg"]))

    # Edge-case C: jump at DST transitions.
    dst_report_pos = pd.DataFrame()
    dst_report_neg = pd.DataFrame()
    if cols.vwap_pos:
        dst_report_pos = _dst_transition_report(hourly, cols.vwap_pos)
    if cols.vwap_neg:
        dst_report_neg = _dst_transition_report(hourly, cols.vwap_neg)

    # Bounded-within-hour forward-fill check (pre-PICASSO at :00 must not be auto-filled from prev hour)
    ffill_bleed_rows = 0
    if cols.price_pos_raw and cols.price_pos_ffill:
        bleed = raw_15m.filter(
            (pl.col("timestamp_utc") < pl.lit(PICASSO_START_UTC).cast(pl.Datetime("us", "UTC")))
            & (pl.col("timestamp_utc").dt.minute() == 0)
            & pl.col(cols.price_pos_raw).is_null()
            & pl.col(cols.price_pos_ffill).is_not_null()
        )
        ffill_bleed_rows += bleed.height
    if cols.price_neg_raw and cols.price_neg_ffill:
        bleed = raw_15m.filter(
            (pl.col("timestamp_utc") < pl.lit(PICASSO_START_UTC).cast(pl.Datetime("us", "UTC")))
            & (pl.col("timestamp_utc").dt.minute() == 0)
            & pl.col(cols.price_neg_raw).is_null()
            & pl.col(cols.price_neg_ffill).is_not_null()
        )
        ffill_bleed_rows += bleed.height
    checks["ffill_cross_hour_bleed_rows"] = ffill_bleed_rows

    print("=== Edge Case Report ===")
    print(f"A) volume>0 but VWAP NaN/null: pos={edge_a_pos.height}, neg={edge_a_neg.height}")
    if edge_a_pos.height:
        print(edge_a_pos.select(["timestamp_utc", "__sum_vol_pos", cols.vwap_pos]).head(20))
    if edge_a_neg.height:
        print(edge_a_neg.select(["timestamp_utc", "__sum_vol_neg", cols.vwap_neg]).head(20))
    print()
    b_total = sum(v for k, v in checks.items() if k.startswith("vwap_vs_price_gt_500pct"))
    print(f"B) VWAP differs from hourly 15m mean price by >500%: rows={b_total}")
    if edge_b_frames:
        print(pl.concat(edge_b_frames, how="diagonal_relaxed").sort("timestamp_utc").head(20))
    print()
    print("C) DST transition jump report (first 10 rows each direction):")
    if not dst_report_pos.empty:
        print("POS:")
        print(dst_report_pos.head(10))
    if not dst_report_neg.empty:
        print("NEG:")
        print(dst_report_neg.head(10))
    print()

    print("=== Rule Audit Summary ===")
    tz_ok = (
        str(raw_15m.schema["timestamp_utc"]).find("time_zone='UTC'") >= 0
        and str(hourly.schema["timestamp_utc"]).find("time_zone='UTC'") >= 0
    )
    print(f"Timezone UTC join integrity: {'PASS' if tz_ok else 'FAIL'}")
    print(
        "Pre-PICASSO ffill hour-bounded: "
        + ("PASS" if checks["ffill_cross_hour_bleed_rows"] == 0 else "FAIL")
        + f" (bleed_rows={checks['ffill_cross_hour_bleed_rows']})"
    )
    for key in sorted(checks):
        if key.startswith("manual_match_"):
            print(f"{key}: {'PASS' if checks[key] == 0 else 'FAIL'} ({checks[key]})")
        if key.startswith("volume_positive_vwap_nan_"):
            print(f"{key}: {'PASS' if checks[key] == 0 else 'FAIL'} ({checks[key]})")
        if key.startswith("vwap_vs_price_gt_500pct_"):
            print(f"{key}: {'PASS' if checks[key] == 0 else 'FAIL'} ({checks[key]})")

    fail_count = sum(1 for k, v in checks.items() if k != "ffill_cross_hour_bleed_rows" and v > 0)
    if not tz_ok:
        fail_count += 1
    if checks["ffill_cross_hour_bleed_rows"] > 0:
        fail_count += 1

    print()
    print("=== Final Verdict ===")
    if fail_count == 0:
        print("SUCCESS: VWAP integrity checks passed.")
    else:
        print(f"FAILURE: {fail_count} integrity check group(s) failed.")


if __name__ == "__main__":
    main()
