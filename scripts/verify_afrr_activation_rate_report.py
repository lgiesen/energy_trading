"""Deep-dive verification report for aFRR VWAP and activation rate.

Usage:
    ./.venv/bin/python scripts/verify_afrr_activation_rate_report.py \
        --in data/processed/all_data_refined.parquet \
        --raw-15m data/raw/regelleistung_15min/afrr_price_volume_15min.parquet \
        --out-dir reports
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


PICASSO_START = pd.Timestamp("2022-06-22T22:00:00Z")


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _rate_stats(df: pl.DataFrame, col: str) -> dict[str, float]:
    out = (
        df.select(pl.col(col).cast(pl.Float64, strict=False).alias("r"))
        .drop_nulls()
        .filter(pl.col("r").is_finite())
        .select(
            [
                pl.len().alias("n"),
                pl.col("r").min().alias("min"),
                pl.col("r").mean().alias("mean"),
                pl.col("r").median().alias("median"),
                pl.col("r").max().alias("max"),
                (pl.col("r") > 1.0).mean().alias("share_gt_1"),
                (pl.col("r") > 2.0).mean().alias("share_gt_2"),
                (pl.col("r") < 0.25).mean().alias("share_lt_0_25"),
            ]
        )
        .to_dicts()[0]
    )
    out["column"] = col
    return out


def _compute_15m_work_price(df_15m: pl.DataFrame, price_raw_col: str) -> pl.DataFrame:
    """Add hour-bounded pre-PICASSO ffill work-price column for POS."""
    return df_15m.with_columns(
        [
            pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False),
            pl.when(pl.col("timestamp_utc") < pl.lit(PICASSO_START))
            .then(
                pl.col(price_raw_col)
                .cast(pl.Float64, strict=False)
                .forward_fill()
                .over(pl.col("timestamp_utc").dt.truncate("1h"))
            )
            .otherwise(pl.col(price_raw_col).cast(pl.Float64, strict=False))
            .alias("__price_pos_work"),
        ]
    )


def _detect_tz_status_from_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {"cache_exists": "false"}
    try:
        c = pd.read_parquet(path)
    except Exception as exc:
        return {"cache_exists": "true", "cache_read_error": str(exc)}
    if "timestamp" not in c.columns:
        return {"cache_exists": "true", "cache_has_timestamp": "false"}
    ts = pd.to_datetime(c["timestamp"], errors="coerce")
    tz = getattr(ts.dt, "tz", None)
    if tz is None:
        return {
            "cache_exists": "true",
            "cache_has_timestamp": "true",
            "cache_timestamp_tz": "naive",
            "applied_conversion": "localized_europe_berlin_then_utc",
        }
    tzname = str(tz)
    if "UTC" in tzname.upper():
        return {
            "cache_exists": "true",
            "cache_has_timestamp": "true",
            "cache_timestamp_tz": tzname,
            "applied_conversion": "already_utc",
        }
    return {
        "cache_exists": "true",
        "cache_has_timestamp": "true",
        "cache_timestamp_tz": tzname,
        "applied_conversion": "aware_to_utc",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep verification for aFRR VWAP and activation rates.")
    parser.add_argument("--in", dest="input_path", default="data/processed/all_data_refined.parquet")
    parser.add_argument("--raw-15m", dest="raw_15m_path", default="data/raw/regelleistung_15min/afrr_price_volume_15min.parquet")
    parser.add_argument("--bids-cache", dest="bids_cache_path", default="data/raw/bids/_afrr_bid_hourly_cache.parquet")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hourly_path = Path(args.input_path)
    raw_15m_path = Path(args.raw_15m_path)
    bids_cache_path = Path(args.bids_cache_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not hourly_path.exists():
        raise FileNotFoundError(hourly_path)
    if not raw_15m_path.exists():
        raise FileNotFoundError(raw_15m_path)

    df = pl.read_parquet(hourly_path).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)
    ).sort("timestamp_utc")

    df_15m = pl.read_parquet(raw_15m_path).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)
    ).sort("timestamp_utc")

    cap_col = _first_existing(
        df.columns,
        ["afrr_capacity_awarded_mw_pos", "awarded_capacity_mw_pos", "afrr_capacity_offered_mw_pos"],
    )
    if cap_col is None:
        raise ValueError(
            "Missing denominator column: expected afrr_capacity_awarded_mw_pos or awarded_capacity_mw_pos"
        )

    # -------- 1) VWAP Extreme-Hour logic check --------
    price_raw_col = _first_existing(
        df_15m.columns,
        ["afrr_avg_activation_price_pos", "afrr_activation_avg_price_pos", "arbeitspreis_pos"],
    )
    vol_col = _first_existing(
        df_15m.columns,
        ["afrr_activated_mw_pos", "activated_volume_pos_mw", "abgerufene_arbeit_pos"],
    )
    official_avg_col = _first_existing(
        df_15m.columns,
        ["durchschnittlicher_arbeitspreis_pos", "afrr_avg_activation_price_pos", "afrr_activation_avg_price_pos"],
    )
    if price_raw_col is None or vol_col is None:
        raise ValueError("Missing 15-min POS price/volume columns in raw parquet.")

    work_15m = _compute_15m_work_price(df_15m, price_raw_col).with_columns(
        [
            pl.col(vol_col).cast(pl.Float64, strict=False).alias("__vol_pos"),
            pl.col("timestamp_utc").dt.truncate("1h").alias("__hour_utc"),
            (pl.col("__price_pos_work") * pl.col(vol_col).cast(pl.Float64, strict=False)).alias("__wc_pos"),
            pl.col(price_raw_col).cast(pl.Float64, strict=False).alias("price_15m_pos_raw"),
        ]
    )
    if official_avg_col is not None:
        work_15m = work_15m.with_columns(
            pl.col(official_avg_col).cast(pl.Float64, strict=False).alias("official_avg_price_15m")
        )
    else:
        work_15m = work_15m.with_columns(pl.lit(None).cast(pl.Float64).alias("official_avg_price_15m"))

    top3 = (
        df.select(["timestamp_utc", "afrr_activation_price_vwap_pos"])
        .drop_nulls()
        .filter(pl.col("afrr_activation_price_vwap_pos").is_finite())
        .sort("afrr_activation_price_vwap_pos", descending=True)
        .head(3)
    )

    extreme_tables: list[pl.DataFrame] = []
    extreme_summary_rows = []
    for row in top3.to_dicts():
        h = row["timestamp_utc"]
        block = (
            work_15m.filter(
                (pl.col("timestamp_utc") >= pl.lit(h))
                & (pl.col("timestamp_utc") < pl.lit(h + pd.Timedelta(hours=1)))
            )
            .select(
                [
                    "timestamp_utc",
                    "price_15m_pos_raw",
                    "__price_pos_work",
                    "__vol_pos",
                    "__wc_pos",
                    "official_avg_price_15m",
                ]
            )
            .sort("timestamp_utc")
        )
        extreme_tables.append(block.with_columns(pl.lit(h).alias("hour_utc")))
        vals = block.select(
            [
                pl.col("__wc_pos").sum().alias("sum_wc"),
                pl.col("__vol_pos").sum().alias("sum_vol"),
                pl.col("price_15m_pos_raw").mean().alias("simple_mean_15m"),
                pl.col("official_avg_price_15m").mean().alias("official_avg_hour"),
            ]
        ).to_dicts()[0]
        manual_vwap = np.nan if vals["sum_vol"] in (None, 0.0) else vals["sum_wc"] / vals["sum_vol"]
        extreme_summary_rows.append(
            {
                "timestamp_utc": h,
                "manual_vwap_from_15m": manual_vwap,
                "pipeline_hourly_vwap": row["afrr_activation_price_vwap_pos"],
                "simple_mean_15m_price": vals["simple_mean_15m"],
                "official_avg_price_hour": vals["official_avg_hour"],
            }
        )

    extreme_breakdown = (
        pl.concat(extreme_tables, how="vertical")
        if extreme_tables
        else pl.DataFrame()
    )
    extreme_summary = pl.from_pandas(pd.DataFrame(extreme_summary_rows))
    extreme_breakdown_path = out_dir / "afrr_vwap_extreme_hours_15m_breakdown.csv"
    extreme_summary_path = out_dir / "afrr_vwap_extreme_hours_summary.csv"
    extreme_breakdown.write_csv(extreme_breakdown_path)
    extreme_summary.write_csv(extreme_summary_path)

    # -------- 2) Activation-rate scarcity quantile check --------
    qr = (
        df.select(
            [
                "timestamp_utc",
                pl.col("afrr_activation_rate_pos").cast(pl.Float64, strict=False).alias("rate"),
                pl.col("afrr_activation_price_vwap_pos").cast(pl.Float64, strict=False).alias("vwap"),
            ]
        )
        .drop_nulls()
        .filter(pl.col("rate").is_finite() & pl.col("vwap").is_finite())
        .to_pandas()
    )
    qr["rate_decile"] = pd.qcut(qr["rate"], 10, labels=False, duplicates="drop") + 1
    decile = (
        qr.groupby("rate_decile", as_index=False)
        .agg(
            n=("vwap", "size"),
            median_rate=("rate", "median"),
            median_vwap_pos_eur_mwh=("vwap", "median"),
            mean_vwap_pos_eur_mwh=("vwap", "mean"),
        )
        .sort_values("rate_decile")
    )
    decile_path = out_dir / "afrr_activation_rate_pos_deciles.csv"
    decile.to_csv(decile_path, index=False)

    # -------- 3) Physical unit factor-4 audit (random 5 rows) --------
    f4 = (
        df.select(
            [
                "timestamp_utc",
                pl.col("afrr_activated_mw_pos").cast(pl.Float64, strict=False).alias("afrr_activated_mw_pos"),
                pl.col("rz_saldo_mw").cast(pl.Float64, strict=False).alias("rz_saldo_mw"),
            ]
        )
        .drop_nulls()
        .filter((pl.col("afrr_activated_mw_pos") > 0.0) & pl.col("rz_saldo_mw").is_finite())
        .with_columns(
            [
                (pl.col("rz_saldo_mw") / pl.col("afrr_activated_mw_pos")).alias("rz_over_activated"),
                (pl.col("rz_saldo_mw").abs() / pl.col("afrr_activated_mw_pos")).alias("abs_rz_over_activated"),
            ]
        )
        .sample(n=min(5, max(1, df.height // 1000)), with_replacement=False, seed=args.seed)
        .sort("timestamp_utc")
    )
    f4_path = out_dir / "afrr_factor4_random_audit.csv"
    f4.write_csv(f4_path)

    # -------- 4) Top-10 sanity table --------
    sanity = (
        df.select(
            [
                "timestamp_utc",
                pl.col("afrr_activated_mw_pos").cast(pl.Float64, strict=False),
                pl.col(cap_col).cast(pl.Float64, strict=False).alias("afrr_capacity_awarded_mw_pos"),
                pl.col("afrr_activation_rate_pos").cast(pl.Float64, strict=False),
                pl.col("afrr_activation_price_vwap_pos").cast(pl.Float64, strict=False),
            ]
        )
        .drop_nulls(["afrr_activation_price_vwap_pos"])
        .sort("afrr_activation_price_vwap_pos", descending=True)
        .head(10)
    )
    sanity_path = out_dir / "afrr_sanity_top10_expensive_hours.csv"
    sanity.write_csv(sanity_path)

    # -------- 4b) Provider-to-grid hours check --------
    provider_col = _first_existing(df.columns, ["bid_provider_to_grid_share_pos"])
    provider_top5_path = out_dir / "afrr_provider_to_grid_top5.csv"
    provider_warning_rows = 0
    if provider_col is not None:
        p2g = (
            df.select(
                [
                    "timestamp_utc",
                    pl.col(provider_col).cast(pl.Float64, strict=False).alias("provider_to_grid_share_pos"),
                    pl.col("afrr_activation_price_vwap_pos").cast(pl.Float64, strict=False).alias("afrr_activation_price_vwap_pos"),
                    pl.col("afrr_activation_rate_pos").cast(pl.Float64, strict=False).alias("afrr_activation_rate_pos"),
                ]
            )
            .drop_nulls(["provider_to_grid_share_pos", "afrr_activation_price_vwap_pos"])
            .filter(pl.col("provider_to_grid_share_pos") > 0.0)
            .sort("provider_to_grid_share_pos", descending=True)
            .head(5)
        )
        p2g.write_csv(provider_top5_path)
        provider_warning_rows = p2g.filter(pl.col("afrr_activation_price_vwap_pos") > 0.0).height
    else:
        pl.DataFrame(
            {
                "info": [
                    "No bid_provider_to_grid_share_pos column found; cannot run provider_to_grid top-5 check."
                ]
            }
        ).write_csv(provider_top5_path)

    # -------- Physical impossibility flags --------
    impossible = (
        df.select(
            [
                "timestamp_utc",
                pl.col("afrr_activated_mw_pos").cast(pl.Float64, strict=False),
                pl.col("afrr_activated_mw_neg").cast(pl.Float64, strict=False),
                pl.col("rz_saldo_mw").cast(pl.Float64, strict=False),
            ]
        )
        .drop_nulls()
        .filter(
            ((pl.col("afrr_activated_mw_pos") > 0.0) & (pl.col("rz_saldo_mw") < -200.0))
            | ((pl.col("afrr_activated_mw_neg") > 0.0) & (pl.col("rz_saldo_mw") > 200.0))
        )
        .sort("timestamp_utc")
    )
    impossible_path = out_dir / "afrr_physical_impossibilities.csv"
    impossible.write_csv(impossible_path)

    # -------- Existing checks (distribution/correlation/scatter) --------
    rate_pos = _rate_stats(df, "afrr_activation_rate_pos")
    rate_neg = _rate_stats(df, "afrr_activation_rate_neg")

    cv = (
        df.select(
            [
                pl.col("afrr_activation_rate_pos").cast(pl.Float64, strict=False).alias("rate"),
                pl.col("afrr_activation_price_vwap_pos").cast(pl.Float64, strict=False).alias("vwap"),
            ]
        )
        .drop_nulls()
        .filter(pl.col("rate").is_finite() & pl.col("vwap").is_finite())
    )
    pearson = cv.select(pl.corr("rate", "vwap")).item()
    cv_pd = cv.to_pandas()
    spearman = cv_pd["rate"].corr(cv_pd["vwap"], method="spearman")

    plot_df = cv_pd.copy()
    q_rate = plot_df["rate"].quantile(0.995)
    q_vwap = plot_df["vwap"].quantile(0.995)
    plot_df = plot_df[(plot_df["rate"] <= q_rate) & (plot_df["vwap"] <= q_vwap)]
    fig = plt.figure(figsize=(8, 5))
    plt.scatter(plot_df["rate"], plot_df["vwap"], s=6, alpha=0.25)
    plt.xlabel("afrr_activation_rate_pos")
    plt.ylabel("afrr_activation_price_vwap_pos")
    plt.title("aFRR Activation Rate vs VWAP (POS)")
    plt.grid(alpha=0.25)
    fig.tight_layout()
    scatter_path = out_dir / "afrr_activation_rate_vs_vwap_scatter.png"
    fig.savefig(scatter_path, dpi=160)
    plt.close(fig)

    # Short summary.
    decile_low = float(decile["median_vwap_pos_eur_mwh"].iloc[0]) if not decile.empty else np.nan
    decile_high = float(decile["median_vwap_pos_eur_mwh"].iloc[-1]) if not decile.empty else np.nan
    summary = {
        "timezone_status": {
            "hourly_timestamp_schema": str(df.schema.get("timestamp_utc")),
            "raw15m_timestamp_schema": str(df_15m.schema.get("timestamp_utc")),
            **_detect_tz_status_from_cache(bids_cache_path),
        },
        "extreme_hours_breakdown_csv": str(extreme_breakdown_path),
        "extreme_hours_summary_csv": str(extreme_summary_path),
        "decile_csv": str(decile_path),
        "factor4_random_audit_csv": str(f4_path),
        "top10_sanity_csv": str(sanity_path),
        "provider_to_grid_top5_csv": str(provider_top5_path),
        "physical_impossibilities_csv": str(impossible_path),
        "scatter_plot": str(scatter_path),
        "rate_stats_pos": rate_pos,
        "rate_stats_neg": rate_neg,
        "correlation_pos_vs_vwap": {
            "pearson": None if pearson is None else float(pearson),
            "spearman": None if spearman is None else float(spearman),
            "n": int(len(cv_pd)),
            "expectation_strong_positive_gt_0_4": bool(pearson is not None and float(pearson) > 0.4),
        },
        "decile_price_lift": {
            "median_vwap_decile_1": decile_low,
            "median_vwap_decile_10": decile_high,
            "ratio_d10_over_d1": (decile_high / decile_low) if np.isfinite(decile_low) and decile_low != 0 else np.nan,
        },
        "physical_impossibility_rows": int(impossible.height),
        "provider_to_grid_positive_vwap_warning_rows": int(provider_warning_rows),
    }

    print("=== aFRR Activation-Rate & VWAP Sanity Report ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("Done.")


if __name__ == "__main__":
    main()
