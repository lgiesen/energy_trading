"""Audit calculated aFRR VWAP against official Result Overview prices.

Usage:
    ./.venv/bin/python scripts/audit_price_source_vwap.py \
        --overview-dir data/raw/marginal_prices \
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


def _read_result_overview_hourly(overview_dir: Path) -> pl.DataFrame:
    files = sorted(overview_dir.glob("RESULT_OVERVIEW_ENERGY_MARKET_aFRR_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No aFRR overview files in {overview_dir}")

    rows: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_excel(path, engine="openpyxl")
        if df.empty:
            continue

        cols = [str(c).strip() for c in df.columns]
        df.columns = cols
        if "PRODUCT" not in df.columns or "DELIVERY_DATE" not in df.columns:
            continue
        avg_col = "GERMANY_AVERAGE_ENERGY_PRICE_[EUR/MWh]"
        if avg_col not in df.columns:
            continue

        if "TYPE_OF_RESERVES" in df.columns:
            df = df[df["TYPE_OF_RESERVES"].astype(str).str.upper().eq("AFRR")]
        if df.empty:
            continue

        df = df[["DELIVERY_DATE", "PRODUCT", avg_col]].copy()
        df["official_avg_price"] = pd.to_numeric(df[avg_col], errors="coerce")
        df = df.dropna(subset=["DELIVERY_DATE", "PRODUCT", "official_avg_price"])

        prod = df["PRODUCT"].astype(str)
        df["direction"] = prod.str.split("_").str[0].str.upper()
        df = df[df["direction"].isin(["POS", "NEG"])]

        qh = pd.to_numeric(prod.str.extract(r"_(\d{3})$")[0], errors="coerce")
        sh = pd.to_numeric(prod.str.extract(r"_(\d{2})_(\d{2})$")[0], errors="coerce")
        eh = pd.to_numeric(prod.str.extract(r"_(\d{2})_(\d{2})$")[1], errors="coerce")

        mask_qh = qh.notna()
        mask_block = sh.notna() & eh.notna()
        parts: list[pd.DataFrame] = []

        if mask_qh.any():
            q = df.loc[mask_qh].copy()
            base_local = pd.to_datetime(q["DELIVERY_DATE"]).dt.floor("D").dt.tz_localize(
                "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"
            )
            q["timestamp_utc"] = (
                base_local + pd.to_timedelta((qh.loc[mask_qh].astype(int) - 1) * 15, unit="m")
            ).dt.tz_convert("UTC")
            parts.append(q[["timestamp_utc", "direction", "official_avg_price"]])

        if mask_block.any():
            b = df.loc[mask_block].copy()
            base_local = pd.to_datetime(b["DELIVERY_DATE"]).dt.floor("D").dt.tz_localize(
                "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"
            )
            rows_block: list[tuple[pd.Timestamp, str, float]] = []
            for ts0, d, p, s_h, e_h in zip(
                base_local,
                b["direction"],
                b["official_avg_price"],
                sh.loc[mask_block].astype(int),
                eh.loc[mask_block].astype(int),
            ):
                end_adj = e_h if e_h > s_h else e_h + 24
                for h in range(s_h, end_adj):
                    ts_local = ts0 + pd.Timedelta(hours=int(h))
                    rows_block.append((ts_local.tz_convert("UTC"), d, float(p)))
            if rows_block:
                parts.append(pd.DataFrame(rows_block, columns=["timestamp_utc", "direction", "official_avg_price"]))

        if parts:
            rows.append(pd.concat(parts, ignore_index=True))

    if not rows:
        raise ValueError("Could not parse any official aFRR overview rows.")

    all_df = pd.concat(rows, ignore_index=True)
    all_df["timestamp_utc"] = pd.to_datetime(all_df["timestamp_utc"], utc=True, errors="coerce")
    all_df = all_df.dropna(subset=["timestamp_utc"])

    # Convert to hourly official average per direction.
    all_df["hour_utc"] = all_df["timestamp_utc"].dt.floor("1h")
    hourly = (
        all_df.groupby(["hour_utc", "direction"], as_index=False)["official_avg_price"]
        .mean()
        .pivot(index="hour_utc", columns="direction", values="official_avg_price")
        .rename(columns={"POS": "official_avg_price_pos", "NEG": "official_avg_price_neg"})
        .reset_index()
        .rename(columns={"hour_utc": "timestamp_utc"})
        .sort_values("timestamp_utc")
    )
    return pl.from_pandas(hourly).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)
    )


def _calc_hourly_vwap_from_15m(path_15m: Path) -> pl.DataFrame:
    df = pl.read_parquet(path_15m).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)
    ).sort("timestamp_utc")

    price_pos = "afrr_avg_activation_price_pos"
    price_neg = "afrr_avg_activation_price_neg"
    vol_pos = "afrr_activated_mw_pos"
    vol_neg = "afrr_activated_mw_neg"
    needed = [c for c in [price_pos, price_neg, vol_pos, vol_neg] if c in df.columns]
    if len(needed) < 4:
        missing = [c for c in [price_pos, price_neg, vol_pos, vol_neg] if c not in df.columns]
        raise ValueError(f"Missing columns in 15m parquet: {missing}")

    # Pre-PICASSO: fill within each hour.
    df = df.with_columns(
        [
            pl.when(pl.col("timestamp_utc") < pl.lit(PICASSO_START))
            .then(pl.col(price_pos).cast(pl.Float64, strict=False).forward_fill().over(pl.col("timestamp_utc").dt.truncate("1h")))
            .otherwise(pl.col(price_pos).cast(pl.Float64, strict=False))
            .alias("__price_pos"),
            pl.when(pl.col("timestamp_utc") < pl.lit(PICASSO_START))
            .then(pl.col(price_neg).cast(pl.Float64, strict=False).forward_fill().over(pl.col("timestamp_utc").dt.truncate("1h")))
            .otherwise(pl.col(price_neg).cast(pl.Float64, strict=False))
            .alias("__price_neg"),
            pl.col(vol_pos).cast(pl.Float64, strict=False).alias("__vol_pos"),
            pl.col(vol_neg).cast(pl.Float64, strict=False).alias("__vol_neg"),
        ]
    )

    hourly = (
        df.with_columns(
            [
                (pl.col("__price_pos") * pl.col("__vol_pos")).alias("__wc_pos"),
                (pl.col("__price_neg") * pl.col("__vol_neg")).alias("__wc_neg"),
            ]
        )
        .group_by_dynamic(
            index_column="timestamp_utc",
            every="1h",
            period="1h",
            closed="left",
            label="left",
        )
        .agg(
            [
                pl.col("__wc_pos").sum().alias("sum_wc_pos"),
                pl.col("__vol_pos").sum().alias("sum_vol_pos"),
                pl.col("__wc_neg").sum().alias("sum_wc_neg"),
                pl.col("__vol_neg").sum().alias("sum_vol_neg"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sum_vol_pos") == 0)
                .then(pl.lit(float("nan")))
                .otherwise(pl.col("sum_wc_pos") / pl.col("sum_vol_pos"))
                .alias("calc_vwap_pos"),
                pl.when(pl.col("sum_vol_neg") == 0)
                .then(pl.lit(float("nan")))
                .otherwise(pl.col("sum_wc_neg") / pl.col("sum_vol_neg"))
                .alias("calc_vwap_neg"),
            ]
        )
        .select(["timestamp_utc", "calc_vwap_pos", "calc_vwap_neg"])
        .sort("timestamp_utc")
    )
    return hourly


def _series_stats(pdf: pd.DataFrame, col: str) -> dict[str, float]:
    s = pd.to_numeric(pdf[col], errors="coerce").dropna()
    return {
        "count": int(s.shape[0]),
        "mean": float(s.mean()) if len(s) else np.nan,
        "median": float(s.median()) if len(s) else np.nan,
        "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        "min": float(s.min()) if len(s) else np.nan,
        "max": float(s.max()) if len(s) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit official hourly average prices vs calculated 15m VWAP.")
    parser.add_argument("--overview-dir", default="data/raw/marginal_prices")
    parser.add_argument("--raw-15m", default="data/raw/regelleistung_15min/afrr_price_volume_15min.parquet")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    overview_dir = Path(args.overview_dir)
    raw_15m = Path(args.raw_15m)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    official = _read_result_overview_hourly(overview_dir)
    calc = _calc_hourly_vwap_from_15m(raw_15m)

    joined = official.join(calc, on="timestamp_utc", how="inner")

    long = pl.concat(
        [
            joined.select(
                [
                    "timestamp_utc",
                    pl.lit("pos").alias("direction"),
                    pl.col("official_avg_price_pos").alias("official_price"),
                    pl.col("calc_vwap_pos").alias("calc_vwap"),
                ]
            ),
            joined.select(
                [
                    "timestamp_utc",
                    pl.lit("neg").alias("direction"),
                    pl.col("official_avg_price_neg").alias("official_price"),
                    pl.col("calc_vwap_neg").alias("calc_vwap"),
                ]
            ),
        ],
        how="vertical",
    ).drop_nulls(["official_price", "calc_vwap"])

    long = long.with_columns(
        [
            (pl.col("calc_vwap") - pl.col("official_price")).abs().alias("abs_deviation"),
            pl.when(pl.col("official_price").abs() > 1e-9)
            .then((pl.col("calc_vwap") - pl.col("official_price")).abs() / pl.col("official_price").abs())
            .otherwise(pl.lit(float("nan")))
            .alias("deviation_pct"),
        ]
    ).with_columns((pl.col("deviation_pct") > 0.10).alias("flag_gt_10pct"))

    top20 = long.sort("deviation_pct", descending=True).head(20)
    report_csv = out_dir / "price_source_audit.csv"
    top20.write_csv(report_csv)

    # stats
    pdf = long.to_pandas()
    stats = {
        "official_stats": _series_stats(pdf, "official_price"),
        "calculated_vwap_stats": _series_stats(pdf, "calc_vwap"),
        "rows_compared": int(len(pdf)),
        "share_flag_gt_10pct": float(np.nanmean(pdf["deviation_pct"] > 0.10)) if len(pdf) else np.nan,
        "corr_pearson": float(pd.Series(pdf["official_price"]).corr(pd.Series(pdf["calc_vwap"]), method="pearson"))
        if len(pdf) > 2
        else np.nan,
    }

    # scatter
    fig = plt.figure(figsize=(8, 5))
    for d, c in [("pos", "#1f77b4"), ("neg", "#d62728")]:
        sub = pdf[pdf["direction"] == d]
        plt.scatter(sub["official_price"], sub["calc_vwap"], s=8, alpha=0.25, label=d, c=c)
    lims = [
        np.nanmin([pdf["official_price"].min(), pdf["calc_vwap"].min()]),
        np.nanmax([pdf["official_price"].max(), pdf["calc_vwap"].max()]),
    ]
    if np.isfinite(lims).all():
        plt.plot(lims, lims, "--", linewidth=1, color="gray")
    plt.xlabel("Official GERMANY_AVERAGE_ENERGY_PRICE [EUR/MWh]")
    plt.ylabel("Calculated VWAP [EUR/MWh]")
    plt.title("Official Hourly Avg vs Calculated VWAP")
    plt.legend()
    plt.grid(alpha=0.25)
    fig.tight_layout()
    scatter_path = out_dir / "price_source_audit_scatter.png"
    fig.savefig(scatter_path, dpi=160)
    plt.close(fig)

    reliability = "reliable_proxy" if stats["share_flag_gt_10pct"] <= 0.10 else "weak_proxy"

    print("=== Price Source Audit ===")
    print(f"official_rows: {official.height}, calculated_rows: {calc.height}, joined_rows: {joined.height}")
    print(f"top20_report: {report_csv}")
    print(f"scatter_plot: {scatter_path}")
    print(f"official_stats: {stats['official_stats']}")
    print(f"calculated_vwap_stats: {stats['calculated_vwap_stats']}")
    print(f"share_flag_gt_10pct: {stats['share_flag_gt_10pct']}")
    print(f"corr_pearson: {stats['corr_pearson']}")
    print(f"proxy_assessment: {reliability}")


if __name__ == "__main__":
    main()
