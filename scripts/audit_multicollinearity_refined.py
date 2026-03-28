"""Audit multicollinearity on refined feature table.

Usage:
    ./.venv/bin/python scripts/audit_multicollinearity_refined.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


def _load_numeric_df(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_pl = pl.read_parquet(path)
    df = df_pl.to_pandas()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found for correlation/VIF audit.")
    return df, df[numeric_cols].copy()


def _correlation_pairs(
    numeric_df: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = numeric_df.corr(method="pearson")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    pairs = (
        corr.where(mask)
        .stack()
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "pearson_r"})
    )
    pairs["abs_r"] = pairs["pearson_r"].abs()
    pairs = pairs.sort_values("abs_r", ascending=False).reset_index(drop=True)
    high_pairs = pairs[pairs["abs_r"] > threshold].reset_index(drop=True)
    return corr, high_pairs


def _compute_vif(numeric_df: pd.DataFrame) -> pd.DataFrame:
    work = numeric_df.copy()
    # Drop constant/all-null columns for stable VIF.
    nunique = work.nunique(dropna=True)
    keep = nunique[nunique > 1].index.tolist()
    work = work[keep]
    if work.empty:
        return pd.DataFrame(columns=["feature", "vif"])

    # Median-impute to avoid dropping too many rows due to sparse gaps.
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.fillna(work.median(numeric_only=True))
    work = work.dropna(axis=1, how="all")

    vals = work.values.astype(float)
    vifs = []
    for i, feature in enumerate(work.columns):
        try:
            v = variance_inflation_factor(vals, i)
        except Exception:
            v = np.inf
        vifs.append((feature, float(v)))
    vif_df = pd.DataFrame(vifs, columns=["feature", "vif"]).sort_values("vif", ascending=False).reset_index(drop=True)
    return vif_df


def _top_heatmap(corr: pd.DataFrame, top_n: int, out_path: Path) -> None:
    abs_corr = corr.abs().copy()
    arr = abs_corr.to_numpy(copy=True)
    np.fill_diagonal(arr, 0.0)
    abs_corr = pd.DataFrame(arr, index=abs_corr.index, columns=abs_corr.columns)
    # Pick features with largest average absolute correlation.
    top_features = abs_corr.mean(axis=1).sort_values(ascending=False).head(top_n).index.tolist()
    if len(top_features) < 2:
        return
    plot_corr = corr.loc[top_features, top_features]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        plot_corr,
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.2,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(f"Top-{len(top_features)} Correlated Features (Pearson)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _recommendations(
    df_all: pd.DataFrame,
    high_pairs: pd.DataFrame,
    high_vif: pd.DataFrame,
) -> pd.DataFrame:
    if high_vif.empty:
        return pd.DataFrame(columns=["feature", "vif", "strongest_pair_abs_r", "suggested_action", "reason"])

    missing_pct = (df_all.isna().mean() * 100.0).to_dict()
    pair_best = {}
    for _, r in high_pairs.iterrows():
        a, b, abs_r = r["feature_a"], r["feature_b"], float(r["abs_r"])
        pair_best[a] = max(pair_best.get(a, 0.0), abs_r)
        pair_best[b] = max(pair_best.get(b, 0.0), abs_r)

    rows = []
    for _, r in high_vif.iterrows():
        f = r["feature"]
        vif = float(r["vif"])
        strongest = float(pair_best.get(f, 0.0))

        # Heuristic recommendation for linear models:
        # If very high VIF and tightly correlated with others -> drop/merge candidate.
        if strongest >= 0.95:
            action = "drop_or_merge"
            reason = "Very high pairwise correlation and VIF; keep canonical representative."
        elif strongest >= 0.85:
            action = "merge_or_regularize"
            reason = "High pairwise correlation with elevated VIF; consider aggregation."
        else:
            action = "keep_with_monitoring"
            reason = "VIF high but pairwise structure less concentrated; validate in CV."

        rows.append(
            {
                "feature": f,
                "vif": vif,
                "strongest_pair_abs_r": strongest,
                "missing_pct": float(missing_pct.get(f, 0.0)),
                "suggested_action": action,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows).sort_values(["vif", "strongest_pair_abs_r"], ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multicollinearity audit for refined feature set.")
    parser.add_argument("--input", "--in", dest="input", default="data/processed/all_data_pruned.parquet", help="Input parquet path.")
    parser.add_argument("--corr-threshold", type=float, default=0.85, help="Absolute Pearson threshold.")
    parser.add_argument("--vif-threshold", type=float, default=10.0, help="VIF threshold.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of features for heatmap.")
    parser.add_argument(
        "--out-dir",
        default="data/reports/processed_audits",
        help="Output directory for reports.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all, numeric_df = _load_numeric_df(in_path)
    corr, high_pairs = _correlation_pairs(numeric_df, args.corr_threshold)
    vif_df = _compute_vif(numeric_df)
    high_vif = vif_df[vif_df["vif"] > args.vif_threshold].reset_index(drop=True)
    recs = _recommendations(df_all, high_pairs, high_vif)

    high_pairs_path = out_dir / "multicollinearity_high_corr_pairs.csv"
    high_vif_path = out_dir / "multicollinearity_high_vif.csv"
    recs_path = out_dir / "multicollinearity_recommendations.csv"
    heatmap_path = out_dir / "multicollinearity_top20_heatmap.png"

    high_pairs.to_csv(high_pairs_path, index=False)
    high_vif.to_csv(high_vif_path, index=False)
    recs.to_csv(recs_path, index=False)
    _top_heatmap(corr, args.top_n, heatmap_path)

    print(f"Input: {in_path}")
    print(f"Numeric features: {numeric_df.shape[1]}")
    print(f"High-correlation pairs (|r|>{args.corr_threshold}): {len(high_pairs)}")
    print(f"High-VIF features (VIF>{args.vif_threshold}): {len(high_vif)}")
    print(f"Saved: {high_pairs_path}")
    print(f"Saved: {high_vif_path}")
    print(f"Saved: {recs_path}")
    print(f"Saved: {heatmap_path}")

    if len(high_vif):
        print("\\nTop VIF offenders:")
        print(high_vif.head(20).to_string(index=False))
        print("\\nSuggested stabilization actions:")
        print(recs.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
