#!/usr/bin/env python3
"""Validate statistical realism of feature artifacts.

This audit script profiles numeric columns, flags suspicious values, and prints
clipping recommendations for heavy-tailed features.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_repo_root() -> Path:
    root = Path.cwd().resolve()
    if (root / "src").exists():
        return root
    for parent in root.parents:
        if (parent / "src").exists():
            return parent
    raise RuntimeError("Could not resolve REPO_ROOT (directory containing 'src').")


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Avoid matplotlib cache permission issues in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str((REPO_ROOT / "data" / ".mplconfig").resolve()))


def _load_style() -> tuple[bool, object | None]:
    try:
        from src.energy_trading.visualization.style import apply_geo_style  # type: ignore

        apply_geo_style()
        return True, apply_geo_style
    except Exception:
        return False, None


def _numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    if num.empty:
        return pd.DataFrame()

    desc = num.describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["NaN_count"] = num.isna().sum()
    desc["p99"] = num.quantile(0.99)
    desc["n_unique"] = num.nunique(dropna=True)
    desc["IQR"] = desc["75%"] - desc["25%"]
    desc = desc.rename_axis("column").reset_index()
    return desc


def _series_extreme_iqr_count(s: pd.Series, median: float, iqr: float) -> int:
    if not np.isfinite(iqr) or iqr <= 0:
        return 0
    x = pd.to_numeric(s, errors="coerce")
    m = pd.to_numeric(pd.Series([median]), errors="coerce").iloc[0]
    return int((x.sub(m).abs() > (10.0 * iqr)).sum())


def _lag_base_name(col: str) -> str | None:
    m = re.match(r"^(.*)_lag_(\d+)h$", col)
    if not m:
        return None
    return m.group(1)


def _distribution_shift_flag(base: pd.Series, lag: pd.Series) -> tuple[bool, str]:
    """Flag suspicious lag-parent distribution mismatch.

    We treat large relative differences in median / std / IQR as suspicious.
    This is intentionally heuristic and conservative to catch shift errors.
    """
    b = pd.to_numeric(base, errors="coerce")
    l = pd.to_numeric(lag, errors="coerce")
    b_med, l_med = b.median(), l.median()
    b_std, l_std = b.std(), l.std()
    b_iqr = b.quantile(0.75) - b.quantile(0.25)
    l_iqr = l.quantile(0.75) - l.quantile(0.25)

    def rel(a: float, b_: float) -> float:
        return float(abs(a - b_) / (abs(a) + 1e-9))

    med_rel = rel(float(b_med), float(l_med))
    std_rel = rel(float(b_std), float(l_std))
    iqr_rel = rel(float(b_iqr), float(l_iqr))
    is_shift = (med_rel > 0.5) or (std_rel > 0.5) or (iqr_rel > 0.5)
    reason = f"lag_dist_shift(med={med_rel:.2f}, std={std_rel:.2f}, iqr={iqr_rel:.2f})"
    return bool(is_shift), reason


def _flag_columns(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    flags = []
    num = df.select_dtypes(include=[np.number]).copy()

    for _, row in stats.iterrows():
        col = row["column"]
        s = num[col]
        col_flags: list[str] = []

        # Constant features.
        if (row["std"] == 0) or (row["n_unique"] <= 1):
            col_flags.append("constant_feature")

        # Extreme outliers by IQR rule.
        extreme_count = _series_extreme_iqr_count(s, median=float(row["50%"]), iqr=float(row["IQR"]))
        if extreme_count > 0:
            col_flags.append(f"extreme_outliers_gt_10xIQR:{extreme_count}")

        # Impossible price ranges (canonical plus lag variants).
        if (
            col == "afrr_activation_price_vwap_pos"
            or col == "da_price"
            or col.startswith("afrr_activation_price_vwap_pos_lag_")
            or col.startswith("da_price_lag_")
        ):
            x = pd.to_numeric(s, errors="coerce")
            bad = ((x > 5000) | (x < -500)).sum()
            if int(bad) > 0:
                col_flags.append(f"impossible_price_range:{int(bad)}")

        # Impossible activation volumes.
        if "afrr_activated_mw" in col:
            x = pd.to_numeric(s, errors="coerce").abs()
            bad = (x > 2000).sum()
            if int(bad) > 0:
                col_flags.append(f"impossible_activation_mw:{int(bad)}")

        # Lag-parent distribution shift check.
        base = _lag_base_name(col)
        if base and base in num.columns:
            shift, reason = _distribution_shift_flag(num[base], s)
            if shift:
                col_flags.append(reason)

        flags.append("; ".join(col_flags))

    out = stats.copy()
    out["flags"] = flags
    out["has_red_flag"] = out["flags"].str.len().fillna(0) > 0
    return out


def _clip_recommendations(stats_flags: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in stats_flags.iterrows():
        col = r["column"]
        p99 = float(r["p99"]) if np.isfinite(r["p99"]) else np.nan
        max_v = float(r["max"]) if np.isfinite(r["max"]) else np.nan
        min_v = float(r["min"]) if np.isfinite(r["min"]) else np.nan
        max_abs = np.nanmax([abs(max_v), abs(min_v)])
        denom = abs(p99) + 1e-9
        ratio = max_abs / denom if np.isfinite(max_abs) and np.isfinite(denom) else np.nan
        if np.isfinite(ratio) and ratio >= 100:
            rows.append(
                {
                    "column": col,
                    "reason": f"max_abs/p99={ratio:.1f}",
                    "recommendation": "Winsorize at 99th pct (and symmetric lower bound where needed)",
                }
            )
    return pd.DataFrame(rows)


def _print_markdown_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("\nNo clipping recommendations based on current fat-tail rule.")
        return
    print("\n## Clipping Strategy Recommendations")
    print("| Column | Reason | Recommendation |")
    print("|---|---|---|")
    for _, r in df.iterrows():
        print(f"| `{r['column']}` | {r['reason']} | {r['recommendation']} |")


def _save_volatility_plot(df: pd.DataFrame, stats_flags: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        return

    top = (
        stats_flags.sort_values("std", ascending=False)
        .loc[:, "column"]
        .head(10)
        .tolist()
    )
    if not top:
        return
    num = df[top].copy().apply(pd.to_numeric, errors="coerce")
    # Keep plotting cheap and robust for large frames.
    if len(num) > 10000:
        num = num.sample(n=10000, random_state=42)
    melted = num.melt(var_name="feature", value_name="value").dropna()
    if melted.empty:
        return

    try:
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=melted, x="feature", y="value", color="#226E9C", fliersize=1)
        plt.xticks(rotation=35, ha="right")
        plt.title("Top 10 Most Volatile Numeric Features")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception as exc:
        print(f"[WARN] Could not save volatility plot: {exc}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate realism of feature artifact statistics.")
    p.add_argument(
        "--path",
        default="data/features/all_data_features.parquet",
        help="Input parquet file path.",
    )
    p.add_argument(
        "--out-csv",
        default="data/reports/feature_statistics_audit.csv",
        help="Output CSV path for audit statistics.",
    )
    p.add_argument(
        "--plot-out",
        default="data/reports/top10_volatile_features_boxplot.png",
        help="Output path for optional volatility boxplot.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = (REPO_ROOT / args.path).resolve() if not Path(args.path).is_absolute() else Path(args.path)
    out_csv = (REPO_ROOT / args.out_csv).resolve() if not Path(args.out_csv).is_absolute() else Path(args.out_csv)
    plot_out = (REPO_ROOT / args.plot_out).resolve() if not Path(args.plot_out).is_absolute() else Path(args.plot_out)

    if not in_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_path}")

    _style_ok, _ = _load_style()

    df = pd.read_parquet(in_path, engine="pyarrow")
    stats = _numeric_profile(df)
    if stats.empty:
        raise ValueError("No numeric columns found in input data.")

    audit = _flag_columns(df=df, stats=stats)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_csv, index=False)

    print(f"[INFO] Input: {in_path}")
    print(f"[INFO] Numeric columns audited: {len(audit)}")
    print(f"[INFO] Red-flag columns: {int(audit['has_red_flag'].sum())}")
    print(f"[INFO] Audit CSV written: {out_csv}")

    clip_df = _clip_recommendations(audit)
    _print_markdown_table(clip_df)

    _save_volatility_plot(df, audit, plot_out)
    print(f"[INFO] Plot saved: {plot_out}")
    if not _style_ok:
        print("[WARN] style.py could not be loaded; default matplotlib style used.")


if __name__ == "__main__":
    main()
