#!/usr/bin/env python3
"""Compare candidate aFRR activation-price datasets and recommend best option.

Core API:
    evaluate_price_options(options_dict)

`options_dict` format:
    {
      "option_name": {
          "df": pandas.DataFrame,                   # optional if path is provided
          "path": "path/to/file.parquet",         # optional if df is provided
          "pos_col": "activation_price_pos_col",
          "neg_col": "activation_price_neg_col",
          "da_col": "da_price_col",               # optional, helps plausibility checks
          "act_rate_pos_col": "rate_pos_col",     # optional
          "act_rate_neg_col": "rate_neg_col",     # optional
          "timestamp_col": "timestamp_utc",       # optional
      },
      ...
    }

Outputs:
- Comparison DataFrame with metrics + warnings + score.
- Recommendation dict with best option and rationale.
- Plot helpers for visual inspection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PLACEHOLDER_VALUES = (9999.0, 9999.99, -9999.0, -9999.99)
HARD_MEDIAN_LIMIT = 500.0
EXTREME_ABS_THRESHOLD = 3000.0


@dataclass
class OptionSpec:
    name: str
    df: pd.DataFrame
    pos_col: str
    neg_col: str
    da_col: str | None = None
    act_rate_pos_col: str | None = None
    act_rate_neg_col: str | None = None
    timestamp_col: str | None = None



def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _series_stats(s: pd.Series) -> dict[str, float | int | None]:
    x = _to_num(s)
    n = int(len(x))
    nn = int(x.notna().sum())
    if nn == 0:
        return {
            "n": n,
            "non_null": 0,
            "nan_pct": 100.0,
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p01": np.nan,
            "p99": np.nan,
            "kurtosis": np.nan,
            "share_abs_gt_3000": np.nan,
        }

    xv = x.dropna()
    return {
        "n": n,
        "non_null": nn,
        "nan_pct": float((1.0 - nn / n) * 100.0) if n else np.nan,
        "mean": float(xv.mean()),
        "median": float(xv.median()),
        "min": float(xv.min()),
        "max": float(xv.max()),
        "p01": float(xv.quantile(0.01)),
        "p99": float(xv.quantile(0.99)),
        "kurtosis": float(xv.kurtosis()),
        "share_abs_gt_3000": float((xv.abs() > EXTREME_ABS_THRESHOLD).mean() * 100.0),
    }


def _placeholder_metrics(s: pd.Series) -> dict[str, float]:
    x = _to_num(s)
    nn = int(x.notna().sum())
    if nn == 0:
        return {
            "placeholder_pct": np.nan,
            "zero_pct": np.nan,
        }
    m_placeholder = pd.Series(False, index=x.index)
    for v in PLACEHOLDER_VALUES:
        m_placeholder = m_placeholder | x.eq(v)
    placeholder_pct = float(m_placeholder.sum() / nn * 100.0)
    zero_pct = float(x.eq(0.0).sum() / nn * 100.0)
    return {
        "placeholder_pct": placeholder_pct,
        "zero_pct": zero_pct,
    }


def _hourly_index_gap_pct(ts: pd.Series) -> float | None:
    t = pd.to_datetime(ts, utc=True, errors="coerce").dropna().sort_values().drop_duplicates()
    if len(t) < 2:
        return np.nan
    full = pd.date_range(start=t.iloc[0], end=t.iloc[-1], freq="1h", tz="UTC")
    if len(full) == 0:
        return np.nan
    missing = len(full.difference(pd.DatetimeIndex(t)))
    return float(missing / len(full) * 100.0)


def _non_activation_diagnostics(
    price: pd.Series,
    rate: pd.Series | None,
) -> dict[str, float | int | None]:
    if rate is None:
        return {
            "no_act_hours": np.nan,
            "nonnull_price_during_no_act": np.nan,
            "nonnull_price_during_no_act_pct": np.nan,
            "ffill_suspicion_pct": np.nan,
        }

    p = _to_num(price)
    r = _to_num(rate).fillna(0.0)
    no_act = r.abs().le(1e-12)
    n_no = int(no_act.sum())
    if n_no == 0:
        return {
            "no_act_hours": 0,
            "nonnull_price_during_no_act": 0,
            "nonnull_price_during_no_act_pct": 0.0,
            "ffill_suspicion_pct": 0.0,
        }

    nonnull_no = int(p[no_act].notna().sum())
    pct_nonnull_no = float(nonnull_no / n_no * 100.0)

    prev_price = p.shift(1)
    p95 = p.dropna().abs().quantile(0.95) if p.notna().any() else np.nan
    ffill_sus = no_act & p.notna() & prev_price.notna() & p.eq(prev_price) & prev_price.abs().ge(p95)
    ffill_pct = float(ffill_sus.sum() / n_no * 100.0)

    return {
        "no_act_hours": n_no,
        "nonnull_price_during_no_act": nonnull_no,
        "nonnull_price_during_no_act_pct": pct_nonnull_no,
        "ffill_suspicion_pct": ffill_pct,
    }


def _aggregation_symptom_flags(pos_stats: dict[str, Any], neg_stats: dict[str, Any], da_stats: dict[str, Any] | None) -> dict[str, Any]:
    flags: list[str] = []

    # Symptom 1: very large p99 tails on both sides
    if np.isfinite(pos_stats["p99"]) and np.isfinite(neg_stats["p01"]):
        if pos_stats["p99"] > 12000 and abs(neg_stats["p01"]) > 12000:
            flags.append("both_tails_extreme")

    # Symptom 2: medians far from DA median
    if da_stats is not None and np.isfinite(da_stats.get("median", np.nan)):
        da_med = abs(float(da_stats["median"]))
        da_med = max(da_med, 1e-6)
        ratio_pos = abs(float(pos_stats["median"])) / da_med if np.isfinite(pos_stats["median"]) else np.nan
        ratio_neg = abs(float(neg_stats["median"])) / da_med if np.isfinite(neg_stats["median"]) else np.nan
        if np.isfinite(ratio_pos) and ratio_pos > 4:
            flags.append("pos_median_far_from_da")
        if np.isfinite(ratio_neg) and ratio_neg > 4:
            flags.append("neg_median_far_from_da")

    return {
        "aggregation_symptom_flags": ";".join(flags) if flags else "",
        "aggregation_symptom_count": len(flags),
    }


def _score_option(row: pd.Series) -> tuple[float, list[str], bool]:
    warnings: list[str] = []
    disqualified = False

    pos_med = float(row.get("pos_median", np.nan))
    neg_med = float(row.get("neg_median", np.nan))
    if (np.isfinite(pos_med) and pos_med > HARD_MEDIAN_LIMIT) or (np.isfinite(neg_med) and neg_med < -HARD_MEDIAN_LIMIT):
        warnings.append("median_outside_plausible_band")
        disqualified = True

    score = 100.0

    # completeness
    score -= 0.8 * float(row.get("avg_nan_pct", 0.0))
    score -= 3.0 * float(row.get("avg_placeholder_pct", 0.0))

    # no activation handling
    score -= 0.4 * float(row.get("avg_nonnull_price_during_no_act_pct", 0.0))
    score -= 0.3 * float(row.get("avg_ffill_suspicion_pct", 0.0))

    # volatility / outlier density
    score -= 0.4 * float(row.get("avg_share_abs_gt_3000_pct", 0.0))

    # aggregation symptoms
    score -= 8.0 * float(row.get("aggregation_symptom_count", 0.0))

    # hard disqualifiers
    if float(row.get("avg_nan_pct", 0.0)) > 40:
        warnings.append("too_many_missing_values")
        disqualified = True
    if float(row.get("avg_placeholder_pct", 0.0)) > 1.0:
        warnings.append("placeholder_values_detected")
    if float(row.get("avg_share_abs_gt_3000_pct", 0.0)) > 20.0:
        warnings.append("extreme_spike_density")

    return score, warnings, disqualified


def evaluate_price_options(options_dict: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute diagnostic metrics and recommendation for candidate datasets."""
    rows: list[dict[str, Any]] = []

    for name, spec_raw in options_dict.items():
        if "df" in spec_raw and spec_raw["df"] is not None:
            df = spec_raw["df"].copy()
        else:
            path = spec_raw.get("path")
            if not path:
                raise ValueError(f"Option '{name}': provide either df or path")
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"Option '{name}': file not found: {p}")
            df = pd.read_parquet(p)

        pos_col = spec_raw["pos_col"]
        neg_col = spec_raw["neg_col"]
        if pos_col not in df.columns or neg_col not in df.columns:
            raise KeyError(f"Option '{name}': missing pos/neg columns in dataframe")

        da_col = spec_raw.get("da_col")
        act_rate_pos_col = spec_raw.get("act_rate_pos_col")
        act_rate_neg_col = spec_raw.get("act_rate_neg_col")
        timestamp_col = spec_raw.get("timestamp_col")

        pos = _to_num(df[pos_col])
        neg = _to_num(df[neg_col])
        da = _to_num(df[da_col]) if da_col and da_col in df.columns else None
        rate_pos = _to_num(df[act_rate_pos_col]) if act_rate_pos_col and act_rate_pos_col in df.columns else None
        rate_neg = _to_num(df[act_rate_neg_col]) if act_rate_neg_col and act_rate_neg_col in df.columns else None

        pos_stats = _series_stats(pos)
        neg_stats = _series_stats(neg)
        da_stats = _series_stats(da) if da is not None else None

        pos_ph = _placeholder_metrics(pos)
        neg_ph = _placeholder_metrics(neg)

        pos_no = _non_activation_diagnostics(pos, rate_pos)
        neg_no = _non_activation_diagnostics(neg, rate_neg)

        gap_pct = np.nan
        if timestamp_col and timestamp_col in df.columns:
            gap_pct = _hourly_index_gap_pct(df[timestamp_col])

        agg_flags = _aggregation_symptom_flags(pos_stats, neg_stats, da_stats)

        row: dict[str, Any] = {
            "option": name,
            "n_rows": int(len(df)),
            "hourly_gap_pct": gap_pct,
            "pos_mean": pos_stats["mean"],
            "pos_median": pos_stats["median"],
            "pos_min": pos_stats["min"],
            "pos_max": pos_stats["max"],
            "pos_p01": pos_stats["p01"],
            "pos_p99": pos_stats["p99"],
            "pos_kurtosis": pos_stats["kurtosis"],
            "pos_nan_pct": pos_stats["nan_pct"],
            "pos_share_abs_gt_3000_pct": pos_stats["share_abs_gt_3000"],
            "neg_mean": neg_stats["mean"],
            "neg_median": neg_stats["median"],
            "neg_min": neg_stats["min"],
            "neg_max": neg_stats["max"],
            "neg_p01": neg_stats["p01"],
            "neg_p99": neg_stats["p99"],
            "neg_kurtosis": neg_stats["kurtosis"],
            "neg_nan_pct": neg_stats["nan_pct"],
            "neg_share_abs_gt_3000_pct": neg_stats["share_abs_gt_3000"],
            "da_median": da_stats["median"] if da_stats else np.nan,
            "pos_placeholder_pct": pos_ph["placeholder_pct"],
            "neg_placeholder_pct": neg_ph["placeholder_pct"],
            "pos_zero_pct": pos_ph["zero_pct"],
            "neg_zero_pct": neg_ph["zero_pct"],
            "pos_nonnull_price_during_no_act_pct": pos_no["nonnull_price_during_no_act_pct"],
            "neg_nonnull_price_during_no_act_pct": neg_no["nonnull_price_during_no_act_pct"],
            "pos_ffill_suspicion_pct": pos_no["ffill_suspicion_pct"],
            "neg_ffill_suspicion_pct": neg_no["ffill_suspicion_pct"],
            **agg_flags,
        }

        row["avg_nan_pct"] = float(np.nanmean([row["pos_nan_pct"], row["neg_nan_pct"]]))
        row["avg_placeholder_pct"] = float(np.nanmean([row["pos_placeholder_pct"], row["neg_placeholder_pct"]]))
        row["avg_share_abs_gt_3000_pct"] = float(
            np.nanmean([row["pos_share_abs_gt_3000_pct"], row["neg_share_abs_gt_3000_pct"]])
        )
        row["avg_nonnull_price_during_no_act_pct"] = float(
            np.nanmean([row["pos_nonnull_price_during_no_act_pct"], row["neg_nonnull_price_during_no_act_pct"]])
        )
        row["avg_ffill_suspicion_pct"] = float(
            np.nanmean([row["pos_ffill_suspicion_pct"], row["neg_ffill_suspicion_pct"]])
        )

        score, warnings, disq = _score_option(pd.Series(row))
        row["score"] = score
        row["warnings"] = ";".join(warnings) if warnings else ""
        row["disqualified"] = bool(disq)

        rows.append(row)

    cmp_df = pd.DataFrame(rows).sort_values(["disqualified", "score"], ascending=[True, False]).reset_index(drop=True)

    eligible = cmp_df.loc[~cmp_df["disqualified"].fillna(False)].copy()
    if eligible.empty:
        best_idx = int(cmp_df["score"].astype(float).idxmax())
        best = cmp_df.loc[best_idx]
        recommendation = {
            "recommended_option": str(best["option"]),
            "reason": "All options disqualified by hard plausibility rules; returning highest-score fallback.",
            "hard_warning": "Review ground truth generation before using any option.",
        }
    else:
        best = eligible.iloc[0]
        recommendation = {
            "recommended_option": str(best["option"]),
            "reason": (
                "Best eligible score with strongest balance of plausible medians, "
                "low missing/imputation pressure, and fewer aggregation/outlier warnings."
            ),
            "score": float(best["score"]),
            "warnings": str(best.get("warnings", "")),
        }

    return cmp_df, recommendation


# ---------- Plot helpers ----------
def plot_option_boxplots(options_dict: dict[str, dict[str, Any]]) -> None:
    """Create boxplots (without fliers) for pos/neg distributions per option."""
    import matplotlib.pyplot as plt

    labels: list[str] = []
    pos_vals: list[np.ndarray] = []
    neg_vals: list[np.ndarray] = []

    for name, spec_raw in options_dict.items():
        if "df" in spec_raw and spec_raw["df"] is not None:
            df = spec_raw["df"]
        else:
            df = pd.read_parquet(spec_raw["path"])

        p = _to_num(df[spec_raw["pos_col"]]).dropna().to_numpy(dtype=float)
        n = _to_num(df[spec_raw["neg_col"]]).dropna().to_numpy(dtype=float)
        labels.append(name)
        pos_vals.append(p)
        neg_vals.append(n)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    axes[0].boxplot(pos_vals, labels=labels, showfliers=False)
    axes[0].set_title("Activation Price POS (boxplot without outliers)")
    axes[0].set_ylabel("EUR/MWh")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].boxplot(neg_vals, labels=labels, showfliers=False)
    axes[1].set_title("Activation Price NEG (boxplot without outliers)")
    axes[1].set_ylabel("EUR/MWh")
    axes[1].tick_params(axis="x", rotation=20)

    plt.show()


def _build_default_options(features_path: Path) -> dict[str, dict[str, Any]]:
    """Convenience options for this repo setup."""
    df = pd.read_parquet(features_path)

    options: dict[str, dict[str, Any]] = {
        "features_targets_raw": {
            "df": df,
            "pos_col": "target_afrr_activation_price_vwap_pos",
            "neg_col": "target_afrr_activation_price_vwap_neg",
            "da_col": "target_da_price" if "target_da_price" in df.columns else None,
            "act_rate_pos_col": "target_afrr_activation_rate_pos" if "target_afrr_activation_rate_pos" in df.columns else None,
            "act_rate_neg_col": "target_afrr_activation_rate_neg" if "target_afrr_activation_rate_neg" in df.columns else None,
            "timestamp_col": "timestamp_utc" if "timestamp_utc" in df.columns else None,
        }
    }

    # if cleaned copy exists from diagnose_activation_ground_truth.py
    clean_pos = "target_afrr_activation_price_vwap_pos_clean"
    clean_neg = "target_afrr_activation_price_vwap_neg_clean"
    if clean_pos in df.columns and clean_neg in df.columns:
        options["features_targets_clean_cols"] = {
            "df": df,
            "pos_col": clean_pos,
            "neg_col": clean_neg,
            "da_col": "target_da_price" if "target_da_price" in df.columns else None,
            "act_rate_pos_col": "target_afrr_activation_rate_pos" if "target_afrr_activation_rate_pos" in df.columns else None,
            "act_rate_neg_col": "target_afrr_activation_rate_neg" if "target_afrr_activation_rate_neg" in df.columns else None,
            "timestamp_col": "timestamp_utc" if "timestamp_utc" in df.columns else None,
        }

    return options


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare candidate activation-price datasets")
    ap.add_argument("--features-path", default="data/features/all_data_features.parquet")
    ap.add_argument("--options-json", default="", help="Optional JSON file describing options_dict")
    ap.add_argument("--out-csv", default="artifacts/reports/activation_option_compare/option_comparison.csv")
    ap.add_argument("--out-json", default="artifacts/reports/activation_option_compare/recommendation.json")
    ap.add_argument("--plot", action="store_true", help="Show boxplots")
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.options_json:
        cfg = json.loads(Path(args.options_json).read_text(encoding="utf-8"))
        options_dict = cfg
    else:
        options_dict = _build_default_options(Path(args.features_path))

    cmp_df, rec = evaluate_price_options(options_dict)
    cmp_df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    print("\n=== Option Comparison ===")
    print(cmp_df[[
        "option", "score", "disqualified",
        "pos_median", "neg_median", "da_median",
        "avg_nan_pct", "avg_placeholder_pct",
        "avg_nonnull_price_during_no_act_pct",
        "avg_share_abs_gt_3000_pct",
        "aggregation_symptom_flags", "warnings",
    ]].to_string(index=False))

    print("\n=== Recommendation ===")
    print(json.dumps(rec, indent=2))
    print(f"\n[OK] comparison csv: {out_csv}")
    print(f"[OK] recommendation: {out_json}")

    if args.plot:
        plot_option_boxplots(options_dict)


if __name__ == "__main__":
    main()
