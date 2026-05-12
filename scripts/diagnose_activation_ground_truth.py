#!/usr/bin/env python3
"""Diagnose and optionally clean aFRR activation-price ground-truth columns.

Checks performed:
1) Distribution audit for target_afrr_activation_price_vwap_{pos,neg}.
2) Placeholder/sentinel detection (e.g., 9999, 15000, very large absolute values).
3) Static aggregation-code audit for 15m->1h logic in fetch_regelleistung.py
   (verify hourly base mean + VWAP numerator/denominator sums).
4) No-activation handling audit and optional cleanup:
   - when target activation rate is zero, target activation price is set to NaN
   - sentinel-like values are set to NaN

By default, this script does NOT overwrite source data.
Use --write-cleaned-path to write a cleaned parquet copy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SeriesStats:
    count: int
    non_null: int
    nulls: int
    mean: float | None
    median: float | None
    std: float | None
    q01: float | None
    q05: float | None
    q25: float | None
    q75: float | None
    q95: float | None
    q99: float | None
    min: float | None
    max: float | None


def _safe_float(v: Any) -> float | None:
    try:
        fv = float(v)
    except Exception:
        return None
    return fv if np.isfinite(fv) else None


def _stats(s: pd.Series) -> SeriesStats:
    x = pd.to_numeric(s, errors="coerce")
    n = int(x.shape[0])
    nn = int(x.notna().sum())
    if nn == 0:
        return SeriesStats(n, 0, n, None, None, None, None, None, None, None, None, None, None, None)
    q = x.quantile([0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
    return SeriesStats(
        count=n,
        non_null=nn,
        nulls=n - nn,
        mean=_safe_float(x.mean()),
        median=_safe_float(x.median()),
        std=_safe_float(x.std(ddof=1)),
        q01=_safe_float(q.loc[0.01]),
        q05=_safe_float(q.loc[0.05]),
        q25=_safe_float(q.loc[0.25]),
        q75=_safe_float(q.loc[0.75]),
        q95=_safe_float(q.loc[0.95]),
        q99=_safe_float(q.loc[0.99]),
        min=_safe_float(x.min()),
        max=_safe_float(x.max()),
    )


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _placeholder_audit(s: pd.Series, placeholder_values: list[float], huge_abs_threshold: float) -> dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce")
    out: dict[str, Any] = {}
    for pv in placeholder_values:
        m = x.eq(pv)
        mn = x.eq(-pv)
        out[f"eq_{int(pv)}"] = int(m.sum())
        out[f"eq_-{int(pv)}"] = int(mn.sum())
    out["abs_ge_huge_threshold"] = int(x.abs().ge(huge_abs_threshold).sum())
    return out


def _static_aggregation_audit(fetch_regelleistung_path: Path) -> dict[str, Any]:
    txt = fetch_regelleistung_path.read_text(encoding="utf-8")
    # coarse but robust static checks
    checks = {
        "has_hourly_base_mean": "resample(\"1h\").mean" in txt,
        "has_weighted_cost_pos": "weighted_cost_pos" in txt,
        "has_weighted_cost_neg": "weighted_cost_neg" in txt,
        "has_sum_weighted_pos": "sum_weighted_pos" in txt and ".sum(" in txt,
        "has_sum_weighted_neg": "sum_weighted_neg" in txt and ".sum(" in txt,
        "has_sum_vol_pos": "sum_vol_pos" in txt,
        "has_sum_vol_neg": "sum_vol_neg" in txt,
        "has_vwap_formula_pos": "sum_weighted_pos / sum_vol_pos" in txt,
        "has_vwap_formula_neg": "sum_weighted_neg / sum_vol_neg" in txt,
        "has_zero_vol_fallback_pos": "np.where(sum_vol_pos != 0" in txt,
        "has_zero_vol_fallback_neg": "np.where(sum_vol_neg != 0" in txt,
        "has_mean_price_fallback": "mean_price_pos" in txt and "mean_price_neg" in txt,
    }
    checks["overall_pass"] = all(checks.values())
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose/clean activation-price target columns in all_data_features.parquet")
    ap.add_argument("--features-path", default="data/features/all_data_features.parquet")
    ap.add_argument("--fetch-regelleistung-path", default="src/energy_trading/ingestion/fetch_regelleistung.py")
    ap.add_argument("--out-dir", default="artifacts/reports/activation_gt_diagnosis")
    ap.add_argument("--huge-abs-threshold", type=float, default=90000.0)
    ap.add_argument("--placeholder-values", nargs="*", type=float, default=[9999.0, 15000.0, 3000.0])
    ap.add_argument("--zero-rate-eps", type=float, default=1e-12)
    ap.add_argument("--write-cleaned-path", default="", help="Optional path to write cleaned parquet copy")
    ap.add_argument(
        "--raw-15m-path",
        default="data/raw/regelleistung_15min/afrr_price_volume_15min.parquet",
        help="Optional raw 15m parquet for dynamic aggregation sanity-check",
    )
    args = ap.parse_args()

    features_path = Path(args.features_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        raise FileNotFoundError(f"features parquet not found: {features_path}")

    df = pd.read_parquet(features_path)

    pos_col = _find_col(df, ["target_afrr_activation_price_vwap_pos", "afrr_activation_price_vwap_pos"])
    neg_col = _find_col(df, ["target_afrr_activation_price_vwap_neg", "afrr_activation_price_vwap_neg"])
    rate_pos_col = _find_col(df, ["target_afrr_activation_rate_pos", "afrr_activation_rate_pos"])
    rate_neg_col = _find_col(df, ["target_afrr_activation_rate_neg", "afrr_activation_rate_neg"])

    missing = [
        name for name, col in [
            ("activation price pos", pos_col),
            ("activation price neg", neg_col),
            ("activation rate pos", rate_pos_col),
            ("activation rate neg", rate_neg_col),
        ]
        if col is None
    ]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Distribution and placeholders (raw)
    raw_pos = pd.to_numeric(df[pos_col], errors="coerce")
    raw_neg = pd.to_numeric(df[neg_col], errors="coerce")
    raw_rate_pos = pd.to_numeric(df[rate_pos_col], errors="coerce")
    raw_rate_neg = pd.to_numeric(df[rate_neg_col], errors="coerce")

    # No-activation masks (target perspective)
    no_act_pos = raw_rate_pos.fillna(0.0).abs().le(args.zero_rate_eps)
    no_act_neg = raw_rate_neg.fillna(0.0).abs().le(args.zero_rate_eps)

    # Candidate cleaning rules:
    # A) sentinel-like placeholders -> NaN
    # B) no activation -> NaN (price economically undefined for that hour)
    sentinel_mask_pos = raw_pos.abs().ge(float(args.huge_abs_threshold))
    sentinel_mask_neg = raw_neg.abs().ge(float(args.huge_abs_threshold))
    for pv in args.placeholder_values:
        sentinel_mask_pos = sentinel_mask_pos | raw_pos.eq(float(pv)) | raw_pos.eq(-float(pv))
        sentinel_mask_neg = sentinel_mask_neg | raw_neg.eq(float(pv)) | raw_neg.eq(-float(pv))

    cleaned_pos = raw_pos.copy()
    cleaned_neg = raw_neg.copy()
    cleaned_pos[sentinel_mask_pos | no_act_pos] = np.nan
    cleaned_neg[sentinel_mask_neg | no_act_neg] = np.nan

    # Prepare report
    report: dict[str, Any] = {
        "features_path": str(features_path.resolve()),
        "columns": {
            "pos_price_col": pos_col,
            "neg_price_col": neg_col,
            "pos_rate_col": rate_pos_col,
            "neg_rate_col": rate_neg_col,
        },
        "raw_stats": {
            "pos": asdict(_stats(raw_pos)),
            "neg": asdict(_stats(raw_neg)),
        },
        "placeholder_audit": {
            "pos": _placeholder_audit(raw_pos, args.placeholder_values, args.huge_abs_threshold),
            "neg": _placeholder_audit(raw_neg, args.placeholder_values, args.huge_abs_threshold),
        },
        "no_activation_audit": {
            "pos_no_activation_hours": int(no_act_pos.sum()),
            "neg_no_activation_hours": int(no_act_neg.sum()),
            "pos_nonnull_price_during_no_activation": int(raw_pos[no_act_pos].notna().sum()),
            "neg_nonnull_price_during_no_activation": int(raw_neg[no_act_neg].notna().sum()),
            "pos_median_price_during_no_activation": _safe_float(raw_pos[no_act_pos].median()),
            "neg_median_price_during_no_activation": _safe_float(raw_neg[no_act_neg].median()),
        },
        "cleaning_effect": {
            "rules": [
                "set sentinel/placeholder activation prices to NaN",
                "set activation price to NaN when activation rate is zero",
            ],
            "pos_rows_set_to_nan": int((raw_pos.notna() & cleaned_pos.isna()).sum()),
            "neg_rows_set_to_nan": int((raw_neg.notna() & cleaned_neg.isna()).sum()),
            "cleaned_stats": {
                "pos": asdict(_stats(cleaned_pos)),
                "neg": asdict(_stats(cleaned_neg)),
            },
        },
        "aggregation_code_audit": _static_aggregation_audit(Path(args.fetch_regelleistung_path)),
    }

    # Optional dynamic aggregation sanity-check vs raw 15m source
    raw_15m_path = Path(args.raw_15m_path)
    if raw_15m_path.exists() and "timestamp_utc" in df.columns:
        raw15 = pd.read_parquet(raw_15m_path)
        if "timestamp_utc" in raw15.columns:
            r = raw15.copy()
            r["timestamp_utc"] = pd.to_datetime(r["timestamp_utc"], utc=True, errors="coerce")
            r = r.dropna(subset=["timestamp_utc"]).set_index("timestamp_utc").sort_index()

            dyn: dict[str, Any] = {"source": str(raw_15m_path.resolve()), "available": True}

            for side in ("pos", "neg"):
                pcol = f"afrr_avg_activation_price_{side}"
                if pcol not in r.columns:
                    dyn[f"{side}_missing"] = pcol
                    continue
                px = pd.to_numeric(r[pcol], errors="coerce")
                h_mean = px.resample("1h").mean()
                h_sum = px.resample("1h").sum(min_count=1)

                feat_col = f"afrr_activation_price_vwap_{side}"
                if feat_col not in df.columns:
                    dyn[f"{side}_feature_missing"] = feat_col
                    continue
                feat = pd.DataFrame(
                    {
                        "timestamp_utc": pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce"),
                        "feature": pd.to_numeric(df[feat_col], errors="coerce"),
                    }
                ).dropna(subset=["timestamp_utc"]).set_index("timestamp_utc").sort_index()

                m = pd.concat([feat["feature"], h_mean.rename("mean15"), h_sum.rename("sum15")], axis=1).dropna()
                if m.empty:
                    dyn[f"{side}_rows_overlap"] = 0
                    continue

                corr_mean = float(m["feature"].corr(m["mean15"]))
                corr_sum = float(m["feature"].corr(m["sum15"]))
                mae_mean = float((m["feature"] - m["mean15"]).abs().mean())
                mae_sum = float((m["feature"] - m["sum15"]).abs().mean())
                dyn[f"{side}_rows_overlap"] = int(len(m))
                dyn[f"{side}_corr_feature_vs_mean15"] = corr_mean
                dyn[f"{side}_corr_feature_vs_sum15"] = corr_sum
                dyn[f"{side}_mae_feature_vs_mean15"] = mae_mean
                dyn[f"{side}_mae_feature_vs_sum15"] = mae_sum
                dyn[f"{side}_looks_like_mean_not_sum"] = bool(mae_mean < mae_sum)

            report["aggregation_dynamic_audit"] = dyn
        else:
            report["aggregation_dynamic_audit"] = {
                "source": str(raw_15m_path.resolve()),
                "available": False,
                "reason": "raw 15m file missing timestamp_utc",
            }
    else:
        report["aggregation_dynamic_audit"] = {
            "source": str(raw_15m_path),
            "available": False,
            "reason": "raw 15m file not found or features has no timestamp_utc",
        }

    # Save detailed CSV for investigation
    out_diag = pd.DataFrame(
        {
            "target_afrr_activation_price_vwap_pos_raw": raw_pos,
            "target_afrr_activation_price_vwap_neg_raw": raw_neg,
            "target_afrr_activation_rate_pos": raw_rate_pos,
            "target_afrr_activation_rate_neg": raw_rate_neg,
            "no_activation_pos": no_act_pos,
            "no_activation_neg": no_act_neg,
            "sentinel_pos": sentinel_mask_pos,
            "sentinel_neg": sentinel_mask_neg,
            "target_afrr_activation_price_vwap_pos_clean": cleaned_pos,
            "target_afrr_activation_price_vwap_neg_clean": cleaned_neg,
        }
    )

    if "timestamp_utc" in df.columns:
        out_diag.insert(0, "timestamp_utc", pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce"))

    report_path = out_dir / "activation_ground_truth_diagnosis_report.json"
    diag_path = out_dir / "activation_ground_truth_diagnosis_rows.parquet"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_diag.to_parquet(diag_path, index=False)

    # Optional cleaned export
    cleaned_path = str(args.write_cleaned_path).strip()
    if cleaned_path:
        cpath = Path(cleaned_path)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        df_out = df.copy()
        # overwrite canonical target columns if present
        if "target_afrr_activation_price_vwap_pos" in df_out.columns:
            df_out["target_afrr_activation_price_vwap_pos"] = cleaned_pos
        if "target_afrr_activation_price_vwap_neg" in df_out.columns:
            df_out["target_afrr_activation_price_vwap_neg"] = cleaned_neg
        # also keep explicit debug columns
        df_out["target_afrr_activation_price_vwap_pos_clean"] = cleaned_pos
        df_out["target_afrr_activation_price_vwap_neg_clean"] = cleaned_neg
        df_out.to_parquet(cpath, index=False)
        print(f"[OK] Wrote cleaned parquet: {cpath}")

    # Console summary
    rs = report["raw_stats"]
    cs = report["cleaning_effect"]["cleaned_stats"]
    print("[SUMMARY] Raw medians:")
    print(f"  pos median: {rs['pos']['median']}")
    print(f"  neg median: {rs['neg']['median']}")
    print("[SUMMARY] Cleaned medians:")
    print(f"  pos median: {cs['pos']['median']}")
    print(f"  neg median: {cs['neg']['median']}")
    print("[SUMMARY] Non-null prices during no activation:")
    na = report["no_activation_audit"]
    print(f"  pos: {na['pos_nonnull_price_during_no_activation']} / {na['pos_no_activation_hours']}")
    print(f"  neg: {na['neg_nonnull_price_during_no_activation']} / {na['neg_no_activation_hours']}")
    print(f"[SUMMARY] Aggregation code audit pass: {report['aggregation_code_audit']['overall_pass']}")
    dyn = report.get("aggregation_dynamic_audit", {})
    if dyn.get("available"):
        for side in ("pos", "neg"):
            key = f"{side}_looks_like_mean_not_sum"
            if key in dyn:
                print(f"[SUMMARY] Dynamic {side}: looks_like_mean_not_sum={dyn[key]}")
    print(f"[OK] Report: {report_path}")
    print(f"[OK] Row diagnostics: {diag_path}")


if __name__ == "__main__":
    main()
