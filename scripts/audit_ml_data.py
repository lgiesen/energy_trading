#!/usr/bin/env python3
"""Audit ML input bundles for leakage, causality, data quality, and drift.

Usage:
    python3 scripts/audit_ml_data.py --base-dir data/model_input
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")
NUM_SENTINELS = (99999.0, -999.0)


@dataclass(frozen=True)
class BundleData:
    bundle: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _read_split(base_dir: Path, bundle: str, split: str) -> pd.DataFrame:
    p = base_dir / bundle / f"{split}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing split file: {p}")
    df = pd.read_parquet(p)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
    return df


def _load_bundle(base_dir: Path, bundle: str) -> BundleData:
    return BundleData(
        bundle=bundle,
        train=_read_split(base_dir, bundle, "train"),
        val=_read_split(base_dir, bundle, "val"),
        test=_read_split(base_dir, bundle, "test"),
    )


def _target_columns(df: pd.DataFrame) -> list[str]:
    # Canonical naming: target_*
    cols = [c for c in df.columns if c.startswith("target_")]
    # Legacy fallback retained for compatibility
    if "target_afrr_rate_h1" in df.columns and "target_afrr_rate_h1" not in cols:
        cols.append("target_afrr_rate_h1")
    return sorted(cols)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    targets = set(_target_columns(df))
    excluded = {"timestamp_utc"} | targets
    return [c for c in df.columns if c not in excluded]


def _numeric_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    out: list[str] = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def audit_correlation_leakage(bundle_data: BundleData, corr_threshold: float) -> pd.DataFrame:
    df = bundle_data.train
    targets = _target_columns(df)
    feats = _numeric_columns(df, _feature_columns(df))
    rows: list[dict[str, Any]] = []
    for t in targets:
        if not pd.api.types.is_numeric_dtype(df[t]):
            continue
        y = pd.to_numeric(df[t], errors="coerce")
        for f in feats:
            x = pd.to_numeric(df[f], errors="coerce")
            valid = x.notna() & y.notna()
            if int(valid.sum()) < 5:
                continue
            xv = x[valid]
            yv = y[valid]
            # Avoid NumPy/Pandas runtime warnings from zero-variance vectors.
            if float(xv.std(ddof=0)) <= 1e-15 or float(yv.std(ddof=0)) <= 1e-15:
                continue
            corr = xv.corr(yv, method="pearson")
            if pd.isna(corr):
                continue
            rows.append(
                {
                    "bundle": bundle_data.bundle,
                    "target": t,
                    "feature": f,
                    "pearson_corr": float(corr),
                    "abs_corr": float(abs(corr)),
                    "high_corr_flag": abs(corr) > corr_threshold,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["high_corr_flag", "abs_corr"], ascending=[False, False]).reset_index(drop=True)


def _parse_lag_hours(col: str) -> int | None:
    # expects ..._lag_24h
    marker = "_lag_"
    if marker not in col:
        return None
    tail = col.split(marker)[-1]
    if not tail.endswith("h"):
        return None
    num = tail[:-1]
    return int(num) if num.isdigit() else None


def audit_causality_and_auction_gap(bundle_data: BundleData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if bundle_data.bundle != "da":
        return pd.DataFrame(rows)

    df = bundle_data.train
    feats = _feature_columns(df)
    suspect_keywords = ("actual", "realized")

    for c in feats:
        cl = c.lower()
        if not any(k in cl for k in suspect_keywords):
            continue
        lag_h = _parse_lag_hours(c)
        # DA auction-causal heuristic: actual/realized signals should not be near-real-time.
        # We flag anything with no lag suffix or lag < 24h as suspicious for DA.
        is_suspicious = (lag_h is None) or (lag_h < 24)
        rows.append(
            {
                "bundle": "da",
                "feature": c,
                "lag_h": lag_h,
                "suspicious_for_da_gate": bool(is_suspicious),
                "reason": "contains_actual_or_realized_and_is_not_day_ahead_lagged",
            }
        )

    # Generic forward-look naming heuristics (for T+1 prediction features at time T)
    forward_patterns = ("_lead_", "t+1", "future", "next_target")
    for c in feats:
        cl = c.lower()
        if any(p in cl for p in forward_patterns):
            rows.append(
                {
                    "bundle": "da",
                    "feature": c,
                    "lag_h": _parse_lag_hours(c),
                    "suspicious_for_da_gate": True,
                    "reason": "forward_looking_name_pattern",
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["suspicious_for_da_gate", "feature"], ascending=[False, True]).reset_index(drop=True)


def audit_sentinel_missing_constant(bundle_data: BundleData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        df = getattr(bundle_data, split)
        feats = _feature_columns(df) + _target_columns(df)
        for c in feats:
            s = df[c]
            if not pd.api.types.is_numeric_dtype(s):
                continue
            x = pd.to_numeric(s, errors="coerce")
            n = len(x)
            if n == 0:
                continue
            nan_pct = float(x.isna().mean() * 100.0)
            unique_non_null = int(x.dropna().nunique())
            var = float(x.var(ddof=0)) if x.notna().any() else np.nan
            const_flag = (unique_non_null <= 1) or (np.isfinite(var) and abs(var) < 1e-15)
            row = {
                "bundle": bundle_data.bundle,
                "split": split,
                "column": c,
                "nan_pct": nan_pct,
                "zero_variance_flag": bool(const_flag),
                "n_unique_non_null": unique_non_null,
            }
            for sv in NUM_SENTINELS:
                key = f"pct_eq_{str(int(sv)).replace('-', 'neg')}"
                row[key] = float((x == sv).mean() * 100.0)
            row["pct_eq_0"] = float((x == 0.0).mean() * 100.0)
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["zero_variance_flag", "nan_pct"], ascending=[False, False]).reset_index(drop=True)


def _last_sunday(year: int, month: int) -> date:
    d = date(year, month, 31)
    while d.weekday() != 6:  # Sunday
        d -= timedelta(days=1)
    return d


def audit_timezone_dst(bundle_data: BundleData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        df = getattr(bundle_data, split)
        if "timestamp_utc" not in df.columns:
            continue
        ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce").dropna().sort_values()
        if ts.empty:
            continue

        diffs = ts.diff().dropna()
        gap_count = int((diffs > pd.Timedelta(hours=1)).sum())
        dup_count = int((diffs == pd.Timedelta(0)).sum())
        irregular_count = int((diffs != pd.Timedelta(hours=1)).sum())
        rows.append(
            {
                "bundle": bundle_data.bundle,
                "split": split,
                "check": "utc_index_regular",
                "status": "warn" if (gap_count > 0 or dup_count > 0) else "ok",
                "details": json.dumps(
                    {
                        "duplicates": dup_count,
                        "gaps_gt_1h": gap_count,
                        "irregular_steps": irregular_count,
                    }
                ),
            }
        )

        years = sorted(set(ts.dt.year.tolist()))
        ts_local = ts.dt.tz_convert("Europe/Berlin")
        local_dates = ts_local.dt.date
        for y in years:
            mar = _last_sunday(y, 3)
            octo = _last_sunday(y, 10)
            for label, day, expected in (("dst_march", mar, 23), ("dst_october", octo, 25)):
                n = int((local_dates == day).sum())
                # If day absent (because split boundaries), don't flag as error.
                if n == 0:
                    status = "skip"
                else:
                    status = "ok" if n == expected else "warn"
                rows.append(
                    {
                        "bundle": bundle_data.bundle,
                        "split": split,
                        "check": label,
                        "status": status,
                        "details": json.dumps({"year": y, "local_date": str(day), "rows": n, "expected": expected}),
                    }
                )
    return pd.DataFrame(rows)


def audit_train_test_shift(bundle_data: BundleData, shift_threshold_pct: float) -> pd.DataFrame:
    train = bundle_data.train
    test = bundle_data.test
    targets = [c for c in _target_columns(train) if c in test.columns]
    rows: list[dict[str, Any]] = []
    for t in targets:
        tr = pd.to_numeric(train[t], errors="coerce")
        te = pd.to_numeric(test[t], errors="coerce")
        tr_mean, te_mean = float(tr.mean()), float(te.mean())
        tr_std, te_std = float(tr.std(ddof=0)), float(te.std(ddof=0))

        if np.isfinite(tr_mean) and abs(tr_mean) > 1e-12:
            mean_shift_pct = float(abs(te_mean - tr_mean) / abs(tr_mean) * 100.0)
        else:
            mean_shift_pct = math.inf if np.isfinite(te_mean) and abs(te_mean) > 1e-12 else 0.0

        warn = bool(mean_shift_pct > shift_threshold_pct)
        rows.append(
            {
                "bundle": bundle_data.bundle,
                "target": t,
                "train_mean": tr_mean,
                "test_mean": te_mean,
                "train_std": tr_std,
                "test_std": te_std,
                "mean_shift_pct": mean_shift_pct,
                "warn_nonstationary_shift": warn,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("mean_shift_pct", ascending=False).reset_index(drop=True)


def run_audit(base_dir: Path, out_dir: Path, corr_threshold: float, shift_threshold_pct: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summary: dict[str, Any] = {}
    for bundle in ("da", "afrr"):
        bd = _load_bundle(base_dir, bundle)

        corr = audit_correlation_leakage(bd, corr_threshold=corr_threshold)
        causality = audit_causality_and_auction_gap(bd)
        quality = audit_sentinel_missing_constant(bd)
        tz = audit_timezone_dst(bd)
        shift = audit_train_test_shift(bd, shift_threshold_pct=shift_threshold_pct)

        corr_p = out_dir / f"{bundle}_correlation_leakage.csv"
        caus_p = out_dir / f"{bundle}_causality_auction_check.csv"
        qual_p = out_dir / f"{bundle}_sentinel_missing_constant.csv"
        tz_p = out_dir / f"{bundle}_timezone_dst_check.csv"
        shift_p = out_dir / f"{bundle}_train_test_shift.csv"

        corr.to_csv(corr_p, index=False)
        causality.to_csv(caus_p, index=False)
        quality.to_csv(qual_p, index=False)
        tz.to_csv(tz_p, index=False)
        shift.to_csv(shift_p, index=False)

        high_corr = int(corr["high_corr_flag"].sum()) if not corr.empty else 0
        suspicious_da = int(causality["suspicious_for_da_gate"].sum()) if not causality.empty else 0
        zero_var = int(quality["zero_variance_flag"].sum()) if not quality.empty else 0
        tz_warn = int((tz["status"] == "warn").sum()) if not tz.empty else 0
        shift_warn = int(shift["warn_nonstationary_shift"].sum()) if not shift.empty else 0

        all_summary[bundle] = {
            "rows_train": int(len(bd.train)),
            "rows_val": int(len(bd.val)),
            "rows_test": int(len(bd.test)),
            "n_targets": int(len(_target_columns(bd.train))),
            "high_corr_flags": high_corr,
            "causality_flags": suspicious_da,
            "zero_variance_flags": zero_var,
            "timezone_warnings": tz_warn,
            "train_test_shift_warnings": shift_warn,
            "reports": {
                "correlation_leakage": str(corr_p),
                "causality_auction": str(caus_p),
                "sentinel_missing_constant": str(qual_p),
                "timezone_dst": str(tz_p),
                "train_test_shift": str(shift_p),
            },
        }

    summary_path = out_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")

    print("\n=== ML Data Integrity Audit Summary ===")
    print(json.dumps(all_summary, indent=2))
    print(f"\n[OK] Wrote summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit ML bundle integrity before training.")
    p.add_argument("--base-dir", default="data/model_input", help="Bundle base directory (contains da/afrr).")
    p.add_argument("--out-dir", default="data/reports/ml_data_audit", help="Output directory for audit reports.")
    p.add_argument("--corr-threshold", type=float, default=0.98, help="Absolute Pearson correlation leakage threshold.")
    p.add_argument(
        "--mean-shift-threshold-pct",
        type=float,
        default=20.0,
        help="Warn if |test_mean-train_mean| / |train_mean| exceeds this percentage.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_audit(
        base_dir=Path(args.base_dir),
        out_dir=Path(args.out_dir),
        corr_threshold=float(args.corr_threshold),
        shift_threshold_pct=float(args.mean_shift_threshold_pct),
    )


if __name__ == "__main__":
    main()
