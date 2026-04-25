#!/usr/bin/env python3
"""Pre-simulation prediction preflight validator.

Checks:
1) required columns present,
2) NaN thresholds,
3) quantile monotonicity,
4) value-range sanity alerts by target.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_trading.simulation.battery_backtest import CANONICAL_PREDICTION_COLUMNS  # noqa: E402


QCOL_RE = re.compile(r"^(?P<base>pred_[a-z0-9_]+)_p(?P<q>\d{2})$", re.IGNORECASE)


@dataclass(frozen=True)
class RangeRule:
    low: float
    high: float
    hard: bool = False  # True => fail when violated; False => alert only


SANITY_RULES: dict[str, RangeRule] = {
    # Hard rules
    "pred_afrr_activation_rate_pos": RangeRule(0.0, 1.0, hard=True),
    "pred_afrr_activation_rate_neg": RangeRule(0.0, 1.0, hard=True),
    # Soft sanity alerts (config/data dependent)
    "pred_da_price": RangeRule(-1000.0, 5000.0, hard=False),
    "pred_afrr_capacity_price_pos": RangeRule(-1000.0, 10000.0, hard=False),
    "pred_afrr_capacity_price_neg": RangeRule(-1000.0, 10000.0, hard=False),
    "pred_afrr_activation_price_pos": RangeRule(-5000.0, 15000.0, hard=False),
    "pred_afrr_activation_price_neg": RangeRule(-5000.0, 15000.0, hard=False),
}
ACTIVATION_RATE_PRED_COLS = {
    "pred_afrr_activation_rate_pos",
    "pred_afrr_activation_rate_neg",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_existing_file(path_like: str | Path, *, manifest_dir: Path) -> Path:
    p = Path(path_like)
    if p.exists():
        return p
    cands = [
        manifest_dir / p.name,
        manifest_dir / "predictions" / p.name,
        REPO_ROOT / "data" / "features" / p.name,
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not resolve file from '{path_like}'.")


def _load_manifest_predictions(run_dir: Path, split: str) -> tuple[pd.DataFrame, dict[str, Path]]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    bundles = manifest.get("bundles", {})
    pred_paths: dict[str, Path] = {}
    for b in ("da", "afrr"):
        p = bundles.get(b, {}).get("predictions", {}).get(split)
        if p:
            pred_paths[b] = _resolve_existing_file(p, manifest_dir=manifest_path.parent)
    if not pred_paths:
        raise FileNotFoundError(f"No prediction files for split='{split}' in manifest.")

    frames: list[pd.DataFrame] = []
    for _, path in pred_paths.items():
        df = pd.read_parquet(path)
        if "timestamp_utc" not in df.columns:
            raise KeyError(f"{path} missing timestamp_utc")
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp_utc"]).copy()
        frames.append(df)
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="timestamp_utc", how="outer", suffixes=("", "_dup"))
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        merged = merged.drop(columns=dup_cols, errors="ignore")
    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)
    return merged, pred_paths


def _check_required_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    missing = [c for c in CANONICAL_PREDICTION_COLUMNS if c not in df.columns]
    return len(missing) == 0, missing


def _nan_profile(df: pd.DataFrame, columns: list[str], threshold: float) -> tuple[pd.DataFrame, bool]:
    rows = []
    ok = True
    n = max(1, len(df))
    for c in columns:
        ser = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series([np.nan] * len(df))
        nan_count = int(ser.isna().sum())
        nan_ratio = float(nan_count / n)
        row_ok = bool(nan_ratio <= threshold)
        ok = ok and row_ok
        rows.append(
            {
                "column": c,
                "nan_count": nan_count,
                "nan_ratio": nan_ratio,
                "threshold": threshold,
                "status": "ok" if row_ok else "fail",
            }
        )
    return pd.DataFrame(rows), ok


def _check_quantile_monotonicity_wide(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    grouped: dict[str, dict[int, str]] = {}
    for c in df.columns:
        m = QCOL_RE.match(c)
        if not m:
            continue
        grouped.setdefault(m.group("base"), {})[int(m.group("q"))] = c

    rows = []
    all_ok = True
    for base, qmap in sorted(grouped.items()):
        qkeys = sorted(qmap.keys())
        if len(qkeys) < 2:
            continue
        violations = 0
        checked = 0
        for q1, q2 in zip(qkeys[:-1], qkeys[1:]):
            a = pd.to_numeric(df[qmap[q1]], errors="coerce")
            b = pd.to_numeric(df[qmap[q2]], errors="coerce")
            m = a.notna() & b.notna()
            checked += int(m.sum())
            violations += int((a[m] > b[m]).sum())
        ok = violations == 0
        all_ok = all_ok and ok
        rows.append(
            {
                "series_base": base,
                "quantiles": ",".join([str(q) for q in qkeys]),
                "rows_checked": checked,
                "violations": violations,
                "status": "ok" if ok else "fail",
            }
        )
    return pd.DataFrame(rows), all_ok


def _check_quantile_monotonicity_long(run_dir: Path, split: str) -> tuple[pd.DataFrame, bool]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return pd.DataFrame(), True
    manifest = _read_json(manifest_path)
    pmap: dict[str, str] = (
        manifest.get("bundles", {}).get("afrr", {}).get("predictions_long", {}).get(split, {}) or {}
    )
    pmap_da: dict[str, str] = (
        manifest.get("bundles", {}).get("da", {}).get("predictions_long", {}).get(split, {}) or {}
    )
    pmap_all = dict(pmap_da)
    pmap_all.update(pmap)

    rows = []
    all_ok = True
    for pred_col, path_raw in sorted(pmap_all.items()):
        path = _resolve_existing_file(path_raw, manifest_dir=manifest_path.parent)
        df = pd.read_parquet(path)
        qcols = [c for c in df.columns if re.match(r"^p\d{2}$", c)]
        qkeys = sorted([int(c[1:]) for c in qcols if c[1:].isdigit()])
        qcols_sorted = [f"p{q:02d}" for q in qkeys if f"p{q:02d}" in df.columns]
        if len(qcols_sorted) < 2:
            continue
        checked = 0
        violations = 0
        for c1, c2 in zip(qcols_sorted[:-1], qcols_sorted[1:]):
            a = pd.to_numeric(df[c1], errors="coerce")
            b = pd.to_numeric(df[c2], errors="coerce")
            m = a.notna() & b.notna()
            checked += int(m.sum())
            violations += int((a[m] > b[m]).sum())
        ok = violations == 0
        all_ok = all_ok and ok
        rows.append(
            {
                "prediction_column": pred_col,
                "file": str(path),
                "quantiles": ",".join(qcols_sorted),
                "rows_checked": checked,
                "violations": violations,
                "status": "ok" if ok else "fail",
            }
        )
    return pd.DataFrame(rows), all_ok


def _range_sanity(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    rows = []
    hard_ok = True
    dynamic_rules = dict(SANITY_RULES)
    # Extend hard [0,1] rules to any wide quantile columns if present.
    for base in ACTIVATION_RATE_PRED_COLS:
        for c in df.columns:
            if c.startswith(base + "_p"):
                dynamic_rules[c] = RangeRule(0.0, 1.0, hard=True)

    for c, rule in dynamic_rules.items():
        if c not in df.columns:
            rows.append(
                {
                    "column": c,
                    "rule_low": rule.low,
                    "rule_high": rule.high,
                    "hard_rule": rule.hard,
                    "rows_checked": 0,
                    "violations": 0,
                    "violation_ratio": np.nan,
                    "status": "missing",
                }
            )
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        m = s.notna()
        checked = int(m.sum())
        if checked == 0:
            viol = 0
            ratio = np.nan
        else:
            viol_mask = (s[m] < rule.low) | (s[m] > rule.high)
            viol = int(viol_mask.sum())
            ratio = float(viol / checked)
        status = "ok" if viol == 0 else ("fail" if rule.hard else "alert")
        if rule.hard and viol > 0:
            hard_ok = False
        rows.append(
            {
                "column": c,
                "rule_low": rule.low,
                "rule_high": rule.high,
                "hard_rule": rule.hard,
                "rows_checked": checked,
                "violations": viol,
                "violation_ratio": ratio,
                "status": status,
            }
        )
    return pd.DataFrame(rows), hard_ok


def _range_sanity_long_activation_quantiles(run_dir: Path, split: str) -> tuple[pd.DataFrame, bool]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return pd.DataFrame(), True
    manifest = _read_json(manifest_path)
    pmap_afrr: dict[str, str] = (
        manifest.get("bundles", {}).get("afrr", {}).get("predictions_long", {}).get(split, {}) or {}
    )
    rows = []
    hard_ok = True
    for pred_col, path_raw in sorted(pmap_afrr.items()):
        if pred_col not in ACTIVATION_RATE_PRED_COLS:
            continue
        path = _resolve_existing_file(path_raw, manifest_dir=manifest_path.parent)
        df = pd.read_parquet(path)
        qcols = [c for c in df.columns if re.match(r"^p\d{2}$", c)]
        check_cols = ["predicted_value", *sorted(qcols)]
        check_cols = [c for c in check_cols if c in df.columns]
        for c in check_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            m = s.notna()
            checked = int(m.sum())
            if checked == 0:
                viol = 0
                ratio = np.nan
            else:
                vm = (s[m] < 0.0) | (s[m] > 1.0)
                viol = int(vm.sum())
                ratio = float(viol / checked)
            status = "ok" if viol == 0 else "fail"
            if viol > 0:
                hard_ok = False
            rows.append(
                {
                    "prediction_column": pred_col,
                    "file": str(path),
                    "column": c,
                    "rows_checked": checked,
                    "violations": viol,
                    "violation_ratio": ratio,
                    "rule_low": 0.0,
                    "rule_high": 1.0,
                    "hard_rule": True,
                    "status": status,
                }
            )
    return pd.DataFrame(rows), hard_ok


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate predictions before running simulation.")
    p.add_argument("--run-dir", required=True, help="Model run dir containing manifest.json.")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--nan-threshold", type=float, default=0.05, help="Max allowed NaN ratio per required column.")
    p.add_argument("--out-dir", default="", help="Output dir (default: <run-dir>/preflight_<split>).")
    p.add_argument("--fail-on-alert", action="store_true", help="Treat soft range alerts as failures.")
    return p


def main() -> None:
    args = _build_cli().parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / f"preflight_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_pred, pred_paths = _load_manifest_predictions(run_dir, args.split)
    req_ok, req_missing = _check_required_columns(merged_pred)
    nan_df, nan_ok = _nan_profile(merged_pred, CANONICAL_PREDICTION_COLUMNS, threshold=float(args.nan_threshold))
    qwide_df, qwide_ok = _check_quantile_monotonicity_wide(merged_pred)
    qlong_df, qlong_ok = _check_quantile_monotonicity_long(run_dir, args.split)
    range_df, range_hard_ok = _range_sanity(merged_pred)
    long_rate_range_df, long_rate_range_ok = _range_sanity_long_activation_quantiles(run_dir, args.split)
    range_alerts = bool((range_df["status"] == "alert").any()) if not range_df.empty else False

    final_ok = req_ok and nan_ok and qwide_ok and qlong_ok and range_hard_ok and long_rate_range_ok
    if args.fail_on_alert:
        final_ok = final_ok and (not range_alerts)

    summary = {
        "run_dir": str(run_dir.resolve()),
        "split": args.split,
        "prediction_paths": {k: str(v.resolve()) for k, v in pred_paths.items()},
        "rows_merged_predictions": int(len(merged_pred)),
        "required_columns_ok": bool(req_ok),
        "required_columns_missing": req_missing,
        "nan_threshold": float(args.nan_threshold),
        "nan_threshold_ok": bool(nan_ok),
        "quantile_monotonicity_wide_ok": bool(qwide_ok),
        "quantile_monotonicity_long_ok": bool(qlong_ok),
        "range_hard_rules_ok": bool(range_hard_ok),
        "activation_rate_long_range_ok": bool(long_rate_range_ok),
        "range_alerts_present": bool(range_alerts),
        "final_ok": bool(final_ok),
    }

    (out_dir / "preflight_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    nan_df.to_csv(out_dir / "preflight_nan_profile.csv", index=False)
    qwide_df.to_csv(out_dir / "preflight_quantile_monotonicity_wide.csv", index=False)
    qlong_df.to_csv(out_dir / "preflight_quantile_monotonicity_long.csv", index=False)
    range_df.to_csv(out_dir / "preflight_value_range_sanity.csv", index=False)
    long_rate_range_df.to_csv(out_dir / "preflight_activation_rate_long_range_sanity.csv", index=False)

    print("[OK] Preflight validation complete.")
    print(f"- final_ok: {summary['final_ok']}")
    print(f"- out_dir: {out_dir}")
    print(f"- summary: {out_dir / 'preflight_summary.json'}")
    print(f"- nan_profile: {out_dir / 'preflight_nan_profile.csv'}")
    print(f"- quantile_wide: {out_dir / 'preflight_quantile_monotonicity_wide.csv'}")
    print(f"- quantile_long: {out_dir / 'preflight_quantile_monotonicity_long.csv'}")
    print(f"- range_sanity: {out_dir / 'preflight_value_range_sanity.csv'}")
    print(f"- activation_rate_long_range_sanity: {out_dir / 'preflight_activation_rate_long_range_sanity.csv'}")


if __name__ == "__main__":
    main()
