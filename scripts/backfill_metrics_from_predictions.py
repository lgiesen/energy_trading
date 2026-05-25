#!/usr/bin/env python3
"""Backfill model metrics JSONs from existing prediction artifacts.

This script recomputes forecast metric suites with the current evaluator
(including newly added metrics such as CRPS/tradeoff diagnostics) and
silently overwrites existing metrics JSON files in-place.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def _load_compute_forecast_metrics():
    metrics_path = SRC_DIR / "energy_trading" / "evaluation" / "metrics.py"
    spec = importlib.util.spec_from_file_location("metrics_module", metrics_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load metrics module: {metrics_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_forecast_metrics


compute_forecast_metrics = _load_compute_forecast_metrics()


TARGET_TO_PRED = {
    "target_da_price": "pred_da_price",
    "target_afrr_activation_price_vwap_pos": "pred_afrr_activation_price_pos",
    "target_afrr_activation_price_vwap_neg": "pred_afrr_activation_price_neg",
    "target_afrr_capacity_price_pos": "pred_afrr_capacity_price_pos",
    "target_afrr_capacity_price_neg": "pred_afrr_capacity_price_neg",
    "target_afrr_activation_rate_pos": "pred_afrr_activation_rate_pos",
    "target_afrr_activation_rate_neg": "pred_afrr_activation_rate_neg",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_existing_file(path_like: str | Path, *, run_dir: Path, subdir_hint: str | None = None) -> Path:
    p = Path(path_like)
    if p.exists():
        return p
    cands: list[Path] = [run_dir / p.name]
    if subdir_hint:
        cands.append(run_dir / subdir_hint / p.name)
    cands.extend(
        [
            REPO_ROOT / "data" / "features" / p.name,
            REPO_ROOT / "data" / "model_input" / p.name,
        ]
    )
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not resolve file from '{path_like}'")


def _load_truth_df(run_dir: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    gt = manifest.get("ground_truth", {}) if isinstance(manifest.get("ground_truth"), dict) else {}
    gt_path_like = gt.get("default_path", "data/features/all_data_features.parquet")
    gt_path = _resolve_existing_file(gt_path_like, run_dir=run_dir)
    truth = pd.read_parquet(gt_path)
    if "timestamp_utc" not in truth.columns:
        raise KeyError(f"Ground truth file missing 'timestamp_utc': {gt_path}")
    truth = truth.copy()
    truth["timestamp_utc"] = pd.to_datetime(truth["timestamp_utc"], utc=True, errors="coerce")
    truth = truth.dropna(subset=["timestamp_utc"]).copy()
    return truth


def _suite_from_long(pred_long_path: Path, *, truth_df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    df = pd.read_parquet(pred_long_path)
    if "target_time_utc" not in df.columns:
        raise KeyError(f"Prediction file missing 'target_time_utc': {pred_long_path}")
    work = df.copy()
    work["target_time_utc"] = pd.to_datetime(work["target_time_utc"], utc=True, errors="coerce")
    work = work.dropna(subset=["target_time_utc"]).copy()
    if "lead_time_h" in work.columns:
        lead1 = work.loc[pd.to_numeric(work["lead_time_h"], errors="coerce") == 1].copy()
        if not lead1.empty:
            work = lead1

    if target_col not in truth_df.columns:
        raise KeyError(f"Target column not found in truth data: {target_col}")

    merged = work.merge(
        truth_df.loc[:, ["timestamp_utc", target_col]],
        left_on="target_time_utc",
        right_on="timestamp_utc",
        how="left",
    )
    merged = merged.rename(columns={target_col: "y_true"})

    y_pred_col = "p50" if "p50" in merged.columns else "predicted_value"
    if y_pred_col not in merged.columns:
        raise KeyError(f"Missing both 'p50' and 'predicted_value' in {pred_long_path}")
    merged = merged.rename(columns={y_pred_col: "y_pred"})

    return compute_forecast_metrics(merged, y_true_col="y_true", y_pred_col="y_pred")


def _update_metric_json(
    metrics_path: Path,
    *,
    target_col: str,
    pred_long_val: Path | None,
    pred_long_test: Path | None,
    truth_df: pd.DataFrame,
) -> None:
    metrics = _read_json(metrics_path)

    val_suite = _suite_from_long(pred_long_val, truth_df=truth_df, target_col=target_col) if pred_long_val else None
    test_suite = _suite_from_long(pred_long_test, truth_df=truth_df, target_col=target_col) if pred_long_test else None

    if val_suite is not None:
        metrics["val_metric_suite_h1"] = val_suite
        metrics["metric_suite_val"] = val_suite
        for k, v in val_suite.items():
            metrics[f"{k}_val_h1"] = v

    if test_suite is not None:
        metrics["test_metric_suite_h1"] = test_suite
        metrics["metric_suite_test"] = test_suite
        for k, v in test_suite.items():
            metrics[f"{k}_test_h1"] = v

    primary_suite = val_suite if val_suite is not None else test_suite
    if primary_suite is not None:
        metrics["metric_suite_h1"] = primary_suite
        for k, v in primary_suite.items():
            metrics[f"{k}_h1"] = v

    per_target = metrics.get("per_target_metrics")
    if isinstance(per_target, dict) and target_col in per_target and isinstance(per_target[target_col], dict):
        target_metrics = per_target[target_col]
        if primary_suite is not None:
            target_metrics["metric_suite_h1"] = primary_suite
            for k, v in primary_suite.items():
                target_metrics[f"{k}_h1"] = v
        per_target[target_col] = target_metrics
        metrics["per_target_metrics"] = per_target

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _iter_metrics_targets(run_dir: Path, manifest: dict[str, Any]) -> list[tuple[Path, str, Path | None, Path | None]]:
    out: list[tuple[Path, str, Path | None, Path | None]] = []
    bundles = manifest.get("bundles", {}) if isinstance(manifest.get("bundles"), dict) else {}

    for bundle_name, bundle_cfg in bundles.items():
        if not isinstance(bundle_cfg, dict):
            continue
        split_long = bundle_cfg.get("predictions_long", {}) if isinstance(bundle_cfg.get("predictions_long"), dict) else {}
        val_map = split_long.get("val", {}) if isinstance(split_long.get("val"), dict) else {}
        test_map = split_long.get("test", {}) if isinstance(split_long.get("test"), dict) else {}

        metric_paths: list[str] = []
        if isinstance(bundle_cfg.get("metrics_path"), str):
            metric_paths.append(bundle_cfg["metrics_path"])
        if isinstance(bundle_cfg.get("metrics_paths"), list):
            metric_paths.extend([str(p) for p in bundle_cfg["metrics_paths"] if isinstance(p, str)])

        for mp_like in metric_paths:
            mp = _resolve_existing_file(mp_like, run_dir=run_dir, subdir_hint="metrics")
            m = _read_json(mp)
            target_col = m.get("target_col")
            if not isinstance(target_col, str) or not target_col:
                continue
            pred_col = TARGET_TO_PRED.get(target_col)
            if pred_col is None:
                continue

            pval = val_map.get(pred_col)
            ptest = test_map.get(pred_col)
            val_path = _resolve_existing_file(pval, run_dir=run_dir, subdir_hint="predictions") if isinstance(pval, str) else None
            test_path = _resolve_existing_file(ptest, run_dir=run_dir, subdir_hint="predictions") if isinstance(ptest, str) else None
            out.append((mp, target_col, val_path, test_path))

    return out


def _backfill_run(run_dir: Path) -> tuple[int, int]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return 0, 0
    manifest = _read_json(manifest_path)
    truth_df = _load_truth_df(run_dir, manifest)
    items = _iter_metrics_targets(run_dir, manifest)

    updated = 0
    failed = 0
    for mp, target_col, pval, ptest in items:
        try:
            _update_metric_json(
                mp,
                target_col=target_col,
                pred_long_val=pval,
                pred_long_test=ptest,
                truth_df=truth_df,
            )
            updated += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[WARN] failed: {mp} ({exc})")
    return updated, failed


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill metrics JSON files from existing prediction artifacts.")
    p.add_argument("--run-dir", default="", help="Optional single model run dir containing manifest.json")
    p.add_argument("--run-root", default="artifacts/model_runs", help="Root dir to scan for model runs")
    args = p.parse_args()

    runs: list[Path] = []
    if args.run_dir:
        runs = [Path(args.run_dir)]
    else:
        root = Path(args.run_root)
        runs = sorted([d for d in root.iterdir() if d.is_dir() and (d / "manifest.json").exists()])

    total_updated = 0
    total_failed = 0
    for run in runs:
        updated, failed = _backfill_run(run)
        total_updated += updated
        total_failed += failed
        print(f"[OK] {run}: updated={updated} failed={failed}")

    print(f"[DONE] updated={total_updated} failed={total_failed}")


if __name__ == "__main__":
    main()
