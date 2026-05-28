#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate forecast benchmark outputs")
    p.add_argument("--benchmark-dir", required=True)
    p.add_argument("--min-join-coverage", type=float, default=0.999)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.benchmark_dir).resolve()
    req = [
        root / "benchmark_manifest.json",
        root / "input_manifest.json",
        root / "benchmark_config_resolved.yaml",
        root / "diagnostics" / "truth_mapping_report.csv",
        root / "diagnostics" / "join_coverage_report.csv",
        root / "diagnostics" / "schema_report.json",
        root / "diagnostics" / "benchmark_input_inventory.csv",
        root / "metrics" / "metrics_overall.csv",
        root / "metrics" / "metrics_by_target.csv",
        root / "metrics" / "metrics_by_model.csv",
        root / "metrics" / "metrics_by_lead.csv",
        root / "metrics" / "metrics_by_horizon_bucket.csv",
        root / "metrics" / "metrics_gate_time.csv",
        root / "metrics" / "metrics_tail_events.csv",
        root / "metrics" / "metrics_calibration.csv",
        root / "metrics" / "metrics_crossing.csv",
        root / "metrics" / "metrics_model_ranking_by_target.csv",
        root / "metrics" / "metrics_model_ranking_global_normalized.csv",
    ]
    errors: list[str] = []
    for p in req:
        if not p.exists():
            errors.append(f"missing required file: {p}")

    if errors:
        print("[FAIL] validation")
        for e in errors:
            print("-", e)
        return 1

    cov = pd.read_csv(root / "diagnostics" / "join_coverage_report.csv")
    if cov.empty:
        errors.append("join_coverage_report.csv is empty")
    if "join_coverage" not in cov.columns:
        errors.append("join_coverage_report.csv missing join_coverage")
    else:
        bad = cov[pd.to_numeric(cov["join_coverage"], errors="coerce") < float(args.min_join_coverage)]
        if not bad.empty:
            errors.append(f"join coverage below threshold in {len(bad)} rows")

    tm = pd.read_csv(root / "diagnostics" / "truth_mapping_report.csv")
    if (tm.get("status", pd.Series(dtype=str)) != "ok").any():
        errors.append("truth_mapping_report contains non-ok rows")

    overall = pd.read_csv(root / "metrics" / "metrics_overall.csv")
    if overall.duplicated(subset=[c for c in ["model", "target", "split"] if c in overall.columns]).any():
        errors.append("metrics_overall has duplicate model/target/split rows")

    crossing = pd.read_csv(root / "metrics" / "metrics_crossing.csv")
    for c in ["crossing_rate_before_repair", "max_crossing_violation_before_repair"]:
        if c not in crossing.columns:
            errors.append(f"metrics_crossing missing {c}")

    ranking = pd.read_csv(root / "metrics" / "metrics_model_ranking_global_normalized.csv")
    if any(c in ranking.columns for c in ["raw_mae_average", "raw_rmse_average"]):
        errors.append("global ranking appears to average raw MAE/RMSE")

    manifest = json.loads((root / "benchmark_manifest.json").read_text(encoding="utf-8"))
    for k in ["config_sha256", "quantiles", "package_versions"]:
        if k not in manifest:
            errors.append(f"benchmark_manifest missing {k}")

    for f in [overall, pd.read_csv(root / "metrics" / "metrics_by_target.csv")]:
        if f.empty:
            errors.append("metrics output unexpectedly empty")
            break
        num = f.select_dtypes(include=[np.number])
        if num.empty:
            errors.append("metrics table has no numeric columns")
            break

    if errors:
        print("[FAIL] validation")
        for e in errors:
            print("-", e)
        return 1

    print("[OK] Forecast benchmark validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
