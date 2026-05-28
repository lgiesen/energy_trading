#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED_FILES = [
    "benchmark_summary.json",
    "benchmark_manifest.json",
    "data_inventory.csv",
    "truth_mapping_audit.csv",
    "benchmark_truth_mapping_audit.csv",
    "forecast_metrics_probabilistic.csv",
    "gate_time_forecast_metrics.csv",
    "tail_performance_value_events.csv",
    "joint_value_event_diagnostics.csv",
    "quantile_pair_diagnostics.csv",
    "quantile_pair_mapping.csv",
    "model_selection_scores.csv",
    "final_model_recommendation_table.csv",
    "final_model_recommendation_table.md",
    "benchmark_score_weights.json",
]

REQUIRED_COLS = {
    "data_inventory.csv": ["model", "target", "split", "join_coverage_pct", "resolved_truth_col", "status"],
    "forecast_metrics_probabilistic.csv": ["model", "target", "split", "mae_p50", "rmse_p50", "mean_pinball_loss", "approx_crps"],
    "gate_time_forecast_metrics.csv": ["model", "target", "split", "status"],
    "tail_performance_value_events.csv": ["model", "target", "tail_side", "tail_quantile", "tail_mae"],
    "joint_value_event_diagnostics.csv": ["model", "joint_event", "joint_event_f1"],
    "quantile_pair_diagnostics.csv": ["model", "target", "scenario", "selected_quantile"],
    "quantile_pair_mapping.csv": ["scenario", "target", "selected_quantile"],
    "model_selection_scores.csv": ["model", "target", "final_composite_score"],
    "final_model_recommendation_table.csv": ["target", "recommended_model", "acceptable_for_simulation"],
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate forecast benchmark output directory")
    ap.add_argument("output_dir", help="artifact output dir")
    ap.add_argument("--coverage-threshold", type=float, default=0.999)
    ap.add_argument("--expect-plots", action="store_true")
    args = ap.parse_args()

    root = Path(args.output_dir)
    errors: list[str] = []

    for f in REQUIRED_FILES:
        if not (root / f).exists():
            errors.append(f"missing file: {root / f}")

    dfs: dict[str, pd.DataFrame] = {}
    for f, cols in REQUIRED_COLS.items():
        p = root / f
        if not p.exists():
            continue
        df = pd.read_csv(p)
        dfs[f] = df
        if df.empty:
            errors.append(f"empty table: {p}")
        miss = [c for c in cols if c not in df.columns]
        if miss:
            errors.append(f"missing required columns in {p.name}: {miss}")

    inv = dfs.get("data_inventory.csv")
    if inv is not None and "join_coverage_pct" in inv.columns:
        bad = inv[pd.to_numeric(inv["join_coverage_pct"], errors="coerce") < (args.coverage_threshold * 100.0)]
        if not bad.empty:
            errors.append(f"join coverage below threshold for {len(bad)} rows")

    if args.expect_plots:
        fig_dir = root / "figures"
        if not fig_dir.exists():
            errors.append("plots expected but figures/ missing")
        else:
            pngs = list(fig_dir.rglob("*.png"))
            if not pngs:
                errors.append("plots expected but no png files found")

    sum_path = root / "benchmark_summary.json"
    if sum_path.exists():
        s = json.loads(sum_path.read_text(encoding="utf-8"))
        for k in ["timestamp_utc", "script_version", "models_evaluated", "targets_evaluated"]:
            if k not in s:
                errors.append(f"benchmark_summary.json missing key: {k}")

    if errors:
        print("[FAIL] forecast benchmark output validation")
        for e in errors:
            print("-", e)
        return 1

    print("[OK] forecast benchmark output validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
