#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_resolved_config(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8")
    try:
        import yaml

        obj = yaml.safe_load(txt)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        try:
            obj = json.loads(txt)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate forecast benchmark outputs")
    p.add_argument("--benchmark-dir", required=True)
    p.add_argument("--min-join-coverage", type=float, default=0.999)
    p.add_argument("--require-figures", action=argparse.BooleanOptionalAction, default=None)
    return p.parse_args()


def _nonempty_csv(path: Path, errors: list[str]) -> pd.DataFrame:
    if not path.exists():
        errors.append(f"missing required file: {path}")
        return pd.DataFrame()
    if path.stat().st_size == 0:
        errors.append(f"empty file: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        errors.append(f"csv is empty: {path}")
    return df


def main() -> int:
    args = parse_args()
    root = Path(args.benchmark_dir).resolve()
    errors: list[str] = []

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
        root / "metrics" / "metrics_residual_patterns.csv",
        root / "metrics" / "metrics_volatility_regimes.csv",
        root / "metrics" / "metrics_directional_bias.csv",
    ]
    for p in req:
        if not p.exists():
            errors.append(f"missing required file: {p}")

    if errors:
        print("[FAIL] validation")
        for e in errors:
            print("-", e)
        return 1

    cov = _nonempty_csv(root / "diagnostics" / "join_coverage_report.csv", errors)
    tm = _nonempty_csv(root / "diagnostics" / "truth_mapping_report.csv", errors)
    overall = _nonempty_csv(root / "metrics" / "metrics_overall.csv", errors)
    by_target = _nonempty_csv(root / "metrics" / "metrics_by_target.csv", errors)
    crossing = _nonempty_csv(root / "metrics" / "metrics_crossing.csv", errors)
    ranking = _nonempty_csv(root / "metrics" / "metrics_model_ranking_global_normalized.csv", errors)
    by_lead = _nonempty_csv(root / "metrics" / "metrics_by_lead.csv", errors)

    if "join_coverage" not in cov.columns:
        errors.append("join_coverage_report.csv missing join_coverage")
    else:
        bad = cov[pd.to_numeric(cov["join_coverage"], errors="coerce") < float(args.min_join_coverage)]
        if not bad.empty:
            errors.append(f"join coverage below threshold in {len(bad)} rows")

    if (tm.get("status", pd.Series(dtype=str)) != "ok").any():
        errors.append("truth_mapping_report contains non-ok rows")

    if overall.duplicated(subset=[c for c in ["model", "target", "split"] if c in overall.columns]).any():
        errors.append("metrics_overall has duplicate model/target/split rows")

    for c in ["crossing_rate_before_repair", "max_crossing_violation_before_repair"]:
        if c not in crossing.columns:
            errors.append(f"metrics_crossing missing {c}")

    if any(c in ranking.columns for c in ["raw_mae_average", "raw_rmse_average"]):
        errors.append("global ranking appears to average raw MAE/RMSE")

    manifest = json.loads((root / "benchmark_manifest.json").read_text(encoding="utf-8"))
    for k in ["config_sha256", "quantiles", "package_versions"]:
        if k not in manifest:
            errors.append(f"benchmark_manifest missing {k}")

    for f in [overall, by_target]:
        if f.empty:
            errors.append("metrics output unexpectedly empty")
            break
        num = f.select_dtypes(include=[np.number])
        if num.empty:
            errors.append("metrics table has no numeric columns")
            break

    cfg = _load_resolved_config(root / "benchmark_config_resolved.yaml")
    infer_require_figures = bool(manifest.get("figures_enabled", cfg.get("figures", {}).get("enabled", False)))
    require_figures = infer_require_figures if args.require_figures is None else bool(args.require_figures)
    save_joined = bool(manifest.get("save_joined_predictions", True))
    if save_joined:
        joined_dir = root / "diagnostics" / "joined_predictions"
        if not joined_dir.exists():
            errors.append("joined_predictions directory missing while save_joined_predictions=true")
        elif not list(joined_dir.glob("*.parquet")):
            errors.append("joined_predictions directory has no parquet files")

    if require_figures:
        if not (root / "diagnostics" / "example_window_report.csv").exists():
            errors.append("missing example_window_report.csv while figures required")
        if by_lead.empty:
            errors.append("metrics_by_lead empty while figures required")
        else:
            for (split, target), _ in by_lead.groupby(["split", "target"]):
                base = root / "figures" / str(split) / str(target)
                required = [
                    base / "leadtime_mae_p50.png",
                    base / "leadtime_mean_pinball.png",
                    base / "leadtime_approx_crps.png",
                    base / "calibration_curve.png",
                    base / "coverage_p10_p90_by_lead.png",
                    base / "interval_width_p10_p90_by_lead.png",
                    base / "error_by_volatility_bucket.png",
                    base / "interval_width_vs_realized_abs_error.png",
                ]
                for p in required:
                    if not p.exists():
                        errors.append(f"missing required figure: {p}")
                    elif p.stat().st_size == 0:
                        errors.append(f"zero-byte figure: {p}")
                sub = overall[(overall["split"] == split) & (overall["target"] == target)]
                for model in sub["model"].unique():
                    model_base = base / str(model)
                    required_model = [
                        model_base / "typical_week_forecast_band.png",
                        model_base / "high_volatility_week_forecast_band.png",
                        model_base / "spike_week_forecast_band.png",
                        model_base / "tail_event_scatter.png",
                        model_base / "residual_by_hour_of_day.png",
                        model_base / "residual_by_lead_time.png",
                        model_base / "residual_by_true_value_bin.png",
                    ]
                    for p in required_model:
                        if not p.exists():
                            errors.append(f"missing required figure: {p}")
                        elif p.stat().st_size == 0:
                            errors.append(f"zero-byte figure: {p}")

    if errors:
        print("[FAIL] validation")
        for e in errors:
            print("-", e)
        return 1

    print("[OK] Forecast benchmark validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
