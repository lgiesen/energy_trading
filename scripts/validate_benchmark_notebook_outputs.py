from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_FILES = {
    "benchmark_data_inventory.csv": [
        "target",
        "model",
        "split",
        "n_rows",
        "available_quantiles",
    ],
    "forecast_metrics_probabilistic.csv": [
        "target",
        "model",
        "split",
        "mae_p50",
        "rmse_p50",
        "mean_pinball",
        "crps_approx",
    ],
    "gate_time_forecast_metrics.csv": [
        "target",
        "model",
        "split",
        "n_obs",
        "mae_p50",
    ],
    "tail_performance_value_events.csv": [
        "target",
        "model",
        "split",
        "tail_bucket",
        "tail_mae",
    ],
    "joint_value_event_diagnostics.csv": [
        "target",
        "model",
        "split",
        "joint_event_recall",
    ],
    "quantile_pair_diagnostics.csv": [
        "scenario",
        "target",
        "model",
        "split",
        "selected_quantile",
    ],
    "quantile_pair_mapping.csv": [
        "scenario",
        "target",
        "selected_quantile",
    ],
    "model_selection_scores.csv": [
        "target",
        "model",
        "split",
        "final_composite_score",
    ],
    "final_model_recommendation_table.csv": [
        "target",
        "split",
        "recommended_model",
        "acceptable_for_simulation",
    ],
    "final_model_recommendation_table.md": None,
    "benchmark_notebook_summary.json": None,
}


def _validate_csv(path: Path, required_cols: list[str]) -> list[str]:
    errs: list[str] = []
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return [f"failed to read csv: {path} ({e})"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errs.append(f"missing columns in {path.name}: {missing}")
    if df.empty:
        errs.append(f"empty csv: {path.name}")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate benchmark notebook outputs")
    ap.add_argument("artifact_dir", help="artifact root, e.g. artifacts/benchmark")
    args = ap.parse_args()

    root = Path(args.artifact_dir)
    if not root.exists():
        raise FileNotFoundError(f"Artifact directory not found: {root}")

    errors: list[str] = []
    for rel, req_cols in REQUIRED_FILES.items():
        p = root / rel
        if not p.exists():
            errors.append(f"missing file: {p}")
            continue
        if req_cols is None:
            if p.suffix == ".json":
                try:
                    json.loads(p.read_text(encoding="utf-8"))
                except Exception as e:
                    errors.append(f"invalid json: {p} ({e})")
            elif p.suffix == ".md":
                txt = p.read_text(encoding="utf-8")
                if len(txt.strip()) == 0:
                    errors.append(f"empty markdown: {p}")
        else:
            errors.extend(_validate_csv(p, req_cols))

    if errors:
        print("[FAIL] benchmark notebook output validation")
        for e in errors:
            print("-", e)
        raise SystemExit(1)

    print("[OK] benchmark notebook output validation passed")


if __name__ == "__main__":
    main()
