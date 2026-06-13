"""Audit BEM submission, rejection and execution activity from simulation artifacts.

This script is read-only. It accepts either one scenario directory or a run root
and discovers scenario folders containing backtest/naive/RHPF hourly parquet
files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_battery_backtest import build_bem_activity_daily, summarize_bem_activity


def _scenario_dirs(root: Path) -> list[Path]:
    if (root / "backtest_hourly.parquet").exists():
        return [root]
    candidates = sorted(p for p in root.rglob("backtest_hourly.parquet") if p.is_file())
    return [p.parent for p in candidates]


def _load_hourly(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        print(f"[WARN] failed to read {path}: {exc}")
        return None


def _hourly_by_path(scenario_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "model": "model_hourly.parquet",
        "naive": "naive_hourly.parquet",
        "rhpf": "rolling_pf_hourly.parquet",
    }
    out: dict[str, pd.DataFrame] = {}
    model = _load_hourly(scenario_dir / files["model"])
    if model is None:
        model = _load_hourly(scenario_dir / "backtest_hourly.parquet")
    if model is not None:
        out["model"] = model
    for path_type in ("naive", "rhpf"):
        frame = _load_hourly(scenario_dir / files[path_type])
        if frame is not None:
            out[path_type] = frame
    return out


def _print_summary(scenario_dir: Path, daily: pd.DataFrame, max_rows: int) -> bool:
    summary_path = scenario_dir / "backtest_summary.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    stats = summarize_bem_activity(daily)
    sim_valid = summary.get("simulation_valid", "")
    thesis = summary.get("thesis_reportable", "")
    invalid = summary.get("invalid_reason", "") or "none"
    print(f"\nScenario: {scenario_dir}")
    print(f"validity: simulation_valid={sim_valid} thesis_reportable={thesis} invalid_reason={invalid}")
    for path_type in ("model", "naive", "rhpf"):
        sub = stats.get(f"{path_type}_bem_submitted_pos_mw_sum", 0.0) + stats.get(
            f"{path_type}_bem_submitted_neg_mw_sum", 0.0
        )
        rej = stats.get(f"{path_type}_bem_rejected_pos_mw_sum", 0.0) + stats.get(
            f"{path_type}_bem_rejected_neg_mw_sum", 0.0
        )
        exe = stats.get(f"{path_type}_bem_executed_pos_mwh_sum", 0.0) + stats.get(
            f"{path_type}_bem_executed_neg_mwh_sum", 0.0
        )
        rev = stats.get(f"{path_type}_bem_activation_revenue_eur_sum", 0.0)
        mismatch = stats.get(f"{path_type}_bem_bid_price_mismatch_count_sum", 0.0)
        stopped = stats.get(f"{path_type}_bem_stopped_submitting_before_end", 0.0)
        last = stats.get(f"{path_type}_bem_last_submission_date", "")
        print(
            f"{path_type:>5}: submitted_mw={float(sub):.3f} rejected_mw={float(rej):.3f} "
            f"executed_mwh={float(exe):.3f} revenue_eur={float(rev):.2f} "
            f"bid_price_mismatch_rows={float(mismatch):.0f} last_submission={last or '-'} "
            f"stopped_before_end={int(float(stopped))}"
        )
    cols = [
        "path_type",
        "date",
        "submitted_pos_mw",
        "submitted_neg_mw",
        "rejected_pos_mw",
        "rejected_neg_mw",
        "accepted_pos_mw",
        "accepted_neg_mw",
        "executed_pos_mwh",
        "executed_neg_mwh",
        "activation_revenue_eur",
        "bid_price_mismatch_count",
        "bem_activity_classification",
    ]
    shown = daily[[c for c in cols if c in daily.columns]].copy()
    print(shown.tail(max_rows).to_string(index=False))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Scenario directory or simulation run root.")
    parser.add_argument("--max-rows", type=int, default=40, help="Maximum daily rows to print per scenario.")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional path for concatenated daily audit CSV.")
    args = parser.parse_args()

    scenarios = _scenario_dirs(args.path)
    if not scenarios:
        print(f"FAIL: no scenario folders with backtest_hourly.parquet found below {args.path}")
        return 2
    all_daily: list[pd.DataFrame] = []
    for scenario_dir in scenarios:
        daily = build_bem_activity_daily(_hourly_by_path(scenario_dir))
        if daily.empty:
            print(f"\nScenario: {scenario_dir}\n[WARN] no BEM activity columns available")
            continue
        daily.insert(0, "scenario_dir", str(scenario_dir))
        all_daily.append(daily)
        _print_summary(scenario_dir, daily.drop(columns=["scenario_dir"]), max_rows=int(args.max_rows))
    if args.csv_out is not None and all_daily:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(all_daily, ignore_index=True, sort=False).to_csv(args.csv_out, index=False)
        print(f"\n[OK] wrote {args.csv_out}")
    print(f"\nPASS scenarios_checked={len(scenarios)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
