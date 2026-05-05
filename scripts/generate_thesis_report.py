#!/usr/bin/env python3
"""Aggregate quantile-sweep simulation outputs into one thesis-ready table."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def _detect_model_from_run_id(run_id: str) -> str:
    rid = run_id.lower()
    if rid.startswith("xgb_") or "xgboost" in rid:
        return "xgboost"
    if rid.startswith("linear_") or "lear" in rid:
        return "linear"
    if rid.startswith("tft_"):
        return "tft"
    return "unknown"


def _collect_summaries(sim_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for csv_path in sorted(sim_root.glob("*/*/quantile_sweep_summary.csv")):
        # artifacts/simulation_runs/<run_id>/<split>/quantile_sweep_summary.csv
        split = csv_path.parent.name
        run_id = csv_path.parent.parent.name
        model = _detect_model_from_run_id(run_id)
        df = pd.read_csv(csv_path)
        df.insert(0, "model", model)
        df.insert(1, "run_id", run_id)
        df.insert(2, "split", split)
        df.insert(3, "source_csv", str(csv_path))
        rows.append(df)
    if not rows:
        raise RuntimeError(
            f"No quantile_sweep_summary.csv files found under {sim_root}. "
            "Run `sim-<model>` with SIM_QUANTILE_PAIRS first."
        )
    out = pd.concat(rows, ignore_index=True)
    sort_cols = [c for c in ["model", "run_id", "split", "scenario"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build thesis benchmark report from quantile sweep results.")
    p.add_argument("--sim-root", default="artifacts/simulation_runs", help="Root directory containing simulation runs.")
    p.add_argument("--out-csv", default="artifacts/thesis_benchmark_report.csv", help="Output CSV path.")
    p.add_argument("--out-md", default="artifacts/thesis_benchmark_report.md", help="Output markdown table path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sim_root = Path(args.sim_root)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    df = _collect_summaries(sim_root)
    keep_cols = [
        "model",
        "run_id",
        "split",
        "scenario",
        "quantile_low",
        "quantile_high",
        "da_quantile_role",
        "realized_total_pnl_eur",
        "predicted_total_pnl_eur",
        "naive_total_pnl_eur",
        "oracle_total_pnl_eur",
        "pnl_gap_total_eur",
        "cost_of_forecast_error_total_eur",
        "economic_opportunity_gap_ratio",
        "roi_on_max_capital",
        "output_dir",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    report = df[keep_cols].copy()
    report.to_csv(out_csv, index=False)
    out_md.write_text(report.to_markdown(index=False), encoding="utf-8")

    print(f"[OK] Thesis report written: {out_csv}")
    print(f"[OK] Thesis report written: {out_md}")
    print(f"[INFO] Rows: {len(report)}")


if __name__ == "__main__":
    main()

