#!/usr/bin/env python3
"""Compute post-hoc decision quality metrics via Kendall's tau.

Best-practice usage: run after simulation on backtest_hourly artifacts.
Default score pair:
- predicted decision score: pred_pnl_eur
- realized economic score: real_pnl_eur
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def _resolve_hourly_path(hourly_path: str, simulation_dir: str) -> Path:
    if hourly_path.strip():
        p = Path(hourly_path.strip())
    else:
        if not simulation_dir.strip():
            raise ValueError("Provide either --hourly-path or --simulation-dir.")
        p = Path(simulation_dir.strip()) / "backtest_hourly.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Hourly file not found: {p}")
    return p


def _safe_tau(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    xs = pd.to_numeric(x, errors="coerce")
    ys = pd.to_numeric(y, errors="coerce")
    m = xs.notna() & ys.notna() & np.isfinite(xs) & np.isfinite(ys)
    if int(m.sum()) < 2:
        return float("nan"), float("nan")
    tau, pval = kendalltau(xs[m].to_numpy(dtype=float), ys[m].to_numpy(dtype=float))
    return float(tau), float(pval)


def main() -> None:
    p = argparse.ArgumentParser(description="Post-hoc Kendall tau decision-quality report.")
    p.add_argument("--hourly-path", default="", help="Path to backtest_hourly.parquet")
    p.add_argument("--simulation-dir", default="", help="Directory containing backtest_hourly.parquet")
    p.add_argument("--timestamp-col", default="timestamp_utc")
    p.add_argument("--pred-score-col", default="pred_pnl_eur")
    p.add_argument("--real-score-col", default="real_pnl_eur")
    p.add_argument("--tau-suff-threshold", type=float, default=0.90)
    p.add_argument("--min-points-per-day", type=int, default=12)
    p.add_argument("--out-json", default="", help="Output JSON path (default: <sim_dir>/decision_quality_kendall.json)")
    p.add_argument("--out-csv", default="", help="Output daily CSV path (default: <sim_dir>/decision_quality_kendall_daily.csv)")
    args = p.parse_args()

    hourly_path = _resolve_hourly_path(args.hourly_path, args.simulation_dir)
    sim_dir = hourly_path.parent
    out_json = Path(args.out_json) if args.out_json.strip() else sim_dir / "decision_quality_kendall.json"
    out_csv = Path(args.out_csv) if args.out_csv.strip() else sim_dir / "decision_quality_kendall_daily.csv"

    df = pd.read_parquet(hourly_path)
    for c in (args.timestamp_col, args.pred_score_col, args.real_score_col):
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' in {hourly_path}")

    d = df[[args.timestamp_col, args.pred_score_col, args.real_score_col]].copy()
    d[args.timestamp_col] = pd.to_datetime(d[args.timestamp_col], utc=True, errors="coerce")
    d = d.dropna(subset=[args.timestamp_col]).sort_values(args.timestamp_col).reset_index(drop=True)
    d["date_utc"] = d[args.timestamp_col].dt.floor("D")

    tau_global, pval_global = _safe_tau(d[args.pred_score_col], d[args.real_score_col])
    pair_agreement = pd.to_numeric(d[args.pred_score_col], errors="coerce") * pd.to_numeric(
        d[args.real_score_col], errors="coerce"
    )
    sign_agreement_rate = float((pair_agreement >= 0).mean()) if len(pair_agreement) else float("nan")

    daily_rows: list[dict[str, float | str | int]] = []
    for dt, g in d.groupby("date_utc", sort=True):
        tau_d, pval_d = _safe_tau(g[args.pred_score_col], g[args.real_score_col])
        n = int(
            pd.to_numeric(g[args.pred_score_col], errors="coerce")
            .notna()
            .mul(pd.to_numeric(g[args.real_score_col], errors="coerce").notna())
            .sum()
        )
        daily_rows.append(
            {
                "date_utc": pd.Timestamp(dt).date().isoformat(),
                "n_points": n,
                "kendall_tau": tau_d,
                "p_value": pval_d,
                "tau_sufficient": float((n >= int(args.min_points_per_day)) and np.isfinite(tau_d) and (tau_d >= args.tau_suff_threshold)),
            }
        )
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        tau_suff_rate = float("nan")
        tau_daily_mean = float("nan")
    else:
        valid = daily["n_points"] >= int(args.min_points_per_day)
        tau_suff_rate = float(daily.loc[valid, "tau_sufficient"].mean()) if bool(valid.any()) else float("nan")
        tau_daily_mean = float(daily.loc[valid, "kendall_tau"].mean()) if bool(valid.any()) else float("nan")

    summary = {
        "hourly_path": str(hourly_path.resolve()),
        "timestamp_col": args.timestamp_col,
        "pred_score_col": args.pred_score_col,
        "real_score_col": args.real_score_col,
        "n_rows_total": int(len(d)),
        "kendall_tau_global": tau_global,
        "kendall_tau_global_p_value": pval_global,
        "tau_suff_threshold": float(args.tau_suff_threshold),
        "min_points_per_day": int(args.min_points_per_day),
        "tau_sufficiency_rate_daily": tau_suff_rate,
        "kendall_tau_daily_mean": tau_daily_mean,
        "sign_agreement_rate": sign_agreement_rate,
        "daily_csv_path": str(out_csv.resolve()),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    daily.to_csv(out_csv, index=False)

    print("[OK] Decision-quality (Kendall tau) report created.")
    print(f"- Global tau: {tau_global:.6f}" if np.isfinite(tau_global) else "- Global tau: nan")
    print(f"- Tau sufficiency daily rate: {tau_suff_rate:.6f}" if np.isfinite(tau_suff_rate) else "- Tau sufficiency daily rate: nan")
    print(f"- JSON: {out_json}")
    print(f"- CSV: {out_csv}")


if __name__ == "__main__":
    main()

