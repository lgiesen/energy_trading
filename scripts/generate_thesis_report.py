#!/usr/bin/env python3
"""Aggregate quantile-sweep simulation outputs into one thesis-ready table.

Also computes post-hoc Economic Elasticity of Accuracy:
    elasticity = (pnl_model - pnl_baseline) / (loss_baseline - loss_model)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def _parse_quantile_token(tok: object) -> int | None:
    if tok is None:
        return None
    s = str(tok).strip().lower()
    if not s:
        return None
    if s.startswith("p") and len(s) == 3 and s[1:].isdigit():
        q = int(s[1:])
        return q if 1 <= q <= 99 else None
    try:
        v = float(s)
    except Exception:
        return None
    if 0.0 < v < 1.0:
        q = int(round(v * 100.0))
        return max(1, min(99, q))
    return None


def _mean_of_existing(payload: dict, keys: list[str]) -> float | None:
    vals: list[float] = []
    for k in keys:
        if k in payload:
            try:
                x = float(payload[k])
            except Exception:
                continue
            if np.isfinite(x):
                vals.append(x)
    if not vals:
        return None
    return float(np.mean(vals))


def _pinball_from_metric_payload(payload: dict, *, q_low: int | None, q_high: int | None, da_role: str | None) -> float | None:
    # 1) Direct common keys.
    direct = _mean_of_existing(
        payload,
        [
            "weighted_pinball_loss_test",
            "weighted_pinball_loss",
            "mean_pinball_loss_test",
            "mean_pinball_loss",
        ],
    )
    if direct is not None:
        return direct

    # 2) Prefer leadtime weighted pinball metrics.
    quantiles: set[int] = set()
    if q_low is not None:
        quantiles.add(int(q_low))
    if q_high is not None:
        quantiles.add(int(q_high))
    if (da_role or "").strip().lower() == "mid":
        quantiles.add(50)
    if not quantiles:
        quantiles = {10, 50, 90}

    weighted_keys = [f"leadtime_pinball_p{q:02d}_test_weighted" for q in sorted(quantiles)]
    weighted = _mean_of_existing(payload, weighted_keys)
    if weighted is not None:
        return weighted

    # 3) Fallback to h1 pinball keys if weighted is unavailable.
    h1_keys = [f"pinball_loss_p{q:02d}_test_h1" for q in sorted(quantiles)]
    h1 = _mean_of_existing(payload, h1_keys)
    if h1 is not None:
        return h1

    # 4) Fallback to any pinball key found in payload.
    any_pinball_vals: list[float] = []
    for k, v in payload.items():
        if "pinball" not in str(k):
            continue
        try:
            x = float(v)
        except Exception:
            continue
        if np.isfinite(x):
            any_pinball_vals.append(x)
    if any_pinball_vals:
        return float(np.mean(any_pinball_vals))
    return None


def _run_pinball_map(model_runs_root: Path, report_df: pd.DataFrame) -> dict[str, float]:
    # For each run_id in thesis report, aggregate pinball proxy over metrics/*.json.
    run_ids = sorted({str(x) for x in report_df["run_id"].dropna().astype(str).unique()})
    out: dict[str, float] = {}
    q_info = {
        rid: (
            _parse_quantile_token(report_df.loc[report_df["run_id"] == rid, "quantile_low"].iloc[0])
            if "quantile_low" in report_df.columns and not report_df.loc[report_df["run_id"] == rid].empty
            else None,
            _parse_quantile_token(report_df.loc[report_df["run_id"] == rid, "quantile_high"].iloc[0])
            if "quantile_high" in report_df.columns and not report_df.loc[report_df["run_id"] == rid].empty
            else None,
            str(report_df.loc[report_df["run_id"] == rid, "da_quantile_role"].iloc[0])
            if "da_quantile_role" in report_df.columns and not report_df.loc[report_df["run_id"] == rid].empty
            else None,
        )
        for rid in run_ids
    }
    for rid in run_ids:
        metrics_dir = model_runs_root / rid / "metrics"
        if not metrics_dir.exists():
            continue
        vals: list[float] = []
        q_low, q_high, da_role = q_info[rid]
        for p in sorted(metrics_dir.glob("*.json")):
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            val = _pinball_from_metric_payload(payload, q_low=q_low, q_high=q_high, da_role=da_role)
            if val is not None and np.isfinite(val):
                vals.append(float(val))
        if vals:
            out[rid] = float(np.mean(vals))
    return out


def _compute_elasticity(
    report: pd.DataFrame,
    *,
    baseline_mode: str,
    persistence_pinball_loss: float | None,
) -> pd.DataFrame:
    d = report.copy()
    d["econ_elasticity_eur_per_loss_unit"] = np.nan

    group_cols = [c for c in ["split", "scenario", "quantile_low", "quantile_high", "da_quantile_role"] if c in d.columns]
    if not group_cols:
        group_cols = ["split"] if "split" in d.columns else []
    if not group_cols:
        group_cols = ["_all"]
        d["_all"] = "all"

    for _, gidx in d.groupby(group_cols, dropna=False).groups.items():
        g = d.loc[gidx].copy()
        if baseline_mode == "linear":
            base = g[g["model"] == "linear"]
            if base.empty:
                continue
            base_pnl = float(pd.to_numeric(base["realized_total_pnl_eur"], errors="coerce").mean())
            base_loss = float(pd.to_numeric(base["pinball_loss_proxy"], errors="coerce").mean())
        else:  # persistence baseline
            # PnL comes from naive benchmark in the same rows.
            if "naive_total_pnl_eur" not in g.columns:
                continue
            base_pnl = float(pd.to_numeric(g["naive_total_pnl_eur"], errors="coerce").mean())
            if persistence_pinball_loss is None or not np.isfinite(float(persistence_pinball_loss)):
                continue
            base_loss = float(persistence_pinball_loss)

        if not np.isfinite(base_pnl) or not np.isfinite(base_loss):
            continue

        pnl = pd.to_numeric(g["realized_total_pnl_eur"], errors="coerce")
        loss = pd.to_numeric(g["pinball_loss_proxy"], errors="coerce")
        denom = base_loss - loss
        num = pnl - base_pnl
        with np.errstate(divide="ignore", invalid="ignore"):
            elast = num / denom
        elast = elast.where(np.isfinite(elast), np.nan)
        d.loc[g.index, "econ_elasticity_eur_per_loss_unit"] = elast

    if "_all" in d.columns:
        d = d.drop(columns=["_all"])
    return d


def _write_elasticity_summary(report: pd.DataFrame, out_csv: Path, out_png: Path) -> None:
    valid = report.copy()
    valid["econ_elasticity_eur_per_loss_unit"] = pd.to_numeric(
        valid["econ_elasticity_eur_per_loss_unit"], errors="coerce"
    )
    valid = valid[np.isfinite(valid["econ_elasticity_eur_per_loss_unit"])]

    grp_cols = [c for c in ["model"] if c in valid.columns]
    if grp_cols:
        summary = (
            valid.groupby(grp_cols, dropna=False)["econ_elasticity_eur_per_loss_unit"]
            .agg(["count", "mean", "median", "min", "max"])
            .reset_index()
            .rename(
                columns={
                    "count": "n_rows",
                    "mean": "elasticity_mean",
                    "median": "elasticity_median",
                    "min": "elasticity_min",
                    "max": "elasticity_max",
                }
            )
        )
    else:
        summary = pd.DataFrame(
            [
                {
                    "n_rows": int(len(valid)),
                    "elasticity_mean": float(valid["econ_elasticity_eur_per_loss_unit"].mean()) if len(valid) else np.nan,
                    "elasticity_median": float(valid["econ_elasticity_eur_per_loss_unit"].median()) if len(valid) else np.nan,
                    "elasticity_min": float(valid["econ_elasticity_eur_per_loss_unit"].min()) if len(valid) else np.nan,
                    "elasticity_max": float(valid["econ_elasticity_eur_per_loss_unit"].max()) if len(valid) else np.nan,
                }
            ]
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    # Profit vs pinball scatter (slope proxy via model-wise linear fit when enough points).
    if "pinball_loss_proxy" in report.columns and "realized_total_pnl_eur" in report.columns:
        plot_df = report.copy()
        plot_df["pinball_loss_proxy"] = pd.to_numeric(plot_df["pinball_loss_proxy"], errors="coerce")
        plot_df["realized_total_pnl_eur"] = pd.to_numeric(plot_df["realized_total_pnl_eur"], errors="coerce")
        plot_df = plot_df[np.isfinite(plot_df["pinball_loss_proxy"]) & np.isfinite(plot_df["realized_total_pnl_eur"])]
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            if "model" in plot_df.columns:
                for model, g in plot_df.groupby("model", dropna=False):
                    ax.scatter(
                        g["pinball_loss_proxy"].to_numpy(dtype=float),
                        g["realized_total_pnl_eur"].to_numpy(dtype=float),
                        label=str(model),
                        alpha=0.8,
                    )
                    if len(g) >= 2:
                        x = g["pinball_loss_proxy"].to_numpy(dtype=float)
                        y = g["realized_total_pnl_eur"].to_numpy(dtype=float)
                        m, b = np.polyfit(x, y, 1)
                        xs = np.linspace(float(np.min(x)), float(np.max(x)), 50)
                        ax.plot(xs, m * xs + b, linewidth=1.5)
            else:
                ax.scatter(
                    plot_df["pinball_loss_proxy"].to_numpy(dtype=float),
                    plot_df["realized_total_pnl_eur"].to_numpy(dtype=float),
                    alpha=0.8,
                )
                if len(plot_df) >= 2:
                    x = plot_df["pinball_loss_proxy"].to_numpy(dtype=float)
                    y = plot_df["realized_total_pnl_eur"].to_numpy(dtype=float)
                    m, b = np.polyfit(x, y, 1)
                    xs = np.linspace(float(np.min(x)), float(np.max(x)), 50)
                    ax.plot(xs, m * xs + b, linewidth=1.5, label=f"slope={m:.2f}")
            ax.set_xlabel("Pinball Loss Proxy (lower is better)")
            ax.set_ylabel("Realized Total PnL [EUR]")
            ax.set_title("Profit vs Forecast Error (Economic Elasticity View)")
            ax.grid(True, alpha=0.3)
            if "model" in plot_df.columns:
                ax.legend()
            fig.tight_layout()
            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_png, dpi=200, bbox_inches="tight")
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build thesis benchmark report from quantile sweep results.")
    p.add_argument("--sim-root", default="artifacts/simulation_runs", help="Root directory containing simulation runs.")
    p.add_argument("--model-runs-root", default="artifacts/model_runs", help="Root directory containing model run metrics.")
    p.add_argument(
        "--elasticity-baseline",
        choices=["linear", "persistence"],
        default="linear",
        help="Baseline used for delta-based elasticity denominator/numerator.",
    )
    p.add_argument(
        "--persistence-pinball-loss",
        type=float,
        default=None,
        help="Required for --elasticity-baseline=persistence when no persistence pinball artifact exists.",
    )
    p.add_argument("--out-csv", default="artifacts/thesis_benchmark_report.csv", help="Output CSV path.")
    p.add_argument("--out-md", default="artifacts/thesis_benchmark_report.md", help="Output markdown table path.")
    p.add_argument(
        "--out-elasticity-summary-csv",
        default="artifacts/thesis_benchmark_elasticity_summary.csv",
        help="Output CSV for aggregated elasticity stats.",
    )
    p.add_argument(
        "--out-elasticity-plot",
        default="artifacts/thesis_benchmark_profit_vs_pinball.png",
        help="Output plot path for profit-vs-pinball relationship.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sim_root = Path(args.sim_root)
    model_runs_root = Path(args.model_runs_root)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_elast_summary_csv = Path(args.out_elasticity_summary_csv)
    out_elast_plot = Path(args.out_elasticity_plot)
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
        "perfect_foresight_total_pnl_eur",
        "pnl_gap_total_eur",
        "cost_of_forecast_error_total_eur",
        "economic_opportunity_gap_ratio",
        "roi_on_max_capital",
        "output_dir",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    report = df[keep_cols].copy()
    pinball_map = _run_pinball_map(model_runs_root, report)
    report["pinball_loss_proxy"] = report["run_id"].astype(str).map(pinball_map)
    report = _compute_elasticity(
        report,
        baseline_mode=args.elasticity_baseline,
        persistence_pinball_loss=args.persistence_pinball_loss,
    )
    report.to_csv(out_csv, index=False)
    out_md.write_text(report.to_markdown(index=False), encoding="utf-8")
    _write_elasticity_summary(report, out_elast_summary_csv, out_elast_plot)

    print(f"[OK] Thesis report written: {out_csv}")
    print(f"[OK] Thesis report written: {out_md}")
    print(f"[OK] Elasticity summary written: {out_elast_summary_csv}")
    print(f"[OK] Elasticity plot written: {out_elast_plot}")
    print(f"[INFO] Rows: {len(report)}")


if __name__ == "__main__":
    main()
