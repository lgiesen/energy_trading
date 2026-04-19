#!/usr/bin/env python3
"""Benchmark multiple evaluated model runs.

Usage example:
    python3 scripts/benchmark_models.py \
      --run-dirs artifacts/model_runs/2026-04-18T14-42-52Z artifacts/model_runs/2026-04-18T14-43-04Z \
      --labels XGBoost TFT \
      --out-dir artifacts/benchmarks/xgb_vs_tft
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


LOWER_IS_BETTER = {
    "mae_mean",
    "rmse_mean",
    "pinball_mean",
    "wis_mean",
    "average_calibration_error",
}
HIGHER_IS_BETTER = {
    "skill_score_mae_mean",
}


def _configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("colorblind")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": True,
            "grid.alpha": 0.3,
        }
    )


def _save_plot_png_pdf(fig: plt.Figure, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _load_summary(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "summary_metrics.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing summary_metrics.json in {run_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def _detect_pred_col(summary: dict[str, Any], requested_pred_col: str) -> str:
    rows = summary.get("avg_metrics_by_prediction_column", [])
    if not rows:
        raise ValueError("summary_metrics.json has no avg_metrics_by_prediction_column entries.")
    available = [str(r.get("prediction_column")) for r in rows if r.get("prediction_column")]
    if requested_pred_col:
        if requested_pred_col not in available:
            raise KeyError(
                f"Requested pred_col '{requested_pred_col}' not available. Found: {sorted(set(available))}"
            )
        return requested_pred_col
    if summary.get("selected_pred_col"):
        return str(summary["selected_pred_col"])
    return available[0]


def _load_per_lead(run_dir: Path, split: str, pred_col: str) -> pd.DataFrame:
    p = run_dir / f"{split}_{pred_col}_metrics_by_lead.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing metrics-by-lead file: {p}")
    return pd.read_csv(p)


def _load_coverage(run_dir: Path, split: str, pred_col: str) -> pd.DataFrame:
    p = run_dir / f"{split}_{pred_col}_quantile_coverage.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing quantile coverage file: {p}")
    return pd.read_csv(p)


def _get_scalar_metric(summary: dict[str, Any], pred_col: str, metric: str) -> float:
    rows = summary.get("avg_metrics_by_prediction_column", [])
    for r in rows:
        if r.get("prediction_column") == pred_col:
            v = r.get(metric, np.nan)
            try:
                return float(v)
            except Exception:
                return float("nan")
    return float("nan")


def _build_summary_table(
    labels: list[str],
    summaries: list[dict[str, Any]],
    pred_col: str,
) -> pd.DataFrame:
    metric_rows = [
        ("MAE", "mae_mean"),
        ("RMSE", "rmse_mean"),
        ("Pinball", "pinball_mean"),
        ("WIS", "wis_mean"),
        ("ACE", "average_calibration_error"),
        ("Skill Score", "skill_score_mae_mean"),
    ]
    out_rows: list[dict[str, float | str]] = []
    for display_name, key in metric_rows:
        row: dict[str, float | str] = {"metric": display_name}
        vals: list[float] = []
        for lbl, summ in zip(labels, summaries):
            v = _get_scalar_metric(summ, pred_col, key)
            row[lbl] = v
            vals.append(v)

        delta = float("nan")
        if len(vals) >= 2 and np.isfinite(vals[0]) and vals[0] != 0 and np.isfinite(vals[1]):
            m1 = vals[0]
            m2 = vals[1]
            if key in LOWER_IS_BETTER:
                delta = 100.0 * (m1 - m2) / abs(m1)
            elif key in HIGHER_IS_BETTER:
                delta = 100.0 * (m2 - m1) / abs(m1)
            else:
                delta = 100.0 * (m2 - m1) / abs(m1)
        row["Delta (%)"] = delta
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _plot_degradation_overlay(
    labels: list[str],
    per_lead: list[pd.DataFrame],
    out_png: Path,
    pred_col: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    markers = ["o", "s", "D", "^", "v", "P", "X"]
    for i, (lbl, df) in enumerate(zip(labels, per_lead)):
        d = df.sort_values("lead_time_h")
        ax.plot(
            d["lead_time_h"],
            d["mae"],
            label=lbl,
            linewidth=2,
            marker=markers[i % len(markers)],
            markevery=max(1, len(d) // 12),
            markersize=4,
        )
    ax.set_title(f"Forecast Degradation Overlay (MAE) - {pred_col}")
    ax.set_xlabel("Lead Time [h]")
    ax.set_ylabel("MAE")
    ax.legend()
    fig.tight_layout()
    _save_plot_png_pdf(fig, out_png)


def _plot_skill_overlay(
    labels: list[str],
    per_lead: list[pd.DataFrame],
    out_png: Path,
    pred_col: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    markers = ["o", "s", "D", "^", "v", "P", "X"]
    for i, (lbl, df) in enumerate(zip(labels, per_lead)):
        d = df.sort_values("lead_time_h")
        ax.plot(
            d["lead_time_h"],
            d["skill_score_mae"],
            label=lbl,
            linewidth=2,
            marker=markers[i % len(markers)],
            markevery=max(1, len(d) // 12),
            markersize=4,
        )
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.5, label="Baseline parity")
    ax.set_title(f"Skill Score Overlay (MAE-based) - {pred_col}")
    ax.set_xlabel("Lead Time [h]")
    ax.set_ylabel("Skill Score")
    ax.legend()
    fig.tight_layout()
    _save_plot_png_pdf(fig, out_png)


def _plot_calibration_overlay(
    labels: list[str],
    coverages: list[pd.DataFrame],
    out_png: Path,
    pred_col: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", linewidth=1.5, label="Perfect calibration")
    markers = ["o", "s", "D", "^", "v", "P", "X"]
    for i, (lbl, cdf) in enumerate(zip(labels, coverages)):
        d = cdf.copy()
        d["quantile"] = pd.to_numeric(d["quantile"], errors="coerce")
        d["empirical_coverage"] = pd.to_numeric(d["empirical_coverage"], errors="coerce")
        d = d.dropna(subset=["quantile", "empirical_coverage"]).sort_values("quantile")
        ax.plot(
            d["quantile"],
            d["empirical_coverage"],
            marker=markers[i % len(markers)],
            linewidth=2,
            label=lbl,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Nominal quantile")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Probabilistic Calibration Overlay - {pred_col}")
    ax.legend()
    fig.tight_layout()
    _save_plot_png_pdf(fig, out_png)


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark multiple evaluated model runs.")
    p.add_argument("--run-dirs", nargs="+", required=True, help="List of run directories.")
    p.add_argument("--labels", nargs="+", required=True, help="Display labels, same length as --run-dirs.")
    p.add_argument("--out-dir", default="artifacts/benchmarks", help="Output directory.")
    p.add_argument("--split", choices=["val", "test"], default="test", help="Which split files to compare.")
    p.add_argument(
        "--pred-col",
        default="",
        help="Prediction column key (e.g., pred_da_price). If omitted, selected automatically from first run.",
    )
    return p


def main() -> None:
    args = _build_cli().parse_args()
    run_dirs = [Path(p) for p in args.run_dirs]
    labels = args.labels
    if len(run_dirs) != len(labels):
        raise ValueError("Length mismatch: --run-dirs and --labels must have same number of entries.")
    if len(run_dirs) < 2:
        raise ValueError("Provide at least 2 run dirs for comparison.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()

    summaries = [_load_summary(rd) for rd in run_dirs]
    pred_col = _detect_pred_col(summaries[0], args.pred_col.strip())

    # Ensure all runs support the same prediction column.
    for i, s in enumerate(summaries):
        _ = _detect_pred_col(s, pred_col)

    per_lead = [_load_per_lead(rd, args.split, pred_col) for rd in run_dirs]
    coverages = [_load_coverage(rd, args.split, pred_col) for rd in run_dirs]

    summary_df = _build_summary_table(labels, summaries, pred_col)
    summary_csv = out_dir / "benchmark_summary.csv"
    summary_tex = out_dir / "benchmark_summary.tex"
    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_latex(
        summary_tex,
        index=False,
        float_format=lambda x: f"{x:.4f}",
        caption=f"Benchmark summary for {pred_col} ({args.split}).",
        label="tab:benchmark_summary",
        escape=False,
    )

    _plot_degradation_overlay(
        labels=labels,
        per_lead=per_lead,
        out_png=out_dir / "plot_a_degradation_overlay_mae.png",
        pred_col=pred_col,
    )
    _plot_skill_overlay(
        labels=labels,
        per_lead=per_lead,
        out_png=out_dir / "plot_b_skill_score_overlay.png",
        pred_col=pred_col,
    )
    _plot_calibration_overlay(
        labels=labels,
        coverages=coverages,
        out_png=out_dir / "plot_c_calibration_overlay.png",
        pred_col=pred_col,
    )

    meta = {
        "run_dirs": [str(p.resolve()) for p in run_dirs],
        "labels": labels,
        "split": args.split,
        "pred_col": pred_col,
        "summary_csv": str(summary_csv.resolve()),
        "summary_tex": str(summary_tex.resolve()),
    }
    (out_dir / "benchmark_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved benchmark outputs:")
    print(f"- {summary_csv}")
    print(f"- {summary_tex}")
    print(f"- {out_dir / 'plot_a_degradation_overlay_mae.png'} (+pdf)")
    print(f"- {out_dir / 'plot_b_skill_score_overlay.png'} (+pdf)")
    print(f"- {out_dir / 'plot_c_calibration_overlay.png'} (+pdf)")
    print(f"- {out_dir / 'benchmark_meta.json'}")


if __name__ == "__main__":
    main()
