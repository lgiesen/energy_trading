#!/usr/bin/env python3
"""Benchmark multiple evaluated model runs.

Usage example:
    python3 scripts/benchmark_models.py \
      --run-dirs artifacts/model_runs/2026-04-19T21-32-05Z artifacts/model_runs/2026-04-19T22-38-28Z \
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
    "spike_mae_top5",
    "spike_rmse_top5",
    "negative_tail_mae_bottom5",
}
HIGHER_IS_BETTER = {
    "skill_score_mae_mean",
    "mean_directional_accuracy",
}


def _available_pred_cols(summary: dict[str, Any]) -> list[str]:
    rows = summary.get("avg_metrics_by_prediction_column", [])
    out = [str(r.get("prediction_column")) for r in rows if r.get("prediction_column")]
    return sorted(set(out))


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


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "manifest.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_truth_path(manifest: dict[str, Any]) -> Path:
    mpath = str(manifest.get("ground_truth", {}).get("default_path", "")).strip()
    if not mpath:
        return Path("data/features/all_data_features.parquet")
    p = Path(mpath)
    if p.exists():
        return p
    # Fallback for downloaded artifacts whose manifest still points to server-local absolute paths.
    return Path("data/features/all_data_features.parquet")


def _resolve_truth_column(summary: dict[str, Any], pred_col: str) -> str:
    for row in summary.get("avg_metrics_by_prediction_column", []):
        if row.get("prediction_column") == pred_col and row.get("truth_column"):
            return str(row["truth_column"])
    raise KeyError(f"Could not resolve truth column for prediction column '{pred_col}' from summary_metrics.json.")


def _discover_long_prediction_file(
    run_dir: Path,
    split: str,
    pred_col: str,
    manifest: dict[str, Any],
) -> Path:
    for bcfg in manifest.get("bundles", {}).values():
        p = bcfg.get("predictions_long", {}).get(split, {}).get(pred_col)
        if p:
            pp = Path(p)
            if pp.exists():
                return pp

    pred_dir = run_dir / "predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction dir missing: {pred_dir}")

    candidates = list(pred_dir.glob(f"*{split}*{pred_col}*long*.parquet"))
    if not candidates:
        candidates = list(pred_dir.glob(f"*{pred_col}*long*{split}*.parquet"))
    if not candidates:
        candidates = list(pred_dir.glob(f"*{pred_col}*{split}*long*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find long prediction parquet for pred_col='{pred_col}', split='{split}' in {pred_dir}"
        )
    return candidates[0]


def _discover_raw_prediction_file(run_dir: Path, split: str, pred_col: str) -> Path:
    direct = run_dir / f"{split}_{pred_col}.parquet"
    if direct.exists():
        return direct

    pred_dir = run_dir / "predictions"
    if pred_dir.exists():
        patterns = [
            f"{split}_{pred_col}.parquet",
            f"*{split}*{pred_col}*.parquet",
            f"*{pred_col}*{split}*.parquet",
        ]
        for pat in patterns:
            candidates = sorted(pred_dir.glob(pat))
            if candidates:
                return candidates[0]

    raise FileNotFoundError(
        f"Could not find raw prediction parquet for pred_col='{pred_col}', split='{split}' "
        f"(expected e.g. {run_dir / f'{split}_{pred_col}.parquet'})"
    )


def _calculate_economic_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    yt = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    yp = pd.to_numeric(y_pred, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[valid]
    yp = yp[valid]
    if yt.size == 0:
        return {
            "spike_mae_top5": float("nan"),
            "spike_rmse_top5": float("nan"),
            "negative_tail_mae_bottom5": float("nan"),
            "mean_directional_accuracy": float("nan"),
        }

    q95 = float(np.nanquantile(yt, 0.95))
    q05 = float(np.nanquantile(yt, 0.05))
    spike_mask = yt >= q95
    tail_mask = yt <= q05

    def _mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b))) if a.size else float("nan")

    def _rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else float("nan")

    dy_true = np.sign(np.diff(yt))
    dy_pred = np.sign(np.diff(yp))
    mda = float(100.0 * np.mean(dy_true == dy_pred)) if dy_true.size else float("nan")

    return {
        "spike_mae_top5": _mae(yt[spike_mask], yp[spike_mask]),
        "spike_rmse_top5": _rmse(yt[spike_mask], yp[spike_mask]),
        "negative_tail_mae_bottom5": _mae(yt[tail_mask], yp[tail_mask]),
        "mean_directional_accuracy": mda,
    }


def _load_economic_metrics_for_run(
    run_dir: Path,
    summary: dict[str, Any],
    split: str,
    pred_col: str,
) -> dict[str, float]:
    manifest = _load_manifest(run_dir)
    try:
        pred_path = _discover_raw_prediction_file(run_dir, split, pred_col)
    except FileNotFoundError:
        # Backward-compatible fallback for older runs that only exported long warehouse files.
        pred_path = _discover_long_prediction_file(run_dir, split, pred_col, manifest)
    pred_df = pd.read_parquet(pred_path)

    truth_path = _resolve_truth_path(manifest)
    if not truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {truth_path}")
    truth_df = pd.read_parquet(truth_path)
    if "timestamp_utc" not in truth_df.columns:
        raise KeyError(f"{truth_path} must contain timestamp_utc")

    truth_col = _resolve_truth_column(summary, pred_col)
    if truth_col not in truth_df.columns:
        raise KeyError(f"Truth column '{truth_col}' not found in {truth_path}")

    # Resolve timestamp + prediction columns from either raw test_<pred_col>.parquet
    # or long-format warehouse files.
    ts_col = next((c for c in ["timestamp", "target_time_utc", "timestamp_utc"] if c in pred_df.columns), None)
    if ts_col is None:
        raise KeyError(
            f"{pred_path} must contain one of timestamp/target_time_utc/timestamp_utc. "
            f"Found: {list(pred_df.columns)}"
        )
    pred_col_candidates = ["p50", "predicted_value", "prediction", "y_pred"]
    pred_value_col = next((c for c in pred_col_candidates if c in pred_df.columns), None)
    if pred_value_col is None:
        raise KeyError(
            f"{pred_path} must contain one of {pred_col_candidates}. Found: {list(pred_df.columns)}"
        )

    pred_df = pred_df.copy()
    pred_df["target_time_utc"] = pd.to_datetime(pred_df[ts_col], utc=True, errors="coerce")
    truth_df = truth_df.copy()
    truth_df["timestamp_utc"] = pd.to_datetime(truth_df["timestamp_utc"], utc=True, errors="coerce")
    truth_df = truth_df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    truth_by_ts = pd.Series(pd.to_numeric(truth_df[truth_col], errors="coerce").values, index=truth_df["timestamp_utc"])

    if truth_col in pred_df.columns:
        pred_df["y_true"] = pd.to_numeric(pred_df[truth_col], errors="coerce")
    elif "y_true" in pred_df.columns:
        pred_df["y_true"] = pd.to_numeric(pred_df["y_true"], errors="coerce")
    elif "actual" in pred_df.columns:
        pred_df["y_true"] = pd.to_numeric(pred_df["actual"], errors="coerce")
    elif "actuals" in pred_df.columns:
        pred_df["y_true"] = pd.to_numeric(pred_df["actuals"], errors="coerce")
    else:
        pred_df["y_true"] = pd.to_numeric(truth_by_ts.reindex(pred_df["target_time_utc"]).values, errors="coerce")
    pred_df["y_pred"] = pd.to_numeric(pred_df[pred_value_col], errors="coerce")

    # Use lead=1 when available to avoid multi-horizon duplicates in directional metrics.
    if "lead_time_h" in pred_df.columns:
        lead1 = pred_df[pd.to_numeric(pred_df["lead_time_h"], errors="coerce") == 1].copy()
        if not lead1.empty:
            pred_df = lead1

    s = (
        pred_df[["target_time_utc", "y_true", "y_pred"]]
        .dropna(subset=["target_time_utc"])
        .sort_values("target_time_utc")
        .groupby("target_time_utc", as_index=False)
        .agg({"y_true": "mean", "y_pred": "mean"})
    )
    return _calculate_economic_metrics(s["y_true"], s["y_pred"])


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


def _resolve_pred_cols(summaries: list[dict[str, Any]], requested_pred_col: str) -> list[str]:
    if requested_pred_col:
        pred = requested_pred_col.strip()
        if not pred:
            raise ValueError("Empty --pred-col after stripping.")
        for i, s in enumerate(summaries):
            avail = _available_pred_cols(s)
            if pred not in avail:
                raise KeyError(f"Requested pred_col '{pred}' not available in run index {i}. Found: {avail}")
        return [pred]

    common: set[str] | None = None
    for s in summaries:
        cur = set(_available_pred_cols(s))
        common = cur if common is None else (common & cur)
    pred_cols = sorted(common or set())
    if not pred_cols:
        raise ValueError("No common prediction columns found across provided runs.")
    return pred_cols


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
    economic_metrics: list[dict[str, float]],
) -> pd.DataFrame:
    metric_rows = [
        ("MAE", "mae_mean"),
        ("RMSE", "rmse_mean"),
        ("Pinball", "pinball_mean"),
        ("WIS", "wis_mean"),
        ("ACE", "average_calibration_error"),
        ("Skill Score", "skill_score_mae_mean"),
        ("Spike MAE (Top 5%)", "spike_mae_top5"),
        ("Spike RMSE (Top 5%)", "spike_rmse_top5"),
        ("Negative Tail MAE (Bottom 5%)", "negative_tail_mae_bottom5"),
        ("MDA (%)", "mean_directional_accuracy"),
    ]
    out_rows: list[dict[str, float | str]] = []
    for display_name, key in metric_rows:
        row: dict[str, float | str] = {"metric": display_name}
        vals: list[float] = []
        for idx, (lbl, summ) in enumerate(zip(labels, summaries)):
            if key in {"spike_mae_top5", "spike_rmse_top5", "negative_tail_mae_bottom5", "mean_directional_accuracy"}:
                v = float(economic_metrics[idx].get(key, np.nan))
            else:
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
    pred_cols = _resolve_pred_cols(summaries, args.pred_col.strip())
    all_summary_tables: list[pd.DataFrame] = []
    plotted_targets: list[str] = []
    skipped_calibration: list[str] = []

    plots_root = out_dir / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    for pred_col in pred_cols:
        per_lead = [_load_per_lead(rd, args.split, pred_col) for rd in run_dirs]
        econ_metrics = [
            _load_economic_metrics_for_run(rd, summ, args.split, pred_col)
            for rd, summ in zip(run_dirs, summaries)
        ]
        summary_df_one = _build_summary_table(labels, summaries, pred_col, econ_metrics)
        summary_df_one.insert(0, "prediction_column", pred_col)
        all_summary_tables.append(summary_df_one)

        target_plot_dir = plots_root / pred_col
        target_plot_dir.mkdir(parents=True, exist_ok=True)
        _plot_degradation_overlay(
            labels=labels,
            per_lead=per_lead,
            out_png=target_plot_dir / "plot_a_degradation_overlay_mae.png",
            pred_col=pred_col,
        )
        _plot_skill_overlay(
            labels=labels,
            per_lead=per_lead,
            out_png=target_plot_dir / "plot_b_skill_score_overlay.png",
            pred_col=pred_col,
        )
        try:
            coverages = [_load_coverage(rd, args.split, pred_col) for rd in run_dirs]
            _plot_calibration_overlay(
                labels=labels,
                coverages=coverages,
                out_png=target_plot_dir / "plot_c_calibration_overlay.png",
                pred_col=pred_col,
            )
        except FileNotFoundError:
            skipped_calibration.append(pred_col)
        plotted_targets.append(pred_col)

    summary_df = pd.concat(all_summary_tables, axis=0, ignore_index=True)
    summary_csv = out_dir / "benchmark_summary.csv"
    summary_tex = out_dir / "benchmark_summary.tex"
    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_latex(
        summary_tex,
        index=False,
        float_format=lambda x: f"{x:.4f}",
        caption=f"Benchmark summary across prediction columns ({args.split}).",
        label="tab:benchmark_summary",
        escape=False,
    )

    meta = {
        "run_dirs": [str(p.resolve()) for p in run_dirs],
        "labels": labels,
        "split": args.split,
        "pred_cols": pred_cols,
        "plotted_targets": plotted_targets,
        "skipped_calibration_targets": skipped_calibration,
        "summary_csv": str(summary_csv.resolve()),
        "summary_tex": str(summary_tex.resolve()),
    }
    (out_dir / "benchmark_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved benchmark outputs:")
    print(f"- {summary_csv}")
    print(f"- {summary_tex}")
    print(f"- plots root: {plots_root}")
    print(f"- {out_dir / 'benchmark_meta.json'}")


if __name__ == "__main__":
    main()
