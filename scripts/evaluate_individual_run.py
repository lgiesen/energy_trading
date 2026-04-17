#!/usr/bin/env python3
"""Evaluate a single model run directory (deep-dive by lead time).

Usage:
    python3 scripts/evaluate_individual_run.py \
      --run-dir artifacts/model_runs/2026-04-13T16-20-00Z \
      --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


QUANTILES: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QCOLS: list[str] = [f"p{int(round(q * 100)):02d}" for q in QUANTILES]

PRED_TO_TARGET = {
    "pred_da_price": "target_da_price_h1",
    "pred_afrr_activation_price_pos": "target_afrr_activation_price_vwap_pos_h1",
    "pred_afrr_activation_price_neg": "target_afrr_activation_price_vwap_neg_h1",
    "pred_afrr_capacity_price_pos": "target_afrr_capacity_price_pos_h1",
    "pred_afrr_capacity_price_neg": "target_afrr_capacity_price_neg_h1",
    "pred_afrr_activation_rate_pos": "target_afrr_rate_h1",
    "pred_afrr_activation_rate_neg": "target_afrr_rate_h1",
}

PRED_TO_TRUE = {
    "pred_da_price": ["da_price", "target_da_price_h1"],
    "pred_afrr_activation_price_pos": ["afrr_activation_price_vwap_pos", "target_afrr_activation_price_vwap_pos_h1"],
    "pred_afrr_activation_price_neg": ["afrr_activation_price_vwap_neg", "target_afrr_activation_price_vwap_neg_h1"],
    "pred_afrr_capacity_price_pos": ["afrr_capacity_price_pos", "target_afrr_capacity_price_pos_h1"],
    "pred_afrr_capacity_price_neg": ["afrr_capacity_price_neg", "target_afrr_capacity_price_neg_h1"],
    "pred_afrr_activation_rate_pos": ["afrr_activation_rate", "afrr_activation_rate_pos", "target_afrr_rate_h1"],
    "pred_afrr_activation_rate_neg": ["afrr_activation_rate", "afrr_activation_rate_neg", "target_afrr_rate_h1"],
}


def _configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-paper")
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


def _save_plot_png_pdf(fig: plt.Figure, out_path_png: Path) -> None:
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_path_png.with_suffix(".pdf"), bbox_inches="tight")


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    e = y_true - y_pred
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def _interval_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    base = upper - lower
    below = (y_true < lower).astype(float)
    above = (y_true > upper).astype(float)
    penalty = (2.0 / alpha) * (lower - y_true) * below + (2.0 / alpha) * (y_true - upper) * above
    return base + penalty


def _wis_p10_p50_p90(y_true: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(p10) & np.isfinite(p50) & np.isfinite(p90)
    if not bool(valid.any()):
        return float("nan")
    yv = y_true[valid]
    l = p10[valid]
    m = p50[valid]
    u = p90[valid]
    alpha = 0.2
    s = _interval_score(yv, l, u, alpha=alpha)
    wis = (0.5 * np.abs(yv - m) + (alpha / 2.0) * s) / 2.0
    return float(np.mean(wis))


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def _discover_long_prediction_files(run_dir: Path, split: str, manifest: dict[str, Any]) -> dict[str, Path]:
    out: dict[str, Path] = {}

    bundles = manifest.get("bundles", {})
    for bcfg in bundles.values():
        pmap = bcfg.get("predictions_long", {}).get(split, {})
        for pred_col, p in pmap.items():
            pp = Path(p)
            if pp.exists():
                out[pred_col] = pp

    if out:
        return out

    pred_dir = run_dir / "predictions"
    if not pred_dir.exists():
        return out
    for p in pred_dir.glob(f"*{split}*long*.parquet"):
        name = p.name
        for pred_col in PRED_TO_TARGET:
            if pred_col in name:
                out[pred_col] = p
                break
    return out


def _resolve_truth_path(manifest: dict[str, Any], arg_truth_path: str) -> Path:
    if arg_truth_path.strip():
        return Path(arg_truth_path)
    mpath = manifest.get("ground_truth", {}).get("default_path", "")
    if mpath:
        return Path(mpath)
    return Path("data/features/all_data_features.parquet")


def _resolve_truth_column(pred_col: str, truth_df: pd.DataFrame) -> str:
    for c in PRED_TO_TRUE.get(pred_col, []):
        if c in truth_df.columns:
            return c
    raise KeyError(f"No truth column found for prediction column '{pred_col}'.")


def _naive_24h_for_target_times(target_times: pd.Series, truth_series_by_ts: pd.Series) -> pd.Series:
    lagged_ts = pd.to_datetime(target_times, utc=True, errors="coerce") - pd.Timedelta(hours=24)
    naive = truth_series_by_ts.reindex(pd.DatetimeIndex(lagged_ts))
    naive.index = target_times.index
    return pd.to_numeric(naive, errors="coerce")


def _compute_metrics_by_lead(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for lead, part in df.groupby("lead_time_h", sort=True):
        y = pd.to_numeric(part["y_true"], errors="coerce").to_numpy(dtype=float)
        p50 = pd.to_numeric(part["p50"], errors="coerce").to_numpy(dtype=float)
        y_naive = pd.to_numeric(part["naive_24h"], errors="coerce").to_numpy(dtype=float)
        p10 = pd.to_numeric(part["p10"], errors="coerce").to_numpy(dtype=float) if "p10" in part.columns else p50
        p90 = pd.to_numeric(part["p90"], errors="coerce").to_numpy(dtype=float) if "p90" in part.columns else p50

        valid = np.isfinite(y) & np.isfinite(p50)
        if not bool(valid.any()):
            rows.append(
                {
                    "lead_time_h": float(lead),
                    "n": 0.0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "pinball_mean": np.nan,
                    "wis_p10_p50_p90": np.nan,
                    "sharpness_p90_p10": np.nan,
                    "naive_mae_24h": np.nan,
                    "naive_rmse_24h": np.nan,
                    "skill_score_mae": np.nan,
                    "skill_score_rmse": np.nan,
                }
            )
            continue

        yv = y[valid]
        p50v = p50[valid]
        mae = float(np.mean(np.abs(yv - p50v)))
        rmse = float(np.sqrt(np.mean((yv - p50v) ** 2)))
        wis = _wis_p10_p50_p90(y, p10, p50, p90)
        sharpness = float(np.nanmean(p90 - p10)) if np.isfinite(np.nanmean(p90 - p10)) else np.nan

        pbl: list[float] = []
        for q, qcol in zip(QUANTILES, QCOLS):
            if qcol not in part.columns:
                continue
            pq = pd.to_numeric(part.loc[valid, qcol], errors="coerce").to_numpy(dtype=float)
            qvalid = np.isfinite(yv) & np.isfinite(pq)
            if bool(qvalid.any()):
                pbl.append(_pinball_loss(yv[qvalid], pq[qvalid], q))
        pinball_mean = float(np.mean(pbl)) if pbl else np.nan

        nvalid = np.isfinite(y) & np.isfinite(y_naive)
        naive_mae = float(np.mean(np.abs(y[nvalid] - y_naive[nvalid]))) if bool(nvalid.any()) else np.nan
        naive_rmse = float(np.sqrt(np.mean((y[nvalid] - y_naive[nvalid]) ** 2))) if bool(nvalid.any()) else np.nan
        skill = 1.0 - (mae / naive_mae) if np.isfinite(naive_mae) and naive_mae > 0 else np.nan
        skill_rmse = 1.0 - (rmse / naive_rmse) if np.isfinite(naive_rmse) and naive_rmse > 0 else np.nan

        rows.append(
            {
                "lead_time_h": float(lead),
                "n": float(int(valid.sum())),
                "mae": mae,
                "rmse": rmse,
                "pinball_mean": pinball_mean,
                "wis_p10_p50_p90": wis,
                "sharpness_p90_p10": sharpness,
                "naive_mae_24h": naive_mae,
                "naive_rmse_24h": naive_rmse,
                "skill_score_mae": skill,
                "skill_score_rmse": skill_rmse,
            }
        )
    return pd.DataFrame(rows).sort_values("lead_time_h").reset_index(drop=True)


def _plot_error_horizon(metrics_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics_df["lead_time_h"], metrics_df["mae"], label="MAE", linewidth=2)
    ax.plot(metrics_df["lead_time_h"], metrics_df["rmse"], label="RMSE", linewidth=2)
    ax.set_xlabel("Lead Time [h]")
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_plot_png_pdf(fig, out_path)
    plt.close(fig)


def _plot_skill_score_horizon(metrics_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics_df["lead_time_h"], metrics_df["skill_score_mae"], linewidth=2, label="Skill Score")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.5, label="Baseline parity (0)")
    ax.set_xlabel("Lead Time [h]")
    ax.set_ylabel("Skill Score (1 - MAE_model / MAE_naive)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_plot_png_pdf(fig, out_path)
    plt.close(fig)


def _plot_decay_horizon(metrics_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(metrics_df["lead_time_h"], metrics_df["mae"], linewidth=2, label="MAE")
    ax.plot(metrics_df["lead_time_h"], metrics_df["pinball_mean"], linewidth=2, label="Pinball")
    if "wis_p10_p50_p90" in metrics_df.columns:
        ax.plot(metrics_df["lead_time_h"], metrics_df["wis_p10_p50_p90"], linewidth=2, label="WIS(P10,P50,P90)")
    ax.set_xlabel("Lead Time [h]")
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_plot_png_pdf(fig, out_path)
    plt.close(fig)


def _plot_probabilistic_window(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    lead: int = 1,
    window_hours: int = 72,
) -> None:
    d = df.loc[df["lead_time_h"] == lead].copy()
    d = d.sort_values("target_time_utc").reset_index(drop=True)
    if d.empty:
        return

    y_roll = pd.to_numeric(d["y_true"], errors="coerce")
    rolling_vol = y_roll.rolling(window=window_hours, min_periods=max(12, window_hours // 2)).std()
    if rolling_vol.notna().any():
        end_idx = int(rolling_vol.idxmax())
        start_idx = max(0, end_idx - window_hours + 1)
        window = d.iloc[start_idx : end_idx + 1].copy()
    else:
        window = d.head(window_hours).copy()

    x = pd.to_datetime(window["target_time_utc"], utc=True, errors="coerce")
    y = pd.to_numeric(window["y_true"], errors="coerce")
    p50 = pd.to_numeric(window["p50"], errors="coerce")
    p10 = pd.to_numeric(window["p10"], errors="coerce") if "p10" in window.columns else p50
    p90 = pd.to_numeric(window["p90"], errors="coerce") if "p90" in window.columns else p50
    p40 = pd.to_numeric(window["p40"], errors="coerce") if "p40" in window.columns else p50
    p60 = pd.to_numeric(window["p60"], errors="coerce") if "p60" in window.columns else p50

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(x, p10, p90, alpha=0.2, label="P10-P90")
    ax.fill_between(x, p40, p60, alpha=0.35, label="P40-P60")
    ax.plot(x, p50, label="P50", linewidth=2)
    ax.plot(x, y, label="True", linewidth=1.8)
    ax.set_xlabel("Target Time (UTC)")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_plot_png_pdf(fig, out_path)
    plt.close(fig)


def _compute_quantile_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    for q, qcol in zip(QUANTILES, QCOLS):
        if qcol not in df.columns:
            continue
        q_pred = pd.to_numeric(df[qcol], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isfinite(q_pred)
        n = int(valid.sum())
        if n == 0:
            empirical = np.nan
            cal_err = np.nan
        else:
            empirical = float(np.mean(y[valid] <= q_pred[valid]))
            cal_err = float(empirical - q)
        rows.append(
            {
                "quantile": float(q),
                "quantile_col": qcol,
                "n": float(n),
                "empirical_coverage": empirical,
                "calibration_error": cal_err,
                "abs_calibration_error": abs(cal_err) if np.isfinite(cal_err) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("quantile").reset_index(drop=True)


def _average_calibration_error(coverage_df: pd.DataFrame) -> float:
    if coverage_df.empty or "abs_calibration_error" not in coverage_df.columns:
        return float("nan")
    s = pd.to_numeric(coverage_df["abs_calibration_error"], errors="coerce")
    s = s[np.isfinite(s)]
    if s.empty:
        return float("nan")
    return float(s.mean())


def _plot_reliability_diagram(coverage_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if coverage_df.empty:
        return
    d = coverage_df.copy()
    d["quantile"] = pd.to_numeric(d["quantile"], errors="coerce")
    d["empirical_coverage"] = pd.to_numeric(d["empirical_coverage"], errors="coerce")
    d = d.dropna(subset=["quantile", "empirical_coverage"]).sort_values("quantile")
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    x = d["quantile"].to_numpy(dtype=float)
    y = d["empirical_coverage"].to_numpy(dtype=float)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.5, color="black", label="Perfect calibration")
    ax.plot(x, y, marker="o", linewidth=2, label="Model coverage")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Theoretical quantile")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_plot_png_pdf(fig, out_path)
    plt.close(fig)


def _directional_and_event_metrics(df: pd.DataFrame, event_q: float = 0.9) -> dict[str, float]:
    d = df.copy()
    d = d.sort_values("target_time_utc").reset_index(drop=True)
    y = pd.to_numeric(d["y_true"], errors="coerce")
    p = pd.to_numeric(d["p50"], errors="coerce")

    y_diff = y.diff()
    p_diff = p.diff()
    valid_dir = y_diff.notna() & p_diff.notna()
    if bool(valid_dir.any()):
        hit = np.sign(y_diff[valid_dir].to_numpy(dtype=float)) == np.sign(p_diff[valid_dir].to_numpy(dtype=float))
        directional_accuracy = float(np.mean(hit))
    else:
        directional_accuracy = float("nan")

    y_valid = y[np.isfinite(y)]
    if y_valid.empty:
        return {
            "directional_accuracy": directional_accuracy,
            "event_threshold_true_q90": float("nan"),
            "event_precision": float("nan"),
            "event_recall": float("nan"),
            "event_f1": float("nan"),
        }

    thr = float(np.nanquantile(y_valid.to_numpy(dtype=float), event_q))
    valid_evt = y.notna() & p.notna()
    if not bool(valid_evt.any()):
        return {
            "directional_accuracy": directional_accuracy,
            "event_threshold_true_q90": thr,
            "event_precision": float("nan"),
            "event_recall": float("nan"),
            "event_f1": float("nan"),
        }

    y_evt = (y[valid_evt].to_numpy(dtype=float) >= thr).astype(int)
    p_evt = (p[valid_evt].to_numpy(dtype=float) >= thr).astype(int)
    tp = int(((p_evt == 1) & (y_evt == 1)).sum())
    fp = int(((p_evt == 1) & (y_evt == 0)).sum())
    fn = int(((p_evt == 0) & (y_evt == 1)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    f1 = float((2 * precision * recall) / (precision + recall)) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else float("nan")
    return {
        "directional_accuracy": directional_accuracy,
        "event_threshold_true_q90": thr,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
    }


def _extract_target_models(payload: Any, target_col: str | None) -> dict[int, dict[str, Any]] | None:
    if not isinstance(payload, dict) or not payload:
        return None

    if all(isinstance(k, int) for k in payload.keys()):
        return payload

    if target_col and target_col in payload and isinstance(payload[target_col], dict):
        return payload[target_col]

    for v in payload.values():
        if isinstance(v, dict):
            return v
    return None


def _feature_gain_series(model: Any) -> pd.Series:
    gain_map = model.get_booster().get_score(importance_type="gain")
    names = model.get_booster().feature_names or []
    if names:
        s = pd.Series(0.0, index=names, dtype=float)
        for k, v in gain_map.items():
            if k in s.index:
                s.loc[k] = float(v)
            elif k.startswith("f") and k[1:].isdigit():
                i = int(k[1:])
                if 0 <= i < len(names):
                    s.iloc[i] = float(v)
    else:
        s = pd.Series({k: float(v) for k, v in gain_map.items()}, dtype=float)
    s = s.sort_values(ascending=False)
    return s


def _plot_xgb_feature_importance(run_dir: Path, out_path: Path, target_col: str | None) -> bool:
    model_files = sorted((run_dir / "models").glob("*model.joblib"))
    if not model_files:
        return False

    payload = joblib.load(model_files[0])
    lead_map = _extract_target_models(payload, target_col)
    if not lead_map or 1 not in lead_map:
        return False

    model_h1 = lead_map[1].get("p50")
    model_h24 = lead_map.get(24, {}).get("p50")
    if model_h1 is None:
        return False

    g1 = _feature_gain_series(model_h1).head(15).sort_values(ascending=True)
    g24 = _feature_gain_series(model_h24).head(15).sort_values(ascending=True) if model_h24 is not None else pd.Series(dtype=float)

    if g24.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(g1.index, g1.values)
        ax.set_title("XGBoost Feature Importance (Gain) - h1")
        ax.set_xlabel("Gain")
        fig.tight_layout()
        _save_plot_png_pdf(fig, out_path)
        plt.close(fig)
        return True

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False)
    axes[0].barh(g1.index, g1.values)
    axes[0].set_title("Top 15 Gain - h1")
    axes[0].set_xlabel("Gain")
    axes[1].barh(g24.index, g24.values)
    axes[1].set_title("Top 15 Gain - h24")
    axes[1].set_xlabel("Gain")
    fig.suptitle("XGBoost Feature Importance")
    fig.tight_layout()
    _save_plot_png_pdf(fig, out_path)
    plt.close(fig)
    return True


def _write_latex_summary_table(summary_df: pd.DataFrame, out_path: Path) -> None:
    table_df = summary_df[
        [
            "prediction_column",
            "truth_column",
            "mae_mean",
            "rmse_mean",
            "pinball_mean",
            "wis_mean",
            "skill_score_mae_mean",
            "skill_score_rmse_mean",
            "average_calibration_error",
            "sharpness_mean",
            "directional_accuracy",
            "event_f1",
        ]
    ].copy()
    table_df = table_df.rename(
        columns={
            "prediction_column": "Prediction",
            "truth_column": "Truth",
            "mae_mean": "MAE",
            "rmse_mean": "RMSE",
            "pinball_mean": "Pinball",
            "wis_mean": "WIS",
            "skill_score_mae_mean": "SkillScore",
            "skill_score_rmse_mean": "SkillScoreRMSE",
            "average_calibration_error": "ACE",
            "sharpness_mean": "Sharpness",
            "directional_accuracy": "HitRate",
            "event_f1": "EventF1",
        }
    )
    latex = table_df.to_latex(
        index=False,
        float_format=lambda x: f"{x:.4f}",
        caption="Average forecast metrics over lead times h1-h48.",
        label="tab:run_summary_metrics",
        escape=False,
    )
    out_path.write_text(latex, encoding="utf-8")


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deep-dive evaluation for a single model run directory.")
    p.add_argument("--run-dir", required=True, help="Path to one run directory under artifacts/model_runs/<run_id>.")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--feature-config", default="data/model_input/feature_config.json")
    p.add_argument("--ground-truth-path", default="", help="Optional explicit ground-truth parquet path.")
    p.add_argument(
        "--pred-col",
        default="",
        help="Optional prediction column key (e.g., pred_da_price). Defaults to first available.",
    )
    p.add_argument(
        "--calibration-lead",
        type=int,
        default=1,
        help="Lead time used for the calibration window plot (default: 1).",
    )
    return p


def main() -> None:
    args = _build_cli().parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    _configure_plot_style()
    _ = json.loads(Path(args.feature_config).read_text(encoding="utf-8"))  # reproducibility context

    manifest = _load_manifest(run_dir)
    pred_long_paths = _discover_long_prediction_files(run_dir, args.split, manifest)
    if not pred_long_paths:
        raise FileNotFoundError(f"No long prediction files found for split='{args.split}' in {run_dir}.")

    truth_path = _resolve_truth_path(manifest, args.ground_truth_path)
    if not truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {truth_path}")
    truth_df = pd.read_parquet(truth_path)
    if "timestamp_utc" not in truth_df.columns:
        raise KeyError("Ground truth parquet must contain 'timestamp_utc'.")
    truth_df["timestamp_utc"] = pd.to_datetime(truth_df["timestamp_utc"], utc=True, errors="coerce")
    truth_df = truth_df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")

    selected_pred_col = args.pred_col.strip() or sorted(pred_long_paths.keys())[0]
    if selected_pred_col not in pred_long_paths:
        raise KeyError(
            f"pred_col '{selected_pred_col}' not available. "
            f"Choose one of: {sorted(pred_long_paths.keys())}"
        )

    all_metrics: dict[str, pd.DataFrame] = {}
    avg_summary_rows: list[dict[str, Any]] = []

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    coverage_by_prediction: dict[str, pd.DataFrame] = {}

    for pred_col, path in sorted(pred_long_paths.items()):
        df = pd.read_parquet(path)
        req = {"snapshot_time_utc", "target_time_utc", "lead_time_h", "p50"}
        missing = req - set(df.columns)
        if missing:
            raise KeyError(f"{path} missing required columns: {sorted(missing)}")

        df["target_time_utc"] = pd.to_datetime(df["target_time_utc"], utc=True, errors="coerce")
        truth_col = _resolve_truth_column(pred_col, truth_df)
        truth_series = pd.to_numeric(truth_df[truth_col], errors="coerce")
        truth_by_ts = pd.Series(truth_series.values, index=truth_df["timestamp_utc"]).sort_index()

        d = df.copy()
        d["y_true"] = pd.to_numeric(truth_by_ts.reindex(d["target_time_utc"]).values, errors="coerce")
        d["naive_24h"] = _naive_24h_for_target_times(d["target_time_utc"], truth_by_ts)
        coverage_df = _compute_quantile_coverage(d)
        coverage_by_prediction[pred_col] = coverage_df
        ace = _average_calibration_error(coverage_df)
        evt = _directional_and_event_metrics(d)

        metrics_df = _compute_metrics_by_lead(d)
        all_metrics[pred_col] = metrics_df
        avg_summary_rows.append(
            {
                "prediction_column": pred_col,
                "truth_column": truth_col,
                "mae_mean": float(metrics_df["mae"].mean()),
                "rmse_mean": float(metrics_df["rmse"].mean()),
                "pinball_mean": float(metrics_df["pinball_mean"].mean()),
                "wis_mean": float(metrics_df["wis_p10_p50_p90"].mean()),
                "skill_score_mae_mean": float(metrics_df["skill_score_mae"].mean()),
                "skill_score_rmse_mean": float(metrics_df["skill_score_rmse"].mean()),
                "average_calibration_error": ace,
                "sharpness_mean": float(metrics_df["sharpness_p90_p10"].mean()),
                "directional_accuracy": evt["directional_accuracy"],
                "event_precision": evt["event_precision"],
                "event_recall": evt["event_recall"],
                "event_f1": evt["event_f1"],
            }
        )

        if pred_col == selected_pred_col:
            _plot_error_horizon(
                metrics_df,
                plots_dir / f"{args.split}_{pred_col}_error_over_horizon.png",
                title=f"{pred_col} - Error Over Horizon ({args.split})",
            )
            _plot_skill_score_horizon(
                metrics_df,
                plots_dir / f"{args.split}_{pred_col}_skill_score_over_horizon.png",
                title=f"{pred_col} - Forecast Skill Score Over Horizon ({args.split})",
            )
            _plot_decay_horizon(
                metrics_df,
                plots_dir / f"{args.split}_{pred_col}_forecast_decay_metrics.png",
                title=f"{pred_col} - Forecast Decay (MAE/Pinball/WIS) ({args.split})",
            )
            _plot_probabilistic_window(
                d,
                plots_dir / f"{args.split}_{pred_col}_probabilistic_calibration.png",
                title=f"{pred_col} - True vs P50 (P10-P90 / P40-P60), highest-volatility 3-day window",
                lead=args.calibration_lead,
                window_hours=72,
            )
            _plot_reliability_diagram(
                coverage_df,
                plots_dir / f"{args.split}_{pred_col}_reliability_diagram.png",
                title=f"{pred_col} - Reliability Diagram ({args.split})",
            )

    target_col = PRED_TO_TARGET.get(selected_pred_col)
    fi_ok = _plot_xgb_feature_importance(
        run_dir=run_dir,
        out_path=plots_dir / f"{args.split}_{selected_pred_col}_feature_importance_xgboost.png",
        target_col=target_col,
    )

    for pred_col, mdf in all_metrics.items():
        mdf.to_csv(run_dir / f"{args.split}_{pred_col}_metrics_by_lead.csv", index=False)
        mdf[["lead_time_h", "mae", "pinball_mean"]].to_csv(
            run_dir / f"{args.split}_{pred_col}_decay_stats.csv",
            index=False,
        )
    for pred_col, cdf in coverage_by_prediction.items():
        cdf.to_csv(run_dir / f"{args.split}_{pred_col}_quantile_coverage.csv", index=False)

    summary_payload = {
        "run_dir": str(run_dir.resolve()),
        "split": args.split,
        "ground_truth_path": str(truth_path.resolve()),
        "selected_pred_col": selected_pred_col,
        "feature_importance_plot_created": bool(fi_ok),
        "avg_metrics_by_prediction_column": avg_summary_rows,
        "quantile_coverage": {
            k: v.to_dict(orient="records") for k, v in coverage_by_prediction.items()
        },
    }
    (run_dir / "summary_metrics.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    summary_df = pd.DataFrame(avg_summary_rows).sort_values("prediction_column").reset_index(drop=True)
    _write_latex_summary_table(summary_df, run_dir / "summary_metrics.tex")

    print("\n=== Average Metrics (over leads h1..h48) ===")
    print(summary_df.to_string(index=False))
    print("\nSaved:")
    print(f"- {run_dir / 'summary_metrics.json'}")
    print(f"- {run_dir / 'summary_metrics.tex'}")
    print(f"- {plots_dir}")


if __name__ == "__main__":
    main()
