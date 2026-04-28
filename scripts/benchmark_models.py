#!/usr/bin/env python3
"""Benchmark multiple evaluated model runs.

Usage example:
    python3 scripts/benchmark_models.py \
      --run-dirs artifacts/model_runs/2026-04-20T15-36-58Z artifacts/model_runs/2026-04-20T15-36-58Z \
      --labels XGBoost TFT \
      --out-dir artifacts/benchmarks/xgb_vs_tft
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_trading.models.cv import PurgedTimeSeriesSplit
from energy_trading.evaluation.gate_synced_analysis import gate_time_dplus1_filter

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


def _infer_model_key(label: str) -> str:
    l = (label or "").strip().lower()
    if "xgb" in l or "xgboost" in l:
        return "xgboost"
    if "tft" in l:
        return "tft"
    return ""


def _matches_model_key(path: Path, model_key: str) -> bool:
    if not model_key:
        return True
    mk = model_key.strip().lower()
    name = path.name.lower()
    if mk in {"xgb", "xgboost"}:
        return "xgboost" in name
    if mk == "tft":
        return "xgboost" not in name
    return mk in name


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
    model_key: str = "",
) -> Path:
    for bcfg in manifest.get("bundles", {}).values():
        p = bcfg.get("predictions_long", {}).get(split, {}).get(pred_col)
        if p:
            pp = Path(p)
            if pp.exists() and _matches_model_key(pp, model_key):
                return pp

    pred_dir = run_dir / "predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction dir missing: {pred_dir}")

    candidates = list(pred_dir.glob(f"*{split}*{pred_col}*long*.parquet"))
    if not candidates:
        candidates = list(pred_dir.glob(f"*{pred_col}*long*{split}*.parquet"))
    if not candidates:
        candidates = list(pred_dir.glob(f"*{pred_col}*{split}*long*.parquet"))
    if model_key:
        candidates = [c for c in candidates if _matches_model_key(c, model_key)]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find long prediction parquet for pred_col='{pred_col}', split='{split}' in {pred_dir}"
        )
    return candidates[0]


def _discover_raw_prediction_file(run_dir: Path, split: str, pred_col: str, model_key: str = "") -> Path:
    direct = run_dir / f"{split}_{pred_col}.parquet"
    if direct.exists() and _matches_model_key(direct, model_key):
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
            if model_key:
                candidates = [c for c in candidates if _matches_model_key(c, model_key)]
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


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    e = y_true - y_pred
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def _load_model_long_with_truth(
    run_dir: Path,
    summary: dict[str, Any],
    split: str,
    pred_col: str,
    model_key: str = "",
) -> pd.DataFrame:
    manifest = _load_manifest(run_dir)
    try:
        pred_path = _discover_raw_prediction_file(run_dir, split, pred_col, model_key=model_key)
    except FileNotFoundError:
        # Backward-compatible fallback for older runs that only exported long warehouse files.
        pred_path = _discover_long_prediction_file(run_dir, split, pred_col, manifest, model_key=model_key)
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

    pred_df = pred_df.copy()
    pred_df["target_time_utc"] = pd.to_datetime(pred_df[ts_col], utc=True, errors="coerce")
    if "lead_time_h" in pred_df.columns:
        pred_df["lead_time_h"] = pd.to_numeric(pred_df["lead_time_h"], errors="coerce")
    else:
        pred_df["lead_time_h"] = 1.0

    if "p50" not in pred_df.columns:
        for c in ["predicted_value", "prediction", "y_pred"]:
            if c in pred_df.columns:
                pred_df["p50"] = pd.to_numeric(pred_df[c], errors="coerce")
                break
    if "p50" not in pred_df.columns:
        raise KeyError(f"{pred_path} must provide p50/predicted_value/prediction/y_pred.")

    q_cols = [c for c in pred_df.columns if re.fullmatch(r"p\d{2}", str(c))]
    if "p50" not in q_cols:
        q_cols.append("p50")
    for c in q_cols:
        pred_df[c] = pd.to_numeric(pred_df[c], errors="coerce")

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
    pred_df["y_pred"] = pd.to_numeric(pred_df["p50"], errors="coerce")
    pred_df["y_naive_24h"] = pd.to_numeric(
        truth_by_ts.reindex(pred_df["target_time_utc"] - pd.Timedelta(hours=24)).values,
        errors="coerce",
    )
    keep = ["target_time_utc", "lead_time_h", "y_true", "y_pred", "y_naive_24h", *sorted(set(q_cols))]
    pred_df = pred_df[keep].dropna(subset=["target_time_utc"]).sort_values(["lead_time_h", "target_time_utc"])
    return pred_df


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


def _compute_per_lead_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, float]] = []
    for lead, d in df.groupby("lead_time_h", dropna=True):
        yt = pd.to_numeric(d["y_true"], errors="coerce").to_numpy(dtype=float)
        yp = pd.to_numeric(d["y_pred"], errors="coerce").to_numpy(dtype=float)
        yn = pd.to_numeric(d["y_naive_24h"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(yt) & np.isfinite(yp)
        mn = m & np.isfinite(yn)
        if not np.any(m):
            continue
        mae = float(np.mean(np.abs(yt[m] - yp[m])))
        rmse = float(np.sqrt(np.mean((yt[m] - yp[m]) ** 2)))
        pin = _pinball_loss(yt[m], yp[m], 0.5)
        skill = float("nan")
        if np.any(mn):
            mae_naive = float(np.mean(np.abs(yt[mn] - yn[mn])))
            if np.isfinite(mae_naive) and mae_naive != 0:
                skill = float(1.0 - (mae / mae_naive))
        out.append(
            {
                "lead_time_h": float(lead),
                "n": float(np.sum(m)),
                "mae": mae,
                "rmse": rmse,
                "pinball_mean": pin,
                "skill_score_mae": skill,
            }
        )
    return pd.DataFrame(out).sort_values("lead_time_h").reset_index(drop=True)


def _compute_coverage(df: pd.DataFrame) -> pd.DataFrame:
    q_cols = sorted([c for c in df.columns if re.fullmatch(r"p\d{2}", str(c))], key=lambda x: int(x[1:]))
    rows: list[dict[str, float]] = []
    yt = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    for c in q_cols:
        q = int(c[1:]) / 100.0
        yp = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(yt) & np.isfinite(yp)
        if not np.any(m):
            continue
        cov = float(np.mean(yt[m] <= yp[m]))
        rows.append({"quantile": q, "empirical_coverage": cov, "n": float(np.sum(m))})
    return pd.DataFrame(rows).sort_values("quantile").reset_index(drop=True)


def _compute_scalar_metrics(df: pd.DataFrame, per_lead: pd.DataFrame, coverage: pd.DataFrame) -> dict[str, float]:
    mae_mean = float(per_lead["mae"].mean()) if not per_lead.empty else float("nan")
    rmse_mean = float(per_lead["rmse"].mean()) if not per_lead.empty else float("nan")
    pinball_mean = float(per_lead["pinball_mean"].mean()) if not per_lead.empty else float("nan")
    skill_mean = float(per_lead["skill_score_mae"].mean()) if not per_lead.empty else float("nan")
    ace = float(np.mean(np.abs(coverage["empirical_coverage"] - coverage["quantile"]))) if not coverage.empty else float("nan")

    # WIS (approx.) from available quantiles around p50.
    q_cols = sorted([c for c in df.columns if re.fullmatch(r"p\d{2}", str(c))], key=lambda x: int(x[1:]))
    wis = float("nan")
    if "p50" in q_cols:
        yt = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
        p50 = pd.to_numeric(df["p50"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(yt) & np.isfinite(p50)
        if np.any(m):
            parts = [0.5 * np.abs(yt[m] - p50[m])]
            k = 0
            for lo, hi in [("p10", "p90"), ("p20", "p80"), ("p30", "p70"), ("p40", "p60")]:
                if lo not in q_cols or hi not in q_cols:
                    continue
                l = pd.to_numeric(df[lo], errors="coerce").to_numpy(dtype=float)
                u = pd.to_numeric(df[hi], errors="coerce").to_numpy(dtype=float)
                mm = m & np.isfinite(l) & np.isfinite(u) & (u >= l)
                if not np.any(mm):
                    continue
                alpha = 2.0 * (1.0 - (int(hi[1:]) / 100.0))
                width = u[mm] - l[mm]
                iscore = width + (2.0 / alpha) * np.maximum(l[mm] - yt[mm], 0.0) + (2.0 / alpha) * np.maximum(
                    yt[mm] - u[mm], 0.0
                )
                parts.append((alpha / 2.0) * iscore)
                k += 1
            if parts:
                wis = float(np.mean(np.sum(np.column_stack(parts), axis=1) / (k + 1.0)))

    return {
        "mae_mean": mae_mean,
        "rmse_mean": rmse_mean,
        "pinball_mean": pinball_mean,
        "wis_mean": wis,
        "average_calibration_error": ace,
        "skill_score_mae_mean": skill_mean,
    }


def _latest_h1_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df.loc[pd.to_numeric(df["lead_time_h"], errors="coerce") == 1].copy()
    if d.empty:
        return d
    d["target_time_utc"] = pd.to_datetime(d["target_time_utc"], utc=True, errors="coerce")
    d["y_true"] = pd.to_numeric(d["y_true"], errors="coerce")
    d["y_pred"] = pd.to_numeric(d["y_pred"], errors="coerce")
    d = d.dropna(subset=["target_time_utc", "y_true", "y_pred"])
    d = d.sort_values("target_time_utc").drop_duplicates(subset=["target_time_utc"], keep="last")
    return d.reset_index(drop=True)


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or len(y_pred) < 2:
        return float("nan")
    dy_true = np.sign(np.diff(y_true))
    dy_pred = np.sign(np.diff(y_pred))
    m = np.isfinite(dy_true) & np.isfinite(dy_pred)
    if not bool(np.any(m)):
        return float("nan")
    return float(np.mean(dy_true[m] == dy_pred[m]))


def _compute_time_cv_fold_metrics(
    aligned: pd.DataFrame,
    *,
    n_splits: int,
    test_size: int,
    gap_hours: int,
    min_train_size: int,
) -> pd.DataFrame:
    splitter = PurgedTimeSeriesSplit(
        n_splits=int(n_splits),
        test_size=int(test_size),
        gap_hours=int(gap_hours),
        frequency="1h",
        min_train_size=int(min_train_size),
    )
    rows: list[dict[str, float | int | str]] = []
    for fold, (_, va_idx) in enumerate(splitter.split(aligned), start=1):
        part = aligned.iloc[va_idx].copy()
        yt = pd.to_numeric(part["y_true"], errors="coerce").to_numpy(dtype=float)
        yp = pd.to_numeric(part["y_pred"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[m]
        yp = yp[m]
        if yt.size == 0:
            continue
        err = yp - yt
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        mbe = float(np.mean(err))
        da = _directional_accuracy(yt, yp)
        rows.append(
            {
                "fold": int(fold),
                "n": int(yt.size),
                "start_utc": str(pd.Timestamp(part["target_time_utc"].min()).isoformat()),
                "end_utc": str(pd.Timestamp(part["target_time_utc"].max()).isoformat()),
                "mae": mae,
                "rmse": rmse,
                "mbe": mbe,
                "directional_accuracy": da,
            }
        )
    return pd.DataFrame(rows)


def _build_time_cv_ranking(
    *,
    pred_col: str,
    labels: list[str],
    h1_frames: list[pd.DataFrame],
    n_splits: int,
    test_size: int,
    gap_hours: int,
    min_train_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not h1_frames or len(h1_frames) != len(labels):
        return pd.DataFrame(), pd.DataFrame()

    common_ts: set[pd.Timestamp] | None = None
    for d in h1_frames:
        ts = set(pd.to_datetime(d["target_time_utc"], utc=True, errors="coerce").dropna().tolist())
        common_ts = ts if common_ts is None else common_ts.intersection(ts)
    common_ts_sorted = sorted(common_ts or [])
    if len(common_ts_sorted) < (n_splits * test_size + gap_hours + 1):
        return pd.DataFrame(), pd.DataFrame()

    fold_tables: list[pd.DataFrame] = []
    rank_rows: list[dict[str, float | str | int]] = []

    for lbl, d in zip(labels, h1_frames):
        cur = d.copy()
        cur["target_time_utc"] = pd.to_datetime(cur["target_time_utc"], utc=True, errors="coerce")
        cur = cur[cur["target_time_utc"].isin(common_ts_sorted)].copy()
        cur = cur.sort_values("target_time_utc").reset_index(drop=True)
        fold_df = _compute_time_cv_fold_metrics(
            cur,
            n_splits=n_splits,
            test_size=test_size,
            gap_hours=gap_hours,
            min_train_size=min_train_size,
        )
        if fold_df.empty:
            continue
        fold_df.insert(0, "model", lbl)
        fold_df.insert(0, "prediction_column", pred_col)
        fold_tables.append(fold_df)
        rank_rows.append(
            {
                "prediction_column": pred_col,
                "model": lbl,
                "cv_folds": int(len(fold_df)),
                "cv_mae_mean": float(fold_df["mae"].mean()),
                "cv_mae_std": float(fold_df["mae"].std(ddof=0)),
                "cv_rmse_mean": float(fold_df["rmse"].mean()),
                "cv_rmse_std": float(fold_df["rmse"].std(ddof=0)),
                "cv_mbe_mean": float(fold_df["mbe"].mean()),
                "cv_directional_accuracy_mean": float(fold_df["directional_accuracy"].mean()),
                "common_h1_rows": int(len(common_ts_sorted)),
            }
        )

    if not rank_rows:
        return pd.DataFrame(), pd.DataFrame()

    ranking = pd.DataFrame(rank_rows).sort_values(
        ["prediction_column", "cv_mae_mean", "cv_rmse_mean", "cv_mae_std"],
        ascending=[True, True, True, True],
    )
    ranking["rank"] = ranking.groupby("prediction_column")["cv_mae_mean"].rank(method="dense").astype(int)
    ranking["winner_by_cv_mae"] = ranking["rank"] == 1

    folds = pd.concat(fold_tables, axis=0, ignore_index=True).sort_values(
        ["prediction_column", "model", "fold"]
    )
    return folds, ranking


def _build_summary_table(
    labels: list[str],
    scalar_metrics: list[dict[str, float]],
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
        for idx, lbl in enumerate(labels):
            if key in {"spike_mae_top5", "spike_rmse_top5", "negative_tail_mae_bottom5", "mean_directional_accuracy"}:
                v = float(economic_metrics[idx].get(key, np.nan))
            else:
                v = float(scalar_metrics[idx].get(key, np.nan))
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
    p.add_argument(
        "--model-keys",
        nargs="+",
        default=[],
        help="Optional model key per run-dir (e.g., xgboost tft). Needed when multiple models share one run folder.",
    )
    p.add_argument("--time-cv-n-splits", type=int, default=5, help="Rolling CV folds for final model choice.")
    p.add_argument("--time-cv-test-size-hours", type=int, default=24 * 14, help="Validation window size per fold.")
    p.add_argument("--time-cv-gap-hours", type=int, default=72, help="Purge gap between train and validation windows.")
    p.add_argument("--time-cv-min-train-hours", type=int, default=24 * 30, help="Minimum train rows per fold.")
    p.add_argument(
        "--gate-eval",
        action="store_true",
        help="Enable strict gate-time D+1 evaluation outputs (execution-relevant benchmarking).",
    )
    p.add_argument(
        "--gate-hours-local",
        nargs="*",
        type=int,
        default=[],
        help="Optional gate-hour override(s) for gate mode, e.g. --gate-hours-local 8 11.",
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
    if args.model_keys and len(args.model_keys) != len(run_dirs):
        raise ValueError("Length mismatch: --model-keys must match --run-dirs when provided.")

    model_keys = args.model_keys if args.model_keys else [_infer_model_key(lbl) for lbl in labels]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()

    summaries = [_load_summary(rd) for rd in run_dirs]
    pred_cols = _resolve_pred_cols(summaries, args.pred_col.strip())
    all_summary_tables: list[pd.DataFrame] = []
    all_time_cv_folds: list[pd.DataFrame] = []
    all_time_cv_rankings: list[pd.DataFrame] = []
    gate_summary_rows: list[dict[str, Any]] = []
    gate_lead_rows: list[dict[str, Any]] = []
    plotted_targets: list[str] = []
    skipped_calibration: list[str] = []

    plots_root = out_dir / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    for pred_col in pred_cols:
        long_frames = [
            _load_model_long_with_truth(rd, summ, args.split, pred_col, model_key=mk)
            for rd, summ, mk in zip(run_dirs, summaries, model_keys)
        ]
        per_lead = [_compute_per_lead_metrics(df) for df in long_frames]
        coverages = [_compute_coverage(df) for df in long_frames]
        scalar_metrics = [_compute_scalar_metrics(df, pl, cv) for df, pl, cv in zip(long_frames, per_lead, coverages)]
        econ_metrics = []
        for df in long_frames:
            d = df.copy()
            if "lead_time_h" in d.columns:
                d1 = d[pd.to_numeric(d["lead_time_h"], errors="coerce") == 1]
                if not d1.empty:
                    d = d1
            d = d.sort_values("target_time_utc").drop_duplicates(subset=["target_time_utc"], keep="last")
            econ_metrics.append(_calculate_economic_metrics(d["y_true"], d["y_pred"]))

        h1_frames = [_latest_h1_frame(df) for df in long_frames]
        cv_folds_df, cv_rank_df = _build_time_cv_ranking(
            pred_col=pred_col,
            labels=labels,
            h1_frames=h1_frames,
            n_splits=int(args.time_cv_n_splits),
            test_size=int(args.time_cv_test_size_hours),
            gap_hours=int(args.time_cv_gap_hours),
            min_train_size=int(args.time_cv_min_train_hours),
        )
        if not cv_folds_df.empty:
            all_time_cv_folds.append(cv_folds_df)
        if not cv_rank_df.empty:
            all_time_cv_rankings.append(cv_rank_df)

        summary_df_one = _build_summary_table(labels, scalar_metrics, pred_col, econ_metrics)
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
            _plot_calibration_overlay(
                labels=labels,
                coverages=coverages,
                out_png=target_plot_dir / "plot_c_calibration_overlay.png",
                pred_col=pred_col,
            )
        except FileNotFoundError:
            skipped_calibration.append(pred_col)

        if args.gate_eval:
            gate_hours = args.gate_hours_local if args.gate_hours_local else [None]
            gate_mae_series: list[tuple[str, pd.DataFrame]] = []
            for lbl, df in zip(labels, long_frames):
                for gh in gate_hours:
                    gdf = gate_time_dplus1_filter(
                        df,
                        pred_col=pred_col,
                        local_tz="Europe/Berlin",
                        gate_hour_override=gh,
                    )
                    if gdf.empty:
                        gate_summary_rows.append(
                            {
                                "prediction_column": pred_col,
                                "model": lbl,
                                "gate_hour_local": gh,
                                "n_rows_gate_dplus1": 0,
                                "gate_dplus1_mae": np.nan,
                                "gate_dplus1_rmse": np.nan,
                                "gate_dplus1_mbe": np.nan,
                                "gate_dplus1_directional_accuracy": np.nan,
                            }
                        )
                        continue

                    yt = pd.to_numeric(gdf["y_true"], errors="coerce").to_numpy(dtype=float)
                    yp = pd.to_numeric(gdf["y_pred"], errors="coerce").to_numpy(dtype=float)
                    m = np.isfinite(yt) & np.isfinite(yp)
                    if not bool(np.any(m)):
                        continue
                    err = yp[m] - yt[m]
                    mae = float(np.mean(np.abs(err)))
                    rmse = float(np.sqrt(np.mean(err**2)))
                    mbe = float(np.mean(err))
                    da = _directional_accuracy(yt[m], yp[m])
                    gate_summary_rows.append(
                        {
                            "prediction_column": pred_col,
                            "model": lbl,
                            "gate_hour_local": int(gh) if gh is not None else np.nan,
                            "n_rows_gate_dplus1": int(np.sum(m)),
                            "gate_dplus1_mae": mae,
                            "gate_dplus1_rmse": rmse,
                            "gate_dplus1_mbe": mbe,
                            "gate_dplus1_directional_accuracy": da,
                        }
                    )

                    per_lead_gate = _compute_per_lead_metrics(gdf)
                    if not per_lead_gate.empty:
                        per_lead_gate = per_lead_gate.copy()
                        per_lead_gate["prediction_column"] = pred_col
                        per_lead_gate["model"] = lbl
                        per_lead_gate["gate_hour_local"] = int(gh) if gh is not None else np.nan
                        gate_lead_rows.extend(per_lead_gate.to_dict(orient="records"))
                        gate_mae_series.append((f"{lbl} (gate={gh if gh is not None else 'default'})", per_lead_gate))

            # Gate-time lead-decay overlay.
            if gate_mae_series:
                fig, ax = plt.subplots(figsize=(10, 5))
                markers = ["o", "s", "D", "^", "v", "P", "X"]
                for i, (name, plg) in enumerate(gate_mae_series):
                    d = plg.sort_values("lead_time_h")
                    ax.plot(
                        d["lead_time_h"],
                        d["mae"],
                        label=name,
                        linewidth=2,
                        marker=markers[i % len(markers)],
                        markevery=max(1, len(d) // 12),
                        markersize=4,
                    )
                ax.set_title(f"Gate-Time D+1 Lead Decay (MAE) - {pred_col}")
                ax.set_xlabel("Lead Time [h]")
                ax.set_ylabel("MAE")
                ax.legend()
                fig.tight_layout()
                _save_plot_png_pdf(fig, target_plot_dir / "plot_d_gate_time_degradation_overlay_mae.png")
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

    time_cv_folds_csv = out_dir / "time_cv_fold_metrics.csv"
    time_cv_rank_csv = out_dir / "time_cv_model_choice.csv"
    if all_time_cv_folds:
        pd.concat(all_time_cv_folds, axis=0, ignore_index=True).to_csv(time_cv_folds_csv, index=False)
    if all_time_cv_rankings:
        rank_df = pd.concat(all_time_cv_rankings, axis=0, ignore_index=True).sort_values(
            ["prediction_column", "rank", "cv_mae_mean"]
        )
        rank_df.to_csv(time_cv_rank_csv, index=False)

    gate_summary_csv = out_dir / "gate_time_benchmark_summary.csv"
    gate_lead_csv = out_dir / "gate_time_lead_metrics.csv"
    if args.gate_eval and gate_summary_rows:
        pd.DataFrame(gate_summary_rows).sort_values(["prediction_column", "model"]).to_csv(gate_summary_csv, index=False)
    if args.gate_eval and gate_lead_rows:
        pd.DataFrame(gate_lead_rows).sort_values(["prediction_column", "model", "lead_time_h"]).to_csv(gate_lead_csv, index=False)

    meta = {
        "run_dirs": [str(p.resolve()) for p in run_dirs],
        "labels": labels,
        "model_keys": model_keys,
        "split": args.split,
        "pred_cols": pred_cols,
        "plotted_targets": plotted_targets,
        "skipped_calibration_targets": skipped_calibration,
        "summary_csv": str(summary_csv.resolve()),
        "summary_tex": str(summary_tex.resolve()),
        "time_cv_folds_csv": str(time_cv_folds_csv.resolve()) if all_time_cv_folds else None,
        "time_cv_model_choice_csv": str(time_cv_rank_csv.resolve()) if all_time_cv_rankings else None,
        "time_cv": {
            "n_splits": int(args.time_cv_n_splits),
            "test_size_hours": int(args.time_cv_test_size_hours),
            "gap_hours": int(args.time_cv_gap_hours),
            "min_train_hours": int(args.time_cv_min_train_hours),
        },
        "gate_eval_enabled": bool(args.gate_eval),
        "gate_hours_local": [int(x) for x in args.gate_hours_local] if args.gate_hours_local else [],
        "gate_time_summary_csv": str(gate_summary_csv.resolve()) if (args.gate_eval and gate_summary_rows) else None,
        "gate_time_lead_csv": str(gate_lead_csv.resolve()) if (args.gate_eval and gate_lead_rows) else None,
    }
    (out_dir / "benchmark_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved benchmark outputs:")
    print(f"- {summary_csv}")
    print(f"- {summary_tex}")
    if all_time_cv_folds:
        print(f"- {time_cv_folds_csv}")
    if all_time_cv_rankings:
        print(f"- {time_cv_rank_csv}")
    if args.gate_eval and gate_summary_rows:
        print(f"- {gate_summary_csv}")
    if args.gate_eval and gate_lead_rows:
        print(f"- {gate_lead_csv}")
    print(f"- plots root: {plots_root}")
    print(f"- {out_dir / 'benchmark_meta.json'}")


if __name__ == "__main__":
    main()
