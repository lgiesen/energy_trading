"""Shared evaluation + canonical logging/export utilities across model families.

This module standardizes:
1) Metric computation
2) Canonical metrics parquet export
3) Canonical predictions parquet export
4) Canonical TensorBoard tag taxonomy:
   [split]/[metric_name]/[target_col]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


def _as_1d_float(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr


def _safe_mean(x: np.ndarray) -> float | None:
    if x.size == 0:
        return None
    if not np.isfinite(x).any():
        return None
    return float(np.nanmean(x))


def _extract_point_prediction(predictions: Any) -> np.ndarray:
    """Extract deterministic/median forecast from either array-like or dict-like input."""
    if isinstance(predictions, dict):
        if "point" not in predictions:
            raise KeyError("predictions_dict must contain key 'point'.")
        return _as_1d_float(predictions["point"])
    return _as_1d_float(predictions)


def _extract_quantile_predictions(predictions: Any) -> dict[float, np.ndarray]:
    """Extract quantile prediction arrays from dict keys like p01/p05/.../p99."""
    if not isinstance(predictions, dict):
        return {}
    out: dict[float, np.ndarray] = {}
    for k, v in predictions.items():
        ks = str(k).strip().lower()
        if ks == "point":
            continue
        if ks.startswith("p") and len(ks) in (3, 4) and ks[1:].isdigit():
            q = float(int(ks[1:])) / 100.0
            if 0.0 < q < 1.0:
                out[q] = _as_1d_float(v)
    return out


def _tail_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> tuple[float | None, float | None]:
    if y_true.size == 0 or mask.size == 0 or not bool(mask.any()):
        return None, None
    yt = y_true[mask]
    yp = y_pred[mask]
    if yt.size == 0:
        return None, None
    return (
        float(mean_absolute_error(yt, yp)),
        float(np.sqrt(mean_squared_error(yt, yp))),
    )


def compute_shared_metrics(y_true: Any, predictions: Any) -> dict[str, float | None]:
    """Compute unified deterministic metric suite.

    Returns:
      rmse, mae, wmape, mbe, directional_accuracy, directional_mae, over_prediction_ratio
    """
    yt = _as_1d_float(y_true)
    yp = _extract_point_prediction(predictions)
    qmap = _extract_quantile_predictions(predictions)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]

    out: dict[str, float | None] = {
        "rmse": None,
        "mae": None,
        "wmape": None,
        "mbe": None,
        "directional_accuracy": None,
        "directional_mae": None,
        "over_prediction_ratio": None,
        "tail_upper_mae": None,
        "tail_upper_rmse": None,
        "spike_precision": None,
        "spike_recall": None,
        "spike_f1": None,
        "tail_lower_mae": None,
        "tail_lower_rmse": None,
    }
    if yt.size == 0:
        return out

    err = yp - yt
    abs_err = np.abs(err)
    sq_err = err * err

    out["rmse"] = float(np.sqrt(np.mean(sq_err)))
    out["mae"] = float(np.mean(abs_err))
    denom = float(np.sum(np.abs(yt)))
    out["wmape"] = (float(np.sum(abs_err)) / denom) if denom > 1e-12 else None
    out["mbe"] = float(np.mean(err))
    out["over_prediction_ratio"] = float(np.mean(yp > yt))

    if yt.size >= 2:
        dyt = np.diff(yt)
        dyp = np.diff(yp)
        dmask = np.isfinite(dyt) & np.isfinite(dyp)
        if dmask.any():
            out["directional_accuracy"] = float(np.mean(np.sign(dyt[dmask]) == np.sign(dyp[dmask])))
            out["directional_mae"] = float(np.mean(np.abs(dyt[dmask] - dyp[dmask])))

    # Tail / extreme-event metrics
    try:
        p90 = float(np.percentile(yt, 90))
        p10 = float(np.percentile(yt, 10))
    except Exception:
        return out

    upper_mask = yt >= p90
    lower_mask = yt <= p10

    upper_mae, upper_rmse = _tail_regression_metrics(yt, yp, upper_mask)
    out["tail_upper_mae"] = upper_mae
    out["tail_upper_rmse"] = upper_rmse

    lower_mae, lower_rmse = _tail_regression_metrics(yt, yp, lower_mask)
    out["tail_lower_mae"] = lower_mae
    out["tail_lower_rmse"] = lower_rmse

    # Spike classification at dynamic upper-tail threshold.
    y_true_spike = (yt >= p90).astype(int)
    y_pred_spike = (yp >= p90).astype(int)
    if y_true_spike.size > 0:
        out["spike_precision"] = float(precision_score(y_true_spike, y_pred_spike, zero_division=0.0))
        out["spike_recall"] = float(recall_score(y_true_spike, y_pred_spike, zero_division=0.0))
        out["spike_f1"] = float(f1_score(y_true_spike, y_pred_spike, zero_division=0.0))

    # Optional probabilistic calibration diagnostics:
    # coverage_q = mean(y_true <= y_q), coverage_error_q = coverage_q - q
    for q, qpred_raw in sorted(qmap.items(), key=lambda kv: kv[0]):
        qpred = np.asarray(qpred_raw, dtype=float).reshape(-1)
        n = min(yt.size, qpred.size)
        if n == 0:
            out[f"coverage_p{int(round(q * 100)):02d}"] = None
            out[f"coverage_error_p{int(round(q * 100)):02d}"] = None
            continue
        yq = qpred[:n]
        yv = yt[:n]
        m = np.isfinite(yv) & np.isfinite(yq)
        if not np.any(m):
            out[f"coverage_p{int(round(q * 100)):02d}"] = None
            out[f"coverage_error_p{int(round(q * 100)):02d}"] = None
            continue
        cov = float(np.mean(yv[m] <= yq[m]))
        out[f"coverage_p{int(round(q * 100)):02d}"] = cov
        out[f"coverage_error_p{int(round(q * 100)):02d}"] = float(cov - q)
    return out


@dataclass
class EvalMetadata:
    model_family: str
    target_col: str
    split: str
    lead_h: int | None = None


def metrics_to_canonical_rows(
    metrics: dict[str, float | None],
    meta: EvalMetadata,
) -> pd.DataFrame:
    rows = []
    for metric_name, metric_value in metrics.items():
        rows.append(
            {
                "model_family": str(meta.model_family),
                "target_col": str(meta.target_col),
                "split": str(meta.split),
                "lead_h": (int(meta.lead_h) if meta.lead_h is not None else pd.NA),
                "metric_name": str(metric_name),
                "metric_value": (float(metric_value) if metric_value is not None and np.isfinite(metric_value) else pd.NA),
            }
        )
    return pd.DataFrame(rows)


def append_canonical_metrics_parquet(
    rows_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model_family", "target_col", "split", "lead_h", "metric_name", "metric_value"]
    missing = [c for c in cols if c not in rows_df.columns]
    if missing:
        raise KeyError(f"rows_df missing canonical columns: {missing}")
    rows_df = rows_df.loc[:, cols].copy()

    if out.exists():
        prev = pd.read_parquet(out)
        merged = pd.concat([prev, rows_df], axis=0, ignore_index=True)
        merged.to_parquet(out, index=False)
    else:
        rows_df.to_parquet(out, index=False)
    return out


def predictions_to_canonical_df(
    *,
    timestamps: Any,
    y_true: Any,
    y_pred: Any,
    meta: EvalMetadata,
) -> pd.DataFrame:
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    yt = pd.to_numeric(pd.Series(y_true), errors="coerce")
    yp = pd.to_numeric(pd.Series(y_pred), errors="coerce")
    n = min(len(ts), len(yt), len(yp))
    df = pd.DataFrame(
        {
            "timestamp": ts.iloc[:n].to_numpy(),
            "model_family": str(meta.model_family),
            "target_col": str(meta.target_col),
            "lead_h": (int(meta.lead_h) if meta.lead_h is not None else pd.NA),
            "actual_value": yt.iloc[:n].to_numpy(dtype=float),
            "predicted_value": yp.iloc[:n].to_numpy(dtype=float),
            "split": str(meta.split),
        }
    )
    return df


def append_canonical_predictions_parquet(
    pred_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "timestamp",
        "model_family",
        "target_col",
        "lead_h",
        "actual_value",
        "predicted_value",
        "split",
    ]
    missing = [c for c in cols if c not in pred_df.columns]
    if missing:
        raise KeyError(f"pred_df missing canonical columns: {missing}")
    pred_df = pred_df.loc[:, cols].copy()

    if out.exists():
        prev = pd.read_parquet(out)
        merged = pd.concat([prev, pred_df], axis=0, ignore_index=True)
        merged.to_parquet(out, index=False)
    else:
        pred_df.to_parquet(out, index=False)
    return out


def log_metrics_tensorboard_canonical(
    writer: Any,
    *,
    split: str,
    target_col: str,
    metrics: dict[str, float | None],
    step: int = 0,
) -> int:
    """Log metrics under canonical tags: [split]/[metric_name]/[target_col]."""
    if writer is None:
        return 0
    n = 0
    for name, value in metrics.items():
        if value is None or not np.isfinite(value):
            continue
        writer.add_scalar(f"{split}/{name}/{target_col}", float(value), int(step))
        n += 1
    return n
