from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mean_error(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(yhat - y))


def median_absolute_error(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.median(np.abs(y - yhat)))


def normalized_mae(y: np.ndarray, yhat: np.ndarray) -> float:
    iqr = float(np.nanpercentile(y, 75) - np.nanpercentile(y, 25))
    denom = iqr if iqr > 1e-12 else float(np.nanmean(np.abs(y)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.mean(np.abs(y - yhat)) / denom)


def pinball_loss(y: np.ndarray, yq: np.ndarray, q: float) -> float:
    e = y - yq
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def mean_pinball_loss(y: np.ndarray, q_preds: dict[float, np.ndarray]) -> float:
    vals = [pinball_loss(y, arr, q) for q, arr in q_preds.items()]
    return float(np.mean(vals)) if vals else float("nan")


def approx_crps(y: np.ndarray, q_preds: dict[float, np.ndarray]) -> float:
    pairs = sorted((float(q), pinball_loss(y, arr, float(q))) for q, arr in q_preds.items())
    if len(pairs) < 2:
        return float("nan")
    qs = np.asarray([p[0] for p in pairs], dtype=float)
    ls = np.asarray([p[1] for p in pairs], dtype=float)
    trapz_fn = getattr(np, "trapezoid", None)
    if trapz_fn is None:
        trapz_fn = getattr(np, "trapz", None)
    if trapz_fn is None:
        raise AttributeError("NumPy has neither trapezoid nor trapz; cannot compute CRPS approximation.")
    return float(2.0 * trapz_fn(ls, qs))


def empirical_coverage(y: np.ndarray, yq: np.ndarray) -> float:
    return float(np.mean(y <= yq))


def interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def winkler_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    width = hi - lo
    s = width.copy()
    below = y < lo
    above = y > hi
    s[below] += (2.0 / alpha) * (lo[below] - y[below])
    s[above] += (2.0 / alpha) * (y[above] - hi[above])
    return float(np.mean(s))


def quantile_crossing_metrics(q_preds: dict[float, np.ndarray]) -> tuple[float, float]:
    if len(q_preds) < 2:
        return 0.0, 0.0
    qs = sorted(q_preds)
    arr = np.vstack([q_preds[q] for q in qs])
    diffs = np.diff(arr, axis=0)
    violated = diffs < 0.0
    crossing_rate = float(np.mean(np.any(violated, axis=0)))
    max_violation = float(np.max(np.where(violated, -diffs, 0.0))) if violated.any() else 0.0
    return crossing_rate, max_violation


def repair_monotone_quantiles(q_preds: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    qs = sorted(q_preds)
    arr = np.vstack([q_preds[q] for q in qs])
    repaired = np.maximum.accumulate(arr, axis=0)
    return {q: repaired[i] for i, q in enumerate(qs)}


@dataclass(frozen=True)
class TailConfig:
    upper_tail_q: float = 0.90
    extreme_upper_tail_q: float = 0.95
    lower_tail_q: float = 0.10
    activation_event_threshold: float = 0.0
    high_activation_rate_q: float = 0.90


def tail_event_metrics(
    *,
    y: np.ndarray,
    yhat_p50: np.ndarray,
    cfg: TailConfig,
    value_weights: np.ndarray | None = None,
    top_k: int = 25,
) -> dict[str, float]:
    q_hi = float(np.nanquantile(y, cfg.upper_tail_q))
    q_lo = float(np.nanquantile(y, cfg.lower_tail_q))
    mask_hi = y >= q_hi
    mask_lo = y <= q_lo
    tail_mask = mask_hi | mask_lo
    if not np.any(tail_mask):
        return {
            "tail_mae": float("nan"),
            "tail_rmse": float("nan"),
            "tail_bias": float("nan"),
            "event_precision": float("nan"),
            "event_recall": float("nan"),
            "event_f1": float("nan"),
            "top_k_capture": float("nan"),
            "value_weighted_abs_error": float("nan"),
        }
    tail_mae = mae(y[tail_mask], yhat_p50[tail_mask])
    tail_rmse = rmse(y[tail_mask], yhat_p50[tail_mask])
    tail_bias = mean_error(y[tail_mask], yhat_p50[tail_mask])

    pred_hi = yhat_p50 >= q_hi
    tp = float(np.sum(pred_hi & mask_hi))
    fp = float(np.sum(pred_hi & (~mask_hi)))
    fn = float(np.sum((~pred_hi) & mask_hi))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0) else float("nan")

    k = min(int(top_k), len(y))
    top_truth_idx = np.argsort(-np.abs(y))[:k]
    top_pred_idx = np.argsort(-np.abs(yhat_p50))[:k]
    top_capture = float(len(set(top_truth_idx).intersection(set(top_pred_idx))) / max(1, k))

    if value_weights is None:
        value_weights = np.ones_like(y)
    vw = np.abs(y - yhat_p50) * np.asarray(value_weights)
    vw_denom = float(np.sum(np.abs(value_weights)))
    vw_err = float(np.sum(vw) / vw_denom) if vw_denom > 1e-12 else float("nan")

    return {
        "tail_mae": tail_mae,
        "tail_rmse": tail_rmse,
        "tail_bias": tail_bias,
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "top_k_capture": top_capture,
        "value_weighted_abs_error": vw_err,
    }


def assign_horizon_bucket(lead_h: float, buckets: dict[str, Iterable[int]]) -> str:
    for name, bounds in buckets.items():
        lo, hi = list(bounds)
        if float(lo) <= float(lead_h) <= float(hi):
            return str(name)
    return "out_of_range"


def assign_gate_context(ts: pd.Timestamp, lead_h: float) -> list[str]:
    contexts = ["all_horizon"]
    if 1 <= float(lead_h) <= 4:
        contexts.append("bem_short_horizon")
    if 12 <= float(lead_h) <= 36:
        contexts.append("da_gate_relevant")
    if 12 <= float(lead_h) <= 36:
        contexts.append("afrr_capacity_gate_relevant")
    return contexts
