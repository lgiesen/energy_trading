"""Split-conformal calibration utilities for quantile forecasts.

This module provides:
1) calculate_conformal_shifts: derive scalar per-quantile additive shifts from a
   calibration set.
2) apply_conformal_shifts: apply these shifts to prediction quantiles and enforce
   monotonic quantile order.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _qcol(alpha: float) -> str:
    return f"p{int(round(alpha * 100)):02d}"


def _finite_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[m], y_pred[m]


def _quantile_with_fallback(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.quantile(x, q))


def calculate_conformal_shifts(
    y_true_calib: Any,
    q_preds_calib_dict: dict[str, Any],
    alphas: list[float] | tuple[float, ...],
) -> dict[str, float]:
    """Compute additive per-quantile shifts from calibration residuals.

    For quantile q:
      shift_q = quantile_q(y_true - y_q_pred)
    so calibrated quantile is:
      y_q_cal = y_q_pred + shift_q
    """
    yt = np.asarray(y_true_calib, dtype=float).reshape(-1)
    shifts: dict[str, float] = {}
    for q in alphas:
        c = _qcol(float(q))
        if c not in q_preds_calib_dict:
            shifts[c] = 0.0
            continue
        yp = np.asarray(q_preds_calib_dict[c], dtype=float).reshape(-1)
        n = min(yt.size, yp.size)
        yt_n, yp_n = _finite_pair(yt[:n], yp[:n])
        e = yt_n - yp_n
        shifts[c] = _quantile_with_fallback(e, float(q))
    return shifts


def apply_conformal_shifts(
    q_preds_test_dict: dict[str, Any],
    shifts_dict: dict[str, float],
) -> dict[str, np.ndarray]:
    """Apply additive shifts and enforce monotonicity across available pXX keys."""
    out: dict[str, np.ndarray] = {}
    q_items: list[tuple[float, str, np.ndarray]] = []
    for k, v in q_preds_test_dict.items():
        ks = str(k).lower().strip()
        if ks.startswith("p") and ks[1:].isdigit():
            q = float(int(ks[1:])) / 100.0
            arr = np.asarray(v, dtype=float).reshape(-1)
            shift = float(shifts_dict.get(ks, 0.0))
            q_items.append((q, ks, arr + shift))
        else:
            out[k] = np.asarray(v, dtype=float).reshape(-1)

    if not q_items:
        return out

    q_items.sort(key=lambda t: t[0])
    n = min(len(a) for _, _, a in q_items)
    stack = np.column_stack([a[:n] for _, _, a in q_items])
    stack = np.sort(stack, axis=1)
    for i, (_, key, _) in enumerate(q_items):
        out[key] = stack[:, i]
    return out

