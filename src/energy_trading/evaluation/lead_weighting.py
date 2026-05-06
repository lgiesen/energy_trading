"""Lead-time weighting helpers for long-horizon model selection/evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def lead_weights(
    leads: np.ndarray | list[float] | pd.Series,
    *,
    start_lead: int = 16,
    end_lead: int = 48,
    max_weight: float = 2.0,
) -> np.ndarray:
    """Piecewise-linear lead weighting.

    - leads < start_lead: weight = 1.0
    - leads >= end_lead: weight = max_weight
    - in between: linear ramp from 1.0 to max_weight
    """
    arr = np.asarray(pd.to_numeric(pd.Series(leads), errors="coerce"), dtype=float)
    w = np.ones_like(arr, dtype=float)
    if not np.isfinite(arr).any():
        return w
    s = float(max(1, int(start_lead)))
    e = float(max(int(end_lead), int(start_lead) + 1))
    mw = float(max(1.0, max_weight))
    mid = (arr >= s) & (arr < e)
    w[arr >= e] = mw
    w[mid] = 1.0 + (mw - 1.0) * ((arr[mid] - s) / (e - s))
    w[~np.isfinite(arr)] = np.nan
    return w


def weighted_metric_from_decay(
    decay_df: pd.DataFrame,
    *,
    value_col: str = "mae",
    count_col: str = "n",
    start_lead: int = 16,
    end_lead: int = 48,
    max_weight: float = 2.0,
) -> float:
    """Compute weighted mean metric from lead-time decay table.

    Expects columns:
    - lead_time_h
    - value_col (e.g., mae/rmse)
    - optional count_col as reliability weight
    """
    if decay_df.empty or "lead_time_h" not in decay_df.columns or value_col not in decay_df.columns:
        return float("nan")
    d = decay_df.copy()
    lead = pd.to_numeric(d["lead_time_h"], errors="coerce").to_numpy(dtype=float)
    val = pd.to_numeric(d[value_col], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(lead) & np.isfinite(val)
    if not np.any(m):
        return float("nan")
    lead = lead[m]
    val = val[m]
    w_lead = lead_weights(
        lead,
        start_lead=start_lead,
        end_lead=end_lead,
        max_weight=max_weight,
    )
    if count_col in d.columns:
        n = pd.to_numeric(d.loc[m, count_col], errors="coerce").to_numpy(dtype=float)
        n = np.where(np.isfinite(n) & (n > 0), n, 1.0)
    else:
        n = np.ones_like(val, dtype=float)
    w = w_lead * n
    return float(np.average(val, weights=w)) if np.isfinite(w).any() and np.sum(w) > 0 else float("nan")

