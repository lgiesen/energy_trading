"""Trading-oriented evaluation metrics for energy-price forecasts.

The PnL metric translates forecast quality into economic value via a simplified
battery dispatch simulation. Use unclipped, unlagged y_true prices to ensure
financial evaluation remains unbiased and causally correct.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BatteryParams:
    """Battery and strategy configuration for PnL simulation."""

    capacity_mwh: float
    power_mw: float
    roundtrip_efficiency: float = 0.9
    initial_soc_mwh: float | None = None
    interval_hours: float = 1.0
    high_percentile: float = 80.0
    low_percentile: float = 20.0


def _validate_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    if yt.shape[0] == 0:
        raise ValueError("y_true/y_pred are empty")
    return yt, yp


def _validate_params(params: BatteryParams) -> None:
    if params.capacity_mwh <= 0:
        raise ValueError("capacity_mwh must be > 0")
    if params.power_mw <= 0:
        raise ValueError("power_mw must be > 0")
    if not (0 < params.roundtrip_efficiency <= 1):
        raise ValueError("roundtrip_efficiency must be in (0, 1]")
    if params.interval_hours <= 0:
        raise ValueError("interval_hours must be > 0")
    if not (0 <= params.low_percentile < params.high_percentile <= 100):
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")


def calculate_pnl(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    battery_params: BatteryParams | dict[str, Any],
) -> dict[str, Any]:
    """Simulate battery trading decisions and return economic KPIs.

    Decision logic:
    - If y_pred is in top percentile: discharge (sell).
    - If y_pred is in bottom percentile: charge (buy).
    - Otherwise: hold.

    Leakage guard:
    - Dispatch decisions are based only on y_pred.
    - Settlement uses y_true (must be raw/unclipped, unlagged actual price).
    """
    params = battery_params if isinstance(battery_params, BatteryParams) else BatteryParams(**battery_params)
    _validate_params(params)
    yt, yp = _validate_arrays(np.asarray(y_true), np.asarray(y_pred))

    high_thr = float(np.nanpercentile(yp, params.high_percentile))
    low_thr = float(np.nanpercentile(yp, params.low_percentile))

    eta_c = np.sqrt(params.roundtrip_efficiency)
    eta_d = np.sqrt(params.roundtrip_efficiency)

    max_internal_per_step = params.power_mw * params.interval_hours
    soc = (
        float(params.initial_soc_mwh)
        if params.initial_soc_mwh is not None
        else 0.5 * params.capacity_mwh
    )
    soc = float(np.clip(soc, 0.0, params.capacity_mwh))

    pnl_total = 0.0
    revenue_total = 0.0
    cost_total = 0.0
    charge_actions = 0
    discharge_actions = 0

    decisions = np.zeros_like(yp, dtype=np.int8)  # -1 charge, 0 hold, +1 discharge
    soc_path = np.zeros_like(yp, dtype=float)
    pnl_path = np.zeros_like(yp, dtype=float)

    for i, (price_true, pred) in enumerate(zip(yt, yp)):
        if not np.isfinite(price_true) or not np.isfinite(pred):
            soc_path[i] = soc
            pnl_path[i] = pnl_total
            continue

        if pred >= high_thr:
            decisions[i] = 1
            # Internal energy removed from battery this step.
            e_internal = min(max_internal_per_step, soc)
            if e_internal > 0:
                e_grid = e_internal * eta_d
                revenue = e_grid * price_true
                revenue_total += revenue
                pnl_total += revenue
                soc -= e_internal
                discharge_actions += 1

        elif pred <= low_thr:
            decisions[i] = -1
            free_capacity = params.capacity_mwh - soc
            e_internal = min(max_internal_per_step, free_capacity)
            if e_internal > 0:
                # Energy bought from grid considering charge efficiency.
                e_grid = e_internal / eta_c
                cost = e_grid * price_true
                cost_total += cost
                pnl_total -= cost
                soc += e_internal
                charge_actions += 1

        soc_path[i] = soc
        pnl_path[i] = pnl_total

    trades = charge_actions + discharge_actions
    avg_trade_value = pnl_total / trades if trades else 0.0

    return {
        "pnl_eur": float(pnl_total),
        "revenue_eur": float(revenue_total),
        "cost_eur": float(cost_total),
        "n_charge_actions": int(charge_actions),
        "n_discharge_actions": int(discharge_actions),
        "n_actions_total": int(trades),
        "avg_pnl_per_action_eur": float(avg_trade_value),
        "threshold_high_pred": high_thr,
        "threshold_low_pred": low_thr,
        "decision": decisions,
        "soc_path_mwh": soc_path,
        "pnl_path_eur": pnl_path,
    }
