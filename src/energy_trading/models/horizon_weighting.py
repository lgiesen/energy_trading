"""Target-specific horizon weighting utilities for training losses only."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


_BASELINE_WEIGHT = 0.05
_HALF_LIFE_HOURS = 6.0


def _as_hours_array(current_hour: int | Iterable[int] | np.ndarray) -> np.ndarray:
    h = np.asarray(current_hour, dtype=float)
    if h.ndim == 0:
        h = h.reshape(1)
    h = np.mod(np.floor(h), 24.0)
    return h.astype(int)


def _target_group(target_name: str) -> str:
    t = str(target_name).lower()
    if ("afrr_activation_price_" in t) or ("afrr_activation_rate_" in t):
        return "activation"
    if "afrr_capacity_price_" in t:
        return "capacity"
    if ("da_price" == t) or t.endswith("da_price") or ("target_da_price" in t):
        return "da"
    raise ValueError(f"Unknown target for horizon weighting: {target_name}")


def _auction_window_mask(current_hours: np.ndarray, horizon: int, gate_hour: int) -> np.ndarray:
    lead = np.arange(1, int(horizon) + 1, dtype=int)[None, :]
    curr = current_hours[:, None]
    delivery_hour_from_today0 = curr + lead
    in_dplus1_window = (delivery_hour_from_today0 >= 24) & (delivery_hour_from_today0 < 48)
    at_last_bidding_hour = curr == int(gate_hour)
    return at_last_bidding_hour & in_dplus1_window


def get_training_weights(
    target_name: str,
    current_hour: int | Iterable[int] | np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Return target-specific horizon weights.

    Output shape:
    - scalar current_hour -> [horizon]
    - vector current_hour length N -> [N, horizon]
    """
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    current_hours = _as_hours_array(current_hour)
    n = int(current_hours.shape[0])
    group = _target_group(target_name)

    if group == "activation":
        lead = np.arange(1, horizon + 1, dtype=float)
        decay = np.exp(-math.log(2.0) * (lead - 1.0) / _HALF_LIFE_HOURS)
        decay = np.maximum(_BASELINE_WEIGHT, decay)
        out = np.tile(decay[None, :], (n, 1))
    elif group == "capacity":
        mask = _auction_window_mask(current_hours, horizon=horizon, gate_hour=8)
        out = np.where(mask, 1.0, _BASELINE_WEIGHT).astype(float)
    elif group == "da":
        mask = _auction_window_mask(current_hours, horizon=horizon, gate_hour=12)
        out = np.where(mask, 1.0, _BASELINE_WEIGHT).astype(float)
    else:
        raise ValueError(f"Unknown target for horizon weighting: {target_name}")

    if np.asarray(current_hour).ndim == 0:
        return out[0]
    return out


def get_lead_sample_weights(
    target_name: str,
    current_hour: int | Iterable[int] | np.ndarray,
    lead_time_h: int,
) -> np.ndarray:
    """Return per-sample weights for one fixed lead time h."""
    lead = int(lead_time_h)
    if lead <= 0:
        raise ValueError("lead_time_h must be > 0")

    current_hours = _as_hours_array(current_hour)
    group = _target_group(target_name)

    if group == "activation":
        scalar = max(
            _BASELINE_WEIGHT,
            math.exp(-math.log(2.0) * (float(lead) - 1.0) / _HALF_LIFE_HOURS),
        )
        out = np.full(current_hours.shape[0], float(scalar), dtype=float)
    elif group == "capacity":
        delivery_hour_from_today0 = current_hours + lead
        mask = (current_hours == 8) & (delivery_hour_from_today0 >= 24) & (delivery_hour_from_today0 < 48)
        out = np.where(mask, 1.0, _BASELINE_WEIGHT).astype(float)
    elif group == "da":
        delivery_hour_from_today0 = current_hours + lead
        mask = (current_hours == 12) & (delivery_hour_from_today0 >= 24) & (delivery_hour_from_today0 < 48)
        out = np.where(mask, 1.0, _BASELINE_WEIGHT).astype(float)
    else:
        raise ValueError(f"Unknown target for horizon weighting: {target_name}")
    return out
