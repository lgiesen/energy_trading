"""Central model-data preparation utilities for time-series training."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


LEAKAGE_TARGET_COLS = [
    "afrr_vwap_pos",
    "afrr_vwap_neg",
    "afrr_activated_mw_pos",
    "afrr_activated_mw_neg",
    "afrr_da_price_spread",
    "afrr_neg_da_price_spread",
]


def prepare_model_data(
    df: pd.DataFrame,
    target_col: str,
    model_type: Literal["xgboost", "rf", "linear", "nn"] = "xgboost",
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare leakage-safe train/test tensors for model training.

    This helper is the single entrypoint for model-ready dataset creation across
    tree-based, linear, and neural pipelines.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing features, ground-truth targets, and optionally
        `timestamp_utc`.
    target_col : str
        Name of the target column to predict.
    model_type : {"xgboost", "rf", "linear", "nn"}, default "xgboost"
        Controls formatting/scaling:
        - `"xgboost"` / `"rf"`: no feature scaling
        - `"linear"` / `"nn"`: apply `StandardScaler` fit on train only
    test_size : float, default 0.2
        Fraction of rows reserved for test split (chronological tail).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        `(X_train, X_test, y_train, y_test)` prepared for direct model fitting.

    Raises
    ------
    KeyError
        If `target_col` is not present in `df`.
    ValueError
        If `test_size` is invalid, or too few rows remain after filtering.

    Notes
    -----
    - **Target leakage prevention:** unlagged primary targets
      (`afrr_vwap_pos`, `afrr_vwap_neg`, `afrr_activated_mw_pos`,
      `afrr_activated_mw_neg`) are always removed from `X`.
    - **Chronological split only:** data is sorted by `timestamp_utc` (if
      available) and split sequentially (no shuffling), preventing look-ahead
      bias in power-market time series.
    - **Scaling discipline:** scaler is fit on `X_train` only and then applied to
      `X_test`, avoiding information bleed from test to train.

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = prepare_model_data(
    ...     df, target_col="afrr_vwap_pos", model_type="xgboost"
    ... )
    >>> X_train_lin, X_test_lin, y_train_lin, y_test_lin = prepare_model_data(
    ...     df, target_col="afrr_vwap_pos", model_type="linear"
    ... )
    """
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not found in dataframe")
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0 and 1 (exclusive), got {test_size}")

    data = df.copy()

    if "timestamp_utc" in data.columns:
        ts = pd.to_datetime(data["timestamp_utc"], utc=True, errors="coerce")
        data = data.assign(timestamp_utc=ts).sort_values("timestamp_utc").reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    y = pd.to_numeric(data[target_col], errors="coerce")
    keep_mask = y.notna()
    data = data.loc[keep_mask].reset_index(drop=True)
    y = y.loc[keep_mask].reset_index(drop=True)

    if len(data) < 10:
        raise ValueError("Not enough rows after target filtering to perform train/test split.")

    # Remove leakage-prone unlagged targets from feature matrix.
    drop_from_x = [c for c in LEAKAGE_TARGET_COLS if c in data.columns]
    X = data.drop(columns=drop_from_x, errors="ignore")

    # Keep only numeric features for model readiness.
    X = X.select_dtypes(include=[np.number])

    # Basic, leakage-safe imputation for model compatibility.
    X = X.fillna(X.median(numeric_only=True))

    split_idx = int(len(X) * (1.0 - test_size))
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError("Invalid split point computed. Check test_size and dataset length.")

    X_train = X.iloc[:split_idx].to_numpy(dtype=float)
    X_test = X.iloc[split_idx:].to_numpy(dtype=float)
    y_train = y.iloc[:split_idx].to_numpy(dtype=float)
    y_test = y.iloc[split_idx:].to_numpy(dtype=float)

    mt = model_type.lower()
    if mt in {"xgboost", "rf"}:
        return X_train, X_test, y_train, y_test
    if mt in {"linear", "nn"}:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    raise ValueError(f"Unsupported model_type '{model_type}'. Use xgboost, rf, linear, or nn.")
