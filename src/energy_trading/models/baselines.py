"""Baseline models for leakage-safe benchmark evaluation.

The persistence baseline uses only historical target values and therefore
respects causality. It is intended as the minimum benchmark an ML model must
beat before being considered useful.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class PersistencePredictor(BaseEstimator, RegressorMixin):
    """168h persistence benchmark (or configurable lag horizon).

    For each timestamp t, prediction is target(t - lag_hours). This uses only
    known history and avoids look-ahead leakage by design.
    """

    def __init__(self, lag_hours: int = 168, frequency: str = "1h") -> None:
        if lag_hours < 1:
            raise ValueError("lag_hours must be >= 1")
        self.lag_hours = int(lag_hours)
        self.frequency = str(frequency)

    def fit(self, X=None, y=None):
        """No-op fit for API compatibility."""
        return self

    def _lag_steps(self) -> int:
        freq_td = pd.to_timedelta(self.frequency)
        hours_per_row = float(freq_td.total_seconds()) / 3600.0
        if hours_per_row <= 0.0:
            raise ValueError("frequency must be positive")
        return int(np.ceil(self.lag_hours / hours_per_row))

    def predict_from_series(self, target_series: pd.Series) -> pd.Series:
        """Create persistence predictions from a raw target series."""
        if not isinstance(target_series, pd.Series):
            target_series = pd.Series(target_series)
        return pd.to_numeric(target_series, errors="coerce").shift(self._lag_steps())

    def add_baseline_column(
        self,
        df: pd.DataFrame,
        target_col: str,
        prediction_col: str = "baseline_prediction",
    ) -> pd.DataFrame:
        """Return copy with persistence baseline column added."""
        if target_col not in df.columns:
            raise KeyError(f"target_col '{target_col}' not found")
        out = df.copy()
        out[prediction_col] = self.predict_from_series(out[target_col])
        return out

    def predict(self, X):
        """scikit-learn-compatible API for direct series-based usage.

        Notes
        -----
        `X` is expected to be a 1D target-history-like array/series.
        """
        pred = self.predict_from_series(pd.Series(X))
        return pred.to_numpy(dtype=float)
