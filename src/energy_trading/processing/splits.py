"""Time-series CV utilities with purge/embargo for sequence targets.

This module implements an expanding-window splitter with a mandatory gap between
training and validation to prevent look-ahead leakage for multi-horizon targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldIndices:
    """Container for one purged fold."""

    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    purge_idx: np.ndarray


class PurgedTimeSeriesSplit:
    """Expanding-window splitter with fixed validation horizon and embargo gap.

    Parameters
    ----------
    n_splits:
        Number of folds.
    min_train_days:
        Minimum initial training window in days.
    test_days:
        Validation window length in days for each fold.
    gap_hours:
        Embargo gap between train end and validation start in hours.
    """

    def __init__(
        self,
        n_splits: int,
        min_train_days: int,
        test_days: int,
        gap_hours: int = 72,
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if min_train_days < 1:
            raise ValueError("min_train_days must be >= 1")
        if test_days < 1:
            raise ValueError("test_days must be >= 1")
        if gap_hours < 0:
            raise ValueError("gap_hours must be >= 0")
        self.n_splits = int(n_splits)
        self.min_train_days = int(min_train_days)
        self.test_days = int(test_days)
        self.gap_hours = int(gap_hours)

    @property
    def min_train_hours(self) -> int:
        return self.min_train_days * 24

    @property
    def test_hours(self) -> int:
        return self.test_days * 24

    def get_n_splits(self) -> int:
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame | pd.Series | np.ndarray,
        timestamps: Iterable[pd.Timestamp] | pd.Series | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yield `(train_idx, val_idx)` pairs.

        The splitter assumes hourly rows in chronological order. If `timestamps`
        are provided, monotonicity is validated.
        """
        n_samples = len(X)
        if n_samples <= 0:
            raise ValueError("X is empty")

        if timestamps is not None:
            ts = pd.to_datetime(pd.Series(list(timestamps)), utc=True, errors="coerce")
            if ts.isna().any():
                raise ValueError("timestamps contain invalid values")
            if not ts.is_monotonic_increasing:
                raise ValueError("timestamps must be sorted ascending")
            if len(ts) != n_samples:
                raise ValueError("timestamps length must match X length")

        required = self.min_train_hours + self.gap_hours + self.test_hours
        if n_samples < required:
            raise ValueError(
                f"Not enough rows for one fold. required={required}, available={n_samples}"
            )

        max_start = n_samples - self.test_hours
        first_start = self.min_train_hours + self.gap_hours
        starts = np.linspace(first_start, max_start, num=self.n_splits, dtype=int)
        starts = np.unique(starts)
        if len(starts) < self.n_splits:
            raise ValueError(
                "Cannot create requested number of unique folds with current settings."
            )

        for val_start in starts:
            val_end = val_start + self.test_hours
            if val_end > n_samples:
                continue
            train_end = val_start - self.gap_hours
            if train_end < self.min_train_hours:
                continue
            train_idx = np.arange(0, train_end, dtype=int)
            val_idx = np.arange(val_start, val_end, dtype=int)
            yield train_idx, val_idx

    def split_with_metadata(
        self,
        X: pd.DataFrame | pd.Series | np.ndarray,
        timestamps: Iterable[pd.Timestamp] | pd.Series | None = None,
    ) -> Generator[FoldIndices, None, None]:
        """Yield fold indices including explicit purge segment."""
        for k, (train_idx, val_idx) in enumerate(self.split(X, timestamps=timestamps), start=1):
            purge_start = train_idx[-1] + 1 if len(train_idx) else 0
            purge_end = val_idx[0]
            purge_idx = np.arange(purge_start, purge_end, dtype=int)
            yield FoldIndices(
                fold=k,
                train_idx=train_idx,
                val_idx=val_idx,
                purge_idx=purge_idx,
            )


def prepare_dual_targets(
    df: pd.DataFrame,
    *,
    target_train_cols: list[str],
    target_true_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract aligned train/eval target matrices from one frame.

    Use `y_train` for optimization stability and `y_true` for unbiased evaluation.
    """
    missing_train = [c for c in target_train_cols if c not in df.columns]
    missing_true = [c for c in target_true_cols if c not in df.columns]
    if missing_train:
        raise KeyError(f"Missing train target columns: {missing_train}")
    if missing_true:
        raise KeyError(f"Missing true target columns: {missing_true}")
    return df[target_train_cols].copy(), df[target_true_cols].copy()

