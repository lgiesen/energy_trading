"""Cross-validation utilities for causally safe time-series model evaluation.

This module implements a purged expanding-window splitter with a strict
causality gap between train and validation segments. The gap prevents target
leakage from delayed publications and horizon overlap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


@dataclass(frozen=True)
class CVFold:
    """Container describing one fold with explicit purge interval."""

    fold: int
    train_idx: np.ndarray
    purge_idx: np.ndarray
    val_idx: np.ndarray


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """Expanding-window splitter with a strict hour-based purge gap.

    Leakage prevention:
    - Validation always starts after `gap_hours / frequency` rows from the end
      of the training fold.
    - No shuffling is used; chronological order is preserved.

    Parameters
    ----------
    n_splits:
        Number of validation folds.
    test_size:
        Number of rows in each validation fold. If None, it is inferred from the
        available sample size.
    gap_hours:
        Purge gap in hours between train end and validation start.
    frequency:
        Sampling frequency of rows. Default is hourly ("1h").
    min_train_size:
        Optional minimum training rows required in each fold.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int | None = None,
        gap_hours: int = 72,
        frequency: str = "1h",
        min_train_size: int | None = None,
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if test_size is not None and test_size < 1:
            raise ValueError("test_size must be >= 1 when provided")
        if gap_hours < 0:
            raise ValueError("gap_hours must be >= 0")
        if min_train_size is not None and min_train_size < 1:
            raise ValueError("min_train_size must be >= 1 when provided")

        self.n_splits = int(n_splits)
        self.test_size = test_size
        self.gap_hours = int(gap_hours)
        self.frequency = str(frequency)
        self.min_train_size = min_train_size

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def _gap_rows_from_frequency(self) -> int:
        freq_td = pd.to_timedelta(self.frequency)
        hours_per_row = float(freq_td.total_seconds()) / 3600.0
        if hours_per_row <= 0.0:
            raise ValueError("frequency must be positive")
        gap_rows = int(np.ceil(self.gap_hours / hours_per_row))
        return max(0, gap_rows)

    def _resolve_test_size(self, n_samples: int, gap_rows: int) -> int:
        if self.test_size is not None:
            return int(self.test_size)
        denom = self.n_splits + 1
        inferred = max(1, (n_samples - gap_rows) // denom)
        return inferred

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        if n_samples < 2:
            raise ValueError("Need at least 2 rows for splitting")

        gap_rows = self._gap_rows_from_frequency()
        test_size = self._resolve_test_size(n_samples=n_samples, gap_rows=gap_rows)

        required = self.n_splits * test_size + gap_rows + 1
        if n_samples < required:
            raise ValueError(
                f"Not enough rows. required>={required}, available={n_samples}, "
                f"n_splits={self.n_splits}, test_size={test_size}, gap_rows={gap_rows}"
            )

        starts = [n_samples - (self.n_splits - k) * test_size for k in range(self.n_splits)]
        for val_start in starts:
            val_end = min(val_start + test_size, n_samples)
            train_end = val_start - gap_rows
            if train_end <= 0:
                continue
            if self.min_train_size is not None and train_end < self.min_train_size:
                continue

            train_idx = np.arange(0, train_end, dtype=int)
            val_idx = np.arange(val_start, val_end, dtype=int)
            if len(val_idx) == 0:
                continue
            yield train_idx, val_idx

    def split_with_metadata(self, X, y=None, groups=None):
        """Yield fold metadata including explicit purge indices."""
        for fold, (train_idx, val_idx) in enumerate(self.split(X, y=y, groups=groups), start=1):
            purge_start = int(train_idx[-1] + 1) if len(train_idx) else 0
            purge_end = int(val_idx[0])
            purge_idx = np.arange(purge_start, purge_end, dtype=int)
            yield CVFold(fold=fold, train_idx=train_idx, purge_idx=purge_idx, val_idx=val_idx)
