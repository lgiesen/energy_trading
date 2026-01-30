"""Helpers for chronological train/val/test splits."""
from __future__ import annotations

from typing import Tuple

import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    time_col: str = "timestamp_utc",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train/val/test by time order.

    The dataframe is sorted by `time_col` and split by row counts. Use a gap
    between splits in your pipeline if you have rolling-window leakage concerns.
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")
    if time_col not in df.columns:
        raise KeyError(f"Missing time column: {time_col}")

    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train = df_sorted.iloc[:train_end].copy()
    val = df_sorted.iloc[train_end:val_end].copy()
    test = df_sorted.iloc[val_end:].copy()
    return train, val, test


def chronological_split_by_time(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    time_col: str = "timestamp_utc",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train/val/test by explicit cutoff timestamps."""
    if time_col not in df.columns:
        raise KeyError(f"Missing time column: {time_col}")

    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    train = df_sorted[df_sorted[time_col] <= train_end].copy()
    val = df_sorted[(df_sorted[time_col] > train_end) & (df_sorted[time_col] <= val_end)].copy()
    test = df_sorted[df_sorted[time_col] > val_end].copy()
    return train, val, test
