"""Model training module."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

# TODO: Keep in mind: predict the downward and upward activation probabilities separately!


def train_with_purged_cv(
    *,
    splitter,
    model,
    metric: Callable[[np.ndarray, np.ndarray], float],
    X: pd.DataFrame,
    y_train: pd.DataFrame | pd.Series,
    y_true: pd.DataFrame | pd.Series,
    timestamps: pd.Series | None = None,
) -> list[float]:
    """Train with purged walk-forward splits and evaluate on unclipped truth.

    The model is fit on `y_train` (for stable optimization) and evaluated
    against `y_true` (market-realistic target).
    """
    if timestamps is None:
        if "timestamp_utc" in X.columns:
            timestamps = pd.to_datetime(X["timestamp_utc"], utc=True, errors="coerce")
        else:
            timestamps = pd.Series(pd.RangeIndex(len(X)))

    fold_scores: list[float] = []
    # Training pattern
    for train_idx, val_idx in splitter.split(X, timestamps=timestamps):
        model.fit(X.iloc[train_idx], y_train.iloc[train_idx])
        y_pred = model.predict(X.iloc[val_idx])
        score = metric(y_pred, y_true.iloc[val_idx])
        fold_scores.append(float(score))
    return fold_scores


def train() -> None:
    """Entrypoint placeholder for project-specific training wiring."""
    raise NotImplementedError(
        "Wire `train_with_purged_cv(...)` with your dataset/model configuration."
    )


if __name__ == "__main__":
    train()
