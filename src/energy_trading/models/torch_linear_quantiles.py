"""PyTorch drop-in linear multi-quantile regressor with sklearn-style API."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "TorchMultiQuantileLinearRegressor requires PyTorch. Install torch first."
    ) from exc


class TorchMultiQuantileLinearRegressor(BaseEstimator, RegressorMixin):
    """Strictly linear multi-output quantile regressor implemented in PyTorch.

    Parameters
    ----------
    quantiles:
        Iterable of quantile levels in (0, 1), e.g. [0.1, 0.5, 0.9].
    learning_rate:
        Adam learning rate.
    epochs:
        Number of full training epochs.
    batch_size:
        Optional mini-batch size. If None, full-batch training is used.
    """

    def __init__(
        self,
        quantiles: Iterable[float],
        learning_rate: float = 0.01,
        epochs: int = 1000,
        batch_size: int | None = None,
    ) -> None:
        self.quantiles = list(quantiles)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = None if batch_size is None else int(batch_size)

        # Fitted attributes
        self.device_: torch.device | None = None
        self.model_: nn.Linear | None = None
        self.quantiles_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def _validate_and_prepare(self, X, y=None):
        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X_np.shape}")
        if y is None:
            return X_np, None
        y_np = np.asarray(y, dtype=np.float32).reshape(-1)
        if X_np.shape[0] != y_np.shape[0]:
            raise ValueError(
                f"X/y length mismatch: X={X_np.shape[0]} y={y_np.shape[0]}"
            )
        return X_np, y_np

    @staticmethod
    def _pinball_loss(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        quantiles: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # y_true: [B], y_pred: [B, Q], quantiles: [Q]
        err = y_true.unsqueeze(1) - y_pred
        loss = torch.maximum(quantiles * err, (quantiles - 1.0) * err)
        if sample_weights is not None:
            # True weighted mean over all quantile/sample terms.
            w = sample_weights.unsqueeze(1).expand_as(loss)
            denom = w.sum().clamp_min(1e-12)
            return (loss * w).sum() / denom
        return loss.mean()

    def fit(self, X, y, sample_weights=None):
        X_np, y_np = self._validate_and_prepare(X, y)

        q = np.asarray(self.quantiles, dtype=np.float32)
        if q.ndim != 1 or q.size == 0:
            raise ValueError("quantiles must be a non-empty 1D iterable.")
        if np.any((q <= 0.0) | (q >= 1.0)):
            raise ValueError("All quantiles must be strictly between 0 and 1.")

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantiles_ = q
        self.n_features_in_ = int(X_np.shape[1])

        self.model_ = nn.Linear(self.n_features_in_, int(q.size)).to(self.device_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        X_t = torch.from_numpy(X_np).to(self.device_)
        y_t = torch.from_numpy(y_np).to(self.device_)
        q_t = torch.from_numpy(q).to(self.device_)
        w_t = None
        if sample_weights is not None:
            w_np = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
            if w_np.shape[0] != X_np.shape[0]:
                raise ValueError(
                    f"sample_weights length mismatch: got {w_np.shape[0]} expected {X_np.shape[0]}"
                )
            finite_w = w_np[np.isfinite(w_np)]
            if finite_w.size == 0:
                raise ValueError("sample_weights contains no finite values.")
            print(
                "[WEIGHT_AUDIT][TorchLinear] "
                f"n={w_np.shape[0]} min={float(np.min(finite_w)):.6f} "
                f"p50={float(np.percentile(finite_w, 50.0)):.6f} "
                f"p95={float(np.percentile(finite_w, 95.0)):.6f} "
                f"mean={float(np.mean(finite_w)):.6f} max={float(np.max(finite_w)):.6f}"
            )
            w_t = torch.from_numpy(w_np).to(self.device_)

        n = X_t.shape[0]
        if self.batch_size is None or self.batch_size <= 0 or self.batch_size >= n:
            batch_size = n
        else:
            batch_size = int(self.batch_size)

        self.model_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device_)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                xb = X_t[idx]
                yb = y_t[idx]
                wb = w_t[idx] if w_t is not None else None
                optimizer.zero_grad(set_to_none=True)
                pred = self.model_(xb)
                loss = self._pinball_loss(yb, pred, q_t, sample_weights=wb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        if self.model_ is None or self.device_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_np, _ = self._validate_and_prepare(X, y=None)
        X_t = torch.from_numpy(X_np).to(self.device_)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(X_t).detach().cpu().numpy()
        return pred

    def get_coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        """Return linear coefficients for downstream MILP use.

        Returns
        -------
        weights, bias
            weights shape: [n_quantiles, n_features]
            bias shape: [n_quantiles]
        """
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        w = self.model_.weight.detach().cpu().numpy().copy()
        b = self.model_.bias.detach().cpu().numpy().copy()
        return w, b
