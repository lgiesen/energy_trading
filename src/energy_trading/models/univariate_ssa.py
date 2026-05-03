from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Union

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

try:
    from sklearn.base import BaseEstimator, RegressorMixin
except Exception:  # pragma: no cover
    class BaseEstimator:  # type: ignore[override]
        """Fallback sklearn-like base class when scikit-learn is unavailable."""

    class RegressorMixin:  # type: ignore[override]
        """Fallback sklearn-like mixin when scikit-learn is unavailable."""

ArrayLike1D = Union[np.ndarray, pd.Series, pd.DataFrame, list[float]]
MetricName = Literal["rmse", "mae"]


def _to_1d_numpy(x: ArrayLike1D, column: str | None = None) -> np.ndarray:
    """Convert supported inputs to a float 1D NumPy array."""
    if isinstance(x, pd.DataFrame):
        if column is not None:
            arr = x[column].to_numpy(dtype=float)
        elif x.shape[1] == 1:
            arr = x.iloc[:, 0].to_numpy(dtype=float)
        else:
            raise ValueError("DataFrame input must have one column or specify `column`.")
    elif isinstance(x, pd.Series):
        arr = x.to_numpy(dtype=float)
    else:
        arr = np.asarray(x, dtype=float)
    arr = np.ravel(arr)
    if arr.ndim != 1:
        raise ValueError("Input must be one-dimensional after conversion.")
    if arr.size == 0:
        raise ValueError("Input series is empty.")
    if not np.isfinite(arr).all():
        raise ValueError("Input series contains NaN/Inf.")
    return arr


def _trajectory_matrix(y: np.ndarray, L: int) -> np.ndarray:
    """Build L x K Hankel trajectory matrix for univariate series y of length N."""
    N = y.size
    if not (2 <= L < N):
        raise ValueError(f"window_length L must satisfy 2 <= L < N. Got L={L}, N={N}.")
    K = N - L + 1
    # Y[:, j] = y[j : j+L]
    return np.column_stack([y[j : j + L] for j in range(K)])


def _diag_averaging(X: np.ndarray) -> np.ndarray:
    """Hankelize matrix X (L x K) via diagonal averaging to length N=L+K-1."""
    L, K = X.shape
    N = L + K - 1
    out = np.zeros(N, dtype=float)
    cnt = np.zeros(N, dtype=float)
    # anti-diagonal index i+j
    for i in range(L):
        for j in range(K):
            idx = i + j
            out[idx] += X[i, j]
            cnt[idx] += 1.0
    return out / cnt


def _reconstruct_with_rank(U: np.ndarray, s: np.ndarray, Vt: np.ndarray, rank: int, scale: float = 1.0) -> np.ndarray:
    """Reconstruct grouped trajectory matrix from first `rank` SVD components."""
    if rank < 1 or rank > s.size:
        raise ValueError(f"rank must be in [1, {s.size}], got {rank}.")
    Xr_scaled = (U[:, :rank] * s[:rank]) @ Vt[:rank, :]
    return Xr_scaled * float(scale)


def _lrf_coefficients_from_U(U: np.ndarray, rank: int, eps: float = 1e-12) -> tuple[np.ndarray, float]:
    """Compute recurrent SSA LRF coefficients R and v^2 from first `rank` columns of U."""
    if rank < 1 or rank > U.shape[1]:
        raise ValueError(f"rank must be in [1, {U.shape[1]}], got {rank}.")
    Ur = U[:, :rank]
    pi = Ur[-1, :]
    v2 = float(np.sum(pi * pi))
    denom = 1.0 - v2
    if abs(denom) <= eps:
        raise FloatingPointError("Numerically unstable LRF denominator (1-v^2) is ~0.")
    U_nabla = Ur[:-1, :]
    R = (U_nabla @ pi) / denom
    return R, v2


def _truncated_svd(Y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute top-k SVD components of dense Y using sparse solver when possible."""
    L, K = Y.shape
    max_rank = min(L, K)
    if not (1 <= k <= max_rank):
        raise ValueError(f"k must be in [1, {max_rank}], got {k}.")

    # svds requires k < min(L,K). Fallback to dense SVD when k == min rank.
    if k < max_rank:
        try:
            U, s, Vt = svds(Y, k=k, which="LM")
            # svds returns ascending singular values; reorder descending.
            order = np.argsort(s)[::-1]
            U, s, Vt = U[:, order], s[order], Vt[order, :]
            if np.isfinite(U).all() and np.isfinite(s).all() and np.isfinite(Vt).all():
                return U, s, Vt
        except Exception:
            pass

    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    return U[:, :k], s[:k], Vt[:k, :]


class UnivariateSSA(BaseEstimator, RegressorMixin):
    """Univariate SSA with recurrent (LRF) forecasting.

    Implements:
    1) Embedding (trajectory matrix)
    2) Truncated SVD
    3) Grouping with first r components
    4) Diagonal averaging reconstruction
    5) Recurrent 1-step forecast with LRF coefficients
    """

    def __init__(self, window_length: int = 48, rank: int = 8, eps: float = 1e-12) -> None:
        self.window_length = int(window_length)
        self.rank = int(rank)
        self.eps = float(eps)

    def fit(self, y: ArrayLike1D, column: str | None = None):
        y_arr = _to_1d_numpy(y, column=column)
        N = y_arr.size
        L = self.window_length
        if not (2 <= L < N):
            raise ValueError(f"window_length must satisfy 2 <= L < N. Got L={L}, N={N}.")
        K = N - L + 1
        max_rank = min(L, K)
        r = self.rank
        if not (1 <= r <= max_rank):
            raise ValueError(f"rank must satisfy 1 <= rank <= {max_rank}. Got rank={r}.")

        Y = _trajectory_matrix(y_arr, L)
        scale = float(np.max(np.abs(Y)))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        Y_scaled = Y / scale

        U, s, Vt = _truncated_svd(Y_scaled, k=r)

        # Grouping + reconstruction with first r components.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            Xr_scaled = (U * s) @ Vt
        if not np.isfinite(Xr_scaled).all():
            # Robust fallback in pathological svds cases.
            U, s, Vt = np.linalg.svd(Y_scaled, full_matrices=False)
            U, s, Vt = U[:, :r], s[:r], Vt[:r, :]
            Xr_scaled = (U * s) @ Vt
        Xr = Xr_scaled * scale
        y_tilde = _diag_averaging(Xr)

        # LRF coefficients from selected eigenvectors U_j.
        pi = U[-1, :]  # last components pi_j
        v2 = float(np.sum(pi * pi))
        denom = 1.0 - v2
        if abs(denom) <= self.eps:
            raise FloatingPointError(
                "Numerically unstable LRF denominator (1-v^2) is ~0; try smaller rank/window_length."
            )
        U_nabla = U[:-1, :]  # first L-1 components
        R = (U_nabla @ pi) / denom  # shape (L-1,)

        self.y_ = y_arr
        self.y_tilde_ = y_tilde
        self.U_ = U
        self.s_ = s
        self.Vt_ = Vt
        self.R_ = R
        self.v2_ = v2
        self.n_features_in_ = 1
        return self

    def forecast_one_step(self) -> float:
        """Forecast y_{N+1} from fitted reconstruction using recurrent LRF."""
        if not hasattr(self, "R_"):
            raise RuntimeError("Model is not fitted.")
        L = self.window_length
        if self.y_tilde_.size < L:
            raise RuntimeError("Insufficient reconstructed history for one-step forecast.")
        # a1*y_N + a2*y_{N-1} + ... + a_{L-1}*y_{N-L+2}
        tail = self.y_tilde_[-(L - 1) :][::-1]
        return float(np.dot(self.R_, tail))

    def predict(self, X=None) -> np.ndarray:
        """sklearn-style predict: returns one-step forecast as array([y_hat])."""
        return np.asarray([self.forecast_one_step()], dtype=float)

    def forecast(self, horizon: int, refit_each_step: bool = True) -> np.ndarray:
        """Recursive multi-step forecast.

        If refit_each_step=True, each step refits SSA on history + prior forecasts,
        re-estimating decomposition and LRF each step (strict recurrent SSA recursion).
        """
        if not hasattr(self, "y_"):
            raise RuntimeError("Model is not fitted.")
        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        hist = self.y_.copy()
        preds = np.empty(horizon, dtype=float)
        if not refit_each_step:
            for h in range(horizon):
                y_next = self.forecast_one_step() if h == 0 else float(np.dot(self.R_, np.r_[preds[h - 1 :: -1], self.y_tilde_][:(self.window_length - 1)]))
                preds[h] = y_next
            return preds

        for h in range(horizon):
            self.fit(hist)
            y_next = self.forecast_one_step()
            preds[h] = y_next
            hist = np.append(hist, y_next)
        # restore original fit for estimator consistency
        self.fit(self.y_)
        return preds


@dataclass
class SSAGridResult:
    window_length: int
    rank: int
    score: float


def _metric(y_true: np.ndarray, y_pred: np.ndarray, metric: MetricName) -> float:
    err = y_true - y_pred
    if metric == "mae":
        return float(np.mean(np.abs(err)))
    if metric == "rmse":
        return float(np.sqrt(np.mean(err * err)))
    raise ValueError(f"Unsupported metric: {metric}")


def rolling_one_step_backtest(
    train: ArrayLike1D,
    val: ArrayLike1D,
    *,
    window_length: int,
    rank: int,
    metric: MetricName = "rmse",
    refit_each_step: bool = True,
) -> tuple[float, np.ndarray]:
    """Evaluate one-step-ahead forecasts over validation with rolling history.

    At each validation step t:
      - fit on train + val[:t] (true observed history)
      - forecast next one-step
    """
    train_arr = _to_1d_numpy(train)
    val_arr = _to_1d_numpy(val)
    if val_arr.size == 0:
        raise ValueError("Validation set is empty.")

    model = UnivariateSSA(window_length=window_length, rank=rank)
    preds = np.empty_like(val_arr, dtype=float)

    if refit_each_step:
        hist = train_arr.copy()
        for i, y_true in enumerate(val_arr):
            model.fit(hist)
            preds[i] = model.forecast_one_step()
            hist = np.append(hist, y_true)
    else:
        model.fit(train_arr)
        preds[:] = model.forecast(horizon=val_arr.size, refit_each_step=True)

    return _metric(val_arr, preds, metric), preds


def grid_search_ssa(
    train: ArrayLike1D,
    val: ArrayLike1D,
    *,
    L_values: Iterable[int] = range(25, 65),
    r_values: Iterable[int] = range(3, 15),
    metric: MetricName = "rmse",
    refit_each_step: bool = True,
    verbose: bool = True,
) -> tuple[SSAGridResult, pd.DataFrame]:
    """Grid search over (L, r) minimizing one-step-ahead validation error."""
    train_arr = _to_1d_numpy(train)
    val_arr = _to_1d_numpy(val)
    rows: list[dict[str, float | int]] = []

    for L in L_values:
        # Minimum history needed for fit after split.
        if train_arr.size <= L:
            continue
        for r in r_values:
            # rank bound depends on N and L at fit time.
            max_rank = min(L, train_arr.size - L + 1)
            if r > max_rank:
                continue
            try:
                score, _ = rolling_one_step_backtest(
                    train_arr,
                    val_arr,
                    window_length=int(L),
                    rank=int(r),
                    metric=metric,
                    refit_each_step=refit_each_step,
                )
                rows.append({"window_length": int(L), "rank": int(r), "score": float(score)})
                if verbose:
                    print(f"L={L:2d} r={r:2d} {metric}={score:.6f}")
            except Exception as ex:
                if verbose:
                    print(f"L={L:2d} r={r:2d} -> failed: {ex}")

    if not rows:
        raise RuntimeError("No valid (L, r) combination evaluated.")

    df = pd.DataFrame(rows).sort_values("score", ascending=True).reset_index(drop=True)
    best = df.iloc[0]
    return SSAGridResult(int(best["window_length"]), int(best["rank"]), float(best["score"])), df


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N = 420
    t = np.arange(N)
    y = (
        40.0
        + 8.0 * np.sin(2 * np.pi * t / 24.0)
        + 2.0 * np.sin(2 * np.pi * t / (24.0 * 7.0))
        + rng.normal(0.0, 1.2, size=N)
    )

    train = y[:320]
    val = y[320:380]

    best, table = grid_search_ssa(
        train,
        val,
        L_values=range(25, 65),
        r_values=range(3, 15),
        metric="rmse",
        refit_each_step=True,
        verbose=False,
    )
    print("Best params:", best)
    print(table.head(5).to_string(index=False))

    model = UnivariateSSA(window_length=best.window_length, rank=best.rank).fit(train)
    fcst_24 = model.forecast(horizon=24, refit_each_step=True)
    print("24-step forecast (first 10):", np.round(fcst_24[:10], 4))
