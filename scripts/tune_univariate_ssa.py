from __future__ import annotations

import os

# Optimization 1: prevent BLAS/OpenMP thread oversubscription before heavy imports.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover
    Parallel = None  # type: ignore[assignment]
    delayed = None  # type: ignore[assignment]

# Ensure project root import path when executed as `python scripts/...`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.energy_trading.models.univariate_ssa import (
    SSAGridResult,
    UnivariateSSA,
    _diag_averaging,
    _lrf_coefficients_from_U,
    _reconstruct_with_rank,
    _to_1d_numpy,
    _trajectory_matrix,
    _truncated_svd,
)


MetricName = str


@dataclass
class HourlySSAResult:
    hour: int
    best: SSAGridResult
    val_score: float
    n_train: int
    n_val: int


def _metric(y_true: np.ndarray, y_pred: np.ndarray, metric: MetricName) -> float:
    err = y_true - y_pred
    if metric == "mae":
        return float(np.mean(np.abs(err)))
    if metric == "rmse":
        return float(np.sqrt(np.mean(err * err)))
    raise ValueError(f"Unsupported metric: {metric}")


def _one_step_from_svd(U: np.ndarray, s: np.ndarray, Vt: np.ndarray, rank: int, scale: float, window_length: int) -> float:
    """Compute recurrent one-step forecast from cached decomposition by slicing first `rank` components."""
    Xr = _reconstruct_with_rank(U, s, Vt, rank=rank, scale=scale)
    y_tilde = _diag_averaging(Xr)
    R, _ = _lrf_coefficients_from_U(U, rank=rank)
    tail = y_tilde[-(window_length - 1) :][::-1]
    return float(np.dot(R, tail))


def _cached_grid_score_one_hour(
    train: np.ndarray,
    val: np.ndarray,
    *,
    L_values: Iterable[int],
    r_values: Iterable[int],
    metric: MetricName,
) -> tuple[SSAGridResult, pd.DataFrame, np.ndarray]:
    """Optimization 2: cache one SVD per (rolling-step, L), reuse for all r."""
    train_arr = _to_1d_numpy(train)
    val_arr = _to_1d_numpy(val)
    if val_arr.size == 0:
        raise ValueError("Validation set is empty.")

    L_values = sorted({int(L) for L in L_values})
    r_values = sorted({int(r) for r in r_values})
    r_min, r_max = min(r_values), max(r_values)

    # Per-(L,r) prediction buffer
    preds: dict[tuple[int, int], np.ndarray] = {}
    for L in L_values:
        for r in r_values:
            preds[(L, r)] = np.full(val_arr.size, np.nan, dtype=float)

    hist = train_arr.copy()
    for i, y_true in enumerate(val_arr):
        # For this rolling step, evaluate each L once, SVD once.
        for L in L_values:
            N = hist.size
            if N <= L:
                continue
            max_rank = min(L, N - L + 1)
            if r_min > max_rank:
                continue
            k = min(r_max, max_rank)

            Y = _trajectory_matrix(hist, L)
            scale = float(np.max(np.abs(Y)))
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0
            Y_scaled = Y / scale

            U, s, Vt = _truncated_svd(Y_scaled, k=k)
            # reuse cached SVD for all r
            for r in r_values:
                if r > k:
                    continue
                try:
                    y_hat = _one_step_from_svd(U, s, Vt, rank=r, scale=scale, window_length=L)
                    preds[(L, r)][i] = y_hat
                except Exception:
                    # keep NaN for invalid combination at this step
                    pass

        # rolling update uses true observed validation value
        hist = np.append(hist, y_true)

    rows: list[dict[str, float | int]] = []
    for L in L_values:
        for r in r_values:
            p = preds[(L, r)]
            mask = np.isfinite(p)
            if not np.any(mask):
                continue
            score = _metric(val_arr[mask], p[mask], metric)
            rows.append({"window_length": L, "rank": r, "score": score, "n_scored": int(mask.sum())})

    if not rows:
        raise RuntimeError("No valid (L, r) evaluated for this hour.")

    table = pd.DataFrame(rows).sort_values("score", ascending=True).reset_index(drop=True)
    best = table.iloc[0]
    return (
        SSAGridResult(int(best["window_length"]), int(best["rank"]), float(best["score"])),
        table,
        val_arr,
    )


def _load_univariate_frame(path: str, target_col: str) -> pd.DataFrame:
    """Strictly isolate [timestamp_utc, target_col], enforce hourly continuity, fill NaNs."""
    df = pd.read_parquet(path)
    required = ["timestamp_utc", target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")

    x = df.loc[:, required].copy()
    x["timestamp_utc"] = pd.to_datetime(x["timestamp_utc"], utc=True, errors="coerce")
    x[target_col] = pd.to_numeric(x[target_col], errors="coerce")
    x = x.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    x = x.drop_duplicates(subset=["timestamp_utc"], keep="last")
    x = x.set_index("timestamp_utc")

    full_idx = pd.date_range(x.index.min(), x.index.max(), freq="h", tz="UTC")
    x = x.reindex(full_idx)
    x[target_col] = x[target_col].interpolate(method="time", limit_direction="both").ffill().bfill()
    if x[target_col].isna().any():
        raise ValueError(f"Unable to fill all NaNs for {target_col} in {path}.")

    x = x.reset_index().rename(columns={"index": "timestamp_utc"})
    return x[["timestamp_utc", target_col]]


def _split_hourly(series_df: pd.DataFrame, target_col: str) -> dict[int, pd.Series]:
    out: dict[int, pd.Series] = {}
    ts = pd.to_datetime(series_df["timestamp_utc"], utc=True)
    for h in range(24):
        mask = ts.dt.hour == h
        out[h] = pd.Series(series_df.loc[mask, target_col].to_numpy(dtype=float)).reset_index(drop=True)
    return out


def _train_hour(
    h: int,
    tr: pd.Series,
    va: pd.Series,
    *,
    L_values: Iterable[int],
    r_values: Iterable[int],
    metric: MetricName,
    min_points_per_hour: int,
) -> HourlySSAResult | None:
    if tr.size < min_points_per_hour or va.size == 0:
        return None
    best, _, _ = _cached_grid_score_one_hour(
        tr.to_numpy(dtype=float),
        va.to_numpy(dtype=float),
        L_values=L_values,
        r_values=r_values,
        metric=metric,
    )
    return HourlySSAResult(
        hour=h,
        best=best,
        val_score=float(best.score),
        n_train=int(tr.size),
        n_val=int(va.size),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimized hourly univariate recurrent SSA tuner.")
    p.add_argument("--train-path", default="data/model_input/da/train.parquet")
    p.add_argument("--val-path", default="data/model_input/da/val.parquet")
    p.add_argument("--target-col", default="target_da_price")
    p.add_argument("--metric", choices=["rmse", "mae"], default="rmse")
    p.add_argument("--horizon-demo", type=int, default=48)
    p.add_argument("--min-points-per-hour", type=int, default=220)
    p.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers across 24 hourly models.")
    p.add_argument(
        "--smoke-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use tiny search (L=25..26, r=3..4) and last 48 validation rows.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Search space
    if args.smoke_test:
        L_values = range(25, 27)   # 25,26
        r_values = range(3, 5)     # 3,4
    else:
        L_values = range(25, 65)   # 24 < L < 65
        r_values = range(3, 15)    # 3 <= r <= 14

    train_df = _load_univariate_frame(args.train_path, args.target_col)
    val_df = _load_univariate_frame(args.val_path, args.target_col)

    # Additional smoke-test acceleration: last 48 hours only.
    if args.smoke_test:
        val_df = val_df.tail(48).reset_index(drop=True)

    train_hourly = _split_hourly(train_df, args.target_col)
    val_hourly = _split_hourly(val_df, args.target_col)

    # Optimization 3: parallelize by independent hourly sub-series.
    if Parallel is None or delayed is None:
        print("[WARN] joblib not installed. Falling back to sequential hourly tuning.")
        raw_results = [
            _train_hour(
                h,
                train_hourly[h],
                val_hourly[h],
                L_values=L_values,
                r_values=r_values,
                metric=args.metric,
                min_points_per_hour=args.min_points_per_hour,
            )
            for h in range(24)
        ]
    else:
        jobs = [
            delayed(_train_hour)(
                h,
                train_hourly[h],
                val_hourly[h],
                L_values=L_values,
                r_values=r_values,
                metric=args.metric,
                min_points_per_hour=args.min_points_per_hour,
            )
            for h in range(24)
        ]
        raw_results = Parallel(n_jobs=args.n_jobs, prefer="processes")(jobs)
    results = [r for r in raw_results if r is not None]

    if not results:
        raise RuntimeError("No hourly models were trained. Check min-points-per-hour / data.")

    weighted_score = float(
        np.average(np.array([r.val_score for r in results]), weights=np.array([r.n_val for r in results]))
    )

    out = pd.DataFrame(
        {
            "hour": [r.hour for r in results],
            "window_length": [r.best.window_length for r in results],
            "rank": [r.best.rank for r in results],
            "val_score": [r.val_score for r in results],
            "n_train": [r.n_train for r in results],
            "n_val": [r.n_val for r in results],
        }
    ).sort_values("hour")

    print("=== Hourly-Specified SSA (Optimized) ===")
    print(f"trained_hours={len(results)}/24")
    print(f"weighted_validation_{args.metric}={weighted_score:.6f}")
    print(out.to_string(index=False))

    # quick recursive forecast demo from one hour-model
    h0 = next((r for r in results if r.hour == 0), results[0])
    model = UnivariateSSA(window_length=h0.best.window_length, rank=h0.best.rank).fit(train_hourly[h0.hour])
    fcst = model.forecast(horizon=int(args.horizon_demo), refit_each_step=True)
    print(f"\nDemo: hour={h0.hour:02d} {args.horizon_demo}-step recursive forecast first10:")
    print(np.array2string(fcst[:10], precision=4, separator=", "))


if __name__ == "__main__":
    main()
