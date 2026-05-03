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
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from datetime import datetime, timezone

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
from src.energy_trading.evaluation.metrics import compute_forecast_metrics

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]


MetricName = str


@dataclass
class HourlySSAResult:
    hour: int
    best: SSAGridResult
    val_score: float
    n_train: int
    n_val: int
    metrics: dict[str, object] | None = None


AFRR_TARGETS = [
    "target_afrr_activation_price_vwap_pos",
    "target_afrr_activation_price_vwap_neg",
    "target_afrr_activation_rate_pos",
    "target_afrr_activation_rate_neg",
    "target_afrr_capacity_price_pos",
    "target_afrr_capacity_price_neg",
]


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


def _directional_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """MAE on first differences (directional movement magnitude error)."""
    if y_true.size < 2 or y_pred.size < 2:
        return None
    dy_true = np.diff(y_true)
    dy_pred = np.diff(y_pred)
    mask = np.isfinite(dy_true) & np.isfinite(dy_pred)
    if not np.any(mask):
        return None
    return float(np.mean(np.abs(dy_true[mask] - dy_pred[mask])))


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


def _rolling_preds_for_params(
    train: np.ndarray,
    val: np.ndarray,
    *,
    window_length: int,
    rank: int,
) -> np.ndarray:
    """Rolling one-step predictions for one fixed (L,r)."""
    train_arr = _to_1d_numpy(train)
    val_arr = _to_1d_numpy(val)
    preds = np.full(val_arr.size, np.nan, dtype=float)

    hist = train_arr.copy()
    L = int(window_length)
    r = int(rank)
    for i, y_true in enumerate(val_arr):
        N = hist.size
        if N > L:
            max_rank = min(L, N - L + 1)
            if r <= max_rank:
                Y = _trajectory_matrix(hist, L)
                scale = float(np.max(np.abs(Y)))
                if not np.isfinite(scale) or scale <= 0.0:
                    scale = 1.0
                U, s, Vt = _truncated_svd(Y / scale, k=r)
                try:
                    preds[i] = _one_step_from_svd(U, s, Vt, rank=r, scale=scale, window_length=L)
                except Exception:
                    preds[i] = np.nan
        hist = np.append(hist, y_true)
    return preds


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
    preds = _rolling_preds_for_params(
        tr.to_numpy(dtype=float),
        va.to_numpy(dtype=float),
        window_length=best.window_length,
        rank=best.rank,
    )
    mdf = pd.DataFrame({"y_true": va.to_numpy(dtype=float), "y_pred": preds})
    suite = compute_forecast_metrics(mdf, y_true_col="y_true", y_pred_col="y_pred")
    suite["directional_mae"] = _directional_mae(
        mdf["y_true"].to_numpy(dtype=float),
        mdf["y_pred"].to_numpy(dtype=float),
    )
    return HourlySSAResult(
        hour=h,
        best=best,
        val_score=float(best.score),
        n_train=int(tr.size),
        n_val=int(va.size),
        metrics=suite,
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
        "--all-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train/tune all canonical univariate targets in one command (DA + 6 aFRR targets).",
    )
    p.add_argument(
        "--afrr-train-path",
        default="data/model_input/afrr/train.parquet",
        help="aFRR train path used when --all-targets is enabled.",
    )
    p.add_argument(
        "--afrr-val-path",
        default="data/model_input/afrr/val.parquet",
        help="aFRR val path used when --all-targets is enabled.",
    )
    p.add_argument(
        "--output-dir",
        default="artifacts/model_runs/ssa_univariate",
        help="Directory where per-target/hour best params and fitted models are saved.",
    )
    p.add_argument(
        "--enable-tensorboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write scalar metrics to TensorBoard logs.",
    )
    p.add_argument(
        "--tensorboard-logdir",
        default="artifacts/tensorlogs/ssa_univariate",
        help="Base TensorBoard log directory.",
    )
    p.add_argument(
        "--smoke-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use tiny search (L=25..26, r=3..4) and last 48 validation rows.",
    )
    return p.parse_args()


def _run_hourly_tuning_for_target(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    L_values: Iterable[int],
    r_values: Iterable[int],
    metric: MetricName,
    min_points_per_hour: int,
    n_jobs: int,
) -> tuple[list[HourlySSAResult], float, dict[int, pd.Series]]:
    train_hourly = _split_hourly(train_df, target_col)
    val_hourly = _split_hourly(val_df, target_col)

    if Parallel is None or delayed is None:
        print("[WARN] joblib not installed. Falling back to sequential hourly tuning.")
        raw_results = [
            _train_hour(
                h,
                train_hourly[h],
                val_hourly[h],
                L_values=L_values,
                r_values=r_values,
                metric=metric,
                min_points_per_hour=min_points_per_hour,
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
                metric=metric,
                min_points_per_hour=min_points_per_hour,
            )
            for h in range(24)
        ]
        raw_results = Parallel(n_jobs=n_jobs, prefer="processes")(jobs)

    results = [r for r in raw_results if r is not None]
    if not results:
        raise RuntimeError(
            f"No hourly models were trained for {target_col}. "
            f"Check min-points-per-hour / split sizes."
        )
    weighted_score = float(
        np.average(np.array([r.val_score for r in results]), weights=np.array([r.n_val for r in results]))
    )
    return results, weighted_score, train_hourly


def _save_target_artifacts(
    *,
    output_dir: Path,
    target_col: str,
    metric: MetricName,
    weighted_score: float,
    results: list[HourlySSAResult],
    train_hourly: dict[int, pd.Series],
) -> None:
    target_dir = output_dir / target_col
    models_dir = target_dir / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in sorted(results, key=lambda x: x.hour):
        rec = {
            "target_col": target_col,
            "hour": r.hour,
            "window_length": r.best.window_length,
            "rank": r.best.rank,
            "val_score": r.val_score,
            "n_train": r.n_train,
            "n_val": r.n_val,
        }
        if r.metrics:
            for k in (
                "mae",
                "rmse",
                "wmape",
                "mbe",
                "directional_accuracy",
                "directional_mae",
                "over_prediction_ratio",
                "n_rows_scored",
            ):
                rec[k] = r.metrics.get(k)
        rows.append(rec)
        # Fit final per-hour model on full train series for this target/hour and persist.
        model = UnivariateSSA(window_length=r.best.window_length, rank=r.best.rank).fit(train_hourly[r.hour])
        with (models_dir / f"hour_{r.hour:02d}.pkl").open("wb") as f:
            pickle.dump(model, f)

    df = pd.DataFrame(rows).sort_values("hour")
    df.to_csv(target_dir / "hourly_best_params.csv", index=False)

    summary = {
        "target_col": target_col,
        "metric": metric,
        "weighted_validation_score": weighted_score,
        "trained_hours": len(results),
        "mean_rmse": float(pd.to_numeric(df.get("rmse"), errors="coerce").mean()) if "rmse" in df.columns else None,
        "mean_mae": float(pd.to_numeric(df.get("mae"), errors="coerce").mean()) if "mae" in df.columns else None,
        "mean_directional_accuracy": float(pd.to_numeric(df.get("directional_accuracy"), errors="coerce").mean())
        if "directional_accuracy" in df.columns
        else None,
        "mean_directional_mae": float(pd.to_numeric(df.get("directional_mae"), errors="coerce").mean())
        if "directional_mae" in df.columns
        else None,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (target_dir / "summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")


def _run_single_target(
    *,
    train_path: str,
    val_path: str,
    target_col: str,
    L_values: Iterable[int],
    r_values: Iterable[int],
    metric: MetricName,
    min_points_per_hour: int,
    n_jobs: int,
    horizon_demo: int,
    output_dir: Path,
    tb_writer: object | None,
) -> dict[str, object]:
    train_df = _load_univariate_frame(train_path, target_col)
    val_df = _load_univariate_frame(val_path, target_col)

    results, weighted_score, train_hourly = _run_hourly_tuning_for_target(
        train_df=train_df,
        val_df=val_df,
        target_col=target_col,
        L_values=L_values,
        r_values=r_values,
        metric=metric,
        min_points_per_hour=min_points_per_hour,
        n_jobs=n_jobs,
    )

    _save_target_artifacts(
        output_dir=output_dir,
        target_col=target_col,
        metric=metric,
        weighted_score=weighted_score,
        results=results,
        train_hourly=train_hourly,
    )

    # demo forecast from hour 00 model if present
    h0 = next((r for r in results if r.hour == 0), results[0])
    model = UnivariateSSA(window_length=h0.best.window_length, rank=h0.best.rank).fit(train_hourly[h0.hour])
    fcst = model.forecast(horizon=int(horizon_demo), refit_each_step=True)

    out = pd.DataFrame(
        {
            "hour": [r.hour for r in results],
            "window_length": [r.best.window_length for r in results],
            "rank": [r.best.rank for r in results],
            "val_score": [r.val_score for r in results],
            "n_train": [r.n_train for r in results],
            "n_val": [r.n_val for r in results],
            "rmse": [None if r.metrics is None else r.metrics.get("rmse") for r in results],
            "mae": [None if r.metrics is None else r.metrics.get("mae") for r in results],
            "directional_accuracy": [None if r.metrics is None else r.metrics.get("directional_accuracy") for r in results],
            "directional_mae": [None if r.metrics is None else r.metrics.get("directional_mae") for r in results],
            "wmape": [None if r.metrics is None else r.metrics.get("wmape") for r in results],
            "mbe": [None if r.metrics is None else r.metrics.get("mbe") for r in results],
            "over_prediction_ratio": [None if r.metrics is None else r.metrics.get("over_prediction_ratio") for r in results],
        }
    ).sort_values("hour")

    print(f"\n=== Target: {target_col} ===")
    print(f"trained_hours={len(results)}/24")
    print(f"weighted_validation_{metric}={weighted_score:.6f}")
    print(out.to_string(index=False))
    print(f"Demo: hour={h0.hour:02d} {horizon_demo}-step first10: {np.round(fcst[:10], 4)}")
    print(f"Saved artifacts: {(output_dir / target_col).resolve()}")

    if tb_writer is not None:
        try:
            tb_writer.add_scalar(f"{target_col}/weighted_validation_{metric}", weighted_score, 0)
            for m in ("rmse", "mae", "directional_accuracy", "directional_mae", "wmape", "mbe", "over_prediction_ratio"):
                vals = pd.to_numeric(out[m], errors="coerce")
                if np.isfinite(vals).any():
                    tb_writer.add_scalar(f"{target_col}/mean_{m}", float(np.nanmean(vals.to_numpy(dtype=float))), 0)
            for _, row in out.iterrows():
                h = int(row["hour"])
                for m in ("rmse", "mae", "directional_accuracy", "directional_mae", "wmape", "mbe", "over_prediction_ratio"):
                    v = pd.to_numeric(pd.Series([row.get(m)]), errors="coerce").iloc[0]
                    if np.isfinite(v):
                        tb_writer.add_scalar(f"{target_col}/hour_{h:02d}/{m}", float(v), 0)
        except Exception:
            pass

    return {
        "target_col": target_col,
        "weighted_score": weighted_score,
        "trained_hours": len(results),
        "mean_rmse": float(pd.to_numeric(out["rmse"], errors="coerce").mean()),
        "mean_mae": float(pd.to_numeric(out["mae"], errors="coerce").mean()),
        "mean_directional_accuracy": float(pd.to_numeric(out["directional_accuracy"], errors="coerce").mean()),
        "mean_directional_mae": float(pd.to_numeric(out["directional_mae"], errors="coerce").mean()),
    }


def main() -> None:
    args = parse_args()

    # Search space
    if args.smoke_test:
        L_values = range(25, 27)   # 25,26
        r_values = range(3, 5)     # 3,4
    else:
        L_values = range(25, 65)   # 24 < L < 65
        r_values = range(3, 15)    # 3 <= r <= 14

    output_dir = Path(args.output_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = None
    if args.enable_tensorboard:
        if SummaryWriter is None:
            print("[WARN] TensorBoard requested but torch/tensorboard is unavailable.")
        else:
            tb_dir = Path(args.tensorboard_logdir) / run_id
            tb_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_dir))

    if args.all_targets:
        targets = [("data/model_input/da/train.parquet", "data/model_input/da/val.parquet", "target_da_price")]
        targets += [(args.afrr_train_path, args.afrr_val_path, t) for t in AFRR_TARGETS]
    else:
        targets = [(args.train_path, args.val_path, args.target_col)]

    summary_rows = []
    for train_path, val_path, target_col in targets:
        train_df = _load_univariate_frame(train_path, target_col)
        val_df = _load_univariate_frame(val_path, target_col)

        if args.smoke_test:
            val_df = val_df.tail(48).reset_index(drop=True)
            results, weighted_score, train_hourly = _run_hourly_tuning_for_target(
                train_df=train_df,
                val_df=val_df,
                target_col=target_col,
                L_values=L_values,
                r_values=r_values,
                metric=args.metric,
                min_points_per_hour=args.min_points_per_hour,
                n_jobs=args.n_jobs,
            )
            _save_target_artifacts(
                output_dir=run_dir,
                target_col=target_col,
                metric=args.metric,
                weighted_score=weighted_score,
                results=results,
                train_hourly=train_hourly,
            )

            h0 = next((r for r in results if r.hour == 0), results[0])
            model = UnivariateSSA(window_length=h0.best.window_length, rank=h0.best.rank).fit(train_hourly[h0.hour])
            fcst = model.forecast(horizon=int(args.horizon_demo), refit_each_step=True)
            print(f"\n=== Target: {target_col} ===")
            print(f"trained_hours={len(results)}/24")
            print(f"weighted_validation_{args.metric}={weighted_score:.6f}")
            print(f"Demo: hour={h0.hour:02d} {args.horizon_demo}-step first10: {np.round(fcst[:10], 4)}")
            print(f"Saved artifacts: {(run_dir / target_col).resolve()}")

            out_smoke = pd.DataFrame(
                {
                    "rmse": [None if rr.metrics is None else rr.metrics.get("rmse") for rr in results],
                    "mae": [None if rr.metrics is None else rr.metrics.get("mae") for rr in results],
                    "directional_accuracy": [
                        None if rr.metrics is None else rr.metrics.get("directional_accuracy") for rr in results
                    ],
                    "directional_mae": [None if rr.metrics is None else rr.metrics.get("directional_mae") for rr in results],
                }
            )
            if tb_writer is not None:
                try:
                    tb_writer.add_scalar(f"{target_col}/weighted_validation_{args.metric}", weighted_score, 0)
                    for m in ("rmse", "mae", "directional_accuracy", "directional_mae"):
                        vals = pd.to_numeric(out_smoke[m], errors="coerce")
                        if np.isfinite(vals).any():
                            tb_writer.add_scalar(f"{target_col}/mean_{m}", float(np.nanmean(vals.to_numpy(dtype=float))), 0)
                except Exception:
                    pass

            summary_rows.append(
                {
                    "target_col": target_col,
                    "weighted_score": weighted_score,
                    "trained_hours": len(results),
                    "mean_rmse": float(pd.to_numeric(out_smoke["rmse"], errors="coerce").mean()),
                    "mean_mae": float(pd.to_numeric(out_smoke["mae"], errors="coerce").mean()),
                    "mean_directional_accuracy": float(
                        pd.to_numeric(out_smoke["directional_accuracy"], errors="coerce").mean()
                    ),
                    "mean_directional_mae": float(
                        pd.to_numeric(out_smoke["directional_mae"], errors="coerce").mean()
                    ),
                }
            )
        else:
            result = _run_single_target(
                train_path=train_path,
                val_path=val_path,
                target_col=target_col,
                L_values=L_values,
                r_values=r_values,
                metric=args.metric,
                min_points_per_hour=args.min_points_per_hour,
                n_jobs=args.n_jobs,
                horizon_demo=args.horizon_demo,
                output_dir=run_dir,
                tb_writer=tb_writer,
            )
            summary_rows.append(result)

    summary_df = pd.DataFrame(summary_rows).sort_values("target_col").reset_index(drop=True)
    summary_df.to_csv(run_dir / "run_summary.csv", index=False)
    print("\n=== Run Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nAll outputs saved under: {run_dir.resolve()}")
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()


if __name__ == "__main__":
    main()
