#!/usr/bin/env python3
"""Target-specific tuning for linear SGD quantile baseline."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import QuantileRegressor

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_trading.evaluation.shared_evaluator import compute_shared_metrics
from energy_trading.models.prepare_ml_bundles import BundleName, load_processed_data


def _asymmetric_mae(y_true: np.ndarray, y_pred: np.ndarray, penalty: float = 3.0) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[m]
    yp = y_pred[m]
    if yt.size == 0:
        return float("nan")
    q90 = float(np.percentile(yt, 90))
    err = yp - yt
    w = np.ones_like(err, dtype=float)
    w[(yt >= q90) & (err < 0)] = float(penalty)
    return float(np.mean(np.abs(err) * w))


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not bool(m.any()):
        return float("nan")
    e = y_true[m] - y_pred[m]
    return float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))


def _mean_pinball_loss(y_true: np.ndarray, preds_by_q: dict[float, np.ndarray]) -> float:
    vals: list[float] = []
    for tau, pred in preds_by_q.items():
        v = _pinball_loss(y_true, pred, tau)
        if np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _parse_hpo_quantiles(raw: str) -> list[float]:
    out = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        q = float(tok)
        if not (0.0 < q < 1.0):
            raise ValueError(f"Invalid quantile in --hpo-quantiles: {q}")
        out.append(q)
    if not out:
        raise ValueError("--hpo-quantiles must contain at least one quantile.")
    return sorted(set(out))


def _build_model(
    alpha: float,
    l1_ratio: float,
    *,
    learning_rate: str,
    eta0: float,
    quantile: float = 0.5,
    seed: int = 42,
) -> Pipeline:
    try:
        model = SGDRegressor(
            loss="quantile",
            quantile=float(quantile),
            alpha=float(alpha),
            penalty="elasticnet",
            l1_ratio=float(l1_ratio),
            max_iter=3000,
            tol=1e-4,
            learning_rate=str(learning_rate),
            eta0=float(eta0),
            power_t=0.25,
            random_state=int(seed),
        )
    except TypeError:
        # Fallback for sklearn versions where SGD quantile is unavailable.
        model = QuantileRegressor(
            quantile=float(quantile),
            alpha=float(alpha),
            solver="highs",
        )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(10.0, 90.0))),
            ("model", model),
        ]
    )


def _resolve_target(bundle: BundleName, cols: list[str], requested: str | None) -> str:
    if requested:
        if requested not in cols:
            raise KeyError(f"Unknown target: {requested}")
        return requested
    if bundle == "da":
        return "target_da_price"
    for c in [
        "target_afrr_activation_price_vwap_pos",
        "target_afrr_activation_price_vwap_neg",
        "target_afrr_capacity_price_pos",
        "target_afrr_capacity_price_neg",
        "target_afrr_activation_rate_pos",
        "target_afrr_activation_rate_neg",
    ]:
        if c in cols:
            return c
    raise KeyError("No supported target found in bundle labels.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune linear SGD quantile model.")
    p.add_argument("--base-dir", default="data/model_input")
    p.add_argument("--bundle", choices=["da", "afrr"], default="afrr")
    p.add_argument("--target-col", default="")
    p.add_argument("--selection-metric", choices=["pinball_mean", "mae", "tail_upper_mae", "asymmetric_mae"], default="pinball_mean")
    p.add_argument("--hpo-quantiles", default="0.1,0.5,0.9")
    p.add_argument("--tail-penalty", type=float, default=3.0)
    p.add_argument("--alpha-grid", default="1e-5,5e-5,1e-4,5e-4,1e-3")
    p.add_argument("--l1-ratio-grid", default="0.05,0.15,0.30")
    p.add_argument("--learning-rate-grid", default="optimal,adaptive")
    p.add_argument("--eta0-grid", default="0.001,0.01,0.1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="artifacts/hpo")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    IS_SMOKE_TEST = os.environ.get("IS_SMOKE_TEST", "0") == "1"
    if IS_SMOKE_TEST:
        print("⚠️ SMOKE TEST MODE AKTIVIERT - Reduziere Rechenlast!")
        args.n_trials = 2
        args.n_estimators = 5
        args.epochs = 1
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle: BundleName = args.bundle  # type: ignore[assignment]
    hpo_quantiles = _parse_hpo_quantiles(args.hpo_quantiles)
    X_tr_df, y_tr_df = load_processed_data(bundle=bundle, split="train", base_dir=args.base_dir)
    X_va_df, y_va_df = load_processed_data(bundle=bundle, split="val", base_dir=args.base_dir)
    target = _resolve_target(bundle, list(y_tr_df.columns), args.target_col.strip() or None)

    y_tr = pd.to_numeric(y_tr_df[target], errors="coerce")
    y_va = pd.to_numeric(y_va_df[target], errors="coerce")
    tr_mask = y_tr.notna()
    va_mask = y_va.notna()
    X_tr = X_tr_df.loc[tr_mask].copy()
    X_va = X_va_df.loc[va_mask].copy()
    y_tr = y_tr.loc[tr_mask].to_numpy(dtype=float)
    y_va = y_va.loc[va_mask].to_numpy(dtype=float)
    if IS_SMOKE_TEST:
        X_tr = X_tr.tail(min(len(X_tr), 3000)).copy()
        X_va = X_va.tail(min(len(X_va), 1000)).copy()
        y_tr = y_tr[-min(len(y_tr), 3000) :]
        y_va = y_va[-min(len(y_va), 1000) :]

    alpha_grid = [float(x) for x in args.alpha_grid.split(",") if x.strip()]
    l1_grid = [float(x) for x in args.l1_ratio_grid.split(",") if x.strip()]
    lr_grid = [str(x).strip() for x in args.learning_rate_grid.split(",") if x.strip()]
    eta0_grid = [float(x) for x in args.eta0_grid.split(",") if x.strip()]
    rows: list[dict[str, float | str]] = []

    total_trials = len(alpha_grid) * len(l1_grid) * len(lr_grid) * len(eta0_grid)
    done_trials = 0
    t0 = time.perf_counter()
    heartbeat_every_s = 20.0
    last_heartbeat = t0

    best_obj = float("inf")
    best_cfg: dict[str, float | str] | None = None
    for alpha in alpha_grid:
        for l1_ratio in l1_grid:
            for learning_rate in lr_grid:
                for eta0 in eta0_grid:
                    trial_t0 = time.perf_counter()
                    preds_by_q: dict[float, np.ndarray] = {}
                    pred_p50: np.ndarray | None = None
                    for q in hpo_quantiles:
                        model_q = _build_model(
                            alpha=alpha,
                            l1_ratio=l1_ratio,
                            learning_rate=learning_rate,
                            eta0=eta0,
                            quantile=float(q),
                            seed=args.seed,
                        )
                        model_q.fit(X_tr, y_tr)
                        pred_q = model_q.predict(X_va)
                        preds_by_q[float(q)] = pred_q
                        if abs(float(q) - 0.5) < 1e-9:
                            pred_p50 = pred_q
                    if pred_p50 is None:
                        closest_q = min(hpo_quantiles, key=lambda z: abs(z - 0.5))
                        pred_p50 = preds_by_q[closest_q]
                    pred = pred_p50
                    pinball_mean = _mean_pinball_loss(y_va, preds_by_q)
                    shared = compute_shared_metrics(y_va, {"point": pred})
                    tail_upper = shared.get("tail_upper_mae")
                    asym = _asymmetric_mae(y_va, pred, penalty=float(args.tail_penalty))
                    if args.selection_metric == "pinball_mean":
                        obj = float(pinball_mean) if np.isfinite(pinball_mean) else float("nan")
                    elif args.selection_metric == "mae":
                        obj = float(shared.get("mae")) if shared.get("mae") is not None else float("nan")
                    elif args.selection_metric == "tail_upper_mae":
                        obj = float(tail_upper) if tail_upper is not None and np.isfinite(tail_upper) else float("nan")
                    else:
                        obj = float(asym) if np.isfinite(asym) else float("nan")
                    if not np.isfinite(obj):
                        obj = float(asym) if np.isfinite(asym) else float("inf")
                    rows.append(
                        {
                            "alpha": alpha,
                            "l1_ratio": l1_ratio,
                            "learning_rate": learning_rate,
                            "eta0": eta0,
                            "objective": obj,
                            "pinball_mean": float(pinball_mean) if np.isfinite(pinball_mean) else np.nan,
                            "hpo_quantiles": ",".join(f"{q:.2f}" for q in hpo_quantiles),
                            "tail_upper_mae": float(tail_upper)
                            if tail_upper is not None and np.isfinite(tail_upper)
                            else np.nan,
                            "asymmetric_mae": float(asym) if np.isfinite(asym) else np.nan,
                            "rmse": float(shared.get("rmse")) if shared.get("rmse") is not None else np.nan,
                            "mae": float(shared.get("mae")) if shared.get("mae") is not None else np.nan,
                        }
                    )
                    if np.isfinite(obj) and obj < best_obj:
                        best_obj = obj
                        best_cfg = {
                            "alpha": alpha,
                            "l1_ratio": l1_ratio,
                            "learning_rate": learning_rate,
                            "eta0": eta0,
                        }
                        print(
                            f"[BEST] trial={done_trials + 1}/{total_trials} "
                            f"objective={best_obj:.6f} params={best_cfg}"
                        )

                    done_trials += 1
                    now = time.perf_counter()
                    if (now - last_heartbeat) >= heartbeat_every_s or done_trials == total_trials:
                        elapsed = now - t0
                        avg_s = elapsed / max(done_trials, 1)
                        remaining = max(total_trials - done_trials, 0)
                        eta_s = remaining * avg_s
                        pct = 100.0 * done_trials / max(total_trials, 1)
                        best_obj_str = f"{best_obj:.6f}" if np.isfinite(best_obj) else "nan"
                        print(
                            f"[HEARTBEAT] progress={done_trials}/{total_trials} ({pct:.1f}%) "
                            f"elapsed={elapsed:.1f}s eta={eta_s:.1f}s "
                            f"last_trial={now - trial_t0:.2f}s best_obj={best_obj_str}"
                        )
                        last_heartbeat = now

    trials = pd.DataFrame(rows).sort_values("objective")
    out_json = out_dir / f"linear_sgd_tuning_{bundle}_{target}.json"
    out_csv = out_dir / f"linear_sgd_tuning_{bundle}_{target}_trials.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(out_csv, index=False)
    payload = {
        "bundle": bundle,
        "target_col": target,
        "selection_metric": args.selection_metric,
        "hpo_quantiles": [float(q) for q in hpo_quantiles],
        "tail_penalty": float(args.tail_penalty),
        "best_objective_value": float(best_obj),
        "best_pinball_mean": float(trials.iloc[0]["pinball_mean"]) if not trials.empty and "pinball_mean" in trials.columns else float("nan"),
        "best_params": best_cfg,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("[OK] Linear SGD tuning finished.")
    print(f"- Best objective: {best_obj:.6f}")
    print(f"- Best params: {best_cfg}")
    print(f"- Result JSON: {out_json}")
    print(f"- Trials CSV: {out_csv}")


if __name__ == "__main__":
    main()
