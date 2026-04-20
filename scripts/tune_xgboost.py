#!/usr/bin/env python3
"""Optuna tuning for XGBoost with spike/tail-weighted optimization.

This script keeps the project's fixed chronological split (train/val) and
optimizes hyperparameters against an asymmetric weighted MAE objective that
penalizes tail hours more strongly.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_trading.models.prepare_ml_bundles import BundleName, load_processed_data
from energy_trading.models.train_xgboost_export import _add_dynamics_features, _resolve_targets

LOGGER = logging.getLogger(__name__)


def _resolve_device(requested: str, allow_cpu: bool) -> str:
    req = (requested or "cuda").strip().lower()
    if req == "cuda":
        try:
            import xgboost as xgb

            # lightweight probe via xgboost version import only.
            _ = xgb.__version__
            return "cuda"
        except Exception:
            if allow_cpu:
                LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
                return "cpu"
            raise
    return "cpu"


def _build_tail_weights(
    y: pd.Series,
    *,
    q_low: float,
    q_high: float,
    tail_weight: float,
) -> tuple[np.ndarray, float, float]:
    yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(yv)
    if not bool(mask.any()):
        return np.ones_like(yv, dtype=float), float("nan"), float("nan")
    lo = float(np.nanquantile(yv[mask], q_low))
    hi = float(np.nanquantile(yv[mask], q_high))
    w = np.ones_like(yv, dtype=float)
    w[yv <= lo] = float(tail_weight)
    w[yv >= hi] = float(tail_weight)
    w[~mask] = 1.0
    return w, lo, hi


def _weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(w) & (w > 0)
    if not bool(m.any()):
        return float("nan")
    return float(np.average(np.abs(y_true[m] - y_pred[m]), weights=w[m]))


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tune XGBoost with tail-weighted objective via Optuna.")
    p.add_argument("--base-dir", default="data/model_input")
    p.add_argument("--bundle", choices=["da", "afrr"], default="afrr")
    p.add_argument("--target-col", default="", help="Optional explicit target column.")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--q-low", type=float, default=0.05)
    p.add_argument("--q-high", type=float, default=0.95)
    p.add_argument("--tail-weight", type=float, default=3.0)
    p.add_argument("--study-name", default="xgb_tail_weighted_tuning")
    p.add_argument("--out-dir", default="artifacts/hpo")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_cli().parse_args()

    try:
        import optuna
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependencies for HPO. Install `optuna` and `xgboost`."
        ) from exc

    np.random.seed(args.seed)
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle: BundleName = args.bundle  # type: ignore[assignment]
    device = _resolve_device(args.device, allow_cpu=args.allow_cpu)

    X_train_df, y_train_df = load_processed_data(bundle=bundle, split="train", base_dir=base_dir)
    X_val_df, y_val_df = load_processed_data(bundle=bundle, split="val", base_dir=base_dir)
    target_cols = _resolve_targets(bundle, list(y_train_df.columns), args.target_col.strip() or None)
    target_col = target_cols[0]

    y_tr = pd.to_numeric(y_train_df[target_col], errors="coerce")
    y_va = pd.to_numeric(y_val_df[target_col], errors="coerce")
    tr_mask = y_tr.notna()
    va_mask = y_va.notna()

    X_tr = _add_dynamics_features(X_train_df.loc[tr_mask].copy(), prune_midterm_lags=True)
    X_va = _add_dynamics_features(X_val_df.loc[va_mask].copy(), prune_midterm_lags=True)
    y_tr = y_tr.loc[tr_mask].copy()
    y_va = y_va.loc[va_mask].copy()

    tr_w, ql, qh = _build_tail_weights(
        y_tr, q_low=float(args.q_low), q_high=float(args.q_high), tail_weight=float(args.tail_weight)
    )
    va_w, _, _ = _build_tail_weights(
        y_va, q_low=float(args.q_low), q_high=float(args.q_high), tail_weight=float(args.tail_weight)
    )
    y_va_np = y_va.to_numpy(dtype=float)

    LOGGER.info(
        "Tuning target=%s bundle=%s rows(train=%s,val=%s) tail-weights=(q%.2f, q%.2f, w=%.2f) thresholds=(%.4f, %.4f)",
        target_col,
        bundle,
        len(X_tr),
        len(X_va),
        args.q_low,
        args.q_high,
        args.tail_weight,
        ql,
        qh,
    )

    trial_rows: list[dict[str, float | int]] = []

    def _objective(trial: "optuna.Trial") -> float:
        max_depth = trial.suggest_int("max_depth", 5, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.12, log=True)
        subsample = trial.suggest_float("subsample", 0.65, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.55, 1.0)
        min_child_weight = trial.suggest_float("min_child_weight", 1.0, 12.0)

        model = XGBRegressor(
            objective="reg:absoluteerror",
            n_estimators=int(args.n_estimators),
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            tree_method="hist",
            device=device,
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            min_child_weight=float(min_child_weight),
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=int(args.seed),
            early_stopping_rounds=max(0, int(args.early_stopping_rounds)),
            n_jobs=-1,
        )
        # Weighted training = asymmetric objective pressure on tail hours.
        model.fit(
            X_tr,
            y_tr,
            sample_weight=tr_w,
            eval_set=[(X_va, y_va)],
            sample_weight_eval_set=[va_w],
            verbose=False,
        )
        pred = model.predict(X_va)
        wmae = _weighted_mae(y_va_np, pred, va_w)
        mae = float(np.mean(np.abs(y_va_np - pred)))
        trial_rows.append(
            {
                "trial": int(trial.number),
                "weighted_mae": float(wmae),
                "mae": float(mae),
                "max_depth": int(max_depth),
                "learning_rate": float(learning_rate),
                "subsample": float(subsample),
                "colsample_bytree": float(colsample_bytree),
                "min_child_weight": float(min_child_weight),
            }
        )
        trial.set_user_attr("mae", float(mae))
        return float(wmae)

    study = optuna.create_study(direction="minimize", study_name=args.study_name)
    study.optimize(_objective, n_trials=int(args.n_trials))

    best = study.best_trial
    result = {
        "study_name": args.study_name,
        "bundle": bundle,
        "target_col": target_col,
        "device": device,
        "n_trials": int(args.n_trials),
        "n_estimators": int(args.n_estimators),
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "tail_weighting": {
            "q_low": float(args.q_low),
            "q_high": float(args.q_high),
            "tail_weight": float(args.tail_weight),
            "train_threshold_low": float(ql),
            "train_threshold_high": float(qh),
        },
        "best_trial": int(best.number),
        "best_weighted_mae": float(best.value),
        "best_mae": float(best.user_attrs.get("mae", np.nan)),
        "best_params": best.params,
    }

    out_json = out_dir / f"xgb_optuna_{bundle}_{target_col}.json"
    out_csv = out_dir / f"xgb_optuna_{bundle}_{target_col}_trials.csv"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(trial_rows).to_csv(out_csv, index=False)

    print("[OK] Optuna tuning finished.")
    print(f"- Best weighted MAE: {result['best_weighted_mae']:.6f}")
    print(f"- Best params: {result['best_params']}")
    print(f"- Result JSON: {out_json}")
    print(f"- Trials CSV: {out_csv}")


if __name__ == "__main__":
    main()

