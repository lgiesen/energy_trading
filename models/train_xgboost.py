"""Phase-2 challenger training: XGBoost + Optuna with Purged CV and PnL objective.

Usage example:
    ./.venv/bin/python models/train_xgboost.py \
      --base-dir data/model_input \
      --bundle afrr \
      --target-col target_afrr_activation_price_vwap_pos_h1 \
      --n-trials 40 \
      --cv-n-splits 3 \
      --cv-test-size 672 \
      --cv-gap-hours 72 \
      --device cpu
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Headless-safe plotting/runtime defaults for batch execution.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "data" / ".mplconfig").resolve()))

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_trading.config import BATTERY_SPECS
from energy_trading.models.baselines import PersistencePredictor
from energy_trading.models.cv import PurgedTimeSeriesSplit
from energy_trading.models.prepare_ml_bundles import BundleName, load_processed_data
from energy_trading.visualization.metrics import BatteryParams, calculate_pnl

LOGGER = logging.getLogger("train_xgboost_phase2")


@dataclass(frozen=True)
class FoldResult:
    fold: int
    pnl_eur: float
    rmse: float


def _fit_with_early_stopping_compat(
    model: XGBRegressor,
    *,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    early_stopping_rounds: int,
) -> None:
    """Fit helper compatible across XGBoost sklearn API variants."""
    # XGBoost variants differ: some accept early_stopping_rounds in fit(...),
    # others require callbacks in constructor.
    try:
        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_eval, y_eval)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
        return
    except TypeError:
        pass

    # Fallback path: use callback API.
    import xgboost as xgb  # local import to avoid hard dependency differences

    model.set_params(
        callbacks=[xgb.callback.EarlyStopping(rounds=int(early_stopping_rounds), save_best=True)]
    )
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_eval, y_eval)],
        verbose=False,
    )


def _resolve_default_target(bundle: BundleName) -> str:
    if bundle == "da":
        return "target_da_price_h1"
    return "target_afrr_activation_price_vwap_pos_h1"


def _drop_unlagged_target_like_features(X: pd.DataFrame) -> pd.DataFrame:
    """Enforce leakage guard by removing unlagged target-like columns from X."""
    # Hard deny-list (raw contemporaneous targets or close aliases).
    blocked_exact = {
        "da_price",
        "afrr_vwap_pos",
        "afrr_vwap_neg",
        "afrr_activation_price_vwap_pos",
        "afrr_activation_price_vwap_neg",
        "afrr_capacity_price_pos",
        "afrr_capacity_price_neg",
        "afrr_activation_rate",
        "afrr_rate",
    }
    blocked_regex = re.compile(
        r"^(target_.*|afrr_.*(vwap|capacity_price|activation_rate).*)$",
        re.IGNORECASE,
    )

    drop_cols: list[str] = []
    for c in X.columns:
        c_low = c.lower()
        is_lagged = "_lag_" in c_low
        is_pit = c_low.endswith("_pit")
        if c_low in blocked_exact:
            drop_cols.append(c)
            continue
        if blocked_regex.match(c_low) and (not is_lagged) and (not is_pit):
            drop_cols.append(c)
            continue

    if drop_cols:
        LOGGER.warning("Leakage guard dropped %d columns from X: %s", len(drop_cols), drop_cols[:15])
        return X.drop(columns=drop_cols, errors="ignore")
    return X


def _battery_params_from_config() -> BatteryParams:
    cap = float(BATTERY_SPECS["capacity_mwh"])
    return BatteryParams(
        capacity_mwh=cap,
        power_mw=float(BATTERY_SPECS["power_mw"]),
        roundtrip_efficiency=float(BATTERY_SPECS["efficiency_rt"]),
        initial_soc_mwh=float(BATTERY_SPECS["initial_soc"]) * cap,
        interval_hours=1.0,
        high_percentile=80.0,
        low_percentile=20.0,
    )


def _pnl_metric(y_true: pd.Series, y_pred: np.ndarray, battery_params: BatteryParams) -> float:
    out = calculate_pnl(y_true=y_true.to_numpy(), y_pred=np.asarray(y_pred), battery_params=battery_params)
    return float(out["pnl_eur"])


def _fit_eval_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    params: dict,
    early_stopping_rounds: int,
    battery_params: BatteryParams,
) -> FoldResult:
    # Small causal holdout inside fold-train for early stopping (no validation leakage).
    n_tr = len(X_tr)
    es_size = max(24 * 3, int(0.1 * n_tr))
    if n_tr <= es_size + 50:
        es_size = max(24, int(0.05 * n_tr))
    fit_end = max(1, n_tr - es_size)

    X_fit = X_tr.iloc[:fit_end]
    y_fit = y_tr.iloc[:fit_end]
    X_es = X_tr.iloc[fit_end:]
    y_es = y_tr.iloc[fit_end:]

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        **params,
    )
    _fit_with_early_stopping_compat(
        model,
        X_fit=X_fit,
        y_fit=y_fit,
        X_eval=X_es,
        y_eval=y_es,
        early_stopping_rounds=early_stopping_rounds,
    )
    pred = model.predict(X_va)
    rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
    pnl = _pnl_metric(y_true=y_va, y_pred=pred, battery_params=battery_params)
    return FoldResult(fold=-1, pnl_eur=pnl, rmse=rmse)


def _run_optuna_hpo(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_trials: int,
    cv_n_splits: int,
    cv_test_size: int,
    cv_gap_hours: int,
    early_stopping_rounds: int,
    battery_params: BatteryParams,
    device: str,
) -> tuple[dict, pd.DataFrame]:
    try:
        import optuna
    except Exception as exc:
        raise RuntimeError("optuna is required. Install with `pip install optuna`.") from exc

    splitter = PurgedTimeSeriesSplit(
        n_splits=cv_n_splits,
        test_size=cv_test_size,
        gap_hours=cv_gap_hours,
        frequency="1h",
        min_train_size=max(500, cv_test_size),
    )

    trial_rows: list[dict] = []

    def objective(trial: "optuna.trial.Trial") -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
            "device": device,
        }

        pnls: list[float] = []
        rmses: list[float] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X_train), start=1):
            X_tr = X_train.iloc[tr_idx].copy()
            y_tr = y_train.iloc[tr_idx].copy()
            X_va = X_train.iloc[va_idx].copy()
            y_va = y_train.iloc[va_idx].copy()

            fr = _fit_eval_fold(
                X_tr,
                y_tr,
                X_va,
                y_va,
                params,
                early_stopping_rounds=early_stopping_rounds,
                battery_params=battery_params,
            )
            pnls.append(fr.pnl_eur)
            rmses.append(fr.rmse)

        avg_pnl = float(np.mean(pnls)) if pnls else float("-inf")
        avg_rmse = float(np.mean(rmses)) if rmses else float("inf")

        trial_rows.append(
            {
                "trial": trial.number,
                "avg_pnl_eur": avg_pnl,
                "avg_rmse": avg_rmse,
                **params,
            }
        )
        LOGGER.info("trial=%s avg_pnl=%.2f avg_rmse=%.4f", trial.number, avg_pnl, avg_rmse)
        return avg_pnl

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    best_params["device"] = device
    trials_df = pd.DataFrame(trial_rows).sort_values("avg_pnl_eur", ascending=False)
    return best_params, trials_df


def _persistence_baseline_pnl(y_train: pd.Series, y_val: pd.Series, battery_params: BatteryParams) -> float:
    baseline = PersistencePredictor(lag_hours=168, frequency="1h")
    full = pd.concat([y_train, y_val], axis=0)
    pred_full = baseline.predict_from_series(full)
    # Use positional tail extraction to avoid duplicate-index alignment issues.
    pred_val = pd.to_numeric(pred_full.iloc[-len(y_val) :], errors="coerce")
    y_val_pos = y_val.iloc[-len(pred_val) :]
    valid = pred_val.notna().to_numpy() & y_val_pos.notna().to_numpy()
    if not bool(valid.any()):
        return float("nan")
    return _pnl_metric(
        y_true=y_val_pos.iloc[valid],
        y_pred=pred_val.iloc[valid].to_numpy(),
        battery_params=battery_params,
    )


def _train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
    early_stopping_rounds: int,
) -> XGBRegressor:
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        **params,
    )
    _fit_with_early_stopping_compat(
        model,
        X_fit=X_train,
        y_fit=y_train,
        X_eval=X_val,
        y_eval=y_val,
        early_stopping_rounds=early_stopping_rounds,
    )
    return model


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train XGBoost challenger with Optuna + Purged CV + PnL objective.")
    p.add_argument("--base-dir", default="data/model_input")
    p.add_argument("--bundle", choices=["da", "afrr"], default="afrr")
    p.add_argument("--target-col", default="")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--cv-n-splits", type=int, default=3)
    p.add_argument("--cv-test-size", type=int, default=24 * 28)
    p.add_argument("--cv-gap-hours", type=int, default=72)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--model-out", default="models/checkpoints/xgboost_afrr_challenger.joblib")
    p.add_argument("--report-out", default="data/reports/xgboost_challenger_report.json")
    p.add_argument("--trials-out", default="data/reports/xgboost_optuna_trials.csv")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = _build_cli().parse_args()

    bundle: BundleName = args.bundle
    target_col = args.target_col.strip() or _resolve_default_target(bundle)

    X_train, y_train_df = load_processed_data(bundle=bundle, split="train", base_dir=args.base_dir)
    X_val, y_val_df = load_processed_data(bundle=bundle, split="val", base_dir=args.base_dir)

    if target_col not in y_train_df.columns or target_col not in y_val_df.columns:
        raise KeyError(
            f"Missing target '{target_col}'. Available train targets: {list(y_train_df.columns)}"
        )

    # Law: explicit X/y split and hard leakage guard.
    X_train = _drop_unlagged_target_like_features(X_train)
    X_val = _drop_unlagged_target_like_features(X_val)

    y_train = pd.to_numeric(y_train_df[target_col], errors="coerce")
    y_val = pd.to_numeric(y_val_df[target_col], errors="coerce")

    mtr = y_train.notna()
    mva = y_val.notna()
    X_train = X_train.loc[mtr].copy()
    y_train = y_train.loc[mtr].copy()
    X_val = X_val.loc[mva].copy()
    y_val = y_val.loc[mva].copy()

    battery_params = _battery_params_from_config()

    best_params, trials_df = _run_optuna_hpo(
        X_train,
        y_train,
        n_trials=args.n_trials,
        cv_n_splits=args.cv_n_splits,
        cv_test_size=args.cv_test_size,
        cv_gap_hours=args.cv_gap_hours,
        early_stopping_rounds=args.early_stopping_rounds,
        battery_params=battery_params,
        device=args.device,
    )

    model = _train_final_model(
        X_train,
        y_train,
        X_val,
        y_val,
        params=best_params,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    pred_val = model.predict(X_val)

    model_pnl = _pnl_metric(y_true=y_val, y_pred=pred_val, battery_params=battery_params)
    model_rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
    baseline_pnl = _persistence_baseline_pnl(y_train=y_train, y_val=y_val, battery_params=battery_params)

    model_out = Path(args.model_out)
    report_out = Path(args.report_out)
    trials_out = Path(args.trials_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    trials_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_out)
    trials_df.to_csv(trials_out, index=False)

    improvement_pct = np.nan
    verdict = "FAILURE"
    if np.isfinite(baseline_pnl) and baseline_pnl != 0:
        improvement_pct = 100.0 * (model_pnl - baseline_pnl) / abs(baseline_pnl)
        verdict = "SUCCESS" if model_pnl > baseline_pnl else "FAILURE"
    elif np.isfinite(baseline_pnl):
        verdict = "SUCCESS" if model_pnl > baseline_pnl else "FAILURE"

    summary = {
        "bundle": bundle,
        "target_col": target_col,
        "rows_train": int(len(X_train)),
        "rows_val": int(len(X_val)),
        "best_params": best_params,
        "val_rmse": model_rmse,
        "val_pnl_eur": model_pnl,
        "baseline_persistence_168h_pnl_eur": baseline_pnl,
        "improvement_pct_vs_baseline": improvement_pct,
        "verdict": verdict,
        "model_out": str(model_out.resolve()),
        "trials_out": str(trials_out.resolve()),
    }
    report_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    comp = pd.DataFrame(
        [
            {"model": "XGBoost Challenger", "PnL_EUR": model_pnl, "RMSE": model_rmse},
            {"model": "Persistence 168h", "PnL_EUR": baseline_pnl, "RMSE": np.nan},
        ]
    )
    print("\n=== Challenger vs Baseline ===")
    print(comp.to_string(index=False))

    if verdict == "SUCCESS":
        print(f"SUCCESS: ML model outperformed Baseline by {improvement_pct:.2f}%")
    else:
        print("FAILURE: Model did not beat the 168h Persistence.")

    print("\nTarget columns to confirm:")
    print("- target_afrr_activation_price_vwap_pos_h1")
    print("- target_afrr_activation_price_vwap_neg_h1")
    print("- target_afrr_capacity_price_pos_h1")
    print("- target_afrr_capacity_price_neg_h1")
    print("- target_afrr_rate_h1")
    print("- target_da_price_h1")

    print("\nBattery params used for PnL metric:")
    print(f"- capacity_mwh={battery_params.capacity_mwh}")
    print(f"- power_mw={battery_params.power_mw}")
    print(f"- roundtrip_efficiency={battery_params.roundtrip_efficiency:.4f}")


if __name__ == "__main__":
    main()
