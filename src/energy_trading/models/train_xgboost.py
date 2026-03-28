"""Train DA-price XGBoost baseline on prepared ML bundles.

This script trains on the DA bundle (train split) and validates on the
chronological val split with early stopping to reduce overfitting.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib

# Keep matplotlib cache inside project for restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "data" / ".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.energy_trading.models.prepare_ml_bundles import load_processed_data


def _resolve_base_dir(preferred: str | Path) -> Path:
    p = Path(preferred)
    if p.exists():
        return p
    legacy = Path("data/processed_ml")
    if legacy.exists():
        return legacy
    raise FileNotFoundError(
        f"Bundle directory not found: {p}. Also checked legacy path: {legacy}."
    )


def _load_da_splits(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, y_train_df = load_processed_data(bundle="da", split="train", base_dir=base_dir)
    X_val, y_val_df = load_processed_data(bundle="da", split="val", base_dir=base_dir)

    if "da_price" not in y_train_df.columns or "da_price" not in y_val_df.columns:
        raise KeyError("Expected DA target column 'da_price' in processed bundles.")

    y_train = pd.to_numeric(y_train_df["da_price"], errors="coerce")
    y_val = pd.to_numeric(y_val_df["da_price"], errors="coerce")

    train_mask = y_train.notna()
    val_mask = y_val.notna()
    X_train = X_train.loc[train_mask].copy()
    y_train = y_train.loc[train_mask].copy()
    X_val = X_val.loc[val_mask].copy()
    y_val = y_val.loc[val_mask].copy()

    return X_train, X_val, y_train, y_val


def _naive_baseline_mae(X_val: pd.DataFrame, y_val: pd.Series) -> float:
    lag_col = "da_price_lag_24h"
    if lag_col not in X_val.columns:
        raise KeyError(
            f"Naive baseline requires '{lag_col}' in X_val, but column is missing."
        )
    y_pred_naive = pd.to_numeric(X_val[lag_col], errors="coerce")
    mask = y_pred_naive.notna() & y_val.notna()
    if not bool(mask.any()):
        raise ValueError("No valid rows available for naive baseline MAE.")
    return float(mean_absolute_error(y_val.loc[mask], y_pred_naive.loc[mask]))


def _plot_top_feature_importance(model, feature_names: list[str], out_path: Path, top_n: int = 20) -> None:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return

    imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    if imp.empty:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    imp.sort_values(ascending=True).plot(kind="barh", color="#2C7FB8")
    plt.title(f"Top {len(imp)} Feature Importances (XGBoost DA)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def train_and_evaluate(
    base_dir: Path,
    model_out: Path,
    importance_out: Path,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 50,
) -> dict[str, float]:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "xgboost is not available. Install xgboost (and libomp on macOS if needed)."
        ) from exc

    X_train, X_val, y_train, y_val = _load_da_splits(base_dir)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        early_stopping_rounds=early_stopping_rounds,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = pd.Series(model.predict(X_val), index=X_val.index)
    mae = float(mean_absolute_error(y_val, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    baseline_mae = _naive_baseline_mae(X_val, y_val)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    _plot_top_feature_importance(model, feature_names=list(X_train.columns), out_path=importance_out, top_n=20)

    return {
        "mae": mae,
        "rmse": rmse,
        "baseline_mae_24h": baseline_mae,
        "rows_train": float(len(X_train)),
        "rows_val": float(len(X_val)),
    }


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DA-price XGBoost model on prepared bundles.")
    p.add_argument("--base-dir", default="data/model_input", help="Bundle base directory.")
    p.add_argument(
        "--model-out",
        default="models/checkpoints/xgboost_da_v1.joblib",
        help="Output path for trained model.",
    )
    p.add_argument(
        "--importance-out",
        default="data/reports/model_training/xgboost_da_top20_feature_importance.png",
        help="Output path for feature-importance figure.",
    )
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    return p


def main() -> None:
    args = _build_cli().parse_args()
    base_dir = _resolve_base_dir(args.base_dir)
    metrics = train_and_evaluate(
        base_dir=base_dir,
        model_out=Path(args.model_out),
        importance_out=Path(args.importance_out),
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    print("[OK] XGBoost DA training finished.")
    print(f"- Base dir: {base_dir}")
    print(f"- Train rows: {int(metrics['rows_train'])}")
    print(f"- Val rows: {int(metrics['rows_val'])}")
    print(f"- MAE (val): {metrics['mae']:.4f}")
    print(f"- RMSE (val): {metrics['rmse']:.4f}")
    print(f"- Naive MAE 24h (val): {metrics['baseline_mae_24h']:.4f}")


if __name__ == "__main__":
    main()
