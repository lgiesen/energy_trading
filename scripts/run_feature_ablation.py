"""Run feature-group ablation with Purged CV for causal model selection.

This script compares baseline-vs-ablation variants on the prepared bundle and
reports MAE/RMSE so feature additions/removals are evidence-based.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from energy_trading.models.cv import PurgedTimeSeriesSplit
from energy_trading.models.prepare_ml_bundles import BundleName, load_processed_data


def _resolve_target(bundle: BundleName, requested: str | None, y_cols: list[str]) -> str:
    if requested:
        if requested not in y_cols:
            raise KeyError(f"Requested target '{requested}' not found in available targets: {y_cols}")
        return requested
    default = "target_da_price_h1" if bundle == "da" else "target_afrr_activation_price_vwap_pos_h1"
    if default not in y_cols:
        raise KeyError(f"Default target '{default}' not found in targets: {y_cols}")
    return default


def _feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "cross_border": [
            c
            for c in feature_cols
            if re.search(r"^(da_spread_(de_at|de_fr|de_nl)|neighbor_spread_avg)", c)
        ],
        "hydro_pumped": [
            c
            for c in feature_cols
            if ("hydro_pumped" in c or c.startswith("generation_hydro_pumped_storage_mw"))
        ],
        "load_error": [c for c in feature_cols if c.startswith("load_error_da")],
        "picasso_flow": [c for c in feature_cols if c.startswith("picasso_flow_rate")],
        "orderbook_depth": [
            c
            for c in feature_cols
            if (
                c.startswith("afrr_activation_offered_mw_")
                or c.startswith("afrr_capacity_offered_mw_")
            )
        ],
    }
    return {k: sorted(set(v)) for k, v in groups.items() if v}


def _fit_eval_cv(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    test_size: int,
    gap_hours: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> tuple[float, float, float]:
    splitter = PurgedTimeSeriesSplit(
        n_splits=n_splits,
        test_size=test_size,
        gap_hours=gap_hours,
        frequency="1h",
    )
    fold_mae: list[float] = []
    fold_rmse: list[float] = []
    fold_spread_capture: list[float] = []

    for tr_idx, va_idx in splitter.split(X):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            device="cpu",
            random_state=42,
            n_jobs=4,
        )
        model.fit(X_tr, y_tr, verbose=False)
        pred = model.predict(X_va)
        fold_mae.append(float(mean_absolute_error(y_va, pred)))
        fold_rmse.append(float(np.sqrt(mean_squared_error(y_va, pred))))
        spread_capture = np.nan
        if "da_price_pit" in X_va.columns:
            da_ref = pd.to_numeric(X_va["da_price_pit"], errors="coerce")
            valid = da_ref.notna() & y_va.notna()
            if bool(valid.any()):
                spread_capture = float(
                    np.mean(
                        np.sign(pred[valid.to_numpy()] - da_ref.loc[valid].to_numpy())
                        * (y_va.loc[valid].to_numpy() - da_ref.loc[valid].to_numpy())
                    )
                )
        fold_spread_capture.append(spread_capture)

    return (
        float(np.mean(fold_mae)),
        float(np.mean(fold_rmse)),
        float(np.nanmean(fold_spread_capture)),
    )


def _fit_eval_holdout(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> tuple[float, float]:
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        device="cpu",
        random_state=42,
        n_jobs=4,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    pred = model.predict(X_val)
    return (
        float(mean_absolute_error(y_val, pred)),
        float(np.sqrt(mean_squared_error(y_val, pred))),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Ablation study for selected feature groups.")
    p.add_argument("--base-dir", default="data/model_input")
    p.add_argument("--bundle", choices=["da", "afrr"], default="afrr")
    p.add_argument("--target-col", default=None)
    p.add_argument("--out-csv", default="data/reports/feature_ablation_report.csv")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.08)
    p.add_argument("--cv-n-splits", type=int, default=3)
    p.add_argument("--cv-test-size", type=int, default=24 * 7)
    p.add_argument("--cv-gap-hours", type=int, default=72)
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    cfg_path = base_dir / "feature_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    bundle = args.bundle
    bundle_cfg = cfg["bundles"][bundle]

    X_train, y_train_df = load_processed_data(bundle=bundle, split="train", base_dir=base_dir)
    X_val, y_val_df = load_processed_data(bundle=bundle, split="val", base_dir=base_dir)

    target = _resolve_target(bundle, args.target_col, bundle_cfg["targets"])
    y_train = pd.to_numeric(y_train_df[target], errors="coerce")
    y_val = pd.to_numeric(y_val_df[target], errors="coerce")

    train_mask = y_train.notna()
    val_mask = y_val.notna()
    X_train = X_train.loc[train_mask].copy()
    y_train = y_train.loc[train_mask].copy()
    X_val = X_val.loc[val_mask].copy()
    y_val = y_val.loc[val_mask].copy()

    all_features = list(X_train.columns)
    groups = _feature_groups(all_features)

    variants: list[tuple[str, list[str]]] = [("base_all", all_features)]
    for g_name, g_cols in groups.items():
        reduced = [c for c in all_features if c not in set(g_cols)]
        variants.append((f"minus_{g_name}", reduced))

    rows: list[dict[str, object]] = []
    for name, cols in variants:
        Xtr = X_train[cols].copy()
        Xva = X_val[cols].copy()
        cv_mae, cv_rmse, cv_spread_capture = _fit_eval_cv(
            Xtr,
            y_train,
            n_splits=args.cv_n_splits,
            test_size=args.cv_test_size,
            gap_hours=args.cv_gap_hours,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
        )
        val_mae, val_rmse = _fit_eval_holdout(
            Xtr,
            y_train,
            Xva,
            y_val,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
        )
        rows.append(
            {
                "variant": name,
                "n_features": len(cols),
                "cv_mae": cv_mae,
                "cv_rmse": cv_rmse,
                "cv_spread_capture": cv_spread_capture,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            }
        )
        print(
            f"[ablation] {name}: n_features={len(cols)} "
            f"cv_mae={cv_mae:.4f} cv_rmse={cv_rmse:.4f} "
            f"cv_spread_capture={cv_spread_capture:.4f} "
            f"val_mae={val_mae:.4f} val_rmse={val_rmse:.4f}"
        )

    rep = pd.DataFrame(rows).sort_values(["cv_mae", "val_mae"], ascending=[True, True]).reset_index(drop=True)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out, index=False)
    print(f"[OK] Wrote ablation report: {out}")


if __name__ == "__main__":
    main()
