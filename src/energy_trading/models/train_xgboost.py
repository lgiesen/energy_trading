"""Train DA-price XGBoost baseline on prepared ML bundles.

This script trains on the DA bundle (train split) and validates on the
chronological val split with early stopping to reduce overfitting.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib

# Keep matplotlib cache inside project for restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "data" / ".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

from src.energy_trading.models.prepare_ml_bundles import BundleName, load_processed_data
from src.energy_trading.models.cv import PurgedTimeSeriesSplit


def _resolve_base_dir(preferred: str | Path) -> Path:
    p = Path(preferred)
    if p.exists():
        return p
    raise FileNotFoundError(
        f"Bundle directory not found: {p}."
    )


def _default_target_for_bundle(bundle: BundleName) -> str:
    if bundle == "da":
        return "target_da_price_h1"
    if bundle == "afrr":
        return "target_afrr_activation_price_vwap_pos_h1"
    raise KeyError(f"Unsupported bundle: {bundle}")


def _default_afrr_targets() -> list[str]:
    return [
        "target_afrr_activation_price_vwap_pos_h1",
        "target_afrr_activation_price_vwap_neg_h1",
        "target_afrr_rate_h1",
        "target_afrr_capacity_price_pos_h1",
        "target_afrr_capacity_price_neg_h1",
    ]


def _resolve_targets(
    bundle: BundleName,
    y_cols: list[str],
    target_col: str | None,
) -> list[str]:
    if target_col:
        if target_col not in y_cols:
            raise KeyError(f"Requested target column '{target_col}' not found in y columns.")
        return [target_col]
    if bundle == "da":
        default = _default_target_for_bundle(bundle)
        if default not in y_cols:
            raise KeyError(f"Default DA target '{default}' missing.")
        return [default]
    if bundle == "afrr":
        targets = [t for t in _default_afrr_targets() if t in y_cols]
        if not targets:
            raise KeyError("No aFRR target columns available for training.")
        return targets
    raise KeyError(f"Unsupported bundle: {bundle}")


def _pred_column_names_for_target(target_col: str) -> list[str]:
    mapping = {
        "target_da_price_h1": ["pred_da_price"],
        "target_afrr_activation_price_vwap_pos_h1": ["pred_afrr_activation_price_pos"],
        "target_afrr_activation_price_vwap_neg_h1": ["pred_afrr_activation_price_neg"],
        "target_afrr_capacity_price_pos_h1": ["pred_afrr_capacity_price_pos"],
        "target_afrr_capacity_price_neg_h1": ["pred_afrr_capacity_price_neg"],
        # Single rate target is reused for both directions by convention.
        "target_afrr_rate_h1": ["pred_afrr_activation_rate_pos", "pred_afrr_activation_rate_neg"],
    }
    return mapping.get(target_col, [f"pred_{target_col}"])


def _source_series_name_for_target(target_col: str) -> str | None:
    mapping = {
        "target_da_price_h1": "da_price",
        "target_afrr_activation_price_vwap_pos_h1": "afrr_activation_price_vwap_pos",
        "target_afrr_activation_price_vwap_neg_h1": "afrr_activation_price_vwap_neg",
        "target_afrr_capacity_price_pos_h1": "afrr_capacity_price_pos",
        "target_afrr_capacity_price_neg_h1": "afrr_capacity_price_neg",
        "target_afrr_rate_h1": "afrr_activation_rate",
    }
    return mapping.get(target_col)


def _target_for_pred_column(pred_col: str) -> str | None:
    mapping = {
        "pred_da_price": "target_da_price_h1",
        "pred_afrr_activation_price_pos": "target_afrr_activation_price_vwap_pos_h1",
        "pred_afrr_activation_price_neg": "target_afrr_activation_price_vwap_neg_h1",
        "pred_afrr_capacity_price_pos": "target_afrr_capacity_price_pos_h1",
        "pred_afrr_capacity_price_neg": "target_afrr_capacity_price_neg_h1",
        "pred_afrr_activation_rate_pos": "target_afrr_rate_h1",
        "pred_afrr_activation_rate_neg": "target_afrr_rate_h1",
    }
    return mapping.get(pred_col)


def _lag_columns_for_source(features: list[str], source_name: str) -> dict[int, str]:
    pattern = re.compile(rf"^{re.escape(source_name)}_lag_(\d+)h$")
    out: dict[int, str] = {}
    for col in features:
        m = pattern.match(col)
        if m:
            out[int(m.group(1))] = col
    return out


def calculate_forecast_decay(
    *,
    pred_long: pd.DataFrame,
    true_h1: pd.Series,
    horizon_hours: int,
) -> pd.DataFrame:
    """Calculate MAE by lead-time for a long-format prediction warehouse table."""
    if pred_long.empty:
        return pd.DataFrame(columns=["lead_time_h", "n", "mae"])

    truth_map: dict[int, pd.Series] = {}
    base = pd.to_numeric(true_h1, errors="coerce")
    for lead in range(1, horizon_hours + 1):
        truth_map[lead] = base.shift(-(lead - 1)).reset_index(drop=True)

    pred = pd.to_numeric(pred_long["predicted_value"], errors="coerce")
    lead = pd.to_numeric(pred_long["lead_time_h"], errors="coerce").astype("Int64")

    rows: list[dict[str, float]] = []
    for l in sorted(set(int(v) for v in lead.dropna().unique())):
        idx = lead == l
        if not bool(idx.any()):
            continue
        truth = truth_map.get(l)
        if truth is None:
            continue
        pred_part = pred[idx].reset_index(drop=True)
        aligned_truth = truth.iloc[: len(pred_part)]
        mask = aligned_truth.notna() & pred_part.notna()
        n = int(mask.sum())
        mae = float(mean_absolute_error(aligned_truth[mask], pred_part[mask])) if n > 0 else np.nan
        rows.append({"lead_time_h": float(l), "n": float(n), "mae": mae})
    return pd.DataFrame(rows).sort_values("lead_time_h").reset_index(drop=True)


def _load_splits(
    base_dir: Path,
    bundle: BundleName,
    target_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    X_train, y_train_df = load_processed_data(bundle=bundle, split="train", base_dir=base_dir)
    X_val, y_val_df = load_processed_data(bundle=bundle, split="val", base_dir=base_dir)

    chosen_target = target_col or _default_target_for_bundle(bundle)
    if chosen_target not in y_train_df.columns or chosen_target not in y_val_df.columns:
        raise KeyError(
            f"Expected target column '{chosen_target}' for bundle '{bundle}' in processed bundles."
        )

    y_train = pd.to_numeric(y_train_df[chosen_target], errors="coerce")
    y_val = pd.to_numeric(y_val_df[chosen_target], errors="coerce")

    train_mask = y_train.notna()
    val_mask = y_val.notna()
    X_train = X_train.loc[train_mask].copy()
    y_train = y_train.loc[train_mask].copy()
    X_val = X_val.loc[val_mask].copy()
    y_val = y_val.loc[val_mask].copy()

    return X_train, X_val, y_train, y_val, chosen_target


def _load_bundle_split_df(
    base_dir: Path,
    bundle: BundleName,
    split: str,
) -> tuple[pd.DataFrame, dict]:
    cfg_path = base_dir / "feature_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing feature_config.json in {base_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    bcfg = cfg["bundles"][bundle]
    split_path = Path(bcfg["files"][split])
    if not split_path.exists():
        raise FileNotFoundError(f"Missing bundle split file: {split_path}")
    df = pd.read_parquet(split_path)
    if "timestamp_utc" not in df.columns:
        raise KeyError(f"`timestamp_utc` missing in split file: {split_path}")
    return df, bcfg


def _naive_baseline_mae(bundle: BundleName, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    lag_col = "da_price_lag_24h" if bundle == "da" else "afrr_activation_price_vwap_pos_lag_24h"
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


def _plot_leadtime_mae_points(decay_df: pd.DataFrame, out_path: Path) -> None:
    """Plot MAE for lead times 1h, 24h, 48h (if available)."""
    if decay_df.empty or "lead_time_h" not in decay_df.columns or "mae" not in decay_df.columns:
        return
    pts = decay_df.copy()
    pts["lead_time_h"] = pd.to_numeric(pts["lead_time_h"], errors="coerce").astype("Int64")
    pts["mae"] = pd.to_numeric(pts["mae"], errors="coerce")
    pts = pts[pts["lead_time_h"].isin([1, 24, 48]) & pts["mae"].notna()].copy()
    if pts.empty:
        return

    pts = pts.sort_values("lead_time_h")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(pts["lead_time_h"].astype(int), pts["mae"], marker="o", linewidth=2, color="#2C7FB8")
    for _, r in pts.iterrows():
        plt.text(int(r["lead_time_h"]), float(r["mae"]), f"{float(r['mae']):.2f}", fontsize=9, va="bottom")
    plt.xticks([1, 24, 48])
    plt.xlabel("Lead Time (h)")
    plt.ylabel("MAE")
    plt.title("MAE by Lead Time (1h / 24h / 48h)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _build_horizon_matrix(y: pd.Series, horizon_hours: int) -> pd.DataFrame:
    """Build direct multi-output target matrix [t+1, ..., t+h]."""
    cols = {f"t_plus_{h}h": pd.to_numeric(y.shift(-(h - 1)), errors="coerce") for h in range(1, horizon_hours + 1)}
    return pd.DataFrame(cols, index=y.index)


def _unwrap_single_xgb(model):
    """Return a single XGBRegressor for importance/SHAP from potentially wrapped model."""
    if hasattr(model, "estimators_") and getattr(model, "estimators_", None):
        return model.estimators_[0]
    return model


def calculate_feature_importance(
    model,
    X_train: pd.DataFrame,
    *,
    report_out: Path,
    shap_plot_out: Path,
    shap_sample_size: int = 5000,
) -> pd.DataFrame:
    """Build empirical importance evidence (Gain + SHAP) for thesis reporting."""
    report_out.parent.mkdir(parents=True, exist_ok=True)
    shap_plot_out.parent.mkdir(parents=True, exist_ok=True)

    features = list(X_train.columns)
    gain_map = model.get_booster().get_score(importance_type="gain")
    gain_series = pd.Series(0.0, index=features, dtype="float64")
    for k, v in gain_map.items():
        # Booster keys are typically f0, f1, ... if no explicit names persisted.
        if k in gain_series.index:
            gain_series.loc[k] = float(v)
        elif k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            if 0 <= idx < len(features):
                gain_series.iloc[idx] = float(v)

    shap_mean_abs = pd.Series(np.nan, index=features, dtype="float64")
    try:
        import shap  # type: ignore

        if len(X_train) > shap_sample_size:
            X_shap = X_train.sample(n=shap_sample_size, random_state=42).copy()
        else:
            X_shap = X_train.copy()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 2 and shap_values.shape[1] == len(features):
            shap_mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=features, dtype="float64")

            plt.figure(figsize=(10, 7))
            shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(shap_plot_out, dpi=160)
            plt.close()
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] SHAP importance skipped: {exc}")

    report = pd.DataFrame(
        {
            "feature": features,
            "xgboost_gain": gain_series.values,
            "shap_mean_abs": shap_mean_abs.values,
        }
    )
    report["gain_rank"] = report["xgboost_gain"].rank(method="min", ascending=False).astype("Int64")
    report["shap_rank"] = report["shap_mean_abs"].rank(method="min", ascending=False).astype("Int64")
    report = report.sort_values(["gain_rank", "shap_rank"], na_position="last").reset_index(drop=True)
    report.to_csv(report_out, index=False)
    return report


def _assert_cuda_available_or_fail(*, device: str, require_cuda: bool) -> None:
    if device != "cuda":
        if require_cuda:
            raise RuntimeError("CUDA is required, but training device is set to CPU.")
        return

    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("CUDA requested but xgboost is unavailable.") from exc

    try:
        X_probe = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y_probe = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        probe = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1,
            max_depth=1,
            learning_rate=0.1,
            tree_method="hist",
            device="cuda",
            n_jobs=1,
        )
        probe.fit(X_probe, y_probe, verbose=False)
    except Exception as exc:
        if require_cuda:
            raise RuntimeError(
                "CUDA training required but not available. "
                "Install GPU-enabled XGBoost/CUDA runtime or run with --allow-cpu."
            ) from exc
        print(f"[WARN] CUDA probe failed; continuing on CPU fallback: {exc}")


def run_purged_cv_with_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int,
    test_size: int,
    gap_hours: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    device: str,
) -> dict[str, float]:
    """Run purged inner CV with strict fold-local fitting (leak-safe).

    Current implementation trains XGBoost directly per fold (no scaling step),
    which is appropriate for tree models and avoids any chance of global
    preprocessor leakage.
    """
    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "xgboost is not available. Install xgboost (and libomp on macOS if needed)."
        ) from exc

    splitter = PurgedTimeSeriesSplit(
        n_splits=n_splits,
        test_size=test_size,
        gap_hours=gap_hours,
        frequency="1h",
        min_train_size=500,
    )

    fold_mae: list[float] = []
    fold_rmse: list[float] = []
    fold_spread_capture: list[float] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X_train), start=1):
        X_tr = X_train.iloc[tr_idx].copy()
        X_va = X_train.iloc[va_idx].copy()
        y_tr = y_train.iloc[tr_idx].copy()
        y_va = y_train.iloc[va_idx].copy()

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method="hist",
            device=device,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr, verbose=False)
        pred = model.predict(X_va)
        mae = float(mean_absolute_error(y_va, pred))
        rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
        spread_capture = np.nan
        if "da_price_pit" in X_va.columns:
            da_ref = pd.to_numeric(X_va["da_price_pit"], errors="coerce")
            valid = da_ref.notna() & y_va.notna()
            if bool(valid.any()):
                # Positive values mean predicted spread direction aligns with
                # realized spread direction/magnitude more often.
                spread_capture = float(
                    np.mean(
                        np.sign(pred[valid.to_numpy()] - da_ref.loc[valid].to_numpy())
                        * (y_va.loc[valid].to_numpy() - da_ref.loc[valid].to_numpy())
                    )
                )
        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_spread_capture.append(spread_capture)
        print(
            f"[CV] fold={fold_idx} train={len(tr_idx)} val={len(va_idx)} "
            f"mae={mae:.4f} rmse={rmse:.4f} spread_capture={spread_capture:.4f}"
        )

    return {
        "cv_mae_mean": float(np.mean(fold_mae)) if fold_mae else np.nan,
        "cv_mae_std": float(np.std(fold_mae)) if fold_mae else np.nan,
        "cv_rmse_mean": float(np.mean(fold_rmse)) if fold_rmse else np.nan,
        "cv_rmse_std": float(np.std(fold_rmse)) if fold_rmse else np.nan,
        "cv_spread_capture_mean": float(np.nanmean(fold_spread_capture)) if fold_spread_capture else np.nan,
        "cv_spread_capture_std": float(np.nanstd(fold_spread_capture)) if fold_spread_capture else np.nan,
        "cv_folds": float(len(fold_mae)),
    }


def train_and_evaluate(
    base_dir: Path,
    bundle: BundleName,
    target_col: str | None,
    model_out: Path,
    importance_out: Path,
    importance_report_out: Path,
    shap_summary_out: Path,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 50,
    run_cv: bool = False,
    cv_n_splits: int = 3,
    cv_test_size: int = 24 * 28,
    cv_gap_hours: int = 72,
    device: str = "cuda",
    require_cuda: bool = True,
    horizon_hours: int = 48,
) -> tuple[dict[str, float], dict[str, object], list[str]]:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "xgboost is not available. Install xgboost (and libomp on macOS if needed)."
        ) from exc

    _assert_cuda_available_or_fail(device=device, require_cuda=require_cuda)

    X_train_df, y_train_df = load_processed_data(bundle=bundle, split="train", base_dir=base_dir)
    X_val_df, y_val_df = load_processed_data(bundle=bundle, split="val", base_dir=base_dir)
    target_cols = _resolve_targets(bundle, list(y_train_df.columns), target_col)
    primary_target = target_cols[0]

    # Train on rows where all selected targets are available.
    train_mask = y_train_df[target_cols].notna().all(axis=1)
    val_mask = y_val_df[target_cols].notna().all(axis=1)
    X_train = X_train_df.loc[train_mask].copy()
    X_val = X_val_df.loc[val_mask].copy()
    y_train_m = y_train_df.loc[train_mask, target_cols].copy()
    y_val_m = y_val_df.loc[val_mask, target_cols].copy()

    # CV remains on primary target only.
    cv_metrics: dict[str, float] = {}
    if run_cv:
        cv_metrics = run_purged_cv_with_pipeline(
            X_train=X_train,
            y_train=pd.to_numeric(y_train_m[primary_target], errors="coerce"),
            n_splits=cv_n_splits,
            test_size=cv_test_size,
            gap_hours=cv_gap_hours,
            n_estimators=max(200, n_estimators // 2),
            max_depth=max_depth,
            learning_rate=learning_rate,
            device=device,
        )

    models_by_target: dict[str, object] = {}
    val_pred_df = pd.DataFrame(index=X_val.index)
    per_target_metrics: dict[str, dict[str, float]] = {}
    primary_X_val_h = None
    primary_y_val_lead1 = None
    for idx, tgt in enumerate(target_cols):
        base_model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method="hist",
            device=device,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42 + idx,
            n_jobs=-1,
        )
        model = MultiOutputRegressor(base_model)

        y_tr_base = pd.to_numeric(y_train_m[tgt], errors="coerce")
        y_va_base = pd.to_numeric(y_val_m[tgt], errors="coerce")
        Y_tr = _build_horizon_matrix(y_tr_base, horizon_hours=horizon_hours)
        Y_va = _build_horizon_matrix(y_va_base, horizon_hours=horizon_hours)

        tr_mask = Y_tr.notna().all(axis=1)
        va_mask = Y_va.notna().all(axis=1)
        X_tr_h = X_train.loc[tr_mask].copy()
        X_va_h = X_val.loc[va_mask].copy()
        Y_tr_h = Y_tr.loc[tr_mask].copy()
        Y_va_h = Y_va.loc[va_mask].copy()
        if X_tr_h.empty or X_va_h.empty:
            raise ValueError(f"Not enough non-null rows for direct multi-output target '{tgt}'.")

        model.fit(X_tr_h, Y_tr_h)
        pred_h = np.asarray(model.predict(X_va_h), dtype=float)
        lead1_pred = pd.Series(pred_h[:, 0], index=X_va_h.index)
        val_pred_df[tgt] = np.nan
        val_pred_df.loc[X_va_h.index, tgt] = lead1_pred

        lead24_mae = np.nan
        lead48_mae = np.nan
        if horizon_hours >= 24:
            lead24_mae = float(mean_absolute_error(Y_va_h.iloc[:, 23], pred_h[:, 23]))
        if horizon_hours >= 48:
            lead48_mae = float(mean_absolute_error(Y_va_h.iloc[:, 47], pred_h[:, 47]))
        per_target_metrics[tgt] = {
            "mae": float(mean_absolute_error(Y_va_h.iloc[:, 0], pred_h[:, 0])),
            "rmse": float(np.sqrt(mean_squared_error(Y_va_h.iloc[:, 0], pred_h[:, 0]))),
            "mae_h24": lead24_mae,
            "mae_h48": lead48_mae,
            "rows_train_horizon": float(len(X_tr_h)),
            "rows_val_horizon": float(len(X_va_h)),
        }
        models_by_target[tgt] = model
        if tgt == primary_target:
            primary_X_val_h = X_va_h
            primary_y_val_lead1 = pd.to_numeric(Y_va_h.iloc[:, 0], errors="coerce")

    if primary_X_val_h is None or primary_y_val_lead1 is None:
        raise ValueError("Primary target horizon-aligned validation set is empty.")
    y_val_primary = primary_y_val_lead1
    y_pred_primary = pd.to_numeric(val_pred_df.loc[primary_X_val_h.index, primary_target], errors="coerce")
    mae = float(mean_absolute_error(y_val_primary, y_pred_primary))
    rmse = float(np.sqrt(mean_squared_error(y_val_primary, y_pred_primary)))
    baseline_mae = _naive_baseline_mae(bundle, primary_X_val_h, y_val_primary)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    model_payload = models_by_target if len(models_by_target) > 1 else models_by_target[primary_target]
    joblib.dump(model_payload, model_out)

    primary_model = _unwrap_single_xgb(models_by_target[primary_target])
    _plot_top_feature_importance(primary_model, feature_names=list(X_train.columns), out_path=importance_out, top_n=20)
    importance_report = calculate_feature_importance(
        primary_model,
        X_train,
        report_out=importance_report_out,
        shap_plot_out=shap_summary_out,
    )

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "baseline_mae_24h": baseline_mae,
        "rows_train": float(len(X_train)),
        "rows_val": float(len(X_val)),
        "n_importance_rows": float(len(importance_report)),
        "target_col": primary_target,
        "target_cols": target_cols,
        "multioutput_horizon_hours": float(horizon_hours),
        "per_target_metrics": per_target_metrics,
    }
    metrics.update(cv_metrics)
    return metrics, models_by_target, target_cols


def _run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _resolve_run_dir(run_dir: str | Path | None) -> Path:
    if run_dir is None or str(run_dir).strip() == "":
        return Path("artifacts/model_runs") / _run_id_now()
    return Path(run_dir)


def _predict_split_frame(
    base_dir: Path,
    bundle: BundleName,
    split: str,
    models_by_target: dict[str, object],
    target_cols: list[str],
) -> pd.DataFrame:
    split_df, bcfg = _load_bundle_split_df(base_dir=base_dir, bundle=bundle, split=split)
    X = split_df[bcfg["features"]].copy()
    out = pd.DataFrame({"timestamp_utc": pd.to_datetime(split_df["timestamp_utc"], utc=True, errors="coerce")})

    # Keep stable canonical schema for simulation auto-load.
    if bundle == "da":
        out["pred_da_price"] = np.nan
    else:
        for c in [
            "pred_afrr_capacity_price_pos",
            "pred_afrr_capacity_price_neg",
            "pred_afrr_activation_price_pos",
            "pred_afrr_activation_price_neg",
            "pred_afrr_activation_rate_pos",
            "pred_afrr_activation_rate_neg",
        ]:
            out[c] = np.nan

    for tgt in target_cols:
        model = models_by_target[tgt]
        pred_h = np.asarray(model.predict(X), dtype=float)
        if pred_h.ndim == 1:
            pred = pd.to_numeric(pd.Series(pred_h, index=split_df.index), errors="coerce")
        else:
            pred = pd.to_numeric(pd.Series(pred_h[:, 0], index=split_df.index), errors="coerce")
        for pred_col in _pred_column_names_for_target(tgt):
            out[pred_col] = pred.values
    return out


def _predict_split_long_multistep(
    *,
    base_dir: Path,
    bundle: BundleName,
    split: str,
    models_by_target: dict[str, object],
    target_cols: list[str],
    horizon_hours: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Generate direct multi-output predictions and return long-format tables.

    Output per prediction column:
    - snapshot_time_utc
    - target_time_utc
    - lead_time_h
    - predicted_value
    """
    split_df, bcfg = _load_bundle_split_df(base_dir=base_dir, bundle=bundle, split=split)
    features = list(bcfg["features"])
    X = split_df[features].copy()
    snapshots = pd.to_datetime(split_df["timestamp_utc"], utc=True, errors="coerce")

    pred_frames: dict[str, list[pd.DataFrame]] = {}

    # Keep true h+1 series per target for decay metric.
    true_h1_by_target: dict[str, pd.Series] = {}
    for tgt in target_cols:
        if tgt in split_df.columns:
            true_h1_by_target[tgt] = pd.to_numeric(split_df[tgt], errors="coerce")

    for tgt in target_cols:
        model = models_by_target[tgt]
        pred_h = np.asarray(model.predict(X), dtype=float)
        if pred_h.ndim == 1:
            pred_h = pred_h.reshape(-1, 1)
        use_h = min(horizon_hours, pred_h.shape[1])
        for lead in range(1, use_h + 1):
            pred = pd.Series(pred_h[:, lead - 1], index=split_df.index)
            target_time = snapshots + pd.to_timedelta(lead, unit="h")
            for pred_col in _pred_column_names_for_target(tgt):
                long_df = pd.DataFrame(
                    {
                        "snapshot_time_utc": snapshots,
                        "target_time_utc": target_time,
                        "lead_time_h": lead,
                        "predicted_value": pd.to_numeric(pred, errors="coerce").values,
                    }
                )
                pred_frames.setdefault(pred_col, []).append(long_df)

    out_long: dict[str, pd.DataFrame] = {}
    out_decay: dict[str, pd.DataFrame] = {}
    for pred_col, parts in pred_frames.items():
        long_df = pd.concat(parts, ignore_index=True)
        long_df = long_df.sort_values(["snapshot_time_utc", "lead_time_h"]).reset_index(drop=True)
        out_long[pred_col] = long_df

        tgt = _target_for_pred_column(pred_col)
        true_h1 = true_h1_by_target.get(tgt or "", pd.Series(index=split_df.index, dtype="float64"))
        out_decay[pred_col] = calculate_forecast_decay(
            pred_long=long_df,
            true_h1=true_h1,
            horizon_hours=horizon_hours,
        )
    return out_long, out_decay


def _write_bundle_manifest_fragment(
    *,
    bundle: BundleName,
    run_dir: Path,
    model_path: Path,
    metrics_path: Path,
    prediction_paths: dict[str, Path],
    prediction_long_paths: dict[str, dict[str, Path]],
    target_cols: list[str],
    metrics: dict[str, float],
) -> dict:
    pred_cols: list[str] = []
    for t in target_cols:
        pred_cols.extend(_pred_column_names_for_target(t))
    pred_cols = list(dict.fromkeys(pred_cols))
    return {
        "bundle": bundle,
        "model_path": str(model_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "predictions": {k: str(v.resolve()) for k, v in prediction_paths.items()},
        "predictions_long": {
            split: {k: str(v.resolve()) for k, v in col_paths.items()}
            for split, col_paths in prediction_long_paths.items()
        },
        "prediction_columns": pred_cols,
        "target_columns": target_cols,
        "run_dir": str(run_dir.resolve()),
        "primary_target": metrics.get("target_col"),
    }


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train XGBoost model on prepared DA/aFRR bundles.")
    p.add_argument("--base-dir", default="data/model_input", help="Bundle base directory.")
    p.add_argument("--bundle", choices=["da", "afrr"], default="da", help="Bundle to train on.")
    p.add_argument("--target-col", default="", help="Optional explicit target column override.")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="XGBoost device.")
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback. If omitted, training fails when CUDA is unavailable.",
    )
    p.add_argument(
        "--model-out",
        default="",
        help="Output path for trained model.",
    )
    p.add_argument(
        "--importance-out",
        default="",
        help="Output path for feature-importance figure.",
    )
    p.add_argument(
        "--importance-report-out",
        default="",
        help="Output CSV path for gain + SHAP feature importance report.",
    )
    p.add_argument(
        "--shap-summary-out",
        default="",
        help="Output path for SHAP summary plot.",
    )
    p.add_argument(
        "--run-dir",
        default="",
        help="Optional run directory. If empty, a timestamped directory under artifacts/model_runs is created.",
    )
    p.add_argument(
        "--export-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export prediction parquet files for configured splits.",
    )
    p.add_argument(
        "--prediction-splits",
        default="val,test",
        help="Comma-separated splits to export predictions for (default: val,test).",
    )
    p.add_argument(
        "--export-predictions-long",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export direct multi-output forecast warehouse in long format.",
    )
    p.add_argument(
        "--forecast-horizon-hours",
        type=int,
        default=48,
        help="Forecast horizon for long-format prediction warehouse.",
    )
    p.add_argument(
        "--metrics-json-out",
        default="",
        help="Optional explicit metrics json output path.",
    )
    p.add_argument(
        "--manifest-fragment-out",
        default="",
        help="Optional path to write bundle manifest fragment JSON.",
    )
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument(
        "--run-cv",
        action="store_true",
        help="Run leakage-safe inner CV with fold-fitted preprocessing pipeline.",
    )
    p.add_argument("--cv-n-splits", type=int, default=3)
    p.add_argument("--cv-test-size", type=int, default=24 * 28, help="Validation rows per fold (hourly rows).")
    p.add_argument("--cv-gap-hours", type=int, default=72)
    return p


def main() -> None:
    args = _build_cli().parse_args()
    base_dir = _resolve_base_dir(args.base_dir)
    run_dir = _resolve_run_dir(args.run_dir)
    bundle_run_dir = run_dir
    models_dir = bundle_run_dir / "models"
    metrics_dir = bundle_run_dir / "metrics"
    pred_dir = bundle_run_dir / "predictions"
    report_dir = bundle_run_dir / "reports"
    for d in (models_dir, metrics_dir, pred_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)

    target_hint = (args.target_col or "").strip()
    if target_hint:
        target_tag = (
            target_hint.replace("target_", "")
            .replace("_h1", "")
            .replace("/", "_")
        )
    else:
        target_tag = ""

    file_tag = f"{args.bundle}_{target_tag}" if target_tag else args.bundle

    default_model_out = models_dir / f"{file_tag}_model.joblib"
    default_importance_out = report_dir / f"{file_tag}_top20_feature_importance.png"
    default_importance_report_out = report_dir / f"{file_tag}_importance_report.csv"
    default_shap_summary_out = report_dir / f"{file_tag}_shap_summary.png"
    default_metrics_json_out = metrics_dir / f"{file_tag}_metrics.json"
    default_manifest_fragment_out = bundle_run_dir / f"{file_tag}_manifest_fragment.json"

    model_out = Path(args.model_out) if args.model_out else default_model_out
    importance_out = Path(args.importance_out) if args.importance_out else default_importance_out
    importance_report_out = Path(args.importance_report_out) if args.importance_report_out else default_importance_report_out
    shap_summary_out = Path(args.shap_summary_out) if args.shap_summary_out else default_shap_summary_out
    metrics_json_out = Path(args.metrics_json_out) if args.metrics_json_out else default_metrics_json_out
    manifest_fragment_out = (
        Path(args.manifest_fragment_out) if args.manifest_fragment_out else default_manifest_fragment_out
    )

    metrics, models_by_target, target_cols = train_and_evaluate(
        base_dir=base_dir,
        bundle=args.bundle,
        target_col=(args.target_col.strip() or None),
        model_out=model_out,
        importance_out=importance_out,
        importance_report_out=importance_report_out,
        shap_summary_out=shap_summary_out,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
        run_cv=args.run_cv,
        cv_n_splits=args.cv_n_splits,
        cv_test_size=args.cv_test_size,
        cv_gap_hours=args.cv_gap_hours,
        device=args.device,
        require_cuda=not args.allow_cpu,
        horizon_hours=args.forecast_horizon_hours,
    )

    metrics_json_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_json_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    prediction_paths: dict[str, Path] = {}
    prediction_long_paths: dict[str, dict[str, Path]] = {}
    splits = [s.strip() for s in args.prediction_splits.split(",") if s.strip()]
    if args.export_predictions:
        for split in splits:
            pred_df = _predict_split_frame(
                base_dir=base_dir,
                bundle=args.bundle,
                split=split,
                models_by_target=models_by_target,
                target_cols=target_cols,
            )
            out_path = pred_dir / f"{file_tag}_{split}.parquet"
            pred_df.to_parquet(out_path, index=False)
            prediction_paths[split] = out_path

            if args.export_predictions_long:
                long_by_col, decay_by_col = _predict_split_long_multistep(
                    base_dir=base_dir,
                    bundle=args.bundle,
                    split=split,
                    models_by_target=models_by_target,
                    target_cols=target_cols,
                    horizon_hours=args.forecast_horizon_hours,
                )
                prediction_long_paths[split] = {}
                for pred_col, long_df in long_by_col.items():
                    long_path = pred_dir / f"{file_tag}_{split}_{pred_col}_long.parquet"
                    long_df.to_parquet(long_path, index=False)
                    prediction_long_paths[split][pred_col] = long_path

                    decay = decay_by_col[pred_col]
                    decay_path = report_dir / f"{file_tag}_{split}_{pred_col}_forecast_decay.csv"
                    decay.to_csv(decay_path, index=False)
                    mae_plot_path = report_dir / f"{file_tag}_{split}_{pred_col}_mae_lead_1_24_48.png"
                    _plot_leadtime_mae_points(decay, mae_plot_path)

    fragment = _write_bundle_manifest_fragment(
        bundle=args.bundle,
        run_dir=run_dir,
        model_path=model_out,
        metrics_path=metrics_json_out,
        prediction_paths=prediction_paths,
        prediction_long_paths=prediction_long_paths,
        target_cols=target_cols,
        metrics=metrics,
    )
    manifest_fragment_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_fragment_out.write_text(json.dumps(fragment, indent=2), encoding="utf-8")

    print("[OK] XGBoost training finished.")
    print(f"- Base dir: {base_dir}")
    print(f"- Run dir: {run_dir}")
    print(f"- Bundle: {args.bundle}")
    print(f"- Device: {args.device}")
    print(f"- Target: {metrics['target_col']}")
    print(f"- Train rows: {int(metrics['rows_train'])}")
    print(f"- Val rows: {int(metrics['rows_val'])}")
    print(f"- MAE (val): {metrics['mae']:.4f}")
    print(f"- RMSE (val): {metrics['rmse']:.4f}")
    print(f"- Naive MAE 24h (val): {metrics['baseline_mae_24h']:.4f}")
    print(f"- Importance rows: {int(metrics['n_importance_rows'])}")
    if "cv_folds" in metrics:
        print(f"- CV folds: {int(metrics['cv_folds'])}")
        print(f"- CV MAE mean±std: {metrics['cv_mae_mean']:.4f} ± {metrics['cv_mae_std']:.4f}")
        print(f"- CV RMSE mean±std: {metrics['cv_rmse_mean']:.4f} ± {metrics['cv_rmse_std']:.4f}")
        if "cv_spread_capture_mean" in metrics:
            print(
                "- CV Spread-Capture mean±std: "
                f"{metrics['cv_spread_capture_mean']:.4f} ± {metrics['cv_spread_capture_std']:.4f}"
            )
    print(f"- Metrics JSON: {metrics_json_out}")
    if prediction_paths:
        print(f"- Prediction files: {', '.join(str(p) for p in prediction_paths.values())}")
    print(f"- Manifest fragment: {manifest_fragment_out}")


if __name__ == "__main__":
    main()
