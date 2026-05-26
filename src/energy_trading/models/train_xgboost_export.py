"""Train DA-price XGBoost baseline on prepared ML bundles.

This script trains on the DA bundle (train split) and validates on the
chronological val split with early stopping to reduce overfitting.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib

# Keep matplotlib cache inside project for restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "data" / ".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_trading.models.prepare_ml_bundles import BundleName, load_processed_data
from energy_trading.models.cv import PurgedTimeSeriesSplit
from energy_trading.models.training_policy import (
    resolve_feature_columns_for_target,
    resolve_xgb_params_for_target,
)
from energy_trading.evaluation.metrics import (
    compute_forecast_metrics,
    compute_gate_closure_metrics,
    compute_horizon_bucket_metrics,
    horizon_bucket_metrics_to_table,
    gate_hour_for_target,
)
from energy_trading.evaluation.lead_weighting import weighted_metric_from_decay
from energy_trading.evaluation.tensorboard_utils import (
    create_summary_writer,
    tensorboard_target_log_dir,
)
from energy_trading.evaluation.conformal_calibration import (
    apply_conformal_shifts,
    calculate_conformal_shifts,
)
from energy_trading.models.horizon_weighting import get_lead_sample_weights

QUANTILES: list[float] = [0.01, 0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 0.95, 0.99]
LOGGER = logging.getLogger(__name__)
_ACTIVATION_RATE_TARGETS = {
    "target_afrr_activation_rate_pos",
    "target_afrr_activation_rate_neg",
}
_ACTIVATION_PRICE_TARGETS = {
    "target_afrr_activation_price_vwap_pos",
    "target_afrr_activation_price_vwap_neg",
}
TAIL_WEIGHT_MULTIPLIER = 3.0


class TargetTransform:
    """Robust target transform for heavy-tailed regression targets."""

    def __init__(self, kind: str, q_low: float, q_high: float, symlog_scale: float) -> None:
        self.kind = str(kind)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.symlog_scale = float(max(1e-9, symlog_scale))

    @staticmethod
    def fit(y: pd.Series, *, kind: str = "symlog_clip", clip_low_q: float = 0.01, clip_high_q: float = 0.99) -> "TargetTransform":
        ys = pd.to_numeric(y, errors="coerce")
        finite = ys[np.isfinite(ys.to_numpy(dtype=float))]
        if finite.empty:
            return TargetTransform(kind=kind, q_low=0.0, q_high=0.0, symlog_scale=1.0)
        q_low = float(finite.quantile(clip_low_q))
        q_high = float(finite.quantile(clip_high_q))
        med = float(finite.median())
        mad = float(np.median(np.abs(finite.to_numpy(dtype=float) - med)))
        scale = max(1.0, 1.4826 * mad)
        return TargetTransform(kind=kind, q_low=q_low, q_high=q_high, symlog_scale=scale)

    def transform_series(self, y: pd.Series) -> pd.Series:
        ys = pd.to_numeric(y, errors="coerce")
        clipped = ys.clip(lower=self.q_low, upper=self.q_high)
        if self.kind == "symlog_clip":
            return np.sign(clipped) * np.log1p(np.abs(clipped) / self.symlog_scale)
        return clipped

    def inverse_array(self, x: np.ndarray) -> np.ndarray:
        xv = np.asarray(x, dtype=float)
        if self.kind == "symlog_clip":
            return np.sign(xv) * (np.expm1(np.abs(xv)) * self.symlog_scale)
        return xv

    def to_dict(self) -> dict[str, float | str]:
        return {
            "kind": self.kind,
            "clip_low": self.q_low,
            "clip_high": self.q_high,
            "symlog_scale": self.symlog_scale,
        }


def _qcol(q: float) -> str:
    return f"p{int(round(q * 100)):02d}"


def _tail_sample_weights(y: pd.Series, multiplier: float = TAIL_WEIGHT_MULTIPLIER) -> np.ndarray:
    """Upweight lower/upper tail samples based on empirical 10th/90th percentiles."""
    yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    w = np.ones_like(yv, dtype=float)
    finite = np.isfinite(yv)
    if not np.any(finite):
        return w
    yt = yv[finite]
    q10 = float(np.percentile(yt, 10))
    q90 = float(np.percentile(yt, 90))
    tail_mask = (yv <= q10) | (yv >= q90)
    w[tail_mask & finite] = float(multiplier)
    return w


def _crossing_metrics_from_stack(q_stack: np.ndarray) -> dict[str, float]:
    arr = np.asarray(q_stack, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return {
            "n_rows": float(arr.shape[0]) if arr.ndim >= 1 else 0.0,
            "crossing_rate_before_repair": 0.0,
            "max_crossing_violation_before_repair": 0.0,
        }
    finite_mask = np.all(np.isfinite(arr), axis=1)
    arr_f = arr[finite_mask]
    if arr_f.shape[0] == 0:
        return {
            "n_rows": 0.0,
            "crossing_rate_before_repair": float("nan"),
            "max_crossing_violation_before_repair": float("nan"),
        }
    violations = np.maximum(arr_f[:, :-1] - arr_f[:, 1:], 0.0)
    row_cross = np.any(violations > 0.0, axis=1)
    return {
        "n_rows": float(arr_f.shape[0]),
        "crossing_rate_before_repair": float(np.mean(row_cross)),
        "max_crossing_violation_before_repair": float(np.max(violations)),
    }


def _resolve_base_dir(preferred: str | Path) -> Path:
    p = Path(preferred)
    if p.exists():
        return p
    raise FileNotFoundError(
        f"Bundle directory not found: {p}."
    )


def _default_target_for_bundle(bundle: BundleName) -> str:
    if bundle == "da":
        return "target_da_price"
    if bundle == "afrr":
        return "target_afrr_activation_price_vwap_pos"
    raise KeyError(f"Unsupported bundle: {bundle}")


def _default_afrr_targets() -> list[str]:
    return [
        "target_afrr_activation_price_vwap_pos",
        "target_afrr_activation_price_vwap_neg",
        "target_afrr_activation_rate_pos",
        "target_afrr_activation_rate_neg",
        "target_afrr_capacity_price_pos",
        "target_afrr_capacity_price_neg",
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
        "target_da_price": ["pred_da_price"],
        "target_afrr_activation_price_vwap_pos": ["pred_afrr_activation_price_pos"],
        "target_afrr_activation_price_vwap_neg": ["pred_afrr_activation_price_neg"],
        "target_afrr_activation_rate_pos": ["pred_afrr_activation_rate_pos"],
        "target_afrr_activation_rate_neg": ["pred_afrr_activation_rate_neg"],
        "target_afrr_capacity_price_pos": ["pred_afrr_capacity_price_pos"],
        "target_afrr_capacity_price_neg": ["pred_afrr_capacity_price_neg"],
    }
    return mapping.get(target_col, [f"pred_{target_col}"])


def _source_series_name_for_target(target_col: str) -> str | None:
    mapping = {
        "target_da_price": "da_price",
        "target_afrr_activation_price_vwap_pos": "afrr_activation_price_vwap_pos",
        "target_afrr_activation_price_vwap_neg": "afrr_activation_price_vwap_neg",
        "target_afrr_activation_rate_pos": "afrr_activation_rate_pos",
        "target_afrr_activation_rate_neg": "afrr_activation_rate_neg",
        "target_afrr_capacity_price_pos": "afrr_capacity_price_pos",
        "target_afrr_capacity_price_neg": "afrr_capacity_price_neg",
    }
    return mapping.get(target_col)


def _target_for_pred_column(pred_col: str) -> str | None:
    mapping = {
        "pred_da_price": "target_da_price",
        "pred_afrr_activation_price_pos": "target_afrr_activation_price_vwap_pos",
        "pred_afrr_activation_price_neg": "target_afrr_activation_price_vwap_neg",
        "pred_afrr_activation_rate_pos": "target_afrr_activation_rate_pos",
        "pred_afrr_activation_rate_neg": "target_afrr_activation_rate_neg",
        "pred_afrr_capacity_price_pos": "target_afrr_capacity_price_pos",
        "pred_afrr_capacity_price_neg": "target_afrr_capacity_price_neg",
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


def calculate_acceptance_probabilities(
    quantiles: dict[float, float] | pd.Series,
    price_bins: list[float] | np.ndarray,
) -> pd.DataFrame:
    """Approximate acceptance probability via piecewise-linear CDF.

    For bid price b:
    p_acc(b) = 1 - CDF(b)
    """
    if isinstance(quantiles, pd.Series):
        q_map = {float(k): float(v) for k, v in quantiles.to_dict().items()}
    else:
        q_map = {float(k): float(v) for k, v in quantiles.items()}
    qs = sorted(q_map.keys())
    if not qs:
        return pd.DataFrame(columns=["price_bin", "cdf", "p_acc"])
    vals = np.array([q_map[q] for q in qs], dtype=float)
    bins = np.asarray(price_bins, dtype=float)
    cdf = np.interp(bins, vals, np.array(qs, dtype=float), left=0.0, right=1.0)
    p_acc = 1.0 - cdf
    return pd.DataFrame({"price_bin": bins, "cdf": cdf, "p_acc": p_acc})


def calculate_forecast_decay(
    *,
    pred_long: pd.DataFrame,
    true_h1: pd.Series,
    horizon_hours: int,
) -> pd.DataFrame:
    """Calculate lead-time decay metrics (MAE + optional pinball by quantile)."""
    if pred_long.empty:
        return pd.DataFrame(columns=["lead_time_h", "n", "mae"])

    truth_map: dict[int, pd.Series] = {}
    base = pd.to_numeric(true_h1, errors="coerce")
    for lead in range(1, horizon_hours + 1):
        truth_map[lead] = base.shift(-(lead - 1)).reset_index(drop=True)

    pred_col = "p50" if "p50" in pred_long.columns else "predicted_value"
    pred = pd.to_numeric(pred_long[pred_col], errors="coerce")
    lead = pd.to_numeric(pred_long["lead_time_h"], errors="coerce").astype("Int64")

    pinball_qs: list[tuple[str, float]] = []
    for c in pred_long.columns:
        m = re.fullmatch(r"p(\d{2})", str(c))
        if not m:
            continue
        q_int = int(m.group(1))
        if 0 < q_int < 100:
            pinball_qs.append((str(c), q_int / 100.0))
    pinball_qs.sort(key=lambda kv: kv[1])

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
        row: dict[str, float] = {"lead_time_h": float(l), "n": float(n), "mae": mae}
        yt = pd.to_numeric(aligned_truth, errors="coerce")
        for q_col, tau in pinball_qs:
            q_pred = pd.to_numeric(pred_long.loc[idx, q_col], errors="coerce").reset_index(drop=True)
            q_mask = yt.notna() & q_pred.notna()
            if int(q_mask.sum()) > 0:
                e = yt.loc[q_mask].to_numpy(dtype=float) - q_pred.loc[q_mask].to_numpy(dtype=float)
                loss = np.maximum(float(tau) * e, (float(tau) - 1.0) * e)
                row[f"pinball_{q_col}"] = float(np.mean(loss))
            else:
                row[f"pinball_{q_col}"] = float("nan")
        rows.append(row)
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


def _add_dynamics_features(df: pd.DataFrame, *, prune_midterm_lags: bool = True) -> pd.DataFrame:
    """Add short-term dynamics features used by direct multi-horizon models."""
    out = df.copy()
    new_cols: list[str] = []

    def _num(col: str) -> pd.Series:
        return pd.to_numeric(out[col], errors="coerce")

    def _first_available_pairs(pairs: list[tuple[str, str]]) -> pd.Series | None:
        for a, b in pairs:
            if {a, b}.issubset(out.columns):
                return _num(a) - _num(b)
        return None

    # Imbalance momentum proxy from nearby NRV lags.
    nrv_velocity = _first_available_pairs(
        [
            ("NRV_balance_lag_2h", "NRV_balance_lag_3h"),
            ("nrv_balance_lag_2h", "nrv_balance_lag_3h"),
        ]
    )
    if nrv_velocity is not None:
        out["nrv_velocity_1h"] = nrv_velocity
        new_cols.append("nrv_velocity_1h")

    # Load ramp (t - t-1) from DA-load forecast history.
    # Fallback to horizon differences when only horizonized columns exist.
    load_signed = None
    for col in ("load_forecast_da_entsoe", "load_forecast_da"):
        if col in out.columns:
            load_signed = _num(col).diff(1)
            break
    if load_signed is None:
        load_signed = _first_available_pairs(
            [
                ("load_forecast_da_entsoe_h1", "load_forecast_da_entsoe"),
                ("load_forecast_da_entsoe_h2", "load_forecast_da_entsoe_h1"),
                ("load_forecast_da_h1", "load_forecast_da"),
                ("load_forecast_da_h2", "load_forecast_da_h1"),
            ]
        )
    if load_signed is not None:
        out["load_ramp_signed_1h"] = load_signed
        out["load_ramp_abs_1h"] = load_signed.abs()
        new_cols.extend(["load_ramp_signed_1h", "load_ramp_abs_1h"])

    # Residual-load ramp from history where available; fallback to horizon columns.
    res_signed = None
    for col in ("residual_load_forecast", "residual_load_forecast_da"):
        if col in out.columns:
            res_signed = _num(col).diff(1)
            break
    if res_signed is None:
        res_signed = _first_available_pairs(
            [
                ("residual_load_forecast_h1", "residual_load_forecast"),
                ("residual_load_forecast_h2", "residual_load_forecast_h1"),
                ("residual_load_forecast_da_h1", "residual_load_forecast_da"),
                ("residual_load_forecast_da_h2", "residual_load_forecast_da_h1"),
            ]
        )
    if res_signed is not None:
        out["res_load_ramp_signed_1h"] = res_signed
        new_cols.append("res_load_ramp_signed_1h")

    # Stress interaction: ramp multiplied by lagged renewable forecast-error proxy.
    if {"res_load_ramp_signed_1h", "wind_total_error_da_lag_2h"}.issubset(out.columns):
        out["res_load_ramp_x_wind_total_error_da_lag_2h"] = (
            _num("res_load_ramp_signed_1h")
            * _num("wind_total_error_da_lag_2h")
        )
        new_cols.append("res_load_ramp_x_wind_total_error_da_lag_2h")
    elif {"load_ramp_signed_1h", "wind_total_error_da_lag_2h"}.issubset(out.columns):
        out["load_ramp_x_wind_total_error_da_lag_2h"] = (
            _num("load_ramp_signed_1h")
            * _num("wind_total_error_da_lag_2h")
        )
        new_cols.append("load_ramp_x_wind_total_error_da_lag_2h")
    elif {"load_ramp_signed_1h", "solar_error_da_lag_2h"}.issubset(out.columns):
        out["load_ramp_x_solar_error_da_lag_2h"] = (
            _num("load_ramp_signed_1h")
            * _num("solar_error_da_lag_2h")
        )
        new_cols.append("load_ramp_x_solar_error_da_lag_2h")

    # Robust NaN handling for newly-created dynamics columns (e.g., first diff row).
    if new_cols:
        for c in list(dict.fromkeys(new_cols)):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # Prune mid-term lag variants once short-term velocity/ramp features exist.
    if prune_midterm_lags and (
        "nrv_velocity_1h" in out.columns
        or "load_ramp_signed_1h" in out.columns
        or "res_load_ramp_signed_1h" in out.columns
    ):
        drop_cols = [c for c in out.columns if c.endswith("_lag_4h") or c.endswith("_lag_6h")]
        if drop_cols:
            out = out.drop(columns=drop_cols, errors="ignore")
    return out


def _resolve_training_device(*, requested_device: str, require_cuda: bool) -> str:
    """Resolve XGBoost device with safe fallbacks for Apple Silicon and CPU-only hosts."""
    req = (requested_device or "cpu").strip().lower()
    if req == "mps":
        print("[WARN] XGBoost does not support MPS device directly; falling back to CPU ('hist').")
        return "cpu"
    if req != "cuda":
        if require_cuda and req == "cpu":
            raise RuntimeError("CUDA is required, but training device is set to CPU.")
        return "cpu"

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
        return "cuda"
    except Exception as exc:
        if require_cuda:
            raise RuntimeError(
                "CUDA training required but not available. Install GPU-enabled XGBoost/CUDA runtime or run with --allow-cpu."
            ) from exc
        print(f"[WARN] CUDA probe failed; continuing on CPU fallback: {exc}")
        return "cpu"


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
    seed: int = 42,
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
            X_shap = X_train.sample(n=shap_sample_size, random_state=seed).copy()
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
    colsample_bytree: float,
    min_child_weight: float,
    reg_alpha: float,
    reg_lambda: float,
    subsample: float,
    device: str,
    seed: int,
    horizon_hours: int,
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
        gap_hours=max(gap_hours, horizon_hours),
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
            objective="reg:quantileerror",
            quantile_alpha=0.5,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            learning_rate=learning_rate,
            tree_method="hist",
            device=device,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(
            X_tr,
            y_tr,
            sample_weight=_tail_sample_weights(y_tr),
            verbose=False,
        )
        pred = _predict_with_device_alignment(model, X_va, resolved_device=device)
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
    run_dir: Path,
    model_out: Path,
    importance_out: Path,
    importance_report_out: Path,
    shap_summary_out: Path,
    n_estimators: int = 1000,
    max_depth: int = 8,
    learning_rate: float = 0.05,
    subsample: float = 0.9,
    colsample_bytree: float = 0.8,
    min_child_weight: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    early_stopping_rounds: int = 50,
    run_cv: bool = False,
    cv_n_splits: int = 3,
    cv_test_size: int = 24 * 28,
    cv_gap_hours: int = 72,
    device: str = "cuda",
    require_cuda: bool = True,
    horizon_hours: int = 48,
    seed: int = 42,
    activation_price_transform: str = "symlog_clip",
) -> tuple[dict[str, float], dict[str, dict[int, dict[str, object]]], list[str]]:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "xgboost is not available. Install xgboost (and libomp on macOS if needed)."
        ) from exc

    resolved_device = _resolve_training_device(requested_device=device, require_cuda=require_cuda)

    X_train_df, y_train_df = load_processed_data(
        bundle=bundle,
        split="train",
        base_dir=base_dir,
        target_col_for_feature_routing=target_col or None,
    )
    X_val_df, y_val_df = load_processed_data(
        bundle=bundle,
        split="val",
        base_dir=base_dir,
        target_col_for_feature_routing=target_col or None,
    )
    target_cols = _resolve_targets(bundle, list(y_train_df.columns), target_col)
    primary_target = target_cols[0]
    cfg = json.loads((base_dir / "feature_config.json").read_text(encoding="utf-8"))
    bcfg = cfg["bundles"][bundle]
    train_ts_full = pd.to_datetime(
        pd.read_parquet(Path(bcfg["files"]["train"]), columns=["timestamp_utc"])["timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    # Train on rows where all selected targets are available.
    train_mask = y_train_df[target_cols].notna().all(axis=1)
    val_mask = y_val_df[target_cols].notna().all(axis=1)
    X_train = X_train_df.loc[train_mask].copy()
    X_val = X_val_df.loc[val_mask].copy()
    y_train_m = y_train_df.loc[train_mask, target_cols].copy()
    y_val_m = y_val_df.loc[val_mask, target_cols].copy()
    train_hours_all = (
        train_ts_full.loc[train_mask]
        .dt.tz_convert("Europe/Berlin")
        .dt.hour
        .to_numpy(dtype=int)
    )

    X_train = _add_dynamics_features(X_train, prune_midterm_lags=True)
    X_val = _add_dynamics_features(X_val, prune_midterm_lags=True)
    all_feature_columns = list(X_train.columns)

    base_xgb_params: dict[str, float] = {
        "max_depth": float(max_depth),
        "min_child_weight": float(min_child_weight),
        "learning_rate": float(learning_rate),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "reg_alpha": float(reg_alpha),
        "reg_lambda": float(reg_lambda),
        "early_stopping_rounds": float(max(0, int(early_stopping_rounds))),
    }
    primary_policy = resolve_xgb_params_for_target(primary_target, base_xgb_params)
    primary_variant, primary_cols = resolve_feature_columns_for_target(all_feature_columns, primary_target)
    X_train_primary = X_train[primary_cols].copy()
    X_val_primary = X_val[primary_cols].copy()

    # CV remains on primary target only.
    cv_metrics: dict[str, float] = {}
    if run_cv:
        cv_metrics = run_purged_cv_with_pipeline(
            X_train=X_train_primary,
            y_train=pd.to_numeric(y_train_m[primary_target], errors="coerce"),
            n_splits=cv_n_splits,
            test_size=cv_test_size,
            gap_hours=cv_gap_hours,
            n_estimators=max(200, n_estimators // 2),
            max_depth=int(primary_policy["max_depth"]),
            min_child_weight=float(primary_policy["min_child_weight"]),
            learning_rate=float(primary_policy["learning_rate"]),
            colsample_bytree=float(primary_policy["colsample_bytree"]),
            reg_alpha=float(primary_policy["reg_alpha"]),
            reg_lambda=float(primary_policy["reg_lambda"]),
            subsample=float(primary_policy["subsample"]),
            device=resolved_device,
            seed=seed,
            horizon_hours=horizon_hours,
        )

    models_by_target: dict[str, dict[int, dict[str, object]]] = {}
    val_pred_df = pd.DataFrame(index=X_val.index)
    per_target_metrics: dict[str, dict[str, float]] = {}
    per_target_policy: dict[str, dict[str, object]] = {}
    per_target_training_seconds: dict[str, float] = {}
    target_transform_by_target: dict[str, dict[str, float | str]] = {}
    per_target_tb_log_dir: dict[str, str] = {}
    xgb_best_iteration_rows: list[dict[str, object]] = []
    per_target_crossing_acc: dict[str, dict[str, float]] = {}
    total_fit_seconds = 0.0
    primary_X_val_h = None
    primary_y_val_lead1 = None
    for idx, tgt in enumerate(target_cols):
        tb_writer = None
        tb_target_log_dir = tensorboard_target_log_dir(
            run_dir=run_dir,
            model_family="xgb",
            bundle=bundle,
            target_col=tgt,
        )
        tb_writer = create_summary_writer(tb_target_log_dir)
        if tb_writer is not None:
            per_target_tb_log_dir[tgt] = str(tb_target_log_dir.resolve())

        target_policy = resolve_xgb_params_for_target(tgt, base_xgb_params)
        target_variant, target_feature_cols = resolve_feature_columns_for_target(all_feature_columns, tgt)
        X_train_t = X_train[target_feature_cols].copy()
        X_val_t = X_val[target_feature_cols].copy()

        target_train_start = time.perf_counter()
        y_tr_base = pd.to_numeric(y_train_m[tgt], errors="coerce")
        y_va_base = pd.to_numeric(y_val_m[tgt], errors="coerce")
        target_transform: TargetTransform | None = None
        if tgt in _ACTIVATION_PRICE_TARGETS and activation_price_transform != "none":
            target_transform = TargetTransform.fit(y_tr_base, kind=activation_price_transform)
            target_transform_by_target[tgt] = target_transform.to_dict()
        Y_tr = _build_horizon_matrix(y_tr_base, horizon_hours=horizon_hours)
        Y_va = _build_horizon_matrix(y_va_base, horizon_hours=horizon_hours)

        q_cols = [_qcol(q) for q in QUANTILES]
        lead_models: dict[int, dict[str, object]] = {}
        pred_by_q = {c: np.full((len(X_val), horizon_hours), np.nan, dtype=float) for c in q_cols}

        rows_train_per_lead: list[int] = []
        rows_val_per_lead: list[int] = []
        best_iteration_by_lead: dict[str, dict[str, object]] = {}
        lead_loop_start = time.time()
        eta_logged = False

        for lead in range(1, horizon_hours + 1):
            y_tr_lead = pd.to_numeric(Y_tr.iloc[:, lead - 1], errors="coerce")
            y_va_lead = pd.to_numeric(Y_va.iloc[:, lead - 1], errors="coerce")

            tr_mask = y_tr_lead.notna()
            va_mask = y_va_lead.notna()
            X_tr_h = X_train_t.loc[tr_mask].copy()
            X_va_h = X_val_t.loc[va_mask].copy()
            y_tr_h = y_tr_lead.loc[tr_mask].copy()
            y_va_h = y_va_lead.loc[va_mask].copy()
            y_tr_model = target_transform.transform_series(y_tr_h) if target_transform is not None else y_tr_h
            y_va_model = target_transform.transform_series(y_va_h) if target_transform is not None else y_va_h

            if X_tr_h.empty or X_va_h.empty:
                raise ValueError(
                    f"Not enough rows for target '{tgt}', lead h{lead} "
                    f"(train={len(X_tr_h)}, val={len(X_va_h)})."
                )

            LOGGER.info(
                "Training target=%s lead=%s/%s (train=%s, val=%s)",
                tgt,
                lead,
                horizon_hours,
                len(X_tr_h),
                len(X_va_h),
            )
            rows_train_per_lead.append(len(X_tr_h))
            rows_val_per_lead.append(len(X_va_h))

            quantile_models_for_lead: dict[str, object] = {}
            lead_pred_by_q: dict[str, np.ndarray] = {}
            lead_best_iteration: dict[str, int | None] = {}
            lead_best_score: dict[str, float | None] = {}
            lead_boosted_rounds: dict[str, int | None] = {}
            for q_idx, q in enumerate(QUANTILES):
                qcol = q_cols[q_idx]
                model = XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=float(q),
                    n_estimators=n_estimators,
                    max_depth=int(target_policy["max_depth"]),
                    min_child_weight=float(target_policy["min_child_weight"]),
                    learning_rate=float(target_policy["learning_rate"]),
                    # Use histogram algorithm explicitly for efficient GPU training.
                    tree_method="hist",
                    device=resolved_device,
                    subsample=float(target_policy["subsample"]),
                    colsample_bytree=float(target_policy["colsample_bytree"]),
                    reg_alpha=float(target_policy["reg_alpha"]),
                    reg_lambda=float(target_policy["reg_lambda"]),
                    random_state=seed + idx * 10_000 + lead * 100 + q_idx,
                    early_stopping_rounds=max(0, int(target_policy["early_stopping_rounds"])),
                    n_jobs=-1,
                )
                # Build training weights explicitly before fit call.
                # Note: this path uses sklearn XGBRegressor API (not xgb.train),
                # so weights are passed via fit(..., sample_weight=...).
                sample_weight_h = (
                    _tail_sample_weights(y_tr_model)
                    * get_lead_sample_weights(
                        target_name=tgt,
                        current_hour=train_hours_all[tr_mask.to_numpy(dtype=bool)],
                        lead_time_h=int(lead),
                    )
                )
                w_mean = float(np.nanmean(sample_weight_h))
                if not np.isfinite(w_mean) or abs(w_mean) <= 1e-12:
                    raise ValueError(
                        f"Invalid mean sample weight for target='{tgt}' lead={lead}: {w_mean}"
                    )
                sample_weight_h = sample_weight_h / w_mean
                finite_w = sample_weight_h[np.isfinite(sample_weight_h)]
                if finite_w.size == 0:
                    raise ValueError(
                        f"Non-finite sample weights for target='{tgt}' lead={lead}."
                    )
                p50_w = float(np.percentile(finite_w, 50.0))
                p95_w = float(np.percentile(finite_w, 95.0))
                LOGGER.info(
                    "[WEIGHT_AUDIT][XGB] target=%s lead=%s n=%s min=%.6f p50=%.6f p95=%.6f mean=%.6f max=%.6f",
                    tgt,
                    lead,
                    int(sample_weight_h.shape[0]),
                    float(np.min(finite_w)),
                    p50_w,
                    p95_w,
                    float(np.mean(finite_w)),
                    float(np.max(finite_w)),
                )
                if ("afrr_activation_price_" in str(tgt).lower()) or (
                    "afrr_activation_rate_" in str(tgt).lower()
                ):
                    LOGGER.info(
                        "[WEIGHT_NOTE][XGB] target=%s lead=%s uses one direct-per-lead model; "
                        "horizon-decay becomes a lead-constant scaling and is neutralized by mean-normalization.",
                        tgt,
                        lead,
                    )
                model.fit(
                    X_tr_h,
                    y_tr_model,
                    sample_weight=sample_weight_h,
                    eval_set=[(X_tr_h, y_tr_model), (X_va_h, y_va_model)],
                    verbose=False,
                )

                # Track early-stopping outcome per lead/quantile.
                best_it = getattr(model, "best_iteration", None)
                if best_it is not None:
                    best_it = int(best_it)
                best_sc = getattr(model, "best_score", None)
                if best_sc is not None:
                    try:
                        best_sc = float(best_sc)
                    except Exception:
                        best_sc = None
                rounds = None
                try:
                    rounds = int(model.get_booster().num_boosted_rounds())
                except Exception:
                    rounds = None
                lead_best_iteration[qcol] = best_it
                lead_best_score[qcol] = best_sc
                lead_boosted_rounds[qcol] = rounds
                xgb_best_iteration_rows.append(
                    {
                        "target_col": str(tgt),
                        "lead_time_h": int(lead),
                        "quantile": str(qcol),
                        "best_iteration": best_it,
                        "best_score": best_sc,
                        "num_boosted_rounds": rounds,
                        "early_stopping_rounds": int(target_policy["early_stopping_rounds"]),
                        "n_train_rows": int(len(X_tr_h)),
                        "n_val_rows": int(len(X_va_h)),
                    }
                )

                # Log one representative target curve per trained target
                # (p50, lead-1) to TensorBoard for direct comparability with TFT logs.
                if tb_writer is not None and abs(float(q) - 0.5) < 1e-12 and lead == 1:
                    try:
                        evals = model.evals_result()
                        train_block = evals.get("validation_0", {})
                        val_block = evals.get("validation_1", {})
                        train_metric = next(iter(train_block.keys()), None)
                        val_metric = next(iter(val_block.keys()), None)
                        train_hist = list(train_block.get(train_metric, [])) if train_metric else []
                        val_hist = list(val_block.get(val_metric, [])) if val_metric else []
                        for ep, (tr_v, va_v) in enumerate(
                            itertools.zip_longest(train_hist, val_hist, fillvalue=np.nan)
                        ):
                            if np.isfinite(float(tr_v)):
                                tb_writer.add_scalar("train_loss_epoch", float(tr_v), ep)
                            if np.isfinite(float(va_v)):
                                tb_writer.add_scalar("val_loss", float(va_v), ep)
                    except Exception as exc:  # pragma: no cover
                        LOGGER.warning(
                            "Could not write TensorBoard loss curves for target=%s: %s",
                            tgt,
                            exc,
                        )

                pred = _predict_with_device_alignment(model, X_va_h, resolved_device=resolved_device)
                if target_transform is not None:
                    pred = target_transform.inverse_array(pred)
                quantile_models_for_lead[qcol] = model
                lead_pred_by_q[qcol] = pred

            # Monotonicity repair for this lead against quantile crossing.
            lead_stack = np.column_stack([lead_pred_by_q[c] for c in q_cols])
            lead_cross = _crossing_metrics_from_stack(lead_stack)
            acc = per_target_crossing_acc.setdefault(
                tgt,
                {"n_rows": 0.0, "n_cross_rows": 0.0, "max_violation": 0.0},
            )
            if np.isfinite(lead_cross["n_rows"]) and lead_cross["n_rows"] > 0.0:
                acc["n_rows"] += lead_cross["n_rows"]
                acc["n_cross_rows"] += lead_cross["crossing_rate_before_repair"] * lead_cross["n_rows"]
                acc["max_violation"] = max(
                    float(acc["max_violation"]),
                    float(lead_cross["max_crossing_violation_before_repair"]),
                )
            lead_stack = np.sort(lead_stack, axis=1)
            for qi, qcol in enumerate(q_cols):
                lead_pred_series = pd.Series(np.nan, index=X_val.index, dtype=float)
                lead_pred_series.loc[X_va_h.index] = lead_stack[:, qi]
                pred_by_q[qcol][:, lead - 1] = lead_pred_series.to_numpy(dtype=float)

            lead_models[lead] = quantile_models_for_lead
            # Lightweight per-lead summary for quick inspection.
            p50_best_it = lead_best_iteration.get("p50")
            p50_rounds = lead_boosted_rounds.get("p50")
            p50_early_stopped = (
                bool((p50_best_it is not None) and (p50_rounds is not None) and (p50_best_it < (p50_rounds - 1)))
                if (p50_best_it is not None and p50_rounds is not None)
                else None
            )
            best_iteration_by_lead[str(int(lead))] = {
                "p10": lead_best_iteration.get("p10"),
                "p50": p50_best_it,
                "p90": lead_best_iteration.get("p90"),
                "p95": lead_best_iteration.get("p95"),
                "p50_num_boosted_rounds": p50_rounds,
                "p50_early_stopped": p50_early_stopped,
            }
            if lead == 1 and horizon_hours > 1 and not eta_logged:
                lead1_minutes = (time.time() - lead_loop_start) / 60.0
                eta_minutes = lead1_minutes * float(horizon_hours - 1)
                LOGGER.info(
                    "ETA for remaining %s leads (target=%s): ~ %.1f minutes",
                    horizon_hours - 1,
                    tgt,
                    eta_minutes,
                )
                eta_logged = True

        lead1_pred = pd.Series(pred_by_q["p50"][:, 0], index=X_val.index)
        val_pred_df[tgt] = np.nan
        val_pred_df.loc[X_val.index, tgt] = lead1_pred

        lead24_mae = np.nan
        lead48_mae = np.nan
        lead1_true = pd.to_numeric(Y_va.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        lead1_pred_np = pred_by_q["p50"][:, 0]
        mask_h1 = np.isfinite(lead1_true) & np.isfinite(lead1_pred_np)
        if not bool(mask_h1.any()):
            raise ValueError(f"No valid h1 validation rows for target '{tgt}'.")
        mae_h1 = float(mean_absolute_error(lead1_true[mask_h1], lead1_pred_np[mask_h1]))
        rmse_h1 = float(np.sqrt(mean_squared_error(lead1_true[mask_h1], lead1_pred_np[mask_h1])))

        h1_metrics_df = pd.DataFrame(
            {
                "y_true": lead1_true,
                "y_pred": lead1_pred_np,
            }
        )
        for qcol in q_cols:
            h1_metrics_df[f"y_pred_{qcol}"] = pred_by_q[qcol][:, 0]
        h1_metric_suite = compute_forecast_metrics(h1_metrics_df, y_true_col="y_true", y_pred_col="y_pred")

        if horizon_hours >= 24:
            y_true_24 = pd.to_numeric(Y_va.iloc[:, 23], errors="coerce").to_numpy(dtype=float)
            y_pred_24 = pred_by_q["p50"][:, 23]
            m24 = np.isfinite(y_true_24) & np.isfinite(y_pred_24)
            if bool(m24.any()):
                lead24_mae = float(mean_absolute_error(y_true_24[m24], y_pred_24[m24]))
        if horizon_hours >= 48:
            y_true_48 = pd.to_numeric(Y_va.iloc[:, 47], errors="coerce").to_numpy(dtype=float)
            y_pred_48 = pred_by_q["p50"][:, 47]
            m48 = np.isfinite(y_true_48) & np.isfinite(y_pred_48)
            if bool(m48.any()):
                lead48_mae = float(mean_absolute_error(y_true_48[m48], y_pred_48[m48]))
        per_target_metrics[tgt] = {
            "mae": mae_h1,
            "rmse": rmse_h1,
            "mae_h24": lead24_mae,
            "mae_h48": lead48_mae,
            "mape_h1": h1_metric_suite.get("mape"),
            "wmape_h1": h1_metric_suite.get("wmape"),
            "r2_h1": h1_metric_suite.get("r2"),
            "mbe_h1": h1_metric_suite.get("mbe"),
            "over_prediction_ratio_h1": h1_metric_suite.get("over_prediction_ratio"),
            "directional_accuracy_h1": h1_metric_suite.get("directional_accuracy"),
            "pinball_loss_p10_h1": h1_metric_suite.get("pinball_loss_p10"),
            "pinball_loss_p50_h1": h1_metric_suite.get("pinball_loss_p50"),
            "pinball_loss_p90_h1": h1_metric_suite.get("pinball_loss_p90"),
            "pinball_loss_p95_h1": h1_metric_suite.get("pinball_loss_p95"),
            "picp_80_h1": h1_metric_suite.get("picp_80"),
            "rows_train_horizon_min": float(min(rows_train_per_lead)),
            "rows_train_horizon_max": float(max(rows_train_per_lead)),
            "rows_val_horizon_min": float(min(rows_val_per_lead)),
            "rows_val_horizon_max": float(max(rows_val_per_lead)),
            "best_iteration_by_lead": best_iteration_by_lead,
            "metric_suite_h1": h1_metric_suite,
            "crossing_rate_before_repair": (
                float(per_target_crossing_acc[tgt]["n_cross_rows"] / per_target_crossing_acc[tgt]["n_rows"])
                if per_target_crossing_acc.get(tgt, {}).get("n_rows", 0.0) > 0.0
                else float("nan")
            ),
            "max_crossing_violation_before_repair": (
                float(per_target_crossing_acc[tgt]["max_violation"])
                if per_target_crossing_acc.get(tgt, {}).get("n_rows", 0.0) > 0.0
                else float("nan")
            ),
        }
        per_target_policy[tgt] = {
            "feature_variant": target_variant,
            "n_features": int(len(target_feature_cols)),
            "xgb_params": {
                "max_depth": int(target_policy["max_depth"]),
                "min_child_weight": float(target_policy["min_child_weight"]),
                "learning_rate": float(target_policy["learning_rate"]),
                "subsample": float(target_policy["subsample"]),
                "colsample_bytree": float(target_policy["colsample_bytree"]),
                "reg_alpha": float(target_policy["reg_alpha"]),
                "reg_lambda": float(target_policy["reg_lambda"]),
                "early_stopping_rounds": int(target_policy["early_stopping_rounds"]),
            },
            "target_transform": target_transform.to_dict() if target_transform is not None else {"kind": "none"},
        }
        models_by_target[tgt] = lead_models
        target_elapsed = time.perf_counter() - target_train_start
        per_target_training_seconds[tgt] = float(target_elapsed)
        total_fit_seconds += float(target_elapsed)
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
        if tgt == primary_target:
            primary_X_val_h = X_val
            primary_y_val_lead1 = pd.to_numeric(Y_va.iloc[:, 0], errors="coerce")

    if primary_X_val_h is None or primary_y_val_lead1 is None:
        raise ValueError("Primary target horizon-aligned validation set is empty.")
    y_val_primary = primary_y_val_lead1
    y_pred_primary = pd.to_numeric(val_pred_df.loc[primary_X_val_h.index, primary_target], errors="coerce")
    mae = float(mean_absolute_error(y_val_primary, y_pred_primary))
    rmse = float(np.sqrt(mean_squared_error(y_val_primary, y_pred_primary)))
    primary_metric_suite_h1 = per_target_metrics.get(primary_target, {}).get("metric_suite_h1", {})
    baseline_mae = _naive_baseline_mae(bundle, primary_X_val_h, y_val_primary)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    model_payload = models_by_target if len(models_by_target) > 1 else models_by_target[primary_target]
    joblib.dump(model_payload, model_out)

    primary_model = _unwrap_single_xgb(models_by_target[primary_target][1]["p50"])
    _plot_top_feature_importance(primary_model, feature_names=list(primary_cols), out_path=importance_out, top_n=20)
    importance_report = calculate_feature_importance(
        primary_model,
        X_train_primary,
        report_out=importance_report_out,
        shap_plot_out=shap_summary_out,
        seed=seed,
    )

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape_h1": primary_metric_suite_h1.get("mape"),
        "wmape_h1": primary_metric_suite_h1.get("wmape"),
        "r2_h1": primary_metric_suite_h1.get("r2"),
        "mbe_h1": primary_metric_suite_h1.get("mbe"),
        "over_prediction_ratio_h1": primary_metric_suite_h1.get("over_prediction_ratio"),
        "directional_accuracy_h1": primary_metric_suite_h1.get("directional_accuracy"),
        "pinball_loss_p10_h1": primary_metric_suite_h1.get("pinball_loss_p10"),
        "pinball_loss_p50_h1": primary_metric_suite_h1.get("pinball_loss_p50"),
        "pinball_loss_p90_h1": primary_metric_suite_h1.get("pinball_loss_p90"),
        "pinball_loss_p95_h1": primary_metric_suite_h1.get("pinball_loss_p95"),
        "picp_80_h1": primary_metric_suite_h1.get("picp_80"),
        "baseline_mae_24h": baseline_mae,
        "rows_train": float(len(X_train)),
        "rows_val": float(len(X_val)),
        "n_importance_rows": float(len(importance_report)),
        "target_col": primary_target,
        "target_cols": target_cols,
        "multioutput_horizon_hours": float(horizon_hours),
        "feature_count_after_refactor": float(len(all_feature_columns)),
        "per_target_metrics": per_target_metrics,
        "per_target_policy": per_target_policy,
        "primary_feature_variant": primary_variant,
        "primary_feature_count": float(len(primary_cols)),
        "resolved_device": resolved_device,
        "timing_training_seconds": float(total_fit_seconds),
        "timing_training_seconds_by_target": per_target_training_seconds,
        "target_transform_by_target": target_transform_by_target,
        "tensorboard_log_dirs_by_target": per_target_tb_log_dir,
        "xgb_best_iteration_by_target_lead_quantile": xgb_best_iteration_rows,
        "crossing_rate_before_repair": per_target_metrics.get(primary_target, {}).get(
            "crossing_rate_before_repair"
        ),
        "max_crossing_violation_before_repair": per_target_metrics.get(primary_target, {}).get(
            "max_crossing_violation_before_repair"
        ),
    }
    # Export all probabilistic metrics available on primary target H1 suite.
    for k, v in primary_metric_suite_h1.items():
        if (
            k.startswith("pinball_loss_p")
            or k.startswith("picp_")
            or k.startswith("winkler_score_")
            or k.startswith("pinaw_")
            or k.startswith("coverage_gap_")
            or k.startswith("tradeoff_score_")
            or k.startswith("crps_")
        ):
            metrics[f"{k}_h1"] = v
    metrics.update(cv_metrics)
    return metrics, models_by_target, target_cols


def _run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _resolve_run_dir(run_dir: str | Path | None) -> Path:
    if run_dir is None or str(run_dir).strip() == "":
        return Path("artifacts/model_runs") / _run_id_now()
    return Path(run_dir)


def _align_features_for_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """Align inference feature frame to the exact feature names seen in training."""
    expected = None
    try:
        expected = model.get_booster().feature_names
    except Exception:
        expected = None
    if not expected:
        return X
    X_aligned = X.reindex(columns=list(expected))
    return X_aligned


def _predict_with_device_alignment(model, X: pd.DataFrame, *, resolved_device: str) -> np.ndarray:
    """Predict with a GPU-compatible path when CUDA is active.

    If CuPy is available and model device is CUDA, use booster.inplace_predict on
    a CuPy array to avoid device mismatch fallback warnings and extra host-device
    conversions. Falls back to sklearn-wrapper predict otherwise.
    """
    X_pred = _align_features_for_model(X, model)
    dev = (resolved_device or "").lower()
    if dev.startswith("cuda"):
        try:
            import cupy as cp  # type: ignore

            booster = model.get_booster()
            booster.set_param({"device": resolved_device})
            x_gpu = cp.asarray(X_pred.to_numpy(dtype=np.float32, copy=False))
            pred_gpu = booster.inplace_predict(x_gpu)
            return np.asarray(cp.asnumpy(pred_gpu), dtype=float).reshape(-1)
        except Exception:
            # If CuPy is unavailable, avoid sklearn's CUDA/CPU inplace-predict
            # mismatch path by explicitly using DMatrix with the booster.
            try:
                import xgboost as xgb  # type: ignore

                booster = model.get_booster()
                booster.set_param({"device": resolved_device})
                x_dm = xgb.DMatrix(
                    X_pred.to_numpy(dtype=np.float32, copy=False),
                    feature_names=list(X_pred.columns),
                )
                pred = booster.predict(x_dm)
                return np.asarray(pred, dtype=float).reshape(-1)
            except Exception:
                # Last-resort fallback for environments with incompatible XGBoost APIs.
                return np.asarray(model.predict(X_pred), dtype=float).reshape(-1)
    return np.asarray(model.predict(X_pred), dtype=float).reshape(-1)


def _predict_split_frame(
    base_dir: Path,
    bundle: BundleName,
    split: str,
    models_by_target: dict[str, dict[int, dict[str, object]]],
    target_cols: list[str],
    target_transform_by_target: dict[str, dict[str, float | str]] | None = None,
    resolved_device: str = "cpu",
) -> pd.DataFrame:
    split_df, bcfg = _load_bundle_split_df(base_dir=base_dir, bundle=bundle, split=split)
    X = split_df[bcfg["features"]].copy()
    X = _add_dynamics_features(X, prune_midterm_lags=True)
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
        model_q = models_by_target[tgt][1]["p50"]
        pred_h = _predict_with_device_alignment(model_q, X, resolved_device=resolved_device)
        tf_cfg = (target_transform_by_target or {}).get(tgt, {})
        if str(tf_cfg.get("kind", "none")) != "none":
            tf = TargetTransform(
                kind=str(tf_cfg.get("kind", "none")),
                q_low=float(tf_cfg.get("clip_low", 0.0)),
                q_high=float(tf_cfg.get("clip_high", 0.0)),
                symlog_scale=float(tf_cfg.get("symlog_scale", 1.0)),
            )
            pred_h = tf.inverse_array(pred_h)
        pred = pd.to_numeric(pd.Series(pred_h, index=split_df.index), errors="coerce")
        if tgt in _ACTIVATION_RATE_TARGETS:
            pred = pred.clip(lower=0.0, upper=1.0)
        for pred_col in _pred_column_names_for_target(tgt):
            out[pred_col] = pred.values
    return out


def _predict_split_long_multistep(
    *,
    base_dir: Path,
    bundle: BundleName,
    split: str,
    models_by_target: dict[str, dict[int, dict[str, object]]],
    target_cols: list[str],
    target_transform_by_target: dict[str, dict[str, float | str]] | None,
    horizon_hours: int,
    model_name: str,
    resolved_device: str = "cpu",
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    """Generate direct multi-output predictions and return long-format tables.

    Output per prediction column includes quantile surface:
    - snapshot_time_utc, target_time_utc, lead_time_h, model_name
    - p10..p90 (+ predicted_value=p50 for compatibility)
    """
    split_df, bcfg = _load_bundle_split_df(base_dir=base_dir, bundle=bundle, split=split)
    features = list(bcfg["features"])
    X = split_df[features].copy()
    X = _add_dynamics_features(X, prune_midterm_lags=True)
    snapshots = pd.to_datetime(split_df["timestamp_utc"], utc=True, errors="coerce")

    pred_frames: dict[str, list[pd.DataFrame]] = {}
    crossing_acc_by_pred_col: dict[str, dict[str, float]] = {}

    # Keep true h+1 series per target for decay metric.
    true_h1_by_target: dict[str, pd.Series] = {}
    for tgt in target_cols:
        if tgt in split_df.columns:
            true_h1_by_target[tgt] = pd.to_numeric(split_df[tgt], errors="coerce")

    for tgt in target_cols:
        q_cols = [_qcol(q) for q in QUANTILES]
        use_h = min(horizon_hours, max(models_by_target[tgt].keys()))
        for lead in range(1, use_h + 1):
            if lead not in models_by_target[tgt]:
                continue
            lead_models = models_by_target[tgt][lead]
            if not all(c in lead_models for c in q_cols):
                continue

            # Predict each quantile model for this lead, then enforce monotonicity.
            lead_pred_q: dict[str, np.ndarray] = {}
            for c in q_cols:
                model_q = lead_models[c]
                lead_pred_q[c] = _predict_with_device_alignment(model_q, X, resolved_device=resolved_device)
            lead_stack = np.column_stack([lead_pred_q[c] for c in q_cols])
            lead_cross = _crossing_metrics_from_stack(lead_stack)
            tf_cfg = (target_transform_by_target or {}).get(tgt, {})
            if str(tf_cfg.get("kind", "none")) != "none":
                tf = TargetTransform(
                    kind=str(tf_cfg.get("kind", "none")),
                    q_low=float(tf_cfg.get("clip_low", 0.0)),
                    q_high=float(tf_cfg.get("clip_high", 0.0)),
                    symlog_scale=float(tf_cfg.get("symlog_scale", 1.0)),
                )
                lead_stack = tf.inverse_array(lead_stack)
            lead_stack = np.sort(lead_stack, axis=1)
            if tgt in _ACTIVATION_RATE_TARGETS:
                # Activation rate predictions are bounded probabilities/fractions.
                lead_stack = np.clip(lead_stack, 0.0, 1.0)

            target_time = snapshots + pd.to_timedelta(lead, unit="h")
            for pred_col in _pred_column_names_for_target(tgt):
                acc = crossing_acc_by_pred_col.setdefault(
                    pred_col,
                    {"n_rows": 0.0, "n_cross_rows": 0.0, "max_violation": 0.0},
                )
                if np.isfinite(lead_cross["n_rows"]) and lead_cross["n_rows"] > 0.0:
                    acc["n_rows"] += lead_cross["n_rows"]
                    acc["n_cross_rows"] += lead_cross["crossing_rate_before_repair"] * lead_cross["n_rows"]
                    acc["max_violation"] = max(
                        float(acc["max_violation"]),
                        float(lead_cross["max_crossing_violation_before_repair"]),
                    )
                payload = {
                    "snapshot_time_utc": snapshots,
                    "target_time_utc": target_time,
                    "lead_time_h": lead,
                    "model_name": model_name,
                }
                for qi, c in enumerate(q_cols):
                    payload[c] = pd.to_numeric(
                        pd.Series(lead_stack[:, qi], index=split_df.index),
                        errors="coerce",
                    ).values
                payload["predicted_value"] = payload["p50"]
                long_df = pd.DataFrame(
                    payload
                )
                pred_frames.setdefault(pred_col, []).append(long_df)

    out_long: dict[str, pd.DataFrame] = {}
    out_decay: dict[str, pd.DataFrame] = {}
    out_crossing: dict[str, dict[str, float]] = {}
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
        acc = crossing_acc_by_pred_col.get(pred_col, {})
        n_rows = float(acc.get("n_rows", 0.0))
        out_crossing[pred_col] = {
            "crossing_rate_before_repair": (
                float(acc["n_cross_rows"] / n_rows) if n_rows > 0.0 else float("nan")
            ),
            "max_crossing_violation_before_repair": (
                float(acc["max_violation"]) if n_rows > 0.0 else float("nan")
            ),
        }
    return out_long, out_decay, out_crossing


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
    p.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda", help="XGBoost device.")
    p.add_argument("--model-name", default="xgboost_v1", help="Name written to long-format prediction export.")
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
    p.add_argument("--lead-weight-start", type=int, default=16)
    p.add_argument("--lead-weight-end", type=int, default=48)
    p.add_argument("--lead-weight-max", type=float, default=2.0)
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
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--reg-alpha", type=float, default=0.0)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument(
        "--run-cv",
        action="store_true",
        help="Run leakage-safe inner CV with fold-fitted preprocessing pipeline.",
    )
    p.add_argument("--cv-n-splits", type=int, default=3)
    p.add_argument("--cv-test-size", type=int, default=24 * 28, help="Validation rows per fold (hourly rows).")
    p.add_argument("--cv-gap-hours", type=int, default=72)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--activation-price-transform",
        choices=["none", "symlog_clip"],
        default="symlog_clip",
        help="Robust target transform for aFRR activation-price targets.",
    )
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    total_start = time.perf_counter()
    args = _build_cli().parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
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

    # Keep model family explicit in artifact filenames for unambiguous tracking.
    file_tag = f"xgboost_{args.bundle}_{target_tag}" if target_tag else f"xgboost_{args.bundle}"

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

    train_eval_start = time.perf_counter()
    metrics, models_by_target, target_cols = train_and_evaluate(
        base_dir=base_dir,
        bundle=args.bundle,
        target_col=(args.target_col.strip() or None),
        run_dir=run_dir,
        model_out=model_out,
        importance_out=importance_out,
        importance_report_out=importance_report_out,
        shap_summary_out=shap_summary_out,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        early_stopping_rounds=args.early_stopping_rounds,
        run_cv=args.run_cv,
        cv_n_splits=args.cv_n_splits,
        cv_test_size=args.cv_test_size,
        cv_gap_hours=args.cv_gap_hours,
        device=args.device,
        require_cuda=not args.allow_cpu,
        horizon_hours=args.forecast_horizon_hours,
        seed=args.seed,
        activation_price_transform=args.activation_price_transform,
    )
    train_eval_elapsed = time.perf_counter() - train_eval_start

    metrics_json_out.parent.mkdir(parents=True, exist_ok=True)
    export_start = time.perf_counter()

    prediction_paths: dict[str, Path] = {}
    prediction_long_paths: dict[str, dict[str, Path]] = {}
    leadtime_weighted_by_split: dict[str, dict[str, dict[str, float]]] = {}
    prediction_export_seconds_by_split: dict[str, float] = {}
    long_export_seconds_by_split: dict[str, float] = {}
    gate_closure_metrics_by_split: dict[str, dict[str, dict[str, object]]] = {}
    splits = [s.strip() for s in args.prediction_splits.split(",") if s.strip()]
    if args.export_predictions:
        resolved_pred_device = str(metrics.get("resolved_device", args.device))
        target_transform_by_target = {
            str(k): v for k, v in (metrics.get("target_transform_by_target", {}) or {}).items()
        }
        calib_long_by_col: dict[str, pd.DataFrame] = {}
        calib_truth_by_target: dict[str, pd.DataFrame] = {}
        for split in splits:
            split_start = time.perf_counter()
            pred_df = _predict_split_frame(
                base_dir=base_dir,
                bundle=args.bundle,
                split=split,
                models_by_target=models_by_target,
                target_cols=target_cols,
                target_transform_by_target=target_transform_by_target,
                resolved_device=resolved_pred_device,
            )
            out_path = pred_dir / f"{file_tag}_{split}.parquet"
            pred_df.to_parquet(out_path, index=False)
            prediction_paths[split] = out_path
            prediction_export_seconds_by_split[split] = float(time.perf_counter() - split_start)

            if args.export_predictions_long:
                long_split_start = time.perf_counter()
                split_df, _ = _load_bundle_split_df(base_dir=base_dir, bundle=args.bundle, split=split)
                long_by_col, decay_by_col, crossing_by_col = _predict_split_long_multistep(
                    base_dir=base_dir,
                    bundle=args.bundle,
                    split=split,
                    models_by_target=models_by_target,
                    target_cols=target_cols,
                    target_transform_by_target=target_transform_by_target,
                    horizon_hours=args.forecast_horizon_hours,
                    model_name=args.model_name,
                    resolved_device=resolved_pred_device,
                )
                if split == "val":
                    calib_long_by_col = {k: v.copy() for k, v in long_by_col.items()}
                    for tgt in target_cols:
                        if tgt in split_df.columns:
                            tdf = split_df.loc[:, ["timestamp_utc", tgt]].rename(
                                columns={"timestamp_utc": "target_time_utc", tgt: "y_true"}
                            )
                            tdf["target_time_utc"] = pd.to_datetime(tdf["target_time_utc"], utc=True, errors="coerce")
                            calib_truth_by_target[tgt] = tdf
                if split == "test" and calib_long_by_col:
                    for pred_col, test_long in list(long_by_col.items()):
                        calib_long = calib_long_by_col.get(pred_col)
                        if calib_long is None:
                            continue
                        tgt_for_pred = _target_for_pred_column(pred_col)
                        if not tgt_for_pred or tgt_for_pred not in split_df.columns:
                            continue
                        q_cols = [c for c in test_long.columns if c.startswith("p") and c[1:].isdigit()]
                        if not q_cols:
                            continue
                        calib_h1 = calib_long.loc[calib_long["lead_time_h"] == 1].copy()
                        calib_h1["target_time_utc"] = pd.to_datetime(calib_h1["target_time_utc"], utc=True, errors="coerce")
                        truth = calib_truth_by_target.get(tgt_for_pred)
                        if truth is None or truth.empty:
                            continue
                        calib_join = calib_h1.merge(truth, on="target_time_utc", how="inner")
                        if calib_join.empty:
                            continue
                        q_dict = {c: pd.to_numeric(calib_join[c], errors="coerce").to_numpy(dtype=float) for c in q_cols}
                        alphas = [float(int(c[1:])) / 100.0 for c in q_cols]
                        shifts = calculate_conformal_shifts(
                            y_true_calib=pd.to_numeric(calib_join["y_true"], errors="coerce").to_numpy(dtype=float),
                            q_preds_calib_dict=q_dict,
                            alphas=alphas,
                        )
                        q_test = {c: pd.to_numeric(test_long[c], errors="coerce").to_numpy(dtype=float) for c in q_cols}
                        q_cal = apply_conformal_shifts(q_test, shifts)
                        for c in q_cols:
                            if c in q_cal:
                                test_long[c] = q_cal[c]
                        if "p50" in test_long.columns:
                            test_long["predicted_value"] = pd.to_numeric(test_long["p50"], errors="coerce")
                        long_by_col[pred_col] = test_long
                prediction_long_paths[split] = {}
                for pred_col, long_df in long_by_col.items():
                    long_path = pred_dir / f"{file_tag}_{split}_{pred_col}_long.parquet"
                    long_df.to_parquet(long_path, index=False)
                    prediction_long_paths[split][pred_col] = long_path

                    decay = decay_by_col[pred_col]
                    decay_path = report_dir / f"{file_tag}_{split}_{pred_col}_forecast_decay.csv"
                    decay.to_csv(decay_path, index=False)
                    leadtime_weighted_by_split.setdefault(split, {})[pred_col] = {
                        "mae_weighted": float(
                            weighted_metric_from_decay(
                                decay,
                                value_col="mae",
                                count_col="n",
                                start_lead=int(args.lead_weight_start),
                                end_lead=int(args.lead_weight_end),
                                max_weight=float(args.lead_weight_max),
                            )
                        )
                    }
                    pinball_cols = [c for c in decay.columns if c.startswith("pinball_p")]
                    for dcol in pinball_cols:
                        qcol = dcol.removeprefix("pinball_")
                        leadtime_weighted_by_split.setdefault(split, {}).setdefault(pred_col, {})[
                            f"pinball_{qcol}_weighted"
                        ] = float(
                            weighted_metric_from_decay(
                                decay,
                                value_col=dcol,
                                count_col="n",
                                start_lead=int(args.lead_weight_start),
                                end_lead=int(args.lead_weight_end),
                                max_weight=float(args.lead_weight_max),
                            )
                        )
                    mae_plot_path = report_dir / f"{file_tag}_{split}_{pred_col}_mae_lead_1_24_48.png"
                    _plot_leadtime_mae_points(decay, mae_plot_path)

                    tgt_for_pred = _target_for_pred_column(pred_col)
                    crossing_payload = crossing_by_col.get(pred_col, {})
                    metrics[f"{split}_{pred_col}_crossing_rate_before_repair"] = crossing_payload.get(
                        "crossing_rate_before_repair"
                    )
                    metrics[
                        f"{split}_{pred_col}_max_crossing_violation_before_repair"
                    ] = crossing_payload.get("max_crossing_violation_before_repair")
                    if tgt_for_pred:
                        gate_hour = gate_hour_for_target(tgt_for_pred)
                        if gate_hour is not None and tgt_for_pred in split_df.columns:
                            gate_metrics = compute_gate_closure_metrics(
                                long_df,
                                truth_df=split_df[["timestamp_utc", tgt_for_pred]].copy(),
                                y_true_col=tgt_for_pred,
                                y_pred_col="p50" if "p50" in long_df.columns else "predicted_value",
                                gate_hour_local=gate_hour,
                                timezone="Europe/Berlin",
                            )
                            gate_closure_metrics_by_split.setdefault(split, {})[pred_col] = gate_metrics
                        if tgt_for_pred in split_df.columns:
                            hb = compute_horizon_bucket_metrics(
                                long_df,
                                split_df[["timestamp_utc", tgt_for_pred]].copy(),
                                target_col=tgt_for_pred,
                                y_pred_col="p50" if "p50" in long_df.columns else "predicted_value",
                                quantile_cols=[c for c in long_df.columns if c.startswith("p") and c[1:].isdigit()],
                                timezone="Europe/Berlin",
                                max_horizon=int(args.forecast_horizon_hours),
                            )
                            for hk, hv in hb.items():
                                metrics[f"{split}_horizon_bucket_{pred_col}_{hk}"] = hv
                            hb_df = horizon_bucket_metrics_to_table(hb, split=split, target_col=tgt_for_pred)
                            hb_csv = report_dir / f"{args.bundle}_{tgt_for_pred}_{split}_horizon_bucket_metrics.csv"
                            hb_df.to_csv(hb_csv, index=False)
                long_export_seconds_by_split[split] = float(time.perf_counter() - long_split_start)

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
    total_elapsed = time.perf_counter() - total_start
    export_elapsed = time.perf_counter() - export_start

    metrics["timing_train_eval_seconds"] = float(train_eval_elapsed)
    metrics["timing_export_seconds"] = float(export_elapsed)
    metrics["timing_export_prediction_seconds_by_split"] = prediction_export_seconds_by_split
    metrics["timing_export_long_seconds_by_split"] = long_export_seconds_by_split
    metrics["leadtime_weighting"] = {
        "start_lead_h": int(args.lead_weight_start),
        "end_lead_h": int(args.lead_weight_end),
        "max_weight": float(args.lead_weight_max),
    }
    metrics["leadtime_weighted_by_split"] = leadtime_weighted_by_split
    primary_pred_col = _pred_column_names_for_target(str(metrics.get("target_col", "")))[0]
    metrics["leadtime_mae_val_weighted"] = (
        leadtime_weighted_by_split.get("val", {}).get(primary_pred_col, {}).get("mae_weighted")
    )
    metrics["leadtime_mae_test_weighted"] = (
        leadtime_weighted_by_split.get("test", {}).get(primary_pred_col, {}).get("mae_weighted")
    )
    metrics["decision_weighted_mae_val"] = metrics.get("leadtime_mae_val_weighted")
    metrics["decision_weighted_mae_test"] = metrics.get("leadtime_mae_test_weighted")
    metrics["crossing_rate_before_repair_val"] = metrics.get(
        f"val_{primary_pred_col}_crossing_rate_before_repair"
    )
    metrics["max_crossing_violation_before_repair_val"] = metrics.get(
        f"val_{primary_pred_col}_max_crossing_violation_before_repair"
    )
    metrics["crossing_rate_before_repair_test"] = metrics.get(
        f"test_{primary_pred_col}_crossing_rate_before_repair"
    )
    metrics["max_crossing_violation_before_repair_test"] = metrics.get(
        f"test_{primary_pred_col}_max_crossing_violation_before_repair"
    )
    val_weighted = leadtime_weighted_by_split.get("val", {}).get(primary_pred_col, {})
    test_weighted = leadtime_weighted_by_split.get("test", {}).get(primary_pred_col, {})
    for key, value in val_weighted.items():
        if key.startswith("pinball_p") and key.endswith("_weighted"):
            qcol = key.removeprefix("pinball_").removesuffix("_weighted")
            metrics[f"leadtime_pinball_{qcol}_val_weighted"] = value
            metrics[f"decision_weighted_pinball_{qcol}_val"] = value
    for key, value in test_weighted.items():
        if key.startswith("pinball_p") and key.endswith("_weighted"):
            qcol = key.removeprefix("pinball_").removesuffix("_weighted")
            metrics[f"leadtime_pinball_{qcol}_test_weighted"] = value
            metrics[f"decision_weighted_pinball_{qcol}_test"] = value
    metrics["gate_closure_metrics_by_split"] = gate_closure_metrics_by_split
    metrics["timing_total_seconds"] = float(total_elapsed)
    # Export early-stopping diagnostics per target/lead/quantile for QA.
    best_it_rows = metrics.get("xgb_best_iteration_by_target_lead_quantile", [])
    best_it_csv_path = report_dir / f"{file_tag}_best_iteration_by_lead.csv"
    if isinstance(best_it_rows, list) and best_it_rows:
        best_it_df = pd.DataFrame(best_it_rows)
        best_it_df.to_csv(best_it_csv_path, index=False)
        metrics["xgb_best_iteration_csv"] = str(best_it_csv_path.resolve())
    metrics_json_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("[OK] XGBoost training finished.")
    print(f"- Base dir: {base_dir}")
    print(f"- Run dir: {run_dir}")
    print(f"- Bundle: {args.bundle}")
    print(f"- Device: {args.device}")
    print(f"- Seed: {args.seed}")
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
