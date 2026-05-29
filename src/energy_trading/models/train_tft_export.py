"""Train TFT model on prepared bundles and export backtest-compatible outputs."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_trading.models.prepare_ml_bundles import BundleName
from energy_trading.models.train_xgboost_export import (
    _add_dynamics_features,
    _pred_column_names_for_target,
    _resolve_targets,
)
from energy_trading.models.training_policy import (
    resolve_feature_columns_for_target,
    resolve_tft_params_for_target,
)
from energy_trading.evaluation.metrics import (
    compute_forecast_metrics,
    compute_gate_closure_metrics,
    gate_hour_for_target,
)
from energy_trading.evaluation.lead_weighting import weighted_metric_from_decay
from energy_trading.evaluation.tensorboard_utils import (
    tensorboard_log_root,
    tensorboard_target_version,
)
from energy_trading.evaluation.conformal_calibration import (
    apply_conformal_shifts,
    calculate_conformal_shifts,
)


CALENDAR_FEATURES = {
    "hour",
    "weekday",
    "dayofweek",
    "month",
    "day",
    "is_weekend",
    "weekofyear",
}
ALLOWED_KNOWN_CATEGORICALS = {"hour", "weekday", "month"}
LAG_FEATURE_RE = re.compile(r".*_lag_\d+h$")
PASSTHROUGH_REAL_RE = re.compile(r".*(_sin|_cos|_slog1p)$")
VOLATILE_PRICE_RE = re.compile(r"(?:^|_)(da_price|afrr_.*price|.*_price_)", re.IGNORECASE)
VOLATILE_VOLUME_FEATURES = {
    "planned_outages_mw",
    "unplanned_outages_mw",
    "total_outages_mw",
    "wind_offshore_forecast_update",
}
TFT_CLIP_MIN = -5.0
TFT_CLIP_MAX = 5.0
QUANTILES: list[float] = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
LOGGER = logging.getLogger(__name__)


def _qcol(q: float) -> str:
    return f"p{int(round(q * 100)):02d}"


def _run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


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


def _resolve_torch_device(requested: str) -> tuple[str, str]:
    """Return (torch_device, lightning_accelerator) with MPS fallback."""
    req = (requested or "mps").strip().lower()
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for TFT training.") from exc

    if req == "mps":
        if torch.backends.mps.is_available():
            return "mps", "mps"
        return "cpu", "cpu"
    if req == "cuda":
        if torch.cuda.is_available():
            return "cuda", "gpu"
        return "cpu", "cpu"
    return "cpu", "cpu"


def _classify_feature_columns(feature_columns: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Classify TFT inputs with strict pruning + known/unknown split.

    Rules:
    - Drop all explicit lag features (`*_lag_<h>h`).
    - known categoricals: calendar ids + binary/flag-style regime indicators.
    - known reals: remaining features containing `forecast` plus forecast-ramp proxies.
    - unknown reals: everything else remaining.
    """
    pruned: list[str] = []
    seen: set[str] = set()
    for c in feature_columns:
        if LAG_FEATURE_RE.match(c):
            continue
        if c in seen:
            continue
        seen.add(c)
        pruned.append(c)

    known_categoricals: list[str] = []
    for c in pruned:
        if c in ALLOWED_KNOWN_CATEGORICALS:
            known_categoricals.append(c)
            continue
        if c.startswith("is_") or c == "holiday_severity":
            known_categoricals.append(c)

    # Explicit assignment for dynamics columns:
    # - forecast/ramp signals are known at prediction time
    # - imbalance momentum from lagged realized signals stays unknown
    explicit_unknown = {"nrv_velocity_1h"}
    explicit_known = {"load_ramp_signed_1h", "load_ramp_abs_1h", "res_load_ramp_signed_1h"}

    known_reals: list[str] = []
    for c in pruned:
        if c in known_categoricals or c in explicit_unknown:
            continue
        lc = c.lower()
        if c in explicit_known or ("forecast" in lc):
            known_reals.append(c)
    unknown_reals = [c for c in pruned if c not in known_categoricals and c not in known_reals]
    return known_reals, known_categoricals, unknown_reals


def _fit_split_scalers(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_col: str,
    categorical_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Fit train-only robust scalers with passthrough + clipped volatile prices.

    Strategy:
    - Real features ending in `_sin`, `_cos`, `_slog1p` bypass scaling.
    - Remaining real features are RobustScaler-fitted on train only.
    - Highly volatile price/volume features are clipped post-scale to improve TFT stability.
    """
    categorical_columns = list(categorical_columns or [])
    categorical_cols = [c for c in categorical_columns if c in feature_columns]
    real_cols = [c for c in feature_columns if c not in categorical_cols]
    passthrough_real_cols = [c for c in real_cols if PASSTHROUGH_REAL_RE.match(c)]
    scaled_real_cols = [c for c in real_cols if c not in passthrough_real_cols]
    clipped_volatile_cols = [
        c for c in scaled_real_cols if VOLATILE_PRICE_RE.search(c) or c in VOLATILE_VOLUME_FEATURES
    ]

    X_tr = train_df[feature_columns].copy()
    X_va = val_df[feature_columns].copy()
    X_te = test_df[feature_columns].copy()
    # split-first, train-fit imputation
    if real_cols:
        med = X_tr[real_cols].median(numeric_only=True)
        X_tr.loc[:, real_cols] = X_tr.loc[:, real_cols].fillna(med)
        X_va.loc[:, real_cols] = X_va.loc[:, real_cols].fillna(med)
        X_te.loc[:, real_cols] = X_te.loc[:, real_cols].fillna(med)
    if categorical_cols:
        mode_df = X_tr[categorical_cols].mode(dropna=True)
        if mode_df.empty:
            cat_fill = pd.Series({c: "0" for c in categorical_cols})
        else:
            cat_fill = mode_df.iloc[0]
        X_tr.loc[:, categorical_cols] = X_tr.loc[:, categorical_cols].fillna(cat_fill).fillna("0")
        X_va.loc[:, categorical_cols] = X_va.loc[:, categorical_cols].fillna(cat_fill).fillna("0")
        X_te.loc[:, categorical_cols] = X_te.loc[:, categorical_cols].fillna(cat_fill).fillna("0")

    if scaled_real_cols:
        f_scaler = RobustScaler()
        f_scaler.fit(X_tr[scaled_real_cols].to_numpy(dtype=float))
    else:
        f_scaler = None

    def _scaled(df: pd.DataFrame, X_in: pd.DataFrame) -> pd.DataFrame:
        out = X_in.copy()
        if scaled_real_cols and f_scaler is not None:
            # Ensure numeric reals can hold scaled float values (pandas >=2.2
            # raises on lossy int8/int16 assignment).
            out = out.astype({c: "float64" for c in scaled_real_cols}, copy=False)
            out.loc[:, scaled_real_cols] = f_scaler.transform(out[scaled_real_cols].to_numpy(dtype=float))
            if clipped_volatile_cols:
                out.loc[:, clipped_volatile_cols] = np.clip(
                    out[clipped_volatile_cols].to_numpy(dtype=float),
                    TFT_CLIP_MIN,
                    TFT_CLIP_MAX,
                )
        if passthrough_real_cols:
            out = out.astype({c: "float64" for c in passthrough_real_cols}, copy=False)
        out[target_col] = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
        out["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        return out

    tr_sc = _scaled(train_df, X_tr)
    va_sc = _scaled(val_df, X_va)
    te_sc = _scaled(test_df, X_te)

    scaler_payload = {
        "feature_scaler": f_scaler,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_cols,
        "real_feature_columns": real_cols,
        "scaled_real_feature_columns": scaled_real_cols,
        "passthrough_real_feature_columns": passthrough_real_cols,
        # Backward-compatible key name kept for downstream consumers.
        "clipped_price_feature_columns": clipped_volatile_cols,
        "clipped_volatile_feature_columns": clipped_volatile_cols,
        "target_col": target_col,
    }
    return tr_sc, va_sc, te_sc, scaler_payload


def _build_long_prediction_table(
    *,
    prediction: np.ndarray,
    decoder_time_idx: np.ndarray,
    idx_to_ts: dict[int, pd.Timestamp],
    model_name: str,
    quantiles: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    # prediction shape typically [batch, decoder, quantile] or [batch, decoder]
    pred = np.asarray(prediction, dtype=float)
    if pred.ndim == 2:
        pred = pred[:, :, None]
    if pred.ndim != 3:
        raise ValueError(f"Unexpected TFT prediction shape: {pred.shape}")
    if pred.shape[2] != len(quantiles):
        raise ValueError(
            f"TFT quantile dimension ({pred.shape[2]}) does not match requested quantiles ({len(quantiles)})."
        )

    dti = np.asarray(decoder_time_idx)
    if dti.ndim != 2 or dti.shape != pred[:, :, 0].shape:
        raise ValueError("decoder_time_idx shape mismatch against prediction tensor.")

    # Enforce monotonic quantile ordering to avoid quantile crossing.
    pred_inv = np.sort(pred, axis=2)
    q_cols = [_qcol(q) for q in quantiles]
    median_col = _qcol(0.5)

    for i in range(pred_inv.shape[0]):
        # snapshot is one step before first decoder hour
        snapshot_idx = int(dti[i, 0]) - 1
        snapshot_ts = idx_to_ts.get(snapshot_idx)
        if snapshot_ts is None:
            continue
        for j in range(pred_inv.shape[1]):
            target_idx = int(dti[i, j])
            target_ts = idx_to_ts.get(target_idx)
            if target_ts is None:
                continue
            q_vals = pred_inv[i, j, :]
            row = {
                "snapshot_time_utc": snapshot_ts,
                "target_time_utc": target_ts,
                "lead_time_h": int(j + 1),
                "model_name": model_name,
            }
            row.update({c: float(v) for c, v in zip(q_cols, q_vals)})
            row["predicted_value"] = float(row[median_col])
            rows.append(
                row
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "snapshot_time_utc",
                "target_time_utc",
                "lead_time_h",
                "model_name",
                *q_cols,
                "predicted_value",
            ]
        )
    out = out.sort_values(["snapshot_time_utc", "lead_time_h", "target_time_utc"]).reset_index(drop=True)
    return out


def _clip_activation_rate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Clip activation-rate prediction columns to [0, 1] in long-format output."""
    if df.empty:
        return df
    out = df.copy()
    q_cols = [_qcol(q) for q in QUANTILES]
    for c in ("predicted_value", *q_cols):
        if c in out.columns:
            out[c] = np.clip(pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float), 0.0, 1.0)
    return out


def _leadtime_mae(
    pred_long: pd.DataFrame,
    *,
    true_h1: pd.Series,
    horizon_hours: int,
) -> pd.DataFrame:
    if pred_long.empty:
        return pd.DataFrame(columns=["lead_time_h", "n", "mae", "rmse"])
    base = pd.to_numeric(true_h1, errors="coerce").reset_index(drop=True)
    truth_by_lead = {h: base.shift(-(h - 1)) for h in range(1, horizon_hours + 1)}
    pred_col = "p50" if "p50" in pred_long.columns else "predicted_value"
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
    for lead in range(1, horizon_hours + 1):
        p = pd.to_numeric(pred_long.loc[pred_long["lead_time_h"] == lead, pred_col], errors="coerce").reset_index(drop=True)
        t = truth_by_lead[lead].iloc[: len(p)]
        m = p.notna() & t.notna()
        n = int(m.sum())
        mae = float(mean_absolute_error(t[m], p[m])) if n > 0 else np.nan
        rmse = float(np.sqrt(mean_squared_error(t[m], p[m]))) if n > 0 else np.nan
        row: dict[str, float] = {"lead_time_h": float(lead), "n": float(n), "mae": mae, "rmse": rmse}
        for q_col, tau in pinball_qs:
            q_pred = pd.to_numeric(
                pred_long.loc[pred_long["lead_time_h"] == lead, q_col],
                errors="coerce",
            ).reset_index(drop=True)
            qm = q_pred.notna() & t.notna()
            if int(qm.sum()) > 0:
                e = t[qm].to_numpy(dtype=float) - q_pred[qm].to_numpy(dtype=float)
                loss = np.maximum(float(tau) * e, (float(tau) - 1.0) * e)
                row[f"pinball_{q_col}"] = float(np.mean(loss))
            else:
                row[f"pinball_{q_col}"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_pinball_from_decay(decay_df: pd.DataFrame) -> float:
    """Compute full-horizon unweighted mean pinball from lead-decay table.

    Per quantile column `pinball_pXX`, first aggregate across leads weighted by
    observation count `n`. Then average quantile means equally.
    """
    if decay_df is None or decay_df.empty or "n" not in decay_df.columns:
        return float("nan")
    n = pd.to_numeric(decay_df["n"], errors="coerce").to_numpy(dtype=float)
    valid_n = np.isfinite(n) & (n > 0)
    if not bool(valid_n.any()):
        return float("nan")

    pinball_cols = [c for c in decay_df.columns if c.startswith("pinball_p")]
    if not pinball_cols:
        return float("nan")

    q_means: list[float] = []
    for col in pinball_cols:
        v = pd.to_numeric(decay_df[col], errors="coerce").to_numpy(dtype=float)
        m = valid_n & np.isfinite(v)
        if not bool(m.any()):
            continue
        q_means.append(float(np.average(v[m], weights=n[m])))
    if not q_means:
        return float("nan")
    return float(np.mean(q_means))


def _save_tft_attention_plot(
    *,
    tft_model,
    dataset,
    out_path: Path,
    batch_size: int,
    num_workers: int,
) -> str | None:
    """Export TFT interpretation/attention plot for thesis diagnostics.

    Uses `predict(mode="raw", return_x=True)` + `interpret_output`.
    Falls back to manual attention plotting if `plot_interpretation` shape/API differs.
    Returns absolute output path string on success, else None.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Could not import matplotlib for attention plotting: %s", exc)
        return None

    try:
        dl = dataset.to_dataloader(
            train=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
        )
        raw_out = tft_model.predict(dl, mode="raw", return_x=True)
        if hasattr(raw_out, "output") and hasattr(raw_out, "x"):
            network_out = raw_out.output
        elif isinstance(raw_out, tuple) and len(raw_out) >= 2:
            network_out = raw_out[0]
        else:
            raise TypeError(f"Unsupported TFT raw predict() return type: {type(raw_out)}")

        interpretation = tft_model.interpret_output(network_out, reduction="mean")
        fig_obj = tft_model.plot_interpretation(interpretation)

        fig = None
        if hasattr(fig_obj, "savefig"):
            fig = fig_obj
        elif isinstance(fig_obj, dict):
            for v in fig_obj.values():
                if hasattr(v, "savefig"):
                    fig = v
                    break
                if hasattr(v, "figure"):
                    fig = v.figure
                    break
        elif isinstance(fig_obj, (list, tuple)) and fig_obj:
            v = fig_obj[0]
            if hasattr(v, "savefig"):
                fig = v
            elif hasattr(v, "figure"):
                fig = v.figure

        if fig is None:
            # Fallback: manual attention visualization.
            att = interpretation.get("attention") if isinstance(interpretation, dict) else None
            if att is None:
                raise ValueError("interpret_output produced no 'attention' entry.")
            att_np = np.asarray(att.detach().cpu() if hasattr(att, "detach") else att, dtype=float)
            if att_np.ndim == 0:
                att_np = att_np.reshape(1)
            if att_np.ndim > 1:
                # average over non-time dimensions
                reduce_axes = tuple(range(att_np.ndim - 1))
                att_np = np.nanmean(att_np, axis=reduce_axes)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(np.arange(att_np.shape[0]), att_np, linewidth=2)
            ax.set_title("TFT Attention Weights (mean)")
            ax.set_xlabel("Relative Time Step")
            ax.set_ylabel("Attention Weight")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(out_path.resolve())
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Could not export TFT attention plot: %s", exc)
        return None


def _save_tft_interpretation_artifacts(
    *,
    tft_model,
    dataset,
    out_dir: Path,
    prefix: str,
    batch_size: int,
    num_workers: int,
    max_encoder_length: int,
) -> dict[str, str]:
    """Export thesis-oriented TFT interpretation artifacts.

    Outputs (if possible):
    - Combined interpretation plot from pytorch-forecasting.
    - Attention-history line plot focused on encoder horizon (t-enc ... t-1).
    - Variable-importance bar plot (top features).
    """
    out: dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Could not import matplotlib for interpretation artifacts: %s", exc)
        return out

    try:
        dl = dataset.to_dataloader(
            train=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
        )
        raw_out = tft_model.predict(dl, mode="raw", return_x=True)
        if hasattr(raw_out, "output") and hasattr(raw_out, "x"):
            network_out = raw_out.output
        elif isinstance(raw_out, tuple) and len(raw_out) >= 2:
            network_out = raw_out[0]
        else:
            raise TypeError(f"Unsupported TFT raw predict() return type: {type(raw_out)}")

        interpretation = tft_model.interpret_output(network_out, reduction="mean")

        # 1) framework interpretation plot
        try:
            fig_obj = tft_model.plot_interpretation(interpretation)
            fig = None
            if hasattr(fig_obj, "savefig"):
                fig = fig_obj
            elif isinstance(fig_obj, dict):
                for v in fig_obj.values():
                    if hasattr(v, "savefig"):
                        fig = v
                        break
                    if hasattr(v, "figure"):
                        fig = v.figure
                        break
            elif isinstance(fig_obj, (list, tuple)) and fig_obj:
                v = fig_obj[0]
                if hasattr(v, "savefig"):
                    fig = v
                elif hasattr(v, "figure"):
                    fig = v.figure
            if fig is not None:
                p = out_dir / f"{prefix}_attention_interpretation.png"
                fig.savefig(p, dpi=300, bbox_inches="tight")
                plt.close(fig)
                out["attention_plot_path"] = str(p.resolve())
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("plot_interpretation export failed: %s", exc)

        # 2) encoder attention history (t-enc ... t-1)
        try:
            att = interpretation.get("attention") if isinstance(interpretation, dict) else None
            if att is not None:
                att_np = np.asarray(att.detach().cpu() if hasattr(att, "detach") else att, dtype=float)
                if att_np.ndim == 1:
                    att_hist = att_np
                else:
                    # reduce all non-time axes
                    reduce_axes = tuple(range(att_np.ndim - 1))
                    att_hist = np.nanmean(att_np, axis=reduce_axes)
                if att_hist.ndim == 1 and att_hist.size > 0:
                    if att_hist.size >= max_encoder_length:
                        hist = att_hist[-max_encoder_length:]
                    else:
                        hist = att_hist
                    x = np.arange(-len(hist), 0, 1)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(x, hist, linewidth=2)
                    ax.set_title("TFT Attention over Encoder History")
                    ax.set_xlabel("Relative time step (hours, t=0 forecast origin)")
                    ax.set_ylabel("Attention weight")
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    p = out_dir / f"{prefix}_attention_history_tminus.png"
                    fig.savefig(p, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    out["attention_history_plot_path"] = str(p.resolve())
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Attention history export failed: %s", exc)

        # 3) variable importance summary
        try:
            blocks = []
            if isinstance(interpretation, dict):
                for key in ["static_variables", "encoder_variables", "decoder_variables"]:
                    val = interpretation.get(key)
                    if val is None:
                        continue
                    arr = np.asarray(val.detach().cpu() if hasattr(val, "detach") else val, dtype=float)
                    if arr.ndim == 0:
                        continue
                    # average all non-feature axes; keep last axis as features
                    if arr.ndim > 1:
                        reduce_axes = tuple(range(arr.ndim - 1))
                        arr = np.nanmean(arr, axis=reduce_axes)
                    blocks.append((key, arr))
            if blocks:
                feat_scores: dict[str, float] = {}
                for key, arr in blocks:
                    for i, v in enumerate(np.asarray(arr).ravel().tolist()):
                        feat_scores[f"{key}[{i}]"] = float(v)
                top = sorted(feat_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:20]
                if top:
                    labels = [k for k, _ in top][::-1]
                    vals = [v for _, v in top][::-1]
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(labels, vals)
                    ax.set_title("TFT Variable Importance (Interpretation Summary)")
                    ax.set_xlabel("Importance score")
                    fig.tight_layout()
                    p = out_dir / f"{prefix}_feature_importance.png"
                    fig.savefig(p, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    out["feature_importance_plot_path"] = str(p.resolve())
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Feature-importance export failed: %s", exc)

    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Could not export TFT interpretation artifacts: %s", exc)
    return out


def _train_tft(
    *,
    base_dir: Path,
    bundle: BundleName,
    target_col: str | None,
    run_dir: Path,
    model_name: str,
    max_encoder_length: int = 168,
    max_prediction_length: int = 48,
    seed: int = 42,
    requested_device: str = "mps",
    num_workers: int = 0,
    cleanup_lightning_checkpoints: bool = False,
    learning_rate: float = 1e-3,
    gradient_clip_val: float = 0.05,
    lead_weight_start: int = 16,
    lead_weight_end: int = 48,
    lead_weight_max: float = 2.0,
    hidden_size_override: int | None = None,
    attention_head_size_override: int | None = None,
    dropout_override: float | None = None,
    max_epochs_override: int | None = None,
    early_stopping_patience_override: int | None = None,
    precision_mode: str = "auto",
) -> dict[str, object]:
    total_start = time.perf_counter()
    try:
        import lightning.pytorch as pl
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data.encoders import EncoderNormalizer, NaNLabelEncoder
        from pytorch_forecasting.metrics import QuantileLoss
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from lightning.pytorch.loggers import TensorBoardLogger
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "TFT dependencies missing. Install pytorch, lightning, pytorch-forecasting."
        ) from exc

    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)
    random.seed(seed)

    train_df, bcfg = _load_bundle_split_df(base_dir=base_dir, bundle=bundle, split="train")
    val_df, _ = _load_bundle_split_df(base_dir=base_dir, bundle=bundle, split="val")
    test_df, _ = _load_bundle_split_df(base_dir=base_dir, bundle=bundle, split="test")
    feature_columns = list(bcfg["features"])

    available_targets = [c for c in bcfg["targets"] if c in train_df.columns]
    selected_targets = _resolve_targets(bundle, available_targets, target_col)
    tgt = selected_targets[0]

    train_df[tgt] = pd.to_numeric(train_df[tgt], errors="coerce")
    val_df[tgt] = pd.to_numeric(val_df[tgt], errors="coerce")
    test_df[tgt] = pd.to_numeric(test_df[tgt], errors="coerce")
    train_df = train_df.loc[train_df[tgt].notna()].copy()
    val_df = val_df.loc[val_df[tgt].notna()].copy()
    test_df = test_df.loc[test_df[tgt].notna()].copy()

    # Keep common feature engineering parity with XGBoost path.
    train_df = _add_dynamics_features(train_df, prune_midterm_lags=True)
    val_df = _add_dynamics_features(val_df, prune_midterm_lags=True)
    test_df = _add_dynamics_features(test_df, prune_midterm_lags=True)
    all_feature_columns = [
        c
        for c in train_df.columns
        if (
            c in feature_columns
            or c.endswith("_ramp_signed_1h")
            or c.endswith("_ramp_abs_1h")
            or c == "nrv_velocity_1h"
            or "_ramp_x_" in c
        )
    ]
    all_feature_columns = [c for c in all_feature_columns if c in train_df.columns]
    target_feature_variant, feature_columns = resolve_feature_columns_for_target(all_feature_columns, tgt)
    known_reals, known_categoricals, unknown_reals = _classify_feature_columns(feature_columns)
    feature_columns = list(dict.fromkeys([*known_reals, *known_categoricals, *unknown_reals]))

    tr_sc, va_sc, te_sc, scalers = _fit_split_scalers(
        train_df,
        val_df,
        test_df,
        feature_columns=feature_columns,
        target_col=tgt,
        categorical_columns=known_categoricals,
    )

    model_dir = run_dir / "models"
    pred_dir = run_dir / "predictions"
    metrics_dir = run_dir / "metrics"
    report_dir = run_dir / "reports"
    for d in (model_dir, pred_dir, metrics_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)

    scaler_path = model_dir / f"{bundle}_{tgt}_tft_scaler.pkl"
    joblib.dump(scalers, scaler_path)

    tr_sc["split"] = "train"
    va_sc["split"] = "val"
    te_sc["split"] = "test"
    full_sc = pd.concat([tr_sc, va_sc, te_sc], axis=0)
    ts_col = "snapshot_time_utc" if "snapshot_time_utc" in full_sc.columns else "timestamp_utc"
    full_sc[ts_col] = pd.to_datetime(full_sc[ts_col], utc=True, errors="coerce")
    full_sc = full_sc.sort_values(ts_col).reset_index(drop=True)

    # Ensure canonical UTC timestamp column for downstream usage.
    if ts_col != "timestamp_utc":
        full_sc["timestamp_utc"] = full_sc[ts_col]

    # Time-series integrity checks on timestamp column.
    ts_full = pd.to_datetime(full_sc["timestamp_utc"], utc=True, errors="coerce")
    if ts_full.isna().any():
        raise ValueError("timestamp_utc contains invalid values; cannot build continuous hour-based time_idx.")
    t0 = ts_full.min()
    full_sc["time_idx"] = ((ts_full - t0).dt.total_seconds() // 3600).astype(int)
    full_sc["series_id"] = "energy_series"

    # Guard against accidental duplicate index points inside a series.
    dup_mask = full_sc.duplicated(subset=["series_id", "time_idx"], keep="last")
    dup_n = int(dup_mask.sum())
    if dup_n > 0:
        print(f"[WARN] Dropping {dup_n} duplicate (series_id, time_idx) rows in TFT input.")
        full_sc = full_sc.loc[~dup_mask].copy()

    # Sanity checks for index semantics.
    if not pd.api.types.is_integer_dtype(full_sc["time_idx"]):
        full_sc["time_idx"] = full_sc["time_idx"].astype(int)
    if bool((full_sc["time_idx"] < 0).any()):
        raise ValueError("Negative values found in time_idx after timestamp-to-hour conversion.")

    if "time_idx" in known_reals:
        known_reals.remove("time_idx")

    # Calendar features must be discrete classes for embedding learning.
    for c in known_categoricals:
        if c in full_sc.columns:
            ser = pd.to_numeric(full_sc[c], errors="coerce")
            if c in ALLOWED_KNOWN_CATEGORICALS:
                ser = ser.round().astype("Int64")
            else:
                ser = ser.round(6)
            full_sc[c] = ser.where(ser.isna(), ser.astype(str))

    idx_train_end = int(full_sc.loc[full_sc["split"] == "train", "time_idx"].max())
    idx_val_end = int(full_sc.loc[full_sc["split"] == "val", "time_idx"].max())

    training = TimeSeriesDataSet(
        full_sc.loc[full_sc["time_idx"] <= idx_train_end].copy(),
        time_idx="time_idx",
        target=tgt,
        group_ids=["series_id"],
        max_encoder_length=max_encoder_length,
        min_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx", *known_reals],
        time_varying_known_categoricals=known_categoricals,
        time_varying_unknown_reals=[tgt, *unknown_reals],
        target_normalizer=EncoderNormalizer(method="robust", center=True),
        categorical_encoders={c: NaNLabelEncoder(add_nan=True) for c in known_categoricals},
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        full_sc.loc[full_sc["time_idx"] <= idx_val_end].copy(),
        min_prediction_idx=idx_train_end + 1,
        stop_randomization=True,
        # IMPORTANT:
        # `predict=True` would keep only the last prediction window per series,
        # which collapses val/test exports to a single forecast block.
        # We need full rolling coverage over the whole split.
        predict=False,
    )
    testing = TimeSeriesDataSet.from_dataset(
        training,
        full_sc.copy(),
        min_prediction_idx=idx_val_end + 1,
        stop_randomization=True,
        # Keep all rolling windows in test for full forecast-vintage export.
        predict=False,
    )
    # Build compact lookup once, then release large intermediate tables.
    idx_to_ts = {
        int(i): pd.Timestamp(ts)
        for i, ts in zip(full_sc["time_idx"], pd.to_datetime(full_sc["timestamp_utc"], utc=True, errors="coerce"))
    }
    del full_sc, tr_sc, va_sc, te_sc, train_df
    gc.collect()

    torch_device, accelerator = _resolve_torch_device(requested_device)
    if torch_device == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:  # pragma: no cover - older torch variants
            pass
    resolved_precision = str(precision_mode).strip().lower()
    if resolved_precision == "auto":
        resolved_precision = "16-mixed" if torch_device == "cuda" else "32-true"
    allowed_precisions = {"32-true", "16-mixed", "bf16-mixed"}
    if resolved_precision not in allowed_precisions:
        raise ValueError(
            f"Unsupported precision mode '{precision_mode}'. "
            f"Use one of: {sorted(allowed_precisions)} or 'auto'."
        )
    batch_size = 64
    train_loader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )

    class _EpochEtaCallback(pl.Callback):
        """Log ETA after first completed epoch."""

        def __init__(self, max_epochs: int) -> None:
            super().__init__()
            self.max_epochs = int(max_epochs)
            self._epoch_start_ts: float | None = None
            self._eta_logged = False

        def on_train_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[override]
            self._epoch_start_ts = time.time()

        def on_train_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
            if self._eta_logged:
                return
            if trainer.current_epoch != 0:
                return
            if self._epoch_start_ts is None:
                return
            elapsed_s = max(0.0, time.time() - self._epoch_start_ts)
            remaining_epochs = max(0, self.max_epochs - 1)
            eta_minutes = (elapsed_s * remaining_epochs) / 60.0
            LOGGER.info(
                "ETA for remaining %s epochs: ~ %.1f minutes",
                remaining_epochs,
                eta_minutes,
            )
            self._eta_logged = True

    base_tft_params = {
        "dropout": 0.1,
        "early_stopping_patience": 10.0,
        "max_epochs": 100.0,
    }
    target_tft_params = resolve_tft_params_for_target(tgt, base_tft_params)
    if dropout_override is not None:
        target_tft_params["dropout"] = float(dropout_override)
    if max_epochs_override is not None:
        target_tft_params["max_epochs"] = float(max_epochs_override)
    if early_stopping_patience_override is not None:
        target_tft_params["early_stopping_patience"] = float(early_stopping_patience_override)
    max_epochs = int(target_tft_params["max_epochs"])
    early_stopping_patience = int(target_tft_params["early_stopping_patience"])
    dropout = float(target_tft_params["dropout"])

    # Asymmetric regularization by target family:
    # - DA targets: medium capacity (memory-aware), light regularization.
    # - aFRR targets: constrained capacity, stronger regularization.
    is_afrr_target = "afrr" in tgt.lower()
    if is_afrr_target:
        hidden_size = 32
        effective_dropout = min(0.30, max(0.25, dropout))
        early_stopping_patience = min(4, max(3, early_stopping_patience))
    else:
        hidden_size = 64
        effective_dropout = float(dropout)
        early_stopping_patience = min(10, max(8, early_stopping_patience))
    if hidden_size_override is not None:
        hidden_size = int(hidden_size_override)
    attention_head_size = int(attention_head_size_override) if attention_head_size_override is not None else 4
    tb_root = tensorboard_log_root()
    tb_version = tensorboard_target_version(
        model_family="tft",
        bundle=bundle,
        target_col=tgt,
    )
    tb_logger = TensorBoardLogger(
        save_dir=str(tb_root),
        name=run_dir.name,
        version=tb_version,
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(model_dir),
        filename=f"{bundle}_{tgt}_best-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stopping_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=early_stopping_patience,
        min_delta=0.0,
        strict=True,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        precision=resolved_precision,
        gradient_clip_val=float(gradient_clip_val),
        enable_checkpointing=True,
        callbacks=[_EpochEtaCallback(max_epochs=max_epochs), checkpoint_cb, early_stopping_cb],
        logger=tb_logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=float(learning_rate),
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=effective_dropout,
        hidden_continuous_size=16,
        loss=QuantileLoss(quantiles=QUANTILES),
        output_size=len(QUANTILES),
        reduce_on_plateau_patience=3,
    )

    tft.to(torch_device)
    fit_start = time.perf_counter()
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    fit_seconds = time.perf_counter() - fit_start

    model_path = model_dir / f"{bundle}_{tgt}_tft_export_model.ckpt"
    best_model_path = checkpoint_cb.best_model_path
    if not best_model_path:
        raise RuntimeError(
            "ModelCheckpoint did not produce a best checkpoint. "
            "Cannot continue without explicit best-weight restoration."
        )
    shutil.copy2(best_model_path, model_path)
    tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    tft.to(torch_device)

    if cleanup_lightning_checkpoints:
        # Lightning writes extra checkpoint files to ./checkpoints by default.
        # Keep only the explicit model checkpoint saved in run_dir/models.
        ckpt_dir = Path("checkpoints")
        if ckpt_dir.exists():
            removed = 0
            for p in ckpt_dir.rglob("*.ckpt"):
                try:
                    p.unlink()
                    removed += 1
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning("Could not delete checkpoint %s: %s", p, exc)
            LOGGER.info("Removed %s Lightning checkpoint file(s) from %s.", removed, ckpt_dir)

    # Predict val/test in long format with model_name.

    def _predict(ds: TimeSeriesDataSet) -> pd.DataFrame:
        dl = ds.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers, pin_memory=False)
        pred_out = tft.predict(dl, mode="quantiles", return_x=True)
        # pytorch-forecasting returns a Prediction object in newer versions
        # (fields: output, x, index, decoder_lengths, y). Keep tuple fallback
        # for compatibility with older API shapes.
        if hasattr(pred_out, "output") and hasattr(pred_out, "x"):
            raw_pred = pred_out.output
            x = pred_out.x
        elif isinstance(pred_out, tuple) and len(pred_out) >= 2:
            raw_pred, x = pred_out[0], pred_out[1]
        else:
            raise TypeError(f"Unsupported TFT predict() return type: {type(pred_out)}")
        if isinstance(raw_pred, dict):
            pred_tensor = raw_pred.get("prediction")
            if pred_tensor is None:
                raise KeyError("TFT predict(mode='quantiles') returned dict without 'prediction'.")
        elif hasattr(raw_pred, "prediction"):
            pred_tensor = raw_pred.prediction
        else:
            pred_tensor = raw_pred
        decoder_idx = x["decoder_time_idx"]
        return _build_long_prediction_table(
            prediction=np.asarray(pred_tensor.detach().cpu()),
            decoder_time_idx=np.asarray(decoder_idx.detach().cpu()),
            idx_to_ts=idx_to_ts,
            model_name=model_name,
            quantiles=QUANTILES,
        )

    pred_val_start = time.perf_counter()
    pred_val_long = _predict(validation)
    pred_val_seconds = time.perf_counter() - pred_val_start
    pred_test_start = time.perf_counter()
    pred_test_long = _predict(testing)
    pred_test_seconds = time.perf_counter() - pred_test_start
    if tgt in {"target_afrr_activation_rate_pos", "target_afrr_activation_rate_neg"}:
        pred_val_long = _clip_activation_rate_predictions(pred_val_long)
        pred_test_long = _clip_activation_rate_predictions(pred_test_long)

    # Split-conformal: derive shifts on validation lead-1 quantiles, apply to test.
    q_cols = [_qcol(q) for q in QUANTILES]
    val_h1 = pred_val_long.loc[pred_val_long["lead_time_h"] == 1].copy()
    if not val_h1.empty:
        val_h1["target_time_utc"] = pd.to_datetime(val_h1["target_time_utc"], utc=True, errors="coerce")
        truth_val = (
            val_df.loc[:, ["timestamp_utc", tgt]]
            .rename(columns={"timestamp_utc": "target_time_utc", tgt: "y_true"})
            .copy()
        )
        truth_val["target_time_utc"] = pd.to_datetime(truth_val["target_time_utc"], utc=True, errors="coerce")
        calib_join = val_h1.merge(truth_val, on="target_time_utc", how="inner")
        if not calib_join.empty:
            q_calib = {c: pd.to_numeric(calib_join[c], errors="coerce").to_numpy(dtype=float) for c in q_cols if c in calib_join.columns}
            shifts = calculate_conformal_shifts(
                y_true_calib=pd.to_numeric(calib_join["y_true"], errors="coerce").to_numpy(dtype=float),
                q_preds_calib_dict=q_calib,
                alphas=list(QUANTILES),
            )
            q_test = {c: pd.to_numeric(pred_test_long[c], errors="coerce").to_numpy(dtype=float) for c in q_cols if c in pred_test_long.columns}
            q_test_cal = apply_conformal_shifts(q_test, shifts)
            for c in q_cols:
                if c in q_test_cal:
                    pred_test_long[c] = q_test_cal[c]
            if "p50" in pred_test_long.columns:
                pred_test_long["predicted_value"] = pd.to_numeric(pred_test_long["p50"], errors="coerce")

    export_start = time.perf_counter()
    interpretation_artifacts = _save_tft_interpretation_artifacts(
        tft_model=tft,
        dataset=validation,
        out_dir=report_dir,
        prefix=f"{bundle}_{tgt}",
        batch_size=batch_size,
        num_workers=num_workers,
        max_encoder_length=max_encoder_length,
    )
    # Training datasets/loaders are no longer needed after interpretation export.
    del training, validation, testing, train_loader, val_loader
    gc.collect()
    pred_col = _pred_column_names_for_target(tgt)[0]
    val_long_path = pred_dir / f"{bundle}_{tgt}_{pred_col}_long.parquet"
    test_long_path = pred_dir / f"{bundle}_{tgt}_{pred_col}_long_test.parquet"
    pred_val_long.to_parquet(val_long_path, index=False)
    pred_test_long.to_parquet(test_long_path, index=False)

    # Lead-1 wide export for existing runner/backtester path.
    def _to_wide_lead1(df_long: pd.DataFrame) -> pd.DataFrame:
        base_cols = ["target_time_utc"]
        if "p50" in df_long.columns:
            base_cols.append("p50")
        elif "predicted_value" in df_long.columns:
            base_cols.append("predicted_value")
        q_cols = [_qcol(q) for q in QUANTILES if _qcol(q) != "p50"]
        for q_col in q_cols:
            if q_col in df_long.columns:
                base_cols.append(q_col)

        d = df_long.loc[df_long["lead_time_h"] == 1, base_cols].copy()
        rename_map = {"target_time_utc": "timestamp_utc"}
        if "p50" in d.columns:
            rename_map["p50"] = pred_col
        elif "predicted_value" in d.columns:
            rename_map["predicted_value"] = pred_col
        for q_col in q_cols:
            if q_col in d.columns:
                rename_map[q_col] = f"{pred_col}_{q_col}"
        d = d.rename(columns=rename_map)
        d = d.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"])
        return d

    pred_val_wide = _to_wide_lead1(pred_val_long)
    pred_test_wide = _to_wide_lead1(pred_test_long)
    val_wide_path = pred_dir / f"{bundle}_{tgt}_val.parquet"
    test_wide_path = pred_dir / f"{bundle}_{tgt}_test.parquet"
    pred_val_wide.to_parquet(val_wide_path, index=False)
    pred_test_wide.to_parquet(test_wide_path, index=False)

    y_val_true = pd.to_numeric(val_df[tgt], errors="coerce").reset_index(drop=True)
    y_test_true = pd.to_numeric(test_df[tgt], errors="coerce").reset_index(drop=True)

    def _h1_metric_suite(pred_long: pd.DataFrame, y_true_h1: pd.Series) -> dict[str, object]:
        if pred_long.empty:
            return {}
        cols = ["lead_time_h"]
        q_cols = [_qcol(q) for q in QUANTILES]
        for c in ("predicted_value", *q_cols):
            if c in pred_long.columns:
                cols.append(c)
        h1 = pred_long.loc[pred_long["lead_time_h"] == 1, cols].copy()
        if h1.empty:
            return {}
        metric_df = pd.DataFrame(
            {
                "y_true": pd.to_numeric(y_true_h1, errors="coerce").iloc[: len(h1)].to_numpy(dtype=float),
                "y_pred": pd.to_numeric(
                    h1["p50"] if "p50" in h1.columns else h1["predicted_value"], errors="coerce"
                ).to_numpy(dtype=float),
            }
        )
        for q in q_cols:
            if q in h1.columns:
                metric_df[f"y_pred_{q}"] = pd.to_numeric(h1[q], errors="coerce").to_numpy(dtype=float)
        return compute_forecast_metrics(metric_df, y_true_col="y_true", y_pred_col="y_pred")

    val_metric_suite_h1 = _h1_metric_suite(pred_val_long, y_val_true)
    test_metric_suite_h1 = _h1_metric_suite(pred_test_long, y_test_true)
    gate_hour = gate_hour_for_target(tgt)
    val_gate_metrics: dict[str, object] = {}
    test_gate_metrics: dict[str, object] = {}
    if gate_hour is not None:
        val_gate_metrics = compute_gate_closure_metrics(
            pred_val_long,
            truth_df=val_df[["timestamp_utc", tgt]].copy(),
            y_true_col=tgt,
            y_pred_col="p50" if "p50" in pred_val_long.columns else "predicted_value",
            gate_hour_local=gate_hour,
            timezone="Europe/Berlin",
        )
        test_gate_metrics = compute_gate_closure_metrics(
            pred_test_long,
            truth_df=test_df[["timestamp_utc", tgt]].copy(),
            y_true_col=tgt,
            y_pred_col="p50" if "p50" in pred_test_long.columns else "predicted_value",
            gate_hour_local=gate_hour,
            timezone="Europe/Berlin",
        )

    decay_val = _leadtime_mae(pred_val_long, true_h1=y_val_true, horizon_hours=max_prediction_length)
    decay_test = _leadtime_mae(pred_test_long, true_h1=y_test_true, horizon_hours=max_prediction_length)
    decay_val_path = report_dir / f"{bundle}_{tgt}_val_forecast_decay.csv"
    decay_test_path = report_dir / f"{bundle}_{tgt}_test_forecast_decay.csv"
    decay_val.to_csv(decay_val_path, index=False)
    decay_test.to_csv(decay_test_path, index=False)
    export_seconds = time.perf_counter() - export_start

    total_seconds = time.perf_counter() - total_start

    def _mae_for_lead(decay_df: pd.DataFrame, lead: int) -> float:
        hit = decay_df.loc[decay_df["lead_time_h"] == int(lead), "mae"]
        if hit.empty:
            return float("nan")
        return float(hit.iloc[0])

    def _rmse_for_lead(decay_df: pd.DataFrame, lead: int) -> float:
        hit = decay_df.loc[decay_df["lead_time_h"] == int(lead), "rmse"]
        if hit.empty:
            return float("nan")
        return float(hit.iloc[0])

    h_last = int(max_prediction_length)
    mae_val_h1 = _mae_for_lead(decay_val, 1)
    mae_val_h_last = _mae_for_lead(decay_val, h_last)
    mae_test_h1 = _mae_for_lead(decay_test, 1)
    mae_test_h_last = _mae_for_lead(decay_test, h_last)
    rmse_val_h1 = _rmse_for_lead(decay_val, 1)
    rmse_val_h_last = _rmse_for_lead(decay_val, h_last)
    rmse_test_h1 = _rmse_for_lead(decay_test, 1)
    rmse_test_h_last = _rmse_for_lead(decay_test, h_last)
    mae_val_weighted = weighted_metric_from_decay(
        decay_val,
        value_col="mae",
        count_col="n",
        start_lead=int(lead_weight_start),
        end_lead=int(lead_weight_end),
        max_weight=float(lead_weight_max),
    )
    mae_test_weighted = weighted_metric_from_decay(
        decay_test,
        value_col="mae",
        count_col="n",
        start_lead=int(lead_weight_start),
        end_lead=int(lead_weight_end),
        max_weight=float(lead_weight_max),
    )
    rmse_val_weighted = weighted_metric_from_decay(
        decay_val,
        value_col="rmse",
        count_col="n",
        start_lead=int(lead_weight_start),
        end_lead=int(lead_weight_end),
        max_weight=float(lead_weight_max),
    )
    rmse_test_weighted = weighted_metric_from_decay(
        decay_test,
        value_col="rmse",
        count_col="n",
        start_lead=int(lead_weight_start),
        end_lead=int(lead_weight_end),
        max_weight=float(lead_weight_max),
    )
    lead_pinball_weighted: dict[str, float] = {}
    pinball_cols_val = [c for c in decay_val.columns if c.startswith("pinball_p")]
    pinball_cols_test = [c for c in decay_test.columns if c.startswith("pinball_p")]
    for dcol in pinball_cols_val:
        qcol = dcol.removeprefix("pinball_")
        if dcol in decay_val.columns:
            lead_pinball_weighted[f"leadtime_pinball_{qcol}_val_weighted"] = weighted_metric_from_decay(
                decay_val,
                value_col=dcol,
                count_col="n",
                start_lead=int(lead_weight_start),
                end_lead=int(lead_weight_end),
                max_weight=float(lead_weight_max),
            )
    for dcol in pinball_cols_test:
        qcol = dcol.removeprefix("pinball_")
        if dcol in decay_test.columns:
            lead_pinball_weighted[f"leadtime_pinball_{qcol}_test_weighted"] = weighted_metric_from_decay(
                decay_test,
                value_col=dcol,
                count_col="n",
                start_lead=int(lead_weight_start),
                end_lead=int(lead_weight_end),
                max_weight=float(lead_weight_max),
            )

    metrics = {
        "bundle": bundle,
        "target_col": tgt,
        "target_feature_variant": target_feature_variant,
        "target_feature_count": int(len(feature_columns)),
        "model_name": model_name,
        "resolved_device": torch_device,
        "accelerator": accelerator,
        "max_encoder_length": max_encoder_length,
        "max_prediction_length": max_prediction_length,
        "leadtime_last_h": h_last,
        "known_reals_count": len(known_reals),
        "unknown_reals_count": len(unknown_reals),
        "leadtime_mae_val_h1": mae_val_h1,
        "leadtime_mae_test_h1": mae_test_h1,
        "leadtime_mae_val_h_last": mae_val_h_last,
        "leadtime_mae_test_h_last": mae_test_h_last,
        "leadtime_rmse_val_h1": rmse_val_h1,
        "leadtime_rmse_test_h1": rmse_test_h1,
        "leadtime_rmse_val_h_last": rmse_val_h_last,
        "leadtime_rmse_test_h_last": rmse_test_h_last,
        "leadtime_mae_val_weighted": mae_val_weighted,
        "leadtime_mae_test_weighted": mae_test_weighted,
        "leadtime_rmse_val_weighted": rmse_val_weighted,
        "leadtime_rmse_test_weighted": rmse_test_weighted,
        "leadtime_weighting": {
            "start_lead_h": int(lead_weight_start),
            "end_lead_h": int(lead_weight_end),
            "max_weight": float(lead_weight_max),
        },
        f"leadtime_mae_val_h{h_last}": mae_val_h_last,
        f"leadtime_mae_test_h{h_last}": mae_test_h_last,
        f"leadtime_rmse_val_h{h_last}": rmse_val_h_last,
        f"leadtime_rmse_test_h{h_last}": rmse_test_h_last,
        # Backward-compatible keys used by older report scripts.
        "leadtime_mae_val_h48": mae_val_h_last if h_last == 48 else float("nan"),
        "leadtime_mae_test_h48": mae_test_h_last if h_last == 48 else float("nan"),
        "leadtime_rmse_val_h48": rmse_val_h_last if h_last == 48 else float("nan"),
        "leadtime_rmse_test_h48": rmse_test_h_last if h_last == 48 else float("nan"),
        "val_metric_suite_h1": val_metric_suite_h1,
        "test_metric_suite_h1": test_metric_suite_h1,
        "wmape_val_h1": val_metric_suite_h1.get("wmape"),
        "wmape_test_h1": test_metric_suite_h1.get("wmape"),
        "mape_val_h1": val_metric_suite_h1.get("mape"),
        "mape_test_h1": test_metric_suite_h1.get("mape"),
        "r2_val_h1": val_metric_suite_h1.get("r2"),
        "r2_test_h1": test_metric_suite_h1.get("r2"),
        "mbe_val_h1": val_metric_suite_h1.get("mbe"),
        "mbe_test_h1": test_metric_suite_h1.get("mbe"),
        "over_prediction_ratio_val_h1": val_metric_suite_h1.get("over_prediction_ratio"),
        "over_prediction_ratio_test_h1": test_metric_suite_h1.get("over_prediction_ratio"),
        "directional_accuracy_val_h1": val_metric_suite_h1.get("directional_accuracy"),
        "directional_accuracy_test_h1": test_metric_suite_h1.get("directional_accuracy"),
        "pinball_loss_p10_val_h1": val_metric_suite_h1.get("pinball_loss_p10"),
        "pinball_loss_p50_val_h1": val_metric_suite_h1.get("pinball_loss_p50"),
        "pinball_loss_p90_val_h1": val_metric_suite_h1.get("pinball_loss_p90"),
        "pinball_loss_p95_val_h1": val_metric_suite_h1.get("pinball_loss_p95"),
        "picp_80_val_h1": val_metric_suite_h1.get("picp_80"),
        "pinball_loss_p10_test_h1": test_metric_suite_h1.get("pinball_loss_p10"),
        "pinball_loss_p50_test_h1": test_metric_suite_h1.get("pinball_loss_p50"),
        "pinball_loss_p90_test_h1": test_metric_suite_h1.get("pinball_loss_p90"),
        "pinball_loss_p95_test_h1": test_metric_suite_h1.get("pinball_loss_p95"),
        "picp_80_test_h1": test_metric_suite_h1.get("picp_80"),
        "gate_closure_hour_local": gate_hour,
        "gate_closure_metrics_val": val_gate_metrics,
        "gate_closure_metrics_test": test_gate_metrics,
        "model_path": str(model_path.resolve()),
        "best_checkpoint_path": str(Path(best_model_path).resolve()) if best_model_path else str(model_path.resolve()),
        "stopped_epoch": int(trainer.current_epoch),
        "max_epochs": int(max_epochs),
        "early_stopping_patience": int(early_stopping_patience),
        "restore_best_weights_equivalent": True,
        "target_tft_params": {
            "dropout": effective_dropout,
            "max_epochs": int(max_epochs),
            "early_stopping_patience": int(early_stopping_patience),
            "is_afrr_target": bool(is_afrr_target),
        },
        "hidden_size": hidden_size,
        "attention_head_size": attention_head_size,
        "dropout": effective_dropout,
        "tensorboard_log_dir": str(Path(tb_logger.log_dir).resolve()),
        "learning_rate": float(learning_rate),
        "gradient_clip_val": float(gradient_clip_val),
        "precision_mode": resolved_precision,
        "attention_plot_path": interpretation_artifacts.get("attention_plot_path"),
        "attention_history_plot_path": interpretation_artifacts.get("attention_history_plot_path"),
        "feature_importance_plot_path": interpretation_artifacts.get("feature_importance_plot_path"),
        "scaler_path": str(scaler_path.resolve()),
        "pred_val_wide": str(val_wide_path.resolve()),
        "pred_test_wide": str(test_wide_path.resolve()),
        "pred_val_long": str(val_long_path.resolve()),
        "pred_test_long": str(test_long_path.resolve()),
        "decay_val_path": str(decay_val_path.resolve()),
        "decay_test_path": str(decay_test_path.resolve()),
        "timing_fit_seconds": float(fit_seconds),
        "timing_predict_val_seconds": float(pred_val_seconds),
        "timing_predict_test_seconds": float(pred_test_seconds),
        "timing_export_seconds": float(export_seconds),
        "timing_total_seconds": float(total_seconds),
    }
    # Export all probabilistic metrics present in H1 suites (all available quantiles + intervals).
    for k, v in val_metric_suite_h1.items():
        if (
            k.startswith("pinball_loss_p")
            or k.startswith("picp_")
            or k.startswith("winkler_score_")
            or k.startswith("pinaw_")
            or k.startswith("coverage_gap_")
            or k.startswith("tradeoff_score_")
            or k.startswith("crps_")
        ):
            metrics[f"{k}_val_h1"] = v
    for k, v in test_metric_suite_h1.items():
        if (
            k.startswith("pinball_loss_p")
            or k.startswith("picp_")
            or k.startswith("winkler_score_")
            or k.startswith("pinaw_")
            or k.startswith("coverage_gap_")
            or k.startswith("tradeoff_score_")
            or k.startswith("crps_")
        ):
            metrics[f"{k}_test_h1"] = v
    metrics.update(lead_pinball_weighted)
    val_pinball_keys = sorted([k for k in val_metric_suite_h1.keys() if k.startswith("pinball_loss_p")])
    test_pinball_keys = sorted([k for k in test_metric_suite_h1.keys() if k.startswith("pinball_loss_p")])
    val_pinball_vals = [float(val_metric_suite_h1[k]) for k in val_pinball_keys if np.isfinite(float(val_metric_suite_h1[k]))]
    test_pinball_vals = [float(test_metric_suite_h1[k]) for k in test_pinball_keys if np.isfinite(float(test_metric_suite_h1[k]))]
    metrics["pinball_mean_val_h1"] = float(np.mean(val_pinball_vals)) if val_pinball_vals else float("nan")
    metrics["pinball_mean_test_h1"] = float(np.mean(test_pinball_vals)) if test_pinball_vals else float("nan")
    metrics["pinball_mean_val"] = _mean_pinball_from_decay(decay_val)
    metrics["pinball_mean_test"] = _mean_pinball_from_decay(decay_test)
    return metrics


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Temporal Fusion Transformer on prepared bundles.")
    p.add_argument("--base-dir", default="data/model_input")
    p.add_argument("--bundle", choices=["da", "afrr"], default="da")
    p.add_argument("--target-col", default="")
    p.add_argument("--run-dir", default="", help="Run directory under artifacts/model_runs.")
    p.add_argument("--model-name", default="tft_v1")
    p.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-encoder-length", type=int, default=168)
    p.add_argument("--max-prediction-length", type=int, default=48)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--gradient-clip-val", type=float, default=0.05)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--attention-head-size", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument(
        "--precision",
        choices=["auto", "32-true", "16-mixed", "bf16-mixed"],
        default="auto",
        help="Training precision. 'auto' uses 16-mixed on CUDA and 32-true otherwise.",
    )
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--early-stopping-patience", type=int, default=None)
    p.add_argument("--lead-weight-start", type=int, default=16)
    p.add_argument("--lead-weight-end", type=int, default=48)
    p.add_argument("--lead-weight-max", type=float, default=2.0)
    p.add_argument(
        "--cleanup-lightning-checkpoints",
        action="store_true",
        help="Delete intermediate Lightning *.ckpt files under ./checkpoints after training.",
    )
    p.add_argument("--metrics-json-out", default="")
    p.add_argument("--manifest-fragment-out", default="")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_cli().parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else Path("artifacts/model_runs") / _run_id_now()
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = _train_tft(
        base_dir=Path(args.base_dir),
        bundle=args.bundle,
        target_col=(args.target_col.strip() or None),
        run_dir=run_dir,
        model_name=args.model_name,
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=args.max_prediction_length,
        seed=args.seed,
        requested_device=args.device,
        num_workers=args.num_workers,
        cleanup_lightning_checkpoints=args.cleanup_lightning_checkpoints,
        learning_rate=float(args.learning_rate),
        gradient_clip_val=float(args.gradient_clip_val),
        lead_weight_start=int(args.lead_weight_start),
        lead_weight_end=int(args.lead_weight_end),
        lead_weight_max=float(args.lead_weight_max),
        hidden_size_override=args.hidden_size,
        attention_head_size_override=args.attention_head_size,
        dropout_override=args.dropout,
        max_epochs_override=args.max_epochs,
        early_stopping_patience_override=args.early_stopping_patience,
        precision_mode=args.precision,
    )

    # Keep metrics path unique per bundle+target to avoid overwrite in target-wise
    # aFRR training loops (train_and_export_runs.py).
    default_metrics_path = run_dir / "metrics" / f"{args.bundle}_{metrics['target_col']}_tft_metrics.json"
    metrics_path = Path(args.metrics_json_out) if args.metrics_json_out else default_metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fragment = {
        "bundle": args.bundle,
        "model_path": metrics["model_path"],
        "metrics_path": str(metrics_path.resolve()),
        "predictions": {
            "val": metrics["pred_val_wide"],
            "test": metrics["pred_test_wide"],
        },
        "predictions_long": {
            "val": {_pred_column_names_for_target(metrics["target_col"])[0]: metrics["pred_val_long"]},
            "test": {_pred_column_names_for_target(metrics["target_col"])[0]: metrics["pred_test_long"]},
        },
        "prediction_columns": _pred_column_names_for_target(metrics["target_col"]),
        "target_columns": [metrics["target_col"]],
        "run_dir": str(run_dir.resolve()),
        "primary_target": metrics["target_col"],
    }
    fragment_path = (
        Path(args.manifest_fragment_out)
        if args.manifest_fragment_out
        else run_dir / f"{args.bundle}_{metrics['target_col']}_tft_manifest_fragment.json"
    )
    fragment_path.write_text(json.dumps(fragment, indent=2), encoding="utf-8")

    print("[OK] TFT training finished.")
    print(f"- Run dir: {run_dir}")
    print(f"- Device: {metrics['resolved_device']} ({metrics['accelerator']})")
    print(f"- Target: {metrics['target_col']}")
    print(f"- MAE val h1/h48: {metrics['leadtime_mae_val_h1']:.4f} / {metrics['leadtime_mae_val_h48']:.4f}")
    print(f"- MAE test h1/h48: {metrics['leadtime_mae_test_h1']:.4f} / {metrics['leadtime_mae_test_h48']:.4f}")
    print(f"- RMSE val h1/h48: {metrics['leadtime_rmse_val_h1']:.4f} / {metrics['leadtime_rmse_val_h48']:.4f}")
    print(f"- RMSE test h1/h48: {metrics['leadtime_rmse_test_h1']:.4f} / {metrics['leadtime_rmse_test_h48']:.4f}")
    print(f"- Metrics JSON: {metrics_path}")
    print(f"- Manifest fragment: {fragment_path}")


if __name__ == "__main__":
    main()
