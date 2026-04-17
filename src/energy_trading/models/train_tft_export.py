"""Train TFT model on prepared bundles and export backtest-compatible outputs."""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
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
CYCLICAL_DROP_FEATURES = {"weekday_sin", "weekday_cos"}
LAG_FEATURE_RE = re.compile(r".*_lag_\d+h$")
QUANTILES: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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
    - Drop `weekday_sin` / `weekday_cos`.
    - known categoricals: only `hour`, `weekday`, `month`.
    - known reals: remaining features containing `forecast` plus forecast-ramp proxies.
    - unknown reals: everything else remaining.
    """
    pruned: list[str] = []
    seen: set[str] = set()
    for c in feature_columns:
        if LAG_FEATURE_RE.match(c):
            continue
        if c in CYCLICAL_DROP_FEATURES:
            continue
        if c in seen:
            continue
        seen.add(c)
        pruned.append(c)

    known_categoricals = [c for c in pruned if c in ALLOWED_KNOWN_CATEGORICALS]

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
    """Fit robust scalers on train only and transform all splits."""
    y_scaler = RobustScaler()
    categorical_columns = list(categorical_columns or [])
    categorical_cols = [c for c in categorical_columns if c in feature_columns]
    real_cols = [c for c in feature_columns if c not in categorical_cols]

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

    y_tr = pd.to_numeric(train_df[target_col], errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
    y_va = pd.to_numeric(val_df[target_col], errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
    y_te = pd.to_numeric(test_df[target_col], errors="coerce").to_numpy(dtype=float).reshape(-1, 1)

    y_scaler.fit(y_tr)

    if real_cols:
        f_scaler = RobustScaler()
        f_scaler.fit(X_tr[real_cols].to_numpy(dtype=float))
    else:
        f_scaler = None

    def _scaled(df: pd.DataFrame, X_in: pd.DataFrame, y_arr: np.ndarray) -> pd.DataFrame:
        out = X_in.copy()
        if real_cols and f_scaler is not None:
            out.loc[:, real_cols] = f_scaler.transform(out[real_cols].to_numpy(dtype=float))
        out[target_col] = y_arr.reshape(-1)
        out["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        return out

    tr_sc = _scaled(train_df, X_tr, y_scaler.transform(y_tr))
    va_sc = _scaled(val_df, X_va, y_scaler.transform(y_va))
    te_sc = _scaled(test_df, X_te, y_scaler.transform(y_te))

    scaler_payload = {
        "feature_scaler": f_scaler,
        "target_scaler": y_scaler,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_cols,
        "real_feature_columns": real_cols,
        "target_col": target_col,
    }
    return tr_sc, va_sc, te_sc, scaler_payload


def _build_long_prediction_table(
    *,
    prediction: np.ndarray,
    decoder_time_idx: np.ndarray,
    idx_to_ts: dict[int, pd.Timestamp],
    target_scaler: RobustScaler,
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

    # Inverse-transform each quantile back to original unit.
    pred_inv = np.zeros_like(pred, dtype=float)
    for q_idx in range(pred.shape[2]):
        flat = pred[:, :, q_idx].reshape(-1, 1)
        inv = target_scaler.inverse_transform(flat).reshape(pred.shape[0], pred.shape[1])
        pred_inv[:, :, q_idx] = inv

    # Enforce monotonic quantile ordering to avoid quantile crossing.
    pred_inv = np.sort(pred_inv, axis=2)
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


def _leadtime_mae(
    pred_long: pd.DataFrame,
    *,
    true_h1: pd.Series,
    horizon_hours: int,
) -> pd.DataFrame:
    if pred_long.empty:
        return pd.DataFrame(columns=["lead_time_h", "n", "mae"])
    base = pd.to_numeric(true_h1, errors="coerce").reset_index(drop=True)
    truth_by_lead = {h: base.shift(-(h - 1)) for h in range(1, horizon_hours + 1)}
    pred_col = "p50" if "p50" in pred_long.columns else "predicted_value"

    rows: list[dict[str, float]] = []
    for lead in range(1, horizon_hours + 1):
        p = pd.to_numeric(pred_long.loc[pred_long["lead_time_h"] == lead, pred_col], errors="coerce").reset_index(drop=True)
        t = truth_by_lead[lead].iloc[: len(p)]
        m = p.notna() & t.notna()
        n = int(m.sum())
        mae = float(mean_absolute_error(t[m], p[m])) if n > 0 else np.nan
        rows.append({"lead_time_h": float(lead), "n": float(n), "mae": mae})
    return pd.DataFrame(rows)


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
) -> dict[str, object]:
    total_start = time.perf_counter()
    try:
        import lightning.pytorch as pl
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data.encoders import NaNLabelEncoder
        from pytorch_forecasting.metrics import QuantileLoss
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
    feature_columns = [
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
    feature_columns = [c for c in feature_columns if c in train_df.columns]
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
            full_sc[c] = full_sc[c].astype("Int64").astype(str)

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
        categorical_encoders={c: NaNLabelEncoder(add_nan=True) for c in known_categoricals},
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        full_sc.loc[full_sc["time_idx"] <= idx_val_end].copy(),
        min_prediction_idx=idx_train_end + 1,
        stop_randomization=True,
        predict=True,
    )
    testing = TimeSeriesDataSet.from_dataset(
        training,
        full_sc.copy(),
        min_prediction_idx=idx_val_end + 1,
        stop_randomization=True,
        predict=True,
    )

    torch_device, accelerator = _resolve_torch_device(requested_device)
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

    max_epochs = 20
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=0.1,
        enable_checkpointing=True,
        callbacks=[_EpochEtaCallback(max_epochs=max_epochs)],
        logger=False,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
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
    trainer.save_checkpoint(str(model_path))

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
    idx_to_ts = {
        int(i): pd.Timestamp(ts)
        for i, ts in zip(full_sc["time_idx"], pd.to_datetime(full_sc["timestamp_utc"], utc=True, errors="coerce"))
    }

    def _predict(ds: TimeSeriesDataSet) -> pd.DataFrame:
        dl = ds.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers, pin_memory=False)
        pred_out = tft.predict(dl, mode="raw", return_x=True)
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
        pred_tensor = raw_pred["prediction"] if isinstance(raw_pred, dict) else raw_pred.prediction
        decoder_idx = x["decoder_time_idx"]
        return _build_long_prediction_table(
            prediction=np.asarray(pred_tensor.detach().cpu()),
            decoder_time_idx=np.asarray(decoder_idx.detach().cpu()),
            idx_to_ts=idx_to_ts,
            target_scaler=scalers["target_scaler"],
            model_name=model_name,
            quantiles=QUANTILES,
        )

    pred_val_start = time.perf_counter()
    pred_val_long = _predict(validation)
    pred_val_seconds = time.perf_counter() - pred_val_start
    pred_test_start = time.perf_counter()
    pred_test_long = _predict(testing)
    pred_test_seconds = time.perf_counter() - pred_test_start

    export_start = time.perf_counter()
    pred_col = _pred_column_names_for_target(tgt)[0]
    val_long_path = pred_dir / f"{bundle}_{tgt}_{pred_col}_long.parquet"
    test_long_path = pred_dir / f"{bundle}_{tgt}_{pred_col}_long_test.parquet"
    pred_val_long.to_parquet(val_long_path, index=False)
    pred_test_long.to_parquet(test_long_path, index=False)

    # Lead-1 wide export for existing runner/backtester path.
    def _to_wide_lead1(df_long: pd.DataFrame) -> pd.DataFrame:
        d = df_long.loc[df_long["lead_time_h"] == 1, ["target_time_utc", "predicted_value"]].copy()
        d = d.rename(columns={"target_time_utc": "timestamp_utc", "predicted_value": pred_col})
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
    decay_val = _leadtime_mae(pred_val_long, true_h1=y_val_true, horizon_hours=max_prediction_length)
    decay_test = _leadtime_mae(pred_test_long, true_h1=y_test_true, horizon_hours=max_prediction_length)
    decay_val_path = report_dir / f"{bundle}_{tgt}_val_forecast_decay.csv"
    decay_test_path = report_dir / f"{bundle}_{tgt}_test_forecast_decay.csv"
    decay_val.to_csv(decay_val_path, index=False)
    decay_test.to_csv(decay_test_path, index=False)
    export_seconds = time.perf_counter() - export_start

    total_seconds = time.perf_counter() - total_start
    metrics = {
        "bundle": bundle,
        "target_col": tgt,
        "model_name": model_name,
        "resolved_device": torch_device,
        "accelerator": accelerator,
        "max_encoder_length": max_encoder_length,
        "max_prediction_length": max_prediction_length,
        "known_reals_count": len(known_reals),
        "unknown_reals_count": len(unknown_reals),
        "leadtime_mae_val_h1": float(decay_val.loc[decay_val["lead_time_h"] == 1, "mae"].iloc[0]),
        "leadtime_mae_val_h48": float(decay_val.loc[decay_val["lead_time_h"] == 48, "mae"].iloc[0]),
        "leadtime_mae_test_h1": float(decay_test.loc[decay_test["lead_time_h"] == 1, "mae"].iloc[0]),
        "leadtime_mae_test_h48": float(decay_test.loc[decay_test["lead_time_h"] == 48, "mae"].iloc[0]),
        "model_path": str(model_path.resolve()),
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
    print(f"- Metrics JSON: {metrics_path}")
    print(f"- Manifest fragment: {fragment_path}")


if __name__ == "__main__":
    main()
