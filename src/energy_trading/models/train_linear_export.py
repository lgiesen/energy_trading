"""Train linear quantile baseline on prepared DA/aFRR bundles and export artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_trading.evaluation.metrics import (  # noqa: E402
    compute_forecast_metrics,
    compute_gate_closure_metrics,
    gate_hour_for_target,
)
from energy_trading.evaluation.tensorboard_utils import (  # noqa: E402
    create_summary_writer,
    log_numeric_scalars,
    tensorboard_target_log_dir,
)
from energy_trading.evaluation.conformal_calibration import (  # noqa: E402
    apply_conformal_shifts,
    calculate_conformal_shifts,
)
from energy_trading.models.prepare_ml_bundles import load_processed_data  # noqa: E402

BundleName = Literal["da", "afrr"]
LOGGER = logging.getLogger(__name__)
QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99)


AFRR_TARGETS = [
    "target_afrr_activation_price_vwap_pos",
    "target_afrr_activation_price_vwap_neg",
    "target_afrr_activation_rate_pos",
    "target_afrr_activation_rate_neg",
    "target_afrr_capacity_price_pos",
    "target_afrr_capacity_price_neg",
]


def _run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _default_target_for_bundle(bundle: BundleName) -> str:
    if bundle == "da":
        return "target_da_price"
    if bundle == "afrr":
        return "target_afrr_activation_price_vwap_pos"
    raise KeyError(f"Unsupported bundle: {bundle}")


def _resolve_targets(bundle: BundleName, y_cols: list[str], target_col: str | None) -> list[str]:
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
        targets = [t for t in AFRR_TARGETS if t in y_cols]
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


def _write_bundle_manifest_fragment(
    *,
    bundle: BundleName,
    run_dir: Path,
    model_path: Path,
    metrics_path: Path,
    prediction_paths: dict[str, Path],
    prediction_long_paths: dict[str, dict[str, Path]],
    target_cols: list[str],
    metrics: dict[str, object],
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


def _qcol(q: float) -> str:
    return f"p{int(round(q * 100)):02d}"


def _build_linear_pipeline(alpha: float, quantile: float) -> Pipeline:
    # Fast linear quantile baseline using SGD quantile loss.
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(10.0, 90.0))),
            (
                "model",
                SGDRegressor(
                    loss="quantile",
                    quantile=float(quantile),
                    alpha=float(alpha),
                    penalty="elasticnet",
                    l1_ratio=0.15,
                    max_iter=3000,
                    tol=1e-4,
                    learning_rate="invscaling",
                    eta0=0.01,
                    power_t=0.25,
                    random_state=42,
                ),
            ),
        ]
    )


def _make_prediction_frame(bundle: BundleName, timestamp: pd.Series, pred_col: str, y_pred: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp_utc": pd.to_datetime(timestamp, utc=True, errors="coerce")})
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
    out[pred_col] = pd.to_numeric(pd.Series(y_pred, index=out.index), errors="coerce")
    return out


def _make_long_rows(
    *,
    h1_target_timestamp: pd.Series,
    lead_time_h: int,
    pred_by_q: dict[str, np.ndarray],
    model_name: str,
) -> pd.DataFrame:
    # Prepared bundles are aligned to h1 target timestamps.
    # For lead=h, snapshot is h hours before target, i.e. target-h.
    h1_ts = pd.to_datetime(h1_target_timestamp, utc=True, errors="coerce")
    target_ts = h1_ts + pd.to_timedelta(int(lead_time_h - 1), unit="h")
    snap_ts = target_ts - pd.to_timedelta(int(lead_time_h), unit="h")
    n = len(h1_ts)
    q_cols = [_qcol(q) for q in QUANTILES]
    q_stack = np.column_stack([pd.to_numeric(pd.Series(pred_by_q[c]), errors="coerce").to_numpy(dtype=float) for c in q_cols])
    q_stack = np.sort(q_stack, axis=1)  # crossing repair
    out = pd.DataFrame(
        {
            "snapshot_time_utc": snap_ts,
            "target_time_utc": target_ts,
            "lead_time_h": int(lead_time_h),
            "model_name": model_name,
            "predicted_value": q_stack[:, q_cols.index("p50")] if n > 0 else np.array([], dtype=float),
        }
    )
    for i, c in enumerate(q_cols):
        out[c] = q_stack[:, i]
    return out


def _train_target(
    *,
    base_dir: Path,
    bundle: BundleName,
    target_col: str,
    alpha: float,
    model_name: str,
    forecast_horizon_hours: int,
) -> tuple[dict[int, dict[str, Pipeline]], dict[str, object], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    cfg = json.loads((base_dir / "feature_config.json").read_text(encoding="utf-8"))
    bcfg = cfg["bundles"][bundle]
    train_ts = pd.to_datetime(
        pd.read_parquet(Path(bcfg["files"]["train"]), columns=["timestamp_utc"])["timestamp_utc"],
        utc=True,
        errors="coerce",
    )
    val_ts = pd.to_datetime(
        pd.read_parquet(Path(bcfg["files"]["val"]), columns=["timestamp_utc"])["timestamp_utc"],
        utc=True,
        errors="coerce",
    )
    test_ts = pd.to_datetime(
        pd.read_parquet(Path(bcfg["files"]["test"]), columns=["timestamp_utc"])["timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    # Target-specific routing is enforced via load_processed_data(...target_col_for_feature_routing=target_col).
    X_train, y_train_df = load_processed_data(
        bundle=bundle,
        split="train",
        base_dir=base_dir,
        target_col_for_feature_routing=target_col,
    )
    X_val, y_val_df = load_processed_data(
        bundle=bundle,
        split="val",
        base_dir=base_dir,
        target_col_for_feature_routing=target_col,
    )
    X_test, y_test_df = load_processed_data(
        bundle=bundle,
        split="test",
        base_dir=base_dir,
        target_col_for_feature_routing=target_col,
    )
    if target_col not in y_train_df.columns:
        raise KeyError(f"Target '{target_col}' not found in train labels.")

    y_train_all = pd.to_numeric(y_train_df[target_col], errors="coerce")
    y_val_all = pd.to_numeric(y_val_df[target_col], errors="coerce")
    y_test_all = pd.to_numeric(y_test_df[target_col], errors="coerce")

    pred_col = _pred_column_names_for_target(target_col)[0]
    lead_models: dict[int, dict[str, Pipeline]] = {}
    lead_rows_val: list[pd.DataFrame] = []
    lead_rows_test: list[pd.DataFrame] = []
    lead_metric_rows: list[dict[str, float]] = []
    fit_seconds_total = 0.0

    # Keep canonical wide-format prediction files as h1 for compatibility.
    pred_frames: dict[str, pd.DataFrame] | None = None
    y_va_h1: pd.Series | None = None
    y_te_h1: pd.Series | None = None
    pred_va_h1: np.ndarray | None = None
    pred_te_h1: np.ndarray | None = None

    for lead_h in range(1, int(forecast_horizon_hours) + 1):
        shift_n = int(lead_h - 1)
        y_tr_shift = y_train_all.shift(-shift_n)
        y_va_shift = y_val_all.shift(-shift_n)
        y_te_shift = y_test_all.shift(-shift_n)

        tr_mask = y_tr_shift.notna()
        va_mask = y_va_shift.notna()
        te_mask = y_te_shift.notna()

        X_tr = X_train.loc[tr_mask].copy()
        y_tr = pd.to_numeric(y_tr_shift.loc[tr_mask], errors="coerce")
        X_va = X_val.loc[va_mask].copy()
        y_va = pd.to_numeric(y_va_shift.loc[va_mask], errors="coerce")
        X_te = X_test.loc[te_mask].copy()
        y_te = pd.to_numeric(y_te_shift.loc[te_mask], errors="coerce")
        ts_va = val_ts.loc[va_mask].copy()
        ts_te = test_ts.loc[te_mask].copy()

        if X_tr.empty or X_va.empty:
            LOGGER.warning(
                "Skipping lead h=%s for target '%s' due to insufficient rows (train=%s, val=%s).",
                lead_h,
                target_col,
                len(X_tr),
                len(X_va),
            )
            continue

        q_models: dict[str, Pipeline] = {}
        preds_va: dict[str, np.ndarray] = {}
        preds_te: dict[str, np.ndarray] = {}
        fit_t0 = time.perf_counter()
        for q in QUANTILES:
            qcol = _qcol(q)
            pipe = _build_linear_pipeline(alpha=alpha, quantile=q)
            pipe.fit(X_tr, y_tr)
            q_models[qcol] = pipe
            preds_va[qcol] = pipe.predict(X_va)
            preds_te[qcol] = pipe.predict(X_te)
        fit_seconds_total += float(time.perf_counter() - fit_t0)

        # Post-process to enforce monotone quantiles row-wise.
        q_cols = [_qcol(q) for q in QUANTILES]
        va_stack = np.column_stack([pd.to_numeric(pd.Series(preds_va[c]), errors="coerce").to_numpy(dtype=float) for c in q_cols])
        te_stack = np.column_stack([pd.to_numeric(pd.Series(preds_te[c]), errors="coerce").to_numpy(dtype=float) for c in q_cols])
        va_stack = np.sort(va_stack, axis=1)
        te_stack = np.sort(te_stack, axis=1)
        for i, c in enumerate(q_cols):
            preds_va[c] = va_stack[:, i]
            preds_te[c] = te_stack[:, i]

        pred_va = preds_va["p50"]
        pred_te = preds_te["p50"]

        lead_models[int(lead_h)] = q_models
        lead_rows_val.append(
            _make_long_rows(
                h1_target_timestamp=ts_va,
                lead_time_h=int(lead_h),
                pred_by_q=preds_va,
                model_name=model_name,
            )
        )
        lead_rows_test.append(
            _make_long_rows(
                h1_target_timestamp=ts_te,
                lead_time_h=int(lead_h),
                pred_by_q=preds_te,
                model_name=model_name,
            )
        )

        val_metric_suite_h = compute_forecast_metrics(
            pd.DataFrame({"y_true": y_va.to_numpy(dtype=float), "y_pred": pred_va}),
            y_true_col="y_true",
            y_pred_col="y_pred",
        )
        test_metric_suite_h = compute_forecast_metrics(
            pd.DataFrame({"y_true": y_te.to_numpy(dtype=float), "y_pred": pred_te}),
            y_true_col="y_true",
            y_pred_col="y_pred",
        )
        lead_metric_rows.append(
            {
                "lead_time_h": float(lead_h),
                "n_val": float(len(y_va)),
                "n_test": float(len(y_te)),
                "mae_val": float(val_metric_suite_h.get("mae", np.nan)),
                "mae_test": float(test_metric_suite_h.get("mae", np.nan)),
                "rmse_val": float(val_metric_suite_h.get("rmse", np.nan)),
                "rmse_test": float(test_metric_suite_h.get("rmse", np.nan)),
            }
        )

        if int(lead_h) == 1:
            y_va_h1 = y_va
            y_te_h1 = y_te
            pred_va_h1 = pred_va
            pred_te_h1 = pred_te
            pred_frames = {
                "val": _make_prediction_frame(bundle=bundle, timestamp=ts_va, pred_col=pred_col, y_pred=pred_va),
                "test": _make_prediction_frame(bundle=bundle, timestamp=ts_te, pred_col=pred_col, y_pred=pred_te),
            }

    if not lead_models:
        raise ValueError(f"No linear lead models trained for target '{target_col}'.")
    if pred_frames is None or y_va_h1 is None or y_te_h1 is None or pred_va_h1 is None or pred_te_h1 is None:
        raise ValueError(f"Lead h1 training failed for target '{target_col}', cannot build canonical exports.")

    long_frames = {
        "val": pd.concat(lead_rows_val, axis=0, ignore_index=True).sort_values(
            ["snapshot_time_utc", "lead_time_h", "target_time_utc"]
        ),
        "test": pd.concat(lead_rows_test, axis=0, ignore_index=True).sort_values(
            ["snapshot_time_utc", "lead_time_h", "target_time_utc"]
        ),
    }

    # Conformal calibration (split-conformal): fit shifts on validation long set,
    # then apply to test long set before export.
    q_cols = [_qcol(q) for q in QUANTILES]
    truth_calib = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(pred_frames["val"]["timestamp_utc"], utc=True, errors="coerce"),
            "y_true": pd.to_numeric(y_va_h1, errors="coerce").to_numpy(dtype=float),
        }
    )
    # Prefer lead-1 calibration to preserve horizon semantics.
    calib_base = long_frames["val"].loc[long_frames["val"]["lead_time_h"] == 1].copy()
    if not calib_base.empty:
        calib_base["target_time_utc"] = pd.to_datetime(calib_base["target_time_utc"], utc=True, errors="coerce")
        calib_join = calib_base.merge(
            truth_calib,
            left_on="target_time_utc",
            right_on="timestamp_utc",
            how="inner",
        )
        if not calib_join.empty:
            q_dict = {c: pd.to_numeric(calib_join[c], errors="coerce").to_numpy(dtype=float) for c in q_cols if c in calib_join.columns}
            shifts = calculate_conformal_shifts(
                y_true_calib=pd.to_numeric(calib_join["y_true"], errors="coerce").to_numpy(dtype=float),
                q_preds_calib_dict=q_dict,
                alphas=list(QUANTILES),
            )
            test_df = long_frames["test"].copy()
            q_test = {c: pd.to_numeric(test_df[c], errors="coerce").to_numpy(dtype=float) for c in q_cols if c in test_df.columns}
            q_cal = apply_conformal_shifts(q_test, shifts)
            for c in q_cols:
                if c in q_cal:
                    test_df[c] = q_cal[c]
            if "p50" in test_df.columns:
                test_df["predicted_value"] = pd.to_numeric(test_df["p50"], errors="coerce")
            long_frames["test"] = test_df

    # Maintain top-level metrics on h1 for continuity + add lead-wise summary.
    val_metric_suite = compute_forecast_metrics(
        pd.DataFrame({"y_true": y_va_h1.to_numpy(dtype=float), "y_pred": pred_va_h1}),
        y_true_col="y_true",
        y_pred_col="y_pred",
    )
    test_metric_suite = compute_forecast_metrics(
        pd.DataFrame({"y_true": y_te_h1.to_numpy(dtype=float), "y_pred": pred_te_h1}),
        y_true_col="y_true",
        y_pred_col="y_pred",
    )

    gate_hour = gate_hour_for_target(target_col)
    val_gate: dict[str, object] = {}
    test_gate: dict[str, object] = {}
    val_h1_long = long_frames["val"].loc[long_frames["val"]["lead_time_h"] == 1].copy()
    test_h1_long = long_frames["test"].loc[long_frames["test"]["lead_time_h"] == 1].copy()
    if gate_hour is not None:
        val_gate = compute_gate_closure_metrics(
            val_h1_long,
            truth_df=pd.DataFrame(
                {
                    "timestamp_utc": pd.to_datetime(val_h1_long["target_time_utc"], utc=True, errors="coerce"),
                    target_col: y_va_h1.to_numpy(dtype=float),
                }
            ),
            y_true_col=target_col,
            y_pred_col="p50",
            gate_hour_local=gate_hour,
            timezone="Europe/Berlin",
        )
        test_gate = compute_gate_closure_metrics(
            test_h1_long,
            truth_df=pd.DataFrame(
                {
                    "timestamp_utc": pd.to_datetime(test_h1_long["target_time_utc"], utc=True, errors="coerce"),
                    target_col: y_te_h1.to_numpy(dtype=float),
                }
            ),
            y_true_col=target_col,
            y_pred_col="p50",
            gate_hour_local=gate_hour,
            timezone="Europe/Berlin",
        )

    lead_df = pd.DataFrame(lead_metric_rows).sort_values("lead_time_h").reset_index(drop=True)
    last_h = int(lead_df["lead_time_h"].max()) if not lead_df.empty else 1
    lead_mae_val_h1 = float(lead_df.loc[lead_df["lead_time_h"] == 1.0, "mae_val"].iloc[0]) if (lead_df["lead_time_h"] == 1.0).any() else np.nan
    lead_mae_test_h1 = float(lead_df.loc[lead_df["lead_time_h"] == 1.0, "mae_test"].iloc[0]) if (lead_df["lead_time_h"] == 1.0).any() else np.nan
    lead_mae_val_last = float(lead_df.loc[lead_df["lead_time_h"] == float(last_h), "mae_val"].iloc[0]) if not lead_df.empty else np.nan
    lead_mae_test_last = float(lead_df.loc[lead_df["lead_time_h"] == float(last_h), "mae_test"].iloc[0]) if not lead_df.empty else np.nan

    metrics: dict[str, object] = {
        "target_col": target_col,
        "model_name": model_name,
        "model_family": "linear_ridge",
        "alpha": float(alpha),
        "n_features": int(X_train.shape[1]),
        "rows_train_h1": int(y_train_all.notna().sum()),
        "rows_val_h1": int(len(y_va_h1)),
        "rows_test_h1": int(len(y_te_h1)),
        "forecast_horizon_hours": int(forecast_horizon_hours),
        "leadtime_last_h": int(last_h),
        "leadtime_mae_val_h1": lead_mae_val_h1,
        "leadtime_mae_test_h1": lead_mae_test_h1,
        f"leadtime_mae_val_h{last_h}": lead_mae_val_last,
        f"leadtime_mae_test_h{last_h}": lead_mae_test_last,
        "leadtime_mae_val_h_last": lead_mae_val_last,
        "leadtime_mae_test_h_last": lead_mae_test_last,
        "leadtime_metrics": lead_df.to_dict(orient="records"),
        "metric_suite_val": val_metric_suite,
        "metric_suite_test": test_metric_suite,
        "mae_val": val_metric_suite.get("mae"),
        "rmse_val": val_metric_suite.get("rmse"),
        "mape_val": val_metric_suite.get("mape"),
        "wmape_val": val_metric_suite.get("wmape"),
        "r2_val": val_metric_suite.get("r2"),
        "mbe_val": val_metric_suite.get("mbe"),
        "over_prediction_ratio_val": val_metric_suite.get("over_prediction_ratio"),
        "mae_test": test_metric_suite.get("mae"),
        "rmse_test": test_metric_suite.get("rmse"),
        "mape_test": test_metric_suite.get("mape"),
        "wmape_test": test_metric_suite.get("wmape"),
        "r2_test": test_metric_suite.get("r2"),
        "mbe_test": test_metric_suite.get("mbe"),
        "over_prediction_ratio_test": test_metric_suite.get("over_prediction_ratio"),
        "directional_accuracy_test": test_metric_suite.get("directional_accuracy"),
        "gate_closure_hour_local": gate_hour,
        "gate_closure_metrics_val": val_gate,
        "gate_closure_metrics_test": test_gate,
        "timing_fit_seconds_total": fit_seconds_total,
        "trained_leads": sorted(int(k) for k in lead_models.keys()),
    }
    return lead_models, metrics, pred_frames, long_frames


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train linear quantile baseline on prepared DA/aFRR bundles.")
    p.add_argument("--base-dir", default="data/model_input")
    p.add_argument("--bundle", choices=["da", "afrr"], default="da")
    p.add_argument("--target-col", default="")
    p.add_argument("--run-dir", default="", help="Run directory under artifacts/model_runs.")
    p.add_argument("--model-name", default="linear_ridge_v1")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--forecast-horizon-hours", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metrics-json-out", default="")
    p.add_argument("--metrics-csv-out", default="")
    p.add_argument("--manifest-fragment-out", default="")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_cli().parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_dir = Path(args.run_dir) if args.run_dir else Path("artifacts/model_runs") / _run_id_now()
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = run_dir / "models"
    pred_dir = run_dir / "predictions"
    metrics_dir = run_dir / "metrics"
    report_dir = run_dir / "reports"
    for d in (model_dir, pred_dir, metrics_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Resolve targets from train labels.
    _, y_train_df = load_processed_data(bundle=args.bundle, split="train", base_dir=args.base_dir)
    target_cols = _resolve_targets(args.bundle, list(y_train_df.columns), args.target_col.strip() or None)
    if len(target_cols) != 1:
        raise ValueError(
            "train_linear_export expects one target per invocation. "
            "Use --target-col for target-wise runs (as done by train_and_export_runs.py)."
        )
    tgt = target_cols[0]

    lead_models, metrics, pred_frames, long_frames = _train_target(
        base_dir=Path(args.base_dir),
        bundle=args.bundle,
        target_col=tgt,
        alpha=float(args.alpha),
        model_name=args.model_name,
        forecast_horizon_hours=max(1, int(args.forecast_horizon_hours)),
    )

    file_tag = f"linear_{args.bundle}_{tgt.replace('target_', '')}"
    model_path = model_dir / f"{file_tag}_model.joblib"
    model_payload: dict[str, Any] = {
        "model_family": "linear_ridge",
        "target_col": tgt,
        "forecast_horizon_hours": int(max(1, int(args.forecast_horizon_hours))),
        "lead_models": lead_models,
    }
    joblib.dump(model_payload, model_path)

    pred_col = _pred_column_names_for_target(tgt)[0]
    pred_val_path = pred_dir / f"{file_tag}_val.parquet"
    pred_test_path = pred_dir / f"{file_tag}_test.parquet"
    pred_val_long_path = pred_dir / f"{file_tag}_val_{pred_col}_long.parquet"
    pred_test_long_path = pred_dir / f"{file_tag}_test_{pred_col}_long.parquet"
    pred_frames["val"].to_parquet(pred_val_path, index=False)
    pred_frames["test"].to_parquet(pred_test_path, index=False)
    long_frames["val"].to_parquet(pred_val_long_path, index=False)
    long_frames["test"].to_parquet(pred_test_long_path, index=False)

    metrics["model_path"] = str(model_path.resolve())
    metrics["pred_val"] = str(pred_val_path.resolve())
    metrics["pred_test"] = str(pred_test_path.resolve())
    metrics["pred_val_long"] = str(pred_val_long_path.resolve())
    metrics["pred_test_long"] = str(pred_test_long_path.resolve())
    metrics["tensorboard_log_dir"] = None

    # Unified TensorBoard path policy across model families.
    tb_log_dir = tensorboard_target_log_dir(
        run_dir=run_dir,
        model_family="linear",
        bundle=args.bundle,
        target_col=tgt,
    )
    tb_writer = create_summary_writer(tb_log_dir)
    if tb_writer is not None:
        # Log nested numeric metrics at step=0 for one-shot linear fit.
        n_logged = log_numeric_scalars(tb_writer, metrics, prefix="metrics", step=0)
        tb_writer.add_scalar("meta/numeric_scalars_logged", float(n_logged), 0)
        tb_writer.flush()
        tb_writer.close()
        metrics["tensorboard_log_dir"] = str(tb_log_dir.resolve())

    metrics_json_path = (
        Path(args.metrics_json_out)
        if args.metrics_json_out
        else metrics_dir / f"{file_tag}_metrics.json"
    )
    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    metrics_csv_path = (
        Path(args.metrics_csv_out)
        if args.metrics_csv_out
        else report_dir / f"{file_tag}_metrics_row.csv"
    )
    flat_row = {
        "target_col": tgt,
        "model_family": "linear_ridge",
        "alpha": float(args.alpha),
        "n_features": int(metrics["n_features"]),
        "mae_val": metrics.get("mae_val"),
        "rmse_val": metrics.get("rmse_val"),
        "mape_val": metrics.get("mape_val"),
        "wmape_val": metrics.get("wmape_val"),
        "r2_val": metrics.get("r2_val"),
        "mbe_val": metrics.get("mbe_val"),
        "over_prediction_ratio_val": metrics.get("over_prediction_ratio_val"),
        "mae_test": metrics.get("mae_test"),
        "rmse_test": metrics.get("rmse_test"),
        "mape_test": metrics.get("mape_test"),
        "wmape_test": metrics.get("wmape_test"),
        "r2_test": metrics.get("r2_test"),
        "mbe_test": metrics.get("mbe_test"),
        "over_prediction_ratio_test": metrics.get("over_prediction_ratio_test"),
        "directional_accuracy_test": metrics.get("directional_accuracy_test"),
        "gate_mae_val": (metrics.get("gate_closure_metrics_val") or {}).get("mae_gate"),
        "gate_rmse_val": (metrics.get("gate_closure_metrics_val") or {}).get("rmse_gate"),
        "gate_mae_test": (metrics.get("gate_closure_metrics_test") or {}).get("mae_gate"),
        "gate_rmse_test": (metrics.get("gate_closure_metrics_test") or {}).get("rmse_gate"),
        "gate_acceptance_rate_test": (metrics.get("gate_closure_metrics_test") or {}).get("acceptance_rate_gate"),
    }
    pd.DataFrame([flat_row]).to_csv(metrics_csv_path, index=False)

    fragment = _write_bundle_manifest_fragment(
        bundle=args.bundle,
        run_dir=run_dir,
        model_path=model_path,
        metrics_path=metrics_json_path,
        prediction_paths={"val": pred_val_path, "test": pred_test_path},
        prediction_long_paths={
            "val": {pred_col: pred_val_long_path},
            "test": {pred_col: pred_test_long_path},
        },
        target_cols=[tgt],
        metrics=metrics,
    )
    fragment_path = (
        Path(args.manifest_fragment_out)
        if args.manifest_fragment_out
        else run_dir / f"{file_tag}_manifest_fragment.json"
    )
    fragment_path.parent.mkdir(parents=True, exist_ok=True)
    fragment_path.write_text(json.dumps(fragment, indent=2), encoding="utf-8")

    print("[OK] Linear Quantile training finished.")
    print(f"- Bundle: {args.bundle}")
    print(f"- Target: {tgt}")
    print(f"- MAE val/test: {metrics.get('mae_val')} / {metrics.get('mae_test')}")
    print(f"- RMSE val/test: {metrics.get('rmse_val')} / {metrics.get('rmse_test')}")
    print(f"- Metrics JSON: {metrics_json_path}")
    print(f"- Metrics CSV: {metrics_csv_path}")
    print(f"- Manifest fragment: {fragment_path}")


if __name__ == "__main__":
    main()
