"""Train Ridge linear baseline on prepared DA/aFRR bundles and export artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
from energy_trading.models.prepare_ml_bundles import load_processed_data  # noqa: E402

BundleName = Literal["da", "afrr"]
LOGGER = logging.getLogger(__name__)


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


def _build_linear_pipeline(alpha: float) -> Pipeline:
    # Linear baseline with robust preprocessing for missing values + scaling.
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=float(alpha))),
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


def _make_long_h1(timestamp: pd.Series, y_pred: np.ndarray, model_name: str) -> pd.DataFrame:
    target_ts = pd.to_datetime(timestamp, utc=True, errors="coerce")
    snap_ts = target_ts - pd.to_timedelta(1, unit="h")
    return pd.DataFrame(
        {
            "snapshot_time_utc": snap_ts,
            "target_time_utc": target_ts,
            "lead_time_h": 1,
            "model_name": model_name,
            "predicted_value": pd.to_numeric(pd.Series(y_pred), errors="coerce"),
            "p50": pd.to_numeric(pd.Series(y_pred), errors="coerce"),
        }
    )


def _train_target(
    *,
    base_dir: Path,
    bundle: BundleName,
    target_col: str,
    alpha: float,
    model_name: str,
) -> tuple[Pipeline, dict[str, object], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
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

    tr_mask = pd.to_numeric(y_train_df[target_col], errors="coerce").notna()
    va_mask = pd.to_numeric(y_val_df[target_col], errors="coerce").notna()
    te_mask = pd.to_numeric(y_test_df[target_col], errors="coerce").notna()

    X_tr = X_train.loc[tr_mask].copy()
    y_tr = pd.to_numeric(y_train_df.loc[tr_mask, target_col], errors="coerce")
    X_va = X_val.loc[va_mask].copy()
    y_va = pd.to_numeric(y_val_df.loc[va_mask, target_col], errors="coerce")
    X_te = X_test.loc[te_mask].copy()
    y_te = pd.to_numeric(y_test_df.loc[te_mask, target_col], errors="coerce")
    ts_va = val_ts.loc[va_mask].copy()
    ts_te = test_ts.loc[te_mask].copy()

    if X_tr.empty or X_va.empty:
        raise ValueError(f"Not enough rows for target '{target_col}' (train={len(X_tr)}, val={len(X_va)}).")

    pipe = _build_linear_pipeline(alpha=alpha)
    fit_t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    fit_seconds = float(time.perf_counter() - fit_t0)

    pred_va = pipe.predict(X_va)
    pred_te = pipe.predict(X_te)

    pred_col = _pred_column_names_for_target(target_col)[0]
    pred_frames = {
        "val": _make_prediction_frame(bundle=bundle, timestamp=ts_va, pred_col=pred_col, y_pred=pred_va),
        "test": _make_prediction_frame(bundle=bundle, timestamp=ts_te, pred_col=pred_col, y_pred=pred_te),
    }
    long_frames = {
        "val": _make_long_h1(timestamp=ts_va, y_pred=pred_va, model_name=model_name),
        "test": _make_long_h1(timestamp=ts_te, y_pred=pred_te, model_name=model_name),
    }

    val_metric_suite = compute_forecast_metrics(
        pd.DataFrame({"y_true": y_va.to_numpy(dtype=float), "y_pred": pred_va}),
        y_true_col="y_true",
        y_pred_col="y_pred",
    )
    test_metric_suite = compute_forecast_metrics(
        pd.DataFrame({"y_true": y_te.to_numpy(dtype=float), "y_pred": pred_te}),
        y_true_col="y_true",
        y_pred_col="y_pred",
    )

    gate_hour = gate_hour_for_target(target_col)
    val_gate: dict[str, object] = {}
    test_gate: dict[str, object] = {}
    if gate_hour is not None:
        val_gate = compute_gate_closure_metrics(
            long_frames["val"],
            truth_df=pd.DataFrame(
                {
                    "timestamp_utc": pd.to_datetime(ts_va, utc=True, errors="coerce"),
                    target_col: y_va.to_numpy(dtype=float),
                }
            ),
            y_true_col=target_col,
            y_pred_col="p50",
            gate_hour_local=gate_hour,
            timezone="Europe/Berlin",
        )
        test_gate = compute_gate_closure_metrics(
            long_frames["test"],
            truth_df=pd.DataFrame(
                {
                    "timestamp_utc": pd.to_datetime(ts_te, utc=True, errors="coerce"),
                    target_col: y_te.to_numpy(dtype=float),
                }
            ),
            y_true_col=target_col,
            y_pred_col="p50",
            gate_hour_local=gate_hour,
            timezone="Europe/Berlin",
        )

    metrics: dict[str, object] = {
        "target_col": target_col,
        "model_name": model_name,
        "model_family": "linear_ridge",
        "alpha": float(alpha),
        "n_features": int(X_tr.shape[1]),
        "rows_train": int(len(X_tr)),
        "rows_val": int(len(X_va)),
        "rows_test": int(len(X_te)),
        "metric_suite_val": val_metric_suite,
        "metric_suite_test": test_metric_suite,
        "mae_val": val_metric_suite.get("mae"),
        "rmse_val": val_metric_suite.get("rmse"),
        "wmape_val": val_metric_suite.get("wmape"),
        "mbe_val": val_metric_suite.get("mbe"),
        "over_prediction_ratio_val": val_metric_suite.get("over_prediction_ratio"),
        "mae_test": test_metric_suite.get("mae"),
        "rmse_test": test_metric_suite.get("rmse"),
        "wmape_test": test_metric_suite.get("wmape"),
        "mbe_test": test_metric_suite.get("mbe"),
        "over_prediction_ratio_test": test_metric_suite.get("over_prediction_ratio"),
        "directional_accuracy_test": test_metric_suite.get("directional_accuracy"),
        "gate_closure_hour_local": gate_hour,
        "gate_closure_metrics_val": val_gate,
        "gate_closure_metrics_test": test_gate,
        "timing_fit_seconds": fit_seconds,
    }
    return pipe, metrics, pred_frames, long_frames


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Ridge linear baseline on prepared DA/aFRR bundles.")
    p.add_argument("--base-dir", default="data/model_input")
    p.add_argument("--bundle", choices=["da", "afrr"], default="da")
    p.add_argument("--target-col", default="")
    p.add_argument("--run-dir", default="", help="Run directory under artifacts/model_runs.")
    p.add_argument("--model-name", default="linear_ridge_v1")
    p.add_argument("--alpha", type=float, default=1.0)
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

    pipe, metrics, pred_frames, long_frames = _train_target(
        base_dir=Path(args.base_dir),
        bundle=args.bundle,
        target_col=tgt,
        alpha=float(args.alpha),
        model_name=args.model_name,
    )

    file_tag = f"linear_{args.bundle}_{tgt.replace('target_', '')}"
    model_path = model_dir / f"{file_tag}_model.joblib"
    joblib.dump(pipe, model_path)

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
        "wmape_val": metrics.get("wmape_val"),
        "mbe_val": metrics.get("mbe_val"),
        "over_prediction_ratio_val": metrics.get("over_prediction_ratio_val"),
        "mae_test": metrics.get("mae_test"),
        "rmse_test": metrics.get("rmse_test"),
        "wmape_test": metrics.get("wmape_test"),
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

    print("[OK] Linear Ridge training finished.")
    print(f"- Bundle: {args.bundle}")
    print(f"- Target: {tgt}")
    print(f"- MAE val/test: {metrics.get('mae_val')} / {metrics.get('mae_test')}")
    print(f"- RMSE val/test: {metrics.get('rmse_val')} / {metrics.get('rmse_test')}")
    print(f"- Metrics JSON: {metrics_json_path}")
    print(f"- Metrics CSV: {metrics_csv_path}")
    print(f"- Manifest fragment: {fragment_path}")


if __name__ == "__main__":
    main()
