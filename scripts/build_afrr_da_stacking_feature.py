"""Create leakage-safe DA stacking feature for aFRR training bundles.

This script adds a new feature column `da_price_predicted_h1` to the aFRR
bundle splits by:
1) training DA models on train folds only (OOF prediction for DA-train),
2) fitting one final DA model on full DA-train for DA-val/test prediction,
3) merging predictions into aFRR train/val/test by timestamp.

Result: a new bundle directory that can be used to train aFRR with stacked DA
signal without leaking future information.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow direct script execution from repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_trading.models.cv import PurgedTimeSeriesSplit
from energy_trading.models.prepare_ml_bundles import load_processed_data


STACK_COL = "da_price_predicted_h1"
DA_TARGET = "target_da_price_h1"


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build DA->aFRR stacking feature (leakage-safe).")
    p.add_argument("--base-dir", default="data/model_input", help="Existing ML bundle directory.")
    p.add_argument("--out-dir", default="data/model_input_stacked", help="Output bundle directory with stacked aFRR.")
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--cv-n-splits", type=int, default=8)
    p.add_argument("--cv-test-size", type=int, default=24 * 7)
    p.add_argument("--cv-gap-hours", type=int, default=72)
    return p


def _read_cfg(base_dir: Path) -> dict:
    cfg_path = base_dir / "feature_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing feature_config.json: {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _load_da_split_df(base_dir: Path, split: str) -> pd.DataFrame:
    cfg = _read_cfg(base_dir)
    p = Path(cfg["bundles"]["da"]["files"][split])
    if not p.exists():
        raise FileNotFoundError(f"Missing DA split file: {p}")
    df = pd.read_parquet(p)
    if "timestamp_utc" not in df.columns:
        raise KeyError(f"`timestamp_utc` missing in DA split: {p}")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    return df


def _fit_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    device: str,
) -> object:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("xgboost is required for DA stacking.") from exc

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
    model.fit(X, y, verbose=False)
    return model


def _build_da_predictions(
    *,
    base_dir: Path,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    device: str,
    cv_n_splits: int,
    cv_test_size: int,
    cv_gap_hours: int,
) -> dict[str, pd.DataFrame]:
    X_train, y_train = load_processed_data(bundle="da", split="train", base_dir=base_dir)
    X_val, _ = load_processed_data(bundle="da", split="val", base_dir=base_dir)
    X_test, _ = load_processed_data(bundle="da", split="test", base_dir=base_dir)

    da_train_df = _load_da_split_df(base_dir, "train")
    da_val_df = _load_da_split_df(base_dir, "val")
    da_test_df = _load_da_split_df(base_dir, "test")

    y_train_s = pd.to_numeric(y_train[DA_TARGET], errors="coerce")
    mask_train = y_train_s.notna()
    X_train = X_train.loc[mask_train].copy()
    y_train_s = y_train_s.loc[mask_train].copy()
    da_train_df = da_train_df.loc[mask_train].copy()

    splitter = PurgedTimeSeriesSplit(
        n_splits=cv_n_splits,
        test_size=cv_test_size,
        gap_hours=cv_gap_hours,
        frequency="1h",
        min_train_size=500,
    )
    oof = pd.Series(np.nan, index=X_train.index, dtype="float64")
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X_train), start=1):
        m = _fit_xgb(
            X_train.iloc[tr_idx],
            y_train_s.iloc[tr_idx],
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            device=device,
        )
        pred = pd.Series(m.predict(X_train.iloc[va_idx]), index=X_train.iloc[va_idx].index)
        oof.loc[pred.index] = pd.to_numeric(pred, errors="coerce")
        print(f"[stack-da] OOF fold={fold} train={len(tr_idx)} val={len(va_idx)}")

    # Fill uncovered early rows with causal baseline; no future information used.
    if "da_price_lag_24h" not in X_train.columns:
        raise KeyError("Missing `da_price_lag_24h` for causal OOF fallback.")
    baseline = pd.to_numeric(X_train["da_price_lag_24h"], errors="coerce")
    oof = oof.fillna(baseline)

    # Last safety fallback for sparse edges.
    oof = oof.fillna(float(y_train_s.median()))

    final_model = _fit_xgb(
        X_train,
        y_train_s,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        device=device,
    )
    val_pred = pd.to_numeric(pd.Series(final_model.predict(X_val), index=X_val.index), errors="coerce")
    test_pred = pd.to_numeric(pd.Series(final_model.predict(X_test), index=X_test.index), errors="coerce")

    out_train = pd.DataFrame({"timestamp_utc": da_train_df["timestamp_utc"], STACK_COL: oof.values})
    out_val = pd.DataFrame({"timestamp_utc": da_val_df["timestamp_utc"], STACK_COL: val_pred.values})
    out_test = pd.DataFrame({"timestamp_utc": da_test_df["timestamp_utc"], STACK_COL: test_pred.values})
    for k in (out_train, out_val, out_test):
        k["timestamp_utc"] = pd.to_datetime(k["timestamp_utc"], utc=True, errors="coerce")
    return {"train": out_train, "val": out_val, "test": out_test}


def _copy_base_bundle(base_dir: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(base_dir, out_dir)


def _merge_into_afrr(out_dir: Path, stack_preds: dict[str, pd.DataFrame]) -> None:
    cfg = _read_cfg(out_dir)
    afrr_files = cfg["bundles"]["afrr"]["files"]
    for split in ("train", "val", "test"):
        p = Path(afrr_files[split])
        df = pd.read_parquet(p)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        merged = df.merge(stack_preds[split], on="timestamp_utc", how="left")
        if merged[STACK_COL].isna().any():
            # Causal and deterministic fallback for remaining gaps.
            if "da_price_lag_24h" in merged.columns:
                merged[STACK_COL] = merged[STACK_COL].fillna(pd.to_numeric(merged["da_price_lag_24h"], errors="coerce"))
            merged[STACK_COL] = merged[STACK_COL].fillna(float(pd.to_numeric(merged[STACK_COL], errors="coerce").median()))
        merged.to_parquet(p, index=False)

    # Save stacked DA predictions for audit/transparency.
    stack_dir = out_dir / "stacking"
    stack_dir.mkdir(parents=True, exist_ok=True)
    for split, sdf in stack_preds.items():
        sdf.to_parquet(stack_dir / f"da_pred_{split}.parquet", index=False)

    # Update config metadata + features.
    afrr_features = cfg["bundles"]["afrr"]["features"]
    if STACK_COL not in afrr_features:
        afrr_features.append(STACK_COL)
    cfg["bundles"]["afrr"]["features"] = afrr_features
    cfg["bundles"]["afrr"]["n_features"] = len(afrr_features)
    cfg.setdefault("stacking", {})
    cfg["stacking"]["enabled"] = True
    cfg["stacking"]["source"] = "da_oof_train_plus_da_model_val_test"
    cfg["stacking"]["column"] = STACK_COL
    cfg["stacking"]["files"] = {
        "train": str((stack_dir / "da_pred_train.parquet").resolve()),
        "val": str((stack_dir / "da_pred_val.parquet").resolve()),
        "test": str((stack_dir / "da_pred_test.parquet").resolve()),
    }

    # Normalize file pointers to this output dir.
    for b in ("da", "afrr"):
        bdir = out_dir / b
        cfg["bundles"][b]["files"] = {
            "train": str((bdir / "train.parquet").resolve()),
            "val": str((bdir / "val.parquet").resolve()),
            "test": str((bdir / "test.parquet").resolve()),
        }
        qrep = bdir / "feature_quality_report.csv"
        if qrep.exists():
            cfg["bundles"][b]["feature_quality_report"] = str(qrep.resolve())

    cfg_path = out_dir / "feature_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def main() -> None:
    args = _build_cli().parse_args()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    _read_cfg(base_dir)  # fail fast

    if args.device == "cuda" and not args.allow_cpu:
        # quick fail-fast probe consistent with training rules
        try:
            from xgboost import XGBRegressor

            probe = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=1,
                max_depth=1,
                learning_rate=0.1,
                tree_method="hist",
                device="cuda",
                n_jobs=1,
            )
            probe.fit(np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float), np.array([0.0, 1.0, 2.0, 3.0], dtype=float), verbose=False)
        except Exception as exc:
            raise RuntimeError("CUDA required for stacking build, but unavailable.") from exc

    print(f"[stack-da] base={base_dir}")
    print(f"[stack-da] out={out_dir}")
    stack_preds = _build_da_predictions(
        base_dir=base_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        device=args.device,
        cv_n_splits=args.cv_n_splits,
        cv_test_size=args.cv_test_size,
        cv_gap_hours=args.cv_gap_hours,
    )
    _copy_base_bundle(base_dir, out_dir)
    _merge_into_afrr(out_dir, stack_preds)
    print("[OK] Built stacked aFRR bundle with leakage-safe DA predictions.")
    print(f"- stacking feature: {STACK_COL}")
    print(f"- output bundle dir: {out_dir}")


if __name__ == "__main__":
    main()
