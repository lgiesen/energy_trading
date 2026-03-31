"""Train DA + aFRR XGBoost models and export a versioned run artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


AFRR_TARGETS = [
    "target_afrr_activation_price_vwap_pos_h1",
    "target_afrr_activation_price_vwap_neg_h1",
    "target_afrr_rate_h1",
    "target_afrr_capacity_price_pos_h1",
    "target_afrr_capacity_price_neg_h1",
]


def _run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _run_train_cmd(cmd: list[str]) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_fragment(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest fragment: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_prediction_files(paths: list[Path], out_path: Path) -> Path:
    merged: pd.DataFrame | None = None
    for p in paths:
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if merged is None:
            merged = df
        else:
            cols = [c for c in df.columns if c != "timestamp_utc"]
            merged = merged.merge(df[["timestamp_utc", *cols]], on="timestamp_utc", how="outer")
    if merged is None:
        raise FileNotFoundError("No prediction files to merge.")
    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and export DA+aFRR model run artifacts.")
    p.add_argument("--base-dir", default="data/model_input")
    p.add_argument("--run-root", default="artifacts/model_runs")
    p.add_argument("--run-id", default="")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--forecast-horizon-hours", type=int, default=48)
    p.add_argument("--ground-truth-path", default="data/features/all_data_features.parquet")
    p.add_argument(
        "--enable-da-stacking-afrr",
        action="store_true",
        help="Build leakage-safe DA stacking feature and train aFRR on stacked bundle dir.",
    )
    p.add_argument(
        "--stacked-base-dir",
        default="data/model_input_stacked",
        help="Output dir for stacked bundles when --enable-da-stacking-afrr is used.",
    )
    p.add_argument("--stack-cv-n-splits", type=int, default=8)
    p.add_argument("--stack-cv-test-size", type=int, default=24 * 7)
    p.add_argument("--stack-cv-gap-hours", type=int, default=72)
    p.add_argument("--skip-latest-pointer", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id.strip() or _run_id_now()
    run_dir = Path(args.run_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    da_fragment = run_dir / "da_manifest_fragment.json"
    afrr_fragment_paths: list[Path] = []
    afrr_base_dir = args.base_dir

    if args.enable_da_stacking_afrr:
        stack_cmd = [
            sys.executable,
            "scripts/build_afrr_da_stacking_feature.py",
            "--base-dir",
            args.base_dir,
            "--out-dir",
            args.stacked_base_dir,
            "--n-estimators",
            str(args.n_estimators),
            "--max-depth",
            str(args.max_depth),
            "--learning-rate",
            str(args.learning_rate),
            "--device",
            args.device,
            "--cv-n-splits",
            str(args.stack_cv_n_splits),
            "--cv-test-size",
            str(args.stack_cv_test_size),
            "--cv-gap-hours",
            str(args.stack_cv_gap_hours),
        ]
        if args.allow_cpu:
            stack_cmd.append("--allow-cpu")
        _run_train_cmd(stack_cmd)
        afrr_base_dir = args.stacked_base_dir

    base_cmd = [
        sys.executable,
        "-m",
        "src.energy_trading.models.train_xgboost",
        "--device",
        args.device,
        "--n-estimators",
        str(args.n_estimators),
        "--max-depth",
        str(args.max_depth),
        "--learning-rate",
        str(args.learning_rate),
        "--early-stopping-rounds",
        str(args.early_stopping_rounds),
        "--run-dir",
        str(run_dir),
        "--prediction-splits",
        "val,test",
        "--forecast-horizon-hours",
        str(args.forecast_horizon_hours),
    ]
    if args.allow_cpu:
        base_cmd.append("--allow-cpu")

    # Train DA model.
    cmd_da = [*base_cmd, "--bundle", "da", "--manifest-fragment-out", str(da_fragment)]
    cmd_da = [*cmd_da, "--base-dir", args.base_dir]
    _run_train_cmd(cmd_da)

    # Train aFRR models target-wise to produce full canonical prediction columns.
    for tgt in AFRR_TARGETS:
        frag = run_dir / f"afrr_{tgt}_manifest_fragment.json"
        afrr_fragment_paths.append(frag)
        cmd_afrr = [
            *base_cmd,
            "--base-dir",
            afrr_base_dir,
            "--bundle",
            "afrr",
            "--target-col",
            tgt,
            "--manifest-fragment-out",
            str(frag),
        ]
        _run_train_cmd(cmd_afrr)

    da_meta = _load_fragment(da_fragment)
    afrr_meta_list = [_load_fragment(p) for p in afrr_fragment_paths if p.exists()]
    if not afrr_meta_list:
        raise RuntimeError("No aFRR manifest fragments found.")

    # Merge aFRR split prediction files into one file per split.
    afrr_pred_val = _merge_prediction_files(
        [Path(m["predictions"]["val"]) for m in afrr_meta_list if "val" in m.get("predictions", {})],
        run_dir / "predictions" / "afrr_val.parquet",
    )
    afrr_pred_test = _merge_prediction_files(
        [Path(m["predictions"]["test"]) for m in afrr_meta_list if "test" in m.get("predictions", {})],
        run_dir / "predictions" / "afrr_test.parquet",
    )

    afrr_long_by_split: dict[str, dict[str, str]] = {"val": {}, "test": {}}
    for meta in afrr_meta_list:
        for split, pred_map in meta.get("predictions_long", {}).items():
            for pred_col, path in pred_map.items():
                afrr_long_by_split.setdefault(split, {})
                afrr_long_by_split[split][pred_col] = path

    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundles": {
            "da": {
                "model_path": da_meta["model_path"],
                "metrics_path": da_meta["metrics_path"],
                "predictions": da_meta["predictions"],
                "predictions_long": da_meta.get("predictions_long", {}),
                "prediction_columns": da_meta["prediction_columns"],
                "target_columns": da_meta["target_columns"],
            },
            "afrr": {
                "model_paths": [m["model_path"] for m in afrr_meta_list],
                "metrics_paths": [m["metrics_path"] for m in afrr_meta_list],
                "predictions": {
                    "val": str(afrr_pred_val.resolve()),
                    "test": str(afrr_pred_test.resolve()),
                },
                "predictions_long": afrr_long_by_split,
                "prediction_columns": [
                    "pred_afrr_capacity_price_pos",
                    "pred_afrr_capacity_price_neg",
                    "pred_afrr_activation_price_pos",
                    "pred_afrr_activation_price_neg",
                    "pred_afrr_activation_rate_pos",
                    "pred_afrr_activation_rate_neg",
                ],
                "target_columns": AFRR_TARGETS,
            },
        },
        "ground_truth": {
            "default_path": str(Path(args.ground_truth_path).resolve()),
        },
        "simulation": {
            "default_split": "test",
        },
    }

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.skip_latest_pointer:
        latest_path = Path(args.run_root) / "latest.json"
        latest_payload = {
            "run_id": run_id,
            "manifest_path": str(manifest_path.resolve()),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        latest_path.write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")

    print("[OK] Run export complete.")
    print(f"- run_id: {run_id}")
    print(f"- run_dir: {run_dir}")
    print(f"- manifest: {manifest_path}")
    if not args.skip_latest_pointer:
        print(f"- latest: {Path(args.run_root) / 'latest.json'}")


if __name__ == "__main__":
    main()
