#!/usr/bin/env python3
"""Budgeted Optuna tuning for TFT on fixed chronological train/val split.

Best-practice principles implemented:
- deterministic seed handling,
- explicit trial budget,
- robust objective fallback,
- persisted artifacts (JSON + trials CSV),
- isolated trial run dirs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tune TFT hyperparameters with Optuna.")
    p.add_argument("--base-dir", default="data/model_input")
    p.add_argument("--bundle", choices=["da", "afrr"], default="da")
    p.add_argument("--target-col", default="target_da_price")
    p.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    p.add_argument(
        "--precision",
        choices=["auto", "32-true", "16-mixed", "bf16-mixed"],
        default="auto",
        help="Forwarded to train_tft_export. Use bf16-mixed or 32-true if 16-mixed overflows.",
    )
    p.add_argument("--n-trials", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout-seconds", type=int, default=0, help="0 disables timeout.")
    p.add_argument("--study-name", default="tft_budgeted_tuning")
    p.add_argument("--selection-metric", default="mae_val")
    p.add_argument("--fallback-metric", default="rmse_val")
    p.add_argument("--out-dir", default="artifacts/hpo")
    p.add_argument("--run-root", default="artifacts/hpo/tft_trials")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lead-weight-start", type=int, default=16)
    p.add_argument("--lead-weight-end", type=int, default=48)
    p.add_argument("--lead-weight-max", type=float, default=2.0)
    p.add_argument("--cleanup-trial-runs", action="store_true", help="Delete trial dirs after successful readout.")
    return p


def _safe_float(v: object) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def main() -> None:
    args = _build_cli().parse_args()
    is_smoke = os.environ.get("IS_SMOKE_TEST", "0") == "1"
    if is_smoke:
        args.n_trials = min(int(args.n_trials), 2)
        args.timeout_seconds = min(int(args.timeout_seconds or 1200), 1200)

    try:
        import optuna
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency 'optuna' for TFT tuning.") from exc

    np.random.seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    trial_rows: list[dict[str, object]] = []
    best_metric = str(args.selection_metric)
    fallback_metric = str(args.fallback_metric)

    def _objective(trial: "optuna.Trial") -> float:
        # Memory-aware capacity ladder:
        # try smaller configurations first and only scale up if they improve the
        # selected validation objective.
        hidden_size = trial.suggest_categorical("hidden_size", [32, 48, 64])
        attention_head_size = trial.suggest_categorical("attention_head_size", [2, 4])
        dropout = trial.suggest_float("dropout", 0.05, 0.35)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.01, 0.30, log=True)
        max_encoder_length = trial.suggest_categorical("max_encoder_length", [96, 168, 240])
        max_epochs = trial.suggest_int("max_epochs", 40, 140, step=20)
        early_stopping_patience = trial.suggest_int("early_stopping_patience", 4, 16, step=2)

        trial_dir = run_root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = trial_dir / "metrics.json"

        cmd = [
            sys.executable,
            "-m",
            "src.energy_trading.models.train_tft_export",
            "--base-dir",
            str(args.base_dir),
            "--bundle",
            str(args.bundle),
            "--target-col",
            str(args.target_col),
            "--run-dir",
            str(trial_dir),
            "--model-name",
            "tft_hpo",
            "--device",
            str(args.device),
            "--precision",
            str(args.precision),
            "--seed",
            str(args.seed),
            "--max-encoder-length",
            str(max_encoder_length),
            "--max-prediction-length",
            "48",
            "--num-workers",
            str(args.num_workers),
            "--learning-rate",
            str(learning_rate),
            "--gradient-clip-val",
            str(gradient_clip_val),
            "--hidden-size",
            str(hidden_size),
            "--attention-head-size",
            str(attention_head_size),
            "--dropout",
            str(dropout),
            "--max-epochs",
            str(max_epochs),
            "--early-stopping-patience",
            str(early_stopping_patience),
            "--lead-weight-start",
            str(args.lead_weight_start),
            "--lead-weight-end",
            str(args.lead_weight_end),
            "--lead-weight-max",
            str(args.lead_weight_max),
            "--metrics-json-out",
            str(metrics_path),
            "--cleanup-lightning-checkpoints",
        ]

        t0 = time.perf_counter()
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metric_primary = _safe_float(payload.get(best_metric))
            metric_fallback = _safe_float(payload.get(fallback_metric))
            objective = metric_primary if np.isfinite(metric_primary) else metric_fallback
            if not np.isfinite(objective):
                objective = float("inf")
            trial.set_user_attr("metric_primary", metric_primary)
            trial.set_user_attr("metric_fallback", metric_fallback)
            trial.set_user_attr("status", "ok")
        except Exception as exc:
            objective = float("inf")
            trial.set_user_attr("metric_primary", float("nan"))
            trial.set_user_attr("metric_fallback", float("nan"))
            trial.set_user_attr("status", f"failed: {exc}")

        dt = time.perf_counter() - t0
        trial_rows.append(
            {
                "trial": int(trial.number),
                "objective": float(objective),
                "status": trial.user_attrs.get("status"),
                "metric_primary": trial.user_attrs.get("metric_primary"),
                "metric_fallback": trial.user_attrs.get("metric_fallback"),
                "hidden_size": int(hidden_size),
                "attention_head_size": int(attention_head_size),
                "dropout": float(dropout),
                "learning_rate": float(learning_rate),
                "gradient_clip_val": float(gradient_clip_val),
                "max_encoder_length": int(max_encoder_length),
                "max_epochs": int(max_epochs),
                "early_stopping_patience": int(early_stopping_patience),
                "duration_seconds": float(dt),
                "trial_dir": str(trial_dir.resolve()),
            }
        )

        if args.cleanup_trial_runs and np.isfinite(objective):
            shutil.rmtree(trial_dir, ignore_errors=True)

        return float(objective)

    sampler = optuna.samplers.TPESampler(seed=int(args.seed), multivariate=True)
    study = optuna.create_study(direction="minimize", study_name=str(args.study_name), sampler=sampler)
    study.optimize(
        _objective,
        n_trials=int(args.n_trials),
        timeout=(int(args.timeout_seconds) if int(args.timeout_seconds) > 0 else None),
    )

    best = study.best_trial
    result = {
        "study_name": str(args.study_name),
        "bundle": str(args.bundle),
        "target_col": str(args.target_col),
        "device": str(args.device),
        "n_trials": int(args.n_trials),
        "selection_metric": best_metric,
        "fallback_metric": fallback_metric,
        "best_trial": int(best.number),
        "best_objective_value": float(best.value),
        "best_primary_metric": _safe_float(best.user_attrs.get("metric_primary")),
        "best_fallback_metric": _safe_float(best.user_attrs.get("metric_fallback")),
        "best_params": best.params,
    }

    out_json = out_dir / f"tft_optuna_{args.bundle}_{args.target_col}.json"
    out_csv = out_dir / f"tft_optuna_{args.bundle}_{args.target_col}_trials.csv"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(trial_rows).sort_values("objective").to_csv(out_csv, index=False)

    print("[OK] TFT tuning finished.")
    print(f"- Selection metric: {best_metric}")
    print(f"- Best objective value: {result['best_objective_value']:.6f}")
    print(f"- Best params: {result['best_params']}")
    print(f"- Result JSON: {out_json}")
    print(f"- Trials CSV: {out_csv}")


if __name__ == "__main__":
    main()
