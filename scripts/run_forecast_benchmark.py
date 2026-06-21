#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from energy_trading.evaluation.forecast_benchmark import TARGETS, _load_latest_or_manifest, _model_name, run_benchmark


DEFAULT_BENCHMARK_CONFIG = "configs/forecast_benchmark.yaml"
DEFAULT_MODEL_RUN_MANIFESTS = [
    "artifacts/model_runs/latest_xgboost.json",
    "artifacts/model_runs/latest_linear.json",
    "artifacts/model_runs/latest_tft.json",
]
DEFAULT_OUT_DIR = "artifacts/rq1_ml_model_benchmark"


def _load_config(path: Path) -> dict[str, Any]:
    txt = path.read_text(encoding="utf-8")
    try:
        import yaml

        obj = yaml.safe_load(txt)
        if not isinstance(obj, dict):
            raise ValueError("config root must be a mapping")
        return obj
    except ImportError:
        return json.loads(txt)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run reproducible forecast benchmark")
    p.add_argument("--benchmark-config", default=DEFAULT_BENCHMARK_CONFIG)
    p.add_argument(
        "--model-run-manifest",
        action="append",
        default=None,
        help=(
            "Model-run manifest path. Can be passed multiple times. "
            "Defaults to latest XGB, RLQR/linear and TFT manifests."
        ),
    )
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--splits", default="test")
    p.add_argument("--truth-source", default="data/features/all_data_features.parquet")
    p.add_argument("--min-join-coverage", type=float, default=0.999)
    p.add_argument("--fail-on-missing-truth", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--make-figures", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-joined-predictions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--example-weeks", default="auto")
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _expected_joined_files(*, model_run_manifests: list[Path], splits: list[str], benchmark_dir: Path) -> list[Path]:
    models: list[str] = []
    seen: set[str] = set()
    for pointer in model_run_manifests:
        manifest_path, manifest = _load_latest_or_manifest(pointer)
        model = _model_name(manifest_path, manifest)
        if model in seen:
            continue
        seen.add(model)
        models.append(model)
    joined_dir = benchmark_dir / "diagnostics" / "joined_predictions"
    return [
        joined_dir / f"{model}__{split}__{target}.parquet"
        for model in models
        for split in splits
        for target in TARGETS
    ]


def _assert_complete_joined_predictions(*, model_run_manifests: list[Path], splits: list[str], benchmark_dir: Path) -> None:
    expected = _expected_joined_files(model_run_manifests=model_run_manifests, splits=splits, benchmark_dir=benchmark_dir)
    missing = [p for p in expected if not p.exists()]
    if not missing:
        return
    preview = "\n".join(f"  - {p}" for p in missing[:12])
    if len(missing) > 12:
        preview += f"\n  - ... {len(missing) - 12} more"
    raise FileNotFoundError(
        "Forecast benchmark did not produce the complete joined-prediction set required by RQ1.\n"
        f"Missing {len(missing)} expected parquet file(s):\n{preview}\n"
        "Check that all model manifests contain test predictions_long entries for all seven RQ1 targets."
    )


def main() -> int:
    args = parse_args()
    cfg = _load_config(Path(args.benchmark_config))
    cfg["example_weeks"] = args.example_weeks
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    model_run_manifests = args.model_run_manifest if args.model_run_manifest is not None else DEFAULT_MODEL_RUN_MANIFESTS
    model_run_manifest_paths = [Path(p).resolve() for p in model_run_manifests]
    make_figures = bool(args.make_figures)
    art = run_benchmark(
        config=cfg,
        model_run_manifests=model_run_manifest_paths,
        out_dir=Path(args.out_dir).resolve(),
        splits=splits,
        truth_source=Path(args.truth_source).resolve(),
        min_join_coverage=float(args.min_join_coverage),
        fail_on_missing_truth=bool(args.fail_on_missing_truth),
        make_figures=make_figures,
        save_joined_predictions=bool(args.save_joined_predictions),
        overwrite=bool(args.overwrite),
    )
    if bool(args.save_joined_predictions):
        _assert_complete_joined_predictions(
            model_run_manifests=model_run_manifest_paths,
            splits=splits,
            benchmark_dir=art.benchmark_dir,
        )
    print(f"[OK] Forecast benchmark completed: {art.benchmark_dir}")
    print(f"[OK] Diagnostics: {art.diagnostics_dir}")
    print(f"[OK] Metrics: {art.metrics_dir}")
    print(f"[OK] Figures: {art.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
