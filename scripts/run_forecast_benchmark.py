#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from energy_trading.evaluation.forecast_benchmark import run_benchmark


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
    p.add_argument("--benchmark-config", required=True)
    p.add_argument("--model-run-manifest", action="append", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--splits", default="val,test")
    p.add_argument("--truth-source", default="data/features/all_data_features.parquet")
    p.add_argument("--min-join-coverage", type=float, default=0.999)
    p.add_argument("--fail-on-missing-truth", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--make-figures", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--example-weeks", default="auto")
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_config(Path(args.benchmark_config))
    cfg["example_weeks"] = args.example_weeks
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    art = run_benchmark(
        config=cfg,
        model_run_manifests=[Path(p).resolve() for p in args.model_run_manifest],
        out_dir=Path(args.out_dir).resolve(),
        splits=splits,
        truth_source=Path(args.truth_source).resolve(),
        min_join_coverage=float(args.min_join_coverage),
        fail_on_missing_truth=bool(args.fail_on_missing_truth),
        make_figures=bool(args.make_figures),
        overwrite=bool(args.overwrite),
    )
    print(f"[OK] Forecast benchmark completed: {art.benchmark_dir}")
    print(f"[OK] Diagnostics: {art.diagnostics_dir}")
    print(f"[OK] Metrics: {art.metrics_dir}")
    print(f"[OK] Figures: {art.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
