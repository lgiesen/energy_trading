#!/usr/bin/env python3
"""Run implemented RQ1 analysis outputs.

The wrapper intentionally delegates to subsection scripts. This keeps the
individual thesis outputs auditable while still providing one reproducible
entry point for RQ1 artifact generation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = "test"
DEFAULT_SPLITS = "test"
DEFAULT_MODELS = "tft,xgboost,linear"


def _split_csv(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _run_step(name: str, cmd: list[str], *, log_dir: Path) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    log_path = log_dir / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        returncode = proc.wait()
    elapsed = time.perf_counter() - started
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return {
        "name": name,
        "command": cmd,
        "log_path": str(log_path),
        "elapsed_seconds": elapsed,
        "returncode": returncode,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run implemented RQ1 thesis analyses.")
    p.add_argument("--benchmark-root", default="artifacts/forecast_benchmarks")
    p.add_argument("--benchmark-dir", default=None)
    p.add_argument("--out-dir", default="artifacts/final_benchmark/rq1")
    p.add_argument("--split", default=DEFAULT_SPLIT, help="Main thesis/reporting split. Defaults to test.")
    p.add_argument(
        "--splits",
        default=DEFAULT_SPLITS,
        help=(
            "Comma-separated splits to load from the forecast benchmark. Defaults to test only for final RQ1 reporting."
        ),
    )
    p.add_argument("--models", default=DEFAULT_MODELS, help="Models to compare. Defaults to all RQ1 models: TFT, XGB and RLQR.")
    p.add_argument("--targets", default="", help="Optional comma-separated targets for example-week plots.")
    p.add_argument("--lead", type=float, default=24.0)
    p.add_argument("--quantile", default="p50")
    p.add_argument("--date", default=None, help="Optional custom week start for example-week plots.")
    p.add_argument("--typical-start", default="2025-03-30T22:00:00Z")
    p.add_argument("--high-volatility-start", default="2025-10-05T22:00:00Z")
    p.add_argument("--window-hours", type=int, default=168)
    p.add_argument("--skip-full-metrics", action="store_true")
    p.add_argument("--skip-calibration", action="store_true")
    p.add_argument("--skip-per-lead", action="store_true")
    p.add_argument("--skip-gate-buckets", action="store_true")
    p.add_argument("--skip-tail-spike", action="store_true")
    p.add_argument("--skip-example-weeks", action="store_true")
    p.add_argument("--skip-organize", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    full_dir = out_dir / "4_1_1_full_unweighted"
    calibration_dir = out_dir / "4_1_2_calibration_uncertainty"
    per_lead_dir = out_dir / "4_1_3_per_lead"
    gate_dir = out_dir / "4_1_4_gate_specific"
    tail_dir = out_dir / "4_1_5_tail_spike"
    example_dir = out_dir / "4_1_7_example_weeks"
    log_dir = out_dir / "logs" / "rq1_wrapper"
    steps: list[dict[str, Any]] = []
    py = sys.executable

    benchmark_args: list[str] = ["--benchmark-root", args.benchmark_root]
    if args.benchmark_dir:
        benchmark_args.extend(["--benchmark-dir", args.benchmark_dir])

    if not args.skip_full_metrics:
        cmd = [
            py,
            "scripts/build_final_full_forecast_metrics.py",
            *benchmark_args,
            "--out-dir",
            str(full_dir),
            "--split",
            str(args.split),
            "--splits",
            str(args.splits),
            "--models",
            str(args.models),
        ]
        steps.append(_run_step("rq1_4_1_1_full_unweighted_metrics", cmd, log_dir=log_dir))

    if not args.skip_calibration:
        cmd = [
            py,
            "scripts/build_final_calibration_uncertainty.py",
            *benchmark_args,
            "--out-dir",
            str(calibration_dir),
            "--split",
            str(args.split),
            "--splits",
            str(args.splits),
            "--models",
            str(args.models),
        ]
        steps.append(_run_step("rq1_4_1_2_calibration_uncertainty", cmd, log_dir=log_dir))

    if not args.skip_per_lead:
        cmd = [
            py,
            "scripts/build_final_per_lead_benchmark.py",
            *benchmark_args,
            "--out-dir",
            "artifacts/final_benchmark",
            "--structured-out-dir",
            str(per_lead_dir),
            "--split",
            str(args.split),
            "--splits",
            str(args.splits),
            "--models",
            str(args.models),
        ]
        steps.append(_run_step("rq1_4_1_3_per_lead_hour", cmd, log_dir=log_dir))

    if not args.skip_gate_buckets:
        cmd = [
            py,
            "scripts/build_final_gate_bucket_benchmark.py",
            *benchmark_args,
            "--out-dir",
            "artifacts/final_benchmark",
            "--structured-out-dir",
            str(gate_dir),
            "--split",
            str(args.split),
            "--splits",
            str(args.splits),
            "--models",
            str(args.models),
        ]
        steps.append(_run_step("rq1_4_1_4_gate_actionable", cmd, log_dir=log_dir))

    if not args.skip_tail_spike:
        cmd = [
            py,
            "scripts/build_final_tail_spike_benchmark.py",
            *benchmark_args,
            "--out-dir",
            "artifacts/final_benchmark",
            "--structured-out-dir",
            str(tail_dir),
            "--split",
            str(args.split),
            "--splits",
            str(args.splits),
            "--models",
            str(args.models),
        ]
        steps.append(_run_step("rq1_4_1_5_tail_spike", cmd, log_dir=log_dir))

    if not args.skip_example_weeks:
        cmd = [
            py,
            "scripts/build_rq1_example_weeks.py",
            *benchmark_args,
            "--out-dir",
            str(example_dir),
            "--split",
            str(args.split),
            "--models",
            str(args.models),
            "--lead",
            str(args.lead),
            "--quantile",
            str(args.quantile),
            "--typical-start",
            str(args.typical_start),
            "--high-volatility-start",
            str(args.high_volatility_start),
            "--window-hours",
            str(args.window_hours),
        ]
        targets = _split_csv(args.targets)
        if targets:
            cmd.extend(["--targets", ",".join(targets)])
        if args.date:
            cmd.extend(["--date", str(args.date)])
        steps.append(_run_step("rq1_4_1_7_example_weeks", cmd, log_dir=log_dir))

    if not args.skip_organize:
        cmd = [
            py,
            "scripts/organize_rq1_outputs.py",
            "--final-root",
            "artifacts/final_benchmark",
            "--rq1-root",
            str(out_dir),
            "--split",
            str(args.split),
        ]
        steps.append(_run_step("rq1_output_organization", cmd, log_dir=log_dir))

    manifest = {
        "description": "RQ1 wrapper manifest for implemented thesis analysis scripts.",
        "out_dir": str(out_dir),
        "split": args.split,
        "models": args.models,
        "subsections": [
            {
                "section": "4.1.1",
                "name": "Full unweighted forecast metrics",
                "status": "skipped" if args.skip_full_metrics else "implemented",
                "output_dir": str(full_dir),
            },
            {
                "section": "4.1.2",
                "name": "Calibration and uncertainty quality",
                "status": "skipped" if args.skip_calibration else "implemented",
                "output_dir": str(calibration_dir),
            },
            {
                "section": "4.1.3",
                "name": "Per-lead-hour performance",
                "status": "skipped" if args.skip_per_lead else "implemented",
                "output_dir": str(per_lead_dir),
            },
            {
                "section": "4.1.4",
                "name": "Gate-specific actionable forecast performance",
                "status": "skipped" if args.skip_gate_buckets else "implemented",
                "output_dir": str(gate_dir),
            },
            {
                "section": "4.1.5",
                "name": "Tail and spike behavior",
                "status": "skipped" if args.skip_tail_spike else "implemented",
                "output_dir": str(tail_dir),
            },
            {
                "section": "4.1.6",
                "name": "Interim answer to RQ1",
                "status": "not_implemented",
                "output_dir": str(out_dir / "4_1_6_interim_answer"),
            },
            {
                "section": "4.1.7",
                "name": "Example weeks",
                "status": "skipped" if args.skip_example_weeks else "implemented",
                "output_dir": str(example_dir),
            },
        ],
        "steps": steps,
        "expected_gate_bucket_outputs": [
            "artifacts/final_benchmark/gate_bucket_metrics.csv",
            f"artifacts/final_benchmark/gate_bucket_metrics_{args.split}.csv",
            "artifacts/final_benchmark/gate_bucket_row_counts.csv",
            "artifacts/final_benchmark/gate_bucket_definitions.csv",
            "artifacts/final_benchmark/gate_bucket_observed_leads.csv",
            "artifacts/final_benchmark/gate_bucket_warnings.csv",
            f"artifacts/final_benchmark/latex/gate_bucket_metrics_{args.split}.tex",
            "artifacts/final_benchmark/figures/gate_bucket_pinball_by_target_group.png",
            "artifacts/final_benchmark/figures/gate_bucket_mae_p50_by_target_group.png",
            "artifacts/final_benchmark/figures/gate_bucket_coverage_p10_p90_by_target_group.png",
            "artifacts/final_benchmark/figures/gate_bucket_observed_leads.png",
        ],
        "expected_tail_spike_outputs": [
            "artifacts/final_benchmark/tail_spike_metrics.csv",
            f"artifacts/final_benchmark/tail_spike_metrics_{args.split}.csv",
            "artifacts/final_benchmark/tail_spike_regime_definitions.csv",
            "artifacts/final_benchmark/tail_spike_thresholds.csv",
            "artifacts/final_benchmark/tail_spike_row_counts.csv",
            "artifacts/final_benchmark/tail_spike_selected_weeks.csv",
            "artifacts/final_benchmark/tail_spike_warnings.csv",
            f"artifacts/final_benchmark/latex/tail_spike_metrics_{args.split}.tex",
            "artifacts/final_benchmark/figures/tail_spike_relative_pinball_by_regime.png",
            "artifacts/final_benchmark/figures/tail_spike_pinball_by_regime.png",
            "artifacts/final_benchmark/figures/tail_spike_mae_p50_by_regime.png",
            "artifacts/final_benchmark/figures/tail_spike_coverage_by_regime.png",
            "artifacts/final_benchmark/figures/tail_spike_true_vs_p50_hexbin.png",
            "artifacts/final_benchmark/figures/tail_spike_residual_distribution_by_regime.png",
        ],
        "expected_per_lead_outputs": [
            "artifacts/final_benchmark/per_lead_metrics.csv",
            f"artifacts/final_benchmark/per_lead_metrics_{args.split}.csv",
            f"artifacts/final_benchmark/per_lead_range_summary_{args.split}.csv",
            f"artifacts/final_benchmark/per_lead_row_counts_{args.split}.csv",
            f"artifacts/final_benchmark/latex/per_lead_range_summary_{args.split}.tex",
            "artifacts/final_benchmark/figures/per_lead_pinball_da_price.png",
            "artifacts/final_benchmark/figures/per_lead_pinball_afrr_capacity_price.png",
            "artifacts/final_benchmark/figures/per_lead_pinball_afrr_activation_price.png",
            "artifacts/final_benchmark/figures/per_lead_pinball_afrr_activation_rate.png",
            "artifacts/final_benchmark/figures/per_lead_relative_pinball_da_price.png",
            "artifacts/final_benchmark/figures/per_lead_relative_pinball_afrr_capacity_price.png",
            "artifacts/final_benchmark/figures/per_lead_relative_pinball_afrr_activation_price.png",
            "artifacts/final_benchmark/figures/per_lead_relative_pinball_afrr_activation_rate.png",
        ],
        "expected_full_metrics_outputs": [
            str(full_dir / "csv" / "rq1_4_1_1_forecast_metrics_full_long.csv"),
            str(full_dir / "csv" / f"rq1_4_1_1_forecast_metrics_full_primary_{args.split}.csv"),
            str(full_dir / "csv" / f"rq1_4_1_1_forecast_metrics_full_detailed_{args.split}.csv"),
            str(full_dir / "csv" / f"rq1_4_1_1_forecast_metrics_full_alignment_diagnostics_{args.split}.csv"),
            str(full_dir / "latex" / f"rq1_4_1_1_forecast_metrics_full_primary_{args.split}.tex"),
            str(full_dir / "latex" / f"rq1_4_1_1_forecast_metrics_full_detailed_{args.split}.tex"),
            str(full_dir / "figures" / f"rq1_4_1_1_forecast_metrics_full_relative_pinball_{args.split}.png"),
        ],
        "expected_calibration_outputs": [
            str(calibration_dir / "csv" / "rq1_4_1_2_calibration_quantile_coverage.csv"),
            str(calibration_dir / "csv" / f"rq1_4_1_2_calibration_quantile_coverage_{args.split}.csv"),
            str(calibration_dir / "csv" / "rq1_4_1_2_calibration_interval_coverage_width.csv"),
            str(calibration_dir / "csv" / f"rq1_4_1_2_calibration_interval_coverage_width_{args.split}.csv"),
            str(calibration_dir / "csv" / "rq1_4_1_2_calibration_quantile_crossing.csv"),
            str(calibration_dir / "csv" / f"rq1_4_1_2_calibration_quantile_crossing_{args.split}.csv"),
            str(calibration_dir / "csv" / "rq1_4_1_2_calibration_summary.csv"),
            str(calibration_dir / "csv" / f"rq1_4_1_2_calibration_summary_{args.split}.csv"),
            str(calibration_dir / "csv" / "rq1_4_1_2_calibration_row_counts.csv"),
            str(calibration_dir / "csv" / "rq1_4_1_2_calibration_warnings.csv"),
            str(calibration_dir / "latex" / f"rq1_4_1_2_calibration_summary_{args.split}.tex"),
            str(calibration_dir / "latex" / f"rq1_4_1_2_calibration_quantile_coverage_{args.split}_appendix.tex"),
            str(calibration_dir / "latex" / f"rq1_4_1_2_calibration_interval_quality_{args.split}_appendix.tex"),
            str(calibration_dir / "latex" / f"rq1_4_1_2_calibration_quantile_crossing_{args.split}_appendix.tex"),
            str(calibration_dir / "figures" / "rq1_4_1_2_calibration_reliability_by_target_group.png"),
            str(calibration_dir / "figures" / "rq1_4_1_2_calibration_interval_coverage_by_target_group.png"),
            str(calibration_dir / "figures" / "rq1_4_1_2_calibration_interval_width_by_target_group.png"),
            str(calibration_dir / "figures" / "rq1_4_1_2_calibration_quantile_crossing_by_target_group.png"),
        ],
        "expected_example_week_outputs_dir": str(example_dir),
        "organized_manifest": str(out_dir / "rq1_output_manifest.json"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for subsection in manifest["subsections"]:
        Path(str(subsection["output_dir"])).mkdir(parents=True, exist_ok=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "rq1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] RQ1 wrapper complete: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
