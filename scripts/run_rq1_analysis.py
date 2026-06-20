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
    p.add_argument("--out-dir", default="artifacts/final_benchmark")
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
    work_dir = out_dir / "_raw_outputs"
    full_dir = out_dir / "4_1_1_full_unweighted"
    calibration_dir = out_dir / "4_1_2_calibration_uncertainty"
    per_lead_dir = out_dir / "4_1_3_per_lead"
    gate_dir = out_dir / "4_1_4_gate_specific"
    tail_dir = out_dir / "4_1_5_tail_spike"
    interim_dir = out_dir / "4_1_6_interim_answer"
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
            str(work_dir / "4_1_1_full_unweighted"),
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
            str(work_dir / "4_1_2_calibration_uncertainty"),
            "--legacy-flat-out-dir",
            str(work_dir / "calibration"),
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
            str(work_dir / "shared"),
            "--structured-out-dir",
            str(work_dir / "4_1_3_per_lead"),
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
            str(work_dir / "shared"),
            "--structured-out-dir",
            str(work_dir / "4_1_4_gate_specific"),
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
            str(work_dir / "shared"),
            "--structured-out-dir",
            str(work_dir / "4_1_5_tail_spike"),
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
            str(work_dir / "4_1_7_example_weeks"),
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
            str(work_dir),
            "--rq1-root",
            str(out_dir),
            "--split",
            str(args.split),
            "--prune-legacy",
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
                "output_dir": str(interim_dir),
            },
            {
                "section": "4.1.7",
                "name": "Example weeks",
                "status": "skipped" if args.skip_example_weeks else "implemented",
                "output_dir": str(example_dir),
            },
        ],
        "steps": steps,
        "canonical_output_layout": {
            "root": str(out_dir),
            "subsection_tiers": ["result_section", "appendix", "backup"],
            "artifact_subfolders": {
                "result_section": ["figures", "tables"],
                "appendix": ["figures", "tables"],
                "backup": ["csv", "diagnostics", "warnings"],
            },
        },
        "expected_example_week_outputs_dir": str(example_dir),
        "organized_manifest": str(out_dir / "rq1_output_manifest.json"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for subsection in manifest["subsections"]:
        Path(str(subsection["output_dir"])).mkdir(parents=True, exist_ok=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = interim_dir / "backup" / "diagnostics" / "rq1_wrapper_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] RQ1 wrapper complete: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
