#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.rq3_pipeline_utils import _tail_file  # noqa: E402


FINAL_POLICIES = "p10-p10,p30-p30,p50-p50,p70-p70,p90-p90,p10-p90,p30-p70,p10-p30,p30-p50,p50-p70,p70-p90"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="DEPRECATED: run_battery_backtest.py manually, then generate_strategy_diagnostics.py on the output folder.")
    ap.add_argument("--preset", choices=["smoke", "final", "custom"], default="smoke")
    ap.add_argument("--simulation-root", required=True)
    ap.add_argument("--models", default=None)
    ap.add_argument("--model-manifest", action="append", default=[])
    ap.add_argument("--strategy", default="multi")
    ap.add_argument("--quantile-policies", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-partial-results", action="store_true")
    ap.add_argument("--include-invalid", action="store_true")
    ap.add_argument("--da-quantile-role", default="mid")
    ap.add_argument("--final-soc-mode", default="hard_min")
    ap.add_argument("--strict-simulation-validity", dest="strict_simulation_validity", action="store_true")
    ap.add_argument("--no-strict-simulation-validity", dest="strict_simulation_validity", action="store_false")
    ap.set_defaults(strict_simulation_validity=True)
    ap.add_argument("--export-afrr-bin-ev-audit", action="store_true")
    ap.add_argument("--allow-invalid-output", action="store_true")
    ap.add_argument("--figures-dir", default=None)
    ap.add_argument("--tables-dir", default=None)
    ap.add_argument("--data-dir", default=None)
    return ap


def _resolved_defaults(args: argparse.Namespace) -> tuple[str, str, int]:
    if args.preset == "smoke":
        models = args.models or "xgb"
        policies = args.quantile_policies or "p50-p50"
        workers = 1 if args.workers is None else int(args.workers)
        return models, policies, workers
    if args.preset == "final":
        models = args.models or "xgb,tft,linear"
        policies = args.quantile_policies or FINAL_POLICIES
        workers = 3 if args.workers is None else int(args.workers)
        return models, policies, workers
    if not args.models or not args.quantile_policies:
        raise ValueError("--preset custom requires --models and --quantile-policies.")
    workers = 1 if args.workers is None else int(args.workers)
    return args.models, args.quantile_policies, workers


def main() -> int:
    args = build_arg_parser().parse_args()
    models, policies, workers = _resolved_defaults(args)
    sim_root = Path(args.simulation_root)
    sim_root.mkdir(parents=True, exist_ok=True)

    sim_cmd = [
        sys.executable,
        "scripts/run_rq3_simulations.py",
        "--simulation-root", str(sim_root),
        "--models", models,
        "--strategy", str(args.strategy),
        "--quantile-policies", policies,
        "--split", str(args.split),
        "--workers", str(workers),
    ]
    for mm in args.model_manifest:
        sim_cmd += ["--model-manifest", mm]
    if args.start:
        sim_cmd += ["--start", str(args.start)]
    if args.end:
        sim_cmd += ["--end", str(args.end)]
    if args.reuse_existing:
        sim_cmd.append("--reuse-existing")
    if args.overwrite:
        sim_cmd.append("--overwrite")
    if args.allow_partial_results:
        sim_cmd.append("--allow-partial-results")
    if args.da_quantile_role:
        sim_cmd += ["--da-quantile-role", str(args.da_quantile_role)]
    if args.final_soc_mode:
        sim_cmd += ["--final-soc-mode", str(args.final_soc_mode)]
    if args.strict_simulation_validity:
        sim_cmd.append("--strict-simulation-validity")
    else:
        sim_cmd.append("--no-strict-simulation-validity")
    if args.export_afrr_bin_ev_audit:
        sim_cmd.append("--export-afrr-bin-ev-audit")
    if args.allow_invalid_output:
        sim_cmd.append("--allow-invalid-output")

    sim_log = sim_root / "pipeline_simulation.log"
    with sim_log.open("w", encoding="utf-8") as f:
        sim_cp = subprocess.run(sim_cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    pipeline_manifest = {
        "preset": args.preset,
        "simulation_command": sim_cmd,
        "simulation_exit_code": sim_cp.returncode,
        "simulation_log": str(sim_log),
        "start": args.start,
        "end": args.end,
        "simulation_root": str(sim_root),
    }

    diag_cmd = [
        sys.executable,
        "scripts/generate_strategy_diagnostics.py",
        "--simulation-root", str(sim_root),
        "--strategy", str(args.strategy),
    ]
    if args.start:
        diag_cmd += ["--start", str(args.start)]
    if args.end:
        diag_cmd += ["--end", str(args.end)]
    if args.include_invalid:
        diag_cmd.append("--include-invalid")
    if args.figures_dir:
        diag_cmd += ["--figures-dir", str(args.figures_dir)]
    if args.tables_dir:
        diag_cmd += ["--tables-dir", str(args.tables_dir)]
    if args.data_dir:
        diag_cmd += ["--data-dir", str(args.data_dir)]

    diag_exit = None
    diag_log = sim_root / "pipeline_diagnostics.log"
    pipeline_failed = sim_root / "pipeline_failed.md"
    if sim_cp.returncode == 0:
        with diag_log.open("w", encoding="utf-8") as f:
            diag_cp = subprocess.run(diag_cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        diag_exit = diag_cp.returncode
        pipeline_manifest["diagnostics_command"] = diag_cmd
        pipeline_manifest["diagnostics_exit_code"] = diag_exit
        pipeline_manifest["diagnostics_log"] = str(diag_log)
    elif args.allow_partial_results:
        run_manifest = sim_root / "run_manifest_rq3.json"
        okish = 0
        if run_manifest.exists():
            try:
                rows = json.loads(run_manifest.read_text(encoding="utf-8")).get("results", [])
                okish = sum(
                    1
                    for r in rows
                    if str(r.get("failure_class", "")) in {"completed_reportable", "reused_reportable"}
                )
            except Exception:
                okish = 0
        if okish > 0:
            with diag_log.open("w", encoding="utf-8") as f:
                diag_cp = subprocess.run(diag_cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            diag_exit = diag_cp.returncode
            pipeline_manifest["diagnostics_command"] = diag_cmd
            pipeline_manifest["diagnostics_exit_code"] = diag_exit
            pipeline_manifest["diagnostics_log"] = str(diag_log)

    if sim_cp.returncode != 0:
        pipeline_failed.write_text(
            "\n".join(
                [
                    "# RQ3 Pipeline Failure",
                    "",
                    f"- Simulation command: {' '.join(sim_cmd)}",
                    f"- Exit code: {sim_cp.returncode}",
                    f"- Simulation log: {sim_log}",
                    "- Log tail:",
                    "```text",
                    _tail_file(sim_log),
                    "```",
                    f"- run_manifest_rq3.json exists: {(sim_root / 'run_manifest_rq3.json').exists()}",
                    f"- failed_runs.md exists: {(sim_root / 'failed_runs.md').exists()}",
                    "",
                    "Next:",
                    f"Inspect {sim_log} and {sim_root / 'failed_runs.md'} if present.",
                ]
            ),
            encoding="utf-8",
        )

    (sim_root / "rq3_pipeline_manifest.json").write_text(json.dumps(pipeline_manifest, indent=2), encoding="utf-8")
    if diag_exit is not None:
        return 0 if (sim_cp.returncode == 0 and diag_exit == 0) else 1
    if sim_cp.returncode != 0:
        return 1
    return int(sim_cp.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
