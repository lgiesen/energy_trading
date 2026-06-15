#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Any

# Allow execution as `python scripts/run_rq3_simulations.py`.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from scripts.rq3_pipeline_utils import (
    build_scenario_out_dir,
    inspect_run_artifacts,
    parse_model_manifest_items,
    parse_models_csv,
    parse_qpair,
    parse_quantile_policies_csv,
    _find_nested_artifact_pair,
    run_logged_subprocess,
    write_failed_run_reports,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="DEPRECATED: run_battery_backtest.py manually for each scenario; this helper is not part of the final RQ3 workflow.")
    ap.add_argument("--simulation-root", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--model-manifest", action="append", default=[])
    ap.add_argument("--strategy", default="multi")
    ap.add_argument("--quantile-policies", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-partial-results", action="store_true")
    ap.add_argument("--allow-missing-models", action="store_true")
    ap.add_argument("--da-quantile-role", default="mid")
    ap.add_argument("--final-soc-mode", default="hard_min")
    ap.add_argument("--strict-simulation-validity", dest="strict_simulation_validity", action="store_true")
    ap.add_argument("--no-strict-simulation-validity", dest="strict_simulation_validity", action="store_false")
    ap.set_defaults(strict_simulation_validity=True)
    ap.add_argument("--export-afrr-bin-ev-audit", action="store_true")
    ap.add_argument("--allow-invalid-output", action="store_true")
    return ap


def _build_commands(args: argparse.Namespace, manifest_map: dict[str, str]) -> list[dict[str, Any]]:
    models = parse_models_csv(args.models)
    policies = parse_quantile_policies_csv(args.quantile_policies)
    simulation_root = Path(args.simulation_root)
    logs_dir = simulation_root / "logs"
    commands: list[dict[str, Any]] = []
    for model in models:
        manifest = manifest_map.get(model, "")
        if not manifest:
            continue
        for qp in policies:
            out_dir = build_scenario_out_dir(simulation_root, model, args.strategy, qp)
            has_summary = (out_dir / "backtest_summary.json").exists()
            has_hourly = (out_dir / "backtest_hourly.parquet").exists() or (out_dir / "backtest_hourly.csv").exists()
            nested_summary, nested_hourly, _ = _find_nested_artifact_pair(out_dir)
            if nested_summary is not None and nested_hourly is not None:
                has_summary = True
                has_hourly = True
            status = "planned"
            if has_summary and has_hourly:
                if args.reuse_existing:
                    status = "reused"
                elif not args.overwrite:
                    raise ValueError(f"Existing outputs found in {out_dir}. Use --reuse-existing or --overwrite.")
            low, high = parse_qpair(qp)
            qarg = f"{low}-{high}" if low and high else qp
            cmd = [
                sys.executable,
                "scripts/run_battery_backtest.py",
                "--run-manifest", manifest,
                "--model-key", model,
                "--split", str(args.split),
                "--trading-strategy", str(args.strategy),
                "--da-quantile-role", str(args.da_quantile_role),
                "--quantile-pairs", qarg,
                "--final-soc-mode", str(args.final_soc_mode),
                "--out-dir", str(out_dir),
            ]
            if args.strict_simulation_validity:
                cmd.append("--strict-simulation-validity")
            if args.export_afrr_bin_ev_audit:
                cmd.append("--export-afrr-bin-ev-audit")
            if args.allow_invalid_output:
                cmd.append("--allow-invalid-output")
            if args.overwrite:
                cmd.append("--clean-output")
            if args.start:
                cmd += ["--start", str(args.start)]
            if args.end:
                cmd += ["--end", str(args.end)]
            log_path = logs_dir / f"{model}_{args.strategy}_{qp}.log"
            commands.append(
                {
                    "model": model,
                    "strategy": args.strategy,
                    "quantile_policy": qp,
                    "command": cmd,
                    "out_dir": str(out_dir),
                    "log_path": str(log_path),
                    "status": status,
                }
            )
    return commands


def main() -> int:
    args = build_arg_parser().parse_args()
    simulation_root = Path(args.simulation_root)
    simulation_root.mkdir(parents=True, exist_ok=True)
    manifest_map = parse_model_manifest_items(args.model_manifest)
    models = parse_models_csv(args.models)
    missing = [m for m in models if not manifest_map.get(m)]
    preflight_failure: dict[str, Any] | None = None
    if missing:
        preflight_failure = {
            "model": ",".join(missing),
            "strategy": args.strategy,
            "quantile_policy": "",
            "command": [],
            "out_dir": "",
            "log_path": "",
            "exit_code": 1,
            "status": "failed_without_artifacts",
            "failure_class": "failed_without_artifacts",
            "summary_path": "",
            "hourly_path": "",
            "artifacts_exist": False,
            "summary_exists": False,
            "hourly_exists": False,
            "simulation_valid": float("nan"),
            "thesis_reportable": float("nan"),
            "invalid_reason": "",
            "exception_type": "ValueError",
            "exception_message": (
                f"Missing manifests for models: {', '.join(missing)}. "
                "Provide --model-manifest <model>=<path> for each requested model."
            ),
            "traceback_tail": "",
            "log_tail": "",
            "suspected_failure_stage": "preflight_manifest_validation",
        }
    else:
        for m, p in manifest_map.items():
            if m in models and not Path(p).exists():
                preflight_failure = {
                    "model": m,
                    "strategy": args.strategy,
                    "quantile_policy": "",
                    "command": [],
                    "out_dir": "",
                    "log_path": str(p),
                    "exit_code": 1,
                    "status": "failed_without_artifacts",
                    "failure_class": "failed_without_artifacts",
                    "summary_path": "",
                    "hourly_path": "",
                    "artifacts_exist": False,
                    "summary_exists": False,
                    "hourly_exists": False,
                    "simulation_valid": float("nan"),
                    "thesis_reportable": float("nan"),
                    "invalid_reason": "",
                    "exception_type": "FileNotFoundError",
                    "exception_message": f"Model manifest for {m} does not exist: {p}",
                    "traceback_tail": "",
                    "log_tail": "",
                    "suspected_failure_stage": "preflight_manifest_validation",
                }
                break

    if preflight_failure is not None:
        run_meta = {
            "simulation_root": str(simulation_root),
            "models": models,
            "strategy": str(args.strategy),
            "quantile_policies": parse_quantile_policies_csv(args.quantile_policies),
            "workers": int(args.workers),
            "results": [preflight_failure],
            "preflight_error": preflight_failure["exception_message"],
        }
        (simulation_root / "run_manifest_rq3.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        pd.DataFrame([preflight_failure]).to_csv(simulation_root / "run_manifest_rq3.csv", index=False)
        cpath, jpath, mpath = write_failed_run_reports(simulation_root, [preflight_failure])
        raise ValueError(
            f"{preflight_failure['exception_message']}\n"
            f"run_manifest_rq3.json={simulation_root / 'run_manifest_rq3.json'}\n"
            f"failed_runs.csv={cpath}\nfailed_runs.json={jpath}\nfailed_runs.md={mpath}"
        )

    commands = _build_commands(args, manifest_map)
    results: list[dict[str, Any]] = []
    for rec in commands:
        if rec["status"] == "reused":
            results.append(
                inspect_run_artifacts(
                    {
                        "model": rec["model"],
                        "strategy": rec["strategy"],
                        "quantile_policy": rec["quantile_policy"],
                        "cmd": rec["command"],
                        "out_dir": rec["out_dir"],
                        "log_path": rec["log_path"],
                        "status": "reused",
                        "exit_code": 0,
                    }
                )
            )

    planned = [rec for rec in commands if rec["status"] == "planned"]

    def _run_one(rec: dict[str, Any]) -> dict[str, Any]:
        exit_code = run_logged_subprocess(rec["command"], Path(rec["log_path"]))
        return inspect_run_artifacts(
            {
                "model": rec["model"],
                "strategy": rec["strategy"],
                "quantile_policy": rec["quantile_policy"],
                "cmd": rec["command"],
                "out_dir": rec["out_dir"],
                "log_path": rec["log_path"],
                "status": "completed" if exit_code == 0 else "failed",
                "exit_code": exit_code,
            }
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = [ex.submit(_run_one, rec) for rec in planned]
        for fu in concurrent.futures.as_completed(futs):
            results.append(fu.result())

    run_meta = {
        "simulation_root": str(simulation_root),
        "models": models,
        "strategy": str(args.strategy),
        "quantile_policies": parse_quantile_policies_csv(args.quantile_policies),
        "workers": int(args.workers),
        "results": results,
    }
    (simulation_root / "run_manifest_rq3.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(simulation_root / "run_manifest_rq3.csv", index=False)

    failed = [
        r for r in results
        if str(r.get("failure_class", "")).startswith(("failed", "process_", "strict_"))
    ]
    if failed:
        cpath, jpath, mpath = write_failed_run_reports(simulation_root, failed)
        if not args.allow_partial_results:
            first = failed[0]
            raise RuntimeError(
                "Simulation runs failed: "
                f"{len(failed)}.\nFirst failure:\n"
                f"  model={first.get('model')}\n"
                f"  strategy={first.get('strategy')}\n"
                f"  quantile_policy={first.get('quantile_policy')}\n"
                f"  exit_code={first.get('exit_code')}\n"
                f"  status={first.get('failure_class')}\n"
                f"  invalid_reason={first.get('invalid_reason','')}\n"
                f"  out_dir={first.get('out_dir')}\n"
                f"  log_path={first.get('log_path')}\n"
                f"  exception_type={first.get('exception_type','')}\n"
                f"  exception_message={first.get('exception_message','')}\n"
                f"  failed_runs_md={mpath}\n"
                f"  failed_runs_csv={cpath}\n"
                f"  failed_runs_json={jpath}"
            )
        okish = [r for r in results if str(r.get("failure_class", "")).startswith(("completed", "reused"))]
        if not okish:
            raise RuntimeError("No successful or reusable simulation scenarios available for diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
