#!/usr/bin/env python3
"""Run implemented RQ1 analysis outputs.

The wrapper intentionally delegates to subsection scripts. This keeps the
individual thesis outputs auditable while still providing one reproducible
entry point for RQ1 artifact generation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = "test"
DEFAULT_SPLITS = "test"
DEFAULT_MODELS = "tft,xgboost,linear"
DEFAULT_BENCHMARK_DIR = "artifacts/benchmark/rq1_ml_model_benchmark"
DEFAULT_EVAL_ORIGIN_START_UTC = "2025-01-13T23:00:00Z"
DEFAULT_EVAL_ORIGIN_END_UTC = "2026-02-26T21:00:00Z"
DEFAULT_EXPORT_DIR = (
    "/Users/leori/Desktop/ uni/3 Master IS/25 MA/"
    "MA___Elevated_Energy_Trading__Forecasting_Expected_Profit_From_Participating_in_the_Day_Ahead_and_Balancing_Markets/"
    "figures/4-results/rq1_ml_model_benchmark"
)
JOINED_TARGETS = [
    "pred_da_price",
    "pred_afrr_capacity_price_pos",
    "pred_afrr_capacity_price_neg",
    "pred_afrr_activation_price_pos",
    "pred_afrr_activation_price_neg",
    "pred_afrr_activation_rate_pos",
    "pred_afrr_activation_rate_neg",
]
MODEL_JOINED_KEYS = {
    "tft": "tft",
    "xgb": "xgb",
    "xgboost": "xgb",
    "linear": "linear",
    "rlqr": "linear",
}


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


def _export_output_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Cannot export missing RQ1 output directory: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.mkdir(parents=True, exist_ok=True)
    excluded_top_level = {
        "_raw_outputs",
        "_rq1_benchmark_inputs",
        "diagnostics",
        "logs",
        "metrics",
    }
    exported_names: set[str] = set()
    for child in sorted(source.iterdir()):
        if child.name in excluded_top_level or child.name.startswith("."):
            continue
        if not (child.is_dir() or child.is_file()):
            continue
        target = destination / child.name
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)
        exported_names.add(child.name)
    if not exported_names:
        raise FileNotFoundError(f"No thesis-facing RQ1 output folders were available for export from {source}")


def _latex_file_has_label(path: Path, label: str) -> bool:
    if not path.exists():
        return False
    return rf"\label{{{label}}}" in path.read_text(encoding="utf-8", errors="ignore")


def _prune_extensions(root: Path, suffixes: set[str]) -> dict[str, int]:
    if not suffixes or not root.exists():
        return {}
    counts = {suffix: 0 for suffix in sorted(suffixes)}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in suffixes:
            continue
        path.unlink()
        counts[suffix] += 1
    return counts


def _prune_subsection_extensions(root: Path, suffixes: set[str]) -> dict[str, int]:
    if not suffixes or not root.exists():
        return {}
    counts = {suffix: 0 for suffix in sorted(suffixes)}
    subsection_dirs = [
        root / "result_section",
        root / "4_1_1_full_unweighted",
        root / "4_1_2_calibration_uncertainty",
        root / "4_1_3_per_lead",
        root / "4_1_4_gate_specific",
        root / "4_1_5_tail_spike",
        root / "4_1_6_example_weeks",
    ]
    for subsection in subsection_dirs:
        if not subsection.exists():
            continue
        for path in sorted(subsection.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in suffixes:
                continue
            path.unlink()
            counts[suffix] += 1
    return counts


def _remove_pruned_manifest_entries(root: Path, suffixes: set[str]) -> int:
    if not suffixes:
        return 0
    manifest_path = root / "rq1_output_manifest.json"
    if not manifest_path.exists():
        return 0
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        return 0
    kept: list[dict[str, Any]] = []
    removed = 0
    for entry in outputs:
        if not isinstance(entry, dict):
            kept.append(entry)
            continue
        path = Path(str(entry.get("path", "")))
        should_prune = path.suffix.lower() in suffixes
        if should_prune and not path.exists():
            removed += 1
            continue
        kept.append(entry)
    if removed:
        manifest["outputs"] = kept
        manifest["pruned_manifest_entries"] = {
            **dict(manifest.get("pruned_manifest_entries", {})),
            **{suffix: sum(1 for entry in outputs if isinstance(entry, dict) and Path(str(entry.get("path", ""))).suffix.lower() == suffix and not Path(str(entry.get("path", ""))).exists()) for suffix in sorted(suffixes)},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return removed


POINT_ERROR_LEGACY_STEMS = [
    "mae_p50_by_target_model",
    "bias_p50_by_target_model",
    "mae_bias_p50_by_target_model",
    "mae_p50_price_targets_by_model",
    "mae_p50_activation_rate_targets_by_model",
    "bias_p50_price_targets_by_model",
    "bias_p50_activation_rate_targets_by_model",
]


def _remove_legacy_point_error_outputs(out_dir: Path) -> int:
    removed = 0
    subdirs = {
        "result_section/csv": [".csv"],
        "result_section/figures": [".png", ".pdf", ".svg"],
        "result_section/latex_figures": [".tex"],
        "result_section/latex_tables": [".tex"],
    }
    for subdir, suffixes in subdirs.items():
        for stem in POINT_ERROR_LEGACY_STEMS:
            for suffix in suffixes:
                path = out_dir / subdir / f"{stem}{suffix}"
                if path.exists():
                    path.unlink()
                    removed += 1
    return removed


def _canonical_model_keys(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in _split_csv(raw):
        key = item.lower()
        if key not in MODEL_JOINED_KEYS:
            raise ValueError(f"Unknown model key {item!r}. Supported: {', '.join(sorted(MODEL_JOINED_KEYS))}")
        canonical = MODEL_JOINED_KEYS[key]
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    if not out:
        raise ValueError("At least one model is required.")
    return out


def _validate_joined_predictions(*, benchmark_dir: Path, splits: list[str], models: list[str]) -> None:
    joined_dir = benchmark_dir / "diagnostics" / "joined_predictions"
    missing = [
        joined_dir / f"{model}__{split}__{target}.parquet"
        for split in splits
        for model in models
        for target in JOINED_TARGETS
        if not (joined_dir / f"{model}__{split}__{target}.parquet").exists()
    ]
    if not missing:
        return
    existing = sorted(joined_dir.glob("*.parquet")) if joined_dir.exists() else []
    preview = "\n".join(f"  - {p}" for p in missing[:12])
    if len(missing) > 12:
        preview += f"\n  - ... {len(missing) - 12} more"
    raise FileNotFoundError(
        "RQ1 requires a complete joined-prediction benchmark before subsection scripts run.\n"
        f"Missing {len(missing)} expected parquet file(s) under {joined_dir}:\n"
        f"{preview}\n"
        f"Currently found {len(existing)} joined parquet file(s). Rebuild the benchmark with:\n"
        "./.venv/bin/python scripts/run_forecast_benchmark.py --no-make-figures"
    )


def _prepare_benchmark_snapshot(*, benchmark_dir: Path, snapshot_dir: Path, skip_json: bool = False) -> Path:
    joined_src = benchmark_dir / "diagnostics" / "joined_predictions"
    if not joined_src.is_dir():
        raise FileNotFoundError(f"Missing joined predictions directory: {joined_src}")
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    joined_dst = snapshot_dir / "diagnostics" / "joined_predictions"
    joined_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(joined_src, joined_dst)
    names = ["benchmark_config_resolved.yaml"] if skip_json else ["input_manifest.json", "benchmark_config_resolved.yaml", "benchmark_manifest.json"]
    for name in names:
        src = benchmark_dir / name
        if src.exists():
            dst = snapshot_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return snapshot_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run implemented RQ1 thesis analyses.")
    p.add_argument("--benchmark-root", default="artifacts")
    p.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR)
    p.add_argument("--out-dir", default="artifacts/benchmark/rq1_ml_model_benchmark")
    p.add_argument("--split", default=DEFAULT_SPLIT, help="Main thesis/reporting split. Defaults to test.")
    p.add_argument(
        "--splits",
        default=DEFAULT_SPLITS,
        help=(
            "Comma-separated splits to load from the forecast benchmark. Defaults to test only for final RQ1 reporting."
        ),
    )
    p.add_argument("--models", default=DEFAULT_MODELS, help="Models to compare. Defaults to all RQ1 models: TFT, XGB and RLQR.")
    p.add_argument("--eval-origin-start", default=DEFAULT_EVAL_ORIGIN_START_UTC, help="Inclusive forecast-origin lower bound for final RQ1 evaluation. Empty string disables the lower bound.")
    p.add_argument("--eval-origin-end", default=DEFAULT_EVAL_ORIGIN_END_UTC, help="Inclusive forecast-origin upper bound for final RQ1 evaluation. Empty string disables the upper bound.")
    p.add_argument("--targets", default="", help="Optional comma-separated targets for example-week plots.")
    p.add_argument("--lead", type=float, default=24.0)
    p.add_argument("--quantile", default="p50")
    p.add_argument("--selection-mode", choices=["algorithmic", "legacy"], default="algorithmic")
    p.add_argument("--date", default=None, help="Optional custom week start for example-week plots.")
    p.add_argument("--typical-start", default="2025-03-30T22:00:00Z")
    p.add_argument("--high-volatility-start", default="2025-10-05T22:00:00Z")
    p.add_argument("--window-hours", type=int, default=168)
    p.add_argument("--skip-full-metrics", action="store_true")
    p.add_argument("--skip-point-error-heatmaps", action="store_true", help="Skip MAE p50 / MBE p50 target-model heatmap generation.")
    p.add_argument("--skip-calibration", action="store_true")
    p.add_argument("--skip-per-lead", action="store_true")
    p.add_argument("--skip-gate-buckets", action="store_true")
    p.add_argument("--skip-tail-spike", action="store_true")
    p.add_argument("--skip-example-weeks", action="store_true")
    p.add_argument(
        "--skip-raw-generation",
        action="store_true",
        help=(
            "Reuse existing _raw_outputs and skip all subsection builder scripts. "
            "This is useful for fast re-organization, LaTeX regeneration, pruning and export."
        ),
    )
    p.add_argument(
        "--organize-only",
        dest="skip_raw_generation",
        action="store_true",
        help="Alias for --skip-raw-generation.",
    )
    p.add_argument("--skip-organize", action="store_true")
    p.add_argument("--skip-pdf", action=argparse.BooleanOptionalAction, default=True, help="Do not keep PDF outputs. Defaults to true; use --no-skip-pdf or --include-pdf to keep PDF outputs.")
    p.add_argument("--include-pdf", dest="skip_pdf", action="store_false", help="Keep PDF outputs.")
    p.add_argument("--skip-svg", action=argparse.BooleanOptionalAction, default=True, help="Do not keep SVG outputs. Defaults to true; use --no-skip-svg or --include-svg to keep SVG outputs.")
    p.add_argument("--include-svg", dest="skip_svg", action="store_false", help="Keep SVG outputs.")
    p.add_argument("--skip-png", action="store_true", help="Prune PNG files from the final organized output tree after LaTeX/table generation.")
    p.add_argument("--skip-csv", action="store_true", help="Prune CSV files from the final organized output tree after derived tables and LaTeX figures are generated.")
    p.add_argument("--skip-json", action="store_true", help="Prune JSON files from the final organized output tree after generation.")
    p.add_argument(
        "--latex-only",
        action="store_true",
        help=(
            "Fast mode for frequent thesis updates: reuse existing _raw_outputs, regenerate organized LaTeX/table "
            "outputs, and prune PNG/PDF/SVG/CSV/JSON from the final exported tree."
        ),
    )
    p.add_argument("--export-dir", default=DEFAULT_EXPORT_DIR, help="Destination thesis folder for the organized rq1_ml_model_benchmark output.")
    p.add_argument("--skip-export", action="store_true", help="Do not copy rq1_ml_model_benchmark to the thesis figures folder.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.latex_only:
        args.skip_raw_generation = True
        args.skip_pdf = True
        args.skip_svg = True
        args.skip_png = True
        args.skip_csv = True
        args.skip_json = True
    if args.skip_raw_generation:
        args.skip_full_metrics = True
        args.skip_calibration = True
        args.skip_per_lead = True
        args.skip_gate_buckets = True
        args.skip_tail_spike = True
        args.skip_example_weeks = True
    out_dir = Path(args.out_dir)
    work_dir = out_dir / "_raw_outputs"
    full_dir = out_dir / "4_1_1_full_unweighted"
    calibration_dir = out_dir / "4_1_2_calibration_uncertainty"
    per_lead_dir = out_dir / "4_1_3_per_lead"
    gate_dir = out_dir / "4_1_4_gate_specific"
    tail_dir = out_dir / "4_1_5_tail_spike"
    example_dir = out_dir / "4_1_6_example_weeks"
    log_dir = out_dir / "logs" / "rq1_wrapper"
    steps: list[dict[str, Any]] = []
    py = sys.executable
    source_benchmark_dir = Path(args.benchmark_dir)
    if not source_benchmark_dir.is_absolute():
        source_benchmark_dir = ROOT / source_benchmark_dir
    benchmark_dir = source_benchmark_dir
    needs_joined = not all(
        [
            args.skip_full_metrics,
            args.skip_calibration,
            args.skip_per_lead,
            args.skip_gate_buckets,
            args.skip_tail_spike,
            args.skip_example_weeks,
        ]
    )
    if needs_joined:
        _validate_joined_predictions(
            benchmark_dir=source_benchmark_dir,
            splits=_split_csv(args.splits),
            models=_canonical_model_keys(args.models),
        )
        benchmark_dir = _prepare_benchmark_snapshot(
            benchmark_dir=source_benchmark_dir,
            snapshot_dir=out_dir / "_rq1_benchmark_inputs",
            skip_json=bool(args.skip_json),
        )

    benchmark_args: list[str] = ["--benchmark-root", args.benchmark_root]
    benchmark_args.extend(["--benchmark-dir", str(benchmark_dir)])
    eval_window_args = [
        "--eval-origin-start",
        str(args.eval_origin_start),
        "--eval-origin-end",
        str(args.eval_origin_end),
    ]

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
            *eval_window_args,
            "--point-error-out-root",
            str(out_dir),
        ]
        if args.skip_pdf:
            cmd.append("--skip-pdf")
        if args.skip_csv:
            cmd.append("--skip-csv")
        if args.skip_json:
            cmd.append("--skip-json")
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
            *eval_window_args,
        ]
        if args.skip_csv:
            cmd.append("--skip-csv")
        if args.skip_json:
            cmd.append("--skip-json")
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
            *eval_window_args,
        ]
        if args.skip_pdf:
            cmd.append("--skip-pdf")
        if args.skip_csv:
            cmd.append("--skip-csv")
        if args.skip_json:
            cmd.append("--skip-json")
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
            *eval_window_args,
        ]
        if args.skip_csv:
            cmd.append("--skip-csv")
        if args.skip_json:
            cmd.append("--skip-json")
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
            *eval_window_args,
        ]
        if args.skip_csv:
            cmd.append("--skip-csv")
        if args.skip_json:
            cmd.append("--skip-json")
        steps.append(_run_step("rq1_4_1_5_tail_spike", cmd, log_dir=log_dir))

    if not args.skip_example_weeks:
        cmd = [
            py,
            "scripts/build_rq1_example_weeks.py",
            *benchmark_args,
            "--out-dir",
            str(work_dir / "4_1_6_example_weeks"),
            "--split",
            str(args.split),
            "--models",
            str(args.models),
            "--lead",
            str(args.lead),
            "--quantile",
            str(args.quantile),
            "--selection-mode",
            str(args.selection_mode),
            "--typical-start",
            str(args.typical_start),
            "--high-volatility-start",
            str(args.high_volatility_start),
            "--window-hours",
            str(args.window_hours),
            *eval_window_args,
        ]
        targets = _split_csv(args.targets)
        if targets:
            cmd.extend(["--targets", ",".join(targets)])
        if args.date:
            cmd.extend(["--date", str(args.date)])
        if args.skip_csv:
            cmd.append("--skip-csv")
        if args.skip_json:
            cmd.append("--skip-json")
        steps.append(_run_step("rq1_4_1_6_example_weeks", cmd, log_dir=log_dir))

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
        if args.skip_csv:
            cmd.append("--skip-csv")
        if args.skip_json:
            cmd.append("--skip-json")
        steps.append(_run_step("rq1_output_organization", cmd, log_dir=log_dir))

    if not args.skip_point_error_heatmaps:
        removed_legacy_point_errors = _remove_legacy_point_error_outputs(out_dir)
        if removed_legacy_point_errors:
            print(f"[OK] Removed deprecated RQ1 point-error outputs: {removed_legacy_point_errors}")
        point_error_csv_dir = work_dir / "4_1_1_full_unweighted" / "csv"
        point_error_candidates = [
            point_error_csv_dir / f"rq1_4_1_1_forecast_metrics_full_detailed_{args.split}.csv",
            point_error_csv_dir / "rq1_4_1_1_forecast_metrics_full_long.csv",
            ROOT
            / "artifacts"
            / "rq1_ml_model_benchmark"
            / "_raw_outputs"
            / "4_1_1_full_unweighted"
            / "csv"
            / f"rq1_4_1_1_forecast_metrics_full_detailed_{args.split}.csv",
            ROOT
            / "artifacts"
            / "rq1_ml_model_benchmark"
            / "_raw_outputs"
            / "4_1_1_full_unweighted"
            / "csv"
            / "rq1_4_1_1_forecast_metrics_full_long.csv",
        ]
        point_error_input = next((p for p in point_error_candidates if p.exists()), point_error_candidates[0])
        if not point_error_input.exists():
            existing_relative_mae_latex = out_dir / "result_section" / "latex_figures" / "mae_p50_relative_to_rlqr_by_target_model.tex"
            existing_mbe_table = out_dir / "result_section" / "latex_tables" / "mbe_p50_raw_by_target_model.tex"
            existing_point_error_outputs = [existing_relative_mae_latex, existing_mbe_table]
            existing_outputs_valid = all(path.exists() for path in existing_point_error_outputs) and _latex_file_has_label(
                existing_mbe_table, "tab:mbe_p50_raw_by_target_model"
            ) and _latex_file_has_label(existing_relative_mae_latex, "fig:mae_p50_relative_to_rlqr_by_target_model")
            if args.skip_full_metrics:
                if existing_outputs_valid:
                    print(
                        "[OK] Reusing existing RQ1 relative MAE p50 heatmap and raw MBE p50 table; "
                        "the internal 4.1.1 metric CSV has already been pruned."
                    )
                else:
                    raise FileNotFoundError(
                        "Cannot reuse RQ1 relative MAE p50 heatmap and raw MBE p50 table because the detailed "
                        "4.1.1 metric CSV and long metric CSV are missing, or existing LaTeX outputs are stale. "
                        f"Checked: {', '.join(str(p) for p in point_error_candidates)}. "
                        "Run RQ1 once without --skip-raw-generation to rebuild that intermediate."
                    )
            else:
                raise FileNotFoundError(f"Missing detailed 4.1.1 metric CSV for point-error heatmaps: {point_error_input}")
        else:
            cmd = [
                py,
                "scripts/generate_rq1_point_error_heatmaps.py",
                "--input",
                str(point_error_input),
                "--out-root",
                str(out_dir),
            ]
            if args.skip_csv:
                cmd.append("--skip-csv")
            if args.skip_png:
                cmd.append("--skip-png")
            if args.skip_pdf:
                cmd.append("--skip-pdf")
            steps.append(_run_step("rq1_4_1_1_point_error_heatmaps", cmd, log_dir=log_dir))

    manifest = {
        "description": "RQ1 wrapper manifest for implemented thesis analysis scripts.",
        "out_dir": str(out_dir),
        "source_benchmark_dir": str(source_benchmark_dir),
        "benchmark_input_snapshot_dir": str(benchmark_dir),
        "split": args.split,
        "models": args.models,
        "eval_origin_start_utc": str(args.eval_origin_start),
        "eval_origin_end_utc": str(args.eval_origin_end),
        "subsections": [
            {
                "section": "4.1.1",
                "name": "Full unweighted forecast metrics",
                "status": "skipped" if args.skip_full_metrics else "implemented",
                "output_dir": str(full_dir),
                "additional_outputs": {
                    "point_error_heatmaps": "skipped" if args.skip_point_error_heatmaps else "implemented",
                    "relative_mae_heatmap": str(out_dir / "result_section" / "figures" / "mae_p50_relative_to_rlqr_by_target_model.png"),
                    "relative_mae_latex": str(out_dir / "result_section" / "latex_figures" / "mae_p50_relative_to_rlqr_by_target_model.tex"),
                    "mbe_table": str(out_dir / "result_section" / "latex_tables" / "mbe_p50_raw_by_target_model.tex"),
                },
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
                "result_section": ["figures", "latex_figures", "tables"],
                "appendix": ["figures", "latex_figures", "tables"],
                "backup": ["csv", "diagnostics", "warnings"],
            },
        },
        "expected_example_week_outputs_dir": str(example_dir),
        "organized_manifest": str(out_dir / "rq1_output_manifest.json"),
        "export_dir": str(args.export_dir),
        "export_skipped": bool(args.skip_export),
        "prune_requested": {
            "pdf": bool(args.skip_pdf),
            "svg": bool(args.skip_svg),
            "png": bool(args.skip_png),
            "csv": bool(args.skip_csv),
            "json": bool(args.skip_json),
        },
        "raw_generation_skipped": bool(args.skip_raw_generation),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for subsection in manifest["subsections"]:
        Path(str(subsection["output_dir"])).mkdir(parents=True, exist_ok=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "diagnostics" / "rq1_wrapper_manifest.json"
    if not args.skip_json:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    prune_suffixes: set[str] = set()
    if args.skip_pdf:
        prune_suffixes.add(".pdf")
    if args.skip_svg:
        prune_suffixes.add(".svg")
    if args.skip_png:
        prune_suffixes.add(".png")
    if args.skip_csv:
        prune_suffixes.add(".csv")
    if args.skip_json:
        prune_suffixes.add(".json")
    # Apply format pruning to the complete RQ1 output tree, not only the
    # organized thesis-facing subsection folders. Several subsection builders
    # still need intermediate CSV/JSON files during generation, but when the
    # user requests skipped formats those files must not remain in the final
    # benchmark directory or be exported to the thesis repository.
    prune_counts = _prune_extensions(out_dir, prune_suffixes)
    if prune_counts:
        manifest_removed = _remove_pruned_manifest_entries(out_dir, prune_suffixes)
        print("[OK] Pruned final RQ1 output files: " + ", ".join(f"{k}={v}" for k, v in sorted(prune_counts.items())))
        if manifest_removed:
            print(f"[OK] Removed pruned entries from RQ1 output manifest: {manifest_removed}")
    if not args.skip_export:
        _export_output_tree(out_dir, Path(args.export_dir))
        print(f"[OK] Exported RQ1 benchmark folder: {args.export_dir}")
    if args.skip_json:
        print(f"[OK] RQ1 wrapper complete; JSON manifests were pruned from {out_dir}")
    else:
        print(f"[OK] RQ1 wrapper complete: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
