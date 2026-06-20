#!/usr/bin/env python3
"""Verify organized RQ1 output structure and manifest consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUBSECTION_DIRS = [
    "4_1_1_full_unweighted",
    "4_1_2_calibration_uncertainty",
    "4_1_3_per_lead",
    "4_1_4_gate_specific",
    "4_1_5_tail_spike",
]

TIERS = [
    "result_section/figures",
    "result_section/tables",
    "appendix/figures",
    "appendix/tables",
    "backup/csv",
    "backup/diagnostics",
    "backup/warnings",
]


def _check_latex_table(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text.startswith(r"\begin{table}"):
        errors.append(f"{path} does not start with a LaTeX table environment.")
    if not text.endswith(r"\end{table}"):
        errors.append(f"{path} does not end with a LaTeX table environment.")
    for token in [r"\toprule", r"\midrule", r"\bottomrule"]:
        if token not in text:
            errors.append(f"{path} is missing {token}.")
    return errors


def verify(rq1_root: Path) -> list[str]:
    errors: list[str] = []
    for subdir in SUBSECTION_DIRS:
        for tier in TIERS:
            path = rq1_root / subdir / tier
            if not path.is_dir():
                errors.append(f"Missing directory: {path}")

    manifest_path = rq1_root / "rq1_output_manifest.json"
    if not manifest_path.exists():
        errors.append(f"Missing manifest: {manifest_path}")
        return errors

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs", [])
    if not isinstance(outputs, list):
        errors.append("Manifest field 'outputs' is not a list.")
        return errors
    missing_outputs = manifest.get("missing_outputs", [])
    if isinstance(missing_outputs, list):
        for entry in missing_outputs:
            if bool(entry.get("required", False)):
                errors.append(f"Required output is missing: {entry.get('path')}")

    for idx, entry in enumerate(outputs):
        for key in ["subsection", "tier", "artifact_type", "path", "metric_family", "thesis_use", "brief_description"]:
            if key not in entry:
                errors.append(f"Manifest output {idx} is missing {key}.")
        path = Path(str(entry.get("path", "")))
        if not path.exists():
            errors.append(f"Manifest path does not exist: {path}")
        if entry.get("artifact_type") == "latex_table" and path.exists():
            errors.extend(_check_latex_table(path))

        if str(entry.get("tier")) == "result_section" and entry.get("artifact_type") == "latex_table" and path.exists():
            text = path.read_text(encoding="utf-8")
            appendix_only = ["RMSE p50", "Bias p50", "p10-p90 width", "Quantile crossing"]
            if any(token in text for token in appendix_only):
                errors.append(f"Result-section table contains appendix-only metric text: {path}")

        if "linear" in str(entry.get("path")).lower() and "RLQR" not in str(entry.get("brief_description", "")):
            errors.append(f"Manifest path references linear without RLQR description: {entry.get('path')}")

    return errors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify organized RQ1 output structure.")
    p.add_argument("--rq1-root", default="artifacts/final_benchmark/rq1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    errors = verify(Path(args.rq1_root))
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print(f"[OK] RQ1 output structure verified: {args.rq1_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
