#!/usr/bin/env python3
"""Verify the generated RQ2 thesis output structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "result_section/tables/1_net_profit_by_model_and_quantile.tex",
    "result_section/figures/2_quantile_sweep_net_profit_by_model.png",
    "result_section/latex_figures/2_quantile_sweep_net_profit_by_model.tex",
    "result_section/figures/3_revenue_cost_components_best_quantile.png",
    "result_section/latex_figures/3_revenue_cost_components_best_quantile.tex",
    "result_section/figures/4_cumulative_net_profit_model_comparison_test_period.png",
    "result_section/latex_figures/4_cumulative_net_profit_model_comparison_test_period.tex",
    "result_section/figures/5_pinball_loss_vs_net_profit_total_normalized.png",
    "result_section/latex_figures/5_pinball_loss_vs_net_profit_total_normalized.tex",
    "result_section/figures/1_profit_heatmap.png",
    "result_section/latex_figures/1_profit_heatmap.tex",
    "appendix/figures/5_pinball_loss_vs_net_profit_da_price.png",
    "appendix/latex_figures/5_pinball_loss_vs_net_profit_da_price.tex",
    "appendix/figures/5_pinball_loss_vs_net_profit_afrr_capacity_price_pos.png",
    "appendix/latex_figures/5_pinball_loss_vs_net_profit_afrr_capacity_price_pos.tex",
    "appendix/figures/5_pinball_loss_vs_net_profit_afrr_capacity_price_neg.png",
    "appendix/latex_figures/5_pinball_loss_vs_net_profit_afrr_capacity_price_neg.tex",
    "appendix/figures/5_pinball_loss_vs_net_profit_afrr_activation_price_pos.png",
    "appendix/latex_figures/5_pinball_loss_vs_net_profit_afrr_activation_price_pos.tex",
    "appendix/figures/5_pinball_loss_vs_net_profit_afrr_activation_price_neg.png",
    "appendix/latex_figures/5_pinball_loss_vs_net_profit_afrr_activation_price_neg.tex",
    "appendix/figures/5_pinball_loss_vs_net_profit_afrr_activation_rate_pos.png",
    "appendix/latex_figures/5_pinball_loss_vs_net_profit_afrr_activation_rate_pos.tex",
    "appendix/figures/5_pinball_loss_vs_net_profit_afrr_activation_rate_neg.png",
    "appendix/latex_figures/5_pinball_loss_vs_net_profit_afrr_activation_rate_neg.tex",
    "appendix/figures/5_pinball_loss_vs_net_profit_total.png",
    "appendix/latex_figures/5_pinball_loss_vs_net_profit_total.tex",
    "appendix/tables/rq2_profit_and_validity_detailed.tex",
    "backup/csv/rq2_scenario_summary_long.csv",
    "backup/csv/1_net_profit_by_model_and_quantile.csv",
    "backup/csv/1_profit_heatmap.csv",
    "backup/csv/2_quantile_sweep_net_profit_by_model.csv",
    "backup/csv/3_revenue_cost_components_best_quantile.csv",
    "backup/csv/4_cumulative_net_profit_model_comparison_test_period.csv",
    "backup/csv/5_pinball_loss_vs_net_profit_scatter_data.csv",
    "backup/csv/5_pinball_loss_vs_net_profit_total_scatter_data.csv",
    "backup/csv/5_pinball_loss_vs_net_profit_total_normalized.csv",
    "backup/csv/rq2_benchmark_values.csv",
    "backup/diagnostics/rq2_input_file_inventory.csv",
    "backup/diagnostics/rq2_validity_diagnostics.csv",
    "backup/warnings/rq2_warnings.csv",
    "rq2_output_manifest.json",
]

INVALIDITY_SEVERITY_FILES = [
    "appendix/tables/simulation_invalidity_severity_summary.tex",
    "appendix/tables/simulation_invalidity_context_table.tex",
    "appendix/tables/simulation_invalidity_reason_matrix.tex",
    "appendix/figures/simulation_invalidity_reason_severity_heatmap.png",
    "appendix/latex_figures/simulation_invalidity_reason_severity_heatmap.tex",
    "backup/diagnostics/simulation_invalidity_severity_summary.csv",
    "backup/diagnostics/simulation_invalidity_severity_by_hour.csv",
    "backup/diagnostics/simulation_invalidity_source_inventory.csv",
    "backup/diagnostics/simulation_invalidity_metric_sources.csv",
    "backup/diagnostics/simulation_invalidity_context_table.csv",
    "backup/diagnostics/simulation_invalidity_reason_matrix.csv",
    "backup/diagnostics/simulation_invalidity_reason_severity_heatmap.csv",
    "backup/diagnostics/simulation_invalidity_limitation_summary.txt",
    "backup/warnings/simulation_invalidity_severity_warnings.csv",
]


def verify(out_root: Path, *, require_invalidity_severity: bool = False) -> list[str]:
    errors: list[str] = []
    required = list(REQUIRED_FILES)
    if require_invalidity_severity:
        required.extend(INVALIDITY_SEVERITY_FILES)
    for rel in required:
        if not (out_root / rel).exists():
            errors.append(f"Missing required file: {out_root / rel}")

    manifest_path = out_root / "rq2_output_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = manifest.get("outputs", [])
        if not isinstance(outputs, list):
            errors.append("Manifest field 'outputs' is not a list.")
        else:
            for entry in outputs:
                rel = entry.get("path", "")
                if rel and not (out_root / rel).exists():
                    errors.append(f"Manifest points to missing file: {rel}")
        if "simulation_days" not in manifest or "annualization_factor" not in manifest:
            errors.append("Manifest is missing simulation_days or annualization_factor.")

    table_path = out_root / "result_section/tables/1_net_profit_by_model_and_quantile.tex"
    if table_path.exists():
        text = table_path.read_text(encoding="utf-8")
        for token in [r"\toprule", r"\midrule", r"\bottomrule", r"\label{tab:1_net_profit_by_model_and_quantile}"]:
            if token not in text:
                errors.append(f"Primary table is missing {token}.")

    diagnostics_path = out_root / "backup/diagnostics/rq2_validity_diagnostics.csv"
    result_csv_path = out_root / "backup/csv/1_net_profit_by_model_and_quantile.csv"
    warnings_path = out_root / "backup/warnings/rq2_warnings.csv"
    if diagnostics_path.exists() and result_csv_path.exists():
        diagnostics = pd.read_csv(diagnostics_path)
        if "included_in_result_section" not in diagnostics.columns:
            errors.append("Validity diagnostics is missing included_in_result_section.")
        if not diagnostics.empty:
            bad = diagnostics.loc[
                (pd.to_numeric(diagnostics.get("simulation_valid"), errors="coerce").fillna(0.0) < 0.5)
                & (diagnostics.get("included_in_result_section").astype(str).str.lower().isin(["true", "1"]))
            ]
            if not bad.empty:
                errors.append("Invalid scenarios are marked as included in result-section diagnostics.")
    if warnings_path.exists():
        warnings = pd.read_csv(warnings_path)
        joined = " ".join(warnings.astype(str).stack().tolist()).lower() if not warnings.empty else ""
        if "naive/rhpf" not in joined and "benchmark" not in joined:
            errors.append("Warnings do not document Naive/RHPF benchmark handling.")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify RQ2 thesis output structure.")
    ap.add_argument("--out-root", default="artifacts/benchmark/rq2_simulation_benchmark")
    ap.add_argument("--require-invalidity-severity", action="store_true", help="Also require invalidity severity limitation outputs.")
    args = ap.parse_args()
    errors = verify(Path(args.out_root), require_invalidity_severity=bool(args.require_invalidity_severity))
    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        return 1
    print(f"[OK] RQ2 output structure verified: {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
