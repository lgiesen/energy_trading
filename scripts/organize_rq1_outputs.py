#!/usr/bin/env python3
"""Organize final RQ1 benchmark outputs into thesis-facing tiers.

This script does not compute forecast metrics. It routes existing RQ1 outputs
into result_section, appendix and backup folders, and writes a manifest for the
organized artifacts. Missing files are recorded explicitly.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from energy_trading.evaluation.style import THESIS_PALETTE, thesis_titlecase
from energy_trading.evaluation.rq1_target_order import MODEL_LABELS, model_sort_key, ordered_model_labels, ordered_unique, sort_model_frame, sort_target_frame, target_sort_key

SUBSECTIONS = {
    "4.1.1": "4_1_1_full_unweighted",
    "4.1.2": "4_1_2_calibration_uncertainty",
    "4.1.3": "4_1_3_per_lead",
    "4.1.4": "4_1_4_gate_specific",
    "4.1.5": "4_1_5_tail_spike",
    "4.1.6": "4_1_6_example_weeks",
}

LEGACY_SUBSECTIONS = {
    "4.1.1": ["4_1_1_full_unweighted_metrics", "4_1_1_full_unweighted", "rq1/4_1_1_full_unweighted_metrics", "rq1/4_1_1_full_unweighted"],
    "4.1.2": ["4_1_2_calibration_uncertainty", "rq1/4_1_2_calibration_uncertainty"],
    "4.1.3": ["4_1_3_per_lead_hour", "4_1_3_per_lead", "rq1/4_1_3_per_lead_hour", "rq1/4_1_3_per_lead"],
    "4.1.4": ["4_1_4_gate_actionable", "4_1_4_gate_specific", "rq1/4_1_4_gate_actionable", "rq1/4_1_4_gate_specific"],
    "4.1.5": ["4_1_5_tail_spike", "rq1/4_1_5_tail_spike"],
    "4.1.6": ["4_1_6_example_weeks", "rq1/4_1_6_example_weeks"],
}

CANONICAL_DIRS = set(SUBSECTIONS.values())


@dataclass(frozen=True)
class Route:
    subsection: str
    tier: str
    artifact_type: str
    dest_rel: str
    sources: tuple[str, ...]
    metric_family: str
    thesis_use: str
    brief_description: str
    required: bool = True


def _route(
    subsection: str,
    tier: str,
    artifact_type: str,
    dest_rel: str,
    sources: list[str],
    metric_family: str,
    thesis_use: str,
    brief_description: str,
    *,
    required: bool = True,
) -> Route:
    return Route(
        subsection=subsection,
        tier=tier,
        artifact_type=artifact_type,
        dest_rel=dest_rel,
        sources=tuple(sources),
        metric_family=metric_family,
        thesis_use=thesis_use,
        brief_description=brief_description,
        required=required,
    )


def _sub_sources(subsection: str, rel: str) -> list[str]:
    return [f"{name}/{rel}" for name in LEGACY_SUBSECTIONS[subsection]]


def build_routes(split: str) -> list[Route]:
    routes = [
        _route("4.1.1", "result_section", "figure", f"figures/forecast_metrics_full_relative_pinball_{split}.png", _sub_sources("4.1.1", f"figures/rq1_4_1_1_forecast_metrics_full_relative_pinball_{split}.png"), "mean_pinball_loss", "main thesis figure", "Relative mean pinball loss against RLQR."),
        _route("4.1.1", "result_section", "figure", "figures/computational_cost_365d.png", _sub_sources("4.1.1", "figures/rq1_4_1_1_computational_cost_365d.png"), "computational_cost", "main thesis figure", "Computational Cost of Each Model Scaled for 365 Days of Training Data."),
        _route("4.1.1", "appendix", "figure", f"figures/forecast_metrics_full_relative_mae_p50_{split}.png", _sub_sources("4.1.1", f"figures/rq1_4_1_1_forecast_metrics_full_relative_mae_p50_{split}.png"), "mae_p50", "appendix figure", "Relative p50 MAE against RLQR by target."),
        _route("4.1.1", "result_section", "latex_table", f"tables/forecast_metrics_full_primary_{split}.tex", _sub_sources("4.1.1", f"latex/rq1_4_1_1_forecast_metrics_full_primary_{split}.tex"), "mean_pinball_loss", "main thesis table", "Primary full-sample mean pinball table."),
        _route("4.1.1", "result_section", "latex_table", "tables/computational_cost_365d.tex", _sub_sources("4.1.1", "latex/rq1_4_1_1_computational_cost_365d.tex"), "computational_cost", "main thesis table", "Observed and 365-day-scaled computational cost table."),
        _route("4.1.1", "appendix", "latex_table", f"tables/forecast_metrics_full_detailed_{split}.tex", _sub_sources("4.1.1", f"latex/rq1_4_1_1_forecast_metrics_full_detailed_{split}.tex"), "point_and_probabilistic_errors", "appendix table", "Detailed full-sample metrics."),
        _route("4.1.1", "appendix", "latex_table", f"tables/da_price_p50_error_tolerance_summary_{split}.tex", _sub_sources("4.1.1", f"latex/rq1_4_1_1_da_price_p50_error_tolerance_summary_{split}.tex"), "p50 absolute error tolerance", "appendix table", "DA price p50 absolute error tolerance threshold summary."),
        _route("4.1.1", "backup", "csv", "csv/forecast_metrics_full_long.csv", _sub_sources("4.1.1", "csv/rq1_4_1_1_forecast_metrics_full_long.csv"), "all_full_metrics", "backup data", "Long-form full metrics CSV."),
        _route("4.1.1", "backup", "csv", f"csv/forecast_metrics_full_primary_{split}.csv", _sub_sources("4.1.1", f"csv/rq1_4_1_1_forecast_metrics_full_primary_{split}.csv"), "mean_pinball_loss", "backup data", "Primary full metrics CSV."),
        _route("4.1.1", "backup", "csv", f"csv/forecast_metrics_full_detailed_{split}.csv", _sub_sources("4.1.1", f"csv/rq1_4_1_1_forecast_metrics_full_detailed_{split}.csv"), "point_and_probabilistic_errors", "backup data", "Detailed full metrics CSV."),
        _route("4.1.1", "backup", "csv", "csv/computational_cost_365d.csv", _sub_sources("4.1.1", "csv/rq1_4_1_1_computational_cost_365d.csv"), "computational_cost", "backup data", "Observed and scaled computational cost source CSV."),
        _route("4.1.1", "backup", "csv", "csv/price_p50_absolute_error_tolerance_curve.csv", _sub_sources("4.1.1", "csv/rq1_4_1_1_price_p50_absolute_error_tolerance_curve.csv"), "p50 absolute error tolerance", "backup data", "Price p50 absolute error tolerance curve source CSV."),
        _route("4.1.1", "backup", "csv", f"csv/price_p50_error_tolerance_summary_{split}.csv", _sub_sources("4.1.1", f"csv/rq1_4_1_1_price_p50_error_tolerance_summary_{split}.csv"), "p50 absolute error tolerance", "backup data", "Price p50 absolute error tolerance summary CSV."),
        _route("4.1.1", "backup", "csv", f"csv/price_p50_error_tolerance_performance_values_{split}.csv", _sub_sources("4.1.1", f"csv/rq1_4_1_1_price_p50_error_tolerance_performance_values_{split}.csv"), "p50 absolute error tolerance", "backup data", "Price p50 absolute error tolerance performance values CSV."),
        _route("4.1.1", "backup", "diagnostics", f"diagnostics/price_p50_error_tolerance_performance_values_{split}.txt", _sub_sources("4.1.1", f"diagnostics/rq1_4_1_1_price_p50_error_tolerance_performance_values_{split}.txt"), "p50 absolute error tolerance", "diagnostics", "Readable price p50 absolute error tolerance performance values."),
        _route("4.1.1", "backup", "diagnostics", f"diagnostics/forecast_metrics_full_alignment_diagnostics_{split}.csv", _sub_sources("4.1.1", f"csv/rq1_4_1_1_forecast_metrics_full_alignment_diagnostics_{split}.csv"), "alignment", "diagnostics", "Alignment diagnostics for full metrics."),
        _route("4.1.2", "result_section", "figure", "figures/calibration_reliability_by_target.png", _sub_sources("4.1.2", "figures/rq1_4_1_2_calibration_reliability_by_target.png"), "calibration", "main thesis figure", "Quantile reliability by target."),
        _route("4.1.2", "result_section", "figure", "figures/calibration_reliability_activation_aggregates.png", _sub_sources("4.1.2", "figures/rq1_4_1_2_calibration_reliability_activation_aggregates.png") + ["_raw_outputs/4_1_2_calibration_uncertainty/figures/rq1_4_1_2_calibration_reliability_activation_aggregates.png", "_raw_outputs/calibration/figures/calibration_reliability_activation_aggregates.png"], "calibration", "main thesis figure", "Quantile reliability for activation price and rate with positive/negative directions merged.", required=False),
        _route("4.1.2", "result_section", "figure", "figures/calibration_interval_coverage_by_target_group.png", _sub_sources("4.1.2", "figures/rq1_4_1_2_calibration_interval_coverage_by_target_group.png"), "interval_coverage", "main thesis figure", "Interval coverage reliability by target group.", required=False),
        _route("4.1.2", "result_section", "latex_table", f"tables/calibration_summary_{split}.tex", _sub_sources("4.1.2", f"latex/rq1_4_1_2_calibration_summary_{split}.tex"), "calibration", "main thesis table", "Calibration summary table."),
        _route("4.1.2", "appendix", "latex_table", f"tables/calibration_quantile_coverage_{split}_appendix.tex", _sub_sources("4.1.2", f"latex/rq1_4_1_2_calibration_quantile_coverage_{split}_appendix.tex"), "calibration", "appendix table", "Quantile coverage appendix table."),
        _route("4.1.2", "appendix", "latex_table", f"tables/calibration_interval_quality_{split}_appendix.tex", _sub_sources("4.1.2", f"latex/rq1_4_1_2_calibration_interval_quality_{split}_appendix.tex"), "interval_coverage_width", "appendix table", "Interval quality appendix table."),
        _route("4.1.2", "appendix", "latex_table", f"tables/calibration_quantile_crossing_{split}_appendix.tex", _sub_sources("4.1.2", f"latex/rq1_4_1_2_calibration_quantile_crossing_{split}_appendix.tex"), "quantile_crossing", "appendix table", "Quantile crossing appendix table."),
        _route("4.1.2", "appendix", "figure", "figures/calibration_interval_width_by_target_group.png", _sub_sources("4.1.2", "figures/rq1_4_1_2_calibration_interval_width_by_target_group.png"), "interval_width", "appendix figure", "Interval width by target group.", required=False),
        _route("4.1.2", "appendix", "figure", "figures/calibration_quantile_crossing_by_target_group.png", _sub_sources("4.1.2", "figures/rq1_4_1_2_calibration_quantile_crossing_by_target_group.png"), "quantile_crossing", "appendix figure", "Quantile crossing by target group.", required=False),
        _route("4.1.2", "appendix", "figure", "figures/calibration_error_heatmap.png", _sub_sources("4.1.2", "figures/rq1_4_1_2_calibration_error_heatmap.png"), "calibration", "appendix figure", "Calibration error heatmap.", required=False),
        _route("4.1.2", "backup", "csv", f"csv/calibration_quantile_coverage_{split}.csv", _sub_sources("4.1.2", f"csv/rq1_4_1_2_calibration_quantile_coverage_{split}.csv"), "calibration", "backup data", "Quantile coverage CSV."),
        _route("4.1.2", "backup", "csv", f"csv/calibration_interval_coverage_width_{split}.csv", _sub_sources("4.1.2", f"csv/rq1_4_1_2_calibration_interval_coverage_width_{split}.csv"), "interval_coverage_width", "backup data", "Interval coverage and width CSV."),
        _route("4.1.2", "backup", "csv", f"csv/calibration_quantile_crossing_{split}.csv", _sub_sources("4.1.2", f"csv/rq1_4_1_2_calibration_quantile_crossing_{split}.csv"), "quantile_crossing", "backup data", "Quantile crossing CSV."),
        _route("4.1.2", "backup", "csv", f"csv/calibration_summary_{split}.csv", _sub_sources("4.1.2", f"csv/rq1_4_1_2_calibration_summary_{split}.csv"), "calibration", "backup data", "Calibration summary CSV."),
        _route("4.1.2", "backup", "diagnostics", "diagnostics/calibration_row_counts.csv", _sub_sources("4.1.2", "csv/rq1_4_1_2_calibration_row_counts.csv"), "row_counts", "diagnostics", "Calibration row counts."),
        _route("4.1.2", "backup", "warnings", "warnings/calibration_warnings.csv", _sub_sources("4.1.2", "csv/rq1_4_1_2_calibration_warnings.csv"), "warnings", "warnings", "Calibration warnings."),
        _route("4.1.3", "result_section", "latex_table", f"tables/per_lead_range_summary_{split}.tex", _sub_sources("4.1.3", f"latex/per_lead_range_summary_{split}.tex") + [f"latex/per_lead_range_summary_{split}.tex"], "mean_pinball_loss", "main thesis table", "Per-lead range mean pinball table."),
        _route("4.1.3", "backup", "csv", f"csv/per_lead_metrics_{split}.csv", _sub_sources("4.1.3", f"per_lead_metrics_{split}.csv") + [f"per_lead_metrics_{split}.csv"], "per_lead_metrics", "backup data", "Per-lead metrics CSV."),
        _route("4.1.3", "backup", "csv", f"csv/per_lead_range_summary_{split}.csv", _sub_sources("4.1.3", f"per_lead_range_summary_{split}.csv") + [f"per_lead_range_summary_{split}.csv"], "mean_pinball_loss", "backup data", "Per-lead range summary CSV."),
        _route("4.1.3", "backup", "diagnostics", f"diagnostics/per_lead_row_counts_{split}.csv", _sub_sources("4.1.3", f"per_lead_row_counts_{split}.csv") + [f"per_lead_row_counts_{split}.csv"], "row_counts", "diagnostics", "Per-lead row counts."),
        _route("4.1.3", "backup", "warnings", "warnings/per_lead_warnings.csv", _sub_sources("4.1.3", "per_lead_warnings.csv") + ["per_lead_warnings.csv"], "warnings", "warnings", "Per-lead warnings."),
        _route("4.1.4", "result_section", "figure", "figures/gate_bucket_pinball_by_target_group.png", _sub_sources("4.1.4", "figures/gate_bucket_pinball_by_target_group.png") + ["figures/gate_bucket_pinball_by_target_group.png"], "relative_mean_pinball_loss", "main thesis figure", "Actionable forecast performance by market gate relative to RLQR; values below 1 indicate lower mean pinball loss than RLQR."),
        _route("4.1.4", "result_section", "latex_table", f"tables/gate_bucket_metrics_{split}.tex", _sub_sources("4.1.4", f"latex/gate_bucket_metrics_{split}.tex") + [f"latex/gate_bucket_metrics_{split}.tex"], "mean_pinball_loss", "main thesis table", "Gate bucket mean pinball table."),
        _route("4.1.4", "appendix", "figure", "figures/gate_bucket_coverage_p10_p90_by_target_group.png", _sub_sources("4.1.4", "figures/gate_bucket_coverage_p10_p90_by_target_group.png") + ["figures/gate_bucket_coverage_p10_p90_by_target_group.png"], "interval_coverage", "appendix figure", "Gate bucket p10-p90 coverage."),
        _route("4.1.4", "appendix", "figure", "figures/gate_bucket_observed_leads.png", _sub_sources("4.1.4", "figures/gate_bucket_observed_leads.png") + ["figures/gate_bucket_observed_leads.png"], "observed_leads", "appendix figure", "Observed leads by gate bucket."),
        _route("4.1.4", "backup", "csv", f"csv/gate_bucket_metrics_{split}.csv", _sub_sources("4.1.4", f"gate_bucket_metrics_{split}.csv") + [f"gate_bucket_metrics_{split}.csv"], "gate_bucket_metrics", "backup data", "Gate bucket metrics CSV."),
        _route("4.1.4", "backup", "csv", "csv/gate_bucket_definitions.csv", _sub_sources("4.1.4", "gate_bucket_definitions.csv") + ["gate_bucket_definitions.csv"], "definitions", "backup data", "Gate bucket definitions."),
        _route("4.1.4", "backup", "diagnostics", "diagnostics/gate_bucket_row_counts.csv", _sub_sources("4.1.4", "gate_bucket_row_counts.csv") + ["gate_bucket_row_counts.csv"], "row_counts", "diagnostics", "Gate bucket row counts."),
        _route("4.1.4", "backup", "diagnostics", "diagnostics/gate_bucket_observed_leads.csv", _sub_sources("4.1.4", "gate_bucket_observed_leads.csv") + ["gate_bucket_observed_leads.csv"], "observed_leads", "diagnostics", "Gate bucket observed leads."),
        _route("4.1.4", "backup", "warnings", "warnings/gate_bucket_warnings.csv", _sub_sources("4.1.4", "gate_bucket_warnings.csv") + ["gate_bucket_warnings.csv"], "warnings", "warnings", "Gate bucket warnings."),
        _route("4.1.5", "result_section", "figure", "figures/tail_spike_relative_pinball_by_regime_price_capacity.png", _sub_sources("4.1.5", "figures/tail_spike_relative_pinball_by_regime_price_capacity.png") + ["_raw_outputs/4_1_5_tail_spike/figures/tail_spike_relative_pinball_by_regime_price_capacity.png", "_raw_outputs/shared/figures/tail_spike_relative_pinball_by_regime_price_capacity.png", "figures/tail_spike_relative_pinball_by_regime_price_capacity.png"], "relative_mean_pinball_loss", "main thesis figure", "Tail/spike relative mean pinball by regime for DA and capacity price targets."),
        _route("4.1.5", "result_section", "figure", "figures/tail_spike_relative_pinball_by_regime_activation.png", _sub_sources("4.1.5", "figures/tail_spike_relative_pinball_by_regime_activation.png") + ["_raw_outputs/4_1_5_tail_spike/figures/tail_spike_relative_pinball_by_regime_activation.png", "_raw_outputs/shared/figures/tail_spike_relative_pinball_by_regime_activation.png", "figures/tail_spike_relative_pinball_by_regime_activation.png"], "relative_mean_pinball_loss", "main thesis figure", "Tail/spike relative mean pinball by regime for activation price and rate targets."),
        _route("4.1.5", "result_section", "figure", "figures/tail_spike_afrr_main_regime_relative_pinball.png", _sub_sources("4.1.5", "figures/tail_spike_afrr_main_regime_relative_pinball.png") + ["_raw_outputs/4_1_5_tail_spike/figures/tail_spike_afrr_main_regime_relative_pinball.png", "_raw_outputs/shared/figures/tail_spike_afrr_main_regime_relative_pinball.png", "figures/tail_spike_afrr_main_regime_relative_pinball.png"], "relative_mean_pinball_loss", "main thesis figure", "aFRR main-regime relative mean pinball by target."),
        _route("4.1.5", "result_section", "figure", "figures/tail_spike_afrr_main_regime_relative_pinball.pdf", _sub_sources("4.1.5", "figures/tail_spike_afrr_main_regime_relative_pinball.pdf") + ["_raw_outputs/4_1_5_tail_spike/figures/tail_spike_afrr_main_regime_relative_pinball.pdf", "_raw_outputs/shared/figures/tail_spike_afrr_main_regime_relative_pinball.pdf", "figures/tail_spike_afrr_main_regime_relative_pinball.pdf"], "relative_mean_pinball_loss", "main thesis figure", "aFRR main-regime relative mean pinball PDF.", required=False),
        _route("4.1.5", "result_section", "latex_figure", "latex_figures/tail_spike_afrr_main_regime_relative_pinball.tex", _sub_sources("4.1.5", "latex_figures/tail_spike_afrr_main_regime_relative_pinball.tex") + ["_raw_outputs/4_1_5_tail_spike/latex_figures/tail_spike_afrr_main_regime_relative_pinball.tex", "_raw_outputs/shared/latex_figures/tail_spike_afrr_main_regime_relative_pinball.tex", "latex_figures/tail_spike_afrr_main_regime_relative_pinball.tex"], "relative_mean_pinball_loss", "main thesis figure", "LaTeX wrapper for aFRR main-regime relative pinball figure."),
        _route("4.1.5", "result_section", "figure", "figures/tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.png", _sub_sources("4.1.5", "figures/tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.png") + ["_raw_outputs/4_1_5_tail_spike/figures/tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.png", "_raw_outputs/shared/figures/tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.png", "figures/tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.png"], "relative_mean_pinball_loss", "main thesis figure", "Corrected aFRR three-regime Figure 9 PNG."),
        _route("4.1.5", "result_section", "figure", "figures/tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.pdf", _sub_sources("4.1.5", "figures/tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.pdf") + ["_raw_outputs/4_1_5_tail_spike/figures/tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.pdf", "_raw_outputs/shared/figures/tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.pdf", "figures/tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.pdf"], "relative_mean_pinball_loss", "main thesis figure", "Corrected aFRR three-regime Figure 9 PDF.", required=False),
        _route("4.1.5", "result_section", "latex_figure", "latex_figures/tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.tex", _sub_sources("4.1.5", "latex_figures/tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.tex") + ["_raw_outputs/4_1_5_tail_spike/latex_figures/tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.tex", "_raw_outputs/shared/latex_figures/tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.tex", "latex_figures/tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.tex"], "relative_mean_pinball_loss", "main thesis figure", "LaTeX wrapper for corrected aFRR three-regime Figure 9."),
        _route("4.1.5", "result_section", "figure", "figures/tail_spike_residual_distribution_by_regime.png", _sub_sources("4.1.5", "figures/tail_spike_residual_distribution_by_regime.png") + ["_raw_outputs/4_1_5_tail_spike/figures/tail_spike_residual_distribution_by_regime.png", "_raw_outputs/shared/figures/tail_spike_residual_distribution_by_regime.png", "figures/tail_spike_residual_distribution_by_regime.png"], "residuals", "main thesis figure", "Tail/spike residual distributions."),
        _route("4.1.5", "result_section", "latex_table", f"tables/tail_spike_metrics_{split}.tex", _sub_sources("4.1.5", f"latex/tail_spike_metrics_{split}.tex") + [f"_raw_outputs/4_1_5_tail_spike/latex/tail_spike_metrics_{split}.tex", f"_raw_outputs/shared/latex/tail_spike_metrics_{split}.tex", f"latex/tail_spike_metrics_{split}.tex"], "mean_pinball_loss", "main thesis table", "Tail/spike mean pinball table."),
        _route("4.1.5", "appendix", "figure", "figures/tail_spike_coverage_by_regime.png", _sub_sources("4.1.5", "figures/tail_spike_coverage_by_regime.png") + ["_raw_outputs/4_1_5_tail_spike/figures/tail_spike_coverage_by_regime.png", "_raw_outputs/shared/figures/tail_spike_coverage_by_regime.png", "figures/tail_spike_coverage_by_regime.png"], "interval_coverage", "appendix figure", "Tail/spike p10-p90 coverage."),
        _route("4.1.5", "appendix", "figure", "figures/tail_spike_mae_p50_by_regime.png", _sub_sources("4.1.5", "figures/tail_spike_mae_p50_by_regime.png") + ["_raw_outputs/4_1_5_tail_spike/figures/tail_spike_mae_p50_by_regime.png", "_raw_outputs/shared/figures/tail_spike_mae_p50_by_regime.png", "figures/tail_spike_mae_p50_by_regime.png"], "mae_p50", "appendix figure", "Tail/spike p50 MAE."),
        _route("4.1.5", "appendix", "figure", "figures/tail_spike_forecast_band_selected_week.png", _sub_sources("4.1.5", "figures/tail_spike_forecast_band_*.png") + ["_raw_outputs/4_1_5_tail_spike/figures/tail_spike_forecast_band_*.png", "_raw_outputs/shared/figures/tail_spike_forecast_band_*.png", "figures/tail_spike_forecast_band_*.png"], "forecast_band", "appendix figure", "Selected-week tail/spike forecast-band example.", required=False),
        _route("4.1.5", "backup", "csv", f"csv/tail_spike_metrics_{split}.csv", _sub_sources("4.1.5", f"tail_spike_metrics_{split}.csv") + [f"_raw_outputs/4_1_5_tail_spike/tail_spike_metrics_{split}.csv", f"_raw_outputs/shared/tail_spike_metrics_{split}.csv", f"tail_spike_metrics_{split}.csv"], "tail_spike_metrics", "backup data", "Tail/spike metrics CSV."),
        _route("4.1.5", "backup", "csv", "csv/tail_spike_afrr_main_regime_metrics.csv", _sub_sources("4.1.5", "tail_spike_afrr_main_regime_metrics.csv") + ["_raw_outputs/4_1_5_tail_spike/tail_spike_afrr_main_regime_metrics.csv", "_raw_outputs/shared/tail_spike_afrr_main_regime_metrics.csv", "tail_spike_afrr_main_regime_metrics.csv"], "tail_spike_metrics", "backup data", "aFRR main-regime metrics CSV."),
        _route("4.1.5", "backup", "csv", "csv/tail_spike_afrr_main_regime_points.csv", _sub_sources("4.1.5", "tail_spike_afrr_main_regime_points.csv") + ["_raw_outputs/4_1_5_tail_spike/tail_spike_afrr_main_regime_points.csv", "_raw_outputs/shared/tail_spike_afrr_main_regime_points.csv", "tail_spike_afrr_main_regime_points.csv"], "tail_spike_points", "backup data", "aFRR main-regime point-level CSV."),
        _route("4.1.5", "backup", "csv", "csv/tail_spike_afrr_three_regime_plot_source.csv", _sub_sources("4.1.5", "tail_spike_afrr_three_regime_plot_source.csv") + ["_raw_outputs/4_1_5_tail_spike/tail_spike_afrr_three_regime_plot_source.csv", "_raw_outputs/shared/tail_spike_afrr_three_regime_plot_source.csv", "tail_spike_afrr_three_regime_plot_source.csv"], "relative_mean_pinball_loss", "backup data", "Corrected aFRR three-regime plot source CSV."),
        _route("4.1.5", "backup", "csv", "csv/tail_spike_regime_definitions.csv", _sub_sources("4.1.5", "tail_spike_regime_definitions.csv") + ["_raw_outputs/4_1_5_tail_spike/tail_spike_regime_definitions.csv", "_raw_outputs/shared/tail_spike_regime_definitions.csv", "tail_spike_regime_definitions.csv"], "definitions", "backup data", "Tail/spike regime definitions."),
        _route("4.1.5", "backup", "csv", "csv/tail_spike_thresholds.csv", _sub_sources("4.1.5", "tail_spike_thresholds.csv") + ["_raw_outputs/4_1_5_tail_spike/tail_spike_thresholds.csv", "_raw_outputs/shared/tail_spike_thresholds.csv", "tail_spike_thresholds.csv"], "thresholds", "backup data", "Tail/spike thresholds."),
        _route("4.1.5", "backup", "diagnostics", "diagnostics/tail_spike_afrr_main_regime_definitions.csv", _sub_sources("4.1.5", "tail_spike_afrr_main_regime_definitions.csv") + ["_raw_outputs/4_1_5_tail_spike/tail_spike_afrr_main_regime_definitions.csv", "_raw_outputs/shared/tail_spike_afrr_main_regime_definitions.csv", "tail_spike_afrr_main_regime_definitions.csv"], "definitions", "diagnostics", "aFRR main-regime definitions and thresholds."),
        _route("4.1.5", "backup", "diagnostics", "diagnostics/tail_spike_afrr_main_regime_overlap.csv", _sub_sources("4.1.5", "tail_spike_afrr_main_regime_overlap.csv") + ["_raw_outputs/4_1_5_tail_spike/tail_spike_afrr_main_regime_overlap.csv", "_raw_outputs/shared/tail_spike_afrr_main_regime_overlap.csv", "tail_spike_afrr_main_regime_overlap.csv"], "overlap", "diagnostics", "aFRR stress-week and high-tail overlap diagnostics."),
        _route("4.1.5", "backup", "diagnostics", "diagnostics/tail_spike_row_counts.csv", _sub_sources("4.1.5", "tail_spike_row_counts.csv") + ["_raw_outputs/4_1_5_tail_spike/tail_spike_row_counts.csv", "_raw_outputs/shared/tail_spike_row_counts.csv", "tail_spike_row_counts.csv"], "row_counts", "diagnostics", "Tail/spike row counts."),
        _route("4.1.5", "backup", "diagnostics", "diagnostics/tail_spike_selected_weeks.csv", _sub_sources("4.1.5", "tail_spike_selected_weeks.csv") + ["_raw_outputs/4_1_5_tail_spike/tail_spike_selected_weeks.csv", "_raw_outputs/shared/tail_spike_selected_weeks.csv", "tail_spike_selected_weeks.csv"], "selected_weeks", "diagnostics", "Tail/spike selected weeks."),
        _route("4.1.5", "backup", "diagnostics", "diagnostics/tail_spike_relative_pinball_by_regime_all_in_one.png", _sub_sources("4.1.5", "figures/tail_spike_relative_pinball_by_regime_all_in_one.png") + ["_raw_outputs/4_1_5_tail_spike/figures/tail_spike_relative_pinball_by_regime_all_in_one.png", "_raw_outputs/shared/figures/tail_spike_relative_pinball_by_regime_all_in_one.png", "figures/tail_spike_relative_pinball_by_regime_all_in_one.png"], "relative_mean_pinball_loss", "diagnostic figure", "Dense all-in-one tail/spike relative mean pinball chart.", required=False),
        _route("4.1.5", "backup", "warnings", "warnings/tail_spike_warnings.csv", _sub_sources("4.1.5", "tail_spike_warnings.csv") + ["_raw_outputs/4_1_5_tail_spike/tail_spike_warnings.csv", "_raw_outputs/shared/tail_spike_warnings.csv", "tail_spike_warnings.csv"], "warnings", "warnings", "Tail/spike warnings."),
        _route("4.1.5", "backup", "warnings", "warnings/tail_spike_afrr_three_regime_warnings.csv", _sub_sources("4.1.5", "tail_spike_afrr_three_regime_warnings.csv") + ["_raw_outputs/4_1_5_tail_spike/tail_spike_afrr_three_regime_warnings.csv", "_raw_outputs/shared/tail_spike_afrr_three_regime_warnings.csv", "tail_spike_afrr_three_regime_warnings.csv"], "warnings", "warnings", "Corrected aFRR three-regime warnings."),
    ]
    price_tolerance_stems = [
        "da_price",
        "afrr_capacity_price_pos",
        "afrr_capacity_price_neg",
        "afrr_activation_price_pos",
        "afrr_activation_price_neg",
    ]
    for stem in price_tolerance_stems:
        routes.extend(
            [
                _route("4.1.1", "result_section", "figure", f"figures/{stem}_p50_absolute_error_tolerance_curve.png", _sub_sources("4.1.1", f"figures/rq1_4_1_1_{stem}_p50_absolute_error_tolerance_curve.png"), "p50 absolute error tolerance", "secondary interpretability figure", f"{stem} cumulative absolute p50 error tolerance curve."),
                _route("4.1.1", "result_section", "figure", f"figures/{stem}_p50_absolute_error_tolerance_curve.pdf", _sub_sources("4.1.1", f"figures/rq1_4_1_1_{stem}_p50_absolute_error_tolerance_curve.pdf"), "p50 absolute error tolerance", "secondary interpretability figure", f"{stem} cumulative absolute p50 error tolerance curve PDF.", required=False),
                _route("4.1.1", "result_section", "latex_figure", f"latex_figures/{stem}_p50_absolute_error_tolerance_curve.tex", _sub_sources("4.1.1", f"latex_figures/rq1_4_1_1_{stem}_p50_absolute_error_tolerance_curve.tex"), "p50 absolute error tolerance", "secondary interpretability figure", f"LaTeX includegraphics snippet for {stem} p50 error tolerance curve."),
            ]
        )
    target_stems = [
        "da_price",
        "afrr_capacity_price",
        "afrr_activation_price",
        "afrr_activation_rate",
    ]
    for stem in target_stems:
        routes.extend(
            [
                _route("4.1.3", "result_section", "figure", f"figures/per_lead_pinball_{stem}.png", _sub_sources("4.1.3", f"figures/per_lead_pinball_{stem}.png") + [f"figures/per_lead_pinball_{stem}.png"], "mean_pinball_loss", "main thesis figure", f"Per-lead mean pinball for {stem}."),
                _route("4.1.3", "result_section", "figure", f"figures/per_lead_relative_pinball_{stem}.png", _sub_sources("4.1.3", f"figures/per_lead_relative_pinball_{stem}.png") + [f"figures/per_lead_relative_pinball_{stem}.png"], "relative_mean_pinball_loss", "main thesis figure", f"Per-lead relative mean pinball for {stem}."),
                _route("4.1.3", "appendix", "figure", f"figures/per_lead_mae_p50_{stem}.png", _sub_sources("4.1.3", f"figures/per_lead_mae_p50_{stem}.png") + [f"figures/per_lead_mae_p50_{stem}.png"], "mae_p50", "appendix figure", f"Per-lead p50 MAE for {stem}.", required=False),
                _route("4.1.3", "appendix", "figure", f"figures/per_lead_rmse_p50_{stem}.png", _sub_sources("4.1.3", f"figures/per_lead_rmse_p50_{stem}.png") + [f"figures/per_lead_rmse_p50_{stem}.png"], "rmse_p50", "appendix figure", f"Per-lead p50 RMSE for {stem}.", required=False),
            ]
        )
    return routes


def _latex_escape(value: Any) -> str:
    s = str(value)
    minus_token = "@@RQ1MINUS@@"
    minus_labels = [
        "aFRR capacity price",
        "aFRR Capacity Price",
        "aFRR activation price",
        "aFRR Activation Price",
        "aFRR activation rate",
        "aFRR Activation Rate",
    ]
    for label in minus_labels:
        s = s.replace(f"{label} -", f"{label} {minus_token}")
        s = s.replace(f"{label} \u2212", f"{label} {minus_token}")
    for old, new in {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }.items():
        s = s.replace(old, new)
    return s.replace(minus_token, "$-$")


def _axis_label(value: Any) -> str:
    s = str(value)
    if s == "mean pinball loss":
        return "Mean pinball loss"
    return s


def _fmt(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "-"
    return f"{x:.4f}" if np.isfinite(x) else "-"


def _ensure_caption_period(caption: Any) -> str:
    text = str(caption).strip()
    if not text:
        return "."
    return text if text.endswith(".") else text + "."


def _write_table(path: Path, headers: list[str], rows: list[list[Any]], caption: str, label: str) -> Path | None:
    if not rows:
        return None
    align = "l" * len(headers)
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        f"    \\caption{{{_latex_escape(_ensure_caption_period(caption))}}}",
        f"    \\label{{{_latex_escape(label)}}}",
        rf"    \begin{{tabular}}{{@{{}}{align}@{{}}}}",
        r"        \toprule",
        "        " + " & ".join(r"\textbf{" + _latex_escape(h) + "}" for h in headers) + r" \\",
        r"        \midrule",
    ]
    for row in rows:
        lines.append("        " + " & ".join(_latex_escape(v) if not isinstance(v, (float, int, np.floating, np.integer)) else _fmt(v) for v in row) + r" \\")
    lines.extend([r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _best(vals: dict[str, float]) -> str:
    finite = {k: v for k, v in vals.items() if np.isfinite(v)}
    return min(finite, key=finite.get) if finite else ""


def _pivot_metric_rows(df: pd.DataFrame, group_cols: list[str], metric: str, *, model_col: str = "model_label") -> pd.DataFrame:
    return df.pivot_table(index=group_cols, columns=model_col, values=metric, aggfunc="mean").reset_index()


def _derive_per_lead_appendix(csv_path: Path, out_path: Path, split: str) -> Path | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        df = df[df["split"].eq(split)]
    rows: list[dict[str, Any]] = []
    metrics = [("mean_pinball_loss", "Mean pinball loss"), ("mae_p50", "MAE p50"), ("rmse_p50", "RMSE p50"), ("bias_p50", "Bias p50")]
    metric_order = {label: idx for idx, (_, label) in enumerate(metrics)}
    for metric, label in metrics:
        if metric not in df.columns:
            continue
        pivot = _pivot_metric_rows(df, ["target_label", "lead_time_h"], metric)
        pivot = sort_target_frame(pivot, target_col="target_label", extra_cols=["lead_time_h"])
        for _, row in pivot.iterrows():
            vals = {m: float(row[m]) for m in MODEL_LABELS if m in row and pd.notna(row[m])}
            n = int(df[(df["target_label"].eq(row["target_label"])) & (df["lead_time_h"].eq(row["lead_time_h"]))]["n_obs"].min())
            rows.append(
                {
                    "Target": row["target_label"],
                    "Lead hour": int(row["lead_time_h"]),
                    "Metric": label,
                    "RLQR": vals.get("RLQR", np.nan),
                    "XGB": vals.get("XGB", np.nan),
                    "TFT": vals.get("TFT", np.nan),
                    "Best model": _best(vals),
                    "N": n,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return None
    out["_metric_order"] = out["Metric"].map(metric_order).fillna(99)
    out = sort_target_frame(out, target_col="Target", extra_cols=["_metric_order", "Lead hour"])
    table_rows = out[["Target", "Lead hour", "Metric", "RLQR", "XGB", "TFT", "Best model", "N"]].values.tolist()
    return _write_table(out_path, ["Target", "Lead hour", "Metric", "RLQR", "XGB", "TFT", "Best model", "N"], table_rows, "Per-lead detailed forecast metrics on the test split.", "tab:per_lead_detailed_metrics_test")


def _derive_gate_interval_appendix(csv_path: Path, out_path: Path, split: str) -> Path | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        df = df[df["split"].eq(split)]
    target_col = "target" if "target" in df.columns else "target_group" if "target_group" in df.columns else None
    if target_col is not None:
        df = sort_target_frame(df, target_col=target_col, extra_cols=["bucket", "model_label"])
    rows: list[list[Any]] = []
    for _, row in df.iterrows():
        rows.append([
            row.get("bucket", ""),
            row.get("target_group", row.get("target", "")),
            row.get("model_label", row.get("model", "")),
            row.get("coverage_p10_p90", np.nan),
            row.get("interval_width_p10_p90_mean", np.nan),
            row.get("n_obs", ""),
        ])
    return _write_table(out_path, ["Bucket", "Target", "Model", "p10-p90 coverage", "p10-p90 width", "N"], rows, "Gate-bucket interval quality on the test split.", "tab:gate_bucket_interval_quality_test")


def _derive_tail_appendix(csv_path: Path, out_path: Path, split: str) -> Path | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        df = df[df["split"].eq(split)]
    rows: list[dict[str, Any]] = []
    metrics = [
        ("mean_pinball_loss", "Mean pinball loss"),
        ("mae_p50", "MAE p50"),
        ("rmse_p50", "RMSE p50"),
        ("bias_p50", "Bias p50"),
        ("coverage_p10_p90", "p10-p90 coverage"),
        ("interval_width_p10_p90_mean", "p10-p90 width"),
    ]
    metric_order = {label: idx for idx, (_, label) in enumerate(metrics)}
    for metric, label in metrics:
        if metric not in df.columns:
            continue
        pivot = _pivot_metric_rows(df, ["target_label", "regime"], metric)
        pivot = sort_target_frame(pivot, target_col="target_label", extra_cols=["regime"])
        for _, row in pivot.iterrows():
            vals = {m: float(row[m]) for m in MODEL_LABELS if m in row and pd.notna(row[m])}
            mask = df["regime"].eq(row["regime"]) & df["target_label"].eq(row["target_label"])
            n = int(df[mask]["n_obs"].min())
            rows.append(
                {
                    "Target": row["target_label"],
                    "Regime": row["regime"],
                    "Metric": label,
                    "RLQR": vals.get("RLQR", np.nan),
                    "XGB": vals.get("XGB", np.nan),
                    "TFT": vals.get("TFT", np.nan),
                    "Best model": _best(vals),
                    "N": n,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return None
    out["_metric_order"] = out["Metric"].map(metric_order).fillna(99)
    out = sort_target_frame(out, target_col="Target", extra_cols=["_metric_order", "Regime"])
    table_rows = out[["Target", "Regime", "Metric", "RLQR", "XGB", "TFT", "Best model", "N"]].values.tolist()
    return _write_table(out_path, ["Target", "Regime", "Metric", "RLQR", "XGB", "TFT", "Best model", "N"], table_rows, "Tail/spike detailed metrics on the test split.", "tab:tail_spike_detailed_metrics_test")


def _find_source(final_root: Path, rq1_root: Path, candidates: tuple[str, ...]) -> Path | None:
    for rel in candidates:
        for base in [rq1_root, final_root]:
            if any(ch in rel for ch in "*?["):
                matches = sorted(base.glob(rel))
                if matches:
                    return matches[0]
            else:
                path = base / rel
                if path.exists():
                    return path
    return None


def _copy_route(route: Route, *, final_root: Path, rq1_root: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    subsection_dir = rq1_root / SUBSECTIONS[route.subsection]
    dest = subsection_dir / route.tier / route.dest_rel
    source = _find_source(final_root, rq1_root, route.sources)
    if source is None and dest.exists():
        source = dest
    if source is None:
        return None, {
            "subsection": route.subsection,
            "tier": route.tier,
            "artifact_type": route.artifact_type,
            "path": str(dest),
            "metric_family": route.metric_family,
            "thesis_use": route.thesis_use,
            "brief_description": route.brief_description,
            "required": route.required,
            "status": "missing",
            "searched": list(route.sources),
        }
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)
    return {
        "subsection": route.subsection,
        "tier": route.tier,
        "artifact_type": route.artifact_type,
        "path": str(dest),
        "metric_family": route.metric_family,
        "thesis_use": route.thesis_use,
        "brief_description": route.brief_description,
    }, None


def _ensure_structure(rq1_root: Path) -> None:
    for name in SUBSECTIONS.values():
        for rel in [
            "result_section/figures",
            "result_section/latex_figures",
            "result_section/tables",
            "appendix/figures",
            "appendix/latex_figures",
            "appendix/tables",
            "backup/csv",
            "backup/diagnostics",
            "backup/warnings",
        ]:
            (rq1_root / name / rel).mkdir(parents=True, exist_ok=True)


def _remove_path(path: Path, removed: list[str]) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
    removed.append(str(path))


def prune_legacy_outputs(*, rq1_root: Path, final_root: Path) -> list[str]:
    """Remove known generated legacy/unstructured output locations."""
    removed: list[str] = []
    legacy_root_names = [
        ".DS_Store",
        "rq1",
        "calibration",
        "figures",
        "latex",
        "gate_bucket_definitions.csv",
        "gate_bucket_manifest.json",
        "gate_bucket_metrics.csv",
        "gate_bucket_metrics_test.csv",
        "gate_bucket_observed_leads.csv",
        "gate_bucket_row_counts.csv",
        "gate_bucket_warnings.csv",
        "per_lead_manifest.json",
        "per_lead_metrics.csv",
        "per_lead_metrics_test.csv",
        "per_lead_range_summary_detail_test.csv",
        "per_lead_range_summary_test.csv",
        "per_lead_row_counts_test.csv",
        "per_lead_warnings.csv",
        "tail_spike_manifest.json",
        "tail_spike_metrics.csv",
        "tail_spike_metrics_test.csv",
        "tail_spike_regime_definitions.csv",
        "tail_spike_row_counts.csv",
        "tail_spike_selected_weeks.csv",
        "tail_spike_thresholds.csv",
        "tail_spike_warnings.csv",
    ]
    for name in legacy_root_names:
        _remove_path(rq1_root / name, removed)
    for path in rq1_root.rglob(".DS_Store"):
        _remove_path(path, removed)

    for subdir in SUBSECTIONS.values():
        root = rq1_root / subdir
        for name in ["csv", "figures", "latex"]:
            _remove_path(root / name, removed)
        for path in root.glob("*.csv"):
            _remove_path(path, removed)
        for path in root.glob("*.json"):
            _remove_path(path, removed)

    _remove_path(
        rq1_root
        / SUBSECTIONS["4.1.5"]
        / "result_section"
        / "figures"
        / "tail_spike_relative_pinball_by_regime.png",
        removed,
    )
    _remove_path(
        rq1_root
        / SUBSECTIONS["4.1.5"]
        / "result_section"
        / "figures"
        / "tail_spike_relative_pinball_by_regime_main.png",
        removed,
    )
    for path in (rq1_root / SUBSECTIONS["4.1.1"] / "result_section" / "figures").glob("*_p50_absolute_error_tolerance_curve.tex"):
        _remove_path(path, removed)

    return removed


def _add_derived_tables(entries: list[dict[str, Any]], missing: list[dict[str, Any]], *, rq1_root: Path, split: str) -> None:
    derived = [
        ("4.1.3", "appendix", "latex_table", f"tables/per_lead_detailed_metrics_{split}.tex", rq1_root / SUBSECTIONS["4.1.3"] / "backup" / "csv" / f"per_lead_metrics_{split}.csv", _derive_per_lead_appendix, "per_lead_metrics", "appendix table", "Per-lead detailed metrics appendix table."),
        ("4.1.4", "appendix", "latex_table", f"tables/gate_bucket_interval_quality_{split}.tex", rq1_root / SUBSECTIONS["4.1.4"] / "backup" / "csv" / f"gate_bucket_metrics_{split}.csv", _derive_gate_interval_appendix, "interval_coverage_width", "appendix table", "Gate bucket interval quality appendix table."),
        ("4.1.5", "appendix", "latex_table", f"tables/tail_spike_detailed_metrics_{split}.tex", rq1_root / SUBSECTIONS["4.1.5"] / "backup" / "csv" / f"tail_spike_metrics_{split}.csv", _derive_tail_appendix, "tail_spike_metrics", "appendix table", "Tail/spike detailed metrics appendix table."),
    ]
    for subsection, tier, artifact_type, rel, source_csv, fn, metric_family, thesis_use, desc in derived:
        dest = rq1_root / SUBSECTIONS[subsection] / tier / rel
        path = fn(source_csv, dest, split)
        if path is None:
            missing.append({
                "subsection": subsection,
                "tier": tier,
                "artifact_type": artifact_type,
                "path": str(dest),
                "metric_family": metric_family,
                "thesis_use": thesis_use,
                "brief_description": desc,
                "required": True,
                "status": "missing_source_csv",
                "searched": [str(source_csv)],
            })
            continue
        entries.append({
            "subsection": subsection,
            "tier": tier,
            "artifact_type": artifact_type,
            "path": str(path),
            "metric_family": metric_family,
            "thesis_use": thesis_use,
            "brief_description": desc,
        })


def _find_example_source_dirs(final_root: Path, rq1_root: Path) -> list[Path]:
    candidates = [
        final_root / "_raw_outputs" / "4_1_6_example_weeks",
        rq1_root / "_raw_outputs" / "4_1_6_example_weeks",
        final_root / "4_1_6_example_weeks",
        rq1_root / "4_1_6_example_weeks",
        rq1_root / "rq1" / "4_1_6_example_weeks",
    ]
    out: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if path.exists() and resolved not in seen:
            out.append(path)
            seen.add(resolved)
    return out


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)


def _add_example_week_outputs(entries: list[dict[str, Any]], missing: list[dict[str, Any]], *, final_root: Path, rq1_root: Path) -> None:
    subsection = "4.1.6"
    target_root = rq1_root / SUBSECTIONS[subsection]
    result_week_keys: set[str] = set()
    for stale_dir in [
        target_root / "appendix" / "figures" / "high_volatility",
        target_root / "appendix" / "latex_figures" / "high_volatility",
        target_root / "result_section" / "figures" / "typical",
        target_root / "result_section" / "figures" / "high_volatility",
        target_root / "result_section" / "latex_figures" / "typical",
        target_root / "result_section" / "latex_figures" / "high_volatility",
    ]:
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
    source_dirs = _find_example_source_dirs(final_root, rq1_root)
    if not source_dirs:
        missing.append(
            {
                "subsection": subsection,
                "tier": "backup",
                "artifact_type": "diagnostics",
                "path": str(target_root / "backup" / "diagnostics" / "example_week_manifest.json"),
                "metric_family": "example_weeks",
                "thesis_use": "diagnostics",
                "brief_description": "Example-week manifest.",
                "required": False,
                "status": "missing_source_dir",
                "searched": [str(p) for p in _find_example_source_dirs(final_root, rq1_root)],
            }
        )
        return
    source_dir = source_dirs[0]

    def _is_stale_week_alias(path: Path) -> bool:
        suffix_pairs = [
            ("_typical", "_typical_week"),
            ("_high_volatility", "_high_volatility_week"),
        ]
        for legacy_suffix, canonical_suffix in suffix_pairs:
            if not path.stem.endswith(legacy_suffix):
                continue
            canonical_name = path.stem[: -len(legacy_suffix)] + canonical_suffix + path.suffix
            return any(candidate.name == canonical_name for candidate in source_dir.rglob(canonical_name))
        return False

    def _prune_target_stale_week_aliases() -> None:
        suffix_pairs = [
            ("_typical", "_typical_week"),
            ("_high_volatility", "_high_volatility_week"),
        ]
        canonical_names = {path.name for path in target_root.glob("**/*") if path.is_file()}
        for path in sorted(target_root.glob("**/*")):
            if not path.is_file() or path.suffix.lower() not in {".tex", ".png"}:
                continue
            for legacy_suffix, canonical_suffix in suffix_pairs:
                if not path.stem.endswith(legacy_suffix):
                    continue
                canonical_name = path.stem[: -len(legacy_suffix)] + canonical_suffix + path.suffix
                if canonical_name in canonical_names:
                    path.unlink()
                break

    metrics = source_dir / "example_week_metrics.csv"
    if not metrics.exists():
        metrics = source_dir / "backup" / "csv" / "example_week_metrics.csv"
    if metrics.exists():
        dst = target_root / "backup" / "csv" / "example_week_metrics.csv"
        _copy_file(metrics, dst)
        entries.append({"subsection": subsection, "tier": "backup", "artifact_type": "csv", "path": str(dst), "metric_family": "example_weeks", "thesis_use": "backup data", "brief_description": "Example-week plot inventory and diagnostics."})
    for src in sorted((source_dir / "backup" / "csv").glob("example_week*.csv")):
        dst = target_root / "backup" / "csv" / src.name
        _copy_file(src, dst)
        entries.append({"subsection": subsection, "tier": "backup", "artifact_type": "csv", "path": str(dst), "metric_family": "market_actionable_example_weeks", "thesis_use": "backup data", "brief_description": f"Market-actionable example-week data: {src.name}."})
    manifest = source_dir / "example_week_manifest.json"
    if not manifest.exists():
        manifest = source_dir / "backup" / "diagnostics" / "example_week_manifest.json"
    if manifest.exists():
        dst = target_root / "backup" / "diagnostics" / "example_week_manifest.json"
        _copy_file(manifest, dst)
        entries.append({"subsection": subsection, "tier": "backup", "artifact_type": "diagnostics", "path": str(dst), "metric_family": "example_weeks", "thesis_use": "diagnostics", "brief_description": "Example-week generation manifest."})
    for src in sorted((source_dir / "backup" / "diagnostics").glob("example_week*.csv")):
        dst = target_root / "backup" / "diagnostics" / src.name
        _copy_file(src, dst)
        entries.append({"subsection": subsection, "tier": "backup", "artifact_type": "diagnostics", "path": str(dst), "metric_family": "market_actionable_example_weeks", "thesis_use": "diagnostics", "brief_description": f"Market-actionable example-week diagnostics: {src.name}."})
    for src in sorted((source_dir / "backup" / "warnings").glob("example_week*.csv")):
        dst = target_root / "backup" / "warnings" / src.name
        _copy_file(src, dst)
        entries.append({"subsection": subsection, "tier": "backup", "artifact_type": "warnings", "path": str(dst), "metric_family": "market_actionable_example_weeks", "thesis_use": "warnings", "brief_description": f"Market-actionable example-week warnings: {src.name}."})
    figures_root = source_dir / "figures"
    if figures_root.exists():
        for src in sorted(figures_root.rglob("*.png")):
            rel = src.relative_to(figures_root)
            tier = "result_section" if rel.parts and rel.parts[0] in result_week_keys else "appendix"
            dst = target_root / tier / "figures" / rel
            _copy_file(src, dst)
            entries.append(
                {
                    "subsection": subsection,
                    "tier": tier,
                    "artifact_type": "figure",
                    "path": str(dst),
                    "metric_family": "example_weeks",
                    "thesis_use": "main thesis figure" if tier == "result_section" else "appendix figure",
                    "brief_description": f"Example-week forecast plot: {rel}",
                }
            )
        for src in sorted(figures_root.rglob("*.tex")):
            if _is_stale_week_alias(src):
                continue
            rel = src.relative_to(figures_root)
            if r"\includegraphics" in src.read_text(encoding="utf-8", errors="ignore"):
                continue
            tier = "result_section" if rel.parts and rel.parts[0] in result_week_keys else "appendix"
            dst = target_root / tier / "latex_figures" / rel
            _copy_file(src, dst)
            entries.append(
                {
                    "subsection": subsection,
                    "tier": tier,
                    "artifact_type": "latex_figure",
                    "path": str(dst),
                    "metric_family": "example_weeks",
                    "thesis_use": "native TikZ/pgfplots figure code",
                    "brief_description": f"Native example-week forecast plot code: {rel}",
                }
            )
    else:
        for tier in ["result_section", "appendix"]:
            for src in sorted((source_dir / tier / "figures").rglob("*.png")):
                rel = src.relative_to(source_dir / tier / "figures")
                dst = target_root / tier / "figures" / rel
                _copy_file(src, dst)
                entries.append(
                    {
                        "subsection": subsection,
                        "tier": tier,
                        "artifact_type": "figure",
                        "path": str(dst),
                        "metric_family": "example_weeks",
                        "thesis_use": "main thesis figure" if tier == "result_section" else "appendix figure",
                        "brief_description": f"Example-week forecast plot: {rel}",
                    }
                )
            for src in sorted((source_dir / tier / "latex_figures").rglob("*.tex")):
                if _is_stale_week_alias(src):
                    continue
                rel = src.relative_to(source_dir / tier / "latex_figures")
                if r"\includegraphics" in src.read_text(encoding="utf-8", errors="ignore"):
                    continue
                dst = target_root / tier / "latex_figures" / rel
                _copy_file(src, dst)
                entries.append(
                    {
                        "subsection": subsection,
                        "tier": tier,
                        "artifact_type": "latex_figure",
                        "path": str(dst),
                        "metric_family": "example_weeks",
                        "thesis_use": "native TikZ/pgfplots figure code",
                        "brief_description": f"Native example-week forecast plot code: {rel}",
                    }
                )

    for tier in ["result_section", "appendix"]:
        figures_dir = source_dir / tier / "figures"
        if figures_dir.exists():
            for src in sorted(figures_dir.rglob("*.png")):
                rel = src.relative_to(figures_dir)
                dst = target_root / tier / "figures" / rel
                _copy_file(src, dst)
                entries.append(
                    {
                        "subsection": subsection,
                        "tier": tier,
                        "artifact_type": "figure",
                        "path": str(dst),
                        "metric_family": "market_actionable_example_weeks",
                        "thesis_use": "main thesis figure" if tier == "result_section" else "appendix figure",
                        "brief_description": f"Market-actionable example-week forecast plot: {rel}",
                    }
                )
        latex_dir = source_dir / tier / "latex_figures"
        if latex_dir.exists():
            for src in sorted(latex_dir.rglob("*.tex")):
                if _is_stale_week_alias(src):
                    continue
                rel = src.relative_to(latex_dir)
                dst = target_root / tier / "latex_figures" / rel
                _copy_file(src, dst)
                entries.append(
                    {
                        "subsection": subsection,
                        "tier": tier,
                        "artifact_type": "latex_figure",
                        "path": str(dst),
                        "metric_family": "market_actionable_example_weeks",
                        "thesis_use": "main thesis figure" if tier == "result_section" else "appendix figure",
                        "brief_description": f"LaTeX includegraphics snippet for market-actionable example-week plot: {rel}",
                    }
                )
    _prune_target_stale_week_aliases()


def _latex_color_name(role: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", role).lower()


def _latex_color_defs() -> list[str]:
    return [
        f"\\definecolor{{{_latex_color_name(role)}}}{{HTML}}{{{hex_color.lstrip('#').upper()}}}"
        for role, hex_color in THESIS_PALETTE.items()
    ]


def _tex_num(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(x):
        return "nan"
    return f"{x:.6g}"


def _tex_hours(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(x):
        return "nan"
    return f"{x:.2f}"


def _tex_symbol(value: Any) -> str:
    raw = str(value)
    raw = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_")
    if not raw:
        raw = "x"
    if raw[0].isdigit():
        raw = "x_" + raw
    return raw


def _tex_label(value: Any) -> str:
    text = str(value).replace("\\", "/").replace("{", "(").replace("}", ")")
    text = text.replace("_", " ")
    replacements = {
        "afrr": "aFRR",
        "da": "DA",
        "id": "ID",
        "vwap": "VWAP",
        "p10": "P10",
        "p50": "P50",
        "p90": "P90",
        "rmse": "RMSE",
        "mae": "MAE",
        "rlqr": "RLQR",
        "xgb": "XGB",
        "tft": "TFT",
    }
    parts = [replacements.get(part.lower(), part) for part in text.split()]
    if parts and parts[-1].lower() == "pos":
        parts[-1] = "+"
    elif parts and parts[-1].lower() == "neg":
        parts[-1] = "-"
    return " ".join(parts)


def _sentence_case_caption(text: str) -> str:
    keep = {"DA", "ID", "VWAP", "RMSE", "MAE", "RLQR", "XGB", "TFT"}
    out_parts: list[str] = []
    sentence_start = True
    for token in text.split(" "):
        stripped = token.strip(".,;:()")
        if sentence_start or stripped in keep or stripped.startswith("aFRR"):
            out_parts.append(token)
        else:
            out_parts.append(token.lower())
        sentence_start = token.endswith((".", "?", "!"))
    out = " ".join(out_parts)
    return out[:1].upper() + out[1:] if out else out


BUCKET_LABELS = {
    "full_h1_48": "Full horizon (h1--h48)",
    "short_h1_8": "Short horizon (h1--h8)",
    "medium_h9_16": "Medium horizon (h9--h16)",
    "long_h17_48": "Long horizon (h17--h48)",
}


ACTIVATION_RELIABILITY_AGGREGATES = (
    ("afrr_activation_price_aggregate", "aFRR activation price"),
    ("afrr_activation_rate_aggregate", "aFRR activation rate"),
)


def _aggregate_activation_reliability(df: pd.DataFrame) -> pd.DataFrame:
    required = {"target_group", "model", "model_label", "quantile", "empirical_coverage", "n_obs"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for aggregate_slug, aggregate_label in ACTIVATION_RELIABILITY_AGGREGATES:
        part = df.loc[df["target_group"].eq(aggregate_label)].copy()
        if part.empty:
            continue
        part["n_obs"] = pd.to_numeric(part["n_obs"], errors="coerce")
        part["empirical_coverage"] = pd.to_numeric(part["empirical_coverage"], errors="coerce")
        part = part.loc[part["n_obs"].notna() & part["empirical_coverage"].notna() & (part["n_obs"] > 0)].copy()
        if part.empty:
            continue
        part["_weighted_empirical_coverage"] = part["empirical_coverage"] * part["n_obs"]
        grouped = (
            part.groupby(["model", "model_label", "quantile"], as_index=False)
            .agg(
                weighted_empirical_coverage=("_weighted_empirical_coverage", "sum"),
                n_obs=("n_obs", "sum"),
            )
        )
        grouped["empirical_coverage"] = grouped["weighted_empirical_coverage"] / grouped["n_obs"]
        grouped["target"] = aggregate_slug
        grouped["target_label"] = aggregate_label
        grouped["target_group"] = aggregate_label
        frames.append(grouped.drop(columns=["weighted_empirical_coverage"]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["target_label", "model_label", "quantile"])


def _bucket_label(bucket: Any) -> str:
    return BUCKET_LABELS.get(str(bucket), _tex_label(bucket))


def _actionable_label(bucket: Any, target_group: Any) -> str:
    bucket_s = str(bucket)
    group_s = str(target_group)
    if bucket_s == "actionable_da_dplus1_11":
        return "DA price D+1 at 11:00"
    if bucket_s == "actionable_bcm_dplus1_08":
        return "BCM capacity price D+1 at 08:00"
    if bucket_s == "actionable_bem_short_h1_8" and "rate" in group_s.lower():
        return "BEM activation rate h1-h8"
    if bucket_s == "actionable_bem_short_h1_8":
        return "BEM activation price h1-h8"
    return _bucket_label(bucket)


def _actionable_label_latex_multiline(label: Any) -> str:
    text = str(label)
    replacements = {
        "DA price D+1 at 11:00": ("DA price", "D+1 at 11:00"),
        "BCM capacity price D+1 at 08:00": ("BCM capacity price", "D+1 at 08:00"),
        "BEM activation price h1-h8": ("BEM activation price", "h1-h8"),
        "BEM activation rate h1-h8": ("BEM activation rate", "h1-h8"),
    }
    parts = replacements.get(text)
    if parts is None:
        return _latex_escape(text)
    return r"\shortstack[r]{" + r"\\".join(_latex_escape(part) for part in parts) + "}"


TAIL_SPIKE_REGIME_LABELS = {
    "normal": "Non-stress regime",
    "da_positive_spike_top5": "Positive spike top 5%",
    "da_negative_spike_bottom5": "Neg. spike bottom 5%",
    "afrr_activation_price_abs_tail_top5": "Abs. tail top 5%",
    "activation_nonzero": "Activation nonzero",
    "activation_zero_or_nearzero": "Activation (near-)zero",
    "high_volatility_week": "High-volatility week",
    "spike_week": "Spike week",
}

TAIL_SPIKE_MAIN_REGIMES_BY_TARGET = {
    "DA price": [
        "normal",
        "da_positive_spike_top5",
        "da_negative_spike_bottom5",
        "high_volatility_week",
        "spike_week",
    ],
    "aFRR capacity price +": [
        "normal",
        "high_volatility_week",
        "spike_week",
    ],
    "aFRR capacity price -": [
        "normal",
        "high_volatility_week",
        "spike_week",
    ],
    "aFRR activation price +": [
        "normal",
        "afrr_activation_price_abs_tail_top5",
        "high_volatility_week",
        "spike_week",
    ],
    "aFRR activation price -": [
        "normal",
        "afrr_activation_price_abs_tail_top5",
        "high_volatility_week",
        "spike_week",
    ],
    "aFRR activation rate +": [
        "activation_zero_or_nearzero",
        "activation_nonzero",
        "high_volatility_week",
        "spike_week",
    ],
    "aFRR activation rate -": [
        "activation_zero_or_nearzero",
        "activation_nonzero",
        "high_volatility_week",
        "spike_week",
    ],
}

TAIL_SPIKE_PRICE_CAPACITY_TARGETS = ["DA price", "aFRR capacity price +", "aFRR capacity price -"]
TAIL_SPIKE_ACTIVATION_TARGETS = [
    "aFRR activation price +",
    "aFRR activation price -",
    "aFRR activation rate +",
    "aFRR activation rate -",
]

P50_TOLERANCE_TEX_CONFIG = {
    "pred_da_price": ("da_price", "DA price", (1.0, 5.0, 10.0), "EUR/MWh"),
    "pred_afrr_capacity_price_pos": ("afrr_capacity_price_pos", "aFRR capacity price +", (1.0, 5.0, 10.0), "EUR/MW"),
    "pred_afrr_capacity_price_neg": ("afrr_capacity_price_neg", "aFRR capacity price $-$", (1.0, 5.0, 10.0), "EUR/MW"),
    "pred_afrr_activation_price_pos": ("afrr_activation_price_pos", "aFRR activation price +", (10.0, 50.0, 100.0), "EUR/MWh"),
    "pred_afrr_activation_price_neg": ("afrr_activation_price_neg", "aFRR activation price $-$", (10.0, 50.0, 100.0), "EUR/MWh"),
}


def _p50_tolerance_title_label(target: str) -> str:
    labels = {
        "pred_da_price": "DA Price",
        "pred_afrr_capacity_price_pos": "aFRR Capacity Price Positive",
        "pred_afrr_capacity_price_neg": "aFRR Capacity Price Negative",
        "pred_afrr_activation_price_pos": "aFRR Activation Price Positive",
        "pred_afrr_activation_price_neg": "aFRR Activation Price Negative",
    }
    return labels.get(str(target), _tex_label(target))


def _p50_tolerance_title(target: str) -> str:
    return f"Cumulative Absolute p50 Error Tolerance for {_p50_tolerance_title_label(target)}"


def _tail_spike_regime_label(regime: Any) -> str:
    return TAIL_SPIKE_REGIME_LABELS.get(str(regime), _tex_label(regime))


def _tail_spike_target_label(target_label: Any) -> str:
    text = str(target_label)
    if text.endswith(" -"):
        return text[:-2] + r" $-$"
    if text.endswith(" +"):
        return text
    return text


def _caption_from_name(stem: str) -> str:
    return stem.replace("_", " ").strip().capitalize() + "."


def _figure_label(section: str, stem: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return f"fig:rq1-{section.replace('.', '-')}-{clean}"


def _tikz_header(caption: str, label: str, *, placement: str = "htbp") -> list[str]:
    return [
        r"% Requires: \usepackage{pgfplots}",
        r"% Requires: \usepackage{xcolor}",
        r"% Recommended in preamble: \pgfplotsset{compat=1.18}",
        *_latex_color_defs(),
        rf"\begin{{figure}}[{placement}]",
        r"    \centering",
        r"    \resizebox{\linewidth}{!}{%",
        r"        \begin{tikzpicture}",
    ]


def _tikz_footer(caption: str, label: str, *, titlecase_caption: bool = True) -> list[str]:
    display_caption = _ensure_caption_period(thesis_titlecase(caption) if titlecase_caption else caption)
    return [
        r"        \end{tikzpicture}}",
        f"    \\caption{{{_latex_escape(display_caption)}}}",
        f"    \\label{{{label}}}",
        r"\end{figure}",
        "",
    ]


def _extract_tikzpicture_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if r"\begin{tikzpicture}" in line), None)
    if start is None:
        raise ValueError(f"Missing tikzpicture start in {path}")
    end = next((i for i in range(start, len(lines)) if r"\end{tikzpicture}" in lines[i]), None)
    if end is None:
        raise ValueError(f"Missing tikzpicture end in {path}")
    fragment = lines[start : end + 1]
    fragment[-1] = re.sub(r"(\\end\{tikzpicture\})\}+\s*$", r"\1", fragment[-1])
    return fragment


def _with_compact_axis_height(lines: list[str], height: str) -> list[str]:
    out: list[str] = []
    for line in lines:
        if re.search(r"\bheight=\d+(?:\.\d+)?cm,", line):
            out.append(re.sub(r"\bheight=\d+(?:\.\d+)?cm,", f"height={height},", line, count=1))
        else:
            out.append(line)
    return out


def _without_embedded_legend(lines: list[str]) -> list[str]:
    return [line for line in lines if r"\legend{" not in line]


def _with_combined_per_lead_layout(lines: list[str], filename: str) -> list[str]:
    out: list[str] = []
    is_afrr_panel = filename.startswith("per_lead_pinball_afrr_")
    is_activation_rate_panel = filename.startswith("per_lead_pinball_afrr_activation_rate")
    ylabel_xshift = "-0.42cm" if is_activation_rate_panel else "-0.25cm"
    for line in lines:
        if is_afrr_panel:
            line = line.replace("horizontal sep=1.25cm", "horizontal sep=2.05cm")
            line = line.replace("horizontal sep=1.65cm", "horizontal sep=2.05cm")
        if is_activation_rate_panel and "y label style={" in line:
            line = re.sub(r"at=\{\(-?[\d.]+,0\.5\)\}", "at={(-0.13,0.5)}", line, count=1)
        if "ylabel style={" in line:
            line = line.replace("ylabel style={", f"ylabel style={{xshift={ylabel_xshift}, ")
        elif "ylabel={" in line:
            out.append(f"                ylabel style={{xshift={ylabel_xshift}}},")
        out.append(line)
    return out


def _percent_tick_options(ticks: tuple[float, ...]) -> list[str]:
    tick_values = ",".join(_tex_num(tick) for tick in ticks)
    tick_labels = ",".join(rf"{tick * 100:.0f}\%" for tick in ticks)
    return [
        f"                xtick={{{tick_values}}},",
        f"                xticklabels={{{tick_labels}}},",
        f"                ytick={{{tick_values}}},",
        f"                yticklabels={{{tick_labels}}},",
    ]


def _write_lines(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _coordinates(rows: list[tuple[Any, Any]]) -> str:
    return " ".join(f"({_tex_symbol(x)},{_tex_num(y)})" for x, y in rows)


def _symbolic_axis_options(labels: list[Any], tick_labels_raw: list[Any] | None = None) -> list[str]:
    symbols = [_tex_symbol(x) for x in labels]
    tick_label_values = tick_labels_raw if tick_labels_raw is not None else labels
    tick_labels = [_latex_escape(_tex_label(x)) for x in tick_label_values]
    return [
        "                symbolic x coords={" + ",".join(symbols) + "},",
        "                xtick={" + ",".join(symbols) + "},",
        "                xticklabels={" + ",".join(tick_labels) + "},",
    ]


def _write_grouped_bar_tex(
    path: Path,
    *,
    data: pd.DataFrame,
    x_col: str,
    series_cols: list[str],
    caption: str,
    label: str,
    ylabel: str,
    colors: list[str] | None = None,
    placement: str = "htbp",
    reference_y: float | None = None,
    reference_label: str | None = None,
    x_tick_label_col: str | None = None,
    axis_height: str = "8cm",
    legend_style: str | None = None,
) -> Path | None:
    if data.empty or not series_cols:
        return None
    del colors
    series_cols = [c for c in ordered_model_labels(series_cols) if c in series_cols]
    labels = data[x_col].astype(str).tolist()
    tick_labels = data[x_tick_label_col].astype(str).tolist() if x_tick_label_col and x_tick_label_col in data.columns else None
    lines = _tikz_header(caption, label, placement=placement)
    lines.extend(
        [
            r"            \begin{axis}[",
            r"                ybar,",
            r"                bar width=13pt,",
            r"                width=0.96\textwidth,",
            rf"                height={axis_height},",
            rf"                ylabel={{{_latex_escape(_axis_label(ylabel))}}},",
            r"                x tick label style={rotate=35, anchor=east},",
            legend_style or r"                legend style={at={(0.5,1.08)}, anchor=south, legend columns=-1, draw=none, fill=none, text=black},",
            r"                legend cell align={left},",
            r"                area legend,",
            r"                axis lines*=left,",
            r"                ymin=0,",
            *_symbolic_axis_options(labels, tick_labels),
            r"            ]",
        ]
    )
    if reference_y is not None and labels:
        first = _tex_symbol(labels[0])
        last = _tex_symbol(labels[-1])
        ref_label = reference_label or f"{reference_y:g}"
        lines.append(rf"                \draw[color=secondary, densely dotted, line width=1.2pt, shorten <=-12mm, shorten >=-12mm] (axis cs:{first},{_tex_num(reference_y)}) -- (axis cs:{last},{_tex_num(reference_y)});")
        lines.append(r"                \addlegendimage{color=secondary, densely dotted, line width=1.2pt}")
        lines.append(rf"                \addlegendentry{{{_latex_escape(_tex_label(ref_label))}}}")
    for col in series_cols:
        if col not in data.columns:
            continue
        color = _model_color_role(col)
        rows = [(x, y) for x, y in zip(data[x_col], pd.to_numeric(data[col], errors="coerce"))]
        lines.append(rf"                \addplot[ybar, fill={color}, draw={color}, area legend] coordinates {{{_coordinates(rows)}}};")
        lines.append(rf"                \addlegendentry{{{_latex_escape(_tex_label(col))}}}")
    lines.extend([r"            \end{axis}", *_tikz_footer(caption, label)])
    return _write_lines(path, lines)


def _write_gate_bucket_relative_tex(
    path: Path,
    *,
    data: pd.DataFrame,
    caption: str,
    label: str,
    placement: str = "htbp",
) -> Path | None:
    required = {"target_group", "bucket", "model_label", "mean_pinball_loss"}
    if data.empty or not required.issubset(data.columns):
        return None
    d = data.copy()
    d = d[d["bucket_family"].eq("actionable")].copy() if "bucket_family" in d.columns else d
    d["mean_pinball_loss"] = pd.to_numeric(d["mean_pinball_loss"], errors="coerce")
    d = d.dropna(subset=["mean_pinball_loss"])
    if d.empty:
        return None
    d = sort_target_frame(d, target_col="target_group", extra_cols=["bucket", "model_label"])
    agg = (
        d.groupby(["target_group", "bucket", "model_label"], as_index=False, sort=False)
        .agg(mean_pinball_loss=("mean_pinball_loss", "mean"))
    )
    pivot = agg.pivot_table(index=["target_group", "bucket"], columns="model_label", values="mean_pinball_loss", aggfunc="mean").reset_index()
    if "RLQR" not in pivot.columns:
        return None
    denom = pd.to_numeric(pivot["RLQR"], errors="coerce")
    pivot = pivot.loc[denom.notna() & denom.abs().gt(1e-12)].copy()
    if pivot.empty:
        return None
    pivot["label"] = [_actionable_label(bucket, group) for group, bucket in zip(pivot["target_group"], pivot["bucket"])]
    label_order = {
        "DA price D+1 at 11:00": 0,
        "BCM capacity price D+1 at 08:00": 1,
        "BEM activation price h1-h8": 2,
        "BEM activation rate h1-h8": 3,
    }
    pivot["_order"] = pivot["label"].map(label_order).fillna(99)
    pivot = pivot.sort_values(["_order", "label"]).reset_index(drop=True)
    labels = pivot["label"].astype(str).tolist()
    y_symbols = [_tex_symbol(label) for label in labels]
    y_symbol_list = ",".join(y_symbols)
    y_tick_labels = ",".join(_actionable_label_latex_multiline(label) for label in labels)

    lines = [
        r"% Requires: \usepackage{pgfplots}",
        r"% Requires: \usepackage{xcolor}",
        r"% Recommended in preamble: \pgfplotsset{compat=1.18}",
        *_latex_color_defs(),
        rf"\begin{{figure}}[{placement}]",
        r"    \centering",
        r"    \resizebox{\linewidth}{!}{%",
        r"        \begin{tikzpicture}",
        r"            \begin{axis}[",
        r"                xbar,",
        r"                bar width=10pt,",
        r"                width=0.96\textwidth,",
        r"                height=6.1cm,",
        r"                xlabel={Mean pinball loss relative to RLQR},",
        r"                legend style={at={(0.5,1.16)}, anchor=south, legend columns=-1, draw=none, fill=none, text=black},",
        r"                legend cell align={left},",
        r"                area legend,",
        r"                axis lines*=left,",
        r"                yticklabel style={font=\small, align=right},",
        r"                xmin=0,",
        r"                grid=major,",
        rf"                symbolic y coords={{{y_symbol_list}}},",
        rf"                ytick={{{y_symbol_list}}},",
        rf"                yticklabels={{{y_tick_labels}}},",
        r"                y dir=reverse,",
        r"            ]",
    ]
    lines.append(rf"                \draw[color=secondary, densely dotted, line width=1.2pt, shorten <=-8mm, shorten >=-8mm] (axis cs:1,{y_symbols[0]}) -- (axis cs:1,{y_symbols[-1]});")
    lines.append(r"                \addlegendimage{color=secondary, densely dotted, line width=1.2pt}")
    lines.append(r"                \addlegendentry{RLQR}")
    for model in ["XGB", "TFT"]:
        if model not in pivot.columns:
            continue
        color = _model_color_role(model)
        coords: list[str] = []
        for _, row in pivot.iterrows():
            base = float(pd.to_numeric(pd.Series([row["RLQR"]]), errors="coerce").iloc[0])
            value = float(pd.to_numeric(pd.Series([row[model]]), errors="coerce").iloc[0])
            if np.isfinite(base) and abs(base) > 1e-12 and np.isfinite(value):
                coords.append(f"({_tex_num(value / base)},{_tex_symbol(row['label'])})")
        if coords:
            lines.append(rf"                \addplot[xbar, fill={color}, draw={color}, area legend] coordinates {{{' '.join(coords)}}};")
            lines.append(rf"                \addlegendentry{{{model}}}")
    lines.extend(
        [
            r"            \end{axis}",
            r"        \end{tikzpicture}}",
            f"    \\caption{{{_latex_escape(_ensure_caption_period(caption))}}}",
            f"    \\label{{{label}}}",
            r"\end{figure}",
            "",
        ]
    )
    return _write_lines(path, lines)


def _write_tail_spike_relative_tex(
    path: Path,
    *,
    data: pd.DataFrame,
    caption: str,
    label: str,
    placement: str = "htbp",
) -> Path | None:
    latex_dir = path.parent
    required = {"target_label", "regime", "model_label", "mean_pinball_loss"}
    if data.empty or not required.issubset(data.columns):
        return None

    thesis_root = "figures/4-results/rq1_ml_model_benchmark/4_1_5_tail_spike/result_section"
    regime_labels = {
        "normal": "Non-stress regime",
        "da_positive_spike_top5": "Positive spike top 5%",
        "da_negative_spike_bottom5": "Neg. spike bottom 5%",
        "afrr_capacity_price_high_tail_top5": "High tail top 5%",
        "afrr_activation_price_abs_tail_top5": "Abs. tail top 5%",
        "afrr_activation_rate_high_tail_top5": "High activation-rate tail top 5%",
        "activation_nonzero": "Activation nonzero",
        "activation_zero_or_nearzero": "Activation (near-)zero",
        "high_volatility_week": "High-volatility week",
        "spike_week": "Spike week",
    }
    main_regimes_by_target = {
        "DA price": [
            "normal",
            "da_positive_spike_top5",
            "da_negative_spike_bottom5",
            "high_volatility_week",
            "spike_week",
        ],
        "aFRR capacity price +": ["normal", "afrr_capacity_price_high_tail_top5", "high_volatility_week", "spike_week"],
        "aFRR capacity price -": ["normal", "afrr_capacity_price_high_tail_top5", "high_volatility_week", "spike_week"],
        "aFRR activation price +": ["normal", "afrr_activation_price_abs_tail_top5", "high_volatility_week", "spike_week"],
        "aFRR activation price -": ["normal", "afrr_activation_price_abs_tail_top5", "high_volatility_week", "spike_week"],
        "aFRR activation rate +": ["activation_zero_or_nearzero", "activation_nonzero", "high_volatility_week", "spike_week"],
        "aFRR activation rate -": ["activation_zero_or_nearzero", "activation_nonzero", "high_volatility_week", "spike_week"],
    }
    target_orders = {
        "da_price": ["DA price"],
        "price_capacity": ["DA price", "aFRR capacity price +", "aFRR capacity price -"],
        "activation": [
            "aFRR activation price +",
            "aFRR activation price -",
            "aFRR activation rate +",
            "aFRR activation rate -",
        ],
        "all_targets": [
            "DA price",
            "aFRR capacity price +",
            "aFRR capacity price -",
            "aFRR activation price",
            "aFRR activation rate",
        ],
    }
    aggregate_targets_by_key = {
        "all_targets": {
            "aFRR activation price": {
                "target_group": "aFRR activation price",
                "regimes": ["normal", "afrr_activation_price_abs_tail_top5", "high_volatility_week", "spike_week"],
            },
            "aFRR activation rate": {
                "target_group": "aFRR activation rate",
                "regimes": ["activation_zero_or_nearzero", "activation_nonzero", "high_volatility_week", "spike_week"],
            },
        }
    }
    aggregate_specs = {
        "capacity_price_aggregate": {
            "target_group": "aFRR capacity price",
            "label": "aFRR capacity price",
            "regimes": ["normal", "high_volatility_week", "spike_week"],
        },
        "activation_price_aggregate": {
            "target_group": "aFRR activation price",
            "label": "aFRR activation price",
            "regimes": ["normal", "afrr_activation_price_abs_tail_top5", "high_volatility_week", "spike_week"],
        },
        "activation_rate_aggregate": {
            "target_group": "aFRR activation rate",
            "label": "aFRR activation rate",
            "regimes": ["activation_zero_or_nearzero", "activation_nonzero", "high_volatility_week", "spike_week"],
        },
    }

    def _clean_regime_label(regime: Any) -> str:
        return regime_labels.get(str(regime), _tex_label(regime))

    def _compact_target_label_tex(target_label: Any) -> str:
        label = str(target_label)
        labels = {
            "DA price": r"\shortstack{DA\\price}",
            "aFRR capacity price": r"\shortstack{aFRR\\capacity\\price}",
            "aFRR capacity price +": r"\shortstack{aFRR\\capacity\\price +}",
            "aFRR capacity price -": r"\shortstack{aFRR\\capacity\\price $-$}",
            "aFRR activation price": r"\shortstack{aFRR\\activation\\prices(+/$-$)}",
            "aFRR activation price +": r"\shortstack{aFRR\\activation\\price +}",
            "aFRR activation price -": r"\shortstack{aFRR\\activation\\price $-$}",
            "aFRR activation rate": r"\shortstack{aFRR\\activation\\rate(+/$-$)}",
            "aFRR activation rate +": r"\shortstack{aFRR\\activation\\rate +}",
            "aFRR activation rate -": r"\shortstack{aFRR\\activation\\rate $-$}",
        }
        return labels.get(label, _latex_escape(label))

    def _write_native_relative_file(
        filename: str,
        *,
        target_key: str | None = None,
        aggregate_key: str | None = None,
        figure_caption: str,
        short_caption: str,
        figure_label: str,
        compact_target_sections: bool = False,
    ) -> Path | None:
        out = latex_dir / filename
        if target_key is not None:
            target_labels = target_orders[target_key]
            aggregate_targets = aggregate_targets_by_key.get(str(target_key), {})
            if aggregate_targets:
                raw_target_labels = [label for label in target_labels if label not in aggregate_targets]
                aggregate_target_groups = [str(spec["target_group"]) for spec in aggregate_targets.values()]
                d = data.loc[data["target_label"].isin(raw_target_labels) | data["target_group"].isin(aggregate_target_groups)].copy()
            else:
                d = data.loc[data["target_label"].isin(target_labels)].copy()
        elif aggregate_key is not None:
            spec = aggregate_specs[aggregate_key]
            d = data.loc[data["target_group"].eq(spec["target_group"])].copy()
            target_labels = [str(spec["label"])]
        else:
            return None
        d["mean_pinball_loss"] = pd.to_numeric(d["mean_pinball_loss"], errors="coerce")
        d = d.dropna(subset=["mean_pinball_loss"])
        if d.empty:
            return None

        def _model_regime_values(frame: pd.DataFrame) -> dict[str, float]:
            if "n_obs" in frame.columns:
                weighted = frame.copy()
                weighted["n_obs"] = pd.to_numeric(weighted["n_obs"], errors="coerce")
                weighted = weighted.loc[weighted["n_obs"].notna() & weighted["n_obs"].gt(0)].copy()
                if not weighted.empty:
                    weighted["_weighted_loss"] = weighted["mean_pinball_loss"] * weighted["n_obs"]
                    grouped = weighted.groupby("model_label", as_index=True).agg(
                        weighted_loss=("_weighted_loss", "sum"),
                        n_obs=("n_obs", "sum"),
                    )
                    grouped = grouped.loc[grouped["n_obs"].gt(0)].copy()
                    if not grouped.empty:
                        return (grouped["weighted_loss"] / grouped["n_obs"]).astype(float).to_dict()
            return frame.groupby("model_label", as_index=True)["mean_pinball_loss"].mean().to_dict()

        rows: list[dict[str, Any]] = []
        section_starts: list[int] = []
        if aggregate_key is not None:
            spec = aggregate_specs[aggregate_key]
            target_label = str(spec["label"])
            for regime in spec["regimes"]:
                regime_part = d.loc[d["regime"].eq(regime)]
                if regime_part.empty:
                    continue
                values = _model_regime_values(regime_part)
                baseline = values.get("RLQR")
                if baseline is None or not np.isfinite(float(baseline)) or abs(float(baseline)) <= 1e-12:
                    continue
                row_label = f"{_tex_label(target_label)} / {_clean_regime_label(regime)}"
                row: dict[str, Any] = {"label": row_label, "target_label": target_label, "regime_label": _clean_regime_label(regime)}
                for model in ["XGB", "TFT"]:
                    value = values.get(model)
                    if value is not None and np.isfinite(float(value)):
                        row[model] = float(value) / float(baseline)
                if "XGB" in row or "TFT" in row:
                    rows.append(row)
        else:
            aggregate_targets = aggregate_targets_by_key.get(str(target_key), {})
            for target_label in target_labels:
                target_spec = aggregate_targets.get(str(target_label))
                if target_spec is not None:
                    target_part = d.loc[d["target_group"].eq(target_spec["target_group"])].copy()
                    regimes = list(target_spec["regimes"])
                else:
                    target_part = d.loc[d["target_label"].eq(target_label)].copy()
                    regimes = main_regimes_by_target.get(target_label, target_part["regime"].dropna().astype(str).drop_duplicates().tolist())
                if target_part.empty:
                    continue
                target_section_start = len(rows)
                for regime in regimes:
                    regime_part = target_part.loc[target_part["regime"].eq(regime)]
                    if regime_part.empty:
                        continue
                    values = _model_regime_values(regime_part)
                    baseline = values.get("RLQR")
                    if baseline is None or not np.isfinite(float(baseline)) or abs(float(baseline)) <= 1e-12:
                        continue
                    regime_label = _clean_regime_label(regime)
                    row_label = f"{_tex_label(target_label)} / {regime_label}"
                    row: dict[str, Any] = {"label": row_label, "target_label": str(target_label), "regime_label": regime_label}
                    for model in ["XGB", "TFT"]:
                        value = values.get(model)
                        if value is not None and np.isfinite(float(value)):
                            row[model] = float(value) / float(baseline)
                    if "XGB" in row or "TFT" in row:
                        rows.append(row)
                if compact_target_sections and len(rows) > target_section_start:
                    section_starts.append(target_section_start)
        if not rows:
            return None

        use_all_target_compact_layout = filename == "tail_spike_relative_pinball_by_regime_all_targets.tex"
        labels = [str(row["label"]) for row in rows]
        y_indices = list(range(len(rows)))
        y_tick_values = ",".join(str(idx) for idx in y_indices)
        section_ranges: list[tuple[int, int, str]] = []
        if compact_target_sections:
            tick_label_parts: list[str] = []
            for idx, row in enumerate(rows):
                tick_label_parts.append(_latex_escape(row.get("regime_label", row["label"])))
            for pos, start_idx in enumerate(section_starts):
                next_start = section_starts[pos + 1] if pos + 1 < len(section_starts) else len(rows)
                section_ranges.append((start_idx, next_start - 1, str(rows[start_idx].get("target_label", ""))))
            y_tick_labels = ",".join(tick_label_parts)
        else:
            y_tick_labels = ",".join(_latex_escape(label) for label in labels)
        y_tick_labels = ",".join("{" + part + "}" for part in y_tick_labels.split(","))
        height_scale = 0.74 if use_all_target_compact_layout else 0.62
        height_padding = 2.6 if use_all_target_compact_layout else 2.0
        height_cm = max(7.8, min(29.0 if compact_target_sections else 16.0, height_scale * len(labels) + height_padding))
        max_x = max(
            [1.0]
            + [
                float(row[model])
                for row in rows
                for model in ["XGB", "TFT"]
                if model in row and np.isfinite(float(row[model]))
            ]
        )
        xmax = max(1.25, min(max_x * 1.12, max_x + 0.35))
        break_start = 2.0
        break_end = 3.5
        break_width = break_end - break_start
        values_for_break = [
            float(row[model])
            for row in rows
            for model in ["XGB", "TFT"]
            if model in row and np.isfinite(float(row[model]))
        ]
        compress_x_axis = (
            use_all_target_compact_layout
            and max_x > break_end
            and not any(break_start < value < break_end for value in values_for_break)
        )

        def _plot_x_value(value: Any) -> float:
            x = float(value)
            if compress_x_axis and x > break_end:
                return x - break_width
            return x

        plot_xmax = _plot_x_value(xmax) if compress_x_axis else xmax
        if compress_x_axis:
            tick_originals = [0.0, 1.0, 2.0]
            tick_originals.extend(float(tick) for tick in range(4, int(np.ceil(xmax)) + 1))
            tick_originals = [tick for tick in tick_originals if tick <= xmax + 1e-9]
            xtick_values = ",".join(_tex_num(_plot_x_value(tick)) for tick in tick_originals)
            xtick_labels = ",".join("{" + _tex_num(tick) + "}" for tick in tick_originals)
            ellipsis_tick = break_start + 0.25
        else:
            xtick_values = ""
            xtick_labels = ""
            ellipsis_tick = None
        table_names = {"XGB": r"\tailSpikeXGBTable", "TFT": r"\tailSpikeTFTTable"}
        model_table_lines: list[str] = []
        model_has_data: dict[str, bool] = {}
        for model in ["XGB", "TFT"]:
            table_rows = ["row,value\\\\"]
            for row_idx, row in enumerate(rows):
                value = row.get(model)
                if value is not None and np.isfinite(float(value)):
                    table_rows.append(f"{row_idx},{_tex_num(_plot_x_value(value))}\\\\")
            model_has_data[model] = len(table_rows) > 1
            if model_has_data[model]:
                model_table_lines.append(r"            \pgfplotstableread[col sep=comma, row sep=\\]{")
                model_table_lines.extend(f"                {line}" for line in table_rows)
                model_table_lines.append(rf"            }}{table_names[model]}")

        lines = [
            r"% Requires: \usepackage{pgfplots}",
            r"% Requires: \usepackage{pgfplotstable}",
            r"% Requires: \usepackage{xcolor}",
            r"% Requires: \usetikzlibrary{decorations.pathreplacing}",
            r"% Recommended in preamble: \pgfplotsset{compat=1.18}",
            *_latex_color_defs(),
            rf"\begin{{figure}}[{placement}]",
            r"    \centering",
            r"    \resizebox{\linewidth}{!}{%",
            r"        \begin{tikzpicture}",
            *model_table_lines,
            r"            \begin{axis}[",
            r"                xbar,",
            r"                bar width=7pt,",
            rf"                width={'0.74' if use_all_target_compact_layout else '0.98'}\textwidth,",
            rf"                height={_tex_num(height_cm)}cm,",
            r"                xlabel={Mean pinball loss relative to RLQR},",
            r"                legend style={at={(0.5,1.12)}, anchor=south, legend columns=-1, draw=none, fill=none, text=black},",
            r"                legend cell align={left},",
            r"                area legend,",
            r"                axis lines*=left,",
            r"                xmin=0,",
            rf"                xmax={_tex_num(plot_xmax)},",
            r"                grid=major,",
            r"                clip=false,",
            r"                ymin=0,",
            rf"                ymax={len(rows) - 1},",
            rf"                ytick={{{y_tick_values}}},",
            rf"                yticklabels={{{y_tick_labels}}},",
            rf"                yticklabel style={{font=\{'small' if use_all_target_compact_layout else 'scriptsize'}, align=right{', xshift=0.18cm' if use_all_target_compact_layout else ''}}},",
            rf"                enlarge y limits={{abs={'0.45' if compact_target_sections else '0.28'}}},",
            r"                y dir=reverse,",
            r"            ]",
            rf"                \draw[color=secondary, densely dotted, line width=1.2pt, shorten <=-8mm, shorten >=-8mm] (axis cs:1,0) -- (axis cs:1,{len(rows) - 1});",
            r"                \addlegendimage{color=secondary, densely dotted, line width=1.2pt}",
            r"                \addlegendentry{RLQR}",
        ]
        if compress_x_axis:
            axis_option_insert_at = lines.index(r"                y dir=reverse,")
            lines[axis_option_insert_at:axis_option_insert_at] = [
                rf"                xtick={{{xtick_values}}},",
                rf"                xticklabels={{{xtick_labels}}},",
                rf"                extra x ticks={{{_tex_num(ellipsis_tick)}}},",
                r"                extra x tick labels={{$\cdots$}},",
                r"                extra x tick style={grid=none, tick style={draw=none}, xticklabel style={font=\small, yshift=-0.4ex}},",
            ]
            lines.append(
                rf"                \node[anchor=south, font=\scriptsize, text=neutraldark] at (axis cs:{_tex_num(ellipsis_tick)},-0.55) {{axis break: {_tex_num(break_start)}--{_tex_num(break_end)}}};"
            )
            lines.append(
                rf"                \draw[color=neutraldark, line width=0.55pt] ([xshift=-3pt,yshift=-3pt]axis cs:{_tex_num(ellipsis_tick)},{len(rows) - 1}) -- ([xshift=1pt,yshift=3pt]axis cs:{_tex_num(ellipsis_tick)},{len(rows) - 1});"
            )
            lines.append(
                rf"                \draw[color=neutraldark, line width=0.55pt] ([xshift=2pt,yshift=-3pt]axis cs:{_tex_num(ellipsis_tick)},{len(rows) - 1}) -- ([xshift=6pt,yshift=3pt]axis cs:{_tex_num(ellipsis_tick)},{len(rows) - 1});"
            )
        if compact_target_sections:
            use_compact_target_labels = use_all_target_compact_layout
            target_label_xshift = "-4.75cm" if use_compact_target_labels else "-4.55cm"
            brace_xshift = "-4.22cm" if use_compact_target_labels else "-1.05cm"
            target_label_font = r"\small" if use_compact_target_labels else r"\scriptsize"
            for start_idx, end_idx, target_label in section_ranges:
                y_min = start_idx - 0.38
                y_max = end_idx + 0.38
                y_mid = (start_idx + end_idx) / 2.0
                target_label_tex = _compact_target_label_tex(target_label) if use_compact_target_labels else _latex_escape(target_label)
                lines.append(
                    rf"                \draw[decorate, decoration={{brace, amplitude=4pt, mirror}}, color=neutraldark, line width=0.55pt] ([xshift={brace_xshift}]axis cs:0,{_tex_num(y_min)}) -- ([xshift={brace_xshift}]axis cs:0,{_tex_num(y_max)});"
                )
                lines.append(
                    rf"                \node[rotate=90, anchor=center, text=neutraldark, font={target_label_font}] at ([xshift={target_label_xshift}]axis cs:0,{_tex_num(y_mid)}) {{{target_label_tex}}};"
                )
            for start_idx in section_starts[1:]:
                lines.append(
                    rf"                \draw[color=naive, dashed, line width=0.6pt] ([xshift=-1.2cm]axis cs:0,{_tex_num(start_idx - 0.5)}) -- (axis cs:{_tex_num(plot_xmax)},{_tex_num(start_idx - 0.5)});"
                )
        bar_shift_by_model = {"XGB": "-1.75pt", "TFT": "1.75pt"}
        for model in ["XGB", "TFT"]:
            if model_has_data.get(model, False):
                color = _model_color_role(model)
                bar_shift = bar_shift_by_model.get(model, "0pt")
                lines.append(rf"                \addplot[xbar, bar shift={bar_shift}, fill={color}, draw={color}, fill opacity=1, draw opacity=1, area legend] table[x=value, y=row] {{{table_names[model]}}};")
                lines.append(rf"                \addlegendentry{{{model}}}")
        lines.extend(
            [
            r"            \end{axis}",
            r"        \end{tikzpicture}}",
            f"    \\caption[{_latex_escape(_ensure_caption_period(short_caption))}]{{{_latex_escape(_ensure_caption_period(figure_caption))}}}",
            f"    \\label{{{figure_label}}}",
            r"\end{figure}",
            "",
            ]
        )
        return _write_lines(out, lines)

    def _model_values_from_metric_frame(frame: pd.DataFrame) -> dict[str, float]:
        if "n_obs" in frame.columns:
            weighted = frame.copy()
            weighted["n_obs"] = pd.to_numeric(weighted["n_obs"], errors="coerce")
            weighted = weighted.loc[weighted["n_obs"].notna() & weighted["n_obs"].gt(0)].copy()
            if not weighted.empty:
                weighted["_weighted_loss"] = weighted["mean_pinball_loss"] * weighted["n_obs"]
                grouped = weighted.groupby("model_label", as_index=True).agg(
                    weighted_loss=("_weighted_loss", "sum"),
                    n_obs=("n_obs", "sum"),
                )
                grouped = grouped.loc[grouped["n_obs"].gt(0)].copy()
                if not grouped.empty:
                    return (grouped["weighted_loss"] / grouped["n_obs"]).astype(float).to_dict()
        return frame.groupby("model_label", as_index=True)["mean_pinball_loss"].mean().to_dict()

    def _metric_relative_row(
        *,
        target_label: str,
        regime_label: str,
        frame: pd.DataFrame,
    ) -> dict[str, Any] | None:
        if frame.empty:
            return None
        values = _model_values_from_metric_frame(frame)
        baseline = values.get("RLQR")
        if baseline is None or not np.isfinite(float(baseline)) or abs(float(baseline)) <= 1e-12:
            return None
        row: dict[str, Any] = {"label": f"{_tex_label(target_label)} / {regime_label}", "target_label": target_label, "regime_label": regime_label}
        for model in ["XGB", "TFT"]:
            value = values.get(model)
            if value is not None and np.isfinite(float(value)):
                row[model] = float(value) / float(baseline)
        return row if ("XGB" in row or "TFT" in row) else None

    def _tail_spike_points_path() -> Path | None:
        section_root = latex_dir.parent.parent
        rq1_root = section_root.parent
        candidates = [
            section_root / "backup" / "csv" / "tail_spike_points_test.csv",
            rq1_root / "_raw_outputs" / "4_1_5_tail_spike" / "tail_spike_points_test.csv",
            rq1_root / "_raw_outputs" / "shared" / "tail_spike_points_test.csv",
        ]
        return next((candidate for candidate in candidates if candidate.exists()), None)

    def _point_relative_row(
        points: pd.DataFrame,
        *,
        target_label: str,
        regime_label: str,
        regimes: set[str],
    ) -> dict[str, Any] | None:
        qcols = sorted([col for col in points.columns if re.fullmatch(r"p\d{1,2}", str(col))], key=lambda col: int(str(col)[1:]))
        required_cols = {"model_label", "target", "forecast_time_utc", "target_time_utc", "lead_time_h", "y_true", *qcols}
        if points.empty or not qcols or not required_cols.issubset(points.columns):
            return None
        d = points.loc[points["regime"].astype(str).isin(regimes)].copy()
        if d.empty:
            return None
        key_cols = ["target", "forecast_time_utc", "target_time_utc", "lead_time_h"]
        numeric_cols = ["y_true", *qcols]
        for col in numeric_cols:
            d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d.dropna(subset=["model_label", *key_cols, *numeric_cols]).copy()
        d = d.drop_duplicates(["model_label", *key_cols], keep="last")
        if d.empty:
            return None
        key_sets = {
            str(model): set(map(tuple, part[key_cols].itertuples(index=False, name=None)))
            for model, part in d.groupby("model_label", sort=False)
        }
        common_models = [model for model in ["RLQR", "XGB", "TFT"] if model in key_sets]
        if len(common_models) < 3:
            return None
        common_keys = set.intersection(*(key_sets[model] for model in common_models))
        if not common_keys:
            return None
        key_df = pd.DataFrame(list(common_keys), columns=key_cols)
        values: dict[str, float] = {}
        for model in common_models:
            part = d.loc[d["model_label"].astype(str).eq(model)].merge(key_df, on=key_cols, how="inner")
            if part.empty:
                continue
            y = pd.to_numeric(part["y_true"], errors="coerce").to_numpy(dtype=float)
            losses: list[np.ndarray] = []
            for qcol in qcols:
                q = int(str(qcol)[1:]) / 100.0
                pred = pd.to_numeric(part[qcol], errors="coerce").to_numpy(dtype=float)
                err = y - pred
                losses.append(np.maximum(q * err, (q - 1.0) * err))
            values[model] = float(np.mean(np.vstack(losses))) if losses else float("nan")
        baseline = values.get("RLQR")
        if baseline is None or not np.isfinite(float(baseline)) or abs(float(baseline)) <= 1e-12:
            return None
        row: dict[str, Any] = {"label": f"{target_label} / {regime_label}", "target_label": target_label, "regime_label": regime_label}
        for model in ["XGB", "TFT"]:
            value = values.get(model)
            if value is not None and np.isfinite(float(value)):
                row[model] = float(value) / float(baseline)
        return row if ("XGB" in row or "TFT" in row) else None

    def _point_spec_mask(chunk: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
        mask = chunk["regime"].astype(str).isin(set(spec["regimes"]))
        if "target_label" in spec:
            mask &= chunk["target_label"].astype(str).eq(str(spec["target_label"]))
        if "target_group" in spec:
            mask &= chunk["target_group"].astype(str).eq(str(spec["target_group"]))
        return mask

    def _point_relative_rows_for_specs(specs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        points_path = _tail_spike_points_path()
        if points_path is None or not specs:
            return {}
        usecols = [
            "forecast_time_utc",
            "target_time_utc",
            "lead_time_h",
            "y_true",
            "p10",
            "p30",
            "p50",
            "p70",
            "p90",
            "model_label",
            "target",
            "target_label",
            "regime",
            "target_group",
        ]
        chunks_by_id: dict[str, list[pd.DataFrame]] = {str(spec["id"]): [] for spec in specs}
        for chunk in pd.read_csv(points_path, usecols=lambda col: col in usecols, chunksize=500_000):
            if not {"target_label", "target_group", "regime"}.issubset(chunk.columns):
                continue
            for spec in specs:
                part = chunk.loc[_point_spec_mask(chunk, spec)].copy()
                if not part.empty:
                    chunks_by_id[str(spec["id"])].append(part)
        rows: dict[str, dict[str, Any]] = {}
        for spec in specs:
            spec_id = str(spec["id"])
            parts = chunks_by_id.get(spec_id, [])
            if not parts:
                continue
            row = _point_relative_row(
                pd.concat(parts, ignore_index=True),
                target_label=str(spec["target_label_out"]),
                regime_label=str(spec["regime_label"]),
                regimes=set(spec["regimes"]),
            )
            if row is not None:
                rows[spec_id] = row
        return rows

    def _write_rows_xbar_file(
        filename: str,
        *,
        rows: list[dict[str, Any]],
        section_starts: list[int],
        figure_caption: str,
        short_caption: str,
        figure_label: str,
    ) -> Path | None:
        if not rows:
            return None
        out = latex_dir / filename
        labels = [str(row["label"]) for row in rows]
        y_tick_values = ",".join(str(idx) for idx in range(len(rows)))
        y_tick_labels = ",".join("{" + _latex_escape(row.get("regime_label", row["label"])) + "}" for row in rows)
        section_ranges: list[tuple[int, int, str]] = []
        for pos, start_idx in enumerate(section_starts):
            next_start = section_starts[pos + 1] if pos + 1 < len(section_starts) else len(rows)
            section_ranges.append((start_idx, next_start - 1, str(rows[start_idx].get("target_label", ""))))
        max_x = max(
            [1.0]
            + [
                float(row[model])
                for row in rows
                for model in ["XGB", "TFT"]
                if model in row and np.isfinite(float(row[model]))
            ]
        )
        xmax = max(1.25, min(max_x * 1.12, max_x + 0.35))
        break_start = 2.0
        break_end = 3.5
        break_width = break_end - break_start
        values_for_break = [
            float(row[model])
            for row in rows
            for model in ["XGB", "TFT"]
            if model in row and np.isfinite(float(row[model]))
        ]
        compress_x_axis = max_x > break_end and not any(break_start < value < break_end for value in values_for_break)

        def _plot_x_value(value: Any) -> float:
            x = float(value)
            if compress_x_axis and x > break_end:
                return x - break_width
            return x

        plot_xmax = _plot_x_value(xmax) if compress_x_axis else xmax
        if compress_x_axis:
            tick_originals = [0.0, 1.0, 2.0]
            tick_originals.extend(float(tick) for tick in range(4, int(np.ceil(xmax)) + 1))
            tick_originals = [tick for tick in tick_originals if tick <= xmax + 1e-9]
            xtick_values = ",".join(_tex_num(_plot_x_value(tick)) for tick in tick_originals)
            xtick_labels = ",".join("{" + _tex_num(tick) + "}" for tick in tick_originals)
            ellipsis_tick = break_start + 0.25
        else:
            xtick_values = ""
            xtick_labels = ""
            ellipsis_tick = None
        height_cm = max(7.8, min(12.9, 0.74 * len(labels) + 2.6))
        table_names = {"XGB": r"\tailSpikeXGBTable", "TFT": r"\tailSpikeTFTTable"}
        model_table_lines: list[str] = []
        model_has_data: dict[str, bool] = {}
        for model in ["XGB", "TFT"]:
            table_rows = ["row,value\\\\"]
            for row_idx, row in enumerate(rows):
                value = row.get(model)
                if value is not None and np.isfinite(float(value)):
                    table_rows.append(f"{row_idx},{_tex_num(_plot_x_value(value))}\\\\")
            model_has_data[model] = len(table_rows) > 1
            if model_has_data[model]:
                model_table_lines.append(r"            \pgfplotstableread[col sep=comma, row sep=\\]{")
                model_table_lines.extend(f"                {line}" for line in table_rows)
                model_table_lines.append(rf"            }}{table_names[model]}")
        lines = [
            r"% Requires: \usepackage{pgfplots}",
            r"% Requires: \usepackage{pgfplotstable}",
            r"% Requires: \usepackage{xcolor}",
            r"% Requires: \usetikzlibrary{decorations.pathreplacing}",
            r"% Recommended in preamble: \pgfplotsset{compat=1.18}",
            *_latex_color_defs(),
            rf"\begin{{figure}}[{placement}]",
            r"    \centering",
            r"    \resizebox{\linewidth}{!}{%",
            r"        \begin{tikzpicture}",
            *model_table_lines,
            r"            \begin{axis}[",
            r"                xbar,",
            r"                bar width=7pt,",
            r"                width=0.74\textwidth,",
            rf"                height={_tex_num(height_cm)}cm,",
            r"                xlabel={Mean pinball loss relative to RLQR},",
            r"                legend style={at={(0.5,1.07)}, anchor=south, legend columns=-1, draw=none, fill=none, text=black},",
            r"                legend cell align={left},",
            r"                area legend,",
            r"                axis lines*=left,",
            r"                xmin=0,",
            rf"                xmax={_tex_num(plot_xmax)},",
            r"                grid=major,",
            r"                clip=false,",
            r"                ymin=0,",
            rf"                ymax={len(rows) - 1},",
            rf"                ytick={{{y_tick_values}}},",
            rf"                yticklabels={{{y_tick_labels}}},",
            r"                yticklabel style={font=\small, align=right, xshift=0.18cm},",
            r"                enlarge y limits={abs=0.45},",
            r"                y dir=reverse,",
            r"            ]",
            rf"                \draw[color=secondary, densely dotted, line width=1.2pt, shorten <=-8mm, shorten >=-8mm] (axis cs:1,0) -- (axis cs:1,{len(rows) - 1});",
            r"                \addlegendimage{color=secondary, densely dotted, line width=1.2pt}",
            r"                \addlegendentry{RLQR}",
        ]
        if compress_x_axis:
            axis_option_insert_at = lines.index(r"                y dir=reverse,")
            lines[axis_option_insert_at:axis_option_insert_at] = [
                rf"                xtick={{{xtick_values}}},",
                rf"                xticklabels={{{xtick_labels}}},",
                rf"                extra x ticks={{{_tex_num(ellipsis_tick)}}},",
                r"                extra x tick labels={{$\cdots$}},",
                r"                extra x tick style={grid=none, tick style={draw=none}, xticklabel style={font=\small, yshift=-0.4ex}},",
            ]
            lines.append(
                rf"                \node[anchor=south, font=\scriptsize, text=neutraldark] at (axis cs:{_tex_num(ellipsis_tick)},-0.55) {{axis break: {_tex_num(break_start)}--{_tex_num(break_end)}}};"
            )
            lines.append(
                rf"                \draw[color=neutraldark, line width=0.55pt] ([xshift=-3pt,yshift=-3pt]axis cs:{_tex_num(ellipsis_tick)},{len(rows) - 1}) -- ([xshift=1pt,yshift=3pt]axis cs:{_tex_num(ellipsis_tick)},{len(rows) - 1});"
            )
            lines.append(
                rf"                \draw[color=neutraldark, line width=0.55pt] ([xshift=2pt,yshift=-3pt]axis cs:{_tex_num(ellipsis_tick)},{len(rows) - 1}) -- ([xshift=6pt,yshift=3pt]axis cs:{_tex_num(ellipsis_tick)},{len(rows) - 1});"
            )
        for start_idx, end_idx, target_label in section_ranges:
            y_min = start_idx - 0.38
            y_max = end_idx + 0.38
            y_mid = (start_idx + end_idx) / 2.0
            lines.append(
                rf"                \draw[decorate, decoration={{brace, amplitude=4pt, mirror}}, color=neutraldark, line width=0.55pt] ([xshift=-4.42cm]axis cs:0,{_tex_num(y_min)}) -- ([xshift=-4.42cm]axis cs:0,{_tex_num(y_max)});"
            )
            lines.append(
                rf"                \node[rotate=90, anchor=center, text=neutraldark, font=\small] at ([xshift=-5.35cm]axis cs:0,{_tex_num(y_mid)}) {{{_compact_target_label_tex(target_label)}}};"
            )
        for start_idx in section_starts[1:]:
            lines.append(
                rf"                \draw[color=naive, dashed, line width=0.6pt] ([xshift=-1.2cm]axis cs:0,{_tex_num(start_idx - 0.5)}) -- (axis cs:{_tex_num(plot_xmax)},{_tex_num(start_idx - 0.5)});"
            )
        bar_shift_by_model = {"XGB": "-1.75pt", "TFT": "1.75pt"}
        for model in ["XGB", "TFT"]:
            if model_has_data.get(model, False):
                color = _model_color_role(model)
                lines.append(rf"                \addplot[xbar, bar shift={bar_shift_by_model[model]}, fill={color}, draw={color}, fill opacity=1, draw opacity=1, area legend] table[x=value, y=row] {{{table_names[model]}}};")
                lines.append(rf"                \addlegendentry{{{model}}}")
        lines.extend(
            [
                r"            \end{axis}",
                r"        \end{tikzpicture}}",
                f"    \\caption[{_latex_escape(_ensure_caption_period(short_caption))}]{{{_latex_escape(_ensure_caption_period(figure_caption))}}}",
                f"    \\label{{{figure_label}}}",
                r"\end{figure}",
                "",
            ]
        )
        return _write_lines(out, lines)

    def _write_reduced_da_all_targets_file() -> Path | None:
        rows: list[dict[str, Any]] = []
        section_starts: list[int] = []
        point_rows = _point_relative_rows_for_specs(
            [
                {
                    "id": "da_positive_spike",
                    "target_label": "DA price",
                    "target_label_out": "DA price",
                    "regime_label": "Positive spike top 5%",
                    "regimes": {"da_positive_spike_top5"},
                },
                {
                    "id": "da_negative_spike",
                    "target_label": "DA price",
                    "target_label_out": "DA price",
                    "regime_label": "Neg. spike bottom 5%",
                    "regimes": {"da_negative_spike_bottom5"},
                },
                {
                    "id": "da_stress_week",
                    "target_label": "DA price",
                    "target_label_out": "DA price",
                    "regime_label": "Stress regime",
                    "regimes": {"high_volatility_week", "spike_week"},
                },
                {
                    "id": "capacity_pos_stress_week",
                    "target_label": "aFRR capacity price +",
                    "target_label_out": "aFRR capacity price +",
                    "regime_label": "Stress regime",
                    "regimes": {"high_volatility_week", "spike_week"},
                },
                {
                    "id": "capacity_neg_stress_week",
                    "target_label": "aFRR capacity price -",
                    "target_label_out": "aFRR capacity price -",
                    "regime_label": "Stress regime",
                    "regimes": {"high_volatility_week", "spike_week"},
                },
                {
                    "id": "activation_price_stress_week",
                    "target_group": "aFRR activation price",
                    "target_label_out": "aFRR activation price",
                    "regime_label": "Stress regime",
                    "regimes": {"high_volatility_week", "spike_week"},
                },
                {
                    "id": "activation_rate_stress_week",
                    "target_group": "aFRR activation rate",
                    "target_label_out": "aFRR activation rate",
                    "regime_label": "Stress regime",
                    "regimes": {"high_volatility_week", "spike_week"},
                },
            ]
        )
        da_start = len(rows)
        da_normal = _metric_relative_row(
            target_label="DA price",
            regime_label="Non-stress regime",
            frame=data.loc[data["target_label"].eq("DA price") & data["regime"].eq("normal")].copy(),
        )
        if da_normal is not None:
            rows.append(da_normal)
        for spec_id in ["da_positive_spike", "da_negative_spike", "da_stress_week"]:
            row = point_rows.get(spec_id)
            if row is not None:
                rows.append(row)
        if len(rows) > da_start:
            section_starts.append(da_start)
        target_specs = [
            ("aFRR capacity price +", data["target_label"].eq("aFRR capacity price +"), ["normal", "afrr_capacity_price_high_tail_top5"], "capacity_pos_stress_week"),
            ("aFRR capacity price -", data["target_label"].eq("aFRR capacity price -"), ["normal", "afrr_capacity_price_high_tail_top5"], "capacity_neg_stress_week"),
            ("aFRR activation price", data["target_group"].eq("aFRR activation price"), ["normal", "afrr_activation_price_abs_tail_top5"], "activation_price_stress_week"),
            ("aFRR activation rate", data["target_group"].eq("aFRR activation rate"), ["activation_zero_or_nearzero", "activation_nonzero"], "activation_rate_stress_week"),
        ]
        for target_label, mask, regimes, stress_spec_id in target_specs:
            start_idx = len(rows)
            for regime in regimes:
                frame = data.loc[mask & data["regime"].eq(regime)].copy()
                row = _metric_relative_row(target_label=target_label, regime_label=_clean_regime_label(regime), frame=frame)
                if row is not None:
                    rows.append(row)
            stress_row = point_rows.get(stress_spec_id)
            if stress_row is not None:
                rows.append(stress_row)
            if len(rows) > start_idx:
                section_starts.append(start_idx)
        return _write_rows_xbar_file(
            "tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.tex",
            rows=rows,
            section_starts=section_starts,
            figure_caption="Tail and spike performance across forecast targets. The stress regime combines target-specific high-volatility weeks, defined as the top 10% of weeks by weekly realized standard deviation, and spike-event weeks, defined as the top 10% of weeks by target-specific spike-event share; overlapping hours are counted once.",
            short_caption="Tail and spike performance across forecast targets",
            figure_label="fig:tail_spike_relative_pinball_all_targets_da_aggregated",
        )

    written: list[Path] = []
    da_price = _write_native_relative_file(
        "tail_spike_relative_pinball_by_regime_da_price.tex",
        target_key="da_price",
        figure_caption="Tail and spike performance for DA price forecasts. Bars show mean pinball loss relative to RLQR across normal, positive spike, negative spike, high-volatility week and spike-week regimes. Values below 1 indicate lower loss than RLQR, while values above 1 indicate worse performance.",
        short_caption="Tail and spike performance: DA price",
        figure_label="fig:tail_spike_relative_pinball_da_price",
    )
    if da_price is not None and da_price.exists():
        written.append(da_price)
    price_capacity = _write_native_relative_file(
        "tail_spike_relative_pinball_by_regime_price_capacity.tex",
        target_key="price_capacity",
        figure_caption="Tail and spike performance for DA and aFRR capacity price forecasts. Bars show mean pinball loss relative to RLQR. Values below 1 indicate lower loss than RLQR, while values above 1 indicate worse performance.",
        short_caption="Tail and spike performance: DA and aFRR capacity price",
        figure_label="fig:tail_spike_relative_pinball_price_capacity",
    )
    if price_capacity is not None and price_capacity.exists():
        written.append(price_capacity)
    activation = _write_native_relative_file(
        "tail_spike_relative_pinball_by_regime_activation.tex",
        target_key="activation",
        figure_caption="Tail and spike performance for aFRR activation forecasts. Bars show mean pinball loss relative to RLQR. Values below 1 indicate lower loss than RLQR, while values above 1 indicate worse performance.",
        short_caption="Tail and spike performance: aFRR activation",
        figure_label="fig:tail_spike_relative_pinball_activation",
    )
    if activation is not None and activation.exists():
        written.append(activation)
    all_targets = _write_native_relative_file(
        path.name,
        target_key="all_targets",
        figure_caption="Tail and spike performance across forecast targets. Bars show mean pinball loss relative to RLQR; values below 1 indicate lower loss.",
        short_caption="Tail and spike performance across forecast targets",
        figure_label="fig:tail_spike_relative_pinball_all_targets",
        compact_target_sections=True,
    )
    if all_targets is not None and all_targets.exists():
        written.append(all_targets)
    all_targets_copy = _write_native_relative_file(
        "tail_spike_relative_pinball_by_regime_all_targets.tex",
        target_key="all_targets",
        figure_caption="Tail and spike performance across forecast targets. Bars show mean pinball loss relative to RLQR; values below 1 indicate lower loss.",
        short_caption="Tail and spike performance across forecast targets",
        figure_label="fig:tail_spike_relative_pinball_all_targets",
        compact_target_sections=True,
    )
    if all_targets_copy is not None and all_targets_copy.exists():
        written.append(all_targets_copy)
    stable_three_regime_wrapper = path.parent / "tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.tex"
    stable_three_regime_png = path.parent.parent / "figures" / "tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.png"
    if stable_three_regime_png.exists() and stable_three_regime_wrapper.exists():
        written.append(stable_three_regime_wrapper)
    else:
        reduced_da_all_targets = _write_reduced_da_all_targets_file()
        if reduced_da_all_targets is not None and reduced_da_all_targets.exists():
            written.append(reduced_da_all_targets)
    for aggregate_key, filename, figure_caption, short_caption, figure_label in [
        (
            "capacity_price_aggregate",
            "tail_spike_relative_pinball_by_regime_capacity_price_aggregate.tex",
            "Tail and spike performance for aFRR capacity price forecasts with positive and negative directions aggregated. Bars show mean pinball loss relative to RLQR. Values below 1 indicate lower loss than RLQR, while values above 1 indicate worse performance.",
            "Tail and spike performance: aggregated aFRR capacity price",
            "fig:tail_spike_relative_pinball_capacity_price_aggregate",
        ),
        (
            "activation_price_aggregate",
            "tail_spike_relative_pinball_by_regime_activation_price_aggregate.tex",
            "Tail and spike performance for aFRR activation price forecasts with positive and negative directions aggregated. Bars show mean pinball loss relative to RLQR. Values below 1 indicate lower loss than RLQR, while values above 1 indicate worse performance.",
            "Tail and spike performance: aggregated aFRR activation price",
            "fig:tail_spike_relative_pinball_activation_price_aggregate",
        ),
        (
            "activation_rate_aggregate",
            "tail_spike_relative_pinball_by_regime_activation_rate_aggregate.tex",
            "Tail and spike performance for aFRR activation rate forecasts with positive and negative directions aggregated. Bars show mean pinball loss relative to RLQR. Values below 1 indicate lower loss than RLQR, while values above 1 indicate worse performance.",
            "Tail and spike performance: aggregated aFRR activation rate",
            "fig:tail_spike_relative_pinball_activation_rate_aggregate",
        ),
    ]:
        _write_native_relative_file(
            filename,
            aggregate_key=aggregate_key,
            figure_caption=figure_caption,
            short_caption=short_caption,
            figure_label=figure_label,
        )
    if all_targets is not None and all_targets.exists():
        return all_targets
    if not written:
        return None

    wrapper_lines = [
        rf"\input{{{thesis_root}/latex_figures/{p.name}}}"
        for p in written
    ]
    wrapper_lines.append("")
    return _write_lines(path, wrapper_lines)


def _model_color_role(model: Any) -> str:
    key = str(model).strip().lower()
    if key in {"tft"}:
        return "tertiary"
    if key in {"xgb", "xgboost"}:
        return "primary"
    if key in {"linear", "rlqr"}:
        return "secondary"
    return "neutraldark"


def _write_model_bar_tex(
    path: Path,
    *,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    model_col: str,
    caption: str,
    label: str,
    ylabel: str,
    placement: str = "htbp",
) -> Path | None:
    if data.empty or x_col not in data.columns or y_col not in data.columns:
        return None
    d = data.copy()
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[y_col])
    if d.empty:
        return None
    labels = d[x_col].astype(str).tolist()
    lines = _tikz_header(caption, label, placement=placement)
    lines.extend(
        [
            r"            \begin{axis}[",
            r"                ybar,",
            r"                bar width=22pt,",
            r"                width=0.78\textwidth,",
            r"                height=7cm,",
            rf"                ylabel={{{_latex_escape(_axis_label(ylabel))}}},",
            r"                nodes near coords,",
            r"                every node near coord/.append style={font=\scriptsize, text=black, /pgf/number format/fixed, /pgf/number format/precision=2},",
            r"                axis lines*=left,",
            r"                ymin=0,",
            *_symbolic_axis_options(labels),
            r"            ]",
        ]
    )
    for _, row in d.iterrows():
        x = row[x_col]
        y = row[y_col]
        role = _model_color_role(row.get(model_col, x))
        lines.append(rf"                \addplot[ybar, bar shift=0pt, fill={role}, draw={role}] coordinates {{({_tex_symbol(x)},{_tex_num(y)})}};")
    lines.extend([r"            \end{axis}", *_tikz_footer(caption, label)])
    return _write_lines(path, lines)


def _write_computational_cost_tex(
    path: Path,
    *,
    data: pd.DataFrame,
    caption: str,
    label: str,
    placement: str = "htbp",
) -> Path | None:
    required = {"model_label", "model", "final_training_scaled_hours_365d"}
    if data.empty or not required.issubset(data.columns):
        return None
    d = data.copy()
    d["final_training_scaled_hours_365d"] = pd.to_numeric(d["final_training_scaled_hours_365d"], errors="coerce")
    d = d.dropna(subset=["final_training_scaled_hours_365d"])
    if d.empty:
        return None
    d = sort_model_frame(d, model_col="model_label")
    labels = d["model_label"].astype(str).tolist()
    x_positions = list(range(len(labels)))
    ymax = max(0.1, float(d["final_training_scaled_hours_365d"].max()) * 1.18)

    lines = _tikz_header(caption, label, placement=placement)
    lines.extend(
        [
            r"            \begin{axis}[",
            r"                width=0.78\textwidth,",
            r"                height=7cm,",
            r"                ylabel={Time (h)},",
            r"                axis lines*=left,",
            r"                ymin=0,",
            rf"                ymax={_tex_num(ymax)},",
            r"                xmin=-0.6,",
            rf"                xmax={_tex_num(len(labels) - 0.4)},",
            "                xtick={" + ",".join(str(x) for x in x_positions) + "},",
            "                xticklabels={" + ",".join(_latex_escape(_tex_label(x)) for x in labels) + "},",
            r"            ]",
        ]
    )
    half_width = 0.28
    for x_pos, (_, row) in zip(x_positions, d.iterrows()):
        role = _model_color_role(row.get("model", row.get("model_label", "")))
        training = float(row["final_training_scaled_hours_365d"])
        x_left = x_pos - half_width
        x_right = x_pos + half_width
        lines.append(rf"                \draw[fill={role}, draw={role}] (axis cs:{_tex_num(x_left)},0) rectangle (axis cs:{_tex_num(x_right)},{_tex_num(training)});")
        lines.append(rf"                \node[font=\scriptsize, text=black, anchor=south] at (axis cs:{_tex_num(x_pos)},{_tex_num(training)}) {{{_tex_hours(training)}}};")
    lines.extend(
        [
            r"            \end{axis}",
            *_tikz_footer(caption, label),
        ]
    )
    return _write_lines(path, lines)


def _write_p50_tolerance_curve_tex(
    path: Path,
    *,
    data: pd.DataFrame,
    target: str,
    split: str,
    placement: str = "htbp",
) -> Path | None:
    if data.empty or target not in P50_TOLERANCE_TEX_CONFIG:
        return None
    slug, label, thresholds, unit = P50_TOLERANCE_TEX_CONFIG[target]
    d = data[(data["split"].astype(str).eq(split)) & (data["target"].astype(str).eq(target))].copy()
    if d.empty:
        return None
    d["threshold"] = pd.to_numeric(d["threshold"], errors="coerce")
    d["share_within_threshold"] = pd.to_numeric(d["share_within_threshold"], errors="coerce")
    d = d.dropna(subset=["threshold", "share_within_threshold"])
    if d.empty:
        return None
    xmax = float(d["threshold"].max())
    x_den = xmax if abs(xmax) > 1e-12 else 1.0
    # Keep dense reference thresholds off the x-axis. Labels such as 1/5/10
    # otherwise collide near the origin for large price-error ranges.
    xticks = [round(xmax, 8)]
    x_tick_values = ",".join(_tex_num(x) for x in xticks)
    x_tick_labels = ",".join(_latex_escape(f"{x:g}") for x in xticks)
    caption = _p50_tolerance_title(target)
    lines = [
        r"% Requires: \usepackage{pgfplots}",
        r"% Requires: \usepackage{xcolor}",
        r"% Recommended in preamble: \pgfplotsset{compat=1.18}",
        *_latex_color_defs(),
        rf"\begin{{figure}}[{placement}]",
        r"    \centering",
        r"    \resizebox{\linewidth}{!}{%",
        r"        \begin{tikzpicture}",
        r"            \begin{axis}[",
        r"                width=0.96\textwidth,",
        r"                height=7cm,",
        rf"                xlabel={{Absolute p50 error threshold in {_latex_escape(unit)}}},",
        r"                ylabel={Share Within Threshold},",
        r"                ymin=0,",
        r"                ymax=1,",
        r"                xmin=0,",
        rf"                xmax={_tex_num(xmax)},",
        rf"                xtick={{{x_tick_values}}},",
        rf"                xticklabels={{{x_tick_labels}}},",
        r"                ytick={0,0.2,0.4,0.6,0.8,1},",
        r"                yticklabels={0\%,20\%,40\%,60\%,80\%,100\%},",
        r"                clip=false,",
        r"                legend style={at={(0.5,1.16)}, anchor=south, legend columns=-1, draw=none, fill=none, text=black},",
        r"                legend cell align={left},",
        r"                axis lines*=left,",
        r"                grid=major,",
        r"            ]",
    ]
    for threshold in thresholds:
        threshold_f = float(threshold)
        if 0.0 <= threshold_f <= xmax:
            lines.append(rf"                \addplot[color=neutraldark, dashed, mark=none, line width=0.8pt, forget plot] coordinates {{({_tex_num(threshold_f)},0) ({_tex_num(threshold_f)},1)}};")
            x_rel = threshold_f / x_den
            threshold_label = _latex_escape(f"{threshold_f:g}")
            lines.append(rf"                \node[font=\scriptsize, anchor=north, fill=white, fill opacity=0.9, text opacity=1, inner sep=1pt, yshift=-1pt] at (rel axis cs:{_tex_num(x_rel)},0) {{{threshold_label}}};")
    legends: list[str] = []
    for model in MODEL_LABELS:
        part = d[d["model"].astype(str).eq(model)].sort_values("threshold")
        if part.empty:
            continue
        coords = " ".join(f"({_tex_num(x)},{_tex_num(y)})" for x, y in zip(part["threshold"], part["share_within_threshold"]))
        color = _model_color_role(model)
        lines.append(rf"                \addplot[color={color}, mark=none, line width=1.3pt] coordinates {{{coords}}};")
        legends.append(_latex_escape(model))
    if legends:
        lines.append("                \\legend{" + ",".join(legends) + "}")
    lines.extend(
        [
            r"            \end{axis}",
            r"        \end{tikzpicture}}",
            f"    \\caption{{{_latex_escape(_ensure_caption_period(caption))}}}",
            rf"    \label{{fig:{slug}_p50_absolute_error_tolerance_curve}}",
            r"\end{figure}",
            "",
        ]
    )
    return _write_lines(path, lines)


def _write_p50_tolerance_pair_wrapper_tex(
    path: Path,
    *,
    pos_target: str,
    neg_target: str,
    caption: str,
    label: str,
) -> Path:
    pos_slug = P50_TOLERANCE_TEX_CONFIG[pos_target][0]
    neg_slug = P50_TOLERANCE_TEX_CONFIG[neg_target][0]
    figure_root = "figures/4-results/rq1_ml_model_benchmark/4_1_1_full_unweighted/result_section/figures"
    pos_caption = _p50_tolerance_title_label(pos_target)
    neg_caption = _p50_tolerance_title_label(neg_target)
    lines = [
        r"\begin{figure}[htbp]",
        r"    \centering",
        r"    \begin{subfigure}[t]{0.495\textwidth}",
        r"        \centering",
        rf"        \includegraphics[width=\linewidth]{{{figure_root}/{pos_slug}_p50_absolute_error_tolerance_curve.png}}",
        rf"        \caption{{{_latex_escape(pos_caption)}.}}",
        rf"        \label{{fig:{pos_slug}_p50_absolute_error_tolerance_curve}}",
        r"    \end{subfigure}\hfill",
        r"    \begin{subfigure}[t]{0.495\textwidth}",
        r"        \centering",
        rf"        \includegraphics[width=\linewidth]{{{figure_root}/{neg_slug}_p50_absolute_error_tolerance_curve.png}}",
        rf"        \caption{{{_latex_escape(neg_caption)}.}}",
        rf"        \label{{fig:{neg_slug}_p50_absolute_error_tolerance_curve}}",
        r"    \end{subfigure}",
        rf"    \caption{{{_latex_escape(_ensure_caption_period(caption))}}}",
        rf"    \label{{{label}}}",
        r"\end{figure}",
        "",
    ]
    return _write_lines(path, lines)


def _write_line_tex(
    path: Path,
    *,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    series_col: str,
    caption: str,
    label: str,
    ylabel: str,
    xlabel: str = "Lead hour",
    placement: str = "htbp",
    reference_y: float | None = None,
    reference_label: str | None = None,
    ideal_diagonal: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    percent_axes: bool = False,
    show_markers: bool = True,
    fragment_only: bool = False,
    highlight_spans: tuple[tuple[float, float], ...] = (),
    axis_height: str | None = None,
    axis_title: str | None = None,
    legend_y: float = 1.08,
    show_legend: bool = True,
    titlecase_caption: bool = True,
) -> Path | None:
    if data.empty:
        return None
    colors = {"TFT": "tertiary", "XGB": "primary", "RLQR": "secondary", "linear": "secondary", "tft": "tertiary", "xgb": "primary"}
    x_values_all = pd.to_numeric(data[x_col], errors="coerce").dropna()
    x_min = float(xlim[0]) if xlim is not None else (float(x_values_all.min()) if not x_values_all.empty else 0.0)
    x_max = float(xlim[1]) if xlim is not None else (float(x_values_all.max()) if not x_values_all.empty else 1.0)
    lines = [r"\begin{tikzpicture}"] if fragment_only else _tikz_header(caption, label, placement=placement)
    lines.extend(
        [
            r"            \begin{axis}[",
            r"                width=0.96\textwidth,",
            rf"                height={axis_height or '7cm'},",
            *([rf"                title={{{_latex_escape(axis_title)}}},"] if axis_title else []),
            rf"                xlabel={{{_latex_escape(xlabel)}}},",
            rf"                ylabel={{{_latex_escape(_axis_label(ylabel))}}},",
            *(
                [
                    rf"                legend style={{at={{(0.5,{_tex_num(legend_y)})}}, anchor=south, legend columns=-1, draw=none, fill=none, text=black}},",
                    r"                legend cell align={left},",
                ]
                if show_legend
                else []
            ),
            r"                axis lines*=left,",
            r"                grid=major,",
            *((
                r"                scaled y ticks=false,",
                r"                y tick label style={/pgf/number format/fixed},",
            ) if y_col == "mean_pinball_loss" else ()),
            *([rf"                xmin={_tex_num(xlim[0])},", rf"                xmax={_tex_num(xlim[1])},"] if xlim is not None else []),
            *([rf"                ymin={_tex_num(ylim[0])},", rf"                ymax={_tex_num(ylim[1])},"] if ylim is not None else []),
            *((
                r"                xtick={0.1,0.3,0.5,0.7,0.9},",
                r"                xticklabels={10\%,30\%,50\%,70\%,90\%},",
                r"                ytick={0.1,0.3,0.5,0.7,0.9},",
                r"                yticklabels={10\%,30\%,50\%,70\%,90\%},",
            ) if percent_axes else ()),
            r"            ]",
        ]
    )
    legends: list[str] = []
    grouped = {str(series): group for series, group in data.groupby(series_col, sort=False)}
    ordered_series = [s for s in ordered_model_labels(grouped.keys()) if s in grouped]
    ordered_series.extend([s for s in grouped if s not in ordered_series])
    for left, right in highlight_spans:
        left_rel = (float(left) - x_min) / (x_max - x_min) if abs(x_max - x_min) > 1e-12 else 0.0
        right_rel = (float(right) - x_min) / (x_max - x_min) if abs(x_max - x_min) > 1e-12 else 1.0
        lines.append(
            rf"                \fill[black!8, draw=none] (rel axis cs:{_tex_num(left_rel)},0) rectangle (rel axis cs:{_tex_num(right_rel)},1);"
        )
    if reference_y is not None:
        x_values = pd.to_numeric(data[x_col], errors="coerce").dropna()
        if not x_values.empty:
            xmin = float(x_values.min())
            xmax = float(x_values.max())
            ref_label = reference_label or f"{reference_y:g}"
            lines.append(rf"                \addplot[color=secondary, densely dotted, mark=none, line width=1.2pt] coordinates {{({_tex_num(xmin)},{_tex_num(reference_y)}) ({_tex_num(xmax)},{_tex_num(reference_y)})}};")
            legends.append(_latex_escape(_tex_label(ref_label)))
    for series in ordered_series:
        group = grouped[series]
        group = group.sort_values(x_col)
        coords = " ".join(f"({_tex_num(x)},{_tex_num(y)})" for x, y in zip(group[x_col], pd.to_numeric(group[y_col], errors="coerce")))
        color = colors.get(str(series), "neutraldark")
        marker_by_series = {
            "XGB": "square*",
            "xgb": "square*",
            "TFT": "triangle*",
            "tft": "triangle*",
            "RFT": "triangle*",
            "rft": "triangle*",
        }
        marker = marker_by_series.get(str(series), "*")
        marker_style = rf"mark={marker}, mark options={{fill={color}, draw={color}}}" if show_markers else "mark=none"
        lines.append(rf"                \addplot[color={color}, {marker_style}, line width=1pt] coordinates {{{coords}}};")
        legends.append(_latex_escape(_tex_label(series)))
    if ideal_diagonal:
        xmin, xmax = xlim if xlim is not None else (0.0, 1.0)
        ymin, ymax = ylim if ylim is not None else (0.0, 1.0)
        lo = max(float(xmin), float(ymin))
        hi = min(float(xmax), float(ymax))
        lines.append(rf"                \addplot[color=neutraldark, dashed, mark=none, line width=1pt] coordinates {{({_tex_num(lo)},{_tex_num(lo)}) ({_tex_num(hi)},{_tex_num(hi)})}};")
        legends.append("Ideal")
    if legends and show_legend:
        lines.append("                \\legend{" + ",".join(legends) + "}")
    if fragment_only:
        lines.extend([r"            \end{axis}", r"\end{tikzpicture}", ""])
    else:
        lines.extend([r"            \end{axis}", *_tikz_footer(caption, label, titlecase_caption=titlecase_caption)])
    return _write_lines(path, lines)


def _write_line_panel_tex(
    path: Path,
    *,
    data: pd.DataFrame,
    panel_col: str,
    x_col: str,
    y_col: str,
    series_col: str,
    caption: str,
    label: str,
    ylabel: str,
    xlabel: str = "Lead hour",
    placement: str = "htbp",
    reference_y: float | None = None,
    reference_label: str | None = None,
    ideal_diagonal: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    percent_ticks: tuple[float, ...] | None = None,
    show_markers: bool = True,
    highlight_spans: tuple[tuple[float, float], ...] = (),
    axis_height: str | None = None,
    y_tick_precision: int | None = None,
    y_scale: float = 1.0,
    y_label_x: float = -0.11,
    titlecase_caption: bool = True,
) -> Path | None:
    if data.empty:
        return None
    x_values_all = pd.to_numeric(data[x_col], errors="coerce").dropna()
    x_min = float(xlim[0]) if xlim is not None else (float(x_values_all.min()) if not x_values_all.empty else 0.0)
    x_max = float(xlim[1]) if xlim is not None else (float(x_values_all.max()) if not x_values_all.empty else 1.0)
    panels = ordered_unique(data[panel_col].dropna().astype(str).drop_duplicates().tolist())
    if not panels:
        return None
    group_cols = min(len(panels), 2)
    group_rows = int(np.ceil(len(panels) / group_cols))
    axis_width = "0.47\\textwidth" if group_cols > 1 else "0.86\\textwidth"
    axis_height = axis_height or ("6.2cm" if y_col == "mean_pinball_loss" else "5.8cm")
    colors = {"TFT": "tertiary", "XGB": "primary", "RLQR": "secondary", "linear": "secondary", "tft": "tertiary", "xgb": "primary"}
    all_series = {str(series) for series in data[series_col].dropna().unique()}
    ordered_all_series = [s for s in ordered_model_labels(all_series) if s in all_series]
    ordered_all_series.extend([s for s in sorted(all_series) if s not in ordered_all_series])
    legend_entries: list[str] = []
    if reference_y is not None:
        legend_entries.append(_latex_escape(_tex_label(reference_label or f"{reference_y:g}")))
    for series in ordered_all_series:
        legend_entries.append(_latex_escape(_tex_label(series)))
    lines = [
        r"% Requires: \usepackage{pgfplots}",
        r"% Requires: \usepackage{xcolor}",
        r"% Requires: \usepgfplotslibrary{groupplots}",
        r"% Recommended in preamble: \pgfplotsset{compat=1.18}",
        *_latex_color_defs(),
        rf"\begin{{figure}}[{placement}]",
        r"    \centering",
        r"    \resizebox{\linewidth}{!}{%",
        r"        \begin{tikzpicture}",
    ]
    lines.extend(
        [
        r"            \begin{groupplot}[",
        rf"                group style={{group size={group_cols} by {group_rows}, horizontal sep=1.25cm, vertical sep=1.15cm}},",
        rf"                width={axis_width},",
        rf"                height={axis_height},",
        rf"                xlabel={{{_latex_escape(xlabel)}}},",
        rf"                ylabel={{{_latex_escape(_axis_label(ylabel))}}},",
        rf"                y label style={{at={{({_tex_num(y_label_x)},0.5)}}}},",
        r"                legend style={at={(0.5,1.22)}, anchor=south, legend columns=-1, draw=none, fill=none, text=black},",
        r"                legend cell align={left},",
        r"                axis lines*=left,",
        r"                grid=major,",
        *((
            r"                scaled y ticks=false,",
            "                y tick label style={/pgf/number format/fixed"
            + (f", /pgf/number format/precision={int(y_tick_precision)}" if y_tick_precision is not None else "")
            + "},",
        ) if y_col == "mean_pinball_loss" else ()),
        *([rf"                xmin={_tex_num(xlim[0])},", rf"                xmax={_tex_num(xlim[1])},"] if xlim is not None else []),
        *([rf"                ymin={_tex_num(float(ylim[0]) * float(y_scale))},", rf"                ymax={_tex_num(float(ylim[1]) * float(y_scale))},"] if ylim is not None else []),
        *(_percent_tick_options(percent_ticks) if percent_ticks is not None else []),
        r"            ]",
        ]
    )
    for panel_i, panel in enumerate(panels):
        panel_df = data[data[panel_col].astype(str).eq(panel)].copy()
        lines.append(rf"                \nextgroupplot[title={{{_latex_escape(thesis_titlecase(_tex_label(panel)))}}}]")
        for left, right in highlight_spans:
            left_rel = (float(left) - x_min) / (x_max - x_min) if abs(x_max - x_min) > 1e-12 else 0.0
            right_rel = (float(right) - x_min) / (x_max - x_min) if abs(x_max - x_min) > 1e-12 else 1.0
            lines.append(
                rf"                    \fill[black!8, draw=none] (rel axis cs:{_tex_num(left_rel)},0) rectangle (rel axis cs:{_tex_num(right_rel)},1);"
            )
        grouped = {str(series): group for series, group in panel_df.groupby(series_col, sort=False)}
        ordered_series = [s for s in ordered_model_labels(grouped.keys()) if s in grouped]
        ordered_series.extend([s for s in grouped if s not in ordered_series])
        if reference_y is not None:
            x_values = pd.to_numeric(panel_df[x_col], errors="coerce").dropna()
            if not x_values.empty:
                xmin = float(x_values.min())
                xmax = float(x_values.max())
                lines.append(rf"                    \addplot[color=secondary, densely dotted, mark=none, line width=1.2pt] coordinates {{({_tex_num(xmin)},{_tex_num(reference_y)}) ({_tex_num(xmax)},{_tex_num(reference_y)})}};")
        for series in ordered_series:
            group = (
                grouped[series]
                .assign(**{x_col: pd.to_numeric(grouped[series][x_col], errors="coerce"), y_col: pd.to_numeric(grouped[series][y_col], errors="coerce")})
                .dropna(subset=[x_col, y_col])
                .sort_values(x_col)
            )
            if group.empty:
                continue
            # Defensive aggregation: one plotted point per panel/model/lead.
            group = group.groupby(x_col, as_index=False)[y_col].mean()
            coords = " ".join(f"({_tex_num(x)},{_tex_num(float(y) * float(y_scale))})" for x, y in zip(group[x_col], group[y_col]))
            color = colors.get(series, "neutraldark")
            marker_style = rf"mark=*, mark options={{fill={color}, draw={color}}}" if show_markers else "mark=none"
            lines.append(rf"                    \addplot[color={color}, {marker_style}, line width=1pt] coordinates {{{coords}}};")
        if ideal_diagonal:
            xmin, xmax = xlim if xlim is not None else (0.0, 1.0)
            ymin, ymax = ylim if ylim is not None else (0.0, 1.0)
            lo = max(float(xmin), float(ymin))
            hi = min(float(xmax), float(ymax))
            lines.append(rf"                    \addplot[color=neutraldark, dashed, mark=none, line width=1pt] coordinates {{({_tex_num(lo)},{_tex_num(lo)}) ({_tex_num(hi)},{_tex_num(hi)})}};")
    if legend_entries:
        lines.append("                \\legend{" + ",".join(legend_entries) + "}")
    lines.extend(
        [
            r"            \end{groupplot}",
            r"        \end{tikzpicture}}",
            f"    \\caption{{{_latex_escape(_ensure_caption_period(thesis_titlecase(caption) if titlecase_caption else caption))}}}",
            f"    \\label{{{label}}}",
            r"\end{figure}",
            "",
        ]
    )
    return _write_lines(path, lines)


def _write_combined_per_lead_pinball_tex(root: Path) -> Path | None:
    specs = [
        (
            "per_lead_pinball_da_price.tex",
            "DA price",
            "fig:rq1-4-1-3-per-lead-pinball-da-price",
            "4.3cm",
        ),
        (
            "per_lead_pinball_afrr_capacity_price.tex",
            "aFRR capacity price",
            "fig:rq1-4-1-3-per-lead-pinball-afrr-capacity-price",
            "4.0cm",
        ),
        (
            "per_lead_pinball_afrr_activation_price.tex",
            "aFRR activation price",
            "fig:rq1-4-1-3-per-lead-pinball-afrr-activation-price",
            "4.0cm",
        ),
        (
            "per_lead_pinball_afrr_activation_rate.tex",
            "aFRR activation rate",
            "fig:rq1-4-1-3-per-lead-pinball-afrr-activation-rate",
            "4.0cm",
        ),
    ]
    latex_dir = root / "result_section" / "latex_figures"
    source_paths = [(latex_dir / filename, caption, label, height) for filename, caption, label, height in specs]
    if any(not path.exists() for path, _, _, _ in source_paths):
        return None
    out = latex_dir / "per_lead_pinball_combined.tex"
    da_capacity_out = latex_dir / "per_lead_pinball_da_capacity.tex"
    activation_out = latex_dir / "per_lead_pinball_activation.tex"
    header_lines = [
        r"% Requires: \usepackage{pgfplots}",
        r"% Requires: \usepackage{xcolor}",
        r"% Requires: \usepackage{subcaption}",
        r"% Requires: \usepgfplotslibrary{groupplots}",
        r"% Recommended in preamble: \pgfplotsset{compat=1.18}",
        *_latex_color_defs(),
    ]

    def append_figure(
        target_lines: list[str],
        figure_specs: list[tuple[Path, str, str, str]],
        *,
        caption: str,
        label: str,
    ) -> None:
        target_lines.extend(
            [
                r"\begin{figure}[tbp]",
                r"    \centering",
                r"    \captionsetup[subfigure]{font=small,skip=0.2em}",
                r"    \begin{tikzpicture}",
                r"        \draw[color=secondary, line width=1pt] (0.00,0) -- (0.70,0);",
                r"        \node[anchor=west, text=black] at (0.90,0) {RLQR};",
                r"        \draw[color=primary, line width=1pt] (2.55,0) -- (3.25,0);",
                r"        \node[anchor=west, text=black] at (3.45,0) {XGB};",
                r"        \draw[color=tertiary, line width=1pt] (5.10,0) -- (5.80,0);",
                r"        \node[anchor=west, text=black] at (6.00,0) {TFT};",
                r"    \end{tikzpicture}",
                r"    \vspace{0.15em}",
            ]
        )

        for i, (path, subcaption, sublabel, height) in enumerate(figure_specs):
            tikz_lines = _extract_tikzpicture_lines(path)
            tikz_lines = _with_compact_axis_height(tikz_lines, height)
            tikz_lines = _without_embedded_legend(tikz_lines)
            tikz_lines = _with_combined_per_lead_layout(tikz_lines, path.name)
            target_lines.extend(
                [
                    r"    \begin{subfigure}[t]{0.98\textwidth}",
                    r"        \centering",
                    r"        \resizebox{0.96\linewidth}{!}{%",
                ]
            )
            target_lines.extend("        " + line for line in tikz_lines)
            target_lines.extend(
                [
                    r"        }",
                    r"        \phantomcaption",
                    f"        \\label{{{sublabel}}}",
                    r"    \end{subfigure}",
                ]
            )
            if i < len(figure_specs) - 1:
                target_lines.append(r"    \vspace{0.25em}")

        target_lines.extend(
            [
                rf"    \caption{{{_latex_escape(_ensure_caption_period(caption))}}}",
                f"    \\label{{{label}}}",
                r"\end{figure}",
                "",
            ]
        )

    da_capacity_lines = list(header_lines)
    append_figure(
        da_capacity_lines,
        source_paths[:2],
        caption="Mean pinball loss by lead hour for DA and aFRR capacity price targets. Grey bands mark decision-relevant lead ranges.",
        label="fig:rq1-4-1-3-per-lead-pinball-combined",
    )
    activation_lines = list(header_lines)
    append_figure(
        activation_lines,
        source_paths[2:],
        caption="Mean pinball loss by lead hour for aFRR activation price and activation rate targets. Grey bands mark decision-relevant lead ranges.",
        label="fig:rq1-4-1-3-per-lead-pinball-combined-activation",
    )
    combined_lines = list(header_lines)
    append_figure(
        combined_lines,
        source_paths[:2],
        caption="Mean pinball loss by lead hour for DA and aFRR capacity price targets. Grey bands mark decision-relevant lead ranges.",
        label="fig:rq1-4-1-3-per-lead-pinball-combined",
    )
    append_figure(
        combined_lines,
        source_paths[2:],
        caption="Mean pinball loss by lead hour for aFRR activation price and activation rate targets. Grey bands mark decision-relevant lead ranges.",
        label="fig:rq1-4-1-3-per-lead-pinball-combined-activation",
    )
    _write_lines(da_capacity_out, da_capacity_lines)
    _write_lines(activation_out, activation_lines)
    return _write_lines(out, combined_lines)


def _padded_ylim(data: pd.DataFrame, value_col: str, *, include: tuple[float, ...] = ()) -> tuple[float, float] | None:
    if data.empty or value_col not in data.columns:
        return None
    values = pd.to_numeric(data[value_col], errors="coerce").dropna()
    if include:
        values = pd.concat([values, pd.Series(include, dtype=float)], ignore_index=True)
    if values.empty:
        return None
    ymin = float(values.min())
    ymax = float(values.max())
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return None
    span = ymax - ymin
    pad = max(abs(ymax) * 0.02, 1e-9) if span <= 0 else span * 0.06
    return ymin - pad, ymax + pad


def _zero_based_ylim(data: pd.DataFrame, value_col: str, *, top_pad_fraction: float = 0.08) -> tuple[float, float] | None:
    if data.empty or value_col not in data.columns:
        return None
    values = pd.to_numeric(data[value_col], errors="coerce").dropna()
    if values.empty:
        return None
    ymax = float(values.max())
    if not np.isfinite(ymax):
        return None
    return 0.0, max(ymax * (1.0 + top_pad_fraction), 1e-9)


def _write_heatmap_tex(
    path: Path,
    *,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    caption: str,
    label: str,
    placement: str = "htbp",
) -> Path | None:
    if data.empty:
        return None
    xs = data[x_col].astype(str).drop_duplicates().tolist()
    ys = data[y_col].astype(str).drop_duplicates().tolist()
    lines = _tikz_header(caption, label, placement=placement)
    lines.extend(
        [
            r"            \begin{axis}[",
            r"                width=0.98\textwidth,",
            r"                height=8cm,",
            r"                view={0}{90},",
            r"                colorbar,",
            r"                colormap={geoSequentialBlue}{rgb255(0cm)=(228,241,247); rgb255(1cm)=(197,225,239); rgb255(2cm)=(158,201,226); rgb255(3cm)=(108,176,214); rgb255(4cm)=(60,147,194); rgb255(5cm)=(34,110,156); rgb255(6cm)=(13,74,112)},",
            r"                point meta min=0,",
            r"                point meta max=1,",
            r"                x tick label style={rotate=35, anchor=east},",
            *_symbolic_axis_options(xs),
            "                symbolic y coords={" + ",".join(_tex_symbol(y) for y in ys) + "},",
            "                ytick={" + ",".join(_tex_symbol(y) for y in ys) + "},",
            "                yticklabels={" + ",".join(_latex_escape(y) for y in ys) + "},",
            r"            ]",
            r"                \addplot[scatter, only marks, mark=square*, mark size=5.8pt, scatter src=explicit] coordinates {",
        ]
    )
    for _, row in data.iterrows():
        lines.append(f"                    ({_tex_symbol(row[x_col])},{_tex_symbol(row[y_col])}) [{_tex_num(row[value_col])}]")
    lines.extend([r"                };", r"            \end{axis}", *_tikz_footer(caption, label)])
    return _write_lines(path, lines)


def _add_tikz_entry(entries: list[dict[str, Any]], *, subsection: str, tier: str, path: Path, metric_family: str, description: str) -> None:
    entries.append(
        {
            "subsection": subsection,
            "tier": tier,
            "artifact_type": "latex_figure",
            "path": str(path),
            "metric_family": metric_family,
            "thesis_use": "native TikZ/pgfplots figure code",
            "brief_description": description,
        }
    )


def _add_latex_figure_entry(
    entries: list[dict[str, Any]],
    *,
    subsection: str,
    tier: str,
    path: Path,
    metric_family: str,
    thesis_use: str,
    description: str,
) -> None:
    entries.append(
        {
            "subsection": subsection,
            "tier": tier,
            "artifact_type": "latex_figure",
            "path": str(path),
            "metric_family": metric_family,
            "thesis_use": thesis_use,
            "brief_description": description,
        }
    )


def _generate_latex_figures(entries: list[dict[str, Any]], *, rq1_root: Path, split: str) -> None:
    # 4.1.1 relative full-sample pinball bar chart.
    sec = "4.1.1"
    root = rq1_root / SUBSECTIONS[sec]
    primary = root / "backup" / "csv" / f"forecast_metrics_full_primary_{split}.csv"
    if primary.exists():
        df = pd.read_csv(primary)
        df = df[~df["target"].astype(str).isin(["Average", "Mean"])].copy()
        df["target_label"] = df["target"].map(lambda x: _tex_label(str(x).replace("pred_", "")))
        df = sort_target_frame(df, target_col="target")
        for col in ["XGB", "TFT"]:
            df[col] = pd.to_numeric(df[col], errors="coerce") / pd.to_numeric(df["RLQR"], errors="coerce")
        avg_row = {col: np.nan for col in df.columns}
        avg_row["target"] = "relative_average"
        avg_row["target_label"] = "Mean"
        avg_row["XGB"] = pd.to_numeric(df["XGB"], errors="coerce").mean()
        avg_row["TFT"] = pd.to_numeric(df["TFT"], errors="coerce").mean()
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
        out = root / "result_section" / "latex_figures" / f"forecast_metrics_full_relative_pinball_{split}.tex"
        path = _write_grouped_bar_tex(
            out,
            data=df,
            x_col="target",
            x_tick_label_col="target_label",
            series_cols=["XGB", "TFT"],
            caption="Mean pinball loss by forecast target relative to RLQR. The dotted line marks the RLQR benchmark at 1. Values below 1 indicate lower loss than RLQR, while values above 1 indicate higher loss.",
            label="fig:rq1-4-1-1-forecast-metrics-full-relative-pinball",
            ylabel="Mean pinball loss relative to RLQR",
            reference_y=1.0,
            reference_label="RLQR",
            axis_height="6.2cm",
            legend_style=r"                legend style={at={(0.5,1.10)}, anchor=south, legend columns=-1, draw=none, fill=none, text=black},",
        )
        if path:
            _add_tikz_entry(entries, subsection=sec, tier="result_section", path=path, metric_family="mean_pinball_loss", description="Native pgfplots relative full-sample pinball bar chart.")
    detailed = root / "backup" / "csv" / f"forecast_metrics_full_detailed_{split}.csv"
    if detailed.exists():
        df = pd.read_csv(detailed)
        df = df.loc[df["metric"].eq("mae_p50")].copy()
        df["target_label"] = df["target"].map(lambda x: _tex_label(str(x).replace("pred_", "")))
        df = sort_target_frame(df, target_col="target")
        denom = pd.to_numeric(df["RLQR"], errors="coerce")
        df = df.loc[denom.notna() & denom.abs().gt(1e-12)].copy()
        denom = pd.to_numeric(df["RLQR"], errors="coerce")
        for col in ["XGB", "TFT"]:
            df[col] = pd.to_numeric(df[col], errors="coerce") / denom
        out = root / "appendix" / "latex_figures" / f"forecast_metrics_full_relative_mae_p50_{split}.tex"
        path = _write_grouped_bar_tex(
            out,
            data=df,
            x_col="target_label",
            series_cols=["XGB", "TFT"],
            caption="Relative MAE p50 by target (RLQR = 1; lower is better).",
            label="fig:rq1-4-1-1-forecast-metrics-full-relative-mae-p50",
            ylabel="MAE p50 relative to RLQR",
            placement="p",
            reference_y=1.0,
            reference_label="RLQR",
        )
        if path:
            _add_tikz_entry(entries, subsection=sec, tier="appendix", path=path, metric_family="mae_p50", description="Native pgfplots relative full-sample p50 MAE bar chart.")
    cost_csv = root / "backup" / "csv" / "computational_cost_365d.csv"
    if cost_csv.exists():
        cost = pd.read_csv(cost_csv)
        out = root / "result_section" / "latex_figures" / "computational_cost_365d.tex"
        path = _write_computational_cost_tex(
            out,
            data=cost,
            caption="Computational Cost of Each Model Scaled for 365 Days of Training Data",
            label="fig:rq1-4-1-1-computational-cost-365d",
        )
        if path:
            _add_tikz_entry(entries, subsection=sec, tier="result_section", path=path, metric_family="computational_cost", description="Native pgfplots computational-cost bar chart.")
    tolerance_candidates = [
        root / "backup" / "csv" / "price_p50_absolute_error_tolerance_curve.csv",
        rq1_root / "_raw_outputs" / "4_1_1_full_unweighted" / "csv" / "rq1_4_1_1_price_p50_absolute_error_tolerance_curve.csv",
        rq1_root / "_raw_outputs" / "4_1_1_full_unweighted_metrics" / "csv" / "rq1_4_1_1_price_p50_absolute_error_tolerance_curve.csv",
    ]
    tolerance_csv = next((path for path in tolerance_candidates if path.exists()), tolerance_candidates[0])
    if tolerance_csv.exists():
        tolerance = pd.read_csv(tolerance_csv)
        for target, (slug, _label, _thresholds, _unit) in P50_TOLERANCE_TEX_CONFIG.items():
            out = root / "result_section" / "latex_figures" / f"{slug}_p50_absolute_error_tolerance_curve.tex"
            path = _write_p50_tolerance_curve_tex(out, data=tolerance, target=target, split=split)
            if path:
                _add_tikz_entry(entries, subsection=sec, tier="result_section", path=path, metric_family="p50 absolute error tolerance", description=f"Native pgfplots p50 absolute error tolerance curve for {slug}.")
        for filename, pos_target, neg_target, caption, label in [
            (
                "afrr_capacity_price_p50_absolute_error_tolerance_curve.tex",
                "pred_afrr_capacity_price_pos",
                "pred_afrr_capacity_price_neg",
                "Cumulative absolute p50 error tolerance for aFRR capacity price targets.",
                "fig:rq1-4-1-1-absolute-error-tolerance-afrr-capacity-price",
            ),
            (
                "afrr_activation_price_p50_absolute_error_tolerance_curve.tex",
                "pred_afrr_activation_price_pos",
                "pred_afrr_activation_price_neg",
                "Cumulative absolute p50 error tolerance for aFRR activation price targets.",
                "fig:rq1-4-1-1-absolute-error-tolerance-afrr-activation-price",
            ),
        ]:
            out = root / "result_section" / "latex_figures" / filename
            path = _write_p50_tolerance_pair_wrapper_tex(out, pos_target=pos_target, neg_target=neg_target, caption=caption, label=label)
            _add_tikz_entry(entries, subsection=sec, tier="result_section", path=path, metric_family="p50 absolute error tolerance", description=f"Appendix wrapper for {caption}")

    # 4.1.2 calibration and uncertainty.
    sec = "4.1.2"
    root = rq1_root / SUBSECTIONS[sec]
    cal_candidates = [
        root / "backup" / "csv" / f"calibration_quantile_coverage_{split}.csv",
        rq1_root
        / "_raw_outputs"
        / "4_1_2_calibration_uncertainty"
        / "csv"
        / f"rq1_4_1_2_calibration_quantile_coverage_{split}.csv",
        rq1_root / "_raw_outputs" / "calibration" / f"calibration_quantile_coverage_{split}.csv",
    ]
    cal = next((path for path in cal_candidates if path.exists()), cal_candidates[0])
    if cal.exists():
        for stale in (root / "result_section" / "latex_figures").glob("calibration_reliability_*.tex"):
            stale.unlink()
        df = pd.read_csv(cal)
        df = (
            df.groupby(["target", "target_label", "target_group", "model", "model_label", "quantile"], as_index=False)
            .agg(empirical_coverage=("empirical_coverage", "mean"), n_obs=("n_obs", "sum"))
        )
        df = sort_target_frame(df, target_col="target", extra_cols=["model_label", "quantile"])
        for target_label in ordered_unique(df["target_label"].dropna().unique()):
            group = df[df["target_label"].eq(target_label)].copy()
            target_slug = _tex_symbol(str(group["target"].iloc[0]).replace("pred_", ""))
            out = root / "result_section" / "latex_figures" / f"calibration_reliability_{target_slug}.tex"
            show_legend = target_slug not in {"afrr_capacity_price_pos", "afrr_capacity_price_neg"}
            path = _write_line_tex(
                out,
                data=group,
                x_col="quantile",
                y_col="empirical_coverage",
                series_col="model_label",
                caption=f"Quantile reliability for {target_label}.",
                label=f"fig:rq1-4-1-2-calibration-reliability-{target_slug.lower()}",
                ylabel="Empirical coverage",
                xlabel="Nominal quantile",
                ideal_diagonal=True,
                xlim=(0.0, 1.0),
                ylim=(0.0, 1.0),
                percent_axes=True,
                fragment_only=True,
                axis_height="6.2cm",
                show_legend=show_legend,
            )
            if path:
                _add_tikz_entry(entries, subsection=sec, tier="result_section", path=path, metric_family="calibration", description=f"Native pgfplots quantile reliability for {target_label}.")
        aggregate_df = _aggregate_activation_reliability(df)
        for aggregate_slug, aggregate_label in ACTIVATION_RELIABILITY_AGGREGATES:
            group = aggregate_df.loc[aggregate_df["target"].eq(aggregate_slug)].copy() if not aggregate_df.empty else pd.DataFrame()
            if group.empty:
                continue
            out = root / "result_section" / "latex_figures" / f"calibration_reliability_{aggregate_slug}.tex"
            path = _write_line_tex(
                out,
                data=group,
                x_col="quantile",
                y_col="empirical_coverage",
                series_col="model_label",
                caption=f"Aggregate quantile reliability for {aggregate_label} with positive and negative directions merged.",
                label=f"fig:rq1-4-1-2-calibration-reliability-{aggregate_slug.replace('_', '-')}",
                ylabel="Empirical coverage",
                xlabel="Nominal quantile",
                ideal_diagonal=True,
                xlim=(0.0, 1.0),
                ylim=(0.0, 1.0),
                percent_axes=True,
                fragment_only=True,
                axis_height="6.2cm",
                show_legend=False,
            )
            if path:
                _add_tikz_entry(entries, subsection=sec, tier="result_section", path=path, metric_family="calibration", description=f"Native pgfplots aggregate quantile reliability for {aggregate_label}.")
    interval = root / "backup" / "csv" / f"calibration_interval_coverage_width_{split}.csv"
    if interval.exists():
        df = pd.read_csv(interval)
        required_cols = {"target_group", "model", "model_label", "nominal_interval_coverage", "interval_coverage", "n_obs"}
        if required_cols.issubset(df.columns):
            df = (
                df.groupby(["target_group", "model", "model_label", "nominal_interval_coverage"], as_index=False)
                .agg(interval_coverage=("interval_coverage", "mean"), n_obs=("n_obs", "sum"))
            )
            df["_target_group_order"] = df["target_group"].map(lambda x: target_sort_key(x)[0])
            df = df.sort_values(["_target_group_order", "model_label", "nominal_interval_coverage"]).drop(columns="_target_group_order")
            out = root / "result_section" / "latex_figures" / "calibration_interval_coverage_by_target_group.tex"
            path = _write_line_panel_tex(
                out,
                data=df,
                panel_col="target_group",
                x_col="nominal_interval_coverage",
                y_col="interval_coverage",
                series_col="model_label",
                caption="Interval coverage vs nominal interval coverage by target group.",
                label="fig:rq1-4-1-2-calibration-interval-coverage",
                ylabel="Empirical interval coverage",
                xlabel="Nominal interval coverage",
                ideal_diagonal=True,
                xlim=(0.0, 1.0),
                ylim=(0.0, 1.0),
                percent_ticks=(0.40, 0.80, 0.90, 0.98),
            )
            if path:
                _add_tikz_entry(entries, subsection=sec, tier="result_section", path=path, metric_family="interval_coverage", description="Native pgfplots interval coverage reliability by target group.")
    # 4.1.3 per-lead line charts.
    sec = "4.1.3"
    root = rq1_root / SUBSECTIONS[sec]
    per_lead_candidates = [
        root / "backup" / "csv" / f"per_lead_metrics_{split}.csv",
        rq1_root / "_raw_outputs" / "4_1_3_per_lead" / f"per_lead_metrics_{split}.csv",
        rq1_root / "_raw_outputs" / "4_1_3_per_lead_hour" / f"per_lead_metrics_{split}.csv",
    ]
    per_lead = next((path for path in per_lead_candidates if path.exists()), per_lead_candidates[0])
    if per_lead.exists():
        df = pd.read_csv(per_lead)
        pinball_highlight_spans = {
            "da_price": ((13.0, 36.0),),
            "afrr_capacity_price": ((16.0, 39.0),),
            "afrr_activation_price": ((0.0, 8.0), (16.0, 39.0)),
            "afrr_activation_rate": ((0.0, 8.0), (16.0, 39.0)),
        }
        for target_slug in sorted(df["target_slug"].dropna().unique(), key=target_sort_key):
            group = sort_target_frame(df[df["target_slug"].eq(target_slug)].copy(), target_col="target_label")
            for metric, stem, tier in [
                ("mean_pinball_loss", "per_lead_pinball", "result_section"),
                ("mae_p50", "per_lead_mae_p50", "appendix"),
                ("rmse_p50", "per_lead_rmse_p50", "appendix"),
            ]:
                out = root / tier / "latex_figures" / f"{stem}_{target_slug}.tex"
                target_label = _tex_label(group["target_label"].iloc[0])
                highlight_spans = pinball_highlight_spans.get(str(target_slug), ()) if metric == "mean_pinball_loss" and tier == "result_section" else ()
                xlim = (0.0, 48.0) if highlight_spans else None
                caption_suffix = " Relevant forecast lead highlighted in grey." if highlight_spans else ""
                if group["target_label"].nunique(dropna=True) > 1:
                    is_afrr_result_pinball = (
                        metric == "mean_pinball_loss"
                        and tier == "result_section"
                        and target_slug in {"afrr_capacity_price", "afrr_activation_price", "afrr_activation_rate"}
                    )
                    if target_slug == "afrr_activation_rate" and metric == "mean_pinball_loss":
                        shared_ylim = _padded_ylim(group, metric)
                        axis_height = "5.0cm" if is_afrr_result_pinball else None
                        y_tick_precision = 2
                        y_scale = 1000.0
                        y_label = "Mean pinball loss (1e-3)"
                    else:
                        shared_ylim = _padded_ylim(group, metric)
                        axis_height = "5.0cm" if is_afrr_result_pinball else None
                        y_tick_precision = None
                        y_scale = 1.0
                        y_label = _tex_label(metric)
                    caption = _sentence_case_caption(f"{_tex_label(metric)} by lead hour for {_tex_label(group['target_group'].iloc[0])}.{caption_suffix}")
                    path = _write_line_panel_tex(
                        out,
                        data=group,
                        panel_col="target_label",
                        x_col="lead_time_h",
                        y_col=metric,
                        series_col="model_label",
                        caption=caption,
                        label=f"fig:rq1-4-1-3-{stem.replace('_','-')}-{target_slug.replace('_','-')}",
                        ylabel=y_label,
                        placement="htbp" if tier == "result_section" else "p",
                        show_markers=metric != "mean_pinball_loss",
                        ylim=shared_ylim,
                        xlim=xlim,
                        highlight_spans=highlight_spans,
                        axis_height=axis_height,
                        y_tick_precision=y_tick_precision,
                        y_scale=y_scale,
                        y_label_x=-0.13 if target_slug == "afrr_activation_rate" and metric == "mean_pinball_loss" and tier == "result_section" else -0.11,
                        titlecase_caption=False,
                    )
                else:
                    axis_height = "5.6cm" if target_slug == "da_price" and metric == "mean_pinball_loss" and tier == "result_section" else None
                    axis_title = "DA Price" if target_slug == "da_price" and metric == "mean_pinball_loss" and tier == "result_section" else None
                    legend_y = 1.22 if axis_title else 1.08
                    caption = _sentence_case_caption(f"{_tex_label(metric)} by lead hour for {target_label}.{caption_suffix}")
                    path = _write_line_tex(out, data=group, x_col="lead_time_h", y_col=metric, series_col="model_label", caption=caption, label=f"fig:rq1-4-1-3-{stem.replace('_','-')}-{target_slug.replace('_','-')}", ylabel=_tex_label(metric), placement="htbp" if tier == "result_section" else "p", show_markers=metric != "mean_pinball_loss", xlim=xlim, highlight_spans=highlight_spans, axis_height=axis_height, axis_title=axis_title, legend_y=legend_y, titlecase_caption=False)
                if path:
                    _add_tikz_entry(entries, subsection=sec, tier=tier, path=path, metric_family=metric, description=f"Native pgfplots {metric} per-lead line chart for {target_slug}.")
            pivot = group.pivot_table(index=["target_label", "lead_time_h"], columns="model_label", values="mean_pinball_loss", aggfunc="first").reset_index()
            if "RLQR" in pivot.columns:
                rel_rows = []
                for _, row in pivot.iterrows():
                    for model in ["XGB", "TFT"]:
                        if model in pivot.columns and pd.notna(row.get(model)) and pd.notna(row.get("RLQR")) and abs(float(row["RLQR"])) > 1e-12:
                            rel_rows.append({"target_label": row["target_label"], "lead_time_h": row["lead_time_h"], "model_label": model, "relative": float(row[model]) / float(row["RLQR"])})
                rel = pd.DataFrame(rel_rows)
                out = root / "result_section" / "latex_figures" / f"per_lead_relative_pinball_{target_slug}.tex"
                target_label = _tex_label(group["target_label"].iloc[0])
                if group["target_label"].nunique(dropna=True) > 1:
                    shared_ylim = _padded_ylim(rel, "relative", include=(1.0,))
                    path = _write_line_panel_tex(
                        out,
                        data=rel,
                        panel_col="target_label",
                        x_col="lead_time_h",
                        y_col="relative",
                        series_col="model_label",
                        caption=f"Relative mean pinball loss by lead hour for {_tex_label(group['target_group'].iloc[0])} (RLQR = 1).",
                        label=f"fig:rq1-4-1-3-per-lead-relative-pinball-{target_slug.replace('_','-')}",
                        ylabel="Mean pinball loss relative to RLQR",
                        reference_y=1.0,
                        reference_label="RLQR",
                        ylim=shared_ylim,
                    )
                else:
                    path = _write_line_tex(out, data=rel, x_col="lead_time_h", y_col="relative", series_col="model_label", caption=f"Relative mean pinball loss by lead hour for {target_label} (RLQR = 1).", label=f"fig:rq1-4-1-3-per-lead-relative-pinball-{target_slug.replace('_','-')}", ylabel="Mean pinball loss relative to RLQR", reference_y=1.0, reference_label="RLQR")
                if path:
                    _add_tikz_entry(entries, subsection=sec, tier="result_section", path=path, metric_family="relative_mean_pinball_loss", description=f"Native pgfplots relative per-lead pinball line chart for {target_slug}.")
        combined = _write_combined_per_lead_pinball_tex(root)
        if combined:
            _add_tikz_entry(
                entries,
                subsection=sec,
                tier="result_section",
                path=combined,
                metric_family="mean_pinball_loss",
                description="Combined native pgfplots per-lead mean-pinball figure with four labelled subfigures.",
            )

    # 4.1.4 gate-specific bars.
    sec = "4.1.4"
    root = rq1_root / SUBSECTIONS[sec]
    gate = root / "backup" / "csv" / f"gate_bucket_metrics_{split}.csv"
    if gate.exists():
        df = pd.read_csv(gate)
        required_gate_cols = {"bucket", "bucket_family", "target", "target_group", "model_label", "mean_pinball_loss"}
        if required_gate_cols.issubset(df.columns):
            d = df.copy()
            d = sort_target_frame(d, target_col="target", extra_cols=["bucket"])
            out = root / "result_section" / "latex_figures" / "gate_bucket_pinball_by_target_group.tex"
            path = _write_gate_bucket_relative_tex(
                out,
                data=d,
                caption="Actionable forecast performance by market gate relative to RLQR; bars show XGB and TFT mean pinball loss relative to RLQR, so values below 1 indicate lower loss than RLQR.",
                label="fig:rq1-4-1-4-gate-bucket-pinball",
            )
            if path:
                _add_tikz_entry(entries, subsection=sec, tier="result_section", path=path, metric_family="relative_mean_pinball_loss", description="Native pgfplots actionable market-gate relative pinball horizontal bar chart.")
            if "coverage_p10_p90" in d.columns:
                d["x"] = d["bucket"].map(_bucket_label) + " / " + d["target_group"].astype(str)
                x_order = d["x"].drop_duplicates().tolist()
                pivot = d.pivot_table(index="x", columns="model_label", values="coverage_p10_p90", aggfunc="mean").reset_index()
                pivot["x"] = pd.Categorical(pivot["x"], categories=x_order, ordered=True)
                pivot = pivot.sort_values("x")
                out = root / "appendix" / "latex_figures" / "gate_bucket_coverage_p10_p90_by_target_group.tex"
                path = _write_grouped_bar_tex(out, data=pivot, x_col="x", series_cols=[c for c in MODEL_LABELS if c in pivot.columns], caption="Gate-specific p10-p90 interval coverage.", label="fig:rq1-4-1-4-gate-bucket-coverage", ylabel="Coverage", placement="p")
                if path:
                    _add_tikz_entry(entries, subsection=sec, tier="appendix", path=path, metric_family="interval_coverage", description="Native pgfplots gate-specific coverage bar chart.")

    # 4.1.5 tail/spike bars from regime metrics.
    sec = "4.1.5"
    root = rq1_root / SUBSECTIONS[sec]
    tail_candidates = [
        root / "backup" / "csv" / f"tail_spike_metrics_{split}.csv",
        rq1_root / "_raw_outputs" / "4_1_5_tail_spike" / f"tail_spike_metrics_{split}.csv",
        rq1_root / "_raw_outputs" / "shared" / f"tail_spike_metrics_{split}.csv",
    ]
    tail = next((candidate for candidate in tail_candidates if candidate.exists()), tail_candidates[0])
    if tail.exists():
        df = pd.read_csv(tail)
        d = df.copy()
        d = sort_target_frame(d, target_col="target", extra_cols=["regime"])
        out = root / "result_section" / "latex_figures" / "tail_spike_relative_pinball_by_regime.tex"
        path = _write_tail_spike_relative_tex(
            out,
            data=d,
            caption="Tail and spike performance by regime. Values below 1 indicate lower mean pinball loss than RLQR, while values above 1 indicate worse performance.",
            label="fig:rq1-4-1-5-tail-spike-relative-pinball",
        )
        if path:
            _add_latex_figure_entry(
                entries,
                subsection=sec,
                tier="result_section",
                path=path,
                metric_family="relative_mean_pinball_loss",
                thesis_use="main thesis figure",
                description="LaTeX wrapper for split tail/spike relative pinball figures.",
            )
            for split_name, description in [
                ("tail_spike_relative_pinball_by_regime_da_price.tex", "Native pgfplots DA price tail/spike relative pinball figure."),
                ("tail_spike_relative_pinball_by_regime_price_capacity.tex", "Native pgfplots DA and aFRR capacity tail/spike relative pinball figure."),
                ("tail_spike_relative_pinball_by_regime_activation.tex", "Native pgfplots aFRR activation tail/spike relative pinball figure."),
                ("tail_spike_relative_pinball_by_regime_all_targets.tex", "Native pgfplots combined tail/spike relative pinball figure across all target variables."),
                ("tail_spike_relative_pinball_by_regime_all_targets_da_aggregated.tex", "Native pgfplots combined tail/spike relative pinball figure with DA spike directions and DA stress weeks aggregated."),
                ("tail_spike_relative_pinball_by_regime_capacity_price_aggregate.tex", "Native pgfplots aFRR capacity tail/spike relative pinball figure with positive and negative directions aggregated."),
                ("tail_spike_relative_pinball_by_regime_activation_price_aggregate.tex", "Native pgfplots aFRR activation-price tail/spike relative pinball figure with positive and negative directions aggregated."),
                ("tail_spike_relative_pinball_by_regime_activation_rate_aggregate.tex", "Native pgfplots aFRR activation-rate tail/spike relative pinball figure with positive and negative directions aggregated."),
            ]:
                split_path = path.parent / split_name
                if split_path.exists():
                    _add_latex_figure_entry(
                        entries,
                        subsection=sec,
                        tier="result_section",
                        path=split_path,
                        metric_family="relative_mean_pinball_loss",
                        thesis_use="main thesis figure",
                        description=description,
                    )
        d["x"] = d["regime"].astype(str) + " / " + d["target_group"].astype(str)
        x_order = d["x"].drop_duplicates().tolist()
        pivot = d.pivot_table(index="x", columns="model_label", values="coverage_p10_p90", aggfunc="mean").reset_index()
        pivot["x"] = pd.Categorical(pivot["x"], categories=x_order, ordered=True)
        pivot = pivot.sort_values("x")
        out = root / "appendix" / "latex_figures" / "tail_spike_coverage_by_regime.tex"
        path = _write_grouped_bar_tex(out, data=pivot, x_col="x", series_cols=[c for c in MODEL_LABELS if c in pivot.columns], caption="Tail/spike p10-p90 interval coverage by regime.", label="fig:rq1-4-1-5-tail-spike-coverage", ylabel="Coverage", placement="p")
        if path:
            _add_tikz_entry(entries, subsection=sec, tier="appendix", path=path, metric_family="interval_coverage", description="Native pgfplots tail/spike coverage bar chart.")


def _prune_latex_figure_imports(rq1_root: Path) -> None:
    for tex in rq1_root.rglob("*.tex"):
        if "latex_figures" not in tex.parts:
            continue
        if "4_1_6_example_weeks" in tex.parts:
            continue
        if tex.name.endswith("_p50_absolute_error_tolerance_curve.tex"):
            continue
        if tex.read_text(encoding="utf-8", errors="ignore").find(r"\includegraphics") >= 0:
            tex.unlink()


def _add_latex_figure_snippets(entries: list[dict[str, Any]], *, rq1_root: Path, split: str) -> None:
    """Generate thesis-facing LaTeX figure snippets from organized outputs."""
    _prune_latex_figure_imports(rq1_root)
    _generate_latex_figures(entries, rq1_root=rq1_root, split=split)


def organize(
    *,
    final_root: Path,
    rq1_root: Path,
    split: str,
    prune_legacy: bool = False,
    skip_csv: bool = False,
    skip_json: bool = False,
) -> dict[str, Any]:
    if not final_root.exists():
        raise FileNotFoundError(f"Missing RQ1 raw output source directory: {final_root}")
    if not any(final_root.rglob("*")):
        raise FileNotFoundError(f"RQ1 raw output source directory is empty: {final_root}")
    _ensure_structure(rq1_root)
    entries: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for route in build_routes(split):
        entry, miss = _copy_route(route, final_root=final_root, rq1_root=rq1_root)
        if entry is not None:
            entries.append(entry)
        if miss is not None:
            missing.append(miss)
    _add_derived_tables(entries, missing, rq1_root=rq1_root, split=split)
    _add_example_week_outputs(entries, missing, final_root=final_root, rq1_root=rq1_root)
    _add_latex_figure_snippets(entries, rq1_root=rq1_root, split=split)
    removed = prune_legacy_outputs(rq1_root=rq1_root, final_root=final_root) if prune_legacy else []
    manifest = {
        "description": "Organized RQ1 thesis benchmark output manifest.",
        "split": split,
        "root": str(rq1_root),
        "outputs": sorted(entries, key=lambda r: (r["subsection"], r["tier"], r["artifact_type"], r["path"])),
        "missing_outputs": sorted(missing, key=lambda r: (r["subsection"], r["tier"], r["artifact_type"], r["path"])),
        "removed_legacy_outputs": removed,
    }
    manifest_path = rq1_root / "rq1_output_manifest.json"
    if not skip_json:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    missing_path = rq1_root / "rq1_output_missing.csv"
    if missing and not skip_csv:
        pd.DataFrame(missing).to_csv(missing_path, index=False)
    elif missing_path.exists():
        missing_path.unlink()
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Organize final RQ1 outputs into thesis-facing tiers.")
    p.add_argument("--final-root", default="artifacts/benchmark/rq1_ml_model_benchmark")
    p.add_argument("--rq1-root", default="artifacts/benchmark/rq1_ml_model_benchmark")
    p.add_argument("--split", default="test")
    p.add_argument("--prune-legacy", action="store_true", help="Remove known generated legacy/unstructured copies after organizing.")
    p.add_argument("--skip-csv", action="store_true", help="Do not write organizer CSV reports.")
    p.add_argument("--skip-json", action="store_true", help="Do not write the organizer JSON manifest.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest = organize(
        final_root=Path(args.final_root),
        rq1_root=Path(args.rq1_root),
        split=str(args.split),
        prune_legacy=bool(args.prune_legacy),
        skip_csv=bool(args.skip_csv),
        skip_json=bool(args.skip_json),
    )
    if not args.skip_json:
        print(f"[OK] Organized RQ1 outputs: {Path(args.rq1_root) / 'rq1_output_manifest.json'}")
    else:
        print(f"[OK] Organized RQ1 outputs without writing JSON manifest: {Path(args.rq1_root)}")
    print(f"[OK] outputs={len(manifest['outputs'])} missing={len(manifest['missing_outputs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
