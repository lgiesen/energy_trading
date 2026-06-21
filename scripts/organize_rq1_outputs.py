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
        _route("4.1.1", "result_section", "figure", "figures/da_price_p50_absolute_error_tolerance_curve.png", _sub_sources("4.1.1", "figures/rq1_4_1_1_da_price_p50_absolute_error_tolerance_curve.png"), "p50 absolute error tolerance", "secondary interpretability figure", "DA price cumulative absolute p50 error tolerance curve."),
        _route("4.1.1", "result_section", "figure", "figures/da_price_p50_absolute_error_tolerance_curve.pdf", _sub_sources("4.1.1", "figures/rq1_4_1_1_da_price_p50_absolute_error_tolerance_curve.pdf"), "p50 absolute error tolerance", "secondary interpretability figure", "DA price cumulative absolute p50 error tolerance curve PDF.", required=False),
        _route("4.1.1", "result_section", "latex_figure", "figures/da_price_p50_absolute_error_tolerance_curve.tex", _sub_sources("4.1.1", "figures/rq1_4_1_1_da_price_p50_absolute_error_tolerance_curve.tex"), "p50 absolute error tolerance", "secondary interpretability figure", "LaTeX includegraphics snippet for DA price p50 error tolerance curve."),
        _route("4.1.1", "appendix", "figure", f"figures/forecast_metrics_full_relative_mae_p50_{split}.png", _sub_sources("4.1.1", f"figures/rq1_4_1_1_forecast_metrics_full_relative_mae_p50_{split}.png"), "mae_p50", "appendix figure", "Relative p50 MAE against RLQR by target."),
        _route("4.1.1", "result_section", "latex_table", f"tables/forecast_metrics_full_primary_{split}.tex", _sub_sources("4.1.1", f"latex/rq1_4_1_1_forecast_metrics_full_primary_{split}.tex"), "mean_pinball_loss", "main thesis table", "Primary full-sample mean pinball table."),
        _route("4.1.1", "result_section", "latex_table", "tables/computational_cost_365d.tex", _sub_sources("4.1.1", "latex/rq1_4_1_1_computational_cost_365d.tex"), "computational_cost", "main thesis table", "Observed and 365-day-scaled computational cost table."),
        _route("4.1.1", "appendix", "latex_table", f"tables/forecast_metrics_full_detailed_{split}.tex", _sub_sources("4.1.1", f"latex/rq1_4_1_1_forecast_metrics_full_detailed_{split}.tex"), "point_and_probabilistic_errors", "appendix table", "Detailed full-sample metrics."),
        _route("4.1.1", "appendix", "latex_table", f"tables/da_price_p50_error_tolerance_summary_{split}.tex", _sub_sources("4.1.1", f"latex/rq1_4_1_1_da_price_p50_error_tolerance_summary_{split}.tex"), "p50 absolute error tolerance", "appendix table", "DA price p50 absolute error tolerance threshold summary."),
        _route("4.1.1", "backup", "csv", "csv/forecast_metrics_full_long.csv", _sub_sources("4.1.1", "csv/rq1_4_1_1_forecast_metrics_full_long.csv"), "all_full_metrics", "backup data", "Long-form full metrics CSV."),
        _route("4.1.1", "backup", "csv", f"csv/forecast_metrics_full_primary_{split}.csv", _sub_sources("4.1.1", f"csv/rq1_4_1_1_forecast_metrics_full_primary_{split}.csv"), "mean_pinball_loss", "backup data", "Primary full metrics CSV."),
        _route("4.1.1", "backup", "csv", f"csv/forecast_metrics_full_detailed_{split}.csv", _sub_sources("4.1.1", f"csv/rq1_4_1_1_forecast_metrics_full_detailed_{split}.csv"), "point_and_probabilistic_errors", "backup data", "Detailed full metrics CSV."),
        _route("4.1.1", "backup", "csv", "csv/computational_cost_365d.csv", _sub_sources("4.1.1", "csv/rq1_4_1_1_computational_cost_365d.csv"), "computational_cost", "backup data", "Observed and scaled computational cost source CSV."),
        _route("4.1.1", "backup", "csv", "csv/da_price_p50_absolute_error_tolerance_curve.csv", _sub_sources("4.1.1", "csv/rq1_4_1_1_da_price_p50_absolute_error_tolerance_curve.csv"), "p50 absolute error tolerance", "backup data", "DA price p50 absolute error tolerance curve source CSV."),
        _route("4.1.1", "backup", "csv", f"csv/da_price_p50_error_tolerance_summary_{split}.csv", _sub_sources("4.1.1", f"csv/rq1_4_1_1_da_price_p50_error_tolerance_summary_{split}.csv"), "p50 absolute error tolerance", "backup data", "DA price p50 absolute error tolerance summary CSV."),
        _route("4.1.1", "backup", "diagnostics", f"diagnostics/forecast_metrics_full_alignment_diagnostics_{split}.csv", _sub_sources("4.1.1", f"csv/rq1_4_1_1_forecast_metrics_full_alignment_diagnostics_{split}.csv"), "alignment", "diagnostics", "Alignment diagnostics for full metrics."),
        _route("4.1.2", "result_section", "figure", "figures/calibration_reliability_by_target.png", _sub_sources("4.1.2", "figures/rq1_4_1_2_calibration_reliability_by_target.png"), "calibration", "main thesis figure", "Quantile reliability by target."),
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
        _route("4.1.4", "result_section", "figure", "figures/gate_bucket_pinball_by_target_group.png", _sub_sources("4.1.4", "figures/gate_bucket_pinball_by_target_group.png") + ["figures/gate_bucket_pinball_by_target_group.png"], "relative_mean_pinball_loss", "main thesis figure", "Actionable forecast performance by market gate relative to RLQR."),
        _route("4.1.4", "result_section", "latex_table", f"tables/gate_bucket_metrics_{split}.tex", _sub_sources("4.1.4", f"latex/gate_bucket_metrics_{split}.tex") + [f"latex/gate_bucket_metrics_{split}.tex"], "mean_pinball_loss", "main thesis table", "Gate bucket mean pinball table."),
        _route("4.1.4", "appendix", "figure", "figures/gate_bucket_coverage_p10_p90_by_target_group.png", _sub_sources("4.1.4", "figures/gate_bucket_coverage_p10_p90_by_target_group.png") + ["figures/gate_bucket_coverage_p10_p90_by_target_group.png"], "interval_coverage", "appendix figure", "Gate bucket p10-p90 coverage."),
        _route("4.1.4", "appendix", "figure", "figures/gate_bucket_observed_leads.png", _sub_sources("4.1.4", "figures/gate_bucket_observed_leads.png") + ["figures/gate_bucket_observed_leads.png"], "observed_leads", "appendix figure", "Observed leads by gate bucket."),
        _route("4.1.4", "backup", "csv", f"csv/gate_bucket_metrics_{split}.csv", _sub_sources("4.1.4", f"gate_bucket_metrics_{split}.csv") + [f"gate_bucket_metrics_{split}.csv"], "gate_bucket_metrics", "backup data", "Gate bucket metrics CSV."),
        _route("4.1.4", "backup", "csv", "csv/gate_bucket_definitions.csv", _sub_sources("4.1.4", "gate_bucket_definitions.csv") + ["gate_bucket_definitions.csv"], "definitions", "backup data", "Gate bucket definitions."),
        _route("4.1.4", "backup", "diagnostics", "diagnostics/gate_bucket_row_counts.csv", _sub_sources("4.1.4", "gate_bucket_row_counts.csv") + ["gate_bucket_row_counts.csv"], "row_counts", "diagnostics", "Gate bucket row counts."),
        _route("4.1.4", "backup", "diagnostics", "diagnostics/gate_bucket_observed_leads.csv", _sub_sources("4.1.4", "gate_bucket_observed_leads.csv") + ["gate_bucket_observed_leads.csv"], "observed_leads", "diagnostics", "Gate bucket observed leads."),
        _route("4.1.4", "backup", "warnings", "warnings/gate_bucket_warnings.csv", _sub_sources("4.1.4", "gate_bucket_warnings.csv") + ["gate_bucket_warnings.csv"], "warnings", "warnings", "Gate bucket warnings."),
        _route("4.1.5", "result_section", "figure", "figures/tail_spike_relative_pinball_by_regime_main.png", _sub_sources("4.1.5", "figures/tail_spike_relative_pinball_by_regime_main.png") + ["figures/tail_spike_relative_pinball_by_regime_main.png"], "mean_pinball_loss", "main thesis figure", "Tail/spike relative mean pinball by regime."),
        _route("4.1.5", "result_section", "figure", "figures/tail_spike_residual_distribution_by_regime.png", _sub_sources("4.1.5", "figures/tail_spike_residual_distribution_by_regime.png") + ["figures/tail_spike_residual_distribution_by_regime.png"], "residuals", "main thesis figure", "Tail/spike residual distributions."),
        _route("4.1.5", "result_section", "latex_table", f"tables/tail_spike_metrics_{split}.tex", _sub_sources("4.1.5", f"latex/tail_spike_metrics_{split}.tex") + [f"latex/tail_spike_metrics_{split}.tex"], "mean_pinball_loss", "main thesis table", "Tail/spike mean pinball table."),
        _route("4.1.5", "appendix", "figure", "figures/tail_spike_coverage_by_regime.png", _sub_sources("4.1.5", "figures/tail_spike_coverage_by_regime.png") + ["figures/tail_spike_coverage_by_regime.png"], "interval_coverage", "appendix figure", "Tail/spike p10-p90 coverage."),
        _route("4.1.5", "appendix", "figure", "figures/tail_spike_mae_p50_by_regime.png", _sub_sources("4.1.5", "figures/tail_spike_mae_p50_by_regime.png") + ["figures/tail_spike_mae_p50_by_regime.png"], "mae_p50", "appendix figure", "Tail/spike p50 MAE."),
        _route("4.1.5", "appendix", "figure", "figures/tail_spike_forecast_band_selected_week.png", _sub_sources("4.1.5", "figures/tail_spike_forecast_band_*.png") + ["figures/tail_spike_forecast_band_*.png"], "forecast_band", "appendix figure", "Selected-week tail/spike forecast-band example.", required=False),
        _route("4.1.5", "backup", "csv", f"csv/tail_spike_metrics_{split}.csv", _sub_sources("4.1.5", f"tail_spike_metrics_{split}.csv") + [f"tail_spike_metrics_{split}.csv"], "tail_spike_metrics", "backup data", "Tail/spike metrics CSV."),
        _route("4.1.5", "backup", "csv", "csv/tail_spike_regime_definitions.csv", _sub_sources("4.1.5", "tail_spike_regime_definitions.csv") + ["tail_spike_regime_definitions.csv"], "definitions", "backup data", "Tail/spike regime definitions."),
        _route("4.1.5", "backup", "csv", "csv/tail_spike_thresholds.csv", _sub_sources("4.1.5", "tail_spike_thresholds.csv") + ["tail_spike_thresholds.csv"], "thresholds", "backup data", "Tail/spike thresholds."),
        _route("4.1.5", "backup", "diagnostics", "diagnostics/tail_spike_row_counts.csv", _sub_sources("4.1.5", "tail_spike_row_counts.csv") + ["tail_spike_row_counts.csv"], "row_counts", "diagnostics", "Tail/spike row counts."),
        _route("4.1.5", "backup", "diagnostics", "diagnostics/tail_spike_selected_weeks.csv", _sub_sources("4.1.5", "tail_spike_selected_weeks.csv") + ["tail_spike_selected_weeks.csv"], "selected_weeks", "diagnostics", "Tail/spike selected weeks."),
        _route("4.1.5", "backup", "diagnostics", "diagnostics/tail_spike_relative_pinball_by_regime_all_in_one.png", _sub_sources("4.1.5", "figures/tail_spike_relative_pinball_by_regime_all_in_one.png") + ["figures/tail_spike_relative_pinball_by_regime_all_in_one.png"], "relative_mean_pinball_loss", "diagnostic figure", "Dense all-in-one tail/spike relative mean pinball chart.", required=False),
        _route("4.1.5", "backup", "warnings", "warnings/tail_spike_warnings.csv", _sub_sources("4.1.5", "tail_spike_warnings.csv") + ["tail_spike_warnings.csv"], "warnings", "warnings", "Tail/spike warnings."),
    ]
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
    for label in ["aFRR capacity price", "aFRR activation price", "aFRR activation rate"]:
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


def _fmt(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "-"
    return f"{x:.4f}" if np.isfinite(x) else "-"


def _write_table(path: Path, headers: list[str], rows: list[list[Any]], caption: str, label: str) -> Path | None:
    if not rows:
        return None
    align = "l" * len(headers)
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        rf"    \begin{{tabular}}{{@{{}}{align}@{{}}}}",
        r"        \toprule",
        "        " + " & ".join(r"\textbf{" + _latex_escape(h) + "}" for h in headers) + r" \\",
        r"        \midrule",
    ]
    for row in rows:
        lines.append("        " + " & ".join(_latex_escape(v) if not isinstance(v, (float, int, np.floating, np.integer)) else _fmt(v) for v in row) + r" \\")
    lines.extend([r"        \bottomrule", r"    \end{tabular}", f"    \\caption{{{_latex_escape(caption)}}}", f"    \\label{{{_latex_escape(label)}}}", r"\end{table}", ""])
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
    metrics = source_dir / "example_week_metrics.csv"
    if not metrics.exists():
        metrics = source_dir / "backup" / "csv" / "example_week_metrics.csv"
    if metrics.exists():
        dst = target_root / "backup" / "csv" / "example_week_metrics.csv"
        _copy_file(metrics, dst)
        entries.append({"subsection": subsection, "tier": "backup", "artifact_type": "csv", "path": str(dst), "metric_family": "example_weeks", "thesis_use": "backup data", "brief_description": "Example-week plot inventory and diagnostics."})
    manifest = source_dir / "example_week_manifest.json"
    if not manifest.exists():
        manifest = source_dir / "backup" / "diagnostics" / "example_week_manifest.json"
    if manifest.exists():
        dst = target_root / "backup" / "diagnostics" / "example_week_manifest.json"
        _copy_file(manifest, dst)
        entries.append({"subsection": subsection, "tier": "backup", "artifact_type": "diagnostics", "path": str(dst), "metric_family": "example_weeks", "thesis_use": "diagnostics", "brief_description": "Example-week generation manifest."})
    figures_root = source_dir / "figures"
    if figures_root.exists():
        for src in sorted(figures_root.rglob("*.png")):
            rel = src.relative_to(figures_root)
            tier = "result_section" if rel.parts and rel.parts[0] == "typical" else "appendix"
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
            rel = src.relative_to(figures_root)
            if r"\includegraphics" in src.read_text(encoding="utf-8", errors="ignore"):
                continue
            tier = "result_section" if rel.parts and rel.parts[0] == "typical" else "appendix"
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
    return " ".join(replacements.get(part.lower(), part) for part in text.split())


BUCKET_LABELS = {
    "full_h1_48": "Full horizon (h1--h48)",
    "short_h1_8": "Short horizon (h1--h8)",
    "medium_h9_16": "Medium horizon (h9--h16)",
    "long_h17_48": "Long horizon (h17--h48)",
}


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


TAIL_SPIKE_REGIME_LABELS = {
    "normal": "Normal",
    "da_abs_tail_top5": "Abs. tail top 5%",
    "da_positive_spike_top5": "Positive spike top 5%",
    "da_negative_spike_bottom5": "Negative spike bottom 5%",
    "afrr_activation_price_abs_tail_top5": "Abs. tail top 5%",
    "activation_nonzero": "Activation nonzero",
    "activation_zero_or_nearzero": "Activation zero / near-zero",
    "high_volatility_week": "High-volatility week",
    "spike_week": "Spike week",
}

TAIL_SPIKE_MAIN_REGIMES = {
    "DA price": [
        "normal",
        "da_abs_tail_top5",
        "da_positive_spike_top5",
        "da_negative_spike_bottom5",
        "high_volatility_week",
        "spike_week",
    ],
    "aFRR capacity price": [
        "normal",
        "high_volatility_week",
        "spike_week",
    ],
    "aFRR activation price": [
        "normal",
        "afrr_activation_price_abs_tail_top5",
        "high_volatility_week",
        "spike_week",
    ],
    "aFRR activation rate": [
        "activation_zero_or_nearzero",
        "activation_nonzero",
        "high_volatility_week",
        "spike_week",
    ],
}


def _tail_spike_regime_label(regime: Any) -> str:
    return TAIL_SPIKE_REGIME_LABELS.get(str(regime), _tex_label(regime))


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


def _tikz_footer(caption: str, label: str) -> list[str]:
    return [
        r"        \end{tikzpicture}}",
        f"    \\caption{{{_latex_escape(thesis_titlecase(caption))}}}",
        f"    \\label{{{label}}}",
        r"\end{figure}",
        "",
    ]


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


def _symbolic_axis_options(labels: list[Any]) -> list[str]:
    symbols = [_tex_symbol(x) for x in labels]
    tick_labels = [_latex_escape(_tex_label(x)) for x in labels]
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
) -> Path | None:
    if data.empty or not series_cols:
        return None
    del colors
    series_cols = [c for c in ordered_model_labels(series_cols) if c in series_cols]
    labels = data[x_col].astype(str).tolist()
    lines = _tikz_header(caption, label, placement=placement)
    lines.extend(
        [
            r"            \begin{axis}[",
            r"                ybar,",
            r"                bar width=13pt,",
            r"                width=0.96\textwidth,",
            r"                height=8cm,",
            rf"                ylabel={{{_latex_escape(ylabel)}}},",
            r"                x tick label style={rotate=35, anchor=east},",
            r"                legend style={at={(0.5,1.08)}, anchor=south, legend columns=-1, draw=none, fill=none, text=black},",
            r"                legend cell align={left},",
            r"                area legend,",
            r"                axis lines*=left,",
            r"                ymin=0,",
            *_symbolic_axis_options(labels),
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
    y_tick_labels = ",".join(_latex_escape(label) for label in labels)

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
        r"                bar width=8pt,",
        r"                width=0.96\textwidth,",
        r"                height=7.2cm,",
        r"                xlabel={Mean pinball loss relative to RLQR},",
        r"                legend style={at={(0.5,1.16)}, anchor=south, legend columns=-1, draw=none, fill=none, text=black},",
        r"                legend cell align={left},",
        r"                area legend,",
        r"                axis lines*=left,",
        r"                xmin=0,",
        r"                grid=major,",
        rf"                symbolic y coords={{{y_symbol_list}}},",
        rf"                ytick={{{y_symbol_list}}},",
        rf"                yticklabels={{{y_tick_labels}}},",
        r"                y dir=reverse,",
        r"            ]",
    ]
    lines.append(rf"                \addplot[color=secondary, densely dotted, mark=none, line width=1.2pt] coordinates {{(1,{y_symbols[0]}) (1,{y_symbols[-1]})}};")
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
            f"    \\caption{{{_latex_escape(thesis_titlecase(caption))}}}",
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
    required = {"target_group", "regime", "model_label", "mean_pinball_loss"}
    if data.empty or not required.issubset(data.columns):
        return None
    d = data.copy()
    d["mean_pinball_loss"] = pd.to_numeric(d["mean_pinball_loss"], errors="coerce")
    d = d.dropna(subset=["mean_pinball_loss"])
    if d.empty:
        return None
    agg = (
        d.groupby(["target_group", "regime", "model_label"], as_index=False, sort=False)
        .agg(mean_pinball_loss=("mean_pinball_loss", "mean"))
    )
    pivot = agg.pivot_table(index=["target_group", "regime"], columns="model_label", values="mean_pinball_loss", aggfunc="mean").reset_index()
    if "RLQR" not in pivot.columns:
        return None
    denom = pd.to_numeric(pivot["RLQR"], errors="coerce")
    pivot = pivot.loc[denom.notna() & denom.abs().gt(1e-12)].copy()
    if pivot.empty:
        return None
    for model in ["XGB", "TFT"]:
        if model in pivot.columns:
            pivot[model] = pd.to_numeric(pivot[model], errors="coerce") / pd.to_numeric(pivot["RLQR"], errors="coerce")

    panels: list[tuple[str, pd.DataFrame, list[str], list[str]]] = []
    for group, regimes in TAIL_SPIKE_MAIN_REGIMES.items():
        part = pivot[pivot["target_group"].astype(str).eq(group) & pivot["regime"].astype(str).isin(regimes)].copy()
        if part.empty:
            continue
        order = {regime: i for i, regime in enumerate(regimes)}
        part["_order"] = part["regime"].astype(str).map(order).fillna(99)
        part = part.sort_values(["_order", "regime"]).reset_index(drop=True)
        labels = [_tail_spike_regime_label(regime) for regime in part["regime"]]
        symbols = [_tex_symbol(f"{group}_{label}") for label in labels]
        panels.append((group, part, labels, symbols))
    if not panels:
        return None

    lines = [
        r"% Requires: \usepackage{pgfplots}",
        r"% Requires: \usepackage{xcolor}",
        r"% Requires: \usepgfplotslibrary{groupplots}",
        r"% Recommended in preamble: \pgfplotsset{compat=1.18}",
        *_latex_color_defs(),
        rf"\begin{{figure}}[{placement}]",
        r"    \centering",
        r"    \begin{tikzpicture}",
        r"        \begin{groupplot}[",
        rf"            group style={{group size=1 by {len(panels)}, vertical sep=1.05cm}},",
        r"            width=0.92\linewidth,",
        r"            height=3.15cm,",
        r"            xlabel={Mean pinball loss relative to RLQR},",
        r"            legend style={at={(0.5,1.22)}, anchor=south, legend columns=-1, draw=none, fill=none, text=black},",
        r"            legend cell align={left},",
        r"            area legend,",
        r"            axis lines*=left,",
        r"            xmin=0,",
        r"            grid=major,",
        r"        ]",
    ]
    legend_written = False
    for group, part, labels, symbols in panels:
        symbol_list = ",".join(symbols)
        label_list = ",".join(_latex_escape(label) for label in labels)
        lines.extend(
            [
                r"            \nextgroupplot[",
                r"                xbar,",
                r"                bar width=6pt,",
                rf"                title={{{_latex_escape(thesis_titlecase(group))}}},",
                rf"                symbolic y coords={{{symbol_list}}},",
                rf"                ytick={{{symbol_list}}},",
                rf"                yticklabels={{{label_list}}},",
                r"                y dir=reverse,",
                r"            ]",
                rf"                \addplot[color=secondary, densely dotted, mark=none, line width=1.2pt] coordinates {{(1,{symbols[0]}) (1,{symbols[-1]})}};",
            ]
        )
        if not legend_written:
            lines.append(r"                \addlegendentry{RLQR baseline}")
        for model in ["XGB", "TFT"]:
            if model not in part.columns:
                continue
            color = _model_color_role(model)
            coords: list[str] = []
            for symbol, value in zip(symbols, pd.to_numeric(part[model], errors="coerce")):
                if pd.notna(value) and np.isfinite(float(value)):
                    coords.append(f"({_tex_num(value)},{symbol})")
            if not coords:
                continue
            lines.append(rf"                \addplot[xbar, fill={color}, draw={color}, area legend] coordinates {{{' '.join(coords)}}};")
            if not legend_written:
                lines.append(rf"                \addlegendentry{{{_latex_escape(model)}}}")
        legend_written = True
    lines.extend(
        [
            r"        \end{groupplot}",
            r"    \end{tikzpicture}",
            f"    \\caption{{{_latex_escape(caption)}}}",
            f"    \\label{{{label}}}",
            r"\end{figure}",
            "",
        ]
    )
    return _write_lines(path, lines)


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
            rf"                ylabel={{{_latex_escape(ylabel)}}},",
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
) -> Path | None:
    if data.empty:
        return None
    colors = {"TFT": "tertiary", "XGB": "primary", "RLQR": "secondary", "linear": "secondary", "tft": "tertiary", "xgb": "primary"}
    lines = [r"\begin{tikzpicture}"] if fragment_only else _tikz_header(caption, label, placement=placement)
    lines.extend(
        [
            r"            \begin{axis}[",
            r"                width=0.96\textwidth,",
            r"                height=7cm,",
            rf"                xlabel={{{_latex_escape(xlabel)}}},",
            rf"                ylabel={{{_latex_escape(ylabel)}}},",
            r"                legend style={at={(0.5,1.08)}, anchor=south, legend columns=-1, draw=none, fill=none, text=black},",
            r"                legend cell align={left},",
            r"                axis lines*=left,",
            r"                grid=major,",
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
        marker_style = rf"mark=*, mark options={{fill={color}, draw={color}}}" if show_markers else "mark=none"
        lines.append(rf"                \addplot[color={color}, {marker_style}, line width=1pt] coordinates {{{coords}}};")
        legends.append(_latex_escape(_tex_label(series)))
    if ideal_diagonal:
        xmin, xmax = xlim if xlim is not None else (0.0, 1.0)
        ymin, ymax = ylim if ylim is not None else (0.0, 1.0)
        lo = max(float(xmin), float(ymin))
        hi = min(float(xmax), float(ymax))
        lines.append(rf"                \addplot[color=neutraldark, dashed, mark=none, line width=1pt] coordinates {{({_tex_num(lo)},{_tex_num(lo)}) ({_tex_num(hi)},{_tex_num(hi)})}};")
        legends.append("Ideal")
    if legends:
        lines.append("                \\legend{" + ",".join(legends) + "}")
    if fragment_only:
        lines.extend([r"            \end{axis}", r"\end{tikzpicture}", ""])
    else:
        lines.extend([r"            \end{axis}", *_tikz_footer(caption, label)])
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
) -> Path | None:
    if data.empty:
        return None
    panels = ordered_unique(data[panel_col].dropna().astype(str).drop_duplicates().tolist())
    if not panels:
        return None
    colors = {"TFT": "tertiary", "XGB": "primary", "RLQR": "secondary", "linear": "secondary", "tft": "tertiary", "xgb": "primary"}
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
        r"            \begin{groupplot}[",
        rf"                group style={{group size=1 by {len(panels)}, vertical sep=1.0cm}},",
        r"                width=0.96\textwidth,",
        r"                height=4.2cm,",
        rf"                xlabel={{{_latex_escape(xlabel)}}},",
        rf"                ylabel={{{_latex_escape(ylabel)}}},",
        r"                legend style={at={(0.5,1.16)}, anchor=south, legend columns=-1, draw=none, fill=none, text=black},",
        r"                legend cell align={left},",
        r"                axis lines*=left,",
        r"                grid=major,",
        *([rf"                xmin={_tex_num(xlim[0])},", rf"                xmax={_tex_num(xlim[1])},"] if xlim is not None else []),
        *([rf"                ymin={_tex_num(ylim[0])},", rf"                ymax={_tex_num(ylim[1])},"] if ylim is not None else []),
        *(_percent_tick_options(percent_ticks) if percent_ticks is not None else []),
        r"            ]",
    ]
    legend_entries: list[str] = []
    for panel_i, panel in enumerate(panels):
        panel_df = data[data[panel_col].astype(str).eq(panel)].copy()
        lines.append(rf"                \nextgroupplot[title={{{_latex_escape(thesis_titlecase(_tex_label(panel)))}}}]")
        grouped = {str(series): group for series, group in panel_df.groupby(series_col, sort=False)}
        ordered_series = [s for s in ordered_model_labels(grouped.keys()) if s in grouped]
        ordered_series.extend([s for s in grouped if s not in ordered_series])
        if reference_y is not None:
            x_values = pd.to_numeric(panel_df[x_col], errors="coerce").dropna()
            if not x_values.empty:
                xmin = float(x_values.min())
                xmax = float(x_values.max())
                lines.append(rf"                    \addplot[color=secondary, densely dotted, mark=none, line width=1.2pt] coordinates {{({_tex_num(xmin)},{_tex_num(reference_y)}) ({_tex_num(xmax)},{_tex_num(reference_y)})}};")
                if panel_i == 0:
                    legend_entries.append(_latex_escape(_tex_label(reference_label or f"{reference_y:g}")))
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
            coords = " ".join(f"({_tex_num(x)},{_tex_num(y)})" for x, y in zip(group[x_col], group[y_col]))
            color = colors.get(series, "neutraldark")
            marker_style = rf"mark=*, mark options={{fill={color}, draw={color}}}" if show_markers else "mark=none"
            lines.append(rf"                    \addplot[color={color}, {marker_style}, line width=1pt] coordinates {{{coords}}};")
            if panel_i == 0:
                legend_entries.append(_latex_escape(_tex_label(series)))
        if ideal_diagonal:
            xmin, xmax = xlim if xlim is not None else (0.0, 1.0)
            ymin, ymax = ylim if ylim is not None else (0.0, 1.0)
            lo = max(float(xmin), float(ymin))
            hi = min(float(xmax), float(ymax))
            lines.append(rf"                    \addplot[color=neutraldark, dashed, mark=none, line width=1pt] coordinates {{({_tex_num(lo)},{_tex_num(lo)}) ({_tex_num(hi)},{_tex_num(hi)})}};")
            if panel_i == 0:
                legend_entries.append("Ideal")
    if legend_entries:
        lines.append("                \\legend{" + ",".join(legend_entries) + "}")
    lines.extend(
        [
            r"            \end{groupplot}",
            r"        \end{tikzpicture}}",
            f"    \\caption{{{_latex_escape(thesis_titlecase(caption))}}}",
            f"    \\label{{{label}}}",
            r"\end{figure}",
            "",
        ]
    )
    return _write_lines(path, lines)


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
            r"                colormap/Blues,",
            r"                x tick label style={rotate=35, anchor=east},",
            *_symbolic_axis_options(xs),
            "                symbolic y coords={" + ",".join(_tex_symbol(y) for y in ys) + "},",
            "                ytick={" + ",".join(_tex_symbol(y) for y in ys) + "},",
            "                yticklabels={" + ",".join(_latex_escape(y) for y in ys) + "},",
            r"            ]",
            r"                \addplot[matrix plot*, point meta=explicit] coordinates {",
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


def _generate_latex_figures(entries: list[dict[str, Any]], *, rq1_root: Path, split: str) -> None:
    # 4.1.1 relative full-sample pinball bar chart.
    sec = "4.1.1"
    root = rq1_root / SUBSECTIONS[sec]
    primary = root / "backup" / "csv" / f"forecast_metrics_full_primary_{split}.csv"
    if primary.exists():
        df = pd.read_csv(primary)
        df["target_label"] = df["target"].map(lambda x: _tex_label(str(x).replace("pred_", "")))
        df = sort_target_frame(df, target_col="target")
        for col in ["XGB", "TFT"]:
            df[col] = pd.to_numeric(df[col], errors="coerce") / pd.to_numeric(df["RLQR"], errors="coerce")
        out = root / "result_section" / "latex_figures" / f"forecast_metrics_full_relative_pinball_{split}.tex"
        path = _write_grouped_bar_tex(
            out,
            data=df,
            x_col="target_label",
            series_cols=["XGB", "TFT"],
            caption="Relative mean pinball loss by target (RLQR = 1; lower is better).",
            label="fig:rq1-4-1-1-forecast-metrics-full-relative-pinball",
            ylabel="Mean pinball loss relative to RLQR",
            reference_y=1.0,
            reference_label="RLQR",
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

    # 4.1.2 calibration and uncertainty.
    sec = "4.1.2"
    root = rq1_root / SUBSECTIONS[sec]
    cal = root / "backup" / "csv" / f"calibration_quantile_coverage_{split}.csv"
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
            )
            if path:
                _add_tikz_entry(entries, subsection=sec, tier="result_section", path=path, metric_family="calibration", description=f"Native pgfplots quantile reliability for {target_label}.")
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
    per_lead = root / "backup" / "csv" / f"per_lead_metrics_{split}.csv"
    if per_lead.exists():
        df = pd.read_csv(per_lead)
        for target_slug in sorted(df["target_slug"].dropna().unique(), key=target_sort_key):
            group = sort_target_frame(df[df["target_slug"].eq(target_slug)].copy(), target_col="target_label")
            for metric, stem, tier in [
                ("mean_pinball_loss", "per_lead_pinball", "result_section"),
                ("mae_p50", "per_lead_mae_p50", "appendix"),
                ("rmse_p50", "per_lead_rmse_p50", "appendix"),
            ]:
                out = root / tier / "latex_figures" / f"{stem}_{target_slug}.tex"
                target_label = _tex_label(group["target_label"].iloc[0])
                if group["target_label"].nunique(dropna=True) > 1:
                    path = _write_line_panel_tex(
                        out,
                        data=group,
                        panel_col="target_label",
                        x_col="lead_time_h",
                        y_col=metric,
                        series_col="model_label",
                        caption=f"{_tex_label(metric)} by lead hour for {_tex_label(group['target_group'].iloc[0])}.",
                        label=f"fig:rq1-4-1-3-{stem.replace('_','-')}-{target_slug.replace('_','-')}",
                        ylabel=_tex_label(metric),
                        placement="htbp" if tier == "result_section" else "p",
                        show_markers=metric != "mean_pinball_loss",
                    )
                else:
                    path = _write_line_tex(out, data=group, x_col="lead_time_h", y_col=metric, series_col="model_label", caption=f"{_tex_label(metric)} by lead hour for {target_label}.", label=f"fig:rq1-4-1-3-{stem.replace('_','-')}-{target_slug.replace('_','-')}", ylabel=_tex_label(metric), placement="htbp" if tier == "result_section" else "p", show_markers=metric != "mean_pinball_loss")
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
                    )
                else:
                    path = _write_line_tex(out, data=rel, x_col="lead_time_h", y_col="relative", series_col="model_label", caption=f"Relative mean pinball loss by lead hour for {target_label} (RLQR = 1).", label=f"fig:rq1-4-1-3-per-lead-relative-pinball-{target_slug.replace('_','-')}", ylabel="Mean pinball loss relative to RLQR", reference_y=1.0, reference_label="RLQR")
                if path:
                    _add_tikz_entry(entries, subsection=sec, tier="result_section", path=path, metric_family="relative_mean_pinball_loss", description=f"Native pgfplots relative per-lead pinball line chart for {target_slug}.")

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
                caption="Actionable Forecast Performance by Market Gate Relative to RLQR.",
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
    tail = root / "backup" / "csv" / f"tail_spike_metrics_{split}.csv"
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
            _add_tikz_entry(entries, subsection=sec, tier="result_section", path=path, metric_family="relative_mean_pinball_loss", description="Native pgfplots tail/spike relative pinball horizontal bar chart.")
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
        if tex.read_text(encoding="utf-8", errors="ignore").find(r"\includegraphics") >= 0:
            tex.unlink()


def _add_latex_figure_snippets(entries: list[dict[str, Any]], *, rq1_root: Path, split: str) -> None:
    """Generate native LaTeX figures from CSV data.

    Deliberately does not emit includegraphics wrappers. If the source figure is
    image-only and the plotted data are not available, no LaTeX figure is
    generated.
    """
    _prune_latex_figure_imports(rq1_root)
    _generate_latex_figures(entries, rq1_root=rq1_root, split=split)


def organize(*, final_root: Path, rq1_root: Path, split: str, prune_legacy: bool = False) -> dict[str, Any]:
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
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    missing_path = rq1_root / "rq1_output_missing.csv"
    if missing:
        pd.DataFrame(missing).to_csv(missing_path, index=False)
    elif missing_path.exists():
        missing_path.unlink()
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Organize final RQ1 outputs into thesis-facing tiers.")
    p.add_argument("--final-root", default="artifacts/rq1_ml_model_benchmark")
    p.add_argument("--rq1-root", default="artifacts/rq1_ml_model_benchmark")
    p.add_argument("--split", default="test")
    p.add_argument("--prune-legacy", action="store_true", help="Remove known generated legacy/unstructured copies after organizing.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest = organize(final_root=Path(args.final_root), rq1_root=Path(args.rq1_root), split=str(args.split), prune_legacy=bool(args.prune_legacy))
    print(f"[OK] Organized RQ1 outputs: {Path(args.rq1_root) / 'rq1_output_manifest.json'}")
    print(f"[OK] outputs={len(manifest['outputs'])} missing={len(manifest['missing_outputs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
