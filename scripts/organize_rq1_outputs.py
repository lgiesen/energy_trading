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

from energy_trading.evaluation.style import THESIS_PALETTE


SUBSECTIONS = {
    "4.1.1": "4_1_1_full_unweighted",
    "4.1.2": "4_1_2_calibration_uncertainty",
    "4.1.3": "4_1_3_per_lead",
    "4.1.4": "4_1_4_gate_specific",
    "4.1.5": "4_1_5_tail_spike",
    "4.1.6": "4_1_6_interim_answer",
    "4.1.7": "4_1_7_example_weeks",
}

LEGACY_SUBSECTIONS = {
    "4.1.1": ["4_1_1_full_unweighted_metrics", "4_1_1_full_unweighted", "rq1/4_1_1_full_unweighted_metrics", "rq1/4_1_1_full_unweighted"],
    "4.1.2": ["4_1_2_calibration_uncertainty", "rq1/4_1_2_calibration_uncertainty"],
    "4.1.3": ["4_1_3_per_lead_hour", "4_1_3_per_lead", "rq1/4_1_3_per_lead_hour", "rq1/4_1_3_per_lead"],
    "4.1.4": ["4_1_4_gate_actionable", "4_1_4_gate_specific", "rq1/4_1_4_gate_actionable", "rq1/4_1_4_gate_specific"],
    "4.1.5": ["4_1_5_tail_spike", "rq1/4_1_5_tail_spike"],
    "4.1.7": ["4_1_7_example_weeks", "rq1/4_1_7_example_weeks"],
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
        _route("4.1.1", "result_section", "latex_table", f"tables/forecast_metrics_full_primary_{split}.tex", _sub_sources("4.1.1", f"latex/rq1_4_1_1_forecast_metrics_full_primary_{split}.tex"), "mean_pinball_loss", "main thesis table", "Primary full-sample mean pinball table."),
        _route("4.1.1", "appendix", "latex_table", f"tables/forecast_metrics_full_detailed_{split}.tex", _sub_sources("4.1.1", f"latex/rq1_4_1_1_forecast_metrics_full_detailed_{split}.tex"), "point_and_probabilistic_errors", "appendix table", "Detailed full-sample metrics."),
        _route("4.1.1", "backup", "csv", "csv/forecast_metrics_full_long.csv", _sub_sources("4.1.1", "csv/rq1_4_1_1_forecast_metrics_full_long.csv"), "all_full_metrics", "backup data", "Long-form full metrics CSV."),
        _route("4.1.1", "backup", "csv", f"csv/forecast_metrics_full_primary_{split}.csv", _sub_sources("4.1.1", f"csv/rq1_4_1_1_forecast_metrics_full_primary_{split}.csv"), "mean_pinball_loss", "backup data", "Primary full metrics CSV."),
        _route("4.1.1", "backup", "csv", f"csv/forecast_metrics_full_detailed_{split}.csv", _sub_sources("4.1.1", f"csv/rq1_4_1_1_forecast_metrics_full_detailed_{split}.csv"), "point_and_probabilistic_errors", "backup data", "Detailed full metrics CSV."),
        _route("4.1.1", "backup", "diagnostics", f"diagnostics/forecast_metrics_full_alignment_diagnostics_{split}.csv", _sub_sources("4.1.1", f"csv/rq1_4_1_1_forecast_metrics_full_alignment_diagnostics_{split}.csv"), "alignment", "diagnostics", "Alignment diagnostics for full metrics."),
        _route("4.1.2", "result_section", "figure", "figures/calibration_reliability_by_target_group.png", _sub_sources("4.1.2", "figures/rq1_4_1_2_calibration_reliability_by_target_group.png"), "calibration", "main thesis figure", "Quantile reliability by target group."),
        _route("4.1.2", "result_section", "figure", "figures/calibration_interval_coverage_by_target_group.png", _sub_sources("4.1.2", "figures/rq1_4_1_2_calibration_interval_coverage_by_target_group.png"), "interval_coverage", "main thesis figure", "Interval coverage by target group."),
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
        _route("4.1.4", "result_section", "figure", "figures/gate_bucket_pinball_by_target_group.png", _sub_sources("4.1.4", "figures/gate_bucket_pinball_by_target_group.png") + ["figures/gate_bucket_pinball_by_target_group.png"], "mean_pinball_loss", "main thesis figure", "Gate bucket mean pinball by target group."),
        _route("4.1.4", "result_section", "latex_table", f"tables/gate_bucket_metrics_{split}.tex", _sub_sources("4.1.4", f"latex/gate_bucket_metrics_{split}.tex") + [f"latex/gate_bucket_metrics_{split}.tex"], "mean_pinball_loss", "main thesis table", "Gate bucket mean pinball table."),
        _route("4.1.4", "appendix", "figure", "figures/gate_bucket_coverage_p10_p90_by_target_group.png", _sub_sources("4.1.4", "figures/gate_bucket_coverage_p10_p90_by_target_group.png") + ["figures/gate_bucket_coverage_p10_p90_by_target_group.png"], "interval_coverage", "appendix figure", "Gate bucket p10-p90 coverage."),
        _route("4.1.4", "appendix", "figure", "figures/gate_bucket_observed_leads.png", _sub_sources("4.1.4", "figures/gate_bucket_observed_leads.png") + ["figures/gate_bucket_observed_leads.png"], "observed_leads", "appendix figure", "Observed leads by gate bucket."),
        _route("4.1.4", "backup", "csv", f"csv/gate_bucket_metrics_{split}.csv", _sub_sources("4.1.4", f"gate_bucket_metrics_{split}.csv") + [f"gate_bucket_metrics_{split}.csv"], "gate_bucket_metrics", "backup data", "Gate bucket metrics CSV."),
        _route("4.1.4", "backup", "csv", "csv/gate_bucket_definitions.csv", _sub_sources("4.1.4", "gate_bucket_definitions.csv") + ["gate_bucket_definitions.csv"], "definitions", "backup data", "Gate bucket definitions."),
        _route("4.1.4", "backup", "diagnostics", "diagnostics/gate_bucket_row_counts.csv", _sub_sources("4.1.4", "gate_bucket_row_counts.csv") + ["gate_bucket_row_counts.csv"], "row_counts", "diagnostics", "Gate bucket row counts."),
        _route("4.1.4", "backup", "diagnostics", "diagnostics/gate_bucket_observed_leads.csv", _sub_sources("4.1.4", "gate_bucket_observed_leads.csv") + ["gate_bucket_observed_leads.csv"], "observed_leads", "diagnostics", "Gate bucket observed leads."),
        _route("4.1.4", "backup", "warnings", "warnings/gate_bucket_warnings.csv", _sub_sources("4.1.4", "gate_bucket_warnings.csv") + ["gate_bucket_warnings.csv"], "warnings", "warnings", "Gate bucket warnings."),
        _route("4.1.5", "result_section", "figure", "figures/tail_spike_relative_pinball_by_regime.png", _sub_sources("4.1.5", "figures/tail_spike_relative_pinball_by_regime.png") + ["figures/tail_spike_relative_pinball_by_regime.png"], "mean_pinball_loss", "main thesis figure", "Tail/spike relative mean pinball by regime."),
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
    return s


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
    rows: list[list[Any]] = []
    metrics = [("mean_pinball_loss", "Mean pinball loss"), ("mae_p50", "MAE p50"), ("rmse_p50", "RMSE p50"), ("bias_p50", "Bias p50")]
    for metric, label in metrics:
        if metric not in df.columns:
            continue
        pivot = _pivot_metric_rows(df, ["target_label", "lead_time_h"], metric)
        for _, row in pivot.iterrows():
            vals = {m: float(row[m]) for m in ["TFT", "XGB", "RLQR"] if m in row and pd.notna(row[m])}
            n = int(df[(df["target_label"].eq(row["target_label"])) & (df["lead_time_h"].eq(row["lead_time_h"]))]["n_obs"].min())
            rows.append([row["target_label"], int(row["lead_time_h"]), label, vals.get("TFT", np.nan), vals.get("XGB", np.nan), vals.get("RLQR", np.nan), _best(vals), n])
    return _write_table(out_path, ["Target", "Lead hour", "Metric", "TFT", "XGB", "RLQR", "Best model", "N"], rows, "Per-lead detailed forecast metrics on the test split.", "tab:per_lead_detailed_metrics_test")


def _derive_gate_interval_appendix(csv_path: Path, out_path: Path, split: str) -> Path | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        df = df[df["split"].eq(split)]
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
    rows: list[list[Any]] = []
    metrics = [
        ("mean_pinball_loss", "Mean pinball loss"),
        ("mae_p50", "MAE p50"),
        ("rmse_p50", "RMSE p50"),
        ("bias_p50", "Bias p50"),
        ("coverage_p10_p90", "p10-p90 coverage"),
        ("interval_width_p10_p90_mean", "p10-p90 width"),
    ]
    for metric, label in metrics:
        if metric not in df.columns:
            continue
        pivot = _pivot_metric_rows(df, ["regime", "target_label"], metric)
        for _, row in pivot.iterrows():
            vals = {m: float(row[m]) for m in ["TFT", "XGB", "RLQR"] if m in row and pd.notna(row[m])}
            mask = df["regime"].eq(row["regime"]) & df["target_label"].eq(row["target_label"])
            n = int(df[mask]["n_obs"].min())
            rows.append([row["regime"], row["target_label"], label, vals.get("TFT", np.nan), vals.get("XGB", np.nan), vals.get("RLQR", np.nan), _best(vals), n])
    return _write_table(out_path, ["Regime", "Target", "Metric", "TFT", "XGB", "RLQR", "Best model", "N"], rows, "Tail/spike detailed metrics on the test split.", "tab:tail_spike_detailed_metrics_test")


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
        "_raw_outputs",
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

    if final_root != rq1_root and final_root.exists():
        _remove_path(final_root, removed)
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
        final_root / "4_1_7_example_weeks",
        rq1_root / "4_1_7_example_weeks",
        rq1_root / "rq1" / "4_1_7_example_weeks",
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
    subsection = "4.1.7"
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
    if metrics.exists():
        dst = target_root / "backup" / "csv" / "example_week_metrics.csv"
        _copy_file(metrics, dst)
        entries.append({"subsection": subsection, "tier": "backup", "artifact_type": "csv", "path": str(dst), "metric_family": "example_weeks", "thesis_use": "backup data", "brief_description": "Example-week plot inventory and diagnostics."})
    manifest = source_dir / "example_week_manifest.json"
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


def _latex_color_name(role: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", role).lower()


def _latex_figure_label(subsection: str, figure_path: Path) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", figure_path.stem.lower()).strip("-")
    section = subsection.replace(".", "-")
    return f"fig:rq1-{section}-{stem}"


def _latex_figure_path(rq1_root: Path, figure_path: Path) -> str:
    rel = figure_path.relative_to(rq1_root).as_posix()
    return rf"\rqonefigroot/{rel}"


def _write_latex_figure_snippet(
    *,
    path: Path,
    figure_path: Path,
    rq1_root: Path,
    subsection: str,
    caption: str,
    placement: str,
) -> None:
    color_lines = [
        f"\\providecolor{{{_latex_color_name(role)}}}{{HTML}}{{{hex_color.lstrip('#').upper()}}}"
        for role, hex_color in THESIS_PALETTE.items()
    ]
    lines = [
        r"% Requires: \usepackage{graphicx}",
        r"% Requires: \usepackage{xcolor}",
        r"% Palette mirrored from src/energy_trading/visualization/style.py:",
        *color_lines,
        r"% Override this once in the thesis preamble if the figures are copied elsewhere.",
        r"\providecommand{\rqonefigroot}{artifacts/final_benchmark}",
        rf"\begin{{figure}}[{placement}]",
        r"    \centering",
        rf"    \includegraphics[width=\textwidth,height=0.82\textheight,keepaspectratio]{{{_latex_figure_path(rq1_root, figure_path)}}}",
        f"    \\caption{{{_latex_escape(caption)}}}",
        f"    \\label{{{_latex_figure_label(subsection, figure_path)}}}",
        r"\end{figure}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _add_latex_figure_snippets(entries: list[dict[str, Any]], *, rq1_root: Path) -> None:
    figure_entries = [
        entry
        for entry in list(entries)
        if entry.get("artifact_type") == "figure"
        and entry.get("tier") in {"result_section", "appendix"}
        and Path(str(entry.get("path", ""))).suffix.lower() in {".png", ".pdf"}
    ]
    for entry in figure_entries:
        figure_path = Path(str(entry["path"]))
        if not figure_path.exists():
            continue
        tier = str(entry["tier"])
        subsection = str(entry["subsection"])
        subsection_dir = rq1_root / SUBSECTIONS[subsection]
        figure_rel = figure_path.relative_to(subsection_dir / tier / "figures")
        snippet_rel = figure_rel.with_suffix(".tex")
        snippet_path = subsection_dir / tier / "latex_figures" / snippet_rel
        _write_latex_figure_snippet(
            path=snippet_path,
            figure_path=figure_path,
            rq1_root=rq1_root,
            subsection=subsection,
            caption=str(entry.get("brief_description", figure_path.stem.replace("_", " "))),
            placement="htbp" if tier == "result_section" else "p",
        )
        entries.append(
            {
                "subsection": subsection,
                "tier": tier,
                "artifact_type": "latex_figure",
                "path": str(snippet_path),
                "metric_family": entry.get("metric_family", "figure"),
                "thesis_use": "copy-paste figure code",
                "brief_description": f"LaTeX includegraphics snippet for {figure_path.name}.",
            }
        )


def organize(*, final_root: Path, rq1_root: Path, split: str, prune_legacy: bool = False) -> dict[str, Any]:
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
    _add_latex_figure_snippets(entries, rq1_root=rq1_root)
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
    p.add_argument("--final-root", default="artifacts/final_benchmark")
    p.add_argument("--rq1-root", default="artifacts/final_benchmark")
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
