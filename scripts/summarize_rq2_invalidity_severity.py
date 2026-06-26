#!/usr/bin/env python3
"""Create thesis-ready RQ2 invalidity limitation outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.build_simulation_invalidity_severity import build_invalidity_context_table, parse_bool, write_invalidity_context_latex
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_simulation_invalidity_severity import build_invalidity_context_table, parse_bool, write_invalidity_context_latex


DEFAULT_RQ2_ROOT = Path("artifacts/final_benchmark/rq2/thesis_final_multi_2m_20260620T091938Z")


COMPACT_COLUMNS = [
    "Scenario",
    "Severity class",
    "Invalid hours (%)",
    "DA infeasible MWh / trade MWh (%)",
    "Fallback optimizations (%)",
    "Missed activation MWh (%)",
    "Max SoC violation (MWh)",
    "Max reserve shortfall (MW)",
]


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def _scenario_label(row: pd.Series) -> str:
    label = str(row.get("model_label", row.get("model", ""))).strip() or "Unknown"
    q = str(row.get("quantile", "")).strip()
    if parse_bool(row.get("is_benchmark")) is True:
        return label
    return f"{label} {q}".strip()


def _severity_or_validity_label(row: pd.Series) -> str:
    severity = str(row.get("invalidity_severity_class", "")).strip()
    if severity and severity.lower() != "nan":
        return severity
    sim_valid = parse_bool(row.get("simulation_valid"))
    reportable = parse_bool(row.get("thesis_reportable"))
    if sim_valid is True and reportable is True:
        return "valid"
    if reportable is False:
        return "invalid"
    return "unknown"


def _fmt_pct(value: Any) -> str:
    x = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(x):
        return "--"
    return f"{100.0 * float(x):.1f}"


def _fmt_num(value: Any, digits: int = 2) -> str:
    x = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(x):
        return "--"
    return f"{float(x):.{digits}f}"


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


def make_compact_table(summary: pd.DataFrame) -> pd.DataFrame:
    d = summary.copy()
    model_order = {"NAIVE": 0, "RHPF": 1, "RLQR": 2, "XGB": 3, "TFT": 4}
    d["_model_order"] = d["model_label"].map(model_order).fillna(99)
    d["_q_order"] = d["quantile"].astype(str).str.extract(r"p(\d+)")[0].astype(float).fillna(50)
    d = d.sort_values(["is_benchmark", "_model_order", "_q_order", "scenario"], ascending=[False, True, True, True])
    out = pd.DataFrame(
        {
            "Scenario": d.apply(_scenario_label, axis=1),
            "Severity class": d.apply(_severity_or_validity_label, axis=1),
            "Invalid hours (%)": d["combined_infeasibility_hours_share"].map(_fmt_pct),
            "DA infeasible MWh / trade MWh (%)": d["da_lockbook_infeasible_mwh_share_of_total_planned_trade"].map(_fmt_pct),
            "Fallback optimizations (%)": d["fallback_optimization_share"].map(_fmt_pct),
            "Missed activation MWh (%)": d["missed_activation_mwh_share"].map(_fmt_pct),
            "Max SoC violation (MWh)": d["max_soc_violation_mwh"].map(lambda x: _fmt_num(x, 2)),
            "Max reserve shortfall (MW)": d["max_reserve_headroom_shortfall_mw"].map(lambda x: _fmt_num(x, 2)),
        }
    )
    out.insert(0, "scenario_id", d["scenario"].astype(str).values)
    out.insert(1, "model_label", d["model_label"].astype(str).values)
    out.insert(2, "quantile", d["quantile"].astype(str).values)
    return out


def write_latex_table(compact: pd.DataFrame, path: Path) -> None:
    display = compact[COMPACT_COLUMNS].copy()
    header = [
        r"\textbf{Scenario}",
        r"\textbf{Severity class}",
        r"\textbf{\begin{tabular}[c]{@{}r@{}}Invalid\\hours (\%)\end{tabular}}",
        r"\textbf{\begin{tabular}[c]{@{}r@{}}DA infeas.\\share (\%)\end{tabular}}",
        r"\textbf{\begin{tabular}[c]{@{}r@{}}Fallback\\opt. (\%)\end{tabular}}",
        r"\textbf{\begin{tabular}[c]{@{}r@{}}Missed act.\\share (\%)\end{tabular}}",
        r"\textbf{\begin{tabular}[c]{@{}r@{}}Max SoC\\viol. (MWh)\end{tabular}}",
        r"\textbf{\begin{tabular}[c]{@{}r@{}}Max reserve\\shortfall (MW)\end{tabular}}",
    ]
    rows = []
    for _, row in display.iterrows():
        rows.append(
            " & ".join(
                [
                    _latex_escape(row["Scenario"]),
                    _latex_escape(row["Severity class"]),
                    _latex_escape(row["Invalid hours (%)"]),
                    _latex_escape(row["DA infeasible MWh / trade MWh (%)"]),
                    _latex_escape(row["Fallback optimizations (%)"]),
                    _latex_escape(row["Missed activation MWh (%)"]),
                    _latex_escape(row["Max SoC violation (MWh)"]),
                    _latex_escape(row["Max reserve shortfall (MW)"]),
                ]
            )
            + r" \\"
        )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{RQ2 invalidity severity diagnostics. Percentages are based on existing diagnostic outputs; unavailable values are shown as --. Ratios are calculated using the available recorded denominator; unavailable denominators are reported as missing rather than zero.}",
        r"\label{tab:rq2_invalidity_severity_summary}",
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    text = "\n".join(lines)
    if re.search(r"(^|[^A-Za-z])[-+]?inf([^A-Za-z]|$)", text, flags=re.IGNORECASE):
        raise ValueError("LaTeX invalidity severity table contains inf.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _reason_counts(summary: pd.DataFrame) -> pd.Series:
    reasons: list[str] = []
    for txt in summary["invalid_reason"].fillna("").astype(str):
        reasons.extend([r.strip() for r in txt.split(",") if r.strip()])
    return pd.Series(reasons, dtype="object").value_counts()


def _stat_line(label: str, series: pd.Series, *, pct: bool = True) -> str:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return f"- {label}: not computable from available diagnostics."
    if pct:
        return f"- {label}: median {100.0 * x.median():.1f}%, max {100.0 * x.max():.1f}%."
    return f"- {label}: median {x.median():.2f}, max {x.max():.2f}."


def write_text_summary(summary: pd.DataFrame, warnings_df: pd.DataFrame, path: Path) -> None:
    total = int(len(summary))
    reportable = summary["thesis_reportable"].map(parse_bool) if "thesis_reportable" in summary else pd.Series(dtype=object)
    invalid = int(reportable.eq(False).sum())
    is_benchmark = summary["is_benchmark"].map(parse_bool).fillna(False) if "is_benchmark" in summary else pd.Series(False, index=summary.index)
    model_based = summary.loc[~is_benchmark].copy()
    model_reportable = model_based["thesis_reportable"].map(parse_bool) if "thesis_reportable" in model_based else pd.Series(dtype=object)
    invalid_model = int(model_reportable.eq(False).sum())
    reasons = _reason_counts(summary).head(8)
    max_soc = pd.to_numeric(summary["max_soc_violation_mwh"], errors="coerce").replace([np.inf, -np.inf], np.nan).max()
    max_reserve = pd.to_numeric(summary["max_reserve_headroom_shortfall_mw"], errors="coerce").replace([np.inf, -np.inf], np.nan).max()
    missing_count = int(len(warnings_df)) if warnings_df is not None else 0
    lines = [
        "RQ2 invalidity limitation summary",
        "",
        f"- Total scenarios: {total}.",
        f"- Invalid / non-reportable scenarios: {invalid} ({100.0 * invalid / max(total, 1):.1f}%).",
        f"- Invalid / non-reportable model-based scenarios: {invalid_model} of {len(model_based)} ({100.0 * invalid_model / max(len(model_based), 1):.1f}%).",
        "- Most frequent invalid reasons: "
        + (", ".join(f"{idx} ({int(val)})" for idx, val in reasons.items()) if not reasons.empty else "none recorded")
        + ".",
        _stat_line("Combined infeasibility hours share", summary["combined_infeasibility_hours_share"]),
        _stat_line("Fallback optimization share", summary["fallback_optimization_share"]),
        _stat_line("Missed activation MWh share", summary["missed_activation_mwh_share"]),
        f"- Largest SoC violation: {_fmt_num(max_soc, 2)} MWh.",
        f"- Largest reserve headroom shortfall: {_fmt_num(max_reserve, 2)} MW/MWh-equivalent.",
        f"- Missing or unavailable severity metrics generated {missing_count} warning rows.",
        "",
        "Limitation statement:",
        "The economic results should be interpreted as diagnostic backtest evidence rather than fully validated physically feasible trading-performance estimates. "
        "The invalidity diagnostics show that model-based runs contain physical or optimization-related violations, including infeasible DA lockbook positions, fallback optimizations, missed activations, SoC violations or reserve headroom shortfalls. "
        "The severity metrics quantify these issues relative to total simulated hours and trading/activation volumes. Invalid runs should therefore not be described as fully valid trading-performance results, and physical feasibility should not be claimed where violations exist.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def update_manifest(root: Path, paths: dict[str, Path]) -> None:
    manifest_path = root / "rq2_output_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.setdefault("outputs", [])
    existing = {str(item.get("path", "")) for item in outputs if isinstance(item, dict)}
    created = datetime.now(timezone.utc).isoformat()
    entries = [
        ("appendix/tables/rq2_invalidity_severity_summary.tex", "appendix", "latex_table", "invalidity severity", "RQ2 limitations table"),
        ("backup/csv/rq2_invalidity_severity_compact.csv", "backup", "csv", "invalidity severity", "source data for limitations table"),
        ("backup/diagnostics/rq2_invalidity_limitation_summary.txt", "backup", "text", "invalidity severity", "thesis-compatible limitation text"),
    ]
    for rel, tier, artifact_type, metric_family, thesis_use in entries:
        if rel in existing:
            continue
        outputs.append(
            {
                "path": rel,
                "tier": tier,
                "artifact_type": artifact_type,
                "metric_family": metric_family,
                "thesis_use": thesis_use,
                "created_at_utc": created,
            }
        )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _resolve_input_path(root: Path, generic_rel: str, legacy_rel: str) -> tuple[Path, bool]:
    generic = root / generic_rel
    if generic.exists():
        return generic, False
    legacy = root / legacy_rel
    if legacy.exists():
        return legacy, True
    return generic, False


def build_outputs(root: Path) -> dict[str, Path]:
    summary_path, summary_legacy = _resolve_input_path(
        root,
        "backup/diagnostics/simulation_invalidity_severity_summary.csv",
        "backup/diagnostics/rq2_invalidity_severity_summary.csv",
    )
    hourly_path, hourly_legacy = _resolve_input_path(
        root,
        "backup/diagnostics/simulation_invalidity_severity_by_hour.csv",
        "backup/diagnostics/rq2_invalidity_severity_by_hour.csv",
    )
    warnings_path, warnings_legacy = _resolve_input_path(
        root,
        "backup/warnings/simulation_invalidity_severity_warnings.csv",
        "backup/warnings/rq2_invalidity_severity_warnings.csv",
    )
    summary = _read_required(summary_path)
    hourly = _read_required(hourly_path)
    warnings_df = pd.read_csv(warnings_path) if warnings_path.exists() else pd.DataFrame(columns=["scenario", "metric", "warning", "details"])
    legacy_inputs = [
        str(path)
        for path, used_legacy in [(summary_path, summary_legacy), (hourly_path, hourly_legacy), (warnings_path, warnings_legacy)]
        if used_legacy
    ]
    if legacy_inputs:
        legacy_warning = pd.DataFrame(
            [
                {
                    "scenario": "",
                    "metric": "input_filename",
                    "warning": "legacy_filename_used",
                    "details": f"Summarizer used legacy rq2_* input file: {path}",
                }
                for path in legacy_inputs
            ]
        )
        warnings_df = pd.concat([warnings_df, legacy_warning], ignore_index=True)

    if hourly.duplicated(["scenario", "timestamp_utc"]).any():
        raise ValueError("Hourly invalidity severity input has duplicate scenario/timestamp rows.")
    compact = make_compact_table(summary)
    if compact.replace([np.inf, -np.inf], np.nan).isna().all(axis=None):
        raise ValueError("Compact invalidity severity table is empty/non-computable.")

    compact_path = root / "backup" / "csv" / "rq2_invalidity_severity_compact.csv"
    latex_path = root / "appendix" / "tables" / "rq2_invalidity_severity_summary.tex"
    text_path = root / "backup" / "diagnostics" / "rq2_invalidity_limitation_summary.txt"
    context_csv_path = root / "backup" / "diagnostics" / "simulation_invalidity_context_table.csv"
    context_latex_path = root / "appendix" / "tables" / "simulation_invalidity_context_table.tex"
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    compact.to_csv(compact_path, index=False)
    write_latex_table(compact, latex_path)
    write_text_summary(summary, warnings_df, text_path)
    context_table = build_invalidity_context_table(summary, include_benchmarks=False)
    context_csv_path.parent.mkdir(parents=True, exist_ok=True)
    context_table.to_csv(context_csv_path, index=False)
    write_invalidity_context_latex(context_table, context_latex_path, "rq2")
    update_manifest(root, {"compact": compact_path, "latex": latex_path, "text": text_path})
    return {"latex": latex_path, "compact": compact_path, "summary": text_path, "context_table": context_latex_path}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize RQ2 invalidity severity diagnostics for thesis limitations.")
    p.add_argument("--rq2-root", default=str(DEFAULT_RQ2_ROOT))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = build_outputs(Path(args.rq2_root))
    for key, path in paths.items():
        print(f"[OK] {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
