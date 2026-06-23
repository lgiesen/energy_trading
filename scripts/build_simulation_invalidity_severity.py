#!/usr/bin/env python3
"""Extract invalidity severity diagnostics directly from simulation run folders.

This is a reporting layer only. It reads completed simulation outputs and never
runs simulations, backtests, model training or HPO.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RUN_ROOT = Path("artifacts/simulation_runs/thesis_final_multi_2m_20260620T091938Z")
DEFAULT_OUT_ROOT = Path("artifacts/final_benchmark/rq2/thesis_final_multi_2m_20260620T091938Z")

MODEL_LABELS = {"linear": "RLQR", "xgb": "XGB", "tft": "TFT"}
BENCHMARK_LABELS = {"benchmarks_naive": ("naive", "Naive"), "benchmarks_rhpf": ("rhpf", "RHPF")}

REQUESTED_CANDIDATE_FILES = [
    "backtest_hourly.parquet",
    "backtest_hourly.csv",
    "validation_events.csv",
    "strict_validity_events.csv",
    "simulation_summary.csv",
    "summary.json",
    "optimization_log.csv",
    "metrics.csv",
    "report.json",
    "manifest.json",
]

PROJECT_CANDIDATE_FILES = [
    "backtest_summary.json",
    "performance_metrics.csv",
    "performance_paths_long.csv",
    "da_precommit_debug.csv",
    "optimization_failure_debug.csv",
    "optimization_infeasibility_attribution.csv",
    "realized_ledger.parquet",
    "realized_ledger.csv",
    "reserve_commitment_debug.csv",
]

CANDIDATE_FILES = REQUESTED_CANDIDATE_FILES + [p for p in PROJECT_CANDIDATE_FILES if p not in REQUESTED_CANDIDATE_FILES]

HOURLY_COLUMNS = [
    "scenario",
    "model",
    "model_label",
    "quantile",
    "timestamp_utc",
    "da_lockbook_infeasible_flag",
    "da_lockbook_infeasible_mwh",
    "fallback_optimization_flag",
    "optimization_infeasible_flag",
    "missed_activation_flag",
    "missed_activation_mwh",
    "activation_event_flag",
    "activation_mwh",
    "soc_violation_flag",
    "soc_violation_mwh",
    "reserve_headroom_shortfall_flag",
    "reserve_headroom_shortfall_mw",
    "combined_infeasibility_flag",
    "source_files_used",
]

SUMMARY_COLUMNS = [
    "scenario",
    "model",
    "model_label",
    "quantile",
    "is_benchmark",
    "simulation_valid",
    "thesis_reportable",
    "invalid_reason",
    "start_time_utc",
    "end_time_utc",
    "total_hours",
    "expected_hours",
    "coverage_share",
    "missing_hours_count",
    "combined_infeasibility_hours",
    "combined_infeasibility_hours_share",
    "da_lockbook_infeasible_hours",
    "da_lockbook_infeasible_mwh",
    "total_planned_trade_mwh",
    "total_planned_trade_mwh_source",
    "da_lockbook_infeasible_mwh_share_of_total_planned_trade",
    "fallback_optimization_count",
    "total_optimization_count",
    "total_optimization_count_source",
    "fallback_optimization_count_source",
    "fallback_present_scenario_flag",
    "fallback_optimization_share",
    "missed_activation_count",
    "total_activation_count",
    "activation_count_semantics",
    "missed_activation_count_semantics",
    "missed_activation_count_share",
    "missed_activation_mwh",
    "total_activation_mwh",
    "total_activation_mwh_source",
    "missed_activation_mwh_share",
    "soc_violation_hours",
    "max_soc_violation_mwh",
    "sum_soc_violation_mwh",
    "reserve_headroom_shortfall_hours",
    "max_reserve_headroom_shortfall_mw",
    "sum_reserve_headroom_shortfall_mw_hours",
    "denominator_completeness_status",
    "invalidity_severity_class",
    "diagnostic_completeness_status",
    "missing_source_columns",
    "source_files_used",
]


@dataclass(frozen=True)
class ScenarioInfo:
    scenario: str
    scenario_root: Path
    output_dir: Path
    model: str
    model_label: str
    quantile: str
    is_benchmark: bool


def normalize_scenario_name(name: str) -> str:
    return re.sub(r"\s+", "", str(name).strip())


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    if text == "":
        return None
    return None


def _bool_to_output(value: Any) -> bool | float:
    parsed = parse_bool(value)
    return parsed if parsed is not None else float("nan")


def parse_scenario_folder(folder: Path) -> tuple[str, str, str, bool, list[dict[str, Any]]]:
    raw = folder.name
    normalized = normalize_scenario_name(raw)
    warnings: list[dict[str, Any]] = []
    if normalized != raw:
        warnings.append(
            {
                "scenario": raw,
                "metric": "scenario_name",
                "warning": "normalized_folder_name",
                "details": f"Normalized scenario folder name from {raw!r} to {normalized!r}.",
            }
        )
    if normalized in BENCHMARK_LABELS:
        model, label = BENCHMARK_LABELS[normalized]
        return model, label, "benchmark", True, warnings
    match = re.fullmatch(r"(?P<model>linear|xgb|tft)_+?(?P<quantile>p\d+)", normalized)
    if match:
        model = match.group("model")
        return model, MODEL_LABELS[model], match.group("quantile"), False, warnings
    warnings.append(
        {
            "scenario": raw,
            "metric": "scenario_name",
            "warning": "unrecognized_scenario_name",
            "details": "Could not infer model/quantile from folder name.",
        }
    )
    return normalized, normalized.upper(), "", False, warnings


def _candidate_dirs(scenario_root: Path) -> list[Path]:
    dirs = [scenario_root]
    dirs.extend(sorted(p for p in scenario_root.glob("*") if p.is_dir()))
    dirs.extend(sorted(p for p in scenario_root.glob("*/*") if p.is_dir()))
    # Preserve order while removing duplicates.
    seen: set[Path] = set()
    out: list[Path] = []
    for path in dirs:
        if path not in seen:
            out.append(path)
            seen.add(path)
    return out


def _dir_has_candidate(path: Path) -> bool:
    return any((path / name).exists() for name in CANDIDATE_FILES)


def discover_scenarios(run_root: Path) -> tuple[list[ScenarioInfo], list[dict[str, Any]]]:
    if not run_root.exists():
        raise FileNotFoundError(f"Missing run root: {run_root}")
    scenarios: list[ScenarioInfo] = []
    warnings: list[dict[str, Any]] = []
    for child in sorted(p for p in run_root.iterdir() if p.is_dir()):
        if child.name == "logs":
            continue
        model, label, quantile, is_benchmark, parse_warnings = parse_scenario_folder(child)
        warnings.extend(parse_warnings)
        candidate_output_dirs = [p for p in _candidate_dirs(child) if _dir_has_candidate(p)]
        if not candidate_output_dirs:
            warnings.append(
                {
                    "scenario": child.name,
                    "metric": "scenario_discovery",
                    "warning": "no_candidate_files",
                    "details": "No targeted simulation output files found at depth <= 2 below scenario folder.",
                }
            )
            continue
        output_dir = sorted(candidate_output_dirs, key=lambda p: (0 if (p / "backtest_hourly.parquet").exists() else 1, len(p.parts)))[0]
        scenarios.append(
            ScenarioInfo(
                scenario=normalize_scenario_name(child.name),
                scenario_root=child,
                output_dir=output_dir,
                model=model,
                model_label=label,
                quantile=quantile,
                is_benchmark=is_benchmark,
            )
        )
    return scenarios, warnings


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_table(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if path.suffix == ".json":
        payload = _read_json(path)
        return pd.DataFrame([payload]) if payload else pd.DataFrame()
    return pd.DataFrame()


def _timestamp_col(df: pd.DataFrame) -> str | None:
    candidates = ["timestamp_utc", "target_time_utc", "delivery_time_utc", "delivery_start_utc", "date"]
    for col in candidates:
        if col in df.columns:
            return col
    normalized = {_normalize_col(c): c for c in df.columns}
    for key in ["timestamputc", "targettimeutc", "deliverytimeutc", "deliverystartutc"]:
        if key in normalized:
            return normalized[key]
    return None


def _normalize_col(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    normalized = {_normalize_col(c): c for c in df.columns}
    for col in candidates:
        match = normalized.get(_normalize_col(col))
        if match:
            return match
    return None


def _find_cols(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    out: list[str] = []
    for col in candidates:
        found = _find_col(df, [col])
        if found and found not in out:
            out.append(found)
    return out


def _normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    col = _timestamp_col(df)
    if col is None:
        return df
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out[col], utc=True, errors="coerce")
    return out.dropna(subset=["timestamp_utc"]).copy()


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _abs_sum(df: pd.DataFrame, candidates: list[str]) -> tuple[pd.Series | None, list[str]]:
    cols = _find_cols(df, candidates)
    if not cols:
        return None, []
    out = pd.Series(0.0, index=df.index, dtype="float64")
    for col in cols:
        out = out + _num(df[col]).abs()
    return out, cols


def _bool_flag(df: pd.DataFrame, candidates: list[str], *, contains: str | None = None) -> tuple[pd.Series | None, list[str]]:
    cols = _find_cols(df, candidates)
    if not cols:
        return None, []
    out = pd.Series(False, index=df.index)
    for col in cols:
        if contains:
            out = out | df[col].fillna("").astype(str).str.contains(contains, case=False, regex=True)
        else:
            parsed = df[col].map(parse_bool)
            numeric = _num(df[col])
            col_flag = parsed.fillna(numeric.abs() > 1e-9).fillna(False).astype(bool)
            out = out | col_flag
    return out.astype(float), cols


def _by_timestamp(df: pd.DataFrame, values: pd.Series, *, op: str) -> pd.Series:
    d = pd.DataFrame({"timestamp_utc": df["timestamp_utc"], "value": values})
    if op == "max":
        return d.groupby("timestamp_utc")["value"].max()
    return d.groupby("timestamp_utc")["value"].sum()


def _safe_ratio(num: float, den: float, *, scenario: str, metric: str, warnings: list[dict[str, Any]]) -> float:
    if not np.isfinite(num):
        warnings.append({"scenario": scenario, "metric": metric, "warning": "numerator_unavailable", "details": "Numerator is unavailable; wrote NaN."})
        return float("nan")
    if not np.isfinite(den):
        warnings.append({"scenario": scenario, "metric": metric, "warning": "denominator_unavailable", "details": "Denominator is unavailable; wrote NaN."})
        return float("nan")
    if abs(float(den)) <= 1e-12:
        warnings.append({"scenario": scenario, "metric": metric, "warning": "zero_denominator", "details": "Denominator is zero; wrote NaN."})
        return float("nan")
    value = float(num) / float(den)
    if not np.isfinite(value):
        warnings.append({"scenario": scenario, "metric": metric, "warning": "nonfinite_share", "details": ""})
        return float("nan")
    return value


def _warn(warnings: list[dict[str, Any]], scenario: str, metric: str, warning: str, details: str) -> None:
    warnings.append({"scenario": scenario, "metric": metric, "warning": warning, "details": details})


def _metric_source(
    rows: list[dict[str, Any]],
    *,
    scenario: str,
    metric_name: str,
    metric_status: str,
    source_file: Path | str | None = None,
    source_column: str | list[str] | None = None,
    source_semantics: str = "",
    source_priority: int = 1,
    fallback_used: bool = False,
    warning: str = "",
) -> None:
    rows.append(
        {
            "scenario": scenario,
            "metric_name": metric_name,
            "metric_status": metric_status,
            "source_file": "" if source_file is None else str(source_file),
            "source_column": ";".join(source_column) if isinstance(source_column, list) else ("" if source_column is None else str(source_column)),
            "source_semantics": source_semantics,
            "source_priority": source_priority,
            "fallback_used": bool(fallback_used),
            "warning": warning,
        }
    )


def _inventory(info: ScenarioInfo) -> tuple[pd.DataFrame, dict[str, Path]]:
    rows: list[dict[str, Any]] = []
    used: dict[str, Path] = {}
    for directory in _candidate_dirs(info.scenario_root):
        for name in CANDIDATE_FILES:
            path = directory / name
            exists = path.exists()
            row_count = np.nan
            columns: list[str] = []
            ts_col = ""
            supported: list[str] = []
            reason = "missing"
            if exists:
                try:
                    sample = _read_table(path, nrows=5 if path.suffix == ".csv" else None)
                    row_count = int(len(sample))
                    columns = list(sample.columns)
                    ts_col = _timestamp_col(sample) or ""
                    reason = ""
                    if name.startswith("backtest_hourly"):
                        supported += ["total_hours", "fallback_optimization", "missed_activation", "soc_violation", "reserve_headroom_shortfall"]
                        used.setdefault("hourly", path)
                    elif name == "da_precommit_debug.csv":
                        supported.append("da_lockbook_infeasible")
                        used.setdefault("da_precommit", path)
                    elif name in {"optimization_log.csv", "optimization_failure_debug.csv", "optimization_infeasibility_attribution.csv"}:
                        supported.append("optimization_infeasible")
                        used.setdefault("optimization", path)
                    elif name.startswith("realized_ledger"):
                        supported.append("activation_volume")
                        used.setdefault("realized_ledger", path)
                    elif name == "reserve_commitment_debug.csv":
                        supported.append("reserve_headroom_shortfall")
                        used.setdefault("reserve", path)
                    elif name in {"performance_metrics.csv", "metrics.csv"}:
                        supported.append("scenario_metrics")
                        used.setdefault("metrics", path)
                    elif name in {"backtest_summary.json", "summary.json", "simulation_summary.csv", "report.json", "manifest.json"}:
                        supported.append("validity_summary")
                        used.setdefault("summary", path)
                    if not supported:
                        reason = "file_exists_but_no_supported_invalidity_metrics_detected"
                except Exception as exc:
                    reason = f"read_error:{type(exc).__name__}:{exc}"
            rows.append(
                {
                    "scenario": info.scenario,
                    "candidate_file": str(path),
                    "exists": bool(exists),
                    "used": bool(supported),
                    "row_count": row_count,
                    "timestamp_column": ts_col,
                    "available_columns": ";".join(columns),
                    "metrics_supported": ";".join(supported),
                    "reason_if_not_used": reason,
                }
            )
    selected_paths = {str(path) for path in used.values()}
    for row in rows:
        row["used"] = str(row["candidate_file"]) in selected_paths
        if row["exists"] and row["metrics_supported"] and not row["used"]:
            row["reason_if_not_used"] = "supported_file_exists_but_another_candidate_was_selected"
    return pd.DataFrame(rows), used


def _empty_hourly() -> pd.DataFrame:
    return pd.DataFrame(columns=HOURLY_COLUMNS)


def _series_to_hourly(hourly_out: pd.DataFrame, series: pd.Series, column: str) -> None:
    hourly_out[column] = hourly_out["timestamp_utc"].map(series) if not series.empty else np.nan


def _sum_if_available(hourly_out: pd.DataFrame, col: str) -> float:
    if col not in hourly_out.columns or hourly_out.empty or hourly_out[col].isna().all():
        return float("nan")
    return float(pd.to_numeric(hourly_out[col], errors="coerce").sum(skipna=True))


def _count_flag(hourly_out: pd.DataFrame, col: str) -> float:
    if col not in hourly_out.columns or hourly_out.empty or hourly_out[col].isna().all():
        return float("nan")
    return float(pd.to_numeric(hourly_out[col], errors="coerce").fillna(0.0).gt(0.5).sum())


def _max_if_available(hourly_out: pd.DataFrame, col: str) -> float:
    if col not in hourly_out.columns or hourly_out.empty or hourly_out[col].isna().all():
        return float("nan")
    return float(pd.to_numeric(hourly_out[col], errors="coerce").max(skipna=True))


def _extract_validity(summary_path: Path | None, metrics: dict[str, Any]) -> tuple[bool | float, bool | float, str]:
    payload = _read_json(summary_path) if summary_path and summary_path.suffix == ".json" else {}
    if summary_path and summary_path.exists() and summary_path.suffix != ".json":
        df = _read_table(summary_path)
        if not df.empty:
            payload.update(df.iloc[0].to_dict())
    valid = payload.get("simulation_valid", metrics.get("simulation_valid", np.nan))
    reportable = payload.get("thesis_reportable", metrics.get("thesis_reportable", np.nan))
    reason = payload.get("invalid_reason", metrics.get("invalid_reason", ""))
    return _bool_to_output(valid), _bool_to_output(reportable), str(reason)


def _warn_nan_metrics(row: dict[str, Any], warnings: list[dict[str, Any]]) -> None:
    for metric in [
        "total_hours",
        "combined_infeasibility_hours",
        "combined_infeasibility_hours_share",
        "da_lockbook_infeasible_mwh",
        "total_planned_trade_mwh",
        "da_lockbook_infeasible_mwh_share_of_total_planned_trade",
        "fallback_optimization_share",
        "missed_activation_mwh",
        "total_activation_mwh",
        "missed_activation_mwh_share",
        "max_soc_violation_mwh",
        "max_reserve_headroom_shortfall_mw",
    ]:
        value = row.get(metric)
        if not np.isfinite(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]):
            _warn(warnings, str(row.get("scenario", "")), metric, "metric_output_nan", "Metric could not be computed from available raw simulation outputs; wrote NaN.")


def _planned_trade_total(
    metrics: dict[str, Any],
    hourly: pd.DataFrame,
    info: ScenarioInfo,
    warnings: list[dict[str, Any]],
) -> tuple[float, str, list[str]]:
    metric_cols = [
        "da_bid_abs_mwh_total",
        "bem_bid_abs_mwh_total",
        "id_abs_mwh_total",
        "bcm_bid_abs_mwh_total",
        "planned_trade_abs_mwh_total",
    ]
    vals = [pd.to_numeric(pd.Series([metrics.get(c)]), errors="coerce").iloc[0] for c in metric_cols if c in metrics]
    vals = [float(v) for v in vals if np.isfinite(v)]
    if vals:
        used = [c for c in metric_cols if c in metrics]
        return float(sum(abs(v) for v in vals)), "planned", used
    planned_candidates = [
        "da_locked_buy_mwh_requested",
        "da_locked_sell_mwh_requested",
        "da_precommit_da_candidate_buy_mw",
        "da_precommit_da_candidate_sell_mw",
        "real_bcm_candidate_pos_mw",
        "real_bcm_candidate_neg_mw",
        "real_bem_candidate_pos_mw",
        "real_bem_candidate_neg_mw",
        "id_buy_mwh",
        "id_sell_mwh",
    ]
    planned, used_cols = _abs_sum(hourly, planned_candidates) if not hourly.empty else (None, [])
    if planned is not None:
        _warn(warnings, info.scenario, "total_planned_trade_mwh", "planned_trade_hourly_proxy", f"Used hourly planned-volume proxy columns: {used_cols}")
        return float(planned.sum(skipna=True)), "planned", used_cols
    realized_candidates = ["real_da_buy_mwh", "real_da_sell_mwh", "real_bem_only_executed_pos_mwh", "real_bem_only_executed_neg_mwh", "real_bcm_linked_pos_activation_mwh", "real_bcm_linked_neg_activation_mwh"]
    realized, used_realized = _abs_sum(hourly, realized_candidates) if not hourly.empty else (None, [])
    if realized is not None:
        _warn(warnings, info.scenario, "total_planned_trade_mwh", "realized_volume_fallback", f"Planned volumes unavailable; used realized-volume fallback columns: {used_realized}")
        return float(realized.sum(skipna=True)), "realized_fallback", used_realized
    _warn(warnings, info.scenario, "total_planned_trade_mwh", "missing_source_columns", "No planned or realized trade volume columns found.")
    return float("nan"), "unavailable", []


def _coverage_stats(hourly_out: pd.DataFrame, info: ScenarioInfo, warnings: list[dict[str, Any]]) -> tuple[str, str, float, float, float]:
    if hourly_out.empty or "timestamp_utc" not in hourly_out.columns:
        _warn(warnings, info.scenario, "coverage_share", "denominator_unavailable", "No hourly timestamps available; wrote NaN.")
        return "", "", float("nan"), float("nan"), float("nan")
    ts = pd.to_datetime(hourly_out["timestamp_utc"], utc=True, errors="coerce").dropna().sort_values()
    if ts.empty:
        _warn(warnings, info.scenario, "coverage_share", "denominator_unavailable", "Hourly timestamps could not be parsed; wrote NaN.")
        return "", "", float("nan"), float("nan"), float("nan")
    start = ts.min()
    end = ts.max()
    total_hours = float(ts.nunique())
    expected = float(len(pd.date_range(start, end, freq="h", tz="UTC")))
    missing = expected - total_hours if np.isfinite(expected) else float("nan")
    coverage = _safe_ratio(total_hours, expected, scenario=info.scenario, metric="coverage_share", warnings=warnings)
    return start.isoformat(), end.isoformat(), expected, coverage, missing


def _severity_class(row: dict[str, Any]) -> str:
    def val(name: str) -> float:
        return pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]

    valid = parse_bool(row.get("simulation_valid"))
    reportable = parse_bool(row.get("thesis_reportable"))
    invalid_share = val("combined_infeasibility_hours_share")
    da_share = val("da_lockbook_infeasible_mwh_share_of_total_planned_trade")
    soc_max = val("max_soc_violation_mwh")
    reserve_max = val("max_reserve_headroom_shortfall_mw")
    known_material = (
        (np.isfinite(invalid_share) and invalid_share > 0.05)
        or (np.isfinite(da_share) and da_share > 0.01)
        or (np.isfinite(soc_max) and soc_max > 0.1)
        or (np.isfinite(reserve_max) and reserve_max > 0.1)
    )
    if known_material:
        return "material_invalidity"
    key_values = [invalid_share, da_share, soc_max, reserve_max]
    if any(not np.isfinite(v) for v in key_values):
        return "unknown_due_to_missing_diagnostics"
    if valid is True and reportable is True and invalid_share == 0.0:
        return "strict_valid"
    if invalid_share <= 0.01:
        return "minor_diagnostic_issue"
    if invalid_share <= 0.05:
        return "moderate_invalidity"
    return "unknown_due_to_missing_diagnostics"


def build_scenario(info: ScenarioInfo) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    metric_sources: list[dict[str, Any]] = []
    inventory, used_paths = _inventory(info)
    source_files_used: set[str] = set()

    metrics_df = _read_table(used_paths["metrics"]) if "metrics" in used_paths else pd.DataFrame()
    metrics = metrics_df.iloc[0].to_dict() if not metrics_df.empty else {}
    summary_path = used_paths.get("summary")
    simulation_valid, thesis_reportable, invalid_reason = _extract_validity(summary_path, metrics)

    hourly = _normalize_timestamps(_read_table(used_paths["hourly"])) if "hourly" in used_paths else pd.DataFrame()
    if hourly.empty:
        _warn(warnings, info.scenario, "total_hours", "missing_source_columns", "No backtest_hourly parquet/csv source found.")
        _metric_source(metric_sources, scenario=info.scenario, metric_name="total_hours", metric_status="unavailable", warning="missing hourly file")
        hourly_out = _empty_hourly()
    else:
        source_files_used.add(str(used_paths["hourly"]))
        _metric_source(metric_sources, scenario=info.scenario, metric_name="total_hours", metric_status="computed", source_file=used_paths["hourly"], source_column="timestamp_utc", source_semantics="unique simulated delivery hours")
        hourly_out = pd.DataFrame({"timestamp_utc": sorted(hourly["timestamp_utc"].dropna().unique())})
        hourly_out.insert(0, "quantile", info.quantile)
        hourly_out.insert(0, "model_label", info.model_label)
        hourly_out.insert(0, "model", info.model)
        hourly_out.insert(0, "scenario", info.scenario)

    # DA lockbook infeasibility.
    da_debug = _normalize_timestamps(_read_table(used_paths["da_precommit"])) if "da_precommit" in used_paths else pd.DataFrame()
    da_mwh_series = pd.Series(dtype=float)
    da_flag_series = pd.Series(dtype=float)
    da_cols: list[str] = []
    if not da_debug.empty:
        source_files_used.add(str(used_paths["da_precommit"]))
        da_mwh, da_cols = _abs_sum(da_debug, ["da_hourly_lock_infeasible_buy_mwh", "da_hourly_lock_infeasible_sell_mwh", "da_lockbook_infeasible_mwh"])
        if da_mwh is None:
            _warn(warnings, info.scenario, "da_lockbook_infeasible_mwh", "missing_source_columns", "Missing DA lockbook infeasibility magnitude columns.")
            _metric_source(metric_sources, scenario=info.scenario, metric_name="da_lockbook_infeasible_mwh", metric_status="unavailable", source_file=used_paths["da_precommit"], warning="missing DA lockbook infeasibility magnitude columns")
        else:
            _metric_source(metric_sources, scenario=info.scenario, metric_name="da_lockbook_infeasible_mwh", metric_status="computed", source_file=used_paths["da_precommit"], source_column=da_cols, source_semantics="absolute infeasible DA lockbook MWh")
            da_flag = da_mwh > 1e-9
            pass_flag, _ = _bool_flag(da_debug, ["da_existing_lockbook_physical_feasibility_passed", "da_combined_lockbook_physical_feasibility_passed"])
            if pass_flag is not None:
                da_flag = da_flag | (pass_flag < 0.5)
            da_mwh_series = _by_timestamp(da_debug, da_mwh, op="sum")
            da_flag_series = _by_timestamp(da_debug, da_flag.astype(float), op="max")
    else:
        _warn(warnings, info.scenario, "da_lockbook_infeasible", "missing_source_columns", "No da_precommit_debug.csv source found.")
        _metric_source(metric_sources, scenario=info.scenario, metric_name="da_lockbook_infeasible_mwh", metric_status="unavailable", warning="missing da_precommit_debug.csv")

    # Fallback and optimization infeasibility.
    opt_debug = _normalize_timestamps(_read_table(used_paths["optimization"])) if "optimization" in used_paths else pd.DataFrame()
    if not opt_debug.empty:
        source_files_used.add(str(used_paths["optimization"]))
    fallback_present_scenario_flag = "fallback_used" in str(invalid_reason).lower()
    fallback_count_source = "unavailable"
    total_opt_count_source = "unavailable"
    fallback_series = pd.Series(dtype=float)
    total_opt_count = float("nan")
    fallback_flag = None
    fallback_cols: list[str] = []
    if not opt_debug.empty:
        fallback_flag, fallback_cols = _bool_flag(opt_debug, ["optimizer_fallback_used", "fallback_used", "is_fallback_hour"])
        if fallback_flag is not None:
            fallback_series = _by_timestamp(opt_debug, fallback_flag, op="max")
            fallback_count_source = "explicit_optimization_log"
            total_opt_count = float(fallback_flag.notna().sum())
            total_opt_count_source = "explicit_optimization_log"
            _metric_source(metric_sources, scenario=info.scenario, metric_name="fallback_optimization_count", metric_status="computed", source_file=used_paths["optimization"], source_column=fallback_cols, source_semantics="explicit optimization log fallback flags", source_priority=1)
    if fallback_flag is None and not hourly.empty:
        fallback_flag, fallback_cols = _bool_flag(hourly, ["optimizer_fallback_used", "fallback_used", "is_fallback_hour"])
        if fallback_flag is not None:
            fallback_series = _by_timestamp(hourly, fallback_flag, op="max")
            fallback_count_source = "hourly_fallback_flags"
            total_opt_count = float(fallback_flag.notna().sum())
            total_opt_count_source = "hourly_fallback_flags"
            _metric_source(metric_sources, scenario=info.scenario, metric_name="fallback_optimization_count", metric_status="computed_from_fallback", source_file=used_paths["hourly"], source_column=fallback_cols, source_semantics="hourly fallback flags", source_priority=2, fallback_used=True)
    if fallback_flag is None:
        fallback_series = pd.Series(dtype=float)
        if fallback_present_scenario_flag:
            _warn(warnings, info.scenario, "fallback_optimization_count", "count_unavailable_from_scenario_flag_only", "Scenario invalid_reason contains fallback_used, but no hourly/log source exists for a count.")
            _metric_source(metric_sources, scenario=info.scenario, metric_name="fallback_optimization_count", metric_status="unavailable", source_semantics="scenario-level invalid reason only", source_priority=3, warning="presence flag only; count unavailable")
        else:
            _warn(warnings, info.scenario, "fallback_optimization", "missing_source_columns", "Missing fallback optimization flag columns.")
            _metric_source(metric_sources, scenario=info.scenario, metric_name="fallback_optimization_count", metric_status="unavailable", warning="missing fallback optimization flag columns")

    opt_flag, opt_cols = _bool_flag(hourly, ["optimization_error_code", "real_optimization_error_code", "solver_status"], contains="infeasible|reserve_infeasible") if not hourly.empty else (None, [])
    if opt_flag is None:
        opt_series = pd.Series(dtype=float)
        _warn(warnings, info.scenario, "optimization_infeasible", "missing_source_columns", "Missing optimization status/error columns.")
    else:
        opt_series = _by_timestamp(hourly, opt_flag, op="max")
        if not np.isfinite(total_opt_count):
            total_opt_count = float(opt_flag.notna().sum())
    if not opt_debug.empty:
        opt_dbg_flag, _ = _bool_flag(opt_debug, ["optimization_error_code", "solver_status", "status"], contains="infeasible|reserve_infeasible")
        if opt_dbg_flag is not None:
            opt_series = pd.concat([opt_series, _by_timestamp(opt_debug, opt_dbg_flag, op="max")]).groupby(level=0).max()
    _metric_source(metric_sources, scenario=info.scenario, metric_name="total_optimization_count", metric_status="computed" if np.isfinite(total_opt_count) else "unavailable", source_file=used_paths.get("optimization") if total_opt_count_source == "explicit_optimization_log" else used_paths.get("hourly"), source_column=fallback_cols, source_semantics=total_opt_count_source, warning="" if np.isfinite(total_opt_count) else "missing optimization denominator")

    # Activation and missed activation.
    realized = _normalize_timestamps(_read_table(used_paths["realized_ledger"])) if "realized_ledger" in used_paths else pd.DataFrame()
    if not realized.empty:
        source_files_used.add(str(used_paths["realized_ledger"]))
    act_source = realized if not realized.empty else hourly
    activation_source_file = used_paths.get("realized_ledger") if not realized.empty else used_paths.get("hourly")
    activation_source_name = "realized_ledger" if not realized.empty else ("hourly" if not hourly.empty else "unavailable")
    act_mwh, act_cols = _abs_sum(act_source, ["real_act_pos_mwh", "real_act_neg_mwh", "act_pos_mwh", "act_neg_mwh", "real_bem_only_executed_pos_mwh", "real_bem_only_executed_neg_mwh", "real_bcm_linked_pos_activation_mwh", "real_bcm_linked_neg_activation_mwh"]) if not act_source.empty else (None, [])
    if act_mwh is None:
        activation_series = pd.Series(dtype=float)
        _warn(warnings, info.scenario, "activation_mwh", "missing_source_columns", "Missing realized activation volume columns.")
        _metric_source(metric_sources, scenario=info.scenario, metric_name="total_activation_mwh", metric_status="unavailable", source_file=activation_source_file, warning="missing realized activation volume columns")
    else:
        activation_series = _by_timestamp(act_source, act_mwh, op="sum")
        _metric_source(metric_sources, scenario=info.scenario, metric_name="total_activation_mwh", metric_status="computed", source_file=activation_source_file, source_column=act_cols, source_semantics=activation_source_name)
    missed_mwh, missed_cols = _abs_sum(act_source, ["real_missed_activation_mwh", "missed_activation_mwh", "real_missed_activation_pos_mwh", "real_missed_activation_neg_mwh"]) if not act_source.empty else (None, [])
    if missed_mwh is None:
        missed_series = pd.Series(dtype=float)
        _warn(warnings, info.scenario, "missed_activation_mwh", "missing_source_columns", "Missing missed activation volume columns.")
        _metric_source(metric_sources, scenario=info.scenario, metric_name="missed_activation_mwh", metric_status="unavailable", source_file=activation_source_file, warning="missing missed activation volume columns")
    else:
        missed_series = _by_timestamp(act_source, missed_mwh, op="sum")
        _metric_source(metric_sources, scenario=info.scenario, metric_name="missed_activation_mwh", metric_status="computed", source_file=activation_source_file, source_column=missed_cols, source_semantics=activation_source_name)

    # Realized/final SoC violations only. Do not use planned/projected SoC.
    soc_violation, soc_cols = _abs_sum(hourly, ["real_protected_soc_violation_pos_mwh", "real_protected_soc_violation_neg_mwh", "final_soc_shortfall_mwh", "real_soc_violation_mwh"]) if not hourly.empty else (None, [])
    if soc_violation is None and not hourly.empty:
        soc_col = _find_col(hourly, ["real_soc_mwh", "final_soc_mwh", "final_soc_after_terminal_closure_mwh"])
        min_col = _find_col(hourly, ["physical_soc_min_mwh", "soc_min_mwh"])
        max_col = _find_col(hourly, ["physical_soc_max_mwh", "soc_max_mwh"])
        if soc_col and min_col and max_col:
            if any(token in soc_col.lower() for token in ["planned", "projected", "optimizer", "forecast"]):
                _warn(warnings, info.scenario, "soc_violation_mwh", "rejected_planned_soc_column", f"Rejected non-realized SoC candidate: {soc_col}")
            else:
                soc = _num(hourly[soc_col])
                soc_violation = (_num(hourly[min_col]) - soc).clip(lower=0.0) + (soc - _num(hourly[max_col])).clip(lower=0.0)
                soc_cols = [soc_col, min_col, max_col]
    if soc_violation is None:
        soc_series = pd.Series(dtype=float)
        _warn(warnings, info.scenario, "soc_violation_mwh", "missing_source_columns", "Missing realized/final SoC violation columns and realized SoC bounds.")
        _metric_source(metric_sources, scenario=info.scenario, metric_name="soc_violation_mwh", metric_status="unavailable", source_file=used_paths.get("hourly"), warning="missing realized/final SoC violation columns and bounds")
    else:
        soc_series = _by_timestamp(hourly, soc_violation, op="sum")
        _metric_source(metric_sources, scenario=info.scenario, metric_name="soc_violation_mwh", metric_status="computed", source_file=used_paths.get("hourly"), source_column=soc_cols, source_semantics="realized/final/post-dispatch SoC only")

    # Reserve headroom shortfall from reserve debug or hourly explicit columns.
    reserve = _normalize_timestamps(_read_table(used_paths["reserve"])) if "reserve" in used_paths else pd.DataFrame()
    if not reserve.empty:
        source_files_used.add(str(used_paths["reserve"]))
        reserve_source = reserve
    else:
        reserve_source = hourly
    reserve_short, reserve_cols = _abs_sum(reserve_source, ["headroom_violation_mwh", "headroom_violation_mw", "real_headroom_violation_pos_mwh", "real_headroom_violation_neg_mwh", "reserve_headroom_shortfall_mw", "real_reserve_headroom_shortfall_mw"]) if not reserve_source.empty else (None, [])
    if reserve_short is None:
        reserve_series = pd.Series(dtype=float)
        _warn(warnings, info.scenario, "reserve_headroom_shortfall", "missing_source_columns", "Missing reserve headroom shortfall columns.")
        _metric_source(metric_sources, scenario=info.scenario, metric_name="reserve_headroom_shortfall_mw", metric_status="unavailable", source_file=used_paths.get("reserve") or used_paths.get("hourly"), warning="missing reserve headroom shortfall columns")
    else:
        reserve_series = _by_timestamp(reserve_source, reserve_short, op="sum")
        _metric_source(metric_sources, scenario=info.scenario, metric_name="reserve_headroom_shortfall_mw", metric_status="computed", source_file=used_paths.get("reserve") or used_paths.get("hourly"), source_column=reserve_cols, source_semantics="MW or MW-hour equivalent from available source")

    if not hourly_out.empty:
        _series_to_hourly(hourly_out, da_flag_series, "da_lockbook_infeasible_flag")
        _series_to_hourly(hourly_out, da_mwh_series, "da_lockbook_infeasible_mwh")
        _series_to_hourly(hourly_out, fallback_series, "fallback_optimization_flag")
        _series_to_hourly(hourly_out, opt_series, "optimization_infeasible_flag")
        _series_to_hourly(hourly_out, missed_series.gt(1e-9).astype(float), "missed_activation_flag")
        _series_to_hourly(hourly_out, missed_series, "missed_activation_mwh")
        _series_to_hourly(hourly_out, activation_series.gt(1e-9).astype(float), "activation_event_flag")
        _series_to_hourly(hourly_out, activation_series, "activation_mwh")
        _series_to_hourly(hourly_out, soc_series.gt(1e-9).astype(float), "soc_violation_flag")
        _series_to_hourly(hourly_out, soc_series, "soc_violation_mwh")
        _series_to_hourly(hourly_out, reserve_series.gt(1e-9).astype(float), "reserve_headroom_shortfall_flag")
        _series_to_hourly(hourly_out, reserve_series, "reserve_headroom_shortfall_mw")
        flag_cols = [
            "da_lockbook_infeasible_flag",
            "fallback_optimization_flag",
            "optimization_infeasible_flag",
            "missed_activation_flag",
            "soc_violation_flag",
            "reserve_headroom_shortfall_flag",
        ]
        available = [c for c in flag_cols if c in hourly_out.columns and not hourly_out[c].isna().all()]
        if available:
            hourly_out["combined_infeasibility_flag"] = hourly_out[available].fillna(0.0).max(axis=1)
        else:
            hourly_out["combined_infeasibility_flag"] = np.nan
            _warn(warnings, info.scenario, "combined_infeasibility_flag", "missing_source_columns", "No hourly invalidity flags available.")
        hourly_out["source_files_used"] = ";".join(sorted(source_files_used))

    total_hours = float(hourly_out["timestamp_utc"].nunique()) if not hourly_out.empty else float("nan")
    combined_hours = _count_flag(hourly_out, "combined_infeasibility_flag")
    da_hours = _count_flag(hourly_out, "da_lockbook_infeasible_flag")
    da_mwh_total = _sum_if_available(hourly_out, "da_lockbook_infeasible_mwh")
    fallback_count = _count_flag(hourly_out, "fallback_optimization_flag")
    missed_count = _count_flag(hourly_out, "missed_activation_flag")
    activation_count = _count_flag(hourly_out, "activation_event_flag")
    missed_total = _sum_if_available(hourly_out, "missed_activation_mwh")
    activation_total = _sum_if_available(hourly_out, "activation_mwh")
    soc_hours = _count_flag(hourly_out, "soc_violation_flag")
    soc_max = _max_if_available(hourly_out, "soc_violation_mwh")
    soc_sum = _sum_if_available(hourly_out, "soc_violation_mwh")
    reserve_hours = _count_flag(hourly_out, "reserve_headroom_shortfall_flag")
    reserve_max = _max_if_available(hourly_out, "reserve_headroom_shortfall_mw")
    reserve_sum = _sum_if_available(hourly_out, "reserve_headroom_shortfall_mw")
    total_planned_trade, total_planned_trade_source, planned_trade_cols = _planned_trade_total(metrics, hourly, info, warnings)
    _metric_source(metric_sources, scenario=info.scenario, metric_name="total_planned_trade_mwh", metric_status="computed_from_fallback" if total_planned_trade_source == "realized_fallback" else ("computed" if np.isfinite(total_planned_trade) else "unavailable"), source_file=used_paths.get("metrics") if total_planned_trade_source == "planned" and planned_trade_cols and planned_trade_cols[0].endswith("_total") else used_paths.get("hourly"), source_column=planned_trade_cols, source_semantics=total_planned_trade_source, fallback_used=total_planned_trade_source == "realized_fallback", warning="" if np.isfinite(total_planned_trade) else "denominator unavailable")
    start_time, end_time, expected_hours, coverage_share, missing_hours = _coverage_stats(hourly_out, info, warnings)
    activation_source = activation_source_name if act_mwh is not None else "unavailable"
    denominator_status_values = [total_planned_trade_source, activation_source, total_opt_count_source]
    denominator_completeness_status = "complete" if all(v not in {"unavailable", ""} for v in denominator_status_values) else "partial"

    row = {
        "scenario": info.scenario,
        "model": info.model,
        "model_label": info.model_label,
        "quantile": info.quantile,
        "is_benchmark": info.is_benchmark,
        "simulation_valid": simulation_valid,
        "thesis_reportable": thesis_reportable,
        "invalid_reason": invalid_reason,
        "start_time_utc": start_time,
        "end_time_utc": end_time,
        "total_hours": total_hours,
        "expected_hours": expected_hours,
        "coverage_share": coverage_share,
        "missing_hours_count": missing_hours,
        "combined_infeasibility_hours": combined_hours,
        "combined_infeasibility_hours_share": _safe_ratio(combined_hours, total_hours, scenario=info.scenario, metric="combined_infeasibility_hours_share", warnings=warnings),
        "da_lockbook_infeasible_hours": da_hours,
        "da_lockbook_infeasible_mwh": da_mwh_total,
        "total_planned_trade_mwh": total_planned_trade,
        "total_planned_trade_mwh_source": total_planned_trade_source,
        "da_lockbook_infeasible_mwh_share_of_total_planned_trade": _safe_ratio(da_mwh_total, total_planned_trade, scenario=info.scenario, metric="da_lockbook_infeasible_mwh_share_of_total_planned_trade", warnings=warnings),
        "fallback_optimization_count": fallback_count,
        "total_optimization_count": total_opt_count,
        "total_optimization_count_source": total_opt_count_source,
        "fallback_optimization_count_source": fallback_count_source,
        "fallback_present_scenario_flag": fallback_present_scenario_flag,
        "fallback_optimization_share": _safe_ratio(fallback_count, total_opt_count, scenario=info.scenario, metric="fallback_optimization_share", warnings=warnings),
        "missed_activation_count": missed_count,
        "total_activation_count": activation_count,
        "activation_count_semantics": "hourly_event_count",
        "missed_activation_count_semantics": "hourly_event_count",
        "missed_activation_count_share": _safe_ratio(missed_count, activation_count, scenario=info.scenario, metric="missed_activation_count_share", warnings=warnings),
        "missed_activation_mwh": missed_total,
        "total_activation_mwh": activation_total,
        "total_activation_mwh_source": activation_source,
        "missed_activation_mwh_share": _safe_ratio(missed_total, activation_total, scenario=info.scenario, metric="missed_activation_mwh_share", warnings=warnings),
        "soc_violation_hours": soc_hours,
        "max_soc_violation_mwh": soc_max,
        "sum_soc_violation_mwh": soc_sum,
        "reserve_headroom_shortfall_hours": reserve_hours,
        "max_reserve_headroom_shortfall_mw": reserve_max,
        "sum_reserve_headroom_shortfall_mw_hours": reserve_sum,
        "denominator_completeness_status": denominator_completeness_status,
        "invalidity_severity_class": "",
        "diagnostic_completeness_status": "complete" if not warnings else "partial",
        "missing_source_columns": ";".join(sorted({w["metric"] for w in warnings if w.get("warning") == "missing_source_columns"})),
        "source_files_used": ";".join(sorted(source_files_used)),
    }
    _metric_source(metric_sources, scenario=info.scenario, metric_name="combined_infeasibility_hours", metric_status="computed" if np.isfinite(combined_hours) else "unavailable", source_file=used_paths.get("hourly"), source_column="combined_infeasibility_flag", source_semantics="unique hours with at least one invalidity flag")
    _warn_nan_metrics(row, warnings)
    row["invalidity_severity_class"] = _severity_class(row)
    row["diagnostic_completeness_status"] = "complete" if not warnings else "partial"
    row["missing_source_columns"] = ";".join(sorted({w["metric"] for w in warnings if w.get("warning") == "missing_source_columns"}))
    for metric in [k for k in row if k.endswith("_share")]:
        value = row[metric]
        if np.isfinite(value) and not (0.0 <= float(value) <= 1.0):
            _warn(warnings, info.scenario, metric, "share_out_of_bounds", str(value))
    if np.isfinite(combined_hours) and np.isfinite(total_hours) and combined_hours > total_hours:
        _warn(warnings, info.scenario, "combined_infeasibility_hours", "validation_error", "combined_infeasibility_hours > total_hours")
    return hourly_out.reindex(columns=HOURLY_COLUMNS), row, inventory, pd.DataFrame(metric_sources), warnings


def _fmt_pct(value: Any) -> str:
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "--" if not np.isfinite(v) else f"{100.0 * float(v):.1f}\\%"


def _fmt_num(value: Any, digits: int = 2) -> str:
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "--" if not np.isfinite(v) else f"{float(v):.{digits}f}"


def _stat(summary: pd.DataFrame, column: str, op: str) -> float:
    if column not in summary:
        return float("nan")
    values = pd.to_numeric(summary[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    if op == "median":
        return float(values.median())
    if op == "max":
        return float(values.max())
    raise ValueError(f"Unsupported stat op: {op}")


def _latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    return "".join(repl.get(ch, ch) for ch in text)


def write_latex_table(summary: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Scenario & Severity class & Invalid hours (\%) & DA infeas. / trade (\%) & Fallback opt. (\%) & Missed act. volume (\%) & Max SoC viol. & Max reserve short. \\",
        r"\midrule",
    ]
    if summary.empty:
        rows.append(r"-- & -- & -- & -- & -- & -- & -- & -- \\")
    else:
        d = summary.copy().sort_values(["is_benchmark", "model_label", "quantile", "scenario"])
        for _, row in d.iterrows():
            rows.append(
                " & ".join(
                    [
                        _latex_escape(row.get("scenario", "")),
                        _latex_escape(row.get("invalidity_severity_class", "")),
                        _fmt_pct(row.get("combined_infeasibility_hours_share")),
                        _fmt_pct(row.get("da_lockbook_infeasible_mwh_share_of_total_planned_trade")),
                        _fmt_pct(row.get("fallback_optimization_share")),
                        _fmt_pct(row.get("missed_activation_mwh_share")),
                        _fmt_num(row.get("max_soc_violation_mwh")),
                        _fmt_num(row.get("max_reserve_headroom_shortfall_mw")),
                    ]
                )
                + r" \\"
            )
    rows += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Simulation invalidity severity diagnostics for {label.upper()}. Unavailable metrics are shown as --.}}",
        rf"\label{{tab:{label}_simulation_invalidity_severity}}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")


def write_limitation_summary(summary: pd.DataFrame, warnings: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(summary)
    valid = summary.get("simulation_valid", pd.Series(dtype=object)).map(parse_bool) if n else pd.Series(dtype=object)
    invalid = int(valid.eq(False).sum()) if n else 0
    model_df = summary.loc[~summary.get("is_benchmark", pd.Series(dtype=bool)).astype(bool)].copy() if n else pd.DataFrame()
    model_valid = model_df.get("simulation_valid", pd.Series(dtype=object)).map(parse_bool) if not model_df.empty else pd.Series(dtype=object)
    invalid_model = int(model_valid.eq(False).sum()) if not model_df.empty else 0
    reasons = []
    if n and "invalid_reason" in summary.columns:
        for value in summary["invalid_reason"].fillna("").astype(str):
            reasons.extend([x.strip() for x in re.split(r"[,;]", value) if x.strip()])
    reason_counts = pd.Series(reasons).value_counts().head(8) if reasons else pd.Series(dtype=int)
    invalid_share = invalid / n if n else float("nan")
    invalid_model_share = invalid_model / len(model_df) if len(model_df) else float("nan")
    lines = [
        f"Simulation invalidity limitation summary ({label})",
        "",
        f"- Total scenarios: {n}",
        f"- Invalid scenarios: {invalid} ({invalid_share:.1%})" if np.isfinite(invalid_share) else f"- Invalid scenarios: {invalid}",
        f"- Invalid model-based scenarios: {invalid_model} ({invalid_model_share:.1%})" if np.isfinite(invalid_model_share) else f"- Invalid model-based scenarios: {invalid_model}",
        "- Most frequent invalid reasons: " + (", ".join(f"{idx}={val}" for idx, val in reason_counts.items()) if not reason_counts.empty else "none recorded"),
        f"- Median invalid-hours share: {_fmt_num(_stat(summary, 'combined_infeasibility_hours_share', 'median'), 4)}",
        f"- Max invalid-hours share: {_fmt_num(_stat(summary, 'combined_infeasibility_hours_share', 'max'), 4)}",
        f"- Median fallback optimization share: {_fmt_num(_stat(summary, 'fallback_optimization_share', 'median'), 4)}",
        f"- Max fallback optimization share: {_fmt_num(_stat(summary, 'fallback_optimization_share', 'max'), 4)}",
        f"- Median missed activation MWh share: {_fmt_num(_stat(summary, 'missed_activation_mwh_share', 'median'), 4)}",
        f"- Max missed activation MWh share: {_fmt_num(_stat(summary, 'missed_activation_mwh_share', 'max'), 4)}",
        f"- Largest SoC violation (MWh): {_fmt_num(_stat(summary, 'max_soc_violation_mwh', 'max'))}",
        f"- Largest reserve headroom shortfall (MW): {_fmt_num(_stat(summary, 'max_reserve_headroom_shortfall_mw', 'max'))}",
        "- Diagnostic completeness caveat: metrics marked unavailable or computed from fallback sources should be treated as audit limitations, not as zero violations.",
        "- Severity class thresholds: strict_valid requires valid/reportable output and zero invalid hours; minor_diagnostic_issue uses invalid-hours share <= 0.01 without large magnitude violations; moderate_invalidity uses invalid-hours share <= 0.05; material_invalidity uses invalid-hours share > 0.05, DA infeasible/trade share > 0.01, max SoC violation > 0.1 MWh, or max reserve shortfall > 0.1 MW; unknown_due_to_missing_diagnostics is used when key diagnostics are unavailable.",
        f"- Warning rows: {len(warnings)}",
        "",
        "Limitation statement:",
        "The economic results should be interpreted as diagnostic backtest evidence rather than fully validated physically feasible trading-performance estimates if invalidities remain. "
        "The invalidity diagnostics quantify physical or optimization-related violations relative to simulated hours and available volume denominators. "
        "Invalid runs should not be described as fully valid trading-performance results, and physical feasibility should not be claimed where violations exist.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_outputs(run_root: Path, out_root: Path, *, label: str) -> dict[str, Path]:
    scenarios, discovery_warnings = discover_scenarios(run_root)
    if not scenarios:
        raise FileNotFoundError(f"No scenario folders with targeted simulation output files found under {run_root}")
    hourly_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    inventory_parts: list[pd.DataFrame] = []
    metric_source_parts: list[pd.DataFrame] = []
    warning_rows: list[dict[str, Any]] = list(discovery_warnings)
    for info in scenarios:
        hourly, summary, inventory, metric_sources, warnings = build_scenario(info)
        hourly_parts.append(hourly)
        summary_rows.append(summary)
        inventory_parts.append(inventory)
        metric_source_parts.append(metric_sources)
        warning_rows.extend(warnings)

    diag_dir = out_root / "backup" / "diagnostics"
    warn_dir = out_root / "backup" / "warnings"
    appendix_tables = out_root / "appendix" / "tables"
    diag_dir.mkdir(parents=True, exist_ok=True)
    warn_dir.mkdir(parents=True, exist_ok=True)
    appendix_tables.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows).reindex(columns=SUMMARY_COLUMNS).sort_values(["is_benchmark", "model_label", "quantile", "scenario"]).reset_index(drop=True)
    hourly_df = pd.concat(hourly_parts, ignore_index=True) if hourly_parts else _empty_hourly()
    if not hourly_df.empty:
        dup = hourly_df.duplicated(["scenario", "timestamp_utc"])
        if dup.any():
            warning_rows.append({"scenario": "", "metric": "hourly_uniqueness", "warning": "duplicate_hourly_rows", "details": f"Dropped {int(dup.sum())} duplicate scenario/timestamp rows."})
        hourly_df = hourly_df.drop_duplicates(["scenario", "timestamp_utc"], keep="last").sort_values(["scenario", "timestamp_utc"]).reset_index(drop=True)
    inventory_df = pd.concat(inventory_parts, ignore_index=True) if inventory_parts else pd.DataFrame()
    metric_sources_df = pd.concat(metric_source_parts, ignore_index=True) if metric_source_parts else pd.DataFrame()
    for col in [c for c in summary_df.columns if c.endswith("_share")]:
        bad = summary_df[col].replace([np.inf, -np.inf], np.nan)
        summary_df[col] = bad
    if np.isinf(summary_df.select_dtypes(include=[np.number]).to_numpy(dtype=float)).any():
        warning_rows.append({"scenario": "", "metric": "validation", "warning": "nonfinite_inf_replaced", "details": "Infinite numeric summary value replaced with NaN."})
        summary_df = summary_df.replace([np.inf, -np.inf], np.nan)
    if not hourly_df.empty and np.isinf(hourly_df.select_dtypes(include=[np.number]).to_numpy(dtype=float)).any():
        warning_rows.append({"scenario": "", "metric": "validation", "warning": "nonfinite_inf_replaced", "details": "Infinite numeric hourly value replaced with NaN."})
        hourly_df = hourly_df.replace([np.inf, -np.inf], np.nan)
    for share_col in [c for c in summary_df.columns if c.endswith("_share")]:
        values = pd.to_numeric(summary_df[share_col], errors="coerce")
        bad = values.notna() & ((values < 0.0) | (values > 1.0))
        if bad.any():
            warning_rows.append({"scenario": "", "metric": share_col, "warning": "share_out_of_bounds", "details": f"{int(bad.sum())} rows outside [0, 1]."})
    warnings_df = pd.DataFrame(warning_rows)

    paths = {
        "summary": diag_dir / "simulation_invalidity_severity_summary.csv",
        "hourly": diag_dir / "simulation_invalidity_severity_by_hour.csv",
        "inventory": diag_dir / "simulation_invalidity_source_inventory.csv",
        "metric_sources": diag_dir / "simulation_invalidity_metric_sources.csv",
        "warnings": warn_dir / "simulation_invalidity_severity_warnings.csv",
        "latex": appendix_tables / "simulation_invalidity_severity_summary.tex",
        "limitation_summary": diag_dir / "simulation_invalidity_limitation_summary.txt",
    }
    summary_df.to_csv(paths["summary"], index=False)
    hourly_df.to_csv(paths["hourly"], index=False)
    inventory_df.to_csv(paths["inventory"], index=False)
    metric_sources_df.to_csv(paths["metric_sources"], index=False)
    warnings_df.to_csv(paths["warnings"], index=False)
    write_latex_table(summary_df, paths["latex"], label)
    write_limitation_summary(summary_df, warnings_df, paths["limitation_summary"], label)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build simulation invalidity severity diagnostics from existing run folders.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--label", default="rq2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = build_outputs(Path(args.run_root), Path(args.out_root), label=str(args.label))
    for name, path in paths.items():
        print(f"[OK] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
