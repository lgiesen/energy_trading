#!/usr/bin/env python3
"""Build RQ2 invalidity severity diagnostics from existing simulation outputs."""

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


DEFAULT_RUN_DIR = Path("artifacts/simulation_runs/thesis_final_multi_2m_20260620T091938Z")
DEFAULT_OUT_DIR = Path("artifacts/benchmark/rq2_simulation_benchmark")

REQUESTED_CANDIDATE_FILES = [
    "backtest_hourly.parquet",
    "backtest_hourly.csv",
    "validation_events.csv",
    "strict_validity_events.csv",
    "simulation_summary.csv",
    "summary.json",
    "optimization_log.csv",
    "metrics.csv",
]

PROJECT_CANDIDATE_FILES = [
    "backtest_summary.json",
    "performance_metrics.csv",
    "performance_paths_long.parquet",
    "performance_paths_long.csv",
    "da_precommit_debug.csv",
    "optimization_failure_debug.csv",
    "optimization_infeasibility_attribution.csv",
    "realized_ledger.parquet",
    "realized_ledger.csv",
    "reserve_commitment_debug.csv",
]

CANDIDATE_FILES = REQUESTED_CANDIDATE_FILES + [p for p in PROJECT_CANDIDATE_FILES if p not in REQUESTED_CANDIDATE_FILES]


@dataclass(frozen=True)
class ScenarioInfo:
    scenario_dir: Path
    scenario: str
    model: str
    model_label: str
    quantile: str
    is_benchmark: bool


def _as_num(s: pd.Series | Any) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce")
    return pd.Series(s, dtype="float64")


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _numeric_sum(df: pd.DataFrame, columns: list[str]) -> pd.Series | None:
    existing = [c for c in columns if c in df.columns]
    if not existing:
        return None
    out = pd.Series(0.0, index=df.index, dtype="float64")
    for col in existing:
        out = out + _as_num(df[col]).fillna(0.0).abs()
    return out


def _safe_ratio(num: float, den: float, *, scenario: str, metric: str, warnings: list[dict[str, Any]]) -> float:
    if not np.isfinite(num) or not np.isfinite(den):
        return float("nan")
    if abs(float(den)) <= 1e-12:
        warnings.append(
            {
                "scenario": scenario,
                "metric": metric,
                "warning": "zero_denominator",
                "details": f"Cannot compute {metric}; denominator is zero.",
            }
        )
        return float("nan")
    value = float(num) / float(den)
    if not np.isfinite(value):
        warnings.append({"scenario": scenario, "metric": metric, "warning": "nonfinite_share", "details": ""})
        return float("nan")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_table(path: Path, *, columns: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path, columns=columns)
        except Exception:
            return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if path.suffix == ".json":
        payload = _read_json(path)
        if isinstance(payload, dict):
            return pd.DataFrame([payload])
    return pd.DataFrame()


def _timestamp_col(df: pd.DataFrame) -> str | None:
    return _first_existing_col(df, ["timestamp_utc", "target_time_utc", "delivery_time_utc", "date"])


def _normalize_timestamps(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if df.empty:
        return df
    col = _timestamp_col(df)
    if col is None:
        return df
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out[col], utc=True, errors="coerce")
    return out.dropna(subset=["timestamp_utc"]).copy()


def _parse_scenario_info(scenario_dir: Path) -> ScenarioInfo:
    run_name = scenario_dir.parents[2].name
    model_raw = scenario_dir.parents[1].name
    scenario = scenario_dir.name
    if model_raw == "benchmarks_naive":
        model, label, quantile, is_benchmark = "naive", "NAIVE", "p50", True
    elif model_raw == "benchmarks_rhpf":
        model, label, quantile, is_benchmark = "rhpf", "RHPF", "p50", True
    else:
        m = re.match(r"(?P<model>linear|xgb|tft)_+?(?P<q>p\d+)$", model_raw)
        model = m.group("model") if m else model_raw
        quantile = m.group("q") if m else scenario.replace("_", "-")
        label = {"linear": "RLQR", "xgb": "XGB", "tft": "TFT"}.get(model, model.upper())
        is_benchmark = False
    return ScenarioInfo(
        scenario=f"{run_name}/{model_raw}/{scenario_dir.parent.name}/{scenario}",
        scenario_dir=scenario_dir,
        model=model,
        model_label=label,
        quantile=quantile,
        is_benchmark=is_benchmark,
    )


def discover_scenarios(run_dir: Path) -> list[ScenarioInfo]:
    paths = sorted(run_dir.glob("*/*/*/backtest_summary.json"))
    return [_parse_scenario_info(p.parent) for p in paths]


def _inventory_for_scenario(info: ScenarioInfo) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    rows: list[dict[str, Any]] = []
    used: dict[str, Path] = {}
    for name in CANDIDATE_FILES:
        path = info.scenario_dir / name
        exists = path.exists()
        df = pd.DataFrame()
        row_count = np.nan
        columns: list[str] = []
        ts_col = ""
        metrics_supported: list[str] = []
        reason = ""
        if exists:
            try:
                df = _read_table(path, nrows=5 if path.suffix == ".csv" else None)
                row_count = int(len(df))
                columns = list(df.columns)
                ts_col = _timestamp_col(df) or ""
                lower_cols = {c.lower() for c in columns}
                if "backtest_hourly" in name:
                    metrics_supported.extend(["total_hours", "fallback_optimization", "missed_activation", "soc_violation"])
                    used["hourly"] = path
                if name in {"da_precommit_debug.csv"}:
                    metrics_supported.append("da_lockbook_infeasible")
                    used["da_precommit"] = path
                if name in {"optimization_failure_debug.csv", "optimization_log.csv"}:
                    metrics_supported.append("optimization_infeasible")
                    used["optimization"] = path
                if name == "optimization_infeasibility_attribution.csv":
                    metrics_supported.append("optimization_infeasibility_attribution")
                    used["optimization_attribution"] = path
                if name.startswith("realized_ledger"):
                    metrics_supported.append("activation_volume")
                    used["realized_ledger"] = path
                if name == "reserve_commitment_debug.csv":
                    metrics_supported.append("reserve_headroom_shortfall")
                    used["reserve"] = path
                if name in {"performance_metrics.csv", "metrics.csv"}:
                    metrics_supported.append("scenario_summary_metrics")
                    used["metrics"] = path
                if name in {"backtest_summary.json", "summary.json", "simulation_summary.csv"}:
                    metrics_supported.append("validity_summary")
                    used["summary"] = path
                if not metrics_supported and lower_cols:
                    reason = "file_exists_but_no_supported_invalidity_metrics_detected"
            except Exception as exc:
                reason = f"read_error:{type(exc).__name__}:{exc}"
        else:
            reason = "missing"
        rows.append(
            {
                "scenario": info.scenario,
                "candidate_file": str(path),
                "exists": bool(exists),
                "used": bool(metrics_supported),
                "row_count": row_count,
                "timestamp_column": ts_col,
                "available_columns": ";".join(columns),
                "metrics_supported": ";".join(metrics_supported),
                "reason_if_not_used": reason,
            }
        )
    return rows, used


def _empty_hourly(info: ScenarioInfo) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
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
    )


def _aggregate_by_timestamp(df: pd.DataFrame, value: pd.Series, *, op: str = "sum") -> pd.Series:
    d = pd.DataFrame({"timestamp_utc": df["timestamp_utc"], "value": value})
    if op == "max":
        return d.groupby("timestamp_utc")["value"].max()
    return d.groupby("timestamp_utc")["value"].sum()


def _add_series(out: pd.DataFrame, series: pd.Series, col: str) -> None:
    if series.empty:
        out[col] = np.nan
        return
    out[col] = out["timestamp_utc"].map(series)


def _metric_warning(warnings: list[dict[str, Any]], info: ScenarioInfo, metric: str, details: str) -> None:
    warnings.append({"scenario": info.scenario, "metric": metric, "warning": "missing_source_columns", "details": details})


def build_scenario(info: ScenarioInfo) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    inventory_rows, used_paths = _inventory_for_scenario(info)
    warnings: list[dict[str, Any]] = []
    source_files_used: set[str] = set()

    summary = _read_json(used_paths.get("summary", info.scenario_dir / "backtest_summary.json"))
    metrics_df = _read_table(used_paths["metrics"]) if "metrics" in used_paths else pd.DataFrame()
    metrics = metrics_df.iloc[0].to_dict() if not metrics_df.empty else {}

    hourly = _normalize_timestamps(_read_table(used_paths["hourly"]), source="hourly") if "hourly" in used_paths else pd.DataFrame()
    if not hourly.empty:
        source_files_used.add(str(used_paths["hourly"]))
        base = pd.DataFrame({"timestamp_utc": sorted(hourly["timestamp_utc"].dropna().unique())})
    else:
        _metric_warning(warnings, info, "total_hours", "No backtest_hourly parquet/csv source found.")
        base = pd.DataFrame({"timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]")})

    if base.empty:
        hourly_out = _empty_hourly(info)
    else:
        hourly_out = base.copy()
        hourly_out.insert(0, "quantile", info.quantile)
        hourly_out.insert(0, "model_label", info.model_label)
        hourly_out.insert(0, "model", info.model)
        hourly_out.insert(0, "scenario", info.scenario)

    # DA lockbook physical infeasibility.
    da_debug = _normalize_timestamps(_read_table(used_paths["da_precommit"]), source="da_precommit") if "da_precommit" in used_paths else pd.DataFrame()
    if not da_debug.empty:
        source_files_used.add(str(used_paths["da_precommit"]))
        da_mwh = _numeric_sum(da_debug, ["da_hourly_lock_infeasible_buy_mwh", "da_hourly_lock_infeasible_sell_mwh"])
        if da_mwh is None:
            _metric_warning(warnings, info, "da_lockbook_infeasible_mwh", "Missing da_hourly_lock_infeasible_buy_mwh/sell_mwh.")
            da_mwh_series = pd.Series(dtype=float)
            da_flag_series = pd.Series(dtype=float)
        else:
            da_mwh_series = _aggregate_by_timestamp(da_debug, da_mwh, op="sum")
            pass_cols = [c for c in ["da_existing_lockbook_physical_feasibility_passed", "da_combined_lockbook_physical_feasibility_passed"] if c in da_debug.columns]
            explicit_flag = da_mwh > 1e-9
            for col in pass_cols:
                explicit_flag = explicit_flag | (_as_num(da_debug[col]).fillna(1.0) < 0.5)
            da_flag_series = _aggregate_by_timestamp(da_debug, explicit_flag.astype(float), op="max")
    else:
        _metric_warning(warnings, info, "da_lockbook_infeasible", "No da_precommit_debug.csv source found.")
        da_mwh_series = pd.Series(dtype=float)
        da_flag_series = pd.Series(dtype=float)

    # Optimizer fallback/infeasibility.
    if not hourly.empty and "optimizer_fallback_used" in hourly.columns:
        fallback_series = _aggregate_by_timestamp(hourly, (_as_num(hourly["optimizer_fallback_used"]).fillna(0.0) > 0.5).astype(float), op="max")
        total_opt_count = int(_as_num(hourly["optimizer_fallback_used"]).notna().sum())
    else:
        fallback_series = pd.Series(dtype=float)
        total_opt_count = np.nan
        _metric_warning(warnings, info, "fallback_optimization", "Missing optimizer_fallback_used in hourly source.")
    opt_code_col = _first_existing_col(hourly, ["optimization_error_code", "real_optimization_error_code"]) if not hourly.empty else None
    if opt_code_col:
        opt_infeasible_series = _aggregate_by_timestamp(hourly, hourly[opt_code_col].fillna("").astype(str).str.contains("infeasible|reserve_infeasible", case=False, regex=True).astype(float), op="max")
        if np.isnan(total_opt_count):
            total_opt_count = int(hourly[opt_code_col].notna().sum())
    else:
        opt_infeasible_series = pd.Series(dtype=float)
        _metric_warning(warnings, info, "optimization_infeasible", "Missing optimization_error_code in hourly source.")
    opt_debug = _normalize_timestamps(_read_table(used_paths["optimization"]), source="optimization") if "optimization" in used_paths else pd.DataFrame()
    if not opt_debug.empty:
        source_files_used.add(str(used_paths["optimization"]))
        if "is_fallback_hour" in opt_debug.columns:
            fallback_series = pd.concat([fallback_series, _aggregate_by_timestamp(opt_debug, (_as_num(opt_debug["is_fallback_hour"]).fillna(0.0) > 0.5).astype(float), op="max")], axis=0).groupby(level=0).max()
        code_col = _first_existing_col(opt_debug, ["optimization_error_code", "solver_status"])
        if code_col:
            opt_infeasible_series = pd.concat([opt_infeasible_series, _aggregate_by_timestamp(opt_debug, opt_debug[code_col].fillna("").astype(str).str.contains("infeasible|reserve_infeasible", case=False, regex=True).astype(float), op="max")], axis=0).groupby(level=0).max()

    # Activation and missed activation.
    realized = _normalize_timestamps(_read_table(used_paths["realized_ledger"]), source="realized_ledger") if "realized_ledger" in used_paths else pd.DataFrame()
    act_source = realized if not realized.empty else hourly
    if not realized.empty:
        source_files_used.add(str(used_paths["realized_ledger"]))
    act_mwh = _numeric_sum(act_source, ["real_act_pos_mwh", "real_act_neg_mwh", "act_pos_mwh", "act_neg_mwh"]) if not act_source.empty else None
    if act_mwh is None:
        activation_series = pd.Series(dtype=float)
        _metric_warning(warnings, info, "activation_mwh", "Missing realized activation volume columns.")
    else:
        activation_series = _aggregate_by_timestamp(act_source, act_mwh, op="sum")
    missed_mwh = _numeric_sum(act_source, ["real_missed_activation_mwh", "missed_activation_mwh", "real_missed_activation_pos_mwh", "real_missed_activation_neg_mwh"]) if not act_source.empty else None
    if missed_mwh is None:
        missed_series = pd.Series(dtype=float)
        _metric_warning(warnings, info, "missed_activation_mwh", "Missing missed activation volume columns.")
    else:
        missed_series = _aggregate_by_timestamp(act_source, missed_mwh, op="sum")

    # Realized protected SoC violations.
    soc_violation = _numeric_sum(hourly, ["real_protected_soc_violation_pos_mwh", "real_protected_soc_violation_neg_mwh"]) if not hourly.empty else None
    if soc_violation is None:
        soc_series = pd.Series(dtype=float)
        _metric_warning(warnings, info, "soc_violation_mwh", "Missing realized protected SoC violation columns.")
    else:
        soc_series = _aggregate_by_timestamp(hourly, soc_violation, op="sum")

    # Reserve headroom shortfall.
    reserve = _normalize_timestamps(_read_table(used_paths["reserve"]), source="reserve") if "reserve" in used_paths else pd.DataFrame()
    if not reserve.empty:
        source_files_used.add(str(used_paths["reserve"]))
        reserve_short = _numeric_sum(reserve, ["headroom_violation_mwh", "real_headroom_violation_pos_mwh", "real_headroom_violation_neg_mwh"])
        if reserve_short is None:
            reserve_series = pd.Series(dtype=float)
            _metric_warning(warnings, info, "reserve_headroom_shortfall", "Missing reserve headroom violation columns.")
        else:
            reserve_series = _aggregate_by_timestamp(reserve, reserve_short, op="sum")
    else:
        reserve_series = pd.Series(dtype=float)
        _metric_warning(warnings, info, "reserve_headroom_shortfall", "No reserve_commitment_debug.csv source found.")

    if not hourly_out.empty:
        _add_series(hourly_out, da_flag_series, "da_lockbook_infeasible_flag")
        _add_series(hourly_out, da_mwh_series, "da_lockbook_infeasible_mwh")
        _add_series(hourly_out, fallback_series, "fallback_optimization_flag")
        _add_series(hourly_out, opt_infeasible_series, "optimization_infeasible_flag")
        _add_series(hourly_out, missed_series.gt(1e-9).astype(float), "missed_activation_flag")
        _add_series(hourly_out, missed_series, "missed_activation_mwh")
        _add_series(hourly_out, activation_series.gt(1e-9).astype(float), "activation_event_flag")
        _add_series(hourly_out, activation_series, "activation_mwh")
        _add_series(hourly_out, soc_series.gt(1e-9).astype(float), "soc_violation_flag")
        _add_series(hourly_out, soc_series, "soc_violation_mwh")
        _add_series(hourly_out, reserve_series.gt(1e-9).astype(float), "reserve_headroom_shortfall_flag")
        _add_series(hourly_out, reserve_series, "reserve_headroom_shortfall_mw")
        flag_cols = [
            "da_lockbook_infeasible_flag",
            "fallback_optimization_flag",
            "optimization_infeasible_flag",
            "missed_activation_flag",
            "soc_violation_flag",
            "reserve_headroom_shortfall_flag",
        ]
        available_flags = [c for c in flag_cols if c in hourly_out.columns and not hourly_out[c].isna().all()]
        if available_flags:
            hourly_out["combined_infeasibility_flag"] = hourly_out[available_flags].fillna(0.0).max(axis=1)
        else:
            hourly_out["combined_infeasibility_flag"] = np.nan
            _metric_warning(warnings, info, "combined_infeasibility_flag", "No hourly invalidity flag source columns available.")
        hourly_out["source_files_used"] = ";".join(sorted(source_files_used))

    total_hours = int(hourly_out["timestamp_utc"].nunique()) if not hourly_out.empty else np.nan
    combined_hours = float(hourly_out["combined_infeasibility_flag"].fillna(0.0).gt(0.5).sum()) if "combined_infeasibility_flag" in hourly_out.columns and not hourly_out.empty else np.nan
    da_hours = float(hourly_out["da_lockbook_infeasible_flag"].fillna(0.0).gt(0.5).sum()) if "da_lockbook_infeasible_flag" in hourly_out.columns and not hourly_out.empty else np.nan
    da_mwh_total = float(hourly_out["da_lockbook_infeasible_mwh"].sum(skipna=True)) if "da_lockbook_infeasible_mwh" in hourly_out.columns and not hourly_out["da_lockbook_infeasible_mwh"].isna().all() else np.nan
    fallback_count = float(hourly_out["fallback_optimization_flag"].fillna(0.0).gt(0.5).sum()) if "fallback_optimization_flag" in hourly_out.columns and not hourly_out.empty else np.nan
    missed_count = float(hourly_out["missed_activation_flag"].fillna(0.0).gt(0.5).sum()) if "missed_activation_flag" in hourly_out.columns and not hourly_out.empty else np.nan
    activation_count = float(hourly_out["activation_event_flag"].fillna(0.0).gt(0.5).sum()) if "activation_event_flag" in hourly_out.columns and not hourly_out["activation_event_flag"].isna().all() else np.nan
    missed_total = float(hourly_out["missed_activation_mwh"].sum(skipna=True)) if "missed_activation_mwh" in hourly_out.columns and not hourly_out["missed_activation_mwh"].isna().all() else np.nan
    activation_total = float(hourly_out["activation_mwh"].sum(skipna=True)) if "activation_mwh" in hourly_out.columns and not hourly_out["activation_mwh"].isna().all() else np.nan
    soc_hours = float(hourly_out["soc_violation_flag"].fillna(0.0).gt(0.5).sum()) if "soc_violation_flag" in hourly_out.columns and not hourly_out.empty else np.nan
    soc_max = float(hourly_out["soc_violation_mwh"].max(skipna=True)) if "soc_violation_mwh" in hourly_out.columns and not hourly_out["soc_violation_mwh"].isna().all() else np.nan
    soc_sum = float(hourly_out["soc_violation_mwh"].sum(skipna=True)) if "soc_violation_mwh" in hourly_out.columns and not hourly_out["soc_violation_mwh"].isna().all() else np.nan
    reserve_hours = float(hourly_out["reserve_headroom_shortfall_flag"].fillna(0.0).gt(0.5).sum()) if "reserve_headroom_shortfall_flag" in hourly_out.columns and not hourly_out.empty else np.nan
    reserve_max = float(hourly_out["reserve_headroom_shortfall_mw"].max(skipna=True)) if "reserve_headroom_shortfall_mw" in hourly_out.columns and not hourly_out["reserve_headroom_shortfall_mw"].isna().all() else np.nan
    reserve_sum = float(hourly_out["reserve_headroom_shortfall_mw"].sum(skipna=True)) if "reserve_headroom_shortfall_mw" in hourly_out.columns and not hourly_out["reserve_headroom_shortfall_mw"].isna().all() else np.nan

    total_planned_trade = np.nan
    planned_cols = ["da_bid_abs_mwh_total", "bem_bid_abs_mwh_total", "id_abs_mwh_total"]
    if metrics:
        vals = [pd.to_numeric(pd.Series([metrics.get(c)]), errors="coerce").iloc[0] for c in planned_cols if c in metrics]
        vals = [float(v) for v in vals if np.isfinite(v)]
        if vals:
            total_planned_trade = float(sum(vals))
    if not np.isfinite(total_planned_trade):
        _metric_warning(warnings, info, "total_planned_trade_mwh", f"Missing performance metric columns: {planned_cols}")

    row = {
        "scenario": info.scenario,
        "model": info.model,
        "model_label": info.model_label,
        "quantile": info.quantile,
        "is_benchmark": info.is_benchmark,
        "simulation_valid": float(summary.get("simulation_valid", metrics.get("simulation_valid", np.nan))),
        "thesis_reportable": float(summary.get("thesis_reportable", metrics.get("thesis_reportable", np.nan))),
        "invalid_reason": str(summary.get("invalid_reason", metrics.get("invalid_reason", ""))),
        "total_hours": total_hours,
        "combined_infeasibility_hours": combined_hours,
        "combined_infeasibility_hours_share": _safe_ratio(combined_hours, total_hours, scenario=info.scenario, metric="combined_infeasibility_hours_share", warnings=warnings),
        "da_lockbook_infeasible_hours": da_hours,
        "da_lockbook_infeasible_mwh": da_mwh_total,
        "total_planned_trade_mwh": total_planned_trade,
        "da_lockbook_infeasible_mwh_share_of_total_planned_trade": _safe_ratio(da_mwh_total, total_planned_trade, scenario=info.scenario, metric="da_lockbook_infeasible_mwh_share_of_total_planned_trade", warnings=warnings),
        "fallback_optimization_count": fallback_count,
        "total_optimization_count": total_opt_count,
        "fallback_optimization_share": _safe_ratio(fallback_count, total_opt_count, scenario=info.scenario, metric="fallback_optimization_share", warnings=warnings),
        "missed_activation_count": missed_count,
        "total_activation_count": activation_count,
        "missed_activation_count_share": _safe_ratio(missed_count, activation_count, scenario=info.scenario, metric="missed_activation_count_share", warnings=warnings),
        "missed_activation_mwh": missed_total,
        "total_activation_mwh": activation_total,
        "missed_activation_mwh_share": _safe_ratio(missed_total, activation_total, scenario=info.scenario, metric="missed_activation_mwh_share", warnings=warnings),
        "soc_violation_hours": soc_hours,
        "max_soc_violation_mwh": soc_max,
        "sum_soc_violation_mwh": soc_sum,
        "reserve_headroom_shortfall_hours": reserve_hours,
        "max_reserve_headroom_shortfall_mw": reserve_max,
        "sum_reserve_headroom_shortfall_mwh_or_mw_hours": reserve_sum,
        "reserve_headroom_shortfall_unit": "MWh-equivalent",
        "diagnostic_completeness_status": "complete" if not warnings else "partial",
        "missing_source_columns": ";".join(sorted({str(w["metric"]) for w in warnings if w.get("warning") == "missing_source_columns"})),
        "source_files_used": ";".join(sorted(source_files_used)),
    }
    if np.isfinite(combined_hours) and np.isfinite(total_hours) and combined_hours > total_hours:
        warnings.append({"scenario": info.scenario, "metric": "combined_infeasibility_hours", "warning": "validation_error", "details": "combined_infeasibility_hours > total_hours"})
    for c in [k for k in row if k.endswith("_share")]:
        v = row[c]
        if np.isfinite(v) and not (0.0 <= float(v) <= 1.0):
            warnings.append({"scenario": info.scenario, "metric": c, "warning": "share_out_of_bounds", "details": str(v)})
    return hourly_out, row, inventory_rows, warnings


def build_outputs(run_dir: Path, out_dir: Path) -> dict[str, Path]:
    scenarios = discover_scenarios(run_dir)
    if not scenarios:
        raise FileNotFoundError(f"No nested scenario folders with backtest_summary.json found under {run_dir}")
    hourly_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    for info in scenarios:
        hourly, summary, inventory, warnings = build_scenario(info)
        hourly_parts.append(hourly)
        summary_rows.append(summary)
        inventory_rows.extend(inventory)
        warning_rows.extend(warnings)

    diag_dir = out_dir / "backup" / "diagnostics"
    warn_dir = out_dir / "backup" / "warnings"
    diag_dir.mkdir(parents=True, exist_ok=True)
    warn_dir.mkdir(parents=True, exist_ok=True)
    summary_path = diag_dir / "rq2_invalidity_severity_summary.csv"
    hourly_path = diag_dir / "rq2_invalidity_severity_by_hour.csv"
    inventory_path = diag_dir / "rq2_invalidity_source_inventory.csv"
    warnings_path = warn_dir / "rq2_invalidity_severity_warnings.csv"

    summary_df = pd.DataFrame(summary_rows).sort_values(["is_benchmark", "model_label", "quantile", "scenario"]).reset_index(drop=True)
    hourly_df = pd.concat(hourly_parts, ignore_index=True) if hourly_parts else _empty_hourly(ScenarioInfo(Path("."), "", "", "", "", False))
    if not hourly_df.empty:
        dup = hourly_df.duplicated(["scenario", "timestamp_utc"])
        if dup.any():
            warning_rows.append({"scenario": "", "metric": "hourly_uniqueness", "warning": "duplicate_hourly_rows", "details": f"{int(dup.sum())} duplicate scenario/timestamp rows"})
        hourly_df = hourly_df.drop_duplicates(["scenario", "timestamp_utc"], keep="last").sort_values(["scenario", "timestamp_utc"]).reset_index(drop=True)
    inventory_df = pd.DataFrame(inventory_rows)
    warnings_df = pd.DataFrame(warning_rows)

    summary_df.to_csv(summary_path, index=False)
    hourly_df.to_csv(hourly_path, index=False)
    inventory_df.to_csv(inventory_path, index=False)
    warnings_df.to_csv(warnings_path, index=False)
    return {
        "summary": summary_path,
        "hourly": hourly_path,
        "inventory": inventory_path,
        "warnings": warnings_path,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build RQ2 invalidity severity diagnostics from existing outputs.")
    p.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = build_outputs(Path(args.run_dir), Path(args.out_dir))
    for name, path in paths.items():
        print(f"[OK] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
