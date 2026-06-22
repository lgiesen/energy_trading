#!/usr/bin/env python3
"""Build thesis-ready RQ2 visualizations from existing simulation outputs.

This script is a reporting layer only. It reads completed simulation folders,
filters result-section outputs to thesis-reportable valid rows, and writes
diagnostic CSVs for missing or invalid scenarios. It never runs simulations.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_trading.evaluation.style import (  # noqa: E402
    GEO_SEQUENTIAL_BLUE,
    MARKET_COLOR_MAP,
    THESIS_PALETTE,
    apply_geo_style,
    get_model_color,
    thesis_titlecase,
)

DEFAULT_RUN_ROOT = Path("artifacts/simulation_runs/thesis_final_multi_2m_20260620T091938Z")
DEFAULT_OUT_ROOT = Path("artifacts/benchmark/rq2_simulation_benchmark")
DEFAULT_FORECAST_BENCHMARK_DIR = Path("artifacts/benchmark/rq1_ml_model_benchmark")
DEFAULT_MODELS = ("linear", "xgb", "tft")
DEFAULT_QUANTILES = ("p10", "p30", "p50", "p70", "p90")
TRUTH_REFERENCE_COLOR = "#000000"
MONTH_ABBR_FULL = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}
MODEL_LABELS = {"linear": "RLQR", "xgb": "XGB", "tft": "TFT"}
MODEL_ORDER = ["RLQR", "XGB", "TFT"]
BENCHMARK_ORDER = ["Naive", "RHPF"]
TARGET_LABELS = {
    "pred_da_price": "DA Price",
    "pred_afrr_capacity_price_pos": "aFRR Capacity Price +",
    "pred_afrr_capacity_price_neg": "aFRR Capacity Price $-$",
    "pred_afrr_activation_price_pos": "aFRR Activation Price +",
    "pred_afrr_activation_price_neg": "aFRR Activation Price $-$",
    "pred_afrr_activation_rate_pos": "aFRR Activation Rate +",
    "pred_afrr_activation_rate_neg": "aFRR Activation Rate $-$",
}
TARGET_SLUGS = {
    "pred_da_price": "da_price",
    "pred_afrr_capacity_price_pos": "afrr_capacity_price_pos",
    "pred_afrr_capacity_price_neg": "afrr_capacity_price_neg",
    "pred_afrr_activation_price_pos": "afrr_activation_price_pos",
    "pred_afrr_activation_price_neg": "afrr_activation_price_neg",
    "pred_afrr_activation_rate_pos": "afrr_activation_rate_pos",
    "pred_afrr_activation_rate_neg": "afrr_activation_rate_neg",
}
COMPONENT_COLUMNS = [
    ("da_net_revenue_eur", "DA net"),
    ("id_net_revenue_eur", "ID net"),
    ("bcm_capacity_revenue_eur", "BCM capacity"),
    ("bcm_linked_activation_revenue_eur", "BCM activation"),
    ("bem_activation_revenue_eur", "BEM activation"),
    ("realized_degradation_cost_eur", "Degradation cost"),
    ("realized_aux_cost_eur", "Auxiliary cost"),
    ("transaction_cost_eur", "Transaction cost"),
    ("offer_cost_eur", "Offer cost"),
    ("penalty_cost_eur", "Penalty cost"),
    ("terminal_soc_repair_cost_eur", "Terminal SoC repair"),
]
COMPONENT_COST_COLUMNS = {
    "realized_degradation_cost_eur",
    "realized_aux_cost_eur",
    "transaction_cost_eur",
    "offer_cost_eur",
    "penalty_cost_eur",
    "terminal_soc_repair_cost_eur",
}
MARKET_COMPONENT_COLORS = {
    "DA net": MARKET_COLOR_MAP["DA"],
    "ID net": MARKET_COLOR_MAP["ID"],
    "BCM capacity": MARKET_COLOR_MAP["BCM capacity"],
    "BCM activation": MARKET_COLOR_MAP["BCM activation"],
    "BEM activation": MARKET_COLOR_MAP["BEM"],
    "Degradation cost": "#F0746E",
    "Auxiliary cost": "#DC3977",
    "Transaction cost": "#7A7A7A",
    "Offer cost": "#9EC9E2",
    "Penalty cost": "#333333",
    "Terminal SoC repair": "#045275",
}
ALL_OUTPUT_DIRS = [
    "result_section/figures",
    "result_section/tables",
    "result_section/latex_figures",
    "appendix/figures",
    "appendix/tables",
    "appendix/latex_figures",
    "backup/csv",
    "backup/diagnostics",
    "backup/warnings",
]


@dataclass(frozen=True)
class ScenarioSpec:
    folder: str
    model_key: str
    model_display: str
    quantile: str
    is_benchmark: bool
    benchmark_name: str = ""


def _parse_csv_arg(value: str, default: tuple[str, ...]) -> list[str]:
    items = [x.strip() for x in str(value or "").split(",") if x.strip()]
    return items or list(default)


def parse_scenario_folder(folder: str) -> ScenarioSpec | None:
    if folder == "benchmarks_naive":
        return ScenarioSpec(folder, "benchmark", "Naive", "benchmark", True, "Naive")
    if folder == "benchmarks_rhpf":
        return ScenarioSpec(folder, "benchmark", "RHPF", "benchmark", True, "RHPF")
    if "_" not in folder:
        return None
    model, quantile = folder.rsplit("_", 1)
    if model not in MODEL_LABELS or not quantile.startswith("p"):
        return None
    return ScenarioSpec(folder, model, MODEL_LABELS[model], quantile, False, "")


def _latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    repl = {
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
    }
    return "".join(repl.get(ch, ch) for ch in text)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def _is_valid_reportable(row: pd.Series) -> bool:
    sim = _safe_float(row.get("simulation_valid", np.nan))
    thesis = _safe_float(row.get("thesis_reportable", np.nan))
    return sim >= 0.5 and thesis >= 0.5


def _read_csv(path: Path, inventory: list[dict[str, Any]], folder: str, file_type: str, used: bool, reason: str) -> pd.DataFrame | None:
    exists = path.exists()
    row_count: int | str = ""
    df: pd.DataFrame | None = None
    if exists:
        try:
            df = pd.read_csv(path)
            row_count = len(df)
        except Exception as exc:
            reason = f"read_failed:{exc}"
            df = None
    inventory.append(
        {
            "folder": folder,
            "path": str(path),
            "file_type": file_type,
            "exists": bool(exists),
            "used": bool(used and df is not None),
            "reason_if_not_used": "" if used and df is not None else reason,
            "row_count": row_count,
        }
    )
    return df


def expected_scenarios(models: list[str], quantiles: list[str]) -> list[ScenarioSpec]:
    specs = [
        ScenarioSpec("benchmarks_naive", "benchmark", "Naive", "benchmark", True, "Naive"),
        ScenarioSpec("benchmarks_rhpf", "benchmark", "RHPF", "benchmark", True, "RHPF"),
    ]
    for model in models:
        label = MODEL_LABELS.get(model, model.upper())
        for quantile in quantiles:
            specs.append(ScenarioSpec(f"{model}_{quantile}", model, label, quantile, False, ""))
    return specs


def _infer_duration_days(rows: pd.DataFrame, supplied_days: float | None) -> tuple[float, str]:
    if supplied_days is not None:
        if supplied_days <= 0:
            raise ValueError("--simulation-days must be positive.")
        return float(supplied_days), "supplied --simulation-days"
    if "n_days" in rows.columns:
        vals = pd.to_numeric(rows["n_days"], errors="coerce").dropna()
        vals = vals[vals > 0]
        if not vals.empty:
            return float(vals.median()), "median n_days from performance_metrics_all_scenarios.csv"
    for start_col, end_col in [("start_utc", "end_utc"), ("start", "end")]:
        if start_col in rows.columns and end_col in rows.columns:
            starts = pd.to_datetime(rows[start_col], errors="coerce", utc=True)
            ends = pd.to_datetime(rows[end_col], errors="coerce", utc=True)
            days = ((ends - starts).dt.total_seconds() / 86400.0).dropna()
            days = days[days > 0]
            if not days.empty:
                return float(days.median()), f"median {start_col}/{end_col} duration"
    raise ValueError("Could not infer simulation duration. Pass --simulation-days.")


def _profit_for_spec(row: pd.Series, spec: ScenarioSpec) -> float:
    if spec.benchmark_name == "Naive":
        return _safe_float(row.get("naive_total_pnl_eur", np.nan))
    if spec.benchmark_name == "RHPF":
        return _safe_float(row.get("rolling_perfect_foresight_same_rules_total_pnl_eur", np.nan))
    for col in ["realized_total_pnl_eur", "realized_net_revenue_eur"]:
        val = _safe_float(row.get(col, np.nan))
        if math.isfinite(val):
            return val
    return math.nan


def collect_summaries(
    run_root: Path,
    models: list[str],
    quantiles: list[str],
    split: str,
    simulation_days: float | None,
    strict_validity: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], float, str]:
    inventory: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    validity: list[dict[str, Any]] = []
    raw_for_duration: list[pd.DataFrame] = []

    for spec in expected_scenarios(models, quantiles):
        folder_path = run_root / spec.folder
        if not folder_path.exists():
            warnings.append({"severity": "warning", "scenario": spec.folder, "message": "missing expected folder"})
            inventory.append({"folder": spec.folder, "path": str(folder_path), "file_type": "folder", "exists": False, "used": False, "reason_if_not_used": "missing expected folder", "row_count": ""})
            continue
        metrics = _read_csv(folder_path / "performance_metrics_all_scenarios.csv", inventory, spec.folder, "performance_metrics", True, "")
        overview = _read_csv(folder_path / "strategy_overview.csv", inventory, spec.folder, "strategy_overview", False, "diagnostic only")
        _read_csv(folder_path / "strategy_overview_valid_only.csv", inventory, spec.folder, "strategy_overview_valid_only", False, "validity cross-check")
        _read_csv(folder_path / "quantile_sweep_summary.csv", inventory, spec.folder, "quantile_sweep_summary", False, "not required for RQ2 result table")
        if metrics is None or metrics.empty:
            warnings.append({"severity": "warning", "scenario": spec.folder, "message": "missing or empty performance_metrics_all_scenarios.csv"})
            continue
        raw_for_duration.append(metrics)
        candidates = metrics.copy()
        if "split" in candidates.columns:
            candidates = candidates.loc[candidates["split"].astype(str) == split].copy()
        if candidates.empty:
            warnings.append({"severity": "warning", "scenario": spec.folder, "message": f"no rows for split={split}"})
            continue
        row = candidates.iloc[0].copy()
        # Strategy overview contains benchmark path validity and PnL for Naive/RHPF.
        if overview is not None and not overview.empty:
            over = overview.copy()
            if "split" in over.columns:
                over = over.loc[over["split"].astype(str) == split].copy()
            if not over.empty:
                o = over.iloc[0]
                if spec.benchmark_name == "Naive":
                    for col in ["simulation_valid", "thesis_reportable", "invalid_reason", "naive_total_pnl_eur", "start", "end"]:
                        if col in o.index:
                            row[col] = o[col]
                elif spec.benchmark_name == "RHPF":
                    for col in ["simulation_valid", "thesis_reportable", "invalid_reason", "rolling_perfect_foresight_same_rules_total_pnl_eur", "start", "end"]:
                        if col in o.index:
                            row[col] = o[col]
        valid = _is_valid_reportable(row)
        profit = _profit_for_spec(row, spec)
        included = bool((valid or not strict_validity) and math.isfinite(profit))
        record = {
            "folder": spec.folder,
            "scenario": str(row.get("scenario", spec.quantile)),
            "split": split,
            "model_key": spec.model_key,
            "model": spec.model_display,
            "quantile": spec.quantile,
            "is_benchmark": spec.is_benchmark,
            "benchmark": spec.benchmark_name,
            "realized_profit_eur": profit,
            "simulation_valid": _safe_float(row.get("simulation_valid", np.nan)),
            "thesis_reportable": _safe_float(row.get("thesis_reportable", np.nan)),
            "invalid_reason": str(row.get("invalid_reason", "")),
            "source_file": str(folder_path / "performance_metrics_all_scenarios.csv"),
            "included_in_result_section": included,
            "start_utc": str(row.get("start_utc", row.get("start", ""))),
            "end_utc": str(row.get("end_utc", row.get("end", ""))),
            "n_days_source": _safe_float(row.get("n_days", np.nan)),
        }
        for col, _label in COMPONENT_COLUMNS:
            record[col] = _safe_float(row.get(col, np.nan))
        rows.append(record)
        validity.append(
            {
                "scenario": spec.folder,
                "model": spec.model_display,
                "quantile": spec.quantile,
                "simulation_valid": record["simulation_valid"],
                "thesis_reportable": record["thesis_reportable"],
                "invalid_reason": record["invalid_reason"],
                "source_file": record["source_file"],
                "included_in_result_section": record["included_in_result_section"],
            }
        )
        if not record["included_in_result_section"]:
            warnings.append(
                {
                    "severity": "warning",
                    "scenario": spec.folder,
                    "message": "excluded from result section because simulation_valid/thesis_reportable is false or profit is missing",
                }
            )
        elif not valid and not strict_validity:
            warnings.append(
                {
                    "severity": "warning",
                    "scenario": spec.folder,
                    "message": "included despite invalid simulation status because --no-strict-validity was used",
                }
            )

    if not rows:
        raise FileNotFoundError(f"No usable RQ2 simulation summaries found under {run_root}.")
    duration_source_df = pd.concat(raw_for_duration, ignore_index=True) if raw_for_duration else pd.DataFrame(rows)
    days, duration_source = _infer_duration_days(duration_source_df, simulation_days)
    factor = 365.0 / days
    summary = pd.DataFrame(rows)
    summary["simulation_days"] = days
    summary["annualization_factor"] = factor
    summary["annualized_profit_eur_per_year"] = pd.to_numeric(summary["realized_profit_eur"], errors="coerce") * factor
    warnings.append({"severity": "info", "scenario": "all", "message": f"simulation_days={days:.6g} inferred from {duration_source}"})
    warnings.append({"severity": "info", "scenario": "benchmarks", "message": "Naive/RHPF benchmark values are treated as global and repeated across quantile rows when valid."})
    return summary, pd.DataFrame(validity), pd.DataFrame(inventory), pd.DataFrame(warnings), warnings, inventory, days, duration_source


def build_result_tables(summary: pd.DataFrame, quantiles: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = summary.loc[summary["included_in_result_section"]].copy()
    for optional_col in ["benchmark", "simulation_valid", "thesis_reportable"]:
        if optional_col not in valid.columns:
            valid[optional_col] = ""
    bench_values = (
        valid.loc[valid["is_benchmark"], ["model", "benchmark", "annualized_profit_eur_per_year", "realized_profit_eur", "simulation_valid", "thesis_reportable"]]
        .drop_duplicates("model")
        .copy()
    )
    rows: list[dict[str, Any]] = []
    for q in quantiles:
        rec: dict[str, Any] = {"quantile": q}
        for b in BENCHMARK_ORDER:
            match = bench_values.loc[bench_values["model"] == b]
            rec[b] = float(match["annualized_profit_eur_per_year"].iloc[0]) if not match.empty else math.nan
        for m in MODEL_ORDER:
            match = valid.loc[(valid["model"] == m) & (valid["quantile"] == q)]
            rec[m] = float(match["annualized_profit_eur_per_year"].iloc[0]) if not match.empty else math.nan
        model_vals = {m: rec[m] for m in MODEL_ORDER if math.isfinite(_safe_float(rec[m]))}
        if model_vals:
            best_model = max(model_vals, key=model_vals.get)
            best_value = model_vals[best_model]
        else:
            best_model = ""
            best_value = math.nan
        rec["best_model"] = best_model
        rec["best_annualized_profit_eur_per_year"] = best_value
        rec["best_vs_naive_pct"] = ((best_value / rec["Naive"]) - 1.0) * 100.0 if math.isfinite(best_value) and math.isfinite(_safe_float(rec["Naive"])) and rec["Naive"] != 0 else math.nan
        rec["best_vs_rhpf_pct"] = ((best_value / rec["RHPF"]) - 1.0) * 100.0 if math.isfinite(best_value) and math.isfinite(_safe_float(rec["RHPF"])) and rec["RHPF"] != 0 else math.nan
        rows.append(rec)
    result = pd.DataFrame(rows)
    heatmap = valid.loc[~valid["is_benchmark"], ["model", "quantile", "annualized_profit_eur_per_year", "realized_profit_eur", "included_in_result_section"]].copy()
    return result, heatmap, bench_values


def build_quantile_sweep_data(summary: pd.DataFrame, quantiles: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for benchmark in BENCHMARK_ORDER:
        match = summary.loc[summary["model"].eq(benchmark)].copy()
        if match.empty:
            continue
        row = match.iloc[0]
        value = _safe_float(row.get("annualized_profit_eur_per_year", np.nan))
        if not math.isfinite(value):
            continue
        for quantile in quantiles:
            rows.append(
                {
                    "series": benchmark,
                    "model": benchmark,
                    "quantile": quantile,
                    "annualized_net_profit_eur_per_year": value,
                    "realized_profit_eur": row.get("realized_profit_eur", np.nan),
                    "simulation_valid": row.get("simulation_valid", np.nan),
                    "thesis_reportable": row.get("thesis_reportable", np.nan),
                    "included_in_result_section": row.get("included_in_result_section", False),
                    "invalid_reason": row.get("invalid_reason", ""),
                }
            )

    model_rows = summary.loc[~summary["is_benchmark"].astype(bool)].copy()
    for _, row in model_rows.iterrows():
        value = _safe_float(row.get("annualized_profit_eur_per_year", np.nan))
        if not math.isfinite(value):
            continue
        rows.append(
            {
                "series": str(row["model"]),
                "model": str(row["model"]),
                "quantile": str(row["quantile"]),
                "annualized_net_profit_eur_per_year": value,
                "realized_profit_eur": row.get("realized_profit_eur", np.nan),
                "simulation_valid": row.get("simulation_valid", np.nan),
                "thesis_reportable": row.get("thesis_reportable", np.nan),
                "included_in_result_section": row.get("included_in_result_section", False),
                "invalid_reason": row.get("invalid_reason", ""),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["quantile"] = pd.Categorical(out["quantile"], categories=quantiles, ordered=True)
    out["series"] = pd.Categorical(out["series"], categories=["Naive", "RHPF", *MODEL_ORDER], ordered=True)
    return out.sort_values(["series", "quantile"]).reset_index(drop=True)


def build_profit_heatmap_data(summary: pd.DataFrame, quantiles: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for q in quantiles:
        rec: dict[str, Any] = {"quantile": q}
        for benchmark in BENCHMARK_ORDER:
            match = summary.loc[summary["model"].eq(benchmark)].copy()
            value = _safe_float(match["annualized_profit_eur_per_year"].iloc[0]) if not match.empty else math.nan
            rec[benchmark] = value
        for model in MODEL_ORDER:
            match = summary.loc[(summary["model"].eq(model)) & (summary["quantile"].astype(str).eq(q))].copy()
            value = _safe_float(match["annualized_profit_eur_per_year"].iloc[0]) if not match.empty else math.nan
            rec[model] = value
        rows.append(rec)
    return pd.DataFrame(rows)


def build_best_quantile_components(summary: pd.DataFrame) -> pd.DataFrame:
    valid = summary.loc[~summary["is_benchmark"].astype(bool)].copy()
    valid["annualized_profit_eur_per_year"] = pd.to_numeric(
        valid["annualized_profit_eur_per_year"], errors="coerce"
    )
    valid = valid.dropna(subset=["annualized_profit_eur_per_year"]).copy()
    rows: list[dict[str, Any]] = []
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "quantile",
                "component",
                "component_value_eur",
                "annualized_component_value_eur_per_year",
                "realized_profit_eur",
                "annualized_profit_eur_per_year",
                "simulation_valid",
                "thesis_reportable",
                "included_in_result_section",
                "invalid_reason",
            ]
        )
    for model in MODEL_ORDER:
        model_rows = valid.loc[valid["model"].eq(model)].copy()
        if model_rows.empty:
            continue
        best = model_rows.sort_values("annualized_profit_eur_per_year", ascending=False).iloc[0]
        factor = _safe_float(best.get("annualization_factor", np.nan))
        for col, label in COMPONENT_COLUMNS:
            raw = _safe_float(best.get(col, np.nan))
            if not math.isfinite(raw):
                continue
            signed = -abs(raw) if col in COMPONENT_COST_COLUMNS else raw
            rows.append(
                {
                    "model": model,
                    "quantile": best["quantile"],
                    "component": label,
                    "component_column": col,
                    "component_value_eur": signed,
                    "annualized_component_value_eur_per_year": signed * factor if math.isfinite(factor) else math.nan,
                    "realized_profit_eur": best["realized_profit_eur"],
                    "annualized_profit_eur_per_year": best["annualized_profit_eur_per_year"],
                    "simulation_valid": best.get("simulation_valid", np.nan),
                    "thesis_reportable": best.get("thesis_reportable", np.nan),
                    "included_in_result_section": best.get("included_in_result_section", False),
                    "invalid_reason": best.get("invalid_reason", ""),
                }
            )
    return pd.DataFrame(rows)


def _best_rows_for_cumulative_paths(summary: pd.DataFrame) -> pd.DataFrame:
    valid = summary.copy()
    selected: list[pd.Series] = []
    for benchmark in BENCHMARK_ORDER:
        rows = valid.loc[valid["model"].eq(benchmark)]
        if not rows.empty:
            selected.append(rows.iloc[0])

    model_rows = valid.loc[~valid["is_benchmark"].astype(bool)].copy()
    model_rows["annualized_profit_eur_per_year"] = pd.to_numeric(
        model_rows["annualized_profit_eur_per_year"], errors="coerce"
    )
    model_rows = model_rows.dropna(subset=["annualized_profit_eur_per_year"])
    for model in MODEL_ORDER:
        rows = model_rows.loc[model_rows["model"].eq(model)].copy()
        if rows.empty:
            continue
        selected.append(rows.sort_values("annualized_profit_eur_per_year", ascending=False).iloc[0])
    return pd.DataFrame(selected) if selected else pd.DataFrame()


def build_cumulative_pnl_paths(summary: pd.DataFrame, *, run_root: Path) -> pd.DataFrame:
    selected = _best_rows_for_cumulative_paths(summary)
    columns = [
        "timestamp_utc",
        "series",
        "model",
        "quantile",
        "path_type",
        "cum_pnl_eur",
        "pnl_eur",
        "folder",
        "simulation_valid",
        "thesis_reportable",
        "included_in_result_section",
        "invalid_reason",
    ]
    if selected.empty:
        return pd.DataFrame(columns=columns)

    frames: list[pd.DataFrame] = []
    for _, row in selected.iterrows():
        folder = str(row["folder"])
        model = str(row["model"])
        quantile = str(row["quantile"])
        path_type = "model"
        if model == "Naive":
            path_type = "naive"
        elif model == "RHPF":
            path_type = "rhpf"

        path = run_root / folder / "performance_paths_long.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "path_type" in df.columns:
            df = df.loc[df["path_type"].astype(str).eq(path_type)].copy()
        if "available" in df.columns:
            df = df.loc[pd.to_numeric(df["available"], errors="coerce").fillna(0.0) >= 0.5].copy()
        if df.empty or "timestamp_utc" not in df.columns:
            continue
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").copy()
        if df.empty:
            continue
        if "cum_pnl_eur" not in df.columns:
            df["cum_pnl_eur"] = pd.to_numeric(df.get("pnl_eur", 0.0), errors="coerce").fillna(0.0).cumsum()
        df["cum_pnl_eur"] = pd.to_numeric(df["cum_pnl_eur"], errors="coerce")
        df["pnl_eur"] = pd.to_numeric(df.get("pnl_eur", np.nan), errors="coerce")
        series = model if model in BENCHMARK_ORDER else f"{model} {quantile}"
        frames.append(
            df.assign(
                series=series,
                model=model,
                quantile=quantile,
                folder=folder,
                simulation_valid=row.get("simulation_valid", np.nan),
                thesis_reportable=row.get("thesis_reportable", np.nan),
                included_in_result_section=row.get("included_in_result_section", False),
                invalid_reason=row.get("invalid_reason", ""),
            )[columns]
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _num_col(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _sum_matching_cols(df: pd.DataFrame, *, prefix: str, suffix: str) -> pd.Series:
    cols = [c for c in df.columns if c.startswith(prefix) and c.endswith(suffix)]
    if not cols:
        return pd.Series(0.0, index=df.index, dtype=float)
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)


def _first_available_num_col(df: pd.DataFrame, candidates: list[str]) -> tuple[pd.Series, str]:
    for column in candidates:
        if column in df.columns:
            return _num_col(df, column), column
    return pd.Series(0.0, index=df.index, dtype=float), ""


def _find_backtest_hourly_path(scenario_dir: Path) -> Path | None:
    direct = scenario_dir / "multi" / "p90_p90" / "backtest_hourly.parquet"
    if direct.exists():
        return direct
    candidates = sorted(scenario_dir.rglob("backtest_hourly.parquet"))
    return candidates[0] if candidates else None


def _first_full_week_dates(timestamps: pd.Series) -> set[Any]:
    ts = pd.to_datetime(timestamps, errors="coerce", utc=True).dropna()
    if ts.empty:
        return set()
    first_date = ts.min().normalize()
    first_monday = first_date + pd.Timedelta(days=(7 - first_date.weekday()) % 7)
    week = pd.date_range(first_monday, first_monday + pd.Timedelta(days=6), freq="1D", tz="UTC")
    return {d.date() for d in week}


def build_market_dispatch_soc_day(
    summary: pd.DataFrame,
    *,
    run_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "timestamp_utc",
        "date",
        "hour_utc",
        "component",
        "market",
        "direction",
        "mw_signed",
        "mw_abs",
        "soc_mwh",
        "pnl_eur",
        "cumulative_pnl_eur",
        "source_column",
        "pnl_source_column",
        "model",
        "quantile",
        "scenario_folder",
        "scenario_output_dir",
    ]
    warnings: list[dict[str, Any]] = []
    model_rows = summary.loc[~summary["is_benchmark"].astype(bool)].copy()
    model_rows["annualized_profit_eur_per_year"] = pd.to_numeric(model_rows["annualized_profit_eur_per_year"], errors="coerce")
    model_rows = model_rows.dropna(subset=["annualized_profit_eur_per_year"]).copy()
    if model_rows.empty:
        warnings.append({"severity": "warning", "scenario": "", "message": "No numeric model scenarios available for dispatch/SOC example-day figure."})
        return pd.DataFrame(columns=columns), pd.DataFrame(warnings)
    best = model_rows.sort_values("annualized_profit_eur_per_year", ascending=False).iloc[0]
    scenario_dir = run_root / str(best["folder"])
    hourly_path = _find_backtest_hourly_path(scenario_dir)
    if hourly_path is None:
        warnings.append({"severity": "warning", "scenario": str(best["folder"]), "message": "Missing nested backtest_hourly.parquet for dispatch/SOC example-day figure."})
        return pd.DataFrame(columns=columns), pd.DataFrame(warnings)

    hourly = pd.read_parquet(hourly_path).copy()
    if "timestamp_utc" not in hourly.columns:
        warnings.append({"severity": "warning", "scenario": str(best["folder"]), "message": f"Missing timestamp_utc in {hourly_path}."})
        return pd.DataFrame(columns=columns), pd.DataFrame(warnings)
    hourly["timestamp_utc"] = pd.to_datetime(hourly["timestamp_utc"], errors="coerce", utc=True)
    hourly = hourly.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").copy()
    if hourly.empty:
        warnings.append({"severity": "warning", "scenario": str(best["folder"]), "message": f"No valid timestamps in {hourly_path}."})
        return pd.DataFrame(columns=columns), pd.DataFrame(warnings)

    d = hourly.copy()
    d["date"] = d["timestamp_utc"].dt.date
    d["da_charge_mw"] = _num_col(d, "real_da_buy_mwh")
    d["da_discharge_mw"] = _num_col(d, "real_da_sell_mwh")
    d["bem_charge_mw"] = _num_col(d, "real_bem_only_executed_neg_mwh")
    d["bem_discharge_mw"] = _num_col(d, "real_bem_only_executed_pos_mwh")
    d["bcm_charge_mw"], bcm_charge_source = _first_available_num_col(
        d,
        [
            "real_bcm_linked_neg_activation_mwh",
            "real_bcm_p90_executed_act_neg_mw",
            "real_executed_afrr_act_neg_mw",
        ],
    )
    d["bcm_discharge_mw"], bcm_discharge_source = _first_available_num_col(
        d,
        [
            "real_bcm_linked_pos_activation_mwh",
            "real_bcm_p90_executed_act_pos_mw",
            "real_executed_afrr_act_pos_mw",
        ],
    )
    if not bcm_charge_source:
        d["bcm_charge_mw"] = _sum_matching_cols(d, prefix="real_executed_afrr_act_neg_bin_", suffix="_mw")
        bcm_charge_source = "real_executed_afrr_act_neg_bin_*_mw"
    if not bcm_discharge_source:
        d["bcm_discharge_mw"] = _sum_matching_cols(d, prefix="real_executed_afrr_act_pos_bin_", suffix="_mw")
        bcm_discharge_source = "real_executed_afrr_act_pos_bin_*_mw"
    d["soc_mwh"] = pd.to_numeric(d.get("real_soc_mwh", d.get("soc_mwh", np.nan)), errors="coerce")
    d["pnl_eur"], pnl_source = _first_available_num_col(
        d,
        [
            "real_pnl_eur",
            "real_da_pnl_eur",
            "perfect_foresight_pnl_eur",
            "pnl_eur",
        ],
    )
    if not pnl_source:
        pnl_parts = [
            _num_col(d, "real_revenue_da_eur") - _num_col(d, "real_cost_da_eur"),
            _num_col(d, "real_revenue_id_eur") - _num_col(d, "real_cost_id_eur"),
            _num_col(d, "real_bcm_capacity_revenue_eur"),
            _num_col(d, "real_bcm_linked_activation_revenue_eur"),
            _num_col(d, "real_bem_only_activation_revenue_eur"),
            -_num_col(d, "real_transaction_cost_eur"),
            -_num_col(d, "real_degradation_cost_eur"),
            -_num_col(d, "real_aux_cost_eur"),
        ]
        d["pnl_eur"] = sum(pnl_parts)
        pnl_source = "reconstructed_real_market_revenue_cost_columns"
    for col in ["real_power_violation_charge_mw", "real_power_violation_discharge_mw", "real_protected_soc_violation_pos_mwh", "real_protected_soc_violation_neg_mwh"]:
        d[col] = _num_col(d, col)
    d["activity_mw"] = d[["da_charge_mw", "da_discharge_mw", "bem_charge_mw", "bem_discharge_mw", "bcm_charge_mw", "bcm_discharge_mw"]].abs().sum(axis=1)
    d["non_da_activity_mw"] = d[["bem_charge_mw", "bem_discharge_mw", "bcm_charge_mw", "bcm_discharge_mw"]].abs().sum(axis=1)
    d["direct_violation"] = d[["real_power_violation_charge_mw", "real_power_violation_discharge_mw", "real_protected_soc_violation_pos_mwh", "real_protected_soc_violation_neg_mwh"]].abs().sum(axis=1)

    candidates = d.copy()
    daily = candidates.groupby("date", as_index=False).agg(
        n_rows=("timestamp_utc", "size"),
        activity_mw=("activity_mw", "sum"),
        non_da_activity_mw=("non_da_activity_mw", "sum"),
        direct_violation=("direct_violation", "sum"),
    )
    complete = daily.loc[daily["n_rows"] >= 24].copy()
    if complete.empty:
        complete = daily.copy()
        warnings.append({"severity": "warning", "scenario": str(best["folder"]), "message": "No complete 24-row day available for dispatch/SOC example-day figure."})
    no_direct_viol = complete.loc[complete["direct_violation"].abs() <= 1e-9].copy()
    selector = no_direct_viol if not no_direct_viol.empty else complete
    selected_row = selector.sort_values(["non_da_activity_mw", "activity_mw", "date"], ascending=[False, False, True]).iloc[0]
    selected_day = selected_row["date"]
    day = d.loc[d["date"].eq(selected_day)].copy()
    day["cumulative_pnl_eur"] = pd.to_numeric(day["pnl_eur"], errors="coerce").fillna(0.0).cumsum()
    warnings.append(
        {
            "severity": "info",
            "scenario": str(best["folder"]),
            "message": "Selected dispatch day maximizes non-DA activation activity among complete no-direct-violation days in the available simulation window.",
            "selected_day": str(selected_day),
            "non_da_activity_mw": float(selected_row.get("non_da_activity_mw", math.nan)),
            "activity_mw": float(selected_row.get("activity_mw", math.nan)),
        }
    )

    scenario_invalid = str(best.get("invalid_reason", "") or "")
    sim_valid = _safe_float(best.get("simulation_valid", np.nan))
    thesis_reportable = _safe_float(best.get("thesis_reportable", np.nan))
    selected_violation = float(day["direct_violation"].sum()) if not day.empty else math.nan
    if sim_valid < 0.5 or thesis_reportable < 0.5 or scenario_invalid:
        warnings.append(
            {
                "severity": "warning",
                "scenario": str(best["folder"]),
                "message": "Selected scenario has global simulation invalidity flags.",
                "simulation_valid": sim_valid,
                "thesis_reportable": thesis_reportable,
                "invalid_reason": scenario_invalid,
                "selected_day": str(selected_day),
            }
        )
    if math.isfinite(selected_violation) and selected_violation > 1e-9:
        warnings.append(
            {
                "severity": "warning",
                "scenario": str(best["folder"]),
                "message": "Selected dispatch day has direct power/protected-SoC violation columns above zero.",
                "selected_day": str(selected_day),
                "direct_violation_sum": selected_violation,
            }
        )
    else:
        warnings.append(
            {
                "severity": "info",
                "scenario": str(best["folder"]),
                "message": "Selected dispatch day has no direct power/protected-SoC violations in the checked columns.",
                "selected_day": str(selected_day),
                "direct_violation_sum": selected_violation,
            }
        )

    specs = [
        ("DA buy", "DA", "charge", "da_charge_mw", "real_da_buy_mwh", 1.0),
        ("BEM negative activation", "BEM", "charge", "bem_charge_mw", "real_bem_only_executed_neg_mwh", 1.0),
        ("BCM negative activation", "BCM activation", "charge", "bcm_charge_mw", bcm_charge_source, 1.0),
        ("DA sell", "DA", "discharge", "da_discharge_mw", "real_da_sell_mwh", -1.0),
        ("BEM positive activation", "BEM", "discharge", "bem_discharge_mw", "real_bem_only_executed_pos_mwh", -1.0),
        ("BCM positive activation", "BCM activation", "discharge", "bcm_discharge_mw", bcm_discharge_source, -1.0),
    ]
    rows: list[pd.DataFrame] = []
    for component, market, direction, value_col, source_col, sign in specs:
        part = day[["timestamp_utc", "date", "soc_mwh", "pnl_eur", "cumulative_pnl_eur", value_col]].copy()
        part["hour_utc"] = part["timestamp_utc"].dt.strftime("%H:%M")
        part["component"] = component
        part["market"] = market
        part["direction"] = direction
        part["mw_abs"] = pd.to_numeric(part[value_col], errors="coerce").fillna(0.0).abs()
        part["mw_signed"] = sign * part["mw_abs"]
        part["source_column"] = source_col
        part["pnl_source_column"] = pnl_source
        part["model"] = str(best["model"])
        part["quantile"] = str(best["quantile"])
        part["scenario_folder"] = str(best["folder"])
        part["scenario_output_dir"] = str(hourly_path.parent)
        rows.append(part[columns])
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    return out.sort_values(["timestamp_utc", "direction", "component"]).reset_index(drop=True), pd.DataFrame(warnings)


def _pinball_loss(y_true: pd.Series, y_pred: pd.Series, quantile: float) -> pd.Series:
    err = y_true - y_pred
    return np.maximum(quantile * err, (quantile - 1.0) * err)


def _model_display_to_key(model: str) -> str:
    mapping = {"RLQR": "linear", "XGB": "xgb", "TFT": "tft"}
    return mapping.get(str(model), str(model).lower())


def build_pinball_net_profit_scatter_data(
    *,
    summary: pd.DataFrame,
    forecast_benchmark_dir: Path,
    split: str,
    quantiles: list[str],
) -> pd.DataFrame:
    joined_dir = forecast_benchmark_dir / "diagnostics" / "joined_predictions"
    pnl = summary.loc[~summary["is_benchmark"].astype(bool)].copy()
    if pnl.empty or not joined_dir.exists():
        return pd.DataFrame()
    pnl["model_key"] = pnl["model"].map(_model_display_to_key)
    pnl = pnl[
        [
            "model",
            "model_key",
            "quantile",
            "realized_profit_eur",
            "annualized_profit_eur_per_year",
            "simulation_valid",
            "thesis_reportable",
            "included_in_result_section",
            "invalid_reason",
        ]
    ].copy()

    rows: list[dict[str, Any]] = []
    key_cols = ["target_time_utc", "lead_time_h"]
    for target in TARGET_SLUGS:
        frames: dict[str, pd.DataFrame] = {}
        for model_key in ["linear", "xgb", "tft"]:
            path = joined_dir / f"{model_key}__{split}__{target}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            keep = [c for c in [*key_cols, "y_true", *quantiles] if c in df.columns]
            frames[model_key] = df[keep].copy()
        if not frames:
            continue
        common_index: set[tuple[Any, ...]] | None = None
        prepared: dict[str, pd.DataFrame] = {}
        for model_key, df in frames.items():
            required = [*key_cols, "y_true"]
            if not set(required).issubset(df.columns):
                continue
            available_q = [q for q in quantiles if q in df.columns]
            if not available_q:
                continue
            d = df.dropna(subset=[*required, *available_q]).copy()
            tuples = set(map(tuple, d[key_cols].to_numpy()))
            common_index = tuples if common_index is None else common_index.intersection(tuples)
            prepared[model_key] = d
        if not prepared or not common_index:
            continue
        common_df = pd.DataFrame(list(common_index), columns=key_cols)
        for model_key, df in prepared.items():
            d = df.merge(common_df, on=key_cols, how="inner")
            for q in quantiles:
                if q not in d.columns:
                    continue
                quantile_value = float(str(q).removeprefix("p")) / 100.0
                loss = _pinball_loss(pd.to_numeric(d["y_true"], errors="coerce"), pd.to_numeric(d[q], errors="coerce"), quantile_value)
                rows.append(
                    {
                        "split": split,
                        "target": target,
                        "target_label": TARGET_LABELS[target].replace("$-$", "-"),
                        "model_key": model_key,
                        "model": MODEL_LABELS.get(model_key, model_key.upper()),
                        "quantile": q,
                        "mean_pinball_loss": float(loss.mean()),
                        "n_obs": int(loss.notna().sum()),
                    }
                )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    out = metrics.merge(pnl, on=["model", "model_key", "quantile"], how="left", suffixes=("", "_simulation"))
    return out.sort_values(["target", "model", "quantile"]).reset_index(drop=True)


def build_total_pinball_net_profit_scatter_data(scatter_data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate target-level mean pinball losses into one total diagnostic.

    This is an observation-weighted raw mean across targets. It is useful as a
    compact diagnostic but remains target-scale dependent.
    """
    columns = [
        "split",
        "target",
        "target_label",
        "model_key",
        "model",
        "quantile",
        "mean_pinball_loss",
        "n_obs",
        "n_targets",
        "realized_profit_eur",
        "annualized_profit_eur_per_year",
        "simulation_valid",
        "thesis_reportable",
        "included_in_result_section",
        "invalid_reason",
    ]
    if scatter_data.empty:
        return pd.DataFrame(columns=columns)

    d = scatter_data.copy()
    d["mean_pinball_loss"] = pd.to_numeric(d["mean_pinball_loss"], errors="coerce")
    d["n_obs"] = pd.to_numeric(d["n_obs"], errors="coerce")
    d = d.dropna(subset=["mean_pinball_loss", "n_obs"])
    d = d.loc[d["n_obs"] > 0].copy()
    if d.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    group_cols = ["split", "model_key", "model", "quantile"]
    value_cols = [
        "realized_profit_eur",
        "annualized_profit_eur_per_year",
        "simulation_valid",
        "thesis_reportable",
        "included_in_result_section",
        "invalid_reason",
    ]
    for keys, group in d.groupby(group_cols, dropna=False, observed=True):
        weights = group["n_obs"].to_numpy(dtype=float)
        losses = group["mean_pinball_loss"].to_numpy(dtype=float)
        base = dict(zip(group_cols, keys))
        first = group.iloc[0]
        rec: dict[str, Any] = {
            **base,
            "target": "all_targets",
            "target_label": "All Target Variables",
            "mean_pinball_loss": float(np.average(losses, weights=weights)),
            "n_obs": int(np.sum(weights)),
            "n_targets": int(group["target"].nunique()),
        }
        for col in value_cols:
            rec[col] = first.get(col, np.nan)
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out[columns].sort_values(["model", "quantile"]).reset_index(drop=True)


def _format_eur(value: Any) -> str:
    val = _safe_float(value)
    if not math.isfinite(val):
        return r"--"
    sign = "-" if val < 0 else ""
    return f"{sign}{abs(val):,.0f} EUR"


def _format_pct(value: Any) -> str:
    val = _safe_float(value)
    if not math.isfinite(val):
        return r"--"
    return f"{val:,.1f}\\%"


def write_primary_table(path: Path, table: pd.DataFrame, simulation_days: float) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{Annualized PnL table. Profit is annualized from the actual simulation duration of {simulation_days:.2f} days.}}",
        r"\label{tab:1_net_profit_by_model_and_quantile}",
        r"\begin{tabular}{@{}lrrrrrlll@{}}",
        r"\toprule",
        r"\textbf{Quantile} & \textbf{Naive} & \textbf{RHPF} & \textbf{RLQR} & \textbf{XGB} & \textbf{TFT} & \textbf{Best model} & \textbf{\begin{tabular}[c]{@{}l@{}}Best vs\\Naive (\%)\end{tabular}} & \textbf{\begin{tabular}[c]{@{}l@{}}Best vs\\RHPF (\%)\end{tabular}} \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        cells = [
            _latex_escape(row["quantile"]),
            _format_eur(row["Naive"]),
            _format_eur(row["RHPF"]),
            _format_eur(row["RLQR"]),
            _format_eur(row["XGB"]),
            _format_eur(row["TFT"]),
            _latex_escape(row.get("best_model", "")) or r"--",
            _format_pct(row.get("best_vs_naive_pct", np.nan)),
            _format_pct(row.get("best_vs_rhpf_pct", np.nan)),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_appendix_table(path: Path, summary: pd.DataFrame) -> None:
    cols = ["folder", "model", "quantile", "realized_profit_eur", "annualized_profit_eur_per_year", "simulation_valid", "thesis_reportable", "invalid_reason"]
    data = summary[cols].copy()
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{RQ2 profit and validity diagnostics by simulation folder.}",
        r"\label{tab:rq2_profit_and_validity_detailed}",
        r"\begin{tabular}{@{}lllrrr p{0.30\linewidth}@{}}",
        r"\toprule",
        r"\textbf{Scenario} & \textbf{Model} & \textbf{Quantile} & \textbf{Profit} & \textbf{Annualized profit} & \textbf{Valid} & \textbf{Invalid reason} \\",
        r"\midrule",
    ]
    for _, row in data.iterrows():
        reason = str(row.get("invalid_reason", "") or "")
        if reason == "none":
            reason = ""
        cells = [
            _latex_escape(row["folder"]),
            _latex_escape(row["model"]),
            _latex_escape(row["quantile"]),
            _format_eur(row["realized_profit_eur"]),
            _format_eur(row["annualized_profit_eur_per_year"]),
            "yes" if _safe_float(row["simulation_valid"]) >= 0.5 and _safe_float(row["thesis_reportable"]) >= 0.5 else "no",
            _latex_escape(reason),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_figure(fig: Any, path_base: Path, formats: list[str]) -> list[Path]:
    written: list[Path] = []
    for fmt in formats:
        out = path_base.with_suffix("." + fmt)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        written.append(out)
    return written


def plot_profit_by_quantile(table: pd.DataFrame, bench: pd.DataFrame, out_base: Path, formats: list[str], run_name: str) -> list[Path]:
    import matplotlib.pyplot as plt

    apply_geo_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(table))
    width = 0.24
    offsets = {"RLQR": -width, "XGB": 0.0, "TFT": width}
    for model in MODEL_ORDER:
        values = pd.to_numeric(table[model], errors="coerce").to_numpy(dtype=float)
        ax.bar(x + offsets[model], values, width=width, label=model, color=get_model_color("linear" if model == "RLQR" else model.lower()), edgecolor="none")
    for benchmark in BENCHMARK_ORDER:
        match = bench.loc[bench["model"] == benchmark]
        if match.empty:
            continue
        value = float(match["annualized_profit_eur_per_year"].iloc[0])
        color = THESIS_PALETTE["naive"] if benchmark == "Naive" else THESIS_PALETTE["perfect_foresight"]
        ax.axhline(value, color=color, linestyle="--", linewidth=1.8, label=f"{benchmark} benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels(table["quantile"].astype(str).tolist())
    ax.set_ylabel("Annualized Net Profit")
    ax.set_xlabel("Quantile policy")
    ax.set_title("Net Profit by Model and Quantile")
    ax.text(0.0, 1.02, f"Multi-market strategy, annualized from {run_name}. Higher is better.", transform=ax.transAxes, fontsize=9)
    ax.legend(ncol=3, loc="best")
    fig.tight_layout()
    written = _save_figure(fig, out_base, formats)
    plt.close(fig)
    return written


def plot_heatmap(table: pd.DataFrame, out_base: Path, formats: list[str]) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import StrMethodFormatter

    apply_geo_style()
    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    rows = ["Naive", "RHPF", *MODEL_ORDER]
    quantiles = table["quantile"].astype(str).tolist()
    pivot = table.set_index("quantile").reindex(quantiles)[rows].T
    pivot_k = pivot / 1000.0
    arr_k = pivot_k.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(arr_k)
    cmap = LinearSegmentedColormap.from_list(
        "geo_sequential_blue",
        [GEO_SEQUENTIAL_BLUE[f"seq_{i}"] for i in range(1, 8)],
    )
    cmap.set_bad("#F2F2F2")
    im = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.grid(False)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks(np.arange(len(quantiles)))
    ax.set_xticklabels(quantiles)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Quantile policy")
    ax.set_ylabel("Strategy")
    ax.set_title("Net Profit")
    for i, model in enumerate(rows):
        for j, q in enumerate(quantiles):
            val = pivot.loc[model, q]
            txt = "n/a" if not math.isfinite(_safe_float(val)) else f"{val/1000:,.0f} kEUR"
            text_color = "white" if model == "RHPF" else THESIS_PALETTE["neutral_dark"]
            ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.grid(False)
    cbar.ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    cbar.set_label("Annualized Net Profit (kEUR/year)")
    fig.tight_layout()
    written = _save_figure(fig, out_base, formats)
    plt.close(fig)
    return written


def plot_net_profit_lines(sweep_data: pd.DataFrame, out_base: Path, formats: list[str], run_name: str, quantiles: list[str]) -> list[Path]:
    import matplotlib.pyplot as plt

    apply_geo_style()
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    x = np.arange(len(quantiles))
    x_by_quantile = {q: i for i, q in enumerate(quantiles)}
    marker_by_model = {"Naive": "D", "RHPF": "X", "RLQR": "o", "XGB": "s", "TFT": "^"}
    color_by_model = {
        "Naive": THESIS_PALETTE["naive"],
        "RHPF": TRUTH_REFERENCE_COLOR,
        "RLQR": get_model_color("linear"),
        "XGB": get_model_color("xgb"),
        "TFT": get_model_color("tft"),
    }
    linestyle_by_model = {"RLQR": "-", "XGB": "-", "TFT": "-"}

    def _label_offset(model: str, quantile: str) -> int:
        if model == "RLQR" and quantile == "p50":
            return -16
        if model == "XGB" and quantile == "p90":
            return -16
        if model == "Naive":
            return -14
        if model == "RHPF":
            return 8
        return 8

    for model in ["Naive", "RHPF", *MODEL_ORDER]:
        g = sweep_data.loc[sweep_data["series"].astype(str).eq(model)].copy()
        if g.empty:
            continue
        if model in BENCHMARK_ORDER:
            q = "p50" if "p50" in x_by_quantile else quantiles[len(quantiles) // 2]
            match = g.loc[g["quantile"].astype(str).eq(q)]
            if match.empty:
                match = g.head(1)
            value = _safe_float(match["annualized_net_profit_eur_per_year"].iloc[0])
            if not math.isfinite(value):
                continue
            ax.plot(
                x,
                np.full(len(x), value, dtype=float),
                linewidth=2.0,
                linestyle="-",
                label=f"{model} benchmark",
                color=color_by_model[model],
                zorder=2,
            )
            plot_x = np.asarray([x_by_quantile.get(q, len(quantiles) // 2)], dtype=float)
            values_arr = np.asarray([value], dtype=float)
            plot_quantiles = [q]
        else:
            values = []
            for q in quantiles:
                match = g.loc[g["quantile"].astype(str).eq(q)]
                values.append(_safe_float(match["annualized_net_profit_eur_per_year"].iloc[0]) if not match.empty else np.nan)
            values_arr = np.asarray(values, dtype=float)
            plot_x = x
            plot_quantiles = quantiles
            ax.plot(
                plot_x,
                values_arr,
                marker=marker_by_model[model],
                linewidth=2.0,
                linestyle=linestyle_by_model[model],
                label=model,
                color=color_by_model[model],
                zorder=3,
            )
        for xi, yi, q in zip(plot_x, values_arr, plot_quantiles):
            if not math.isfinite(float(yi)):
                continue
            ax.annotate(
                f"{yi/1000:,.0f} kEUR",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, _label_offset(model, q)),
                ha="center",
                fontsize=8,
                color=THESIS_PALETTE["neutral_dark"],
            )
    ax.set_xticks(x)
    ax.set_xticklabels(quantiles)
    ax.set_ylabel("Annualized Net Profit (EUR/year)")
    ax.set_xlabel("Quantile policy")
    ax.set_title("Quantile Sweep: Net Profit by Model", pad=14)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=True)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    written = _save_figure(fig, out_base, formats)
    plt.close(fig)
    return written


def plot_best_quantile_components(component_data: pd.DataFrame, out_base: Path, formats: list[str], run_name: str) -> list[Path]:
    import matplotlib.pyplot as plt

    apply_geo_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    if component_data.empty:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No model scenarios available for component decomposition.",
            ha="center",
            va="center",
            fontsize=11,
            color=THESIS_PALETTE["neutral_dark"],
            wrap=True,
        )
        ax.set_title("RQ2 Revenue and Cost Components at Best Quantile")
        fig.tight_layout()
        written = _save_figure(fig, out_base, formats)
        plt.close(fig)
        return written

    models = [m for m in MODEL_ORDER if m in set(component_data["model"])]
    x = np.arange(len(models))
    width = 0.62
    component_order = [label for _col, label in COMPONENT_COLUMNS if label in set(component_data["component"])]
    pivot = component_data.pivot_table(
        index="model",
        columns="component",
        values="annualized_component_value_eur_per_year",
        aggfunc="sum",
    ).reindex(index=models, columns=component_order).fillna(0.0)
    pos_bottom = np.zeros(len(models))
    neg_bottom = np.zeros(len(models))
    for component in component_order:
        values = pivot[component].to_numpy(dtype=float)
        bottoms = np.where(values >= 0, pos_bottom, neg_bottom)
        ax.bar(
            x,
            values,
            width=width,
            bottom=bottoms,
            label=component,
            color=MARKET_COMPONENT_COLORS.get(component, THESIS_PALETTE["neutral_dark"]),
            edgecolor="white",
            linewidth=0.6,
        )
        pos_bottom = np.where(values >= 0, pos_bottom + values, pos_bottom)
        neg_bottom = np.where(values < 0, neg_bottom + values, neg_bottom)
    best_quantiles = (
        component_data[["model", "quantile"]]
        .drop_duplicates("model")
        .set_index("model")
        .reindex(models)["quantile"]
        .fillna("")
        .astype(str)
        .tolist()
    )
    ax.axhline(0, color=THESIS_PALETTE["neutral_dark"], linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{q}" if q else m for m, q in zip(models, best_quantiles)])
    ax.set_ylabel("Annualized component value (EUR/year)")
    ax.set_xlabel("Model and best quantile")
    ax.set_title("RQ2 Revenue and Cost Components at Best Quantile", pad=12)
    handles, labels = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    revenue_labels = [label for label in component_order if label not in {display for _col, display in COMPONENT_COLUMNS if _col in COMPONENT_COST_COLUMNS}]
    cost_labels = [label for _col, label in COMPONENT_COLUMNS if _col in COMPONENT_COST_COLUMNS and label in handle_by_label]
    from matplotlib.patches import Patch

    spacer_count = max(0, len(cost_labels) - len(revenue_labels))
    legend_labels = revenue_labels + [" "] * spacer_count + cost_labels
    legend_handles = [handle_by_label[label] if label in handle_by_label else Patch(facecolor="none", edgecolor="none") for label in legend_labels]
    ax.legend(legend_handles, legend_labels, ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    written = _save_figure(fig, out_base, formats)
    plt.close(fig)
    return written


def plot_market_dispatch_soc_day(dispatch_data: pd.DataFrame, out_base: Path, formats: list[str]) -> list[Path]:
    import matplotlib.pyplot as plt

    apply_geo_style()
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    if dispatch_data.empty:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No market dispatch/SOC example-day data available.",
            ha="center",
            va="center",
            fontsize=11,
            color=THESIS_PALETTE["neutral_dark"],
            wrap=True,
        )
        ax.set_title("Market Dispatch and SoC on Selected Day")
        fig.tight_layout()
        written = _save_figure(fig, out_base, formats)
        plt.close(fig)
        return written

    data = dispatch_data.copy()
    data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], errors="coerce", utc=True)
    times = sorted(data["timestamp_utc"].dropna().unique())
    x = np.arange(len(times))
    time_index = {ts: i for i, ts in enumerate(times)}
    data["x"] = data["timestamp_utc"].map(time_index)

    component_order = [
        "DA buy",
        "BEM negative activation",
        "BCM negative activation",
        "DA sell",
        "BEM positive activation",
        "BCM positive activation",
    ]
    label_map = {
        "DA buy": "DA buy/sell",
        "DA sell": "DA buy/sell",
        "BEM negative activation": "BEM activation +/-",
        "BEM positive activation": "BEM activation +/-",
        "BCM negative activation": "BCM activation +/-",
        "BCM positive activation": "BCM activation +/-",
    }
    color_map = {
        "DA buy": MARKET_COLOR_MAP["DA"],
        "DA sell": MARKET_COLOR_MAP["DA"],
        "BEM negative activation": MARKET_COLOR_MAP["BEM"],
        "BEM positive activation": MARKET_COLOR_MAP["BEM"],
        "BCM negative activation": MARKET_COLOR_MAP["BCM activation"],
        "BCM positive activation": MARKET_COLOR_MAP["BCM activation"],
    }
    pos_bottom = np.zeros(len(times))
    neg_bottom = np.zeros(len(times))
    width = 0.82
    for component in component_order:
        g = data.loc[data["component"].eq(component)].copy()
        if g.empty:
            continue
        values = np.zeros(len(times))
        for _, row in g.iterrows():
            idx = int(row["x"])
            values[idx] = float(row["mw_signed"])
        bottom = pos_bottom if np.nanmean(values) >= 0 else neg_bottom
        label = label_map[component]
        ax.bar(
            x,
            values,
            width=width,
            bottom=bottom,
            color=color_map[component],
            edgecolor="none",
            linewidth=0.0,
            label=None,
        )
        if np.nanmean(values) >= 0:
            pos_bottom = pos_bottom + values
        else:
            neg_bottom = neg_bottom + values

    soc = (
        data[["timestamp_utc", "soc_mwh"]]
        .drop_duplicates("timestamp_utc")
        .sort_values("timestamp_utc")
        .set_index("timestamp_utc")
        .reindex(times)["soc_mwh"]
        .to_numpy(dtype=float)
    )
    ax2 = ax.twinx()
    ax2.plot(x, soc, color=THESIS_PALETTE["perfect_foresight"], linewidth=2.2, label="SoC")
    ax2.set_ylabel("SoC (MWh)")
    ax2.tick_params(axis="y", colors=THESIS_PALETTE["perfect_foresight"])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(THESIS_PALETTE["perfect_foresight"])
    ax2.spines["right"].set_linewidth(1.0)
    pnl_keur = (
        data[["timestamp_utc", "cumulative_pnl_eur"]]
        .drop_duplicates("timestamp_utc")
        .sort_values("timestamp_utc")
        .set_index("timestamp_utc")
        .reindex(times)["cumulative_pnl_eur"]
        .to_numpy(dtype=float)
        / 1000.0
    )
    pnl_color = GEO_SEQUENTIAL_BLUE["seq_7"]
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.08))
    ax3.plot(x, pnl_keur, color=pnl_color, linewidth=2.0, linestyle="--", label="Cumulative Net Profit")
    ax3.set_ylabel("Cumulative Net Profit (kEUR)")
    ax3.tick_params(axis="y", colors=pnl_color)
    ax3.spines["right"].set_visible(True)
    ax3.spines["right"].set_color(pnl_color)
    ax3.spines["right"].set_linewidth(1.0)

    ax.axhline(0, color=THESIS_PALETTE["neutral_dark"], linewidth=0.9)
    ax.grid(False)
    ax2.grid(False)
    ax3.grid(False)
    ax.set_axisbelow(True)
    ax.set_ylim(-10, 10)
    if len(x) > 0:
        ax.set_xlim(0, len(x) - 1)
    ax.set_ylabel("Power (MW): charge (+), discharge (-)")
    selected_date = pd.Timestamp(times[0]).date() if times else None
    selected_date_label = f"{selected_date.day} {MONTH_ABBR_FULL[selected_date.month]} {selected_date.year}" if selected_date is not None else ""
    ax.set_xlabel(f"Time ({selected_date_label})" if selected_date_label else "Time")
    labels = [pd.Timestamp(ts).strftime("%H:%M") for ts in times]
    ax.set_xticks(x[:: max(1, len(x) // 8)])
    ax.set_xticklabels(labels[:: max(1, len(x) // 8)])
    fig.suptitle("Exemplary Trading Day with TFT p90 and a Multi-Market Strategy", y=0.98)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_components = ["DA buy", "BEM negative activation", "BCM negative activation"]
    blank = Patch(facecolor="none", edgecolor="none", label=" ")
    handles = [
        Patch(facecolor=color_map["DA buy"], edgecolor="none", label=label_map["DA buy"]),
        blank,
        Patch(facecolor=color_map["BEM negative activation"], edgecolor="none", label=label_map["BEM negative activation"]),
        Patch(facecolor=color_map["BCM negative activation"], edgecolor="none", label=label_map["BCM negative activation"]),
        blank,
        blank,
        Line2D([0], [0], color=THESIS_PALETTE["perfect_foresight"], linewidth=2.2, label="SoC"),
        Line2D([0], [0], color=pnl_color, linewidth=2.0, linestyle="--", label="Cumulative Net Profit"),
    ]
    ax.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=True)
    fig.tight_layout(rect=(0, 0, 0.94, 0.94))
    written = _save_figure(fig, out_base, formats)
    plt.close(fig)
    return written


def plot_cumulative_pnl(cumulative: pd.DataFrame, out_base: Path, formats: list[str], run_name: str) -> list[Path]:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    apply_geo_style()
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    if cumulative.empty:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No thesis-reportable cumulative Net Profit paths available.",
            ha="center",
            va="center",
            fontsize=11,
            color=THESIS_PALETTE["neutral_dark"],
            wrap=True,
        )
        ax.set_title("Cumulative Net Profit: Model Comparison Over Test Period")
        fig.tight_layout()
        written = _save_figure(fig, out_base, formats)
        plt.close(fig)
        return written

    color_map = {
        "Naive": THESIS_PALETTE["naive"],
        "RHPF": TRUTH_REFERENCE_COLOR,
        "RLQR": get_model_color("linear"),
        "XGB": get_model_color("xgb"),
        "TFT": get_model_color("tft"),
    }
    line_style = {"Naive": "-", "RHPF": "-", "RLQR": "-", "XGB": "-", "TFT": "-"}
    order = ["Naive", "RHPF", "RLQR", "XGB", "TFT"]
    cumulative = cumulative.copy()
    cumulative["timestamp_utc"] = pd.to_datetime(cumulative["timestamp_utc"], errors="coerce", utc=True)
    for model in order:
        group = cumulative.loc[cumulative["model"].eq(model)].sort_values("timestamp_utc")
        if group.empty:
            continue
        label = group["series"].iloc[0]
        ax.plot(
            group["timestamp_utc"],
            pd.to_numeric(group["cum_pnl_eur"], errors="coerce") / 1000.0,
            label=label,
            color=color_map.get(model, THESIS_PALETTE["neutral_dark"]),
            linestyle=line_style.get(model, "-"),
            linewidth=2.2 if model not in BENCHMARK_ORDER else 1.9,
        )
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_ylabel("Cumulative Net Profit (kEUR)")
    ax.set_xlabel("Time (2025)")
    ax.set_title("Cumulative Net Profit: Model Comparison Over Test Period")
    ax.legend(ncol=3, loc="best")
    fig.tight_layout()
    written = _save_figure(fig, out_base, formats)
    plt.close(fig)
    return written


def plot_pinball_net_profit_scatter(scatter_data: pd.DataFrame, figures_dir: Path, formats: list[str]) -> list[Path]:
    import matplotlib.pyplot as plt

    apply_geo_style()
    written: list[Path] = []
    color_map = {"RLQR": get_model_color("linear"), "XGB": get_model_color("xgb"), "TFT": get_model_color("tft")}
    marker_map = {"RLQR": "o", "XGB": "s", "TFT": "^"}
    for target, label in TARGET_LABELS.items():
        slug = TARGET_SLUGS[target]
        df = scatter_data.loc[scatter_data.get("target", pd.Series(dtype=str)).astype(str).eq(target)].copy() if not scatter_data.empty else pd.DataFrame()
        fig, ax = plt.subplots(figsize=(8.2, 5.6))
        if df.empty:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"No pinball-loss / Net Profit data available for {label.replace('$-$', '-')}.",
                ha="center",
                va="center",
                fontsize=11,
                color=THESIS_PALETTE["neutral_dark"],
                wrap=True,
            )
        else:
            for model in MODEL_ORDER:
                g = df.loc[df["model"].eq(model)].copy()
                if g.empty:
                    continue
                y_k = pd.to_numeric(g["annualized_profit_eur_per_year"], errors="coerce") / 1000.0
                ax.scatter(
                    g["mean_pinball_loss"],
                    y_k,
                    label=model,
                    color=color_map[model],
                    marker=marker_map[model],
                    s=58,
                    edgecolor="black",
                    linewidth=0.45,
                    alpha=0.9,
                )
                for _, row in g.iterrows():
                    ax.annotate(
                        str(row["quantile"]),
                        (row["mean_pinball_loss"], _safe_float(row["annualized_profit_eur_per_year"]) / 1000.0),
                        textcoords="offset points",
                        xytext=(4, 3),
                        fontsize=8,
                        color=THESIS_PALETTE["neutral_dark"],
                    )
            ax.set_xlabel("Mean pinball loss")
            ax.set_ylabel("Annualized Net Profit (kEUR / year)")
            ax.legend(title="Model", loc="best")
        ax.set_title(f"RQ2 Pinball Loss vs Net Profit: {label}")
        fig.tight_layout()
        written.extend(_save_figure(fig, figures_dir / f"5_pinball_loss_vs_net_profit_{slug}", formats))
        plt.close(fig)
    return written


def build_normalized_total_pinball_profit_data(total_scatter_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize total mean pinball loss and profit to [0, 1] over observed model-quantile rows."""
    if total_scatter_data.empty:
        return pd.DataFrame()
    df = total_scatter_data.copy()
    df["mean_pinball_loss"] = pd.to_numeric(df["mean_pinball_loss"], errors="coerce")
    df["annualized_profit_eur_per_year"] = pd.to_numeric(df["annualized_profit_eur_per_year"], errors="coerce")
    df = df.dropna(subset=["mean_pinball_loss", "annualized_profit_eur_per_year"]).copy()
    if df.empty:
        return pd.DataFrame()

    loss_min = float(df["mean_pinball_loss"].min())
    loss_max = float(df["mean_pinball_loss"].max())
    profit_min = float(df["annualized_profit_eur_per_year"].min())
    profit_max = float(df["annualized_profit_eur_per_year"].max())
    loss_range = loss_max - loss_min
    profit_range = profit_max - profit_min
    df["normalized_forecast_loss"] = 0.0 if math.isclose(loss_range, 0.0) else (df["mean_pinball_loss"] - loss_min) / loss_range
    df["normalized_annualized_net_profit"] = 0.0 if math.isclose(profit_range, 0.0) else (df["annualized_profit_eur_per_year"] - profit_min) / profit_range
    df["loss_min_observed"] = loss_min
    df["loss_max_observed"] = loss_max
    df["profit_min_observed_eur_per_year"] = profit_min
    df["profit_max_observed_eur_per_year"] = profit_max
    return df


def plot_normalized_total_pinball_profit_scatter(normalized_data: pd.DataFrame, out_base: Path, formats: list[str]) -> list[Path]:
    import matplotlib.pyplot as plt

    apply_geo_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    color_map = {"RLQR": get_model_color("linear"), "XGB": get_model_color("xgb"), "TFT": get_model_color("tft")}
    marker_map = {"RLQR": "o", "XGB": "s", "TFT": "^"}
    if normalized_data.empty:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "No normalized total pinball-loss / Net Profit data available.",
                ha="center",
                va="center",
                fontsize=11,
            color=THESIS_PALETTE["neutral_dark"],
            wrap=True,
        )
    else:
        ax.plot(
            [0.0, 1.0],
            [1.0, 0.0],
            color=THESIS_PALETTE["naive"],
            linestyle="--",
            linewidth=1.3,
            alpha=0.8,
            label="VoF reference line",
            zorder=1,
        )
        for model in MODEL_ORDER:
            g = normalized_data.loc[normalized_data["model"].eq(model)].copy()
            if g.empty:
                continue
            ax.scatter(
                g["normalized_forecast_loss"],
                g["normalized_annualized_net_profit"],
                label=model,
                color=color_map[model],
                marker=marker_map[model],
                s=62,
                edgecolor="black",
                linewidth=0.45,
                alpha=0.92,
                zorder=3,
            )
            for _, row in g.iterrows():
                ax.annotate(
                    str(row["quantile"]),
                    (row["normalized_forecast_loss"], row["normalized_annualized_net_profit"]),
                    textcoords="offset points",
                    xytext=(4, 3),
                    fontsize=8,
                    color=THESIS_PALETTE["neutral_dark"],
                )
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("Normalized total mean pinball loss")
        ax.set_ylabel("Normalized annualized net profit")
        ax.legend(title="Series", loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=8, frameon=True)
    ax.set_title("Normalized Total Forecast Loss vs Net Profit")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    written = _save_figure(fig, out_base, formats)
    plt.close(fig)
    return written


def plot_total_pinball_net_profit_scatter(scatter_data: pd.DataFrame, out_base: Path, formats: list[str]) -> list[Path]:
    import matplotlib.pyplot as plt

    apply_geo_style()
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    color_map = {"RLQR": get_model_color("linear"), "XGB": get_model_color("xgb"), "TFT": get_model_color("tft")}
    marker_map = {"RLQR": "o", "XGB": "s", "TFT": "^"}
    if scatter_data.empty:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No total pinball-loss / Net Profit data available.",
            ha="center",
            va="center",
            fontsize=11,
            color=THESIS_PALETTE["neutral_dark"],
            wrap=True,
        )
    else:
        for model in MODEL_ORDER:
            g = scatter_data.loc[scatter_data["model"].eq(model)].copy()
            if g.empty:
                continue
            y_k = pd.to_numeric(g["annualized_profit_eur_per_year"], errors="coerce") / 1000.0
            ax.scatter(
                g["mean_pinball_loss"],
                y_k,
                label=model,
                color=color_map[model],
                marker=marker_map[model],
                s=62,
                edgecolor="black",
                linewidth=0.45,
                alpha=0.9,
            )
            for _, row in g.iterrows():
                ax.annotate(
                    str(row["quantile"]),
                    (row["mean_pinball_loss"], _safe_float(row["annualized_profit_eur_per_year"]) / 1000.0),
                    textcoords="offset points",
                    xytext=(4, 3),
                    fontsize=8,
                    color=THESIS_PALETTE["neutral_dark"],
                )
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.plot(
            [xmax, xmin],
            [ymin, ymax],
            color=THESIS_PALETTE["naive"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            zorder=1,
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Total mean pinball loss")
        ax.set_ylabel("Annualized Net Profit (kEUR / year)")
        ax.legend(title="Model", loc="best")
    ax.set_title("Total Mean Pinball Loss vs Net Profit")
    ax.text(
        0.0,
        1.02,
        "Observation-weighted raw mean pinball loss across all target variables. Lower forecast loss and higher Net Profit are preferable.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    written = _save_figure(fig, out_base, formats)
    plt.close(fig)
    return written


def _write_latex_include(path: Path, figure_rel: str, caption: str, label: str) -> None:
    path.write_text(
        "\n".join(
            [
                r"\begin{figure}[htbp]",
                r"\centering",
                rf"\includegraphics[width=\linewidth]{{{figure_rel}}}",
                rf"\caption{{{_latex_escape(caption)}}}",
                rf"\label{{{label}}}",
                r"\end{figure}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _thesis_figure_rel(out_root: Path, figure_name: str) -> str:
    """Return thesis-project include path for a generated result-section figure."""
    return f"figures/4-results/{out_root.name}/result_section/figures/{figure_name}"


def _thesis_appendix_figure_rel(out_root: Path, figure_name: str) -> str:
    """Return thesis-project include path for a generated appendix figure."""
    return f"figures/4-results/{out_root.name}/appendix/figures/{figure_name}"


def _manifest_entry(path: Path, root: Path, tier: str, artifact_type: str, metric_family: str, thesis_use: str, run_root: Path, created: str, days: float, factor: float) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "tier": tier,
        "artifact_type": artifact_type,
        "metric_family": metric_family,
        "thesis_use": thesis_use,
        "source_run_root": str(run_root),
        "created_at_utc": created,
        "simulation_days": days,
        "annualization_factor": factor,
    }


def build_outputs(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root)
    out_root = Path(args.out_root)
    forecast_benchmark_dir = Path(args.forecast_benchmark_dir)
    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)
    for rel in ALL_OUTPUT_DIRS:
        (out_root / rel).mkdir(parents=True, exist_ok=True)
    formats = [f.strip().lower() for f in str(args.formats).split(",") if f.strip()]
    if args.no_figures:
        formats = []

    models = _parse_csv_arg(args.models, DEFAULT_MODELS)
    quantiles = _parse_csv_arg(args.quantiles, DEFAULT_QUANTILES)
    summary, validity, inventory, warning_df, _, _, days, duration_source = collect_summaries(
        run_root=run_root,
        models=models,
        quantiles=quantiles,
        split=str(args.split),
        simulation_days=args.simulation_days,
        strict_validity=bool(args.strict_validity),
    )
    if not args.annualize:
        summary["annualized_profit_eur_per_year"] = summary["realized_profit_eur"]
        summary["annualization_factor"] = 1.0
    table, heatmap, bench = build_result_tables(summary, quantiles)
    heatmap_table = build_profit_heatmap_data(summary, quantiles)
    sweep_data = build_quantile_sweep_data(summary, quantiles)
    component_data = build_best_quantile_components(summary)
    cumulative_data = build_cumulative_pnl_paths(summary, run_root=run_root)
    scatter_data = build_pinball_net_profit_scatter_data(
        summary=summary,
        forecast_benchmark_dir=forecast_benchmark_dir,
        split=str(args.split),
        quantiles=quantiles,
    )
    total_scatter_data = build_total_pinball_net_profit_scatter_data(scatter_data)
    normalized_total_scatter_data = build_normalized_total_pinball_profit_data(total_scatter_data)
    dispatch_soc_data, dispatch_soc_warnings = build_market_dispatch_soc_day(summary, run_root=run_root)

    csv_dir = out_root / "backup/csv"
    diag_dir = out_root / "backup/diagnostics"
    warn_dir = out_root / "backup/warnings"
    figures_dir = out_root / "result_section/figures"
    tables_dir = out_root / "result_section/tables"
    latex_figures_dir = out_root / "result_section/latex_figures"
    appendix_figures_dir = out_root / "appendix/figures"
    appendix_tables_dir = out_root / "appendix/tables"
    appendix_latex_figures_dir = out_root / "appendix/latex_figures"

    summary.to_csv(csv_dir / "rq2_scenario_summary_long.csv", index=False)
    table.to_csv(csv_dir / "1_net_profit_by_model_and_quantile.csv", index=False)
    heatmap_table.to_csv(csv_dir / "1_profit_heatmap.csv", index=False)
    sweep_data.to_csv(csv_dir / "2_quantile_sweep_net_profit_by_model.csv", index=False)
    component_data.to_csv(csv_dir / "3_revenue_cost_components_best_quantile.csv", index=False)
    cumulative_data.to_csv(csv_dir / "4_cumulative_net_profit_model_comparison_test_period.csv", index=False)
    scatter_data.to_csv(csv_dir / "5_pinball_loss_vs_net_profit_scatter_data.csv", index=False)
    total_scatter_data.to_csv(csv_dir / "5_pinball_loss_vs_net_profit_total_scatter_data.csv", index=False)
    normalized_total_scatter_data.to_csv(csv_dir / "5_pinball_loss_vs_net_profit_total_normalized.csv", index=False)
    dispatch_soc_data.to_csv(csv_dir / "6_market_dispatch_soc_selected_day.csv", index=False)
    bench.to_csv(csv_dir / "rq2_benchmark_values.csv", index=False)
    inventory.to_csv(diag_dir / "rq2_input_file_inventory.csv", index=False)
    validity.to_csv(diag_dir / "rq2_validity_diagnostics.csv", index=False)
    warning_df = pd.concat([warning_df, dispatch_soc_warnings], ignore_index=True, sort=False)
    warning_df.to_csv(warn_dir / "rq2_warnings.csv", index=False)
    dispatch_soc_warnings.to_csv(warn_dir / "6_market_dispatch_soc_selected_day_warnings.csv", index=False)

    write_primary_table(tables_dir / "1_net_profit_by_model_and_quantile.tex", table, days)
    write_appendix_table(appendix_tables_dir / "rq2_profit_and_validity_detailed.tex", summary)

    result_figure_paths: list[Path] = []
    appendix_figure_paths: list[Path] = []
    if formats:
        result_figure_paths += plot_net_profit_lines(sweep_data, figures_dir / "2_quantile_sweep_net_profit_by_model", formats, run_root.name, quantiles)
        result_figure_paths += plot_best_quantile_components(component_data, figures_dir / "3_revenue_cost_components_best_quantile", formats, run_root.name)
        result_figure_paths += plot_cumulative_pnl(cumulative_data, figures_dir / "4_cumulative_net_profit_model_comparison_test_period", formats, run_root.name)
        appendix_figure_paths += plot_pinball_net_profit_scatter(scatter_data, appendix_figures_dir, formats)
        result_figure_paths += plot_normalized_total_pinball_profit_scatter(normalized_total_scatter_data, figures_dir / "5_pinball_loss_vs_net_profit_total_normalized", formats)
        appendix_figure_paths += plot_total_pinball_net_profit_scatter(total_scatter_data, appendix_figures_dir / "5_pinball_loss_vs_net_profit_total", formats)
        result_figure_paths += plot_market_dispatch_soc_day(dispatch_soc_data, figures_dir / "6_market_dispatch_soc_selected_day", formats)
        result_figure_paths += plot_heatmap(heatmap_table, figures_dir / "1_profit_heatmap", formats)
        _write_latex_include(
            latex_figures_dir / "2_quantile_sweep_net_profit_by_model.tex",
            _thesis_figure_rel(out_root, "2_quantile_sweep_net_profit_by_model.png"),
            "Quantile sweep of annualized Net Profit by model and benchmark. The plot shows all numeric simulation rows; validity flags are reported in the source CSV.",
            "fig:2_quantile_sweep_net_profit_by_model",
        )
        _write_latex_include(
            latex_figures_dir / "3_revenue_cost_components_best_quantile.tex",
            _thesis_figure_rel(out_root, "3_revenue_cost_components_best_quantile.png"),
            "Revenue and cost components for each model at its best numeric quantile policy. Values are annualized from the simulation duration; costs are plotted below zero. Validity flags are reported in the source CSV.",
            "fig:3_revenue_cost_components_best_quantile",
        )
        _write_latex_include(
            latex_figures_dir / "4_cumulative_net_profit_model_comparison_test_period.tex",
            _thesis_figure_rel(out_root, "4_cumulative_net_profit_model_comparison_test_period.png"),
            "Cumulative Net Profit over the test period for Naive, RHPF and each model at its best numeric quantile policy. If a model path exceeds RHPF, interpret this as an accounting and timing diagnostic: rising energy prices may create value that was not fully valued at the decision time, and RHPF is a same-rules diagnostic rather than a global oracle. Validity flags are reported in the source CSV.",
            "fig:4_cumulative_net_profit_model_comparison_test_period",
        )
        _write_latex_include(
            latex_figures_dir / "5_pinball_loss_vs_net_profit_total_normalized.tex",
            _thesis_figure_rel(out_root, "5_pinball_loss_vs_net_profit_total_normalized.png"),
            "Normalized total mean pinball loss and annualized Net Profit by model and quantile policy. The grey dashed line is the VoF reference line.",
            "fig:5_pinball_loss_vs_net_profit_total_normalized",
        )
        for target, label in TARGET_LABELS.items():
            slug = TARGET_SLUGS[target]
            caption = f"Mean pinball loss and annualized Net Profit for {label.replace('$-$', '-')} by model and quantile policy."
            tex_path = appendix_latex_figures_dir / f"5_pinball_loss_vs_net_profit_{slug}.tex"
            figure_rel = _thesis_appendix_figure_rel(out_root, f"5_pinball_loss_vs_net_profit_{slug}.png")
            _write_latex_include(
                tex_path,
                figure_rel,
                caption,
                f"fig:5_pinball_loss_vs_net_profit_{slug}",
            )
        _write_latex_include(
            appendix_latex_figures_dir / "5_pinball_loss_vs_net_profit_total.tex",
            _thesis_appendix_figure_rel(out_root, "5_pinball_loss_vs_net_profit_total.png"),
            "Total mean pinball loss and annualized Net Profit by model and quantile policy. The forecast metric is the observation-weighted raw mean pinball loss across all target variables.",
            "fig:5_pinball_loss_vs_net_profit_total",
        )
        _write_latex_include(
            latex_figures_dir / "6_market_dispatch_soc_selected_day.tex",
            _thesis_figure_rel(out_root, "6_market_dispatch_soc_selected_day.png"),
            "Market dispatch, state of charge and cumulative Net Profit for the selected TFT p90 example day. Charging actions are stacked above zero, discharging actions below zero, the green line shows SoC, and the blue dashed line shows cumulative realized Net Profit in kEUR.",
            "fig:6_market_dispatch_soc_selected_day",
        )
        _write_latex_include(
            latex_figures_dir / "1_profit_heatmap.tex",
            _thesis_figure_rel(out_root, "1_profit_heatmap.png"),
            "Net Profit heatmap by strategy and quantile policy. Values show all numeric simulation rows from the scenario folders; validity flags are reported in the source CSV.",
            "fig:1_profit_heatmap",
        )

    created = datetime.now(timezone.utc).isoformat()
    factor = 1.0 if not args.annualize else 365.0 / days
    entries: list[dict[str, Any]] = []
    for path, tier, artifact_type, metric_family, thesis_use in [
        (tables_dir / "1_net_profit_by_model_and_quantile.tex", "result_section", "latex_table", "annualized realized net profit", "primary RQ2 result table"),
        (appendix_tables_dir / "rq2_profit_and_validity_detailed.tex", "appendix", "latex_table", "profit and validity", "validity audit table"),
        (csv_dir / "rq2_scenario_summary_long.csv", "backup", "csv", "scenario summary", "reproducibility backup"),
        (csv_dir / "1_net_profit_by_model_and_quantile.csv", "backup", "csv", "annualized realized net profit", "source data for primary table"),
        (csv_dir / "1_profit_heatmap.csv", "backup", "csv", "annualized realized net profit", "source data for diagnostic heatmap with all numeric rows"),
        (csv_dir / "2_quantile_sweep_net_profit_by_model.csv", "backup", "csv", "annualized Net Profit", "source data for quantile sweep line figure"),
        (csv_dir / "3_revenue_cost_components_best_quantile.csv", "backup", "csv", "revenue and cost components", "source data for best-quantile stacked component figure"),
        (csv_dir / "4_cumulative_net_profit_model_comparison_test_period.csv", "backup", "csv", "cumulative Net Profit", "source data for best-quantile cumulative Net Profit figure"),
        (csv_dir / "5_pinball_loss_vs_net_profit_scatter_data.csv", "backup", "csv", "forecast accuracy vs Net Profit", "source data for target-specific pinball-loss scatter figures"),
        (csv_dir / "5_pinball_loss_vs_net_profit_total_scatter_data.csv", "backup", "csv", "forecast accuracy vs Net Profit", "source data for total pinball-loss scatter figure"),
        (csv_dir / "5_pinball_loss_vs_net_profit_total_normalized.csv", "backup", "csv", "normalized forecast accuracy vs Net Profit", "source data for normalized total pinball-loss scatter figure"),
        (csv_dir / "6_market_dispatch_soc_selected_day.csv", "backup", "csv", "market dispatch and SoC", "source data for selected-day stacked dispatch/SOC figure"),
        (csv_dir / "rq2_benchmark_values.csv", "backup", "csv", "benchmark values", "Naive/RHPF benchmark source"),
        (diag_dir / "rq2_input_file_inventory.csv", "backup", "diagnostics", "input inventory", "reproducibility audit"),
        (diag_dir / "rq2_validity_diagnostics.csv", "backup", "diagnostics", "validity diagnostics", "invalid-row audit"),
        (warn_dir / "rq2_warnings.csv", "backup", "warnings", "warnings", "generation warnings"),
        (warn_dir / "6_market_dispatch_soc_selected_day_warnings.csv", "backup", "warnings", "market dispatch and SoC", "selected-day invalidity and direct violation checks"),
    ]:
        entries.append(_manifest_entry(path, out_root, tier, artifact_type, metric_family, thesis_use, run_root, created, days, factor))
    for fig in result_figure_paths:
        entries.append(_manifest_entry(fig, out_root, "result_section", "figure", "annualized realized net profit", "RQ2 result-section visualization", run_root, created, days, factor))
    for fig in appendix_figure_paths:
        entries.append(_manifest_entry(fig, out_root, "appendix", "figure", "forecast accuracy vs Net Profit", "appendix scatter visualization", run_root, created, days, factor))
    for tex in latex_figures_dir.glob("*.tex"):
        entries.append(_manifest_entry(tex, out_root, "result_section", "latex_figure", "annualized realized net profit", "LaTeX figure wrapper", run_root, created, days, factor))
    for tex in appendix_latex_figures_dir.glob("*.tex"):
        entries.append(_manifest_entry(tex, out_root, "appendix", "latex_figure", "forecast accuracy vs Net Profit", "appendix LaTeX figure wrapper", run_root, created, days, factor))

    manifest = {
        "schema_version": "rq2_output_manifest_v1",
        "source_run_root": str(run_root),
        "forecast_benchmark_dir": str(forecast_benchmark_dir),
        "output_root": str(out_root),
        "created_at_utc": created,
        "split": str(args.split),
        "strict_validity": bool(args.strict_validity),
        "duration_source": duration_source,
        "simulation_days": days,
        "annualization_factor": factor,
        "outputs": entries,
    }
    (out_root / "rq2_output_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build RQ2 thesis visualizations from existing simulation outputs.")
    ap.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--forecast-benchmark-dir", default=str(DEFAULT_FORECAST_BENCHMARK_DIR), help="RQ1 forecast benchmark directory with diagnostics/joined_predictions for pinball-loss scatter plots.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--annualize", action="store_true", default=True)
    ap.add_argument("--no-annualize", dest="annualize", action="store_false")
    ap.add_argument("--strict-validity", action="store_true", default=True)
    ap.add_argument("--no-strict-validity", dest="strict_validity", action="store_false")
    ap.add_argument("--simulation-days", type=float, default=None)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--quantiles", default=",".join(DEFAULT_QUANTILES))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--formats", default="png", help="Comma-separated figure formats. Defaults to png only; pass png,pdf,svg to also write PDF/SVG.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_outputs(args)
    print(f"[OK] RQ2 outputs written: {manifest['output_root']}")
    print(f"[OK] simulation_days={manifest['simulation_days']:.6g} annualization_factor={manifest['annualization_factor']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
