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
    return f"{sign}€{abs(val):,.0f}"


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
    ax.set_ylabel("Annualized Net Profit (EUR/year)")
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

    apply_geo_style()
    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    rows = ["Naive", "RHPF", *MODEL_ORDER]
    quantiles = table["quantile"].astype(str).tolist()
    pivot = table.set_index("quantile").reindex(quantiles)[rows].T
    arr = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(arr)
    cmap = LinearSegmentedColormap.from_list(
        "geo_sequential_blue",
        [GEO_SEQUENTIAL_BLUE[f"seq_{i}"] for i in range(1, 8)],
    )
    cmap.set_bad("#F2F2F2")
    im = ax.imshow(masked, aspect="auto", cmap=cmap)
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
            txt = "n/a" if not math.isfinite(_safe_float(val)) else f"€{val/1000:,.0f}k"
            text_color = "white" if model == "RHPF" else THESIS_PALETTE["neutral_dark"]
            ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Annualized Net Profit (EUR/year)")
    fig.tight_layout()
    written = _save_figure(fig, out_base, formats)
    plt.close(fig)
    return written


def plot_net_profit_lines(sweep_data: pd.DataFrame, out_base: Path, formats: list[str], run_name: str, quantiles: list[str]) -> list[Path]:
    import matplotlib.pyplot as plt

    apply_geo_style()
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    x = np.arange(len(quantiles))
    marker_by_model = {"Naive": "D", "RHPF": "X", "RLQR": "o", "XGB": "s", "TFT": "^"}
    color_by_model = {
        "Naive": THESIS_PALETTE["naive"],
        "RHPF": THESIS_PALETTE["perfect_foresight"],
        "RLQR": get_model_color("linear"),
        "XGB": get_model_color("xgb"),
        "TFT": get_model_color("tft"),
    }
    linestyle_by_model = {"Naive": "--", "RHPF": "--", "RLQR": "-", "XGB": "-", "TFT": "-"}
    label_offsets = {"Naive": 6, "RHPF": -12, "RLQR": 6, "XGB": 6, "TFT": 6}
    for model in ["Naive", "RHPF", *MODEL_ORDER]:
        g = sweep_data.loc[sweep_data["series"].astype(str).eq(model)].copy()
        if g.empty:
            continue
        values = []
        for q in quantiles:
            match = g.loc[g["quantile"].astype(str).eq(q)]
            values.append(_safe_float(match["annualized_net_profit_eur_per_year"].iloc[0]) if not match.empty else np.nan)
        values_arr = np.asarray(values, dtype=float)
        ax.plot(
            x,
            values_arr,
            marker=marker_by_model[model],
            linewidth=2.0,
            linestyle=linestyle_by_model[model],
            label=model if model not in BENCHMARK_ORDER else f"{model} benchmark",
            color=color_by_model[model],
        )
        for xi, yi in zip(x, values_arr):
            if not math.isfinite(float(yi)):
                continue
            ax.annotate(
                f"€{yi/1000:,.0f}k",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, label_offsets[model]),
                ha="center",
                fontsize=8,
                color=THESIS_PALETTE["neutral_dark"],
            )
    ax.set_xticks(x)
    ax.set_xticklabels(quantiles)
    ax.set_ylabel("Annualized Net Profit (EUR/year)")
    ax.set_xlabel("Quantile policy")
    ax.set_title("Quantile Sweep: Net Profit by Model")
    ax.text(
        0.0,
        1.02,
        f"All numeric model/benchmark rows, annualized from {run_name}. Validity flags are reported in the source CSV.",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.legend(ncol=3, loc="best")
    fig.tight_layout()
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
    ax.set_title("RQ2 Revenue and Cost Components at Best Quantile")
    ax.text(
        0.0,
        1.02,
        f"Best numeric quantile per model, annualized from {run_name}. Costs are plotted below zero; validity flags are reported in the source CSV.",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    written = _save_figure(fig, out_base, formats)
    plt.close(fig)
    return written


def plot_cumulative_pnl(cumulative: pd.DataFrame, out_base: Path, formats: list[str], run_name: str) -> list[Path]:
    import matplotlib.pyplot as plt

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
        "RHPF": THESIS_PALETTE["perfect_foresight"],
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
            group["cum_pnl_eur"],
            label=label,
            color=color_map.get(model, THESIS_PALETTE["neutral_dark"]),
            linestyle=line_style.get(model, "-"),
            linewidth=2.2 if model not in BENCHMARK_ORDER else 1.9,
        )
    ax.set_ylabel("Cumulative Net Profit (EUR)")
    ax.set_xlabel("Time")
    ax.set_title("Cumulative Net Profit: Model Comparison Over Test Period")
    ax.text(
        0.0,
        1.02,
        f"Best numeric quantile per model, from {run_name}. Validity flags are reported in the source CSV.",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.legend(ncol=3, loc="best")
    fig.autofmt_xdate(rotation=0)
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
                ax.scatter(
                    g["mean_pinball_loss"],
                    g["annualized_profit_eur_per_year"],
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
                        (row["mean_pinball_loss"], row["annualized_profit_eur_per_year"]),
                        textcoords="offset points",
                        xytext=(4, 3),
                        fontsize=8,
                        color=THESIS_PALETTE["neutral_dark"],
                    )
            ax.set_xlabel("Mean pinball loss")
            ax.set_ylabel("Annualized Net Profit (EUR/year)")
            ax.legend(title="Model", loc="best")
        ax.set_title(f"RQ2 Pinball Loss vs Net Profit: {label}")
        fig.tight_layout()
        written.extend(_save_figure(fig, figures_dir / f"5_pinball_loss_vs_net_profit_{slug}", formats))
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
            ax.scatter(
                g["mean_pinball_loss"],
                g["annualized_profit_eur_per_year"],
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
                    (row["mean_pinball_loss"], row["annualized_profit_eur_per_year"]),
                    textcoords="offset points",
                    xytext=(4, 3),
                    fontsize=8,
                    color=THESIS_PALETTE["neutral_dark"],
                )
        ax.set_xlabel("Total mean pinball loss")
        ax.set_ylabel("Annualized Net Profit (EUR/year)")
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

    csv_dir = out_root / "backup/csv"
    diag_dir = out_root / "backup/diagnostics"
    warn_dir = out_root / "backup/warnings"
    figures_dir = out_root / "result_section/figures"
    tables_dir = out_root / "result_section/tables"
    latex_figures_dir = out_root / "result_section/latex_figures"
    appendix_tables_dir = out_root / "appendix/tables"

    summary.to_csv(csv_dir / "rq2_scenario_summary_long.csv", index=False)
    table.to_csv(csv_dir / "1_net_profit_by_model_and_quantile.csv", index=False)
    heatmap_table.to_csv(csv_dir / "1_profit_heatmap.csv", index=False)
    sweep_data.to_csv(csv_dir / "2_quantile_sweep_net_profit_by_model.csv", index=False)
    component_data.to_csv(csv_dir / "3_revenue_cost_components_best_quantile.csv", index=False)
    cumulative_data.to_csv(csv_dir / "4_cumulative_net_profit_model_comparison_test_period.csv", index=False)
    scatter_data.to_csv(csv_dir / "5_pinball_loss_vs_net_profit_scatter_data.csv", index=False)
    total_scatter_data.to_csv(csv_dir / "5_pinball_loss_vs_net_profit_total_scatter_data.csv", index=False)
    bench.to_csv(csv_dir / "rq2_benchmark_values.csv", index=False)
    inventory.to_csv(diag_dir / "rq2_input_file_inventory.csv", index=False)
    validity.to_csv(diag_dir / "rq2_validity_diagnostics.csv", index=False)
    warning_df.to_csv(warn_dir / "rq2_warnings.csv", index=False)

    write_primary_table(tables_dir / "1_net_profit_by_model_and_quantile.tex", table, days)
    write_appendix_table(appendix_tables_dir / "rq2_profit_and_validity_detailed.tex", summary)

    figure_paths: list[Path] = []
    if formats:
        figure_paths += plot_net_profit_lines(sweep_data, figures_dir / "2_quantile_sweep_net_profit_by_model", formats, run_root.name, quantiles)
        figure_paths += plot_best_quantile_components(component_data, figures_dir / "3_revenue_cost_components_best_quantile", formats, run_root.name)
        figure_paths += plot_cumulative_pnl(cumulative_data, figures_dir / "4_cumulative_net_profit_model_comparison_test_period", formats, run_root.name)
        figure_paths += plot_pinball_net_profit_scatter(scatter_data, figures_dir, formats)
        figure_paths += plot_total_pinball_net_profit_scatter(total_scatter_data, figures_dir / "5_pinball_loss_vs_net_profit_total", formats)
        figure_paths += plot_heatmap(heatmap_table, figures_dir / "1_profit_heatmap", formats)
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
        for target, label in TARGET_LABELS.items():
            slug = TARGET_SLUGS[target]
            _write_latex_include(
                latex_figures_dir / f"5_pinball_loss_vs_net_profit_{slug}.tex",
                _thesis_figure_rel(out_root, f"5_pinball_loss_vs_net_profit_{slug}.png"),
                f"Mean pinball loss and annualized Net Profit for {label.replace('$-$', '-')} by model and quantile policy.",
                f"fig:5_pinball_loss_vs_net_profit_{slug}",
            )
        _write_latex_include(
            latex_figures_dir / "5_pinball_loss_vs_net_profit_total.tex",
            _thesis_figure_rel(out_root, "5_pinball_loss_vs_net_profit_total.png"),
            "Total mean pinball loss and annualized Net Profit by model and quantile policy. The forecast metric is the observation-weighted raw mean pinball loss across all target variables.",
            "fig:5_pinball_loss_vs_net_profit_total",
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
        (csv_dir / "rq2_benchmark_values.csv", "backup", "csv", "benchmark values", "Naive/RHPF benchmark source"),
        (diag_dir / "rq2_input_file_inventory.csv", "backup", "diagnostics", "input inventory", "reproducibility audit"),
        (diag_dir / "rq2_validity_diagnostics.csv", "backup", "diagnostics", "validity diagnostics", "invalid-row audit"),
        (warn_dir / "rq2_warnings.csv", "backup", "warnings", "warnings", "generation warnings"),
    ]:
        entries.append(_manifest_entry(path, out_root, tier, artifact_type, metric_family, thesis_use, run_root, created, days, factor))
    for fig in figure_paths:
        entries.append(_manifest_entry(fig, out_root, "result_section", "figure", "annualized realized net profit", "RQ2 result-section visualization", run_root, created, days, factor))
    for tex in latex_figures_dir.glob("*.tex"):
        entries.append(_manifest_entry(tex, out_root, "result_section", "latex_figure", "annualized realized net profit", "LaTeX figure wrapper", run_root, created, days, factor))

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
