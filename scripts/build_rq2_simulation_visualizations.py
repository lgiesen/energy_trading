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
    ("penalty_cost_eur", "Penalty cost"),
    ("terminal_soc_repair_cost_eur", "Terminal SoC repair"),
]
COMPONENT_COST_COLUMNS = {
    "realized_degradation_cost_eur",
    "realized_aux_cost_eur",
    "transaction_cost_eur",
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
    "Penalty cost": "#333333",
    "Terminal SoC repair": "#045275",
}
ALL_OUTPUT_DIRS = [
    "result_section/figures",
    "result_section/tables",
    "result_section/latex_figures",
    "result_section/csv",
    "appendix/figures",
    "appendix/tables",
    "appendix/latex_figures",
    "backup/csv",
    "backup/diagnostics",
    "backup/warnings",
]

BIDDING_ACTIVITY_MARKETS = [
    (
        "DA",
        "DA",
        [
            ("buy", "real_submitted_da_buy_mw", "submitted"),
            ("sell", "real_submitted_da_sell_mw", "submitted"),
        ],
    ),
    (
        "BCM",
        "aFRR capacity",
        [
            ("positive", "real_bcm_precommit_candidate_pos_mw", "submitted"),
            ("negative", "real_bcm_precommit_candidate_neg_mw", "submitted"),
        ],
    ),
    (
        "BEM",
        "aFRR activation",
        [
            ("positive", "real_bem_only_submitted_pos_mw", "submitted"),
            ("negative", "real_bem_only_submitted_neg_mw", "submitted"),
        ],
    ),
    (
        "ID",
        "ID",
        [
            ("buy", "real_executed_id_charge_mw_for_soc_feedback", "realized_fallback"),
            ("sell", "real_executed_id_discharge_mw_for_soc_feedback", "realized_fallback"),
        ],
    ),
]

BIDDING_ACTIVITY_SUBMITTED_CLEARED_SPECS = [
    {
        "market": "DA",
        "market_label": "DA",
        "submitted_groups": [[("real_submitted_da_buy_mw", "mw"), ("real_submitted_da_sell_mw", "mw")]],
        "cleared_groups": [
            [("real_da_auction_accepted_buy_mwh", "mwh"), ("real_da_auction_accepted_sell_mwh", "mwh")],
            [("da_executed_buy_mwh", "mwh"), ("da_executed_sell_mwh", "mwh")],
            [("real_executed_charge_mw", "mw"), ("real_executed_discharge_mw", "mw")],
        ],
        "metric_semantics": "submitted_and_cleared",
    },
    {
        "market": "BCM",
        "market_label": "BCM",
        "submitted_groups": [
            [("reserve_submitted_pos_mw", "mw"), ("reserve_submitted_neg_mw", "mw")],
            [("real_bcm_precommit_candidate_pos_mw", "mw"), ("real_bcm_precommit_candidate_neg_mw", "mw")],
            [("real_submitted_bcm_capacity_pos_mw", "mw"), ("real_submitted_bcm_capacity_neg_mw", "mw")],
        ],
        "cleared_groups": [
            [("reserve_awarded_pos_mw", "mw"), ("reserve_awarded_neg_mw", "mw")],
            [("real_bcm_precommit_locked_pos_mw", "mw"), ("real_bcm_precommit_locked_neg_mw", "mw")],
            [("bcm_precommit_written_pos_mw", "mw"), ("bcm_precommit_written_neg_mw", "mw")],
            [("real_bcm_capacity_awarded_pos_mw", "mw"), ("real_bcm_capacity_awarded_neg_mw", "mw")],
            [("real_executed_bcm_capacity_pos_mw", "mw"), ("real_executed_bcm_capacity_neg_mw", "mw")],
        ],
        "bid_price_groups": [
            [
                ("real_bcm_precommit_candidate_pos_mw", "real_ev_bcm_capacity_bid_price_pos_bin_0_eur_per_mw_h"),
                ("real_bcm_precommit_candidate_neg_mw", "real_ev_bcm_capacity_bid_price_neg_bin_0_eur_per_mw_h"),
            ],
            [
                ("reserve_submitted_pos_mw", "real_ev_bcm_capacity_bid_price_pos_bin_0_eur_per_mw_h"),
                ("reserve_submitted_neg_mw", "real_ev_bcm_capacity_bid_price_neg_bin_0_eur_per_mw_h"),
            ],
            [
                ("real_submitted_bcm_capacity_pos_mw", "real_bcm_capacity_bid_price_pos_eur_per_mw_h"),
                ("real_submitted_bcm_capacity_neg_mw", "real_bcm_capacity_bid_price_neg_eur_per_mw_h"),
            ],
        ],
        "clearing_price_groups": [
            [
                ("real_bcm_precommit_candidate_pos_mw", "real_bcm_capacity_clearing_price_pos_eur_per_mw_h"),
                ("real_bcm_precommit_candidate_neg_mw", "real_bcm_capacity_clearing_price_neg_eur_per_mw_h"),
            ],
            [
                ("reserve_submitted_pos_mw", "real_bcm_capacity_clearing_price_pos_eur_per_mw_h"),
                ("reserve_submitted_neg_mw", "real_bcm_capacity_clearing_price_neg_eur_per_mw_h"),
            ],
            [
                ("real_submitted_bcm_capacity_pos_mw", "real_bcm_capacity_clearing_price_pos_eur_per_mw_h"),
                ("real_submitted_bcm_capacity_neg_mw", "real_bcm_capacity_clearing_price_neg_eur_per_mw_h"),
            ],
        ],
        "metric_semantics": "precommit_submitted_and_awarded",
    },
    {
        "market": "BEM",
        "market_label": "BEM",
        "submitted_groups": [[("real_bem_only_submitted_pos_mw", "mw"), ("real_bem_only_submitted_neg_mw", "mw")]],
        "cleared_groups": [[("real_bem_only_executed_pos_mw", "mw"), ("real_bem_only_executed_neg_mw", "mw")]],
        "metric_semantics": "submitted_and_cleared",
    },
    {
        "market": "ID",
        "market_label": "ID",
        "submitted_groups": [[("real_executed_id_charge_mw_for_soc_feedback", "mw"), ("real_executed_id_discharge_mw_for_soc_feedback", "mw")]],
        "cleared_groups": [[("real_executed_id_charge_mw_for_soc_feedback", "mw"), ("real_executed_id_discharge_mw_for_soc_feedback", "mw")]],
        "metric_semantics": "realized_only_fallback",
    },
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


def _ensure_caption_period(caption: Any) -> str:
    text = str(caption).strip()
    if not text:
        return "."
    return text if text.endswith(".") else text + "."


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


def _find_model_hourly_path(scenario_dir: Path) -> Path | None:
    candidates = sorted(scenario_dir.rglob("model_hourly.parquet"))
    return candidates[0] if candidates else None


def _infer_timestep_hours(timestamps: pd.Series) -> float:
    ts = pd.to_datetime(timestamps, errors="coerce", utc=True).dropna().sort_values()
    if len(ts) < 2:
        return 1.0
    diffs = ts.diff().dropna().dt.total_seconds() / 3600.0
    diffs = diffs.loc[(diffs > 0) & np.isfinite(diffs)]
    if diffs.empty:
        return 1.0
    return float(diffs.median())


def _infer_time_coverage(timestamps: pd.Series) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT, float, float, float, int]:
    ts = pd.to_datetime(timestamps, errors="coerce", utc=True).dropna().sort_values()
    if ts.empty:
        return pd.NaT, pd.NaT, math.nan, math.nan, math.nan, 0
    timestep_hours = _infer_timestep_hours(ts)
    start = ts.min()
    end = ts.max()
    observed = int(ts.nunique())
    if not math.isfinite(timestep_hours) or timestep_hours <= 0.0:
        timestep_hours = 1.0
    total_hours = ((end - start).total_seconds() / 3600.0) + timestep_hours
    test_period_days = total_hours / 24.0 if total_hours > 0.0 else math.nan
    expected = int(round(total_hours / timestep_hours)) if total_hours > 0.0 else observed
    missing = max(expected - observed, 0)
    coverage_share = observed / expected if expected > 0 else math.nan
    return start, end, test_period_days, coverage_share, timestep_hours, missing


def _selected_model_strategy_rows(summary: pd.DataFrame) -> pd.DataFrame:
    model_rows = summary.loc[~summary["is_benchmark"].astype(bool)].copy()
    model_rows["annualized_profit_eur_per_year"] = pd.to_numeric(model_rows.get("annualized_profit_eur_per_year", np.nan), errors="coerce")
    model_rows = model_rows.dropna(subset=["annualized_profit_eur_per_year"]).copy()
    if model_rows.empty:
        return pd.DataFrame()
    best = (
        model_rows.sort_values(["model", "annualized_profit_eur_per_year"], ascending=[True, False])
        .groupby("model", as_index=False)
        .head(1)
        .copy()
    )
    order = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    best["_order"] = best["model"].map(order).fillna(999)
    return best.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def _all_model_strategy_rows(summary: pd.DataFrame) -> pd.DataFrame:
    model_rows = summary.loc[~summary["is_benchmark"].astype(bool)].copy()
    model_rows = model_rows.loc[model_rows["model"].astype(str).isin(MODEL_ORDER)].copy()
    if model_rows.empty:
        return pd.DataFrame()
    order = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    q_order = {q: idx for idx, q in enumerate(DEFAULT_QUANTILES)}
    model_rows["_model_order"] = model_rows["model"].map(order).fillna(999)
    model_rows["_quantile_order"] = model_rows["quantile"].astype(str).map(q_order).fillna(999)
    return model_rows.sort_values(["_model_order", "_quantile_order"]).drop(columns=["_model_order", "_quantile_order"]).reset_index(drop=True)


def _choose_bidding_source_group(df: pd.DataFrame, groups: list[list[tuple[str, str]]]) -> list[tuple[str, str]]:
    for group in groups:
        if all(column in df.columns for column, _unit in group):
            return group
    return []


def _bidding_count_and_volume(df: pd.DataFrame, source_group: list[tuple[str, str]], timestep_hours: float) -> tuple[int, float]:
    count = 0
    volume = 0.0
    for column, unit in source_group:
        values = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        count += int((values.abs() > 1e-9).sum())
        multiplier = timestep_hours if unit == "mw" else 1.0
        volume += float(values.abs().sum() * multiplier)
    return count, volume


def _choose_price_source_group(
    df: pd.DataFrame,
    groups: list[list[tuple[str, str]]],
) -> list[tuple[str, str]]:
    for group in groups:
        if all(volume_col in df.columns and price_col in df.columns for volume_col, price_col in group):
            return group
    return []


def _weighted_average_price(df: pd.DataFrame, source_group: list[tuple[str, str]]) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0
    for volume_col, price_col in source_group:
        volume = pd.to_numeric(df[volume_col], errors="coerce")
        price = pd.to_numeric(df[price_col], errors="coerce")
        mask = volume.abs().gt(1e-9) & price.notna() & np.isfinite(price)
        if not bool(mask.any()):
            continue
        weights = volume.loc[mask].abs()
        weighted_sum += float((weights * price.loc[mask]).sum())
        weight_sum += float(weights.sum())
    return float(weighted_sum / weight_sum) if weight_sum > 1e-12 else math.nan


def _difference_or_nan(submitted: float, cleared: float, *, tolerance: float = 1e-6) -> tuple[float, bool]:
    diff = submitted - cleared
    if diff < -tolerance:
        return math.nan, True
    return max(diff, 0.0), False


def _bcm_precommit_price_rejection_adjusted_activity(
    df: pd.DataFrame,
    *,
    timestep_hours: float,
) -> dict[str, float | str] | None:
    required = [
        "real_bcm_precommit_candidate_pos_mw",
        "real_bcm_precommit_candidate_neg_mw",
        "real_bcm_precommit_locked_pos_mw",
        "real_bcm_precommit_locked_neg_mw",
        "real_ev_bcm_capacity_bid_price_pos_bin_0_eur_per_mw_h",
        "real_ev_bcm_capacity_bid_price_neg_bin_0_eur_per_mw_h",
        "real_bcm_capacity_clearing_price_pos_eur_per_mw_h",
        "real_bcm_capacity_clearing_price_neg_eur_per_mw_h",
    ]
    if not all(c in df.columns for c in required):
        return None

    cleared_count = 0
    cleared_volume = 0.0
    price_rejected_count = 0
    price_rejected_volume = 0.0
    excluded_count = 0
    excluded_volume = 0.0
    excluded_missing_or_zero_price_count = 0
    excluded_missing_or_zero_price_volume = 0.0
    excluded_physical_count = 0
    excluded_physical_volume = 0.0
    excluded_ev_count = 0
    excluded_ev_volume = 0.0
    excluded_optimizer_count = 0
    excluded_optimizer_volume = 0.0
    bid_weighted_sum = 0.0
    bid_weight = 0.0
    clearing_weighted_sum = 0.0
    clearing_weight = 0.0

    reason = (
        df["real_bcm_precommit_zero_reason"].fillna("").astype(str).str.lower()
        if "real_bcm_precommit_zero_reason" in df.columns
        else pd.Series("", index=df.index)
    )

    for side in ("pos", "neg"):
        cand = pd.to_numeric(df[f"real_bcm_precommit_candidate_{side}_mw"], errors="coerce").fillna(0.0).clip(lower=0.0)
        locked = pd.to_numeric(df[f"real_bcm_precommit_locked_{side}_mw"], errors="coerce").fillna(0.0).clip(lower=0.0)
        bid = pd.to_numeric(df[f"real_ev_bcm_capacity_bid_price_{side}_bin_0_eur_per_mw_h"], errors="coerce")
        clearing = pd.to_numeric(df[f"real_bcm_capacity_clearing_price_{side}_eur_per_mw_h"], errors="coerce")
        rejected_mw = (cand - locked).clip(lower=0.0)
        cleared_mask = locked.gt(1e-9)
        rejected_mask = rejected_mw.gt(1e-9)
        finite_positive_price = bid.notna() & np.isfinite(bid) & bid.gt(1e-12)
        finite_clearing = clearing.notna() & np.isfinite(clearing)
        price_rejected_mask = rejected_mask & finite_positive_price & finite_clearing & bid.gt(clearing)
        missing_or_zero_price_mask = rejected_mask & (~finite_positive_price | ~finite_clearing)
        physical_mask = rejected_mask & reason.str.contains(
            "headroom|physical|protected_soc|reserve_infeasible|power_infeasible|soc_infeasible|id_recovery|terminal",
            regex=True,
        )
        ev_mask = rejected_mask & reason.str.contains("candidate_incremental_ev_negative|negative_ev|ev_nan", regex=True)
        optimizer_mask = rejected_mask & reason.str.contains("optimizer_validation", regex=True)
        excluded_mask = rejected_mask & ~price_rejected_mask

        cleared_count += int(cleared_mask.sum())
        cleared_volume += float(locked.abs().sum() * timestep_hours)
        price_rejected_count += int(price_rejected_mask.sum())
        price_rejected_volume += float(rejected_mw.where(price_rejected_mask, 0.0).abs().sum() * timestep_hours)
        excluded_count += int(excluded_mask.sum())
        excluded_volume += float(rejected_mw.where(excluded_mask, 0.0).abs().sum() * timestep_hours)
        excluded_missing_or_zero_price_count += int(missing_or_zero_price_mask.sum())
        excluded_missing_or_zero_price_volume += float(
            rejected_mw.where(missing_or_zero_price_mask, 0.0).abs().sum() * timestep_hours
        )
        excluded_physical_count += int((excluded_mask & physical_mask).sum())
        excluded_physical_volume += float(rejected_mw.where(excluded_mask & physical_mask, 0.0).abs().sum() * timestep_hours)
        excluded_ev_count += int((excluded_mask & ev_mask).sum())
        excluded_ev_volume += float(rejected_mw.where(excluded_mask & ev_mask, 0.0).abs().sum() * timestep_hours)
        excluded_optimizer_count += int((excluded_mask & optimizer_mask).sum())
        excluded_optimizer_volume += float(rejected_mw.where(excluded_mask & optimizer_mask, 0.0).abs().sum() * timestep_hours)

        auction_mw = locked + rejected_mw.where(price_rejected_mask, 0.0)
        bid_mask = auction_mw.gt(1e-9) & bid.notna() & np.isfinite(bid)
        clearing_mask = auction_mw.gt(1e-9) & clearing.notna() & np.isfinite(clearing)
        if bool(bid_mask.any()):
            weights = auction_mw.loc[bid_mask].abs()
            bid_weighted_sum += float((weights * bid.loc[bid_mask]).sum())
            bid_weight += float(weights.sum())
        if bool(clearing_mask.any()):
            weights = auction_mw.loc[clearing_mask].abs()
            clearing_weighted_sum += float((weights * clearing.loc[clearing_mask]).sum())
            clearing_weight += float(weights.sum())

    submitted_count = cleared_count + price_rejected_count
    submitted_volume = cleared_volume + price_rejected_volume
    return {
        "submitted_count": float(submitted_count),
        "cleared_count": float(cleared_count),
        "not_cleared_count": float(price_rejected_count),
        "submitted_volume": float(submitted_volume),
        "cleared_volume": float(cleared_volume),
        "not_cleared_volume": float(price_rejected_volume),
        "excluded_count": float(excluded_count),
        "excluded_volume": float(excluded_volume),
        "excluded_missing_or_zero_price_count": float(excluded_missing_or_zero_price_count),
        "excluded_missing_or_zero_price_volume": float(excluded_missing_or_zero_price_volume),
        "excluded_physical_count": float(excluded_physical_count),
        "excluded_physical_volume": float(excluded_physical_volume),
        "excluded_ev_count": float(excluded_ev_count),
        "excluded_ev_volume": float(excluded_ev_volume),
        "excluded_optimizer_count": float(excluded_optimizer_count),
        "excluded_optimizer_volume": float(excluded_optimizer_volume),
        "submitted_bid_price_avg": float(bid_weighted_sum / bid_weight) if bid_weight > 1e-12 else math.nan,
        "clearing_price_avg": float(clearing_weighted_sum / clearing_weight) if clearing_weight > 1e-12 else math.nan,
        "semantics": "BCM precommit auction-adjusted: submitted=awarded plus price-rejected; feasibility/EV/optimizer/missing-price filtered candidates excluded",
    }


def build_bidding_activity_by_market_model(
    summary: pd.DataFrame,
    *,
    run_root: Path,
    selected_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [
        "run_root",
        "scenario",
        "model",
        "quantile",
        "strategy_label",
        "market",
        "market_label",
        "test_period_start_utc",
        "test_period_end_utc",
        "test_period_days",
        "annualization_factor",
        "submitted_bid_count_total",
        "cleared_bid_count_total",
        "not_cleared_bid_count_total",
        "submitted_bid_volume_mwh_total",
        "cleared_bid_volume_mwh_total",
        "not_cleared_bid_volume_mwh_total",
        "submitted_bid_count_annualized",
        "cleared_bid_count_annualized",
        "not_cleared_bid_count_annualized",
        "submitted_bid_volume_mwh_annualized",
        "cleared_bid_volume_mwh_annualized",
        "not_cleared_bid_volume_mwh_annualized",
        "submitted_bid_price_eur_per_mw_h_weighted_avg",
        "clearing_price_eur_per_mw_h_weighted_avg",
        "excluded_filtered_bid_count_total",
        "excluded_filtered_bid_volume_mwh_total",
        "excluded_physical_bid_count_total",
        "excluded_physical_bid_volume_mwh_total",
        "excluded_candidate_ev_bid_count_total",
        "excluded_candidate_ev_bid_volume_mwh_total",
        "excluded_optimizer_bid_count_total",
        "excluded_optimizer_bid_volume_mwh_total",
        "excluded_missing_or_zero_price_bid_count_total",
        "excluded_missing_or_zero_price_bid_volume_mwh_total",
        "excluded_filtering_semantics",
        "bid_price_source",
        "clearing_price_source",
        "submitted_source",
        "cleared_source",
        "volume_unit_source",
        "count_semantics",
        "volume_semantics",
        "metric_semantics",
        "coverage_share",
        "missing_hours_count",
        "warnings",
    ]
    warning_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    selected = _selected_model_strategy_rows(summary) if selected_only else _all_model_strategy_rows(summary)
    if selected.empty:
        scope = "selected model strategies" if selected_only else "model-quantile strategies"
        warning_rows.append({"severity": "warning", "scenario": "", "market": "", "message": f"No {scope} available for bidding-activity aggregation."})
        return pd.DataFrame(columns=columns), pd.DataFrame(warning_rows), pd.DataFrame(inventory_rows)

    for _, scenario_row in selected.iterrows():
        scenario = str(scenario_row.get("folder", ""))
        model = str(scenario_row.get("model", ""))
        quantile = str(scenario_row.get("quantile", ""))
        strategy_label = f"{model} {quantile}".strip()
        scenario_dir = run_root / scenario
        model_hourly_path = _find_model_hourly_path(scenario_dir)
        if model_hourly_path is None:
            warning_rows.append({"severity": "warning", "scenario": scenario, "market": "", "message": "Missing model_hourly.parquet; scenario omitted from bidding-activity figure."})
            inventory_rows.append(
                {
                    "scenario": scenario,
                    "candidate_file": str(scenario_dir / "multi" / f"{quantile}_{quantile}" / "model_hourly.parquet"),
                    "exists": False,
                    "used": False,
                    "row_count": 0,
                    "available_columns": "",
                    "metrics_supported": "",
                    "reason_if_not_used": "missing_model_hourly",
                }
            )
            continue
        df = pd.read_parquet(model_hourly_path).copy()
        if "timestamp_utc" in df.columns:
            start, end, test_period_days, coverage_share, timestep_hours, missing_hours = _infer_time_coverage(df["timestamp_utc"])
        else:
            start, end, test_period_days, coverage_share, timestep_hours, missing_hours = pd.NaT, pd.NaT, math.nan, math.nan, 1.0, 0
            warning_rows.append({"severity": "warning", "scenario": scenario, "market": "", "message": "Missing timestamp_utc; annualization uses NaN test-period days."})
        annualization_factor = 365.0 / test_period_days if math.isfinite(test_period_days) and test_period_days > 0.0 else math.nan
        coverage_warning = ""
        if math.isfinite(coverage_share) and coverage_share < 0.999:
            coverage_warning = f"incomplete timestamp coverage: coverage_share={coverage_share:.4f}, missing_hours={missing_hours}"
            warning_rows.append({"severity": "warning", "scenario": scenario, "market": "", "message": coverage_warning})
        inventory_rows.append(
            {
                "scenario": scenario,
                "candidate_file": str(model_hourly_path),
                "exists": True,
                "used": True,
                "row_count": len(df),
                "available_columns": ",".join(str(c) for c in df.columns),
                "metrics_supported": "submitted_DA,submitted_BCM,submitted_BEM,realized_ID_fallback",
                "reason_if_not_used": "",
            }
        )
        for spec in BIDDING_ACTIVITY_SUBMITTED_CLEARED_SPECS:
            market = str(spec["market"])
            market_label = str(spec["market_label"])
            submitted_group = _choose_bidding_source_group(df, spec["submitted_groups"])  # type: ignore[arg-type]
            cleared_group = _choose_bidding_source_group(df, spec["cleared_groups"])  # type: ignore[arg-type]
            bid_price_group = _choose_price_source_group(df, spec.get("bid_price_groups", []))  # type: ignore[arg-type]
            clearing_price_group = _choose_price_source_group(df, spec.get("clearing_price_groups", []))  # type: ignore[arg-type]
            row_warnings: list[str] = []
            if not submitted_group:
                row_warnings.append("missing submitted bid source columns")
                warning_rows.append({"severity": "warning", "scenario": scenario, "market": market, "message": "Missing submitted bid source columns."})
            if not cleared_group:
                row_warnings.append("missing cleared/realized source columns")
                warning_rows.append({"severity": "warning", "scenario": scenario, "market": market, "message": "Missing cleared/realized source columns."})
            if market == "ID":
                row_warnings.append("submitted ID bid data unavailable; using realized ID recourse activity as submitted=cleared fallback")
                warning_rows.append(
                    {
                        "severity": "warning",
                        "scenario": scenario,
                        "market": market,
                        "message": "Submitted ID bid data unavailable; using realized ID recourse activity as submitted=cleared fallback.",
                    }
                )
            if not submitted_group or not cleared_group:
                row_warnings.append("market omitted: no usable bid/activity columns")
                warning_rows.append({"severity": "warning", "scenario": scenario, "market": market, "message": "No usable bid/activity columns for market."})
            submitted_count, submitted_volume = _bidding_count_and_volume(df, submitted_group, timestep_hours) if submitted_group else (math.nan, math.nan)
            cleared_count, cleared_volume = _bidding_count_and_volume(df, cleared_group, timestep_hours) if cleared_group else (math.nan, math.nan)
            submitted_bid_price_avg = _weighted_average_price(df, bid_price_group) if bid_price_group else math.nan
            clearing_price_avg = _weighted_average_price(df, clearing_price_group) if clearing_price_group else math.nan
            not_cleared_count, neg_count = _difference_or_nan(float(submitted_count), float(cleared_count)) if math.isfinite(float(submitted_count)) and math.isfinite(float(cleared_count)) else (math.nan, False)
            not_cleared_volume, neg_volume = _difference_or_nan(float(submitted_volume), float(cleared_volume)) if math.isfinite(float(submitted_volume)) and math.isfinite(float(cleared_volume)) else (math.nan, False)
            excluded_filtered_count = math.nan
            excluded_filtered_volume = math.nan
            excluded_physical_count = math.nan
            excluded_physical_volume = math.nan
            excluded_ev_count = math.nan
            excluded_ev_volume = math.nan
            excluded_optimizer_count = math.nan
            excluded_optimizer_volume = math.nan
            excluded_missing_or_zero_price_count = math.nan
            excluded_missing_or_zero_price_volume = math.nan
            excluded_filtering_semantics = ""
            if market == "BCM":
                adjusted = _bcm_precommit_price_rejection_adjusted_activity(df, timestep_hours=timestep_hours)
                if adjusted is not None:
                    submitted_count = float(adjusted["submitted_count"])
                    cleared_count = float(adjusted["cleared_count"])
                    not_cleared_count = float(adjusted["not_cleared_count"])
                    submitted_volume = float(adjusted["submitted_volume"])
                    cleared_volume = float(adjusted["cleared_volume"])
                    not_cleared_volume = float(adjusted["not_cleared_volume"])
                    submitted_bid_price_avg = float(adjusted["submitted_bid_price_avg"])
                    clearing_price_avg = float(adjusted["clearing_price_avg"])
                    excluded_filtered_count = float(adjusted["excluded_count"])
                    excluded_filtered_volume = float(adjusted["excluded_volume"])
                    excluded_physical_count = float(adjusted["excluded_physical_count"])
                    excluded_physical_volume = float(adjusted["excluded_physical_volume"])
                    excluded_ev_count = float(adjusted["excluded_ev_count"])
                    excluded_ev_volume = float(adjusted["excluded_ev_volume"])
                    excluded_optimizer_count = float(adjusted["excluded_optimizer_count"])
                    excluded_optimizer_volume = float(adjusted["excluded_optimizer_volume"])
                    excluded_missing_or_zero_price_count = float(adjusted["excluded_missing_or_zero_price_count"])
                    excluded_missing_or_zero_price_volume = float(adjusted["excluded_missing_or_zero_price_volume"])
                    excluded_filtering_semantics = str(adjusted["semantics"])
                    row_warnings.append("BCM graph excludes feasibility/EV/optimizer/missing-price filtered candidates from submitted/not-cleared segments")
                    warning_rows.append(
                        {
                            "severity": "info",
                            "scenario": scenario,
                            "market": market,
                            "message": "BCM bidding graph adjusted to awarded plus price-rejected candidates; non-auction filters are excluded and reported in CSV columns.",
                        }
                    )
                    neg_count = False
                    neg_volume = False
            if neg_count:
                row_warnings.append("negative not-cleared bid count; source values inconsistent")
                warning_rows.append({"severity": "warning", "scenario": scenario, "market": market, "message": "Negative not-cleared bid count; set to NaN."})
            if neg_volume:
                row_warnings.append("negative not-cleared bid volume; source values inconsistent")
                warning_rows.append({"severity": "warning", "scenario": scenario, "market": market, "message": "Negative not-cleared bid volume; set to NaN."})
            if math.isfinite(float(submitted_volume)) and abs(float(submitted_volume)) <= 1e-12:
                row_warnings.append("zero submitted activity volume")
                warning_rows.append({"severity": "warning", "scenario": scenario, "market": market, "message": "Zero submitted activity volume."})

            if coverage_warning:
                row_warnings.append(coverage_warning)
            metric_semantics = str(spec["metric_semantics"])
            count_semantics = "nonzero submitted and cleared/realized directional source rows"
            volume_semantics = "absolute source volumes; MW source columns converted to MWh using inferred timestep"
            def annualize(value: float) -> float:
                return value * annualization_factor if math.isfinite(value) and math.isfinite(annualization_factor) else math.nan
            rows.append(
                {
                    "run_root": str(run_root),
                    "scenario": scenario,
                    "model": model,
                    "quantile": quantile,
                    "strategy_label": strategy_label,
                    "market": market,
                    "market_label": market_label,
                    "test_period_start_utc": start.isoformat() if pd.notna(start) else "",
                    "test_period_end_utc": end.isoformat() if pd.notna(end) else "",
                    "test_period_days": test_period_days,
                    "annualization_factor": annualization_factor,
                    "submitted_bid_count_total": submitted_count,
                    "cleared_bid_count_total": cleared_count,
                    "not_cleared_bid_count_total": not_cleared_count,
                    "submitted_bid_volume_mwh_total": submitted_volume,
                    "cleared_bid_volume_mwh_total": cleared_volume,
                    "not_cleared_bid_volume_mwh_total": not_cleared_volume,
                    "submitted_bid_count_annualized": annualize(float(submitted_count)),
                    "cleared_bid_count_annualized": annualize(float(cleared_count)),
                    "not_cleared_bid_count_annualized": annualize(float(not_cleared_count)),
                    "submitted_bid_volume_mwh_annualized": annualize(float(submitted_volume)),
                    "cleared_bid_volume_mwh_annualized": annualize(float(cleared_volume)),
                    "not_cleared_bid_volume_mwh_annualized": annualize(float(not_cleared_volume)),
                    "submitted_bid_price_eur_per_mw_h_weighted_avg": submitted_bid_price_avg,
                    "clearing_price_eur_per_mw_h_weighted_avg": clearing_price_avg,
                    "excluded_filtered_bid_count_total": excluded_filtered_count,
                    "excluded_filtered_bid_volume_mwh_total": excluded_filtered_volume,
                    "excluded_physical_bid_count_total": excluded_physical_count,
                    "excluded_physical_bid_volume_mwh_total": excluded_physical_volume,
                    "excluded_candidate_ev_bid_count_total": excluded_ev_count,
                    "excluded_candidate_ev_bid_volume_mwh_total": excluded_ev_volume,
                    "excluded_optimizer_bid_count_total": excluded_optimizer_count,
                    "excluded_optimizer_bid_volume_mwh_total": excluded_optimizer_volume,
                    "excluded_missing_or_zero_price_bid_count_total": excluded_missing_or_zero_price_count,
                    "excluded_missing_or_zero_price_bid_volume_mwh_total": excluded_missing_or_zero_price_volume,
                    "excluded_filtering_semantics": excluded_filtering_semantics,
                    "bid_price_source": ";".join(f"{volume_col}->{price_col}" for volume_col, price_col in bid_price_group),
                    "clearing_price_source": ";".join(f"{volume_col}->{price_col}" for volume_col, price_col in clearing_price_group),
                    "submitted_source": ";".join(f"{col}:{unit}" for col, unit in submitted_group),
                    "cleared_source": ";".join(f"{col}:{unit}" for col, unit in cleared_group),
                    "volume_unit_source": f"MW columns multiplied by inferred timestep {timestep_hours:.6g}h; MWh columns used directly",
                    "count_semantics": count_semantics,
                    "volume_semantics": volume_semantics,
                    "metric_semantics": metric_semantics,
                    "coverage_share": coverage_share,
                    "missing_hours_count": missing_hours,
                    "warnings": " | ".join(row_warnings),
                }
            )
    out = pd.DataFrame(rows, columns=columns)
    market_order = {str(spec["market"]): idx for idx, spec in enumerate(BIDDING_ACTIVITY_SUBMITTED_CLEARED_SPECS)}
    model_order = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    if not out.empty:
        out["_market_order"] = out["market"].map(market_order).fillna(999)
        out["_model_order"] = out["model"].map(model_order).fillna(999)
        out = out.sort_values(["_market_order", "_model_order"]).drop(columns=["_market_order", "_model_order"]).reset_index(drop=True)
    return out, pd.DataFrame(warning_rows), pd.DataFrame(inventory_rows)


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
                        "mean_absolute_error": float(
                            np.mean(
                                np.abs(
                                    pd.to_numeric(d["y_true"], errors="coerce")
                                    - pd.to_numeric(d[q], errors="coerce")
                                )
                            )
                        ),
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
        "mean_absolute_error",
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
    d["mean_absolute_error"] = pd.to_numeric(d.get("mean_absolute_error", np.nan), errors="coerce")
    d["n_obs"] = pd.to_numeric(d["n_obs"], errors="coerce")
    d = d.dropna(subset=["mean_pinball_loss", "mean_absolute_error", "n_obs"])
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
        absolute_errors = group["mean_absolute_error"].to_numpy(dtype=float)
        base = dict(zip(group_cols, keys))
        first = group.iloc[0]
        rec: dict[str, Any] = {
            **base,
            "target": "all_targets",
            "target_label": "All Target Variables",
            "mean_pinball_loss": float(np.average(losses, weights=weights)),
            "mean_absolute_error": float(np.average(absolute_errors, weights=weights)),
            "n_obs": int(np.sum(weights)),
            "n_targets": int(group["target"].nunique()),
        }
        for col in value_cols:
            rec[col] = first.get(col, np.nan)
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out[columns].sort_values(["model", "quantile"]).reset_index(drop=True)


def build_target_normalized_mean_pinball_loss_heatmap_data(scatter_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Aggregate target-level pinball losses after scaling each target to RLQR.

    For each forecast target and quantile policy, the model loss is divided by
    the RLQR loss for the same target and quantile. The resulting relative
    losses are then aggregated across targets.
    """
    columns = [
        "split",
        "model_key",
        "model",
        "quantile",
        "target_normalized_mean_pinball_loss",
        "n_targets",
        "n_obs",
        "aggregation_weighting",
    ]
    omitted_columns = ["split", "target", "target_label", "quantile", "reason", "rlqr_mean_pinball_loss"]
    if scatter_data.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=omitted_columns), "unavailable_empty_input"

    required = {"split", "target", "target_label", "model_key", "model", "quantile", "mean_pinball_loss"}
    missing = sorted(required - set(scatter_data.columns))
    if missing:
        raise ValueError(
            "Cannot build target-normalized mean pinball heatmap; missing required columns: "
            + ", ".join(missing)
        )

    d = scatter_data.copy()
    d["mean_pinball_loss"] = pd.to_numeric(d["mean_pinball_loss"], errors="coerce")
    if "n_obs" in d.columns:
        d["n_obs"] = pd.to_numeric(d["n_obs"], errors="coerce")
        has_weights = d["n_obs"].notna().any() and (d["n_obs"] > 0).any()
    else:
        d["n_obs"] = np.nan
        has_weights = False
    weighting = "observation_weighted" if has_weights else "simple_target_average"

    rlqr = (
        d.loc[d["model"].astype(str).eq("RLQR"), ["split", "target", "target_label", "quantile", "mean_pinball_loss"]]
        .rename(columns={"mean_pinball_loss": "rlqr_mean_pinball_loss"})
        .drop_duplicates(["split", "target", "quantile"])
    )
    merged = d.merge(rlqr, on=["split", "target", "target_label", "quantile"], how="left")
    bad = merged["rlqr_mean_pinball_loss"].isna() | ~np.isfinite(merged["rlqr_mean_pinball_loss"]) | (merged["rlqr_mean_pinball_loss"].abs() <= 1e-12)
    omitted = (
        merged.loc[bad, ["split", "target", "target_label", "quantile", "rlqr_mean_pinball_loss"]]
        .drop_duplicates(["split", "target", "quantile"])
        .copy()
    )
    if not omitted.empty:
        omitted["reason"] = np.where(
            omitted["rlqr_mean_pinball_loss"].isna(),
            "missing_rlqr_denominator",
            "zero_or_nonfinite_rlqr_denominator",
        )
        omitted = omitted[omitted_columns]
    else:
        omitted = pd.DataFrame(columns=omitted_columns)

    valid = merged.loc[~bad].copy()
    valid["relative_pinball_loss"] = valid["mean_pinball_loss"] / valid["rlqr_mean_pinball_loss"]
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna(subset=["relative_pinball_loss"])
    if valid.empty:
        return pd.DataFrame(columns=columns), omitted, weighting

    rows: list[dict[str, Any]] = []
    group_cols = ["split", "model_key", "model", "quantile"]
    for keys, group in valid.groupby(group_cols, dropna=False, observed=True):
        rel = group["relative_pinball_loss"].to_numpy(dtype=float)
        if has_weights:
            weights = pd.to_numeric(group["n_obs"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if float(np.sum(weights)) <= 0:
                value = float(np.nanmean(rel))
                n_obs = math.nan
            else:
                value = float(np.average(rel, weights=weights))
                n_obs = int(np.sum(weights))
        else:
            value = float(np.nanmean(rel))
            n_obs = math.nan
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "target_normalized_mean_pinball_loss": value,
                "n_targets": int(group["target"].nunique()),
                "n_obs": n_obs,
                "aggregation_weighting": weighting,
            }
        )
    out = pd.DataFrame(rows)
    return out[columns].sort_values(["model", "quantile"]).reset_index(drop=True), omitted.reset_index(drop=True), weighting


def build_rlqr_relative_mean_pinball_loss_heatmap_data(scatter_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Build Figure 7 from target-level losses normalized to RLQR first."""
    detail_columns = [
        "split",
        "target",
        "target_label",
        "model_key",
        "model",
        "quantile_policy",
        "raw_mean_pinball_loss",
        "rlqr_mean_pinball_loss",
        "relative_mean_pinball_loss",
        "n_obs",
        "aggregation_level",
    ]
    aggregate_columns = [
        "split",
        "target",
        "target_label",
        "model_key",
        "model",
        "quantile_policy",
        "raw_mean_pinball_loss",
        "rlqr_mean_pinball_loss",
        "relative_mean_pinball_loss",
        "n_obs",
        "n_targets",
        "aggregation_weighting",
        "aggregation_level",
    ]
    if scatter_data.empty:
        return pd.DataFrame(columns=aggregate_columns), pd.DataFrame(columns=detail_columns), "unavailable_empty_input"

    required = {"split", "target", "target_label", "model_key", "model", "quantile", "mean_pinball_loss", "n_obs"}
    missing = sorted(required - set(scatter_data.columns))
    if missing:
        raise ValueError(
            "Cannot build RLQR-relative mean pinball heatmap; missing required columns: "
            + ", ".join(missing)
        )

    d = scatter_data.copy()
    d["mean_pinball_loss"] = pd.to_numeric(d["mean_pinball_loss"], errors="coerce")
    d["n_obs"] = pd.to_numeric(d["n_obs"], errors="coerce")
    rlqr = (
        d.loc[d["model"].astype(str).eq("RLQR"), ["split", "target", "target_label", "quantile", "mean_pinball_loss"]]
        .rename(columns={"mean_pinball_loss": "rlqr_mean_pinball_loss"})
        .drop_duplicates(["split", "target", "quantile"])
    )
    merged = d.merge(rlqr, on=["split", "target", "target_label", "quantile"], how="left")
    bad = merged["rlqr_mean_pinball_loss"].isna() | ~np.isfinite(merged["rlqr_mean_pinball_loss"]) | (merged["rlqr_mean_pinball_loss"].abs() <= 1e-12)
    if bad.any():
        bad_keys = (
            merged.loc[bad, ["split", "target", "target_label", "quantile", "rlqr_mean_pinball_loss"]]
            .drop_duplicates(["split", "target", "quantile"])
            .sort_values(["split", "target", "quantile"])
        )
        details = "; ".join(
            f"{row.split}/{row.target}/{row.quantile}: {row.rlqr_mean_pinball_loss}"
            for row in bad_keys.itertuples(index=False)
        )
        raise ValueError(
            "Cannot build RLQR-relative mean pinball heatmap because RLQR denominators are missing, "
            f"zero, NaN or non-finite for target/quantile combinations: {details}"
        )

    merged["relative_mean_pinball_loss"] = merged["mean_pinball_loss"] / merged["rlqr_mean_pinball_loss"]
    rel_bad = merged["relative_mean_pinball_loss"].isna() | ~np.isfinite(merged["relative_mean_pinball_loss"])
    if rel_bad.any():
        bad_keys = merged.loc[rel_bad, ["split", "target", "target_label", "model", "quantile"]].drop_duplicates()
        details = "; ".join(f"{row.split}/{row.target}/{row.model}/{row.quantile}" for row in bad_keys.itertuples(index=False))
        raise ValueError(f"Cannot build RLQR-relative mean pinball heatmap because relative losses are invalid for: {details}")

    detail = merged.rename(
        columns={
            "quantile": "quantile_policy",
            "mean_pinball_loss": "raw_mean_pinball_loss",
        }
    ).copy()
    detail["aggregation_level"] = "target"
    detail = detail[detail_columns]

    rows: list[dict[str, Any]] = []
    group_cols = ["split", "model_key", "model", "quantile_policy"]
    weighting = "observation_weighted"
    for keys, group in detail.groupby(group_cols, dropna=False, observed=True):
        weights = pd.to_numeric(group["n_obs"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if float(np.sum(weights)) <= 0:
            raise ValueError(
                "Cannot aggregate RLQR-relative mean pinball heatmap because n_obs weights are missing or non-positive for "
                + "/".join(str(x) for x in keys)
            )
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "target": "all_targets",
                "target_label": "All Target Variables",
                "raw_mean_pinball_loss": float(np.average(group["raw_mean_pinball_loss"].to_numpy(dtype=float), weights=weights)),
                "rlqr_mean_pinball_loss": float(np.average(group["rlqr_mean_pinball_loss"].to_numpy(dtype=float), weights=weights)),
                "relative_mean_pinball_loss": float(np.average(group["relative_mean_pinball_loss"].to_numpy(dtype=float), weights=weights)),
                "n_obs": int(np.sum(weights)),
                "n_targets": int(group["target"].nunique()),
                "aggregation_weighting": weighting,
                "aggregation_level": "all_targets_after_normalization",
            }
        )
    aggregate = pd.DataFrame(rows)
    aggregate = aggregate[aggregate_columns].sort_values(["model", "quantile_policy"]).reset_index(drop=True)
    detail = detail.sort_values(["target", "model", "quantile_policy"]).reset_index(drop=True)
    return aggregate, detail, weighting


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
    cols = ["model", "quantile", "realized_profit_eur", "annualized_profit_eur_per_year", "simulation_valid", "thesis_reportable", "invalid_reason"]
    data = summary[cols].copy()
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{RQ2 profit and validity diagnostics by simulation folder.}",
        r"\label{tab:rq2_profit_and_validity_detailed}",
        r"\begin{tabular}{@{}llrrc p{0.42\linewidth}@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Quantile} & \textbf{Profit} & \textbf{Annualized profit} & \textbf{Valid} & \textbf{Invalid reason} \\",
        r"\midrule",
    ]
    for _, row in data.iterrows():
        reason = str(row.get("invalid_reason", "") or "")
        if reason == "none":
            reason = ""
        reason_tex = _latex_escape(reason).replace(",", r",\allowbreak{}")
        cells = [
            _latex_escape(row["model"]),
            _latex_escape(row["quantile"]),
            _format_eur(row["realized_profit_eur"]),
            _format_eur(row["annualized_profit_eur_per_year"]),
            "yes" if _safe_float(row["simulation_valid"]) >= 0.5 and _safe_float(row["thesis_reportable"]) >= 0.5 else "no",
            reason_tex,
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_k_eur_table(value: Any, digits: int = 1) -> str:
    val = _safe_float(value)
    if not math.isfinite(val):
        return r"--"
    return f"{val:,.{digits}f}"


def _bold_latex_cell(value: str) -> str:
    if value == r"--":
        return value
    return r"\textbf{" + value + "}"


def _revenue_cost_table_component_label(component: str) -> str:
    labels = {
        "DA net": "DA net revenue",
        "ID net": "ID net revenue",
        "BCM capacity": "BCM capacity revenue",
        "BCM activation": "BCM-linked BEM activation revenue",
        "BEM activation": "BEM activation revenue",
        "Degradation cost": "Degradation cost",
        "Auxiliary cost": "Auxiliary cost",
        "Transaction cost": "Transaction cost",
        "Penalty cost": "Penalty cost",
        "Terminal SoC repair": "Terminal SoC repair",
    }
    return labels.get(str(component), str(component))


def _revenue_cost_legend_component_label(component: str) -> str:
    if str(component) == "BCM activation":
        return r"\shortstack[l]{BCM-linked BEM\\activation revenue}"
    return _latex_escape(_revenue_cost_table_component_label(component))


def write_revenue_cost_component_table(path: Path, component_data: pd.DataFrame) -> None:
    models = [m for m in MODEL_ORDER if not component_data.empty and m in set(component_data["model"])]
    component_order = [label for _col, label in COMPONENT_COLUMNS if not component_data.empty and label in set(component_data["component"])]
    cost_labels = {display for col, display in COMPONENT_COLUMNS if col in COMPONENT_COST_COLUMNS}
    quantile_by_model = (
        component_data[["model", "quantile"]]
        .dropna()
        .drop_duplicates("model")
        .set_index("model")["quantile"]
        .astype(str)
        .to_dict()
        if not component_data.empty
        else {}
    )
    pivot = (
        component_data.pivot_table(
            index="component",
            columns="model",
            values="annualized_component_value_eur_per_year",
            aggfunc="sum",
        ).reindex(index=component_order, columns=MODEL_ORDER)
        if not component_data.empty and component_order
        else pd.DataFrame(index=component_order, columns=MODEL_ORDER)
    )
    header_cells = [r"\textbf{Component}"]
    for model in models:
        q = quantile_by_model.get(model, "")
        label = (_latex_escape(model) + (r"~" + _latex_escape(q) if q else ""))
        header_cells.append(r"\textbf{" + label + r"}")

    def _row(component: str, *, cost: bool) -> str:
        cells = [_latex_escape(_revenue_cost_table_component_label(component))]
        values: dict[str, float] = {}
        for model in models:
            value = _safe_float(pivot.loc[component, model]) / 1000.0 if component in pivot.index and model in pivot.columns else math.nan
            if cost and math.isfinite(value):
                value = abs(value)
            values[model] = value
        finite_values = [v for v in values.values() if math.isfinite(v)]
        best_value = (min(finite_values) if cost else max(finite_values)) if finite_values else math.nan
        for model in models:
            value = values.get(model, math.nan)
            formatted = _format_k_eur_table(value, 1)
            if math.isfinite(value) and math.isfinite(best_value) and math.isclose(value, best_value, rel_tol=1e-9, abs_tol=1e-9):
                formatted = _bold_latex_cell(formatted)
            cells.append(formatted)
        return " & ".join(cells) + r" \\"

    revenue_components = [c for c in component_order if c not in cost_labels]
    cost_components = [c for c in component_order if c in cost_labels]
    n_cols = len(models) + 1
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Revenue and cost components for each model at its best numeric quantile policy. Values are annualized and reported in kEUR per year. Cost rows are reported as positive cost magnitudes.}",
        r"\label{tab:3_revenue_cost_components_best_quantile}",
        r"\begin{tabular}{@{}l" + "r" * len(models) + r"@{}}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
        rf"\multicolumn{{{n_cols}}}{{@{{}}l}}{{\textbf{{Revenue}}}} \\",
    ]
    lines.extend(_row(component, cost=False) for component in revenue_components)
    lines.extend(
        [
            r"\addlinespace",
            rf"\multicolumn{{{n_cols}}}{{@{{}}l}}{{\textbf{{Costs}}}} \\",
        ]
    )
    lines.extend(_row(component, cost=True) for component in cost_components)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_bid_activity_best_quantile_table(path: Path, bidding_activity_data: pd.DataFrame) -> None:
    market_labels = [str(spec["market_label"]) for spec in BIDDING_ACTIVITY_SUBMITTED_CLEARED_SPECS]
    model_order = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    market_order = {market: idx for idx, market in enumerate(market_labels)}
    data = bidding_activity_data.copy()
    if data.empty:
        rows = []
    else:
        data["_model_order"] = data["model"].map(model_order).fillna(999)
        data["_market_order"] = data["market_label"].map(market_order).fillna(999)
        data = data.sort_values(["_model_order", "_market_order"]).drop(columns=["_model_order", "_market_order"])
        rows = data.to_dict("records")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Submitted and cleared annualized bid volumes at the best quantile strategy of each model. Volumes are shown in kMWh/year.}",
        r"\label{tab:4_bid_activity_best_quantile}",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Market} & \textbf{\shortstack{Submitted\\(kMWh)}} & \textbf{\shortstack{Cleared\\(kMWh)}} & \textbf{Clearing ratio (\%)} \\",
        r"\midrule",
    ]
    for row in rows:
        submitted = _safe_float(row.get("submitted_bid_volume_mwh_annualized")) / 1000.0
        cleared = _safe_float(row.get("cleared_bid_volume_mwh_annualized")) / 1000.0
        ratio = (cleared / submitted * 100.0) if math.isfinite(submitted) and submitted > 0 else math.nan
        cells = [
            _latex_escape(row.get("strategy_label") or f"{row.get('model', '')} {row.get('quantile', '')}".strip()),
            _latex_escape(row.get("market_label", "")),
            _format_k_eur_table(submitted, 1),
            _format_k_eur_table(cleared, 1),
            _format_k_eur_table(ratio, 1),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
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


def plot_mean_pinball_loss_heatmap(relative_pinball_data: pd.DataFrame, out_base: Path, formats: list[str]) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    apply_geo_style()
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    rows = list(MODEL_ORDER)
    quantiles = list(DEFAULT_QUANTILES)
    if not relative_pinball_data.empty and "quantile_policy" in relative_pinball_data.columns:
        quantiles = [q for q in DEFAULT_QUANTILES if q in set(relative_pinball_data["quantile_policy"].astype(str))]
    pivot = (
        relative_pinball_data.pivot_table(
            index="model",
            columns="quantile_policy",
            values="relative_mean_pinball_loss",
            aggfunc="mean",
        ).reindex(index=rows, columns=quantiles)
        if not relative_pinball_data.empty
        else pd.DataFrame(index=rows, columns=quantiles)
    )
    arr = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(arr)
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
    ax.set_ylabel("Model")
    finite = arr[np.isfinite(arr)]
    midpoint = float(np.nanmedian(finite)) if finite.size else math.nan
    for i, model in enumerate(rows):
        for j, q in enumerate(quantiles):
            val = _safe_float(pivot.loc[model, q]) if model in pivot.index and q in pivot.columns else math.nan
            txt = "n/a" if not math.isfinite(val) else f"{val:.2f}"
            text_color = "white" if math.isfinite(val) and math.isfinite(midpoint) and val >= midpoint else THESIS_PALETTE["neutral_dark"]
            ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.grid(False)
    cbar.set_label("Relative mean pinball loss")
    fig.tight_layout()
    written = _save_figure(fig, out_base, formats)
    plt.close(fig)
    return written


def plot_target_normalized_mean_pinball_loss_heatmap(data: pd.DataFrame, out_base: Path, formats: list[str]) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    apply_geo_style()
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    rows = list(MODEL_ORDER)
    quantiles = list(DEFAULT_QUANTILES)
    if not data.empty and "quantile" in data.columns:
        quantiles = [q for q in DEFAULT_QUANTILES if q in set(data["quantile"].astype(str))]
    pivot = (
        data.pivot_table(
            index="model",
            columns="quantile",
            values="target_normalized_mean_pinball_loss",
            aggfunc="mean",
        ).reindex(index=rows, columns=quantiles)
        if not data.empty
        else pd.DataFrame(index=rows, columns=quantiles)
    )
    arr = pivot.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    span = max(float(np.max(np.abs(finite - 1.0))), 0.10) if finite.size else 0.10
    norm = TwoSlopeNorm(vmin=1.0 - span, vcenter=1.0, vmax=1.0 + span)
    cmap = LinearSegmentedColormap.from_list(
        "rq2_relative_pinball",
        [THESIS_PALETTE["primary"], "#FFFFFF", THESIS_PALETTE["tertiary"]],
    )
    cmap.set_bad("#F2F2F2")
    im = ax.imshow(np.ma.masked_invalid(arr), aspect="auto", cmap=cmap, norm=norm)
    ax.grid(False)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks(np.arange(len(quantiles)))
    ax.set_xticklabels(quantiles)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Quantile policy")
    ax.set_ylabel("Model")
    for i, model in enumerate(rows):
        for j, q in enumerate(quantiles):
            val = _safe_float(pivot.loc[model, q]) if model in pivot.index and q in pivot.columns else math.nan
            txt = "n/a" if not math.isfinite(val) else f"{val:.2f}"
            text_color = "white" if math.isfinite(val) and abs(val - 1.0) >= 0.55 * span else THESIS_PALETTE["neutral_dark"]
            ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.grid(False)
    cbar.set_label("Mean pinball loss relative to RLQR")
    fig.tight_layout()
    written = _save_figure(fig, out_base, formats)
    plt.close(fig)
    return written


def plot_bid_volume_heatmap(activity: pd.DataFrame, out_base: Path, formats: list[str], *, metric: str) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    apply_geo_style()
    if metric not in {"submitted", "cleared"}:
        raise ValueError(f"Unsupported bid-volume heatmap metric: {metric}")
    value_col = f"{metric}_bid_volume_mwh_annualized"
    metric_label = "Submitted" if metric == "submitted" else "Cleared"
    models = list(MODEL_ORDER)
    quantiles = list(DEFAULT_QUANTILES)
    markets = [str(spec["market_label"]) for spec in BIDDING_ACTIVITY_SUBMITTED_CLEARED_SPECS]
    fig, axes_arr = plt.subplots(2, 2, figsize=(9.8, 6.8), sharey=True)
    axes = axes_arr.ravel()
    cmap = LinearSegmentedColormap.from_list(
        "geo_sequential_blue",
        [GEO_SEQUENTIAL_BLUE[f"seq_{i}"] for i in range(1, 8)],
    )
    cmap.set_bad("#F2F2F2")
    all_values: list[float] = []
    pivots: dict[str, pd.DataFrame] = {}
    for market in markets:
        d = activity.loc[activity["market_label"].astype(str).eq(market)].copy() if not activity.empty else pd.DataFrame()
        pivot = (
            d.pivot_table(
                index="model",
                columns="quantile",
                values=value_col,
                aggfunc="sum",
            ).reindex(index=models, columns=quantiles) / 1000.0
            if not d.empty
            else pd.DataFrame(index=models, columns=quantiles)
        )
        pivots[market] = pivot
        all_values.extend([float(v) for v in pivot.to_numpy(dtype=float).ravel() if math.isfinite(float(v))])
    vmax = max(all_values) if all_values else 1.0
    vmin = 0.0
    last_im = None
    for ax, market in zip(axes, markets):
        pivot = pivots[market]
        arr = pivot.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(arr)
        last_im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.grid(False)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks(np.arange(len(quantiles)))
        ax.set_xticklabels(quantiles, rotation=0)
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels(models)
        ax.set_xlabel("Quantile")
        for i, model in enumerate(models):
            for j, q in enumerate(quantiles):
                val = _safe_float(pivot.loc[model, q]) if model in pivot.index and q in pivot.columns else math.nan
                txt = "n/a" if not math.isfinite(val) else f"{val:,.0f}"
                text_color = "white" if math.isfinite(val) and vmax > 0 and val >= 0.55 * vmax else THESIS_PALETTE["neutral_dark"]
                ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=8)
    for ax in axes[len(markets):]:
        ax.axis("off")
    axes[0].set_ylabel("Model")
    axes[2].set_ylabel("Model")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=list(axes), shrink=0.88)
        cbar.ax.grid(False)
        cbar.set_label(f"{metric_label} bid volume (GWh/year)")
    fig.tight_layout(rect=(0, 0, 0.94, 0.94))
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

    def _label_position(model: str, quantile: str) -> tuple[int, int, str]:
        if model == "RLQR" and quantile in {"p10", "p30", "p70", "p90"}:
            return (0, -16, "center")
        if model == "RLQR" and quantile == "p50":
            return (0, -16, "center")
        if model == "XGB" and quantile == "p90":
            return (0, 8, "center")
        if model == "TFT" and quantile == "p10":
            return (0, 8, "center")
        if model == "TFT" and quantile == "p90":
            return (8, 0, "left")
        if model == "Naive":
            return (0, -14, "center")
        if model == "RHPF":
            return (0, 8, "center")
        return (0, 8, "center")

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
            dx, dy, ha = _label_position(model, q)
            ax.annotate(
                f"{yi/1000:,.0f} kEUR",
                (xi, yi),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=ha,
                va="center",
                fontsize=8,
                color=THESIS_PALETTE["neutral_dark"],
            )
    ax.set_xticks(x)
    ax.set_xticklabels(quantiles)
    ax.set_ylabel("Annualized Net Profit (EUR/year)")
    ax.set_xlabel("Quantile policy")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=True)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    written = _save_figure(fig, out_base, formats)
    plt.close(fig)
    return written


def plot_best_quantile_components(component_data: pd.DataFrame, out_base: Path, formats: list[str], run_name: str) -> list[Path]:
    import matplotlib.pyplot as plt

    apply_geo_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
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
        fig.tight_layout()
        written = _save_figure(fig, out_base, formats)
        plt.close(fig)
        return written

    models = [m for m in MODEL_ORDER if m in set(component_data["model"])]
    x = np.arange(len(models))
    width = 0.72
    component_order = [label for _col, label in COMPONENT_COLUMNS if label in set(component_data["component"])]
    pivot = component_data.pivot_table(
        index="model",
        columns="component",
        values="annualized_component_value_eur_per_year",
        aggfunc="sum",
    ).reindex(index=models, columns=component_order).fillna(0.0) / 1000.0
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
    ax.set_ylabel("Annualized component value (kEUR/year)")
    ax.set_xlabel("Model and best quantile")
    handles, labels = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    revenue_labels = [label for label in component_order if label not in {display for _col, display in COMPONENT_COLUMNS if _col in COMPONENT_COST_COLUMNS}]
    cost_labels = [label for _col, label in COMPONENT_COLUMNS if _col in COMPONENT_COST_COLUMNS and label in handle_by_label]
    from matplotlib.patches import Patch

    empty = Patch(facecolor="none", edgecolor="none")
    legend_handles = [empty, empty, empty, empty]
    legend_labels = ["Revenue", " ", "Costs", " "]
    legend_rows = max(math.ceil(len(revenue_labels) / 2), math.ceil(len(cost_labels) / 2))
    for row_idx in range(legend_rows):
        for group in (revenue_labels, cost_labels):
            row_labels = group[2 * row_idx : 2 * row_idx + 2]
            legend_handles.extend([handle_by_label[label] for label in row_labels])
            legend_labels.extend(row_labels)
            legend_handles.extend([empty] * (2 - len(row_labels)))
            legend_labels.extend([" "] * (2 - len(row_labels)))
    ax.legend(
        legend_handles,
        legend_labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    written = _save_figure(fig, out_base, formats)
    plt.close(fig)
    return written


def plot_bidding_activity_submitted_cleared(
    activity: pd.DataFrame,
    out_base: Path,
    formats: list[str],
    *,
    metric: str,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    apply_geo_style()
    is_volume = metric == "volume"
    cleared_col = "cleared_bid_volume_mwh_annualized" if is_volume else "cleared_bid_count_annualized"
    not_cleared_col = "not_cleared_bid_volume_mwh_annualized" if is_volume else "not_cleared_bid_count_annualized"
    scale = 1000.0
    title = "Annualized Bid Volume by Market and Model Strategy" if is_volume else "Annualized Bid Count by Market and Model Strategy"
    ylabel = "Submitted bid volume (GWh/year)" if is_volume else "Submitted bids (1,000/year)"
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    if activity.empty:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No bidding-activity data available for selected model strategies.",
            ha="center",
            va="center",
            fontsize=11,
            color=THESIS_PALETTE["neutral_dark"],
            wrap=True,
        )
        fig.tight_layout()
        written = _save_figure(fig, out_base, formats)
        plt.close(fig)
        return written

    market_labels = [str(spec["market_label"]) for spec in BIDDING_ACTIVITY_SUBMITTED_CLEARED_SPECS]
    models = [m for m in MODEL_ORDER if m in set(activity["model"].astype(str))]
    x = np.arange(len(market_labels))
    width = 0.26
    offsets = {model: (idx - (len(models) - 1) / 2) * width for idx, model in enumerate(models)}
    color_map = {"RLQR": get_model_color("linear"), "XGB": get_model_color("xgb"), "TFT": get_model_color("tft")}
    for model in models:
        cleared = (
            activity.pivot_table(index="market_label", columns="model", values=cleared_col, aggfunc="sum")
            .reindex(index=market_labels, columns=models)
            .get(model, pd.Series(index=market_labels, dtype=float))
        )
        not_cleared = (
            activity.pivot_table(index="market_label", columns="model", values=not_cleared_col, aggfunc="sum")
            .reindex(index=market_labels, columns=models)
            .get(model, pd.Series(index=market_labels, dtype=float))
        )
        cleared_values = pd.to_numeric(cleared, errors="coerce").to_numpy(dtype=float) / scale
        not_cleared_values = pd.to_numeric(not_cleared, errors="coerce").to_numpy(dtype=float) / scale
        color = color_map.get(model, THESIS_PALETTE["neutral_dark"])
        label = f"{model} {str(activity.loc[activity['model'].astype(str).eq(model), 'quantile'].iloc[0])}"
        ax.bar(x + offsets[model], cleared_values, width=width, label=label, color=color, edgecolor="white", linewidth=0.6, alpha=1.0)
        ax.bar(x + offsets[model], not_cleared_values, width=width, bottom=cleared_values, color=color, edgecolor="white", linewidth=0.6, alpha=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(market_labels)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.35)
    model_handles = [Patch(facecolor=color_map[m], edgecolor="white", label=f"{m} {str(activity.loc[activity['model'].astype(str).eq(m), 'quantile'].iloc[0])}") for m in models]
    status_handles = [
        Patch(facecolor=THESIS_PALETTE["neutral_dark"], alpha=1.0, edgecolor="white", label="Cleared bids"),
        Patch(facecolor=THESIS_PALETTE["neutral_dark"], alpha=0.35, edgecolor="white", label="Uncleared bids"),
    ]
    ax.legend(handles=[*model_handles, *status_handles], ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
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
        "BCM negative activation": "BCM-linked BEM activation revenue",
        "BCM positive activation": "BCM-linked BEM activation revenue",
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
    raw_min = min(0.0, float(np.nanmin(neg_bottom)) if len(neg_bottom) else 0.0)
    raw_max = max(0.0, float(np.nanmax(pos_bottom)) if len(pos_bottom) else 0.0)
    span = max(raw_max - raw_min, 1.0)
    pad = max(0.25, 0.08 * span)
    power_ymin = math.floor((raw_min - pad) * 2.0) / 2.0
    power_ymax = math.ceil((raw_max + pad) * 2.0) / 2.0

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
    ax.set_ylim(power_ymin, power_ymax)
    if len(x) > 0:
        ax.set_xlim(0, len(x) - 1)
    ax.set_ylabel("Battery dispatch power (MW)")
    selected_date = pd.Timestamp(times[0]).date() if times else None
    selected_date_label = f"{selected_date.day} {MONTH_ABBR_FULL[selected_date.month]} {selected_date.year}" if selected_date is not None else ""
    ax.set_xlabel(f"Time ({selected_date_label})" if selected_date_label else "Time")
    labels = [pd.Timestamp(ts).strftime("%H:%M") for ts in times]
    ax.set_xticks(x[:: max(1, len(x) // 8)])
    ax.set_xticklabels(labels[:: max(1, len(x) // 8)])
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
    plot_dates = pd.Series(cumulative["timestamp_utc"]).dropna().dt.date.sort_values().unique().tolist()
    tick_positions = _select_spaced_tick_positions(len(plot_dates), max_ticks=7, min_gap=5)
    if tick_positions:
        tick_dates = [pd.Timestamp(plot_dates[i]) for i in tick_positions]
        ax.set_xticks(tick_dates)
        ax.set_xticklabels([d.strftime("%d %b") for d in tick_dates])
    ax.set_ylabel("Cumulative Net Profit (kEUR)")
    ax.set_xlabel("Time (2025)")
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


def build_normalized_total_mae_profit_data(total_scatter_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize total selected-quantile MAE and profit to [0, 1]."""
    if total_scatter_data.empty or "mean_absolute_error" not in total_scatter_data.columns:
        return pd.DataFrame()
    df = total_scatter_data.copy()
    df["mean_absolute_error"] = pd.to_numeric(df["mean_absolute_error"], errors="coerce")
    df["annualized_profit_eur_per_year"] = pd.to_numeric(df["annualized_profit_eur_per_year"], errors="coerce")
    df = df.dropna(subset=["mean_absolute_error", "annualized_profit_eur_per_year"]).copy()
    if df.empty:
        return pd.DataFrame()

    loss_min = float(df["mean_absolute_error"].min())
    loss_max = float(df["mean_absolute_error"].max())
    profit_min = float(df["annualized_profit_eur_per_year"].min())
    profit_max = float(df["annualized_profit_eur_per_year"].max())
    loss_range = loss_max - loss_min
    profit_range = profit_max - profit_min
    df["normalized_forecast_loss"] = 0.0 if math.isclose(loss_range, 0.0) else (df["mean_absolute_error"] - loss_min) / loss_range
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

    def _label_position(model: str, quantile: str) -> tuple[int, int, str, str]:
        if model == "XGB" and quantile == "p70":
            return (-8, 0, "right", "center")
        if model == "XGB" and quantile == "p90":
            return (8, 0, "left", "center")
        if model == "RLQR" and quantile == "p30":
            return (-8, 0, "right", "center")
        if model == "RLQR" and quantile == "p50":
            return (0, 8, "center", "bottom")
        if model == "TFT" and quantile == "p90":
            return (0, -8, "center", "top")
        return (4, 3, "left", "bottom")

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
            label="Loss-profit reference line",
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
                dx, dy, ha, va = _label_position(model, str(row["quantile"]))
                ax.annotate(
                    str(row["quantile"]),
                    (row["normalized_forecast_loss"], row["normalized_annualized_net_profit"]),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    ha=ha,
                    va=va,
                    fontsize=8,
                    color=THESIS_PALETTE["neutral_dark"],
                )
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("Normalized total mean pinball loss")
        ax.set_ylabel("Normalized annualized net profit")
        ax.legend(title="Series", loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=8, frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    written = _save_figure(fig, out_base, formats)
    plt.close(fig)
    return written


def plot_normalized_total_mae_profit_scatter(normalized_data: pd.DataFrame, out_base: Path, formats: list[str]) -> list[Path]:
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
            "No normalized total MAE / Net Profit data available.",
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
            label="Loss-profit reference line",
            zorder=1,
        )

        def _mae_label_position(model: str, quantile: str) -> tuple[int, int, str, str]:
            key = (str(model), str(quantile).lower())
            if key in {("XGB", "p70"), ("XGB", "p10")}:
                return 0, 5, "center", "bottom"
            if key in {("TFT", "p90"), ("TFT", "p50")}:
                return 0, -5, "center", "top"
            return 4, 3, "left", "bottom"

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
                dx, dy, ha, va = _mae_label_position(model, str(row["quantile"]))
                ax.annotate(
                    str(row["quantile"]),
                    (row["normalized_forecast_loss"], row["normalized_annualized_net_profit"]),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    ha=ha,
                    va=va,
                    fontsize=8,
                    color=THESIS_PALETTE["neutral_dark"],
                )
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("Normalized total MAE")
        ax.set_ylabel("Normalized annualized net profit")
        ax.legend(title="Series", loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=8, frameon=True)
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
                rf"\caption{{{_latex_escape(_ensure_caption_period(caption))}}}",
                rf"\label{{{label}}}",
                r"\end{figure}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _hex_to_rgb01(color: str) -> tuple[float, float, float]:
    color = str(color).strip().lstrip("#")
    if len(color) != 6:
        return (0.0, 0.0, 0.0)
    return (int(color[0:2], 16) / 255.0, int(color[2:4], 16) / 255.0, int(color[4:6], 16) / 255.0)


def _tex_color_def(name: str, color: str) -> str:
    r, g, b = _hex_to_rgb01(color)
    return rf"\definecolor{{{name}}}{{rgb}}{{{r:.4f},{g:.4f},{b:.4f}}}"


def _rq2_latex_color_defs() -> list[str]:
    return [
        _tex_color_def("rqTwoNeutral", THESIS_PALETTE["neutral_dark"]),
        _tex_color_def("rqTwoGrid", THESIS_PALETTE.get("grid", "#D8D8D8")),
        _tex_color_def("rqTwoNaive", THESIS_PALETTE["naive"]),
        _tex_color_def("rqTwoRHPF", TRUTH_REFERENCE_COLOR),
        _tex_color_def("rqTwoRLQR", get_model_color("linear")),
        _tex_color_def("rqTwoXGB", get_model_color("xgb")),
        _tex_color_def("rqTwoTFT", get_model_color("tft")),
        _tex_color_def("rqTwoDA", MARKET_COLOR_MAP["DA"]),
        _tex_color_def("rqTwoID", MARKET_COLOR_MAP["ID"]),
        _tex_color_def("rqTwoBCMCapacity", MARKET_COLOR_MAP["BCM capacity"]),
        _tex_color_def("rqTwoBCMActivation", MARKET_COLOR_MAP["BCM activation"]),
        _tex_color_def("rqTwoBEM", MARKET_COLOR_MAP["BEM"]),
        _tex_color_def("rqTwoPNL", GEO_SEQUENTIAL_BLUE["seq_7"]),
        _tex_color_def("rqTwoSoC", THESIS_PALETTE.get("perfect_foresight", "#2E7D32")),
        _tex_color_def("rqTwoReference", GEO_SEQUENTIAL_BLUE["seq_4"]),
        _tex_color_def("rqTwoCostDegradation", MARKET_COMPONENT_COLORS["Degradation cost"]),
        _tex_color_def("rqTwoCostAuxiliary", MARKET_COMPONENT_COLORS["Auxiliary cost"]),
        _tex_color_def("rqTwoCostTransaction", MARKET_COMPONENT_COLORS["Transaction cost"]),
        _tex_color_def("rqTwoCostPenalty", MARKET_COMPONENT_COLORS["Penalty cost"]),
        _tex_color_def("rqTwoCostTerminal", MARKET_COMPONENT_COLORS["Terminal SoC repair"]),
    ]


def _tex_model_color(model: str) -> str:
    return {
        "Naive": "rqTwoNaive",
        "RHPF": "rqTwoRHPF",
        "RLQR": "rqTwoRLQR",
        "XGB": "rqTwoXGB",
        "TFT": "rqTwoTFT",
    }.get(str(model), "rqTwoNeutral")


def _tex_component_color(component: str) -> str:
    return {
        "DA net": "rqTwoDA",
        "ID net": "rqTwoID",
        "BCM capacity": "rqTwoBCMCapacity",
        "BCM activation": "rqTwoBCMActivation",
        "BEM activation": "rqTwoBEM",
        "Degradation cost": "rqTwoCostDegradation",
        "Auxiliary cost": "rqTwoCostAuxiliary",
        "Transaction cost": "rqTwoCostTransaction",
        "Penalty cost": "rqTwoCostPenalty",
        "Terminal SoC repair": "rqTwoCostTerminal",
    }.get(str(component), "rqTwoNeutral")


def _tex_float(value: Any, digits: int = 3) -> str:
    number = _safe_float(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def _tex_k_eur(value: Any, digits: int = 1) -> str:
    number = _safe_float(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number / 1000.0:.{digits}f}"


def _tex_k_eur_label(value: Any) -> str:
    number = _safe_float(value)
    if not math.isfinite(number):
        return "--"
    return f"{number:,.0f}"


def _select_spaced_tick_positions(n_items: int, *, max_ticks: int = 7, min_gap: int = 5) -> list[int]:
    """Select readable tick positions while keeping the final date if possible."""
    if n_items <= 0:
        return []
    if n_items <= max_ticks:
        return list(range(n_items))
    positions = sorted({int(round(x)) for x in np.linspace(0, n_items - 1, max_ticks)})
    if positions[0] != 0:
        positions.insert(0, 0)
    if positions[-1] != n_items - 1:
        positions.append(n_items - 1)

    cleaned: list[int] = []
    for pos in positions:
        if cleaned and pos - cleaned[-1] < min_gap:
            if pos == n_items - 1:
                cleaned[-1] = pos
            continue
        cleaned.append(pos)
    return cleaned


def _write_native_latex(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _axis_common_options() -> list[str]:
    return [
        "tick align=outside,",
        "axis line style={rqTwoNeutral},",
        "tick style={rqTwoNeutral},",
        "label style={font=\\small},",
        "tick label style={font=\\small},",
        "title style={font=\\normalfont\\small},",
        "legend style={font=\\small, draw=none, fill=none},",
        "grid=major,",
        "grid style={rqTwoGrid!55, line width=0.2pt},",
    ]


def write_latex_profit_heatmap(path: Path, table: pd.DataFrame) -> None:
    rows = ["Naive", "RHPF", *MODEL_ORDER]
    quantiles = table["quantile"].astype(str).tolist() if "quantile" in table.columns else list(DEFAULT_QUANTILES)
    points: list[tuple[int, int, float, str, str]] = []
    vals: list[float] = []
    for yi, row_label in enumerate(rows):
        for xi, q in enumerate(quantiles):
            match = table.loc[table["quantile"].astype(str).eq(q)] if "quantile" in table.columns else pd.DataFrame()
            value = _safe_float(match[row_label].iloc[0]) if not match.empty and row_label in match.columns else math.nan
            if math.isfinite(value):
                vals.append(value / 1000.0)
            txt = "n/a" if not math.isfinite(value) else f"{value / 1000.0:,.0f}"
            text_color = "white" if row_label == "RHPF" else "rqTwoNeutral"
            points.append((xi, yi, value / 1000.0 if math.isfinite(value) else math.nan, txt, text_color))
    meta_min = min(vals) if vals else 0.0
    meta_max = max(vals) if vals else 1.0
    coordinates = "\n".join(
        f"({xi},{yi}) [{_tex_float(meta, 3)}]" for xi, yi, meta, _txt, _tc in points if math.isfinite(meta)
    )
    nodes = "\n".join(
        rf"\node[text={tc}, font=\scriptsize] at (axis cs:{xi},{yi}) {{{_latex_escape(txt)}}};"
        for xi, yi, _meta, txt, tc in points
    )
    lines = [
        r"% Requires \usepackage{pgfplots}",
        r"% Requires \pgfplotsset{compat=1.18}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\begin{tikzpicture}",
        *_rq2_latex_color_defs(),
        r"\begin{axis}[",
        *_axis_common_options(),
        r"width=\linewidth,",
        r"height=0.52\linewidth,",
        r"xlabel={Quantile policy},",
        r"ylabel={Strategy},",
        "xmin=-0.5, xmax=" + _tex_float(len(quantiles) - 0.5, 1) + ",",
        "ymin=-0.5, ymax=" + _tex_float(len(rows) - 0.5, 1) + ",",
        "xtick={" + ",".join(str(i) for i in range(len(quantiles))) + "},",
        "xticklabels={" + ",".join(_latex_escape(q) for q in quantiles) + "},",
        "ytick={" + ",".join(str(i) for i in range(len(rows))) + "},",
        "yticklabels={" + ",".join(_latex_escape(r) for r in rows) + "},",
        r"y dir=reverse,",
        r"point meta min=" + _tex_float(meta_min, 2) + ",",
        r"point meta max=" + _tex_float(meta_max, 2) + ",",
        r"colormap={rq2blue}{rgb255(0cm)=(228,241,247); rgb255(1cm)=(197,225,239); rgb255(2cm)=(158,201,226); rgb255(3cm)=(108,176,214); rgb255(4cm)=(60,147,194); rgb255(5cm)=(34,110,156); rgb255(6cm)=(13,74,112)},",
        r"colorbar,",
        r"colorbar style={ylabel={Annualized Net Profit (kEUR/year)}, tick label style={font=\small}},",
        r"]",
        r"\addplot[scatter, only marks, mark=square*, mark size=18pt, scatter/use mapped color={fill=mapped color, draw=white}] coordinates {",
        coordinates,
        r"};",
        nodes,
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\caption{Net Profit by strategy and quantile policy. Cell values are annualized Net Profit in kEUR per year; validity flags are reported in the source CSV.}",
        r"\label{fig:1_profit_heatmap}",
        r"\end{figure}",
    ]
    _write_native_latex(path, lines)


def write_latex_mean_pinball_loss_heatmap(path: Path, relative_pinball_data: pd.DataFrame, quantiles: list[str]) -> None:
    rows = list(MODEL_ORDER)
    quantile_order = [q for q in quantiles if relative_pinball_data.empty or q in set(relative_pinball_data["quantile_policy"].astype(str))]
    if not quantile_order:
        quantile_order = list(quantiles)
    pivot = (
        relative_pinball_data.pivot_table(
            index="model",
            columns="quantile_policy",
            values="relative_mean_pinball_loss",
            aggfunc="mean",
        ).reindex(index=rows, columns=quantile_order)
        if not relative_pinball_data.empty
        else pd.DataFrame(index=rows, columns=quantile_order)
    )
    points: list[tuple[int, int, float, str, str]] = []
    vals: list[float] = []
    palette = [GEO_SEQUENTIAL_BLUE[f"seq_{i}"] for i in range(1, 8)]
    for yi, model in enumerate(rows):
        for xi, q in enumerate(quantile_order):
            value = _safe_float(pivot.loc[model, q]) if model in pivot.index and q in pivot.columns else math.nan
            if math.isfinite(value):
                vals.append(value)
            txt = "n/a" if not math.isfinite(value) else f"{value:.2f}"
            points.append((xi, yi, value, txt, "rqTwoNeutral"))
    meta_min = min(vals) if vals else 0.0
    meta_max = max(vals) if vals else 1.0
    midpoint = float(np.nanmedian(vals)) if vals else math.nan

    def _cell_color(meta: float) -> str:
        if not math.isfinite(meta):
            return "#F2F2F2"
        if meta_max <= meta_min:
            idx = 3
        else:
            idx = int(round((meta - meta_min) / (meta_max - meta_min) * (len(palette) - 1)))
        idx = max(0, min(len(palette) - 1, idx))
        return palette[idx]

    cells: list[str] = []
    for xi, yi, meta, txt, tc in points:
        color = _cell_color(meta)
        text_color = "white" if math.isfinite(meta) and math.isfinite(midpoint) and meta >= midpoint else tc
        cells.extend(
            [
                rf"\definecolor{{rqTwoHeatCell{xi}{yi}}}{{HTML}}{{{color.lstrip('#')}}}",
                rf"\filldraw[fill=rqTwoHeatCell{xi}{yi}, draw=white, line width=0.6pt] (axis cs:{xi - 0.5},{yi - 0.5}) rectangle (axis cs:{xi + 0.5},{yi + 0.5});",
                rf"\node[text={text_color}, font=\scriptsize] at (axis cs:{xi},{yi}) {{{_latex_escape(txt)}}};",
            ]
        )
    lines = [
        r"% Requires \usepackage{pgfplots}",
        r"% Requires \pgfplotsset{compat=1.18}",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        *_rq2_latex_color_defs(),
        r"\begin{axis}[",
        *_axis_common_options(),
        r"width=0.78\linewidth,",
        r"height=0.32\linewidth,",
        r"xlabel={Quantile policy},",
        r"ylabel={Model},",
        "xmin=-0.5, xmax=" + _tex_float(len(quantile_order) - 0.5, 1) + ",",
        "ymin=-0.5, ymax=" + _tex_float(len(rows) - 0.5, 1) + ",",
        "xtick={" + ",".join(str(i) for i in range(len(quantile_order))) + "},",
        "xticklabels={" + ",".join(_latex_escape(q) for q in quantile_order) + "},",
        "ytick={" + ",".join(str(i) for i in range(len(rows))) + "},",
        "yticklabels={" + ",".join(_latex_escape(r) for r in rows) + "},",
        r"y dir=reverse,",
        r"point meta min=" + _tex_float(meta_min, 2) + ",",
        r"point meta max=" + _tex_float(meta_max, 2) + ",",
        r"colormap={rq2blue}{rgb255(0cm)=(228,241,247); rgb255(1cm)=(197,225,239); rgb255(2cm)=(158,201,226); rgb255(3cm)=(108,176,214); rgb255(4cm)=(60,147,194); rgb255(5cm)=(34,110,156); rgb255(6cm)=(13,74,112)},",
        r"colorbar,",
        r"colorbar style={ylabel={Relative mean pinball loss}, tick label style={font=\small}},",
        r"]",
        r"\addplot[scatter, only marks, mark=none, draw=none, opacity=0, scatter/use mapped color={draw opacity=0, fill opacity=0}] coordinates {",
        rf"(0,0) [{_tex_float(meta_min, 3)}]",
        rf"(0,0) [{_tex_float(meta_max, 3)}]",
        r"};",
        *cells,
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\caption{Mean pinball loss relative to RLQR by model and quantile policy. Values below 1 indicate lower pinball loss than RLQR; values above 1 indicate higher loss.}",
        r"\label{fig:7_mean_pinball_loss_heatmap}",
        r"\end{figure}",
    ]
    _write_native_latex(path, lines)


def write_latex_target_normalized_mean_pinball_loss_heatmap(path: Path, data: pd.DataFrame, quantiles: list[str]) -> None:
    rows = list(MODEL_ORDER)
    quantile_order = [q for q in quantiles if data.empty or q in set(data["quantile"].astype(str))]
    if not quantile_order:
        quantile_order = list(quantiles)
    pivot = (
        data.pivot_table(
            index="model",
            columns="quantile",
            values="target_normalized_mean_pinball_loss",
            aggfunc="mean",
        ).reindex(index=rows, columns=quantile_order)
        if not data.empty
        else pd.DataFrame(index=rows, columns=quantile_order)
    )
    arr = pivot.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    span = max(float(np.max(np.abs(finite - 1.0))), 0.10) if finite.size else 0.10
    vmin = 1.0 - span
    vmax = 1.0 + span
    cells: list[str] = []
    for yi, model in enumerate(rows):
        for xi, q in enumerate(quantile_order):
            value = _safe_float(pivot.loc[model, q]) if model in pivot.index and q in pivot.columns else math.nan
            if not math.isfinite(value):
                fill = "rqTwoMissing"
                text_color = "rqTwoNeutral"
                intensity = 0
                txt = "n/a"
            else:
                intensity = int(round(min(100.0, abs(value - 1.0) / max(span, 1e-12) * 100.0)))
                fill = f"rqTwoBetter!{intensity}!white" if value < 1.0 else f"rqTwoWorse!{intensity}!white"
                text_color = "white" if intensity >= 62 else "rqTwoNeutral"
                txt = f"{value:.2f}"
            cells.extend(
                [
                    rf"\filldraw[fill={fill}, draw=white, line width=0.6pt] (axis cs:{xi - 0.5},{yi - 0.5}) rectangle (axis cs:{xi + 0.5},{yi + 0.5});",
                    rf"\node[text={text_color}, font=\scriptsize] at (axis cs:{xi},{yi}) {{{_latex_escape(txt)}}};",
                ]
            )
    lines = [
        r"% Requires \usepackage{pgfplots}",
        r"% Requires \pgfplotsset{compat=1.18}",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        *_rq2_latex_color_defs(),
        r"\definecolor{rqTwoBetter}{HTML}{226E9C}",
        r"\definecolor{rqTwoWorse}{HTML}{7C1D6F}",
        r"\definecolor{rqTwoMissing}{HTML}{F2F2F2}",
        r"\begin{axis}[",
        *_axis_common_options(),
        r"width=0.78\linewidth,",
        r"height=0.32\linewidth,",
        r"xlabel={Quantile policy},",
        r"ylabel={Model},",
        "xmin=-0.5, xmax=" + _tex_float(len(quantile_order) - 0.5, 1) + ",",
        "ymin=-0.5, ymax=" + _tex_float(len(rows) - 0.5, 1) + ",",
        "xtick={" + ",".join(str(i) for i in range(len(quantile_order))) + "},",
        "xticklabels={" + ",".join(_latex_escape(q) for q in quantile_order) + "},",
        "ytick={" + ",".join(str(i) for i in range(len(rows))) + "},",
        "yticklabels={" + ",".join(_latex_escape(r) for r in rows) + "},",
        r"y dir=reverse,",
        "point meta min=" + _tex_float(vmin, 3) + ",",
        "point meta max=" + _tex_float(vmax, 3) + ",",
        (
            r"colormap={rq2relpinball}{rgb255(0cm)=(34,110,156); "
            r"rgb255(1cm)=(255,255,255); rgb255(2cm)=(124,29,111)},"
        ),
        r"colorbar,",
        r"colorbar style={ylabel={Mean pinball loss relative to RLQR}, tick label style={font=\small}},",
        r"]",
        r"\addplot[scatter, only marks, mark=none, draw=none, opacity=0, scatter/use mapped color={draw opacity=0, fill opacity=0}] coordinates {",
        rf"(0,0) [{_tex_float(vmin, 3)}]",
        rf"(0,0) [{_tex_float(vmax, 3)}]",
        r"};",
        *cells,
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\caption{Target-normalized mean pinball loss by model and quantile policy. Pinball loss is first scaled relative to the RLQR baseline within each forecast target and quantile policy and is then averaged across targets. Values below 1 indicate lower pinball loss than RLQR, while values above 1 indicate higher pinball loss.}",
        r"\label{fig:8_target_normalized_mean_pinball_loss_heatmap}",
        r"\end{figure}",
    ]
    _write_native_latex(path, lines)


def write_latex_bid_volume_heatmap(path: Path, activity: pd.DataFrame, quantiles: list[str], *, metric: str) -> None:
    if metric not in {"submitted", "cleared"}:
        raise ValueError(f"Unsupported bid-volume heatmap metric: {metric}")
    value_col = f"{metric}_bid_volume_mwh_annualized"
    metric_label = "Submitted" if metric == "submitted" else "Cleared"
    rows = list(MODEL_ORDER)
    markets = [str(spec["market_label"]) for spec in BIDDING_ACTIVITY_SUBMITTED_CLEARED_SPECS]
    quantile_order = [q for q in quantiles if q in set(activity["quantile"].astype(str))] if not activity.empty else list(quantiles)
    if not quantile_order:
        quantile_order = list(quantiles)
    panels: list[tuple[str, pd.DataFrame]] = []
    vals: list[float] = []
    for market in markets:
        d = activity.loc[activity["market_label"].astype(str).eq(market)].copy() if not activity.empty else pd.DataFrame()
        pivot = (
            d.pivot_table(
                index="model",
                columns="quantile",
                values=value_col,
                aggfunc="sum",
            ).reindex(index=rows, columns=quantile_order) / 1000.0
            if not d.empty
            else pd.DataFrame(index=rows, columns=quantile_order)
        )
        panels.append((market, pivot))
        vals.extend([float(v) for v in pivot.to_numpy(dtype=float).ravel() if math.isfinite(float(v))])
    panels_by_market = {market: pivot for market, pivot in panels}
    panel_order = [m for m in ["DA", "aFRR capacity", "ID", "aFRR activation"] if m in panels_by_market]
    panel_order.extend([m for m in markets if m in panels_by_market and m not in panel_order])
    panels = [(market, panels_by_market[market]) for market in panel_order]
    meta_min = 0.0
    meta_max = max(vals) if vals else 1.0
    palette = [GEO_SEQUENTIAL_BLUE[f"seq_{i}"] for i in range(1, 8)]

    def _cell_color(value: float) -> str:
        if not math.isfinite(value):
            return "#F2F2F2"
        if meta_max <= meta_min:
            idx = 3
        else:
            idx = int(round((value - meta_min) / (meta_max - meta_min) * (len(palette) - 1)))
        idx = max(0, min(len(palette) - 1, idx))
        return palette[idx]

    panel_blocks: list[str] = []
    positions = [(0, 0), (6.35, 0), (0, -5.25), (6.35, -5.25)]
    for panel_idx, (market, pivot) in enumerate(panels):
        x0, y0 = positions[panel_idx]
        block: list[str] = [
            rf"\begin{{scope}}[shift={{({x0:.2f},{y0:.2f})}}]",
            rf"\node[font=\normalsize] at (2.50,0.55) {{{_latex_escape(market)}}};",
            r"\draw[rqTwoNeutral, line width=0.35pt] (-0.05,-0.05) rectangle (5.05,-3.05);",
        ]
        for xi, q in enumerate(quantile_order):
            block.append(rf"\node[font=\small] at ({xi + 0.5:.2f},-3.38) {{{_latex_escape(q)}}};")
        block.append(r"\node[font=\small] at (2.50,-3.85) {Quantile};")
        if panel_idx in {0, 2}:
            for yi, model in enumerate(rows):
                block.append(rf"\node[anchor=east, font=\small] at (-0.20,{-yi - 0.5:.2f}) {{{_latex_escape(model)}}};")
        for yi, model in enumerate(rows):
            for xi, q in enumerate(quantile_order):
                value = _safe_float(pivot.loc[model, q]) if model in pivot.index and q in pivot.columns else math.nan
                txt = "n/a" if not math.isfinite(value) else f"{value:,.0f}"
                color = _cell_color(value)
                text_color = "white" if math.isfinite(value) and meta_max > 0 and value >= 0.55 * meta_max else "rqTwoNeutral"
                cname = f"rqTwoBidCell{metric}{panel_idx}{xi}{yi}"
                block.extend(
                    [
                        rf"\definecolor{{{cname}}}{{HTML}}{{{color.lstrip('#')}}}",
                        rf"\filldraw[fill={cname}, draw=white, line width=0.45pt] ({xi:.2f},{-yi:.2f}) rectangle ({xi + 1:.2f},{-yi - 1:.2f});",
                        rf"\node[text={text_color}, font=\scriptsize] at ({xi + 0.5:.2f},{-yi - 0.5:.2f}) {{{_latex_escape(txt)}}};",
                    ]
                )
        block.append(r"\end{scope}")
        panel_blocks.extend(block)

    colorbar_x = 12.25
    colorbar_y_top = 0.0
    colorbar_height = 8.25
    colorbar_width = 0.34
    colorbar_lines: list[str] = []
    for idx, color in enumerate(palette):
        y1 = colorbar_y_top - colorbar_height * idx / len(palette)
        y2 = colorbar_y_top - colorbar_height * (idx + 1) / len(palette)
        cname = f"rqTwoBidColorbar{metric}{idx}"
        colorbar_lines.extend(
            [
                rf"\definecolor{{{cname}}}{{HTML}}{{{color.lstrip('#')}}}",
                rf"\filldraw[fill={cname}, draw=none] ({colorbar_x:.2f},{y1:.3f}) rectangle ({colorbar_x + colorbar_width:.2f},{y2:.3f});",
            ]
        )
    colorbar_lines.append(
        rf"\draw[rqTwoNeutral, line width=0.35pt] ({colorbar_x:.2f},{colorbar_y_top:.3f}) rectangle ({colorbar_x + colorbar_width:.2f},{colorbar_y_top - colorbar_height:.3f});"
    )
    tick_values = np.linspace(meta_min, meta_max, 5)
    for tick in tick_values:
        frac = 0.0 if meta_max <= meta_min else (float(tick) - meta_min) / (meta_max - meta_min)
        y = colorbar_y_top - colorbar_height * frac
        tick_label = f"{float(tick):.0f}"
        colorbar_lines.extend(
            [
                rf"\draw[rqTwoNeutral, line width=0.25pt] ({colorbar_x + colorbar_width:.2f},{y:.3f}) -- ({colorbar_x + colorbar_width + 0.12:.2f},{y:.3f});",
                rf"\node[anchor=west, font=\scriptsize] at ({colorbar_x + colorbar_width + 0.18:.2f},{y:.3f}) {{{_latex_escape(tick_label)}}};",
            ]
        )
    colorbar_lines.append(
        rf"\node[rotate=90, anchor=center, font=\small] at ({colorbar_x + colorbar_width + 1.05:.2f},{colorbar_y_top - colorbar_height / 2:.3f}) {{{metric_label} bid volume (GWh/year)}};"
    )

    lines = [
        r"% Requires \usepackage{tikz}",
        r"% Requires \usepackage{pgfplots}",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}[x=0.96cm,y=0.88cm]",
        *_rq2_latex_color_defs(),
        *panel_blocks,
        *colorbar_lines,
        r"\end{tikzpicture}",
        rf"\caption{{{metric_label} bid volume by market, model and quantile policy. Cell values are annualized {metric.lower()} bid volumes in GWh/year.}}",
        rf"\label{{fig:8_{metric}_bid_volume_heatmap_by_market_model_quantile}}",
        r"\end{figure}",
    ]
    _write_native_latex(path, lines)


def write_latex_quantile_sweep(path: Path, sweep_data: pd.DataFrame, quantiles: list[str]) -> None:
    x_by_q = {q: i for i, q in enumerate(quantiles)}
    plotted_values: list[float] = []
    if not sweep_data.empty and "annualized_net_profit_eur_per_year" in sweep_data.columns:
        plotted_values = [
            _safe_float(v) / 1000.0
            for v in sweep_data["annualized_net_profit_eur_per_year"]
            if math.isfinite(_safe_float(v))
        ]
    y_min = min([0.0] + plotted_values) if plotted_values else 0.0
    y_max = max(plotted_values) if plotted_values else 1.0
    y_pad = max(60.0, 0.12 * (y_max - y_min))
    y_max_plot = y_max + y_pad
    below_point = ("north", "0pt", "-6pt")
    label_offsets = {
        ("RLQR", "p10"): below_point,
        ("RLQR", "p30"): below_point,
        ("RLQR", "p50"): below_point,
        ("RLQR", "p70"): below_point,
        ("RLQR", "p90"): below_point,
        ("XGB", "p90"): ("south", "0pt", "5pt"),
        ("TFT", "p10"): ("south", "0pt", "5pt"),
        ("TFT", "p90"): ("west", "6pt", "0pt"),
    }
    model_plot_options = {
        "RLQR": r"color=rqTwoRLQR, mark=*, mark options={solid, draw=rqTwoRLQR, fill=rqTwoRLQR}, line width=1.8pt",
        "XGB": r"color=rqTwoXGB, mark=square*, mark options={solid, draw=rqTwoXGB, fill=rqTwoXGB}, line width=1.8pt",
        "TFT": r"color=rqTwoTFT, mark=triangle*, mark options={solid, draw=rqTwoTFT, fill=rqTwoTFT}, line width=1.8pt",
    }
    lines = [
        r"% Requires \usepackage{pgfplots}",
        r"% Requires \pgfplotsset{compat=1.18}",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        *_rq2_latex_color_defs(),
        r"\begin{axis}[",
        *_axis_common_options(),
        r"width=0.92\linewidth,",
        r"height=0.56\linewidth,",
        r"xlabel={Quantile policy},",
        r"ylabel={Annualized Net Profit (kEUR)},",
        r"ymin=" + _tex_float(y_min, 2) + ",",
        r"ymax=" + _tex_float(y_max_plot, 2) + ",",
        r"clip=false,",
        r"xtick={" + ",".join(str(i) for i in range(len(quantiles))) + "},",
        r"xticklabels={" + ",".join(_latex_escape(q) for q in quantiles) + "},",
        r"legend columns=5,",
        r"legend style={at={(0.5,1.04)}, anchor=south, legend columns=5, draw=none, fill=none, font=\small},",
        r"]",
    ]
    for model in ["Naive", "RHPF"]:
        g = sweep_data.loc[sweep_data["series"].astype(str).eq(model)].copy() if not sweep_data.empty else pd.DataFrame()
        if g.empty:
            continue
        q = "p50" if "p50" in x_by_q else quantiles[len(quantiles) // 2]
        match = g.loc[g["quantile"].astype(str).eq(q)]
        if match.empty:
            match = g.head(1)
        value = _safe_float(match["annualized_net_profit_eur_per_year"].iloc[0]) / 1000.0
        if not math.isfinite(value):
            continue
        lines += [
            rf"\addplot+[color={_tex_model_color(model)}, mark=none, line width=1.6pt] coordinates {{(0,{_tex_float(value, 2)}) ({len(quantiles)-1},{_tex_float(value, 2)})}};",
            rf"\addlegendentry{{{model}}}",
        ]
        anchor = "north east" if model == "RHPF" else "south east"
        yshift = "-3pt" if model == "RHPF" else "3pt"
        lines.append(
            rf"\node[font=\scriptsize, fill=white, fill opacity=0.85, text opacity=1, inner sep=1pt, anchor={anchor}, yshift={yshift}] "
            rf"at (axis cs:{len(quantiles)-1},{_tex_float(value, 2)}) {{{model} {_tex_k_eur_label(value)}}};"
        )
    for model in MODEL_ORDER:
        g = sweep_data.loc[sweep_data["series"].astype(str).eq(model)].copy() if not sweep_data.empty else pd.DataFrame()
        if g.empty:
            continue
        coords = []
        labels: list[tuple[str, float, float]] = []
        for q in quantiles:
            match = g.loc[g["quantile"].astype(str).eq(q)]
            if match.empty:
                continue
            value = _safe_float(match["annualized_net_profit_eur_per_year"].iloc[0]) / 1000.0
            if math.isfinite(value):
                coords.append(f"({x_by_q[q]},{_tex_float(value, 2)})")
                labels.append((q, float(x_by_q[q]), value))
        if not coords:
            continue
        lines += [
            rf"\addplot+[{model_plot_options.get(model, 'color=rqTwoNeutral, mark=*, line width=1.8pt')}] coordinates {{{' '.join(coords)}}};",
            rf"\addlegendentry{{{model}}}",
        ]
        for q, x_pos, value in labels:
            anchor, xshift, yshift = label_offsets.get((model, q), ("south", "0pt", "5pt"))
            lines.append(
                rf"\node[font=\scriptsize, text=rqTwoNeutral, fill=white, fill opacity=0.85, text opacity=1, inner sep=1pt, anchor={anchor}, xshift={xshift}, yshift={yshift}] "
                rf"at (axis cs:{_tex_float(x_pos, 1)},{_tex_float(value, 2)}) {{{_tex_k_eur_label(value)}}};"
            )
    lines += [
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\caption{Quantile sweep of annualized net profit by model and benchmark. Naive and RHPF are shown as benchmark reference lines.}",
        r"\label{fig:2_quantile_sweep_net_profit_by_model}",
        r"\end{figure}",
    ]
    _write_native_latex(path, lines)


def write_latex_revenue_cost_components(path: Path, component_data: pd.DataFrame) -> None:
    models = list(MODEL_ORDER) if not component_data.empty else []
    component_order = [label for _col, label in COMPONENT_COLUMNS if not component_data.empty and label in set(component_data["component"])]
    cost_labels = {display for col, display in COMPONENT_COLUMNS if col in COMPONENT_COST_COLUMNS}
    group_gap = 1.75
    pair_gap = 0.55
    x_positions = {model: (group_gap * idx, group_gap * idx + pair_gap) for idx, model in enumerate(models)}
    tick_positions = [pos for model in models for pos in x_positions[model]]
    x_ticks = ",".join(_tex_float(i, 2) for i in tick_positions)
    x_tick_labels = ",".join(["Revenue", "Cost"] * len(models))
    extra_x_ticks = ",".join(_tex_float(sum(x_positions[model]) / 2.0, 2) for model in models)
    quantile_by_model = (
        component_data[["model", "quantile"]]
        .dropna()
        .drop_duplicates("model")
        .set_index("model")["quantile"]
        .astype(str)
        .to_dict()
        if not component_data.empty
        else {}
    )
    extra_x_tick_labels = ",".join(_latex_escape(f"{model} {quantile_by_model.get(model, '')}".strip()) for model in models)
    lines = [
        r"% Requires \usepackage{pgfplots}",
        r"% Requires \pgfplotsset{compat=1.18}",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        *_rq2_latex_color_defs(),
        r"\begin{axis}[",
        *_axis_common_options(),
        r"width=0.88\linewidth,",
        r"height=0.44\linewidth,",
        r"ybar stacked,",
        r"bar width=22pt,",
        r"ylabel={Annualized component value (kEUR/year)},",
        r"ymin=0,",
        r"enlarge x limits=0.12,",
        r"xtick={" + x_ticks + "},",
        r"xticklabels={" + x_tick_labels + "},",
        r"extra x ticks={" + extra_x_ticks + "},",
        r"extra x tick labels={" + extra_x_tick_labels + "},",
        r"extra x tick style={tick label style={yshift=-1.6em, font=\small}, tick style={draw=none}},",
        r"legend columns=4,",
        r"legend cell align=left,",
        r"legend style={at={(0.5,-0.24)}, anchor=north, font=\scriptsize, draw=none, fill=none, /tikz/every even column/.append style={column sep=0.45cm}},",
        r"]",
    ]
    if not component_data.empty and models and component_order:
        pivot = component_data.pivot_table(
            index="model",
            columns="component",
            values="annualized_component_value_eur_per_year",
            aggfunc="sum",
        ).reindex(index=MODEL_ORDER, columns=component_order).fillna(0.0)
        revenue_components = [c for c in component_order if c not in cost_labels]
        cost_components = [c for c in component_order if c in cost_labels]
        ordered = revenue_components + cost_components
        for component in ordered:
            coords_parts: list[str] = []
            for model in models:
                profit_x, cost_x = x_positions[model]
                raw_value = _safe_float(pivot.loc[model, component]) / 1000.0
                if component in cost_labels:
                    profit_value = 0.0
                    cost_value = abs(raw_value) if math.isfinite(raw_value) else 0.0
                elif math.isfinite(raw_value) and raw_value < 0.0:
                    profit_value = 0.0
                    cost_value = abs(raw_value)
                else:
                    profit_value = raw_value if math.isfinite(raw_value) else 0.0
                    cost_value = 0.0
                coords_parts.append(f"({_tex_float(profit_x, 2)},{_tex_float(profit_value, 2)})")
                coords_parts.append(f"({_tex_float(cost_x, 2)},{_tex_float(cost_value, 2)})")
            if len(coords_parts) != 2 * len(models):
                raise ValueError(
                    f"Grouped stacked revenue/cost LaTeX component {component!r} has {len(coords_parts)} coordinates "
                    f"but expected {2 * len(models)} coordinates."
                )
            coords = " ".join(coords_parts)
            lines += [
                rf"\addplot+[fill={_tex_component_color(component)}, draw=white, forget plot] coordinates {{{coords}}};",
            ]
        lines += [
            r"\addlegendimage{empty legend}",
            r"\addlegendentry{Revenue}",
            r"\addlegendimage{empty legend}",
            r"\addlegendentry{}",
            r"\addlegendimage{empty legend}",
            r"\addlegendentry{Costs}",
            r"\addlegendimage{empty legend}",
            r"\addlegendentry{}",
        ]
        legend_rows = max(math.ceil(len(revenue_components) / 2), math.ceil(len(cost_components) / 2))
        for row_idx in range(legend_rows):
            for components in (revenue_components, cost_components):
                row_components = components[2 * row_idx : 2 * row_idx + 2]
                for component in row_components:
                    lines += [
                        rf"\addlegendimage{{area legend, fill={_tex_component_color(component)}, draw=white}}",
                        rf"\addlegendentry{{{_revenue_cost_legend_component_label(component)}}}",
                    ]
                for _ in range(2 - len(row_components)):
                    lines += [
                        r"\addlegendimage{empty legend}",
                        r"\addlegendentry{}",
                    ]
    lines += [
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\caption{Revenue and cost components for each model at its best numeric quantile policy. Profit components and cost/loss components are shown as adjacent stacked bars. Values are annualized and reported in kEUR per year.}",
        r"\label{fig:3_revenue_cost_components_best_quantile}",
        r"\end{figure}",
    ]
    _write_native_latex(path, lines)


def write_latex_cumulative_pnl(path: Path, cumulative: pd.DataFrame) -> None:
    data = cumulative.copy()
    if not data.empty:
        data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], errors="coerce", utc=True)
        data = data.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").copy()
        data["date"] = data["timestamp_utc"].dt.date
        daily = data.sort_values("timestamp_utc").groupby(["model", "series", "date"], as_index=False).tail(1)
        dates = sorted(daily["date"].unique())
    else:
        daily = pd.DataFrame()
        dates = []
    date_index = {d: i for i, d in enumerate(dates)}
    tick_positions = _select_spaced_tick_positions(len(dates), max_ticks=7, min_gap=5)
    tick_labels = [pd.Timestamp(dates[i]).strftime("%d %b") for i in tick_positions] if dates else []
    lines = [
        r"% Requires \usepackage{pgfplots}",
        r"% Requires \pgfplotsset{compat=1.18}",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        *_rq2_latex_color_defs(),
        r"\begin{axis}[",
        *_axis_common_options(),
        r"width=\linewidth,",
        r"height=0.56\linewidth,",
        r"xlabel={Date},",
        r"ylabel={Cumulative Net Profit (kEUR)},",
        r"xmin=0,",
        r"xmax=" + _tex_float(max(len(dates) - 1, 0), 1) + ",",
        r"xtick={" + ",".join(str(i) for i in tick_positions) + "},",
        r"xticklabels={" + ",".join(_latex_escape(x) for x in tick_labels) + "},",
        r"legend columns=3,",
        r"legend cell align=left,",
        r"legend style={at={(0.02,0.98)}, anchor=north west, font=\small, draw=none, fill=none, /tikz/every even column/.append style={column sep=0.35cm}},",
        r"]",
    ]
    for model in ["Naive", "RHPF", *MODEL_ORDER]:
        g = daily.loc[daily["model"].astype(str).eq(model)].sort_values("date") if not daily.empty else pd.DataFrame()
        if g.empty:
            continue
        coords = " ".join(
            f"({date_index[row['date']]},{_tex_float(_safe_float(row['cum_pnl_eur']) / 1000.0, 2)})"
            for _, row in g.iterrows()
            if row["date"] in date_index and math.isfinite(_safe_float(row["cum_pnl_eur"]))
        )
        if not coords:
            continue
        label = str(g["series"].iloc[0])
        lines += [
            rf"\addplot+[color={_tex_model_color(model)}, mark=none, line width=1.9pt] coordinates {{{coords}}};",
            rf"\addlegendentry{{{_latex_escape(label)}}}",
        ]
    lines += [
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\caption{Cumulative Net Profit over the test period for Naive, RHPF and each model at its best numeric quantile policy.}",
        r"\label{fig:4_cumulative_net_profit_model_comparison_test_period}",
        r"\end{figure}",
    ]
    _write_native_latex(path, lines)


def _forecast_profit_correlation_caption(data: pd.DataFrame, *, x_col: str, label: str) -> str:
    if data.empty or x_col not in data.columns or "normalized_annualized_net_profit" not in data.columns:
        return ""
    d = data[[x_col, "normalized_annualized_net_profit"]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d["normalized_annualized_net_profit"] = pd.to_numeric(d["normalized_annualized_net_profit"], errors="coerce")
    d = d.dropna()
    if len(d) < 2:
        return ""
    x = d[x_col].to_numpy(dtype=float)
    y = d["normalized_annualized_net_profit"].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or math.isclose(float(np.std(x)), 0.0) or math.isclose(float(np.std(y)), 0.0):
        return ""
    pearson = float(pd.Series(x).corr(pd.Series(y), method="pearson"))
    spearman = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    return rf" Descriptive correlations for {label} versus profit are $r = {pearson:.2f}$ and $\rho = {spearman:.2f}$."


def write_latex_normalized_pinball_profit(path: Path, normalized_data: pd.DataFrame) -> None:
    correlation_caption = _forecast_profit_correlation_caption(
        normalized_data,
        x_col="normalized_forecast_loss",
        label="normalized total mean pinball loss",
    )
    lines = [
        r"% Requires \usepackage{pgfplots}",
        r"% Requires \pgfplotsset{compat=1.18}",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        *_rq2_latex_color_defs(),
        r"\begin{axis}[",
        *_axis_common_options(),
        r"label style={font=\normalsize},",
        r"tick label style={font=\normalsize},",
        r"width=0.76\linewidth,",
        r"height=0.54\linewidth,",
        r"xlabel={Normalized total mean pinball loss},",
        r"ylabel={Normalized annualized net profit},",
        r"xmin=-0.04, xmax=1.04, ymin=-0.04, ymax=1.04,",
        r"legend columns=4,",
        r"legend style={at={(0.5,1.04)}, anchor=south, font=\normalsize, draw=none, fill=none},",
        r"]",
        r"\addplot+[color=rqTwoNaive, mark=none, dashed, line width=1.2pt] coordinates {(0,1) (1,0)};",
        r"\addlegendentry{Loss-profit reference line}",
    ]
    for model in MODEL_ORDER:
        g = normalized_data.loc[normalized_data["model"].astype(str).eq(model)].copy() if not normalized_data.empty else pd.DataFrame()
        if g.empty:
            continue
        marker = {"RLQR": "*", "XGB": "square*", "TFT": "triangle*"}.get(model, "*")
        model_color = _tex_model_color(model)
        coords = " ".join(
            f"({_tex_float(row['normalized_forecast_loss'], 4)},{_tex_float(row['normalized_annualized_net_profit'], 4)})"
            for _, row in g.iterrows()
        )
        lines += [
            rf"\addplot+[only marks, color={model_color}, mark={marker}, mark size=2.4pt, mark options={{draw={model_color}, fill={model_color}}}] coordinates {{{coords}}};",
            rf"\addlegendentry{{{model}}}",
        ]
        for _, row in g.iterrows():
            quantile = str(row["quantile"])
            node_options = "font=\\small, anchor=west, text=rqTwoNeutral"
            if model == "XGB" and quantile == "p70":
                node_options = "font=\\small, anchor=east, xshift=-3pt, text=rqTwoNeutral"
            elif model == "XGB" and quantile == "p90":
                node_options = "font=\\small, anchor=west, xshift=3pt, text=rqTwoNeutral"
            elif model == "RLQR" and quantile == "p30":
                node_options = "font=\\small, anchor=east, xshift=-3pt, text=rqTwoNeutral"
            elif model == "RLQR" and quantile == "p50":
                node_options = "font=\\small, anchor=south, yshift=3pt, text=rqTwoNeutral"
            elif model == "TFT" and quantile == "p90":
                node_options = "font=\\small, anchor=north, yshift=-3pt, text=rqTwoNeutral"
            lines.append(
                rf"\node[{node_options}] at (axis cs:{_tex_float(row['normalized_forecast_loss'], 4)},{_tex_float(row['normalized_annualized_net_profit'], 4)}) {{{_latex_escape(quantile)}}};"
            )
    lines += [
        r"\end{axis}",
        r"\end{tikzpicture}",
        rf"\caption{{Normalized total mean pinball loss versus annualized Net Profit by model-quantile policy; the dashed line marks the loss-profit reference.{correlation_caption}}}",
        r"\label{fig:5_pinball_loss_vs_net_profit_total_normalized}",
        r"\end{figure}",
    ]
    _write_native_latex(path, lines)


def write_latex_normalized_mae_profit(path: Path, normalized_data: pd.DataFrame) -> None:
    correlation_caption = _forecast_profit_correlation_caption(
        normalized_data,
        x_col="normalized_forecast_loss",
        label="normalized MAE p50",
    )

    def _mae_node_options(model: str, quantile: str) -> str:
        key = (str(model), str(quantile).lower())
        if key in {("XGB", "p70"), ("XGB", "p10")}:
            return "font=\\small, anchor=south, yshift=3pt, text=rqTwoNeutral"
        if key == ("RLQR", "p50"):
            return "font=\\small, anchor=east, xshift=-2pt, text=rqTwoNeutral"
        if key in {("TFT", "p90"), ("TFT", "p50")}:
            return "font=\\small, anchor=north, yshift=-3pt, text=rqTwoNeutral"
        return "font=\\small, anchor=west, xshift=2pt, text=rqTwoNeutral"

    lines = [
        r"% Requires \usepackage{pgfplots}",
        r"% Requires \pgfplotsset{compat=1.18}",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        *_rq2_latex_color_defs(),
        r"\begin{axis}[",
        *_axis_common_options(),
        r"label style={font=\normalsize},",
        r"tick label style={font=\normalsize},",
        r"width=0.82\linewidth,",
        r"height=0.62\linewidth,",
        r"xlabel={Normalized total MAE},",
        r"ylabel={Normalized annualized net profit},",
        r"xmin=-0.04, xmax=1.04, ymin=-0.04, ymax=1.04,",
        r"legend columns=4,",
        r"legend style={at={(0.5,-0.18)}, anchor=north, font=\normalsize, draw=none, fill=none},",
        r"]",
        r"\addplot+[color=rqTwoNaive, mark=none, dashed, line width=1.2pt] coordinates {(0,1) (1,0)};",
        r"\addlegendentry{Loss-profit reference line}",
    ]
    for model in MODEL_ORDER:
        g = normalized_data.loc[normalized_data["model"].astype(str).eq(model)].copy() if not normalized_data.empty else pd.DataFrame()
        if g.empty:
            continue
        marker = {"RLQR": "*", "XGB": "square*", "TFT": "triangle*"}.get(model, "*")
        model_color = _tex_model_color(model)
        coords = " ".join(
            f"({_tex_float(row['normalized_forecast_loss'], 4)},{_tex_float(row['normalized_annualized_net_profit'], 4)})"
            for _, row in g.iterrows()
        )
        lines += [
            rf"\addplot+[only marks, color={model_color}, mark={marker}, mark size=2.4pt, mark options={{draw={model_color}, fill={model_color}}}] coordinates {{{coords}}};",
            rf"\addlegendentry{{{model}}}",
        ]
        for _, row in g.iterrows():
            quantile = str(row["quantile"])
            node_options = _mae_node_options(model, quantile)
            lines.append(
                rf"\node[{node_options}] at (axis cs:{_tex_float(row['normalized_forecast_loss'], 4)},{_tex_float(row['normalized_annualized_net_profit'], 4)}) {{{_latex_escape(quantile)}}};"
            )
    lines += [
        r"\end{axis}",
        r"\end{tikzpicture}",
        rf"\caption{{Normalized total MAE versus annualized Net Profit by model-quantile policy; the dashed line marks the loss-profit reference.{correlation_caption}}}",
        r"\label{fig:5_mae_vs_net_profit_total_normalized}",
        r"\end{figure}",
    ]
    _write_native_latex(path, lines)


def write_latex_market_dispatch_soc(path: Path, dispatch_data: pd.DataFrame) -> None:
    data = dispatch_data.copy()
    if not data.empty:
        data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], errors="coerce", utc=True)
        data = data.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").copy()
    times = sorted(data["timestamp_utc"].dropna().unique()) if not data.empty else []
    x_by_time = {ts: i for i, ts in enumerate(times)}
    selected_date = pd.Timestamp(times[0]).date() if times else None
    selected_date_label = f"{selected_date.day} {MONTH_ABBR_FULL[selected_date.month]} {selected_date.year}" if selected_date else ""
    components = [
        ("DA buy", "rqTwoDA", "DA buy/sell", False),
        ("DA sell", "rqTwoDA", "DA buy/sell", True),
        ("BEM negative activation", "rqTwoBEM", "BEM activation +/-", False),
        ("BEM positive activation", "rqTwoBEM", "BEM activation +/-", True),
        ("BCM negative activation", "rqTwoBCMActivation", "BCM-linked BEM activation revenue", False),
        ("BCM positive activation", "rqTwoBCMActivation", "BCM-linked BEM activation revenue", True),
    ]
    tick_step = 4
    tick_positions = list(range(0, len(times), tick_step)) if times else []
    if times and tick_positions[-1] != len(times) - 1:
        tick_positions.append(len(times) - 1)
    tick_labels = [pd.Timestamp(times[i]).strftime("%H:%M") for i in tick_positions] if times else []
    power_ymin = -1.0
    power_ymax = 1.0
    axis_width = r"0.82\linewidth"
    if not data.empty and times:
        stack = (
            data.pivot_table(index="timestamp_utc", columns="component", values="mw_signed", aggfunc="sum")
            .reindex(times)
            .fillna(0.0)
        )
        positive_stack = stack.clip(lower=0.0).sum(axis=1)
        negative_stack = stack.clip(upper=0.0).sum(axis=1)
        raw_min = min(0.0, _safe_float(negative_stack.min()))
        raw_max = max(0.0, _safe_float(positive_stack.max()))
        span = max(raw_max - raw_min, 1.0)
        pad = max(0.25, 0.08 * span)
        power_ymin = math.floor((raw_min - pad) * 2.0) / 2.0
        power_ymax = math.ceil((raw_max + pad) * 2.0) / 2.0
    lines = [
        r"% Requires \usepackage{pgfplots}",
        r"% Requires \pgfplotsset{compat=1.18}",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        *_rq2_latex_color_defs(),
        r"\begin{axis}[",
        *_axis_common_options(),
        r"name=dispatchaxis,",
        r"xshift=-0.18cm,",
        rf"width={axis_width},",
        r"height=0.56\linewidth,",
        r"ybar stacked,",
        r"bar width=7pt,",
        rf"xlabel={{Time ({_latex_escape(selected_date_label)})}},",
        r"ylabel={Battery dispatch power (MW)},",
        r"ylabel style={xshift=0.18cm},",
        rf"ymin={_tex_float(power_ymin, 2)}, ymax={_tex_float(power_ymax, 2)},",
        r"xmin=-0.5, xmax=" + _tex_float(max(len(times) - 0.5, 0.5), 1) + ",",
        r"xtick={" + ",".join(str(i) for i in tick_positions) + "},",
        r"xticklabels={" + ",".join(_latex_escape(x) for x in tick_labels) + "},",
        r"legend columns=3,",
        r"legend style={at={(0.5,-0.20)}, anchor=north, font=\scriptsize, draw=none, fill=none},",
        r"]",
    ]
    legend_seen: set[str] = set()
    for component, color, label, already_seen in components:
        coords: list[str] = []
        g = data.loc[data["component"].astype(str).eq(component)].copy() if not data.empty else pd.DataFrame()
        by_x = {x_by_time[row["timestamp_utc"]]: _safe_float(row["mw_signed"]) for _, row in g.iterrows() if row["timestamp_utc"] in x_by_time}
        for i in range(len(times)):
            coords.append(f"({i},{_tex_float(by_x.get(i, 0.0), 3)})")
        plot_options = f"fill={color}, draw=white"
        if label in legend_seen:
            plot_options += ", forget plot"
        lines.append(rf"\addplot+[{plot_options}] coordinates {{{' '.join(coords)}}};")
        if label not in legend_seen:
            lines.append(rf"\addlegendentry{{{_latex_escape(label)}}}")
            legend_seen.add(label)
    lines += [
        r"\addplot+[black, mark=none, line width=0.8pt, forget plot] coordinates {(0,0) (" + str(max(len(times) - 1, 0)) + r",0)};",
        r"\addlegendimage{color=rqTwoSoC, mark=none, line width=2.0pt}",
        r"\addlegendentry{Charge}",
        r"\addlegendimage{color=rqTwoPNL, mark=none, dashed, line width=1.8pt}",
        r"\addlegendentry{Cumulative Net Profit}",
        r"\end{axis}",
        r"\begin{axis}[",
        r"name=socaxis,",
        r"at={(dispatchaxis.south west)},",
        r"anchor=south west,",
        rf"width={axis_width},",
        r"height=0.56\linewidth,",
        r"axis x line=none,",
        r"axis y line*=right,",
        r"ylabel={Charge (MWh)},",
        r"ylabel style={font=\small, text=rqTwoSoC, at={(axis description cs:1.10,0.50)}, anchor=center},",
        r"yticklabel style={font=\small, text=rqTwoSoC},",
        r"tick style={rqTwoSoC},",
        r"axis line style={rqTwoSoC},",
        r"ymin=2, ymax=18,",
        r"xmin=-0.5, xmax=" + _tex_float(max(len(times) - 0.5, 0.5), 1) + ",",
        r"grid=none,",
        r"]",
    ]
    if times:
        soc = data[["timestamp_utc", "soc_mwh"]].drop_duplicates("timestamp_utc").set_index("timestamp_utc").reindex(times)["soc_mwh"]
        pnl = data[["timestamp_utc", "cumulative_pnl_eur"]].drop_duplicates("timestamp_utc").set_index("timestamp_utc").reindex(times)["cumulative_pnl_eur"] / 1000.0
        soc_coords = " ".join(f"({i},{_tex_float(v, 3)})" for i, v in enumerate(soc.to_numpy(dtype=float)) if math.isfinite(_safe_float(v)))
        pnl_coords = " ".join(f"({i},{_tex_float(v, 3)})" for i, v in enumerate(pnl.to_numpy(dtype=float)) if math.isfinite(_safe_float(v)))
        lines += [
            rf"\addplot+[color=rqTwoSoC, mark=none, line width=2.0pt] coordinates {{{soc_coords}}};",
            r"\end{axis}",
            r"\begin{axis}[",
            r"name=pnlaxis,",
            r"at={(dispatchaxis.south west)},",
            r"anchor=south west,",
            rf"width={axis_width},",
            r"height=0.56\linewidth,",
            r"axis x line=none,",
            r"axis y line*=right,",
            r"ylabel={Cumulative Net Profit (kEUR)},",
            r"ylabel style={font=\small, text=rqTwoPNL, at={(axis description cs:1.19,0.50)}, anchor=center},",
            r"yticklabel style={font=\small, text=rqTwoPNL, xshift=3.8em},",
            r"tick style={rqTwoPNL, xshift=3.5em},",
            r"axis line style={rqTwoPNL, xshift=3.5em},",
            r"xmin=-0.5, xmax=" + _tex_float(max(len(times) - 0.5, 0.5), 1) + ",",
            r"grid=none,",
            r"]",
            rf"\addplot+[color=rqTwoPNL, mark=none, dashed, line width=1.8pt] coordinates {{{pnl_coords}}};",
        ]
    lines += [
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\caption{Market dispatch, state of charge and cumulative Net Profit for the selected TFT p90 example day. Charging actions are stacked above zero and discharging actions below zero.}",
        r"\label{fig:6_market_dispatch_soc_selected_day}",
        r"\end{figure}",
    ]
    _write_native_latex(path, lines)


def write_latex_bidding_activity_submitted_cleared(path: Path, activity: pd.DataFrame, *, metric: str) -> None:
    is_volume = metric == "volume"
    market_labels = [str(spec["market_label"]) for spec in BIDDING_ACTIVITY_SUBMITTED_CLEARED_SPECS]
    models = [m for m in MODEL_ORDER if not activity.empty and m in set(activity["model"].astype(str))]
    model_labels = {
        model: f"{model} {str(activity.loc[activity['model'].astype(str).eq(model), 'quantile'].iloc[0])}"
        for model in models
    }
    shifts = {"RLQR": "-8pt", "XGB": "0pt", "TFT": "8pt"}
    cleared_col = "cleared_bid_volume_mwh_annualized" if is_volume else "cleared_bid_count_annualized"
    not_cleared_col = "not_cleared_bid_volume_mwh_annualized" if is_volume else "not_cleared_bid_count_annualized"
    ylabel = "Submitted bid volume (GWh/year)" if is_volume else "Submitted bids (1,000/year)"
    caption = (
        "Annualized submitted bid volume by market and model strategy, split into cleared and uncleared bids; labels show clearing ratios."
        if is_volume
        else "Annualized submitted bid count by market and model strategy, split into cleared and uncleared bids."
    )
    label = "fig:annualized_bid_volume_by_market_model" if is_volume else "fig:annualized_bid_count_by_market_model"
    bar_shift = {"RLQR": -0.16, "XGB": 0.0, "TFT": 0.16}
    bar_half_width = 0.070
    max_total = 0.0
    bar_draws: list[str] = []
    ratio_labels: list[tuple[float, float, str]] = []
    full_clearing_by_market: dict[str, list[tuple[float, float, float, float]]] = {label: [] for label in market_labels}
    ratio_label_x_offsets = {
        ("DA", "RLQR"): -0.055,
        ("DA", "TFT"): 0.055,
        ("BEM", "RLQR"): -0.065,
        ("BEM", "TFT"): 0.065,
        ("ID", "RLQR"): -0.055,
        ("ID", "TFT"): 0.055,
    }
    for model in models:
        for xi, market_label in enumerate(market_labels):
            match = activity.loc[activity["market_label"].astype(str).eq(market_label) & activity["model"].astype(str).eq(model)]
            cleared = _safe_float(match[cleared_col].iloc[0]) / 1000.0 if not match.empty else 0.0
            not_cleared = _safe_float(match[not_cleared_col].iloc[0]) / 1000.0 if not match.empty else 0.0
            if not math.isfinite(cleared):
                cleared = 0.0
            if not math.isfinite(not_cleared):
                not_cleared = 0.0
            total = max(0.0, cleared + not_cleared)
            max_total = max(max_total, total)
            x_mid = xi + bar_shift.get(model, 0.0)
            x_left = x_mid - bar_half_width
            x_right = x_mid + bar_half_width
            color = _tex_model_color(model)
            if is_volume and total > 0:
                clearing_ratio = cleared / total
                if math.isfinite(clearing_ratio):
                    if abs(clearing_ratio - 1.0) <= 1e-9:
                        full_clearing_by_market.setdefault(market_label, []).append((x_left, x_right, total, clearing_ratio))
                    else:
                        label_x = x_mid + ratio_label_x_offsets.get((market_label, model), 0.0)
                        ratio_labels.append((label_x, total, f"{clearing_ratio * 100.0:.0f}\\%"))
            if cleared > 0:
                bar_draws.append(
                    rf"\filldraw[fill={color}, draw=white, line width=0.4pt] (axis cs:{_tex_float(x_left, 3)},0) rectangle (axis cs:{_tex_float(x_right, 3)},{_tex_float(cleared, 3)});"
                )
            if not_cleared > 0:
                bar_draws.append(
                    rf"\filldraw[fill={color}, fill opacity=0.35, draw=white, line width=0.4pt] (axis cs:{_tex_float(x_left, 3)},{_tex_float(cleared, 3)}) rectangle (axis cs:{_tex_float(x_right, 3)},{_tex_float(total, 3)});"
                )
    y_max = max(1.0, max_total * (1.24 if is_volume else 1.12))
    label_offset = max(y_max * 0.015, 0.08)
    ratio_label_draws = [
        rf"\node[anchor=south, font=\normalsize, text=rqTwoNeutral] at (axis cs:{_tex_float(x_mid, 3)},{_tex_float(min(y_max * 0.985, total + label_offset), 3)}) {{{label}}};"
        for x_mid, total, label in ratio_labels
    ]
    full_clearing_braces: list[str] = []
    if is_volume:
        for market_label, entries in full_clearing_by_market.items():
            if len(entries) < len(models) or not entries:
                continue
            x_left = min(row[0] for row in entries)
            x_right = max(row[1] for row in entries)
            group_top = max(row[2] for row in entries)
            y = min(y_max * 0.945, group_top + max(y_max * 0.040, 0.18))
            x_mid = 0.5 * (x_left + x_right)
            full_clearing_braces.append(
                rf"\node[anchor=south, font=\normalsize, text=rqTwoNeutral] "
                rf"at (axis cs:{_tex_float(x_mid, 3)},{_tex_float(y, 3)}) {{$\overbrace{{\hspace{{1.55cm}}}}^{{100\%}}$}};"
            )
    lines = [
        r"% Requires \usepackage{pgfplots}",
        r"% Requires \pgfplotsset{compat=1.18}",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        *_rq2_latex_color_defs(),
        r"\begin{axis}[",
        *_axis_common_options(),
        r"width=\linewidth,",
        r"height=0.54\linewidth,",
        rf"ylabel={{{ylabel}}},",
        r"xlabel={},",
        r"xmin=-0.5, xmax=" + _tex_float(max(len(market_labels) - 0.5, 0.5), 1) + ",",
        r"ymin=0, ymax=" + _tex_float(y_max, 3) + ",",
        r"xtick={" + ",".join(str(i) for i in range(len(market_labels))) + "},",
        r"xticklabels={" + ",".join(_latex_escape(label) for label in market_labels) + "},",
        r"legend columns=3,",
        r"legend cell align=left,",
        r"legend style={at={(0.5,-0.18)}, anchor=north, font=\small, draw=none, fill=none, /tikz/every even column/.append style={column sep=0.45cm}},",
        r"]",
    ]
    for model in models:
        lines.append(rf"\addlegendimage{{area legend, fill={_tex_model_color(model)}, draw=white}}")
        lines.append(rf"\addlegendentry{{{_latex_escape(model_labels.get(model, model))}}}")
    lines.extend(bar_draws)
    lines.extend(ratio_label_draws)
    lines.extend(full_clearing_braces)
    lines += [
        r"\addlegendimage{area legend, fill=rqTwoNeutral, draw=white}",
        r"\addlegendentry{Cleared bids}",
        r"\addlegendimage{area legend, fill=rqTwoNeutral, fill opacity=0.35, draw=white}",
        r"\addlegendentry{Uncleared bids}",
        r"\end{axis}",
        r"\end{tikzpicture}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{figure}",
    ]
    _write_native_latex(path, lines)


def write_result_section_native_latex_figures(
    *,
    latex_figures_dir: Path,
    heatmap_table: pd.DataFrame,
    sweep_data: pd.DataFrame,
    component_data: pd.DataFrame,
    cumulative_data: pd.DataFrame,
    relative_pinball_heatmap_data: pd.DataFrame,
    target_normalized_pinball_data: pd.DataFrame,
    normalized_total_scatter_data: pd.DataFrame,
    normalized_total_mae_data: pd.DataFrame,
    dispatch_soc_data: pd.DataFrame,
    bidding_activity_data: pd.DataFrame,
    bidding_heatmap_data: pd.DataFrame,
    quantiles: list[str],
) -> None:
    write_latex_profit_heatmap(latex_figures_dir / "1_profit_heatmap.tex", heatmap_table)
    write_latex_quantile_sweep(latex_figures_dir / "2_quantile_sweep_net_profit_by_model.tex", sweep_data, quantiles)
    write_latex_revenue_cost_components(latex_figures_dir / "3_revenue_cost_components_best_quantile.tex", component_data)
    write_latex_cumulative_pnl(latex_figures_dir / "4_cumulative_net_profit_model_comparison_test_period.tex", cumulative_data)
    write_latex_normalized_pinball_profit(latex_figures_dir / "5_pinball_loss_vs_net_profit_total_normalized.tex", normalized_total_scatter_data)
    write_latex_normalized_mae_profit(latex_figures_dir / "5_mae_vs_net_profit_total_normalized.tex", normalized_total_mae_data)
    write_latex_market_dispatch_soc(latex_figures_dir / "6_market_dispatch_soc_selected_day.tex", dispatch_soc_data)
    write_latex_mean_pinball_loss_heatmap(latex_figures_dir / "7_mean_pinball_loss_heatmap.tex", relative_pinball_heatmap_data, quantiles)
    write_latex_target_normalized_mean_pinball_loss_heatmap(latex_figures_dir / "8_target_normalized_mean_pinball_loss_heatmap.tex", target_normalized_pinball_data, quantiles)
    write_latex_bid_volume_heatmap(latex_figures_dir / "8_submitted_bid_volume_heatmap_by_market_model_quantile.tex", bidding_heatmap_data, quantiles, metric="submitted")
    write_latex_bid_volume_heatmap(latex_figures_dir / "8_cleared_bid_volume_heatmap_by_market_model_quantile.tex", bidding_heatmap_data, quantiles, metric="cleared")
    write_latex_bidding_activity_submitted_cleared(latex_figures_dir / "annualized_bid_volume_by_market_model.tex", bidding_activity_data, metric="volume")
    write_latex_bidding_activity_submitted_cleared(latex_figures_dir / "annualized_bid_count_by_market_model.tex", bidding_activity_data, metric="count")


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
    relative_pinball_heatmap_data, relative_pinball_heatmap_detail, relative_pinball_heatmap_weighting = (
        build_rlqr_relative_mean_pinball_loss_heatmap_data(scatter_data)
    )
    target_normalized_pinball_data, target_normalized_pinball_omissions, target_normalized_pinball_weighting = (
        build_target_normalized_mean_pinball_loss_heatmap_data(scatter_data)
    )
    normalized_total_scatter_data = build_normalized_total_pinball_profit_data(total_scatter_data)
    normalized_total_mae_data = build_normalized_total_mae_profit_data(total_scatter_data)
    dispatch_soc_data, dispatch_soc_warnings = build_market_dispatch_soc_day(summary, run_root=run_root)
    bidding_activity_data, bidding_activity_warnings, bidding_activity_inventory = build_bidding_activity_by_market_model(summary, run_root=run_root, selected_only=True)
    bidding_heatmap_data, bidding_heatmap_warnings, bidding_heatmap_inventory = build_bidding_activity_by_market_model(summary, run_root=run_root, selected_only=False)

    csv_dir = out_root / "backup/csv"
    result_csv_dir = out_root / "result_section/csv"
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
    pd.concat([relative_pinball_heatmap_detail, relative_pinball_heatmap_data], ignore_index=True, sort=False).to_csv(
        result_csv_dir / "7_mean_pinball_loss_heatmap.csv",
        index=False,
    )
    target_normalized_pinball_data.to_csv(result_csv_dir / "8_target_normalized_mean_pinball_loss_heatmap.csv", index=False)
    target_normalized_pinball_omissions.to_csv(diag_dir / "8_target_normalized_mean_pinball_loss_omitted_target_quantiles.csv", index=False)
    normalized_total_scatter_data.to_csv(csv_dir / "5_pinball_loss_vs_net_profit_total_normalized.csv", index=False)
    normalized_total_mae_data.to_csv(csv_dir / "5_mae_vs_net_profit_total_normalized.csv", index=False)
    dispatch_soc_data.to_csv(csv_dir / "6_market_dispatch_soc_selected_day.csv", index=False)
    bidding_activity_data.to_csv(csv_dir / "bidding_activity_submitted_cleared_by_market_model.csv", index=False)
    bidding_activity_data.to_csv(result_csv_dir / "bidding_activity_submitted_cleared_by_market_model.csv", index=False)
    bidding_heatmap_data.to_csv(csv_dir / "8_submitted_bid_volume_heatmap_by_market_model_quantile.csv", index=False)
    bidding_heatmap_data.to_csv(result_csv_dir / "8_submitted_bid_volume_heatmap_by_market_model_quantile.csv", index=False)
    bidding_heatmap_data.to_csv(csv_dir / "8_cleared_bid_volume_heatmap_by_market_model_quantile.csv", index=False)
    bidding_heatmap_data.to_csv(result_csv_dir / "8_cleared_bid_volume_heatmap_by_market_model_quantile.csv", index=False)
    bench.to_csv(csv_dir / "rq2_benchmark_values.csv", index=False)
    inventory.to_csv(diag_dir / "rq2_input_file_inventory.csv", index=False)
    bidding_activity_inventory.to_csv(diag_dir / "bidding_activity_submitted_cleared_source_inventory.csv", index=False)
    bidding_heatmap_inventory.to_csv(diag_dir / "8_submitted_bid_volume_heatmap_source_inventory.csv", index=False)
    validity.to_csv(diag_dir / "rq2_validity_diagnostics.csv", index=False)
    warning_df = pd.concat([warning_df, dispatch_soc_warnings, bidding_activity_warnings, bidding_heatmap_warnings], ignore_index=True, sort=False)
    if not target_normalized_pinball_omissions.empty:
        warning_df = pd.concat(
            [
                warning_df,
                target_normalized_pinball_omissions.assign(
                    severity="warning",
                    message="Omitted target-quantile pair from target-normalized mean pinball aggregation because RLQR denominator was missing, non-finite or zero.",
                ),
            ],
            ignore_index=True,
            sort=False,
        )
    warning_df.to_csv(warn_dir / "rq2_warnings.csv", index=False)
    dispatch_soc_warnings.to_csv(warn_dir / "6_market_dispatch_soc_selected_day_warnings.csv", index=False)
    bidding_activity_warnings.to_csv(warn_dir / "bidding_activity_submitted_cleared_warnings.csv", index=False)
    bidding_heatmap_warnings.to_csv(warn_dir / "8_submitted_bid_volume_heatmap_warnings.csv", index=False)

    write_primary_table(tables_dir / "1_net_profit_by_model_and_quantile.tex", table, days)
    write_revenue_cost_component_table(tables_dir / "3_revenue_cost_components_best_quantile.tex", component_data)
    write_bid_activity_best_quantile_table(tables_dir / "4_bid_activity_best_quantile.tex", bidding_activity_data)
    write_appendix_table(appendix_tables_dir / "rq2_profit_and_validity_detailed.tex", summary)

    result_figure_paths: list[Path] = []
    appendix_figure_paths: list[Path] = []
    if formats:
        result_figure_paths += plot_net_profit_lines(sweep_data, figures_dir / "2_quantile_sweep_net_profit_by_model", formats, run_root.name, quantiles)
        result_figure_paths += plot_best_quantile_components(component_data, figures_dir / "3_revenue_cost_components_best_quantile", formats, run_root.name)
        result_figure_paths += plot_cumulative_pnl(cumulative_data, figures_dir / "4_cumulative_net_profit_model_comparison_test_period", formats, run_root.name)
        appendix_figure_paths += plot_pinball_net_profit_scatter(scatter_data, appendix_figures_dir, formats)
        result_figure_paths += plot_normalized_total_pinball_profit_scatter(normalized_total_scatter_data, figures_dir / "5_pinball_loss_vs_net_profit_total_normalized", formats)
        result_figure_paths += plot_normalized_total_mae_profit_scatter(normalized_total_mae_data, figures_dir / "5_mae_vs_net_profit_total_normalized", formats)
        appendix_figure_paths += plot_total_pinball_net_profit_scatter(total_scatter_data, appendix_figures_dir / "5_pinball_loss_vs_net_profit_total", formats)
        result_figure_paths += plot_market_dispatch_soc_day(dispatch_soc_data, figures_dir / "6_market_dispatch_soc_selected_day", formats)
        result_figure_paths += plot_mean_pinball_loss_heatmap(relative_pinball_heatmap_data, figures_dir / "7_mean_pinball_loss_heatmap", formats)
        result_figure_paths += plot_target_normalized_mean_pinball_loss_heatmap(target_normalized_pinball_data, figures_dir / "8_target_normalized_mean_pinball_loss_heatmap", formats)
        result_figure_paths += plot_bidding_activity_submitted_cleared(bidding_activity_data, figures_dir / "annualized_bid_volume_by_market_model", formats, metric="volume")
        result_figure_paths += plot_bidding_activity_submitted_cleared(bidding_activity_data, figures_dir / "annualized_bid_count_by_market_model", formats, metric="count")
        result_figure_paths += plot_bid_volume_heatmap(bidding_heatmap_data, figures_dir / "8_submitted_bid_volume_heatmap_by_market_model_quantile", formats, metric="submitted")
        result_figure_paths += plot_bid_volume_heatmap(bidding_heatmap_data, figures_dir / "8_cleared_bid_volume_heatmap_by_market_model_quantile", formats, metric="cleared")
        result_figure_paths += plot_heatmap(heatmap_table, figures_dir / "1_profit_heatmap", formats)
        write_result_section_native_latex_figures(
            latex_figures_dir=latex_figures_dir,
            heatmap_table=heatmap_table,
            sweep_data=sweep_data,
            component_data=component_data,
            cumulative_data=cumulative_data,
            relative_pinball_heatmap_data=relative_pinball_heatmap_data,
            target_normalized_pinball_data=target_normalized_pinball_data,
            normalized_total_scatter_data=normalized_total_scatter_data,
            normalized_total_mae_data=normalized_total_mae_data,
            dispatch_soc_data=dispatch_soc_data,
            bidding_activity_data=bidding_activity_data,
            bidding_heatmap_data=bidding_heatmap_data,
            quantiles=quantiles,
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

    created = datetime.now(timezone.utc).isoformat()
    factor = 1.0 if not args.annualize else 365.0 / days
    entries: list[dict[str, Any]] = []
    for path, tier, artifact_type, metric_family, thesis_use in [
        (tables_dir / "1_net_profit_by_model_and_quantile.tex", "result_section", "latex_table", "annualized realized net profit", "primary RQ2 result table"),
        (tables_dir / "3_revenue_cost_components_best_quantile.tex", "result_section", "latex_table", "revenue and cost components", "best-quantile revenue/cost component table"),
        (tables_dir / "4_bid_activity_best_quantile.tex", "result_section", "latex_table", "bidding activity", "best-quantile submitted and cleared bid volume table"),
        (appendix_tables_dir / "rq2_profit_and_validity_detailed.tex", "appendix", "latex_table", "profit and validity", "validity audit table"),
        (csv_dir / "rq2_scenario_summary_long.csv", "backup", "csv", "scenario summary", "reproducibility backup"),
        (csv_dir / "1_net_profit_by_model_and_quantile.csv", "backup", "csv", "annualized realized net profit", "source data for primary table"),
        (csv_dir / "1_profit_heatmap.csv", "backup", "csv", "annualized realized net profit", "source data for diagnostic heatmap with all numeric rows"),
        (csv_dir / "2_quantile_sweep_net_profit_by_model.csv", "backup", "csv", "annualized Net Profit", "source data for quantile sweep line figure"),
        (csv_dir / "3_revenue_cost_components_best_quantile.csv", "backup", "csv", "revenue and cost components", "source data for best-quantile stacked component figure"),
        (csv_dir / "4_cumulative_net_profit_model_comparison_test_period.csv", "backup", "csv", "cumulative Net Profit", "source data for best-quantile cumulative Net Profit figure"),
        (csv_dir / "5_pinball_loss_vs_net_profit_scatter_data.csv", "backup", "csv", "forecast accuracy vs Net Profit", "source data for target-specific pinball-loss scatter figures"),
        (csv_dir / "5_pinball_loss_vs_net_profit_total_scatter_data.csv", "backup", "csv", "forecast accuracy vs Net Profit", "source data for total pinball-loss scatter figure"),
        (result_csv_dir / "7_mean_pinball_loss_heatmap.csv", "result_section", "csv", "RLQR-relative mean pinball loss", "source data for RLQR-relative mean pinball heatmap"),
        (result_csv_dir / "8_target_normalized_mean_pinball_loss_heatmap.csv", "result_section", "csv", "target-normalized mean pinball loss", "source data for target-normalized mean pinball heatmap"),
        (diag_dir / "8_target_normalized_mean_pinball_loss_omitted_target_quantiles.csv", "backup", "diagnostics", "target-normalized mean pinball loss", "omitted RLQR denominator diagnostics"),
        (csv_dir / "5_pinball_loss_vs_net_profit_total_normalized.csv", "backup", "csv", "normalized forecast accuracy vs Net Profit", "source data for normalized total pinball-loss scatter figure"),
        (csv_dir / "5_mae_vs_net_profit_total_normalized.csv", "backup", "csv", "normalized forecast accuracy vs Net Profit", "source data for normalized total MAE scatter figure"),
        (csv_dir / "6_market_dispatch_soc_selected_day.csv", "backup", "csv", "market dispatch and SoC", "source data for selected-day stacked dispatch/SOC figure"),
        (csv_dir / "bidding_activity_submitted_cleared_by_market_model.csv", "backup", "csv", "bidding activity", "backup source data for submitted/cleared bidding-activity figures"),
        (result_csv_dir / "bidding_activity_submitted_cleared_by_market_model.csv", "result_section", "csv", "bidding activity", "source data for submitted/cleared bidding-activity figures"),
        (csv_dir / "8_submitted_bid_volume_heatmap_by_market_model_quantile.csv", "backup", "csv", "bidding activity", "source data for submitted bid-size heatmap"),
        (result_csv_dir / "8_submitted_bid_volume_heatmap_by_market_model_quantile.csv", "result_section", "csv", "bidding activity", "source data for submitted bid-size heatmap"),
        (csv_dir / "8_cleared_bid_volume_heatmap_by_market_model_quantile.csv", "backup", "csv", "bidding activity", "source data for cleared bid-size heatmap"),
        (result_csv_dir / "8_cleared_bid_volume_heatmap_by_market_model_quantile.csv", "result_section", "csv", "bidding activity", "source data for cleared bid-size heatmap"),
        (csv_dir / "rq2_benchmark_values.csv", "backup", "csv", "benchmark values", "Naive/RHPF benchmark source"),
        (diag_dir / "rq2_input_file_inventory.csv", "backup", "diagnostics", "input inventory", "reproducibility audit"),
        (diag_dir / "bidding_activity_submitted_cleared_source_inventory.csv", "backup", "diagnostics", "bidding activity", "source inventory for submitted/cleared bidding-activity aggregation"),
        (diag_dir / "rq2_validity_diagnostics.csv", "backup", "diagnostics", "validity diagnostics", "invalid-row audit"),
        (warn_dir / "rq2_warnings.csv", "backup", "warnings", "warnings", "generation warnings"),
        (warn_dir / "6_market_dispatch_soc_selected_day_warnings.csv", "backup", "warnings", "market dispatch and SoC", "selected-day invalidity and direct violation checks"),
        (warn_dir / "bidding_activity_submitted_cleared_warnings.csv", "backup", "warnings", "bidding activity", "submitted/cleared bidding-activity aggregation warnings"),
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
        "forecast_pinball_input": str(forecast_benchmark_dir / "diagnostics" / "joined_predictions"),
        "rlqr_relative_mean_pinball_aggregation": relative_pinball_heatmap_weighting,
        "target_normalized_mean_pinball_aggregation": target_normalized_pinball_weighting,
        "target_normalized_mean_pinball_omitted_target_quantiles": int(len(target_normalized_pinball_omissions)),
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
    print(f"[OK] RQ2 pinball input: {forecast_benchmark_dir / 'diagnostics' / 'joined_predictions'}")
    print("[OK] detected pinball columns: model, quantile, target, mean_pinball_loss, n_obs")
    print(f"[OK] RLQR-relative pinball aggregation: {relative_pinball_heatmap_weighting}")
    print(f"[OK] RLQR-relative heatmap CSV: {result_csv_dir / '7_mean_pinball_loss_heatmap.csv'}")
    print(f"[OK] RLQR-relative heatmap LaTeX: {latex_figures_dir / '7_mean_pinball_loss_heatmap.tex'}")
    print(f"[OK] target-normalized pinball aggregation: {target_normalized_pinball_weighting}")
    print(f"[OK] omitted target-quantile combinations: {len(target_normalized_pinball_omissions)}")
    print(f"[OK] target-normalized heatmap CSV: {result_csv_dir / '8_target_normalized_mean_pinball_loss_heatmap.csv'}")
    print(f"[OK] target-normalized heatmap LaTeX: {latex_figures_dir / '8_target_normalized_mean_pinball_loss_heatmap.tex'}")
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
