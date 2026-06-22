#!/usr/bin/env python3
"""Descriptive regret-driver diagnostics for existing BESS simulation runs.

The script is intentionally read-only with respect to simulation artifacts. It
discovers already-written scenario outputs, maps columns conservatively, and
emits diagnostic tables/figures plus debug files that document every missing
component. The outputs are descriptive; they do not claim causal attribution.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from energy_trading.evaluation.style import THESIS_PALETTE, apply_geo_style, get_model_color


DEFAULT_RUN_DIR = Path("artifacts/simulation_runs/thesis_final_multi_2m_20260620T091938Z")
DEFAULT_OUT_DIR = Path("artifacts/benchmark/rq2_simulation_benchmark/regret_drivers")
MODEL_DISPLAY = {"linear": "RLQR", "rlqr": "RLQR", "xgb": "XGB", "xgboost": "XGB", "tft": "TFT"}
REASON_EMPTY = {"", "none", "nan", "null", "[]", "{}"}


ROLE_CANDIDATES: dict[str, list[str]] = {
    "realized_profit": [
        "realized_net_revenue_eur",
        "realized_total_pnl_eur",
        "realized_profit",
        "realized_net_profit",
        "net_profit",
        "net_revenue_eur",
        "PnL Model (€)",
        "model_profit_eur",
    ],
    "planned_profit": [
        "predicted_net_revenue_eur",
        "planned_net_revenue_eur",
        "predicted_total_pnl_eur",
        "planned_profit",
        "predicted_profit",
        "planned_net_profit",
        "PnL Planned (€)",
    ],
    "rhpf_profit": [
        "rhpf_profit",
        "profit_rhpf",
        "PnL RHPF (€)",
        "rolling_pf_profit",
        "rolling_perfect_foresight_profit_eur",
        "rolling_pf_net_revenue_eur",
        "oracle_profit_eur",
    ],
    "naive_profit": ["naive_profit", "PnL Naive (€)", "naive_net_revenue_eur"],
    "simulation_valid": ["simulation_valid", "valid", "is_valid"],
    "thesis_reportable": ["thesis_reportable", "reportable", "is_reportable"],
    "invalid_reason": ["invalid_reason", "invalid_reasons", "validation_reason"],
}

ANNUALIZED_PROFIT_CANDIDATES = [
    "annualized_realized_net_revenue_eur",
    "annualized_realized_net_profit_eur",
    "annualized_net_profit_eur",
    "annualized_profit_eur",
    "annualized_realized_total_pnl_eur",
    "annualized_realized_net_revenue_eur_per_year",
]

COMPONENT_CANDIDATES: dict[str, list[str]] = {
    "DA_profit_eur": ["da_net_revenue_eur", "da_pnl_eur", "da_profit_eur", "DA revenue/profit"],
    "ID_profit_eur": ["id_net_revenue_eur", "id_recourse_pnl_eur", "id_profit_eur", "ID revenue/profit"],
    "BCM_capacity_profit_eur": ["bcm_capacity_revenue_eur", "afrr_capacity_revenue_eur", "bcm_capacity_profit_eur"],
    "BCM_activation_profit_eur": ["bcm_activation_revenue_eur", "bcm_linked_activation_revenue_eur"],
    "BEM_activation_profit_eur": ["bem_net_revenue_eur", "bem_activation_revenue_eur", "bem_pnl_eur"],
    "degradation_cost_eur": ["realized_degradation_cost_eur", "degradation_cost_eur"],
    "auxiliary_cost_eur": ["realized_aux_cost_eur", "aux_cost_eur", "auxiliary_cost_eur"],
    "penalties_eur": ["penalty_cost_eur", "infeasibility_penalty_eur"],
    "terminal_soc_repair_cost_eur": ["terminal_soc_repair_cost_eur", "emergency_id_repair_cost_eur"],
    "total_costs_eur": ["total_costs_eur", "gross_market_costs_eur"],
    "net_profit_eur": ["realized_net_revenue_eur", "net_revenue_eur", "net_profit", "PnL Model (€)"],
}

PLANNED_COMPONENT_CANDIDATES: dict[str, tuple[list[str], list[str]]] = {
    "net_profit_eur": (ROLE_CANDIDATES["planned_profit"], COMPONENT_CANDIDATES["net_profit_eur"]),
    "DA_profit_eur": (["predicted_da_net_revenue_eur", "planned_da_net_revenue_eur"], COMPONENT_CANDIDATES["DA_profit_eur"]),
    "ID_profit_eur": (["predicted_id_net_revenue_eur", "planned_id_net_revenue_eur"], COMPONENT_CANDIDATES["ID_profit_eur"]),
    "BCM_capacity_profit_eur": (["predicted_bcm_capacity_revenue_eur", "planned_bcm_capacity_revenue_eur"], COMPONENT_CANDIDATES["BCM_capacity_profit_eur"]),
    "BEM_activation_profit_eur": (["predicted_bem_net_revenue_eur", "planned_bem_net_revenue_eur"], COMPONENT_CANDIDATES["BEM_activation_profit_eur"]),
}


class MissingRequiredColumnError(ValueError):
    """Raised when strict regret calculation cannot find a required column."""


@dataclass
class ColumnLookup:
    mapping_rows: list[dict[str, Any]] = field(default_factory=list)
    missing_rows: list[dict[str, Any]] = field(default_factory=list)
    daily_alignment_rows: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def record_mapping(self, *, source: str, role: str, column: str | None, candidates: Iterable[str], required: bool = False) -> None:
        self.mapping_rows.append(
            {
                "source": source,
                "role": role,
                "column_used": column,
                "required": bool(required),
                "candidates": "; ".join(candidates),
            }
        )
        if column is None:
            self.missing_rows.append(
                {
                    "source": source,
                    "role": role,
                    "missing_candidates": "; ".join(candidates),
                    "required": bool(required),
                }
            )


@dataclass
class Scenario:
    folder: str
    model_key: str
    model: str
    quantile: str
    strategy: str
    scenario_dir: Path
    is_benchmark: bool
    benchmark_name: str | None = None
    files: dict[str, Path] = field(default_factory=dict)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    hourly: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class TailEventSpec:
    tail_event_type: str
    target_family: str
    direction: str
    tail_side: str
    candidate_columns: tuple[str, ...]


TAIL_EVENT_SPECS: tuple[TailEventSpec, ...] = (
    TailEventSpec(
        tail_event_type="DA price high tail",
        target_family="da_price",
        direction="none",
        tail_side="high",
        candidate_columns=("realized_da_price_eur_per_mwh", "da_price", "price_da", "target_da_price"),
    ),
    TailEventSpec(
        tail_event_type="DA price low tail",
        target_family="da_price",
        direction="none",
        tail_side="low",
        candidate_columns=("realized_da_price_eur_per_mwh", "da_price", "price_da", "target_da_price"),
    ),
    TailEventSpec(
        tail_event_type="aFRR capacity price + high tail",
        target_family="afrr_capacity_price",
        direction="pos",
        tail_side="high",
        candidate_columns=(
            "realized_capacity_price_pos",
            "realized_afrr_capacity_price_pos",
            "afrr_capacity_price_pos",
            "capacity_price_pos",
            "target_afrr_capacity_price_pos",
            "real_bcm_settlement_capacity_price_resolved_pos_eur_per_mw_h",
        ),
    ),
    TailEventSpec(
        tail_event_type="aFRR capacity price - high tail",
        target_family="afrr_capacity_price",
        direction="neg",
        tail_side="high",
        candidate_columns=(
            "realized_capacity_price_neg",
            "realized_afrr_capacity_price_neg",
            "afrr_capacity_price_neg",
            "capacity_price_neg",
            "target_afrr_capacity_price_neg",
            "real_bcm_settlement_capacity_price_resolved_neg_eur_per_mw_h",
        ),
    ),
    TailEventSpec(
        tail_event_type="aFRR activation price + high tail",
        target_family="afrr_activation_price",
        direction="pos",
        tail_side="high",
        candidate_columns=(
            "realized_activation_price_pos",
            "realized_afrr_activation_price_pos",
            "afrr_activation_price_pos",
            "real_act_price_pos",
            "target_afrr_activation_price_vwap_pos",
            "real_bem_submitted_activation_price_pos_eur_per_mwh",
            "real_bcm_true_activation_price_pos",
            "real_bcm_p50_executed_act_pos_price_eur_mwh",
        ),
    ),
    TailEventSpec(
        tail_event_type="aFRR activation price - high tail",
        target_family="afrr_activation_price",
        direction="neg",
        tail_side="high",
        candidate_columns=(
            "realized_activation_price_neg",
            "realized_afrr_activation_price_neg",
            "afrr_activation_price_neg",
            "real_act_price_neg",
            "target_afrr_activation_price_vwap_neg",
            "real_bem_submitted_activation_price_neg_eur_per_mwh",
            "real_bcm_true_activation_price_neg",
            "real_bcm_p50_executed_act_neg_price_eur_mwh",
        ),
    ),
    TailEventSpec(
        tail_event_type="aFRR activation rate + high tail",
        target_family="afrr_activation_rate",
        direction="pos",
        tail_side="high",
        candidate_columns=(
            "realized_activation_rate_pos",
            "realized_afrr_activation_rate_pos",
            "afrr_activation_rate_pos",
            "act_pos_rate",
            "real_act_rate_pos",
            "target_afrr_activation_rate_pos",
            "da_precommit_da_settlement_equiv_replay_act_pos_rate",
        ),
    ),
    TailEventSpec(
        tail_event_type="aFRR activation rate - high tail",
        target_family="afrr_activation_rate",
        direction="neg",
        tail_side="high",
        candidate_columns=(
            "realized_activation_rate_neg",
            "realized_afrr_activation_rate_neg",
            "afrr_activation_rate_neg",
            "act_neg_rate",
            "real_act_rate_neg",
            "target_afrr_activation_rate_neg",
            "da_precommit_da_settlement_equiv_replay_act_neg_rate",
        ),
    ),
)


def normalize_column_name(value: str) -> str:
    text = str(value).lower()
    text = text.replace("€", "eur")
    text = text.replace("%", "pct")
    return re.sub(r"[^a-z0-9]+", "", text)


def discover_column(
    columns: Iterable[str],
    candidates: Iterable[str],
    *,
    role: str = "",
    source: str = "",
    lookup: ColumnLookup | None = None,
    required: bool = False,
) -> str | None:
    cols = [str(c) for c in columns]
    candidates_list = [str(c) for c in candidates]
    exact = {c: c for c in cols}
    lower = {c.lower(): c for c in cols}
    used: str | None = None
    for candidate in candidates_list:
        if candidate in exact:
            used = exact[candidate]
            break
    if used is None:
        for candidate in candidates_list:
            if candidate.lower() in lower:
                used = lower[candidate.lower()]
                break
    if used is None:
        normalized = {normalize_column_name(c): c for c in cols}
        for candidate in candidates_list:
            key = normalize_column_name(candidate)
            if key in normalized:
                used = normalized[key]
                break
    if lookup is not None:
        lookup.record_mapping(source=source, role=role, column=used, candidates=candidates_list, required=required)
    if required and used is None:
        raise MissingRequiredColumnError(
            f"Missing required column for role '{role}' in {source or 'dataframe'}. "
            f"Tried candidates: {', '.join(candidates_list)}"
        )
    return used


def _safe_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def _boolish(value: Any) -> bool | float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return math.nan


def _split_reasons(value: Any) -> list[str]:
    text = str(value or "").strip()
    if text.lower() in REASON_EMPTY:
        return []
    return [p.strip() for p in re.split(r"[,;|]", text) if p.strip() and p.strip().lower() not in REASON_EMPTY]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _scenario_identity(path: Path, run_dir: Path) -> tuple[str, str, str, str, str, bool, str | None]:
    rel = path.relative_to(run_dir)
    folder = rel.parts[0]
    strategy = rel.parts[1] if len(rel.parts) > 1 else ""
    policy = rel.parts[2] if len(rel.parts) > 2 else ""
    is_benchmark = folder.startswith("benchmarks_")
    if is_benchmark:
        key = folder.replace("benchmarks_", "")
        model = {"rhpf": "RHPF", "naive": "Naive"}.get(key, key.upper())
        return folder, key, model, "benchmark", strategy, True, model
    parts = folder.split("_")
    key = parts[0]
    quantile = next((p for p in parts[1:] if re.fullmatch(r"p\d+", p)), "")
    if not quantile:
        quantile = policy.split("_")[0] if policy else ""
    return folder, key, MODEL_DISPLAY.get(key, key.upper()), quantile, strategy, False, None


def _candidate_files(path: Path) -> dict[str, Path]:
    names = {
        "performance_metrics": "performance_metrics.csv",
        "daily_metrics": "daily_performance_metrics.csv",
        "hourly": "backtest_hourly.parquet",
        "model_hourly": "model_hourly.parquet",
        "realized_ledger": "realized_ledger.parquet",
        "planned_ledger": "planned_ledger.parquet",
        "executed_ledger": "executed_ledger.parquet",
        "summary_json": "backtest_summary.json",
        "model_summary_json": "model_summary.json",
        "rolling_pf_summary_json": "rolling_pf_summary.json",
        "infeasibility_attribution": "optimization_infeasibility_attribution.csv",
        "solver_failures": "solver_failure_diagnostics.csv",
        "event_log": "backtest_milp_event_log.parquet",
    }
    return {key: path / name for key, name in names.items() if (path / name).exists()}


def discover_scenarios(run_dir: Path, *, models: set[str], quantiles: set[str]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for summary in sorted(run_dir.glob("*/multi/*/backtest_summary.json")):
        folder, key, model, quantile, strategy, is_benchmark, benchmark_name = _scenario_identity(summary.parent, run_dir)
        if not is_benchmark:
            wanted_model = key in models or model.lower() in models or (key == "linear" and "rlqr" in models)
            wanted_quantile = quantile in quantiles
            if not wanted_model or not wanted_quantile:
                continue
        files = _candidate_files(summary.parent)
        scenario = Scenario(
            folder=folder,
            model_key=key,
            model=model,
            quantile=quantile,
            strategy=strategy,
            scenario_dir=summary.parent,
            is_benchmark=is_benchmark,
            benchmark_name=benchmark_name,
            files=files,
        )
        if "performance_metrics" in files:
            scenario.metrics = _read_csv(files["performance_metrics"])
        if "daily_metrics" in files:
            scenario.daily = _read_csv(files["daily_metrics"])
        if "hourly" in files:
            scenario.hourly = _read_parquet(files["hourly"])
        scenarios.append(scenario)
    return scenarios


def _first_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    return df.iloc[0]


def _series_value(row: pd.Series, column: str | None) -> Any:
    if column is None or row.empty or column not in row.index:
        return math.nan
    return row[column]


def calculate_regret_values(realized_profit: float, benchmark_profit: float, planned_profit: float = math.nan) -> dict[str, float]:
    realized = _safe_float(realized_profit)
    benchmark = _safe_float(benchmark_profit)
    planned = _safe_float(planned_profit)
    total = benchmark - realized if math.isfinite(realized) and math.isfinite(benchmark) else math.nan
    relative = total / abs(benchmark) if math.isfinite(total) and math.isfinite(benchmark) and benchmark != 0 else math.nan
    model_vs_benchmark = realized / benchmark * 100.0 if math.isfinite(realized) and math.isfinite(benchmark) and benchmark != 0 else math.nan
    gap = planned - realized if math.isfinite(planned) and math.isfinite(realized) else math.nan
    return {
        "planning_gap_eur": gap,
        "total_regret_eur": total,
        "relative_regret": relative,
        "model_vs_benchmark_pct": model_vs_benchmark,
    }


def _annualization_factor(row: pd.Series, lookup: ColumnLookup, source: str) -> tuple[float, str | None]:
    factor_col = discover_column(row.index, ["annualization_factor"], role="annualization_factor", source=source, lookup=lookup)
    factor = _safe_float(_series_value(row, factor_col))
    if math.isfinite(factor):
        return factor, factor_col
    days_col = discover_column(row.index, ["n_days", "duration_days", "simulation_days"], role="annualization_days", source=source, lookup=lookup)
    days = _safe_float(_series_value(row, days_col))
    if math.isfinite(days) and days > 0:
        return 365.0 / days, days_col
    hours_col = discover_column(row.index, ["n_hours", "duration_hours", "simulation_hours"], role="annualization_hours", source=source, lookup=lookup)
    hours = _safe_float(_series_value(row, hours_col))
    if math.isfinite(hours) and hours > 0:
        return 365.0 * 24.0 / hours, hours_col
    return math.nan, None


def _annualized_profit_from_row(row: pd.Series, *, source: str, lookup: ColumnLookup, role_prefix: str) -> tuple[float, str | None]:
    annualized_col = discover_column(
        row.index,
        ANNUALIZED_PROFIT_CANDIDATES,
        role=f"{role_prefix}_annualized_profit",
        source=source,
        lookup=lookup,
    )
    if annualized_col is not None:
        return _safe_float(_series_value(row, annualized_col)), annualized_col
    realized_col = discover_column(
        row.index,
        ROLE_CANDIDATES["realized_profit"],
        role=f"{role_prefix}_realized_profit_for_annualization",
        source=source,
        lookup=lookup,
    )
    realized = _safe_float(_series_value(row, realized_col))
    factor, factor_col = _annualization_factor(row, lookup, source)
    if math.isfinite(realized) and math.isfinite(factor):
        return realized * factor, f"{realized_col}*annualization_factor({factor_col})"
    return math.nan, None


def _benchmark_annualized_profit_for(
    scenario: Scenario,
    benchmark_scenario: Scenario | None,
    benchmark: str,
    lookup: ColumnLookup,
) -> tuple[float, str | None]:
    row = _first_row(scenario.metrics)
    source = str(scenario.scenario_dir / "performance_metrics.csv")
    benchmark_specific = [
        f"annualized_{benchmark.lower()}_profit_eur",
        f"{benchmark.lower()}_annualized_profit_eur",
        f"annualized_{benchmark.lower()}_net_revenue_eur",
        f"{benchmark.lower()}_annualized_net_revenue_eur",
        f"PnL {benchmark.upper()} annualized (€)",
    ]
    col = discover_column(row.index, benchmark_specific, role=f"{benchmark}_annualized_profit", source=source, lookup=lookup)
    if col is not None:
        return _safe_float(_series_value(row, col)), col
    if benchmark_scenario is not None and not benchmark_scenario.metrics.empty:
        bench_row = _first_row(benchmark_scenario.metrics)
        bench_source = str(benchmark_scenario.scenario_dir / "performance_metrics.csv")
        value, used = _annualized_profit_from_row(bench_row, source=bench_source, lookup=lookup, role_prefix=f"benchmark_{benchmark}")
        if used is not None:
            return value, f"benchmark_scenario:{benchmark_scenario.scenario_dir}:{used}"
    return math.nan, None


def build_annualized_regret_table(scenarios: list[Scenario], *, benchmark: str, lookup: ColumnLookup) -> pd.DataFrame:
    benchmark_scenario = _benchmark_scenario(scenarios, benchmark)
    columns = [
        "Model",
        "Quantile",
        "Annualized net profit",
        "RHPF annualized profit",
        "Regret vs RHPF",
        "Regret share",
        "Model/RHPF (%)",
        "model_annualized_column_used",
        "benchmark_annualized_column_used",
    ]
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario.is_benchmark:
            continue
        row = _first_row(scenario.metrics)
        source = str(scenario.scenario_dir / "performance_metrics.csv")
        annualized_model, model_col = _annualized_profit_from_row(row, source=source, lookup=lookup, role_prefix="model")
        annualized_benchmark, benchmark_col = _benchmark_annualized_profit_for(scenario, benchmark_scenario, benchmark, lookup)
        regret = annualized_benchmark - annualized_model if math.isfinite(annualized_model) and math.isfinite(annualized_benchmark) else math.nan
        regret_share = regret / abs(annualized_benchmark) * 100.0 if math.isfinite(regret) and math.isfinite(annualized_benchmark) and annualized_benchmark != 0 else math.nan
        model_vs_benchmark = annualized_model / annualized_benchmark * 100.0 if math.isfinite(annualized_model) and math.isfinite(annualized_benchmark) and annualized_benchmark != 0 else math.nan
        rows.append(
            {
                "Model": scenario.model,
                "Quantile": scenario.quantile,
                "Annualized net profit": annualized_model,
                "RHPF annualized profit": annualized_benchmark,
                "Regret vs RHPF": regret,
                "Regret share": regret_share,
                "Model/RHPF (%)": model_vs_benchmark,
                "model_annualized_column_used": model_col,
                "benchmark_annualized_column_used": benchmark_col,
            }
        )
    out = pd.DataFrame(rows, columns=columns)
    if not out.empty:
        model_order = {"RLQR": 0, "XGB": 1, "TFT": 2}
        quantile_order = {"p10": 0, "p30": 1, "p50": 2, "p70": 3, "p90": 4}
        out = out.assign(
            _model_order=out["Model"].map(model_order).fillna(99),
            _quantile_order=out["Quantile"].map(quantile_order).fillna(99),
        ).sort_values(["_model_order", "_quantile_order", "Model", "Quantile"]).drop(columns=["_model_order", "_quantile_order"]).reset_index(drop=True)
    return out


def _benchmark_scenario(scenarios: list[Scenario], benchmark: str) -> Scenario | None:
    candidates = [s for s in scenarios if s.is_benchmark and (s.benchmark_name or "").lower() == benchmark.lower()]
    if candidates:
        return candidates[0]
    candidates = [s for s in scenarios if s.is_benchmark and benchmark.lower() in s.folder.lower()]
    return candidates[0] if candidates else None


def _benchmark_profit_for(
    scenario: Scenario,
    benchmark_scenario: Scenario | None,
    benchmark: str,
    lookup: ColumnLookup,
) -> tuple[float, str | None, bool | float, str]:
    row = _first_row(scenario.metrics)
    role = f"{benchmark.lower()}_profit"
    candidates = ROLE_CANDIDATES.get(role, ROLE_CANDIDATES["rhpf_profit"])
    col = discover_column(row.index, candidates, role=role, source=str(scenario.scenario_dir / "performance_metrics.csv"), lookup=lookup)
    if col is not None:
        return _safe_float(row[col]), col, math.nan, ""
    if benchmark_scenario is not None and not benchmark_scenario.metrics.empty:
        bench_row = _first_row(benchmark_scenario.metrics)
        realized_col = discover_column(
            bench_row.index,
            ROLE_CANDIDATES["realized_profit"],
            role="benchmark_realized_profit",
            source=str(benchmark_scenario.scenario_dir / "performance_metrics.csv"),
            lookup=lookup,
        )
        valid_col = discover_column(
            bench_row.index,
            ROLE_CANDIDATES["simulation_valid"],
            role="benchmark_valid",
            source=str(benchmark_scenario.scenario_dir / "performance_metrics.csv"),
            lookup=lookup,
        )
        invalid_col = discover_column(
            bench_row.index,
            ROLE_CANDIDATES["invalid_reason"],
            role="benchmark_invalid_reason",
            source=str(benchmark_scenario.scenario_dir / "performance_metrics.csv"),
            lookup=lookup,
        )
        source = f"benchmark_scenario:{benchmark_scenario.scenario_dir}:{realized_col}"
        return _safe_float(_series_value(bench_row, realized_col)), source, _boolish(_series_value(bench_row, valid_col)), str(_series_value(bench_row, invalid_col) or "")
    lookup.warnings.append(f"No benchmark profit source found for {scenario.folder}; benchmark={benchmark}.")
    return math.nan, None, math.nan, "missing_benchmark_profit"


def build_regret_bridge(scenarios: list[Scenario], *, benchmark: str, lookup: ColumnLookup, strict: bool) -> pd.DataFrame:
    bench = _benchmark_scenario(scenarios, benchmark)
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario.is_benchmark:
            continue
        row = _first_row(scenario.metrics)
        source = str(scenario.scenario_dir / "performance_metrics.csv")
        realized_col = discover_column(row.index, ROLE_CANDIDATES["realized_profit"], role="realized_profit", source=source, lookup=lookup, required=strict)
        planned_col = discover_column(row.index, ROLE_CANDIDATES["planned_profit"], role="planned_profit", source=source, lookup=lookup)
        valid_col = discover_column(row.index, ROLE_CANDIDATES["simulation_valid"], role="simulation_valid", source=source, lookup=lookup)
        report_col = discover_column(row.index, ROLE_CANDIDATES["thesis_reportable"], role="thesis_reportable", source=source, lookup=lookup)
        reason_col = discover_column(row.index, ROLE_CANDIDATES["invalid_reason"], role="invalid_reason", source=source, lookup=lookup)
        realized = _safe_float(_series_value(row, realized_col))
        planned = _safe_float(_series_value(row, planned_col))
        benchmark_profit, benchmark_col, benchmark_valid, benchmark_reason = _benchmark_profit_for(scenario, bench, benchmark, lookup)
        if strict and not math.isfinite(benchmark_profit):
            raise MissingRequiredColumnError(f"Missing benchmark profit for {scenario.folder}; benchmark={benchmark}.")
        calc = calculate_regret_values(realized, benchmark_profit, planned)
        rows.append(
            {
                "run_name": scenario.scenario_dir.parents[2].name if len(scenario.scenario_dir.parents) >= 3 else "",
                "strategy": scenario.strategy,
                "model": scenario.model,
                "quantile": scenario.quantile,
                "planned_profit_eur": planned,
                "realized_profit_eur": realized,
                "benchmark_profit_eur": benchmark_profit,
                **calc,
                "simulation_valid": _boolish(_series_value(row, valid_col)),
                "thesis_reportable": _boolish(_series_value(row, report_col)),
                "invalid_reason": str(_series_value(row, reason_col) or ""),
                "benchmark_valid": benchmark_valid,
                "benchmark_invalid_reason": benchmark_reason,
                "benchmark_column_used": benchmark_col,
                "planned_profit_column_used": planned_col,
                "realized_profit_column_used": realized_col,
            }
        )
    return pd.DataFrame(rows)


def build_pnl_by_market(scenarios: list[Scenario], lookup: ColumnLookup) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario.is_benchmark:
            continue
        row = _first_row(scenario.metrics)
        out: dict[str, Any] = {"strategy": scenario.strategy, "model": scenario.model, "quantile": scenario.quantile}
        for component, candidates in COMPONENT_CANDIDATES.items():
            col = discover_column(row.index, candidates, role=f"component:{component}", source=str(scenario.scenario_dir / "performance_metrics.csv"), lookup=lookup)
            out[component] = _safe_float(_series_value(row, col))
            out[f"{component}_column_used"] = col
            out[f"{component}_status"] = "available" if col is not None else "missing_column"
        rows.append(out)
    return pd.DataFrame(rows)


def build_planned_vs_realized(scenarios: list[Scenario], lookup: ColumnLookup) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario.is_benchmark:
            continue
        row = _first_row(scenario.metrics)
        realized_profit_col = discover_column(row.index, ROLE_CANDIDATES["realized_profit"], role="realized_profit_for_planning_share", source=str(scenario.scenario_dir), lookup=lookup)
        planned_profit_col = discover_column(row.index, ROLE_CANDIDATES["planned_profit"], role="planned_profit_for_planning_share", source=str(scenario.scenario_dir), lookup=lookup)
        total_gap = _safe_float(_series_value(row, planned_profit_col)) - _safe_float(_series_value(row, realized_profit_col))
        if not math.isfinite(total_gap):
            total_gap = math.nan
        for component, (planned_candidates, realized_candidates) in PLANNED_COMPONENT_CANDIDATES.items():
            planned_col = discover_column(row.index, planned_candidates, role=f"planned_component:{component}", source=str(scenario.scenario_dir), lookup=lookup)
            realized_col = discover_column(row.index, realized_candidates, role=f"realized_component:{component}", source=str(scenario.scenario_dir), lookup=lookup)
            planned = _safe_float(_series_value(row, planned_col))
            realized = _safe_float(_series_value(row, realized_col))
            gap = planned - realized if math.isfinite(planned) and math.isfinite(realized) else math.nan
            share = gap / total_gap if math.isfinite(gap) and math.isfinite(total_gap) and total_gap != 0 else math.nan
            rows.append(
                {
                    "strategy": scenario.strategy,
                    "model": scenario.model,
                    "quantile": scenario.quantile,
                    "component": component,
                    "planned_component_eur": planned,
                    "realized_component_eur": realized,
                    "component_gap_eur": gap,
                    "share_of_total_planning_gap": share,
                    "planned_column_used": planned_col,
                    "realized_column_used": realized_col,
                    "data_status": "available" if planned_col and realized_col else "unavailable_missing_column",
                }
            )
    return pd.DataFrame(rows)


def build_bid_execution(scenarios: list[Scenario], lookup: ColumnLookup) -> pd.DataFrame:
    market_specs = {
        "DA": (["da_bid_abs_mwh_total", "da_bid_buy_mwh_total"], ["da_realized_abs_mwh_total", "da_realized_buy_mwh_total"], "mwh"),
        "ID": (["id_abs_mwh_total", "id_buy_mwh_total"], ["id_abs_mwh_total", "id_sell_mwh_total"], "mwh"),
        "BCM": (["bcm_bid_capacity_abs_mw_mean", "bcm_bid_capacity_pos_mw_mean"], ["bcm_realized_capacity_abs_mw_mean", "bcm_realized_capacity_pos_mw_mean"], "mw"),
        "BEM": (["bem_bid_abs_mwh_total", "bem_bid_pos_mwh_total"], ["bem_realized_abs_mwh_total", "bem_realized_pos_mwh_total"], "mwh"),
    }
    revenue_candidates = {
        "DA": ["da_net_revenue_eur"],
        "ID": ["id_net_revenue_eur"],
        "BCM": ["bcm_capacity_revenue_eur", "afrr_capacity_revenue_eur"],
        "BEM": ["bem_net_revenue_eur", "bem_activation_revenue_eur"],
    }
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario.is_benchmark:
            continue
        row = _first_row(scenario.metrics)
        for market, (offered_candidates, executed_candidates, unit) in market_specs.items():
            offered_col = discover_column(row.index, offered_candidates, role=f"{market}:offered", source=str(scenario.scenario_dir), lookup=lookup)
            executed_col = discover_column(row.index, executed_candidates, role=f"{market}:executed", source=str(scenario.scenario_dir), lookup=lookup)
            revenue_col = discover_column(row.index, revenue_candidates[market], role=f"{market}:realized_revenue", source=str(scenario.scenario_dir), lookup=lookup)
            offered = _safe_float(_series_value(row, offered_col))
            executed = _safe_float(_series_value(row, executed_col))
            non_exec = offered - executed if math.isfinite(offered) and math.isfinite(executed) else math.nan
            ratio = executed / offered if math.isfinite(offered) and offered != 0 and math.isfinite(executed) else math.nan
            rows.append(
                {
                    "model": scenario.model,
                    "quantile": scenario.quantile,
                    "market": market,
                    "offered_volume_mwh": offered if unit == "mwh" else math.nan,
                    "offered_capacity_mw": offered if unit == "mw" else math.nan,
                    "awarded_volume_mwh": executed if unit == "mwh" and market in {"DA", "ID"} else math.nan,
                    "awarded_capacity_mw": executed if unit == "mw" else math.nan,
                    "executed_volume_mwh": executed if unit == "mwh" else math.nan,
                    "non_executed_volume_mwh": non_exec if unit == "mwh" else math.nan,
                    "execution_ratio": ratio,
                    "expected_revenue_eur": math.nan,
                    "realized_revenue_eur": _safe_float(_series_value(row, revenue_col)),
                    "missed_profitable_execution_eur": math.nan,
                    "avoided_unprofitable_execution_eur": math.nan,
                    "net_execution_effect_eur": math.nan,
                    "data_status": "available" if offered_col and executed_col else "missing_column",
                    "offered_column_used": offered_col,
                    "executed_column_used": executed_col,
                    "realized_revenue_column_used": revenue_col,
                }
            )
    return pd.DataFrame(rows)


def _find_time_column(df: pd.DataFrame) -> str | None:
    return discover_column(df.columns, ["timestamp_utc", "timestamp", "date_utc", "datetime_utc", "time"], role="timestamp")


def _numeric_sum(df: pd.DataFrame, candidates: Iterable[str], lookup: ColumnLookup, source: str, role: str) -> tuple[float, str | None]:
    col = discover_column(df.columns, candidates, role=role, source=source, lookup=lookup)
    if col is None:
        return math.nan, None
    values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.dropna().sum()) if values.notna().any() else math.nan, col


def build_activation_surprise(scenarios: list[Scenario], lookup: ColumnLookup) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario.is_benchmark:
            continue
        hourly = scenario.hourly
        source = str(scenario.files.get("hourly", scenario.scenario_dir / "backtest_hourly.parquet"))
        if hourly.empty:
            rows.append({"model": scenario.model, "quantile": scenario.quantile, "aggregation_level": "aggregate", "data_status": "missing_hourly_file"})
            continue
        specs = {
            "activation_rate": (["forecast_activation_rate", "ev_pred_act_rate_pos_p90", "ev_pred_act_rate_pos_guard"], ["realized_activation_rate", "real_act_rate_pos", "act_pos_rate"]),
            "activation_price": (["forecast_activation_price", "pred_afrr_activation_price_pos", "ev_pred_act_price_pos"], ["realized_activation_price", "afrr_activation_price_pos", "real_act_price_pos"]),
            "activation_mwh": (["planned_activation_mwh", "bem_bid_pos_mwh", "bcm_bid_activation_pos_mwh"], ["realized_activation_mwh", "bem_realized_pos_mwh", "realized_activation_pos_mwh"]),
        }
        cols = {
            key: (
                discover_column(hourly.columns, pred, role=f"activation_surprise:{key}:forecast", source=source, lookup=lookup),
                discover_column(hourly.columns, real, role=f"activation_surprise:{key}:realized", source=source, lookup=lookup),
            )
            for key, (pred, real) in specs.items()
        }
        rates = cols["activation_rate"]
        prices = cols["activation_price"]
        vols = cols["activation_mwh"]
        data_status = "available_forecast_errors" if rates[0] and rates[1] else "missing_rate_columns"
        monetary_status = "direction_unavailable"
        if prices[0] and prices[1] and vols[0] and vols[1]:
            # Positive-direction approximation only when explicit signed columns are found.
            monetary_status = "computed_positive_direction_only"
        pred_rate_mean = _safe_float(pd.to_numeric(hourly[rates[0]], errors="coerce").mean()) if rates[0] else math.nan
        real_rate_mean = _safe_float(pd.to_numeric(hourly[rates[1]], errors="coerce").mean()) if rates[1] else math.nan
        planned_mwh, _ = _numeric_sum(hourly, [vols[0]] if vols[0] else [], lookup, source, "planned_activation_mwh_sum")
        realized_mwh, _ = _numeric_sum(hourly, [vols[1]] if vols[1] else [], lookup, source, "realized_activation_mwh_sum")
        forecast_price_mean = _safe_float(pd.to_numeric(hourly[prices[0]], errors="coerce").mean()) if prices[0] else math.nan
        realized_price_mean = _safe_float(pd.to_numeric(hourly[prices[1]], errors="coerce").mean()) if prices[1] else math.nan
        volume_effect = price_effect = interaction = total = math.nan
        if monetary_status == "computed_positive_direction_only":
            planned_v = pd.to_numeric(hourly[vols[0]], errors="coerce")
            realized_v = pd.to_numeric(hourly[vols[1]], errors="coerce")
            forecast_p = pd.to_numeric(hourly[prices[0]], errors="coerce")
            realized_p = pd.to_numeric(hourly[prices[1]], errors="coerce")
            valid = planned_v.notna() & realized_v.notna() & forecast_p.notna() & realized_p.notna()
            if valid.any():
                dv = realized_v[valid] - planned_v[valid]
                dp = realized_p[valid] - forecast_p[valid]
                volume_effect = float((dv * forecast_p[valid]).sum())
                price_effect = float((realized_v[valid] * dp).sum())
                interaction = float((dv * dp).sum())
                total = volume_effect + price_effect + interaction
        rows.append(
            {
                "model": scenario.model,
                "quantile": scenario.quantile,
                "aggregation_level": "aggregate",
                "forecast_activation_rate": pred_rate_mean,
                "realized_activation_rate": real_rate_mean,
                "activation_rate_error": real_rate_mean - pred_rate_mean if math.isfinite(pred_rate_mean) and math.isfinite(real_rate_mean) else math.nan,
                "planned_activation_mwh": planned_mwh,
                "realized_activation_mwh": realized_mwh,
                "activation_volume_error": realized_mwh - planned_mwh if math.isfinite(planned_mwh) and math.isfinite(realized_mwh) else math.nan,
                "forecast_activation_price": forecast_price_mean,
                "realized_activation_price": realized_price_mean,
                "activation_price_error": realized_price_mean - forecast_price_mean if math.isfinite(forecast_price_mean) and math.isfinite(realized_price_mean) else math.nan,
                "volume_effect_eur": volume_effect,
                "price_effect_eur": price_effect,
                "interaction_effect_eur": interaction,
                "activation_surprise_total_eur": total,
                "data_status": data_status,
                "monetary_effect_status": monetary_status,
            }
        )
    return pd.DataFrame(rows)


def build_penalty_infeasibility(scenarios: list[Scenario], lookup: ColumnLookup) -> pd.DataFrame:
    penalty_cols = {
        "missed_activation_penalty_eur": ["missed_activation_penalty_eur", "penalty_cost_eur"],
        "reserve_shortfall_penalty_eur": ["reserve_shortfall_penalty_eur", "penalty_cost_eur"],
        "infeasibility_penalty_eur": ["infeasibility_penalty_eur", "penalty_cost_eur"],
        "emergency_id_repair_cost_eur": ["emergency_id_repair_cost_eur", "terminal_soc_repair_cost_eur"],
        "total_penalty_cost_eur": ["total_penalty_cost_eur", "penalty_cost_eur"],
    }
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario.is_benchmark:
            continue
        row = _first_row(scenario.metrics)
        reasons = _split_reasons(_series_value(row, discover_column(row.index, ROLE_CANDIDATES["invalid_reason"], role="invalid_reason_for_penalties", source=str(scenario.scenario_dir), lookup=lookup)))
        out: dict[str, Any] = {"model": scenario.model, "quantile": scenario.quantile, "invalid_reason_tokens": "; ".join(reasons)}
        for name, candidates in penalty_cols.items():
            col = discover_column(row.index, candidates, role=f"penalty:{name}", source=str(scenario.scenario_dir), lookup=lookup)
            out[name] = _safe_float(_series_value(row, col))
            out[f"{name}_column_used"] = col
        reason_text = " ".join(r.lower() for r in reasons)
        for label, patterns in {
            "reserve_headroom_shortfall_count": ["headroom", "reserve_shortfall"],
            "protected_soc_count": ["protected_soc"],
            "missed_activation_count": ["missed_activation"],
            "fallback_used_count": ["fallback"],
            "optimization_infeasible_count": ["infeasible"],
            "optimization_infeasible_debug_dump_count": ["debug_dump"],
        }.items():
            out[label] = sum(1 for p in patterns if p in reason_text)
        hourly = scenario.hourly
        affected = math.nan
        if not hourly.empty:
            flag_cols = [c for c in hourly.columns if any(tok in c.lower() for tok in ["fallback", "infeasible", "shortfall", "missed", "violation"])]
            if flag_cols:
                flags = hourly[flag_cols].apply(lambda s: pd.to_numeric(s, errors="coerce").abs().gt(1e-12) if pd.api.types.is_numeric_dtype(s) else s.astype(str).str.lower().isin({"true", "1", "yes"}))
                affected = float(flags.any(axis=1).sum())
        out["affected_hours"] = affected
        out["associated_regret_eur"] = math.nan
        rows.append(out)
    return pd.DataFrame(rows)


def _daily_benchmark(scenarios: list[Scenario], benchmark: str) -> pd.DataFrame:
    bench = _benchmark_scenario(scenarios, benchmark)
    return bench.daily.copy() if bench is not None and not bench.daily.empty else pd.DataFrame()


def _parse_daily_dates(df: pd.DataFrame, column: str, *, source: str) -> pd.Series:
    parsed = pd.to_datetime(df[column], errors="coerce", utc=True)
    bad = parsed.isna()
    if bad.any():
        examples = df.loc[bad, column].astype(str).head(5).tolist()
        raise ValueError(f"Failed to parse daily date column '{column}' in {source}. Invalid examples: {examples}")
    return parsed.dt.strftime("%Y-%m-%d")


def build_daily_regret(scenarios: list[Scenario], *, benchmark: str, lookup: ColumnLookup) -> pd.DataFrame:
    bench_daily = _daily_benchmark(scenarios, benchmark)
    if bench_daily.empty:
        lookup.warnings.append(f"No daily benchmark file found for benchmark={benchmark}; daily regret unavailable.")
        return pd.DataFrame()
    date_col_b = discover_column(bench_daily.columns, ["date_utc", "date", "timestamp_utc"], role="daily_benchmark_date", source="benchmark_daily", lookup=lookup)
    profit_col_b = discover_column(bench_daily.columns, ["net_revenue_eur", "realized_net_revenue_eur", "PnL RHPF (€)"], role="daily_benchmark_profit", source="benchmark_daily", lookup=lookup)
    if not date_col_b or not profit_col_b:
        return pd.DataFrame()
    bench = pd.DataFrame(
        {
            "date": _parse_daily_dates(bench_daily, date_col_b, source="benchmark_daily"),
            "benchmark_daily_profit_eur": pd.to_numeric(bench_daily[profit_col_b], errors="coerce"),
        }
    )
    benchmark_days = set(bench["date"].astype(str))
    rows: list[pd.DataFrame] = []
    for scenario in scenarios:
        if scenario.is_benchmark or scenario.daily.empty:
            continue
        daily = scenario.daily.copy()
        date_col = discover_column(daily.columns, ["date_utc", "date", "timestamp_utc"], role="daily_model_date", source=str(scenario.scenario_dir), lookup=lookup)
        profit_col = discover_column(daily.columns, ["net_revenue_eur", "realized_net_revenue_eur"], role="daily_model_profit", source=str(scenario.scenario_dir), lookup=lookup)
        if not date_col or not profit_col:
            continue
        model_daily = daily.copy()
        model_daily["date"] = _parse_daily_dates(model_daily, date_col, source=str(scenario.scenario_dir / "daily_performance_metrics.csv"))
        model_daily["model_daily_profit_eur"] = pd.to_numeric(model_daily[profit_col], errors="coerce")
        model_days = set(model_daily["date"].astype(str))
        missing_model_days = sorted(benchmark_days - model_days)
        missing_benchmark_days = sorted(model_days - benchmark_days)
        merged = model_daily.merge(bench, on="date", how="inner")
        lookup.daily_alignment_rows.append(
            {
                "model": scenario.model,
                "quantile": scenario.quantile,
                "benchmark": benchmark,
                "model_days": len(model_days),
                "benchmark_days": len(benchmark_days),
                "merged_days": len(merged),
                "missing_model_days": "; ".join(missing_model_days),
                "missing_benchmark_days": "; ".join(missing_benchmark_days),
                "date_min": min(model_days | benchmark_days) if (model_days or benchmark_days) else "",
                "date_max": max(model_days | benchmark_days) if (model_days or benchmark_days) else "",
            }
        )
        if missing_model_days:
            lookup.warnings.append(f"{scenario.model} {scenario.quantile}: benchmark has {len(missing_model_days)} day(s) missing from model daily output.")
        if missing_benchmark_days:
            lookup.warnings.append(f"{scenario.model} {scenario.quantile}: model has {len(missing_benchmark_days)} day(s) missing from benchmark daily output.")
        if merged.empty:
            continue
        out = pd.DataFrame(
            {
                "date": merged["date"].astype(str),
                "model": scenario.model,
                "quantile": scenario.quantile,
                "model_daily_profit_eur": pd.to_numeric(merged["model_daily_profit_eur"], errors="coerce"),
                "benchmark_daily_profit_eur": pd.to_numeric(merged["benchmark_daily_profit_eur"], errors="coerce"),
            }
        )
        out["realized_model_profit_eur"] = out["model_daily_profit_eur"]
        out["benchmark_profit_eur"] = out["benchmark_daily_profit_eur"]
        out["daily_regret_eur"] = out["benchmark_daily_profit_eur"] - out["model_daily_profit_eur"]
        for name, candidates in {
            "DA_profit_eur": ["da_pnl_eur", "da_net_revenue_eur"],
            "ID_profit_eur": ["id_recourse_pnl_eur", "id_net_revenue_eur"],
            "BCM_profit_eur": ["bcm_pnl_eur", "bcm_total_revenue_eur"],
            "BEM_profit_eur": ["bem_pnl_eur", "bem_total_revenue_eur"],
            "penalties_eur": ["penalty_cost_eur"],
        }.items():
            col = discover_column(merged.columns, candidates, role=f"daily:{name}", source=str(scenario.scenario_dir), lookup=lookup)
            out[name] = pd.to_numeric(merged[col], errors="coerce") if col else math.nan
        out["tail_event_flag"] = False
        out["activation_surprise_flag"] = False
        out["infeasibility_flag"] = merged.get("has_non_ok_optimization_hour", False)
        out = out.sort_values("date").reset_index(drop=True)
        out["cumulative_model_profit_eur"] = out["model_daily_profit_eur"].cumsum()
        out["cumulative_benchmark_profit_eur"] = out["benchmark_daily_profit_eur"].cumsum()
        out["cumulative_regret_eur"] = out["daily_regret_eur"].cumsum()
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_tail_event_regret(daily_regret: pd.DataFrame, scenarios: list[Scenario], lookup: ColumnLookup) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    def unavailable_row(scenario: Scenario, spec: TailEventSpec, *, column_used: str | None, data_status: str) -> dict[str, Any]:
        return {
            "model": scenario.model,
            "quantile": scenario.quantile,
            "tail_event_type": spec.tail_event_type,
            "target_family": spec.target_family,
            "direction": spec.direction,
            "tail_side": spec.tail_side,
            "column_used": column_used or "",
            "candidate_columns": "; ".join(spec.candidate_columns),
            "threshold_used": math.nan,
            "n_periods": math.nan,
            "n_tail_days": math.nan,
            "model_profit_in_tail_periods": math.nan,
            "benchmark_profit_in_tail_periods": math.nan,
            "tail_regret_eur": math.nan,
            "tail_regret_share_of_total_regret": math.nan,
            "non_tail_regret_eur": math.nan,
            "data_status": data_status,
        }

    for scenario in scenarios:
        if scenario.is_benchmark:
            continue
        if scenario.hourly.empty:
            for spec in TAIL_EVENT_SPECS:
                rows.append(unavailable_row(scenario, spec, column_used=None, data_status="missing_hourly_file"))
            continue
        hourly = scenario.hourly
        source = str(scenario.files.get("hourly", scenario.scenario_dir))
        time_col = _find_time_column(hourly)
        model_daily = daily_regret[(daily_regret.get("model") == scenario.model) & (daily_regret.get("quantile") == scenario.quantile)] if not daily_regret.empty else pd.DataFrame()
        if not model_daily.empty:
            model_daily = model_daily.sort_values("date").reset_index(drop=True)
        total_regret = pd.to_numeric(model_daily.get("daily_regret_eur", pd.Series(dtype=float)), errors="coerce").sum() if not model_daily.empty else math.nan
        for spec in TAIL_EVENT_SPECS:
            col = discover_column(hourly.columns, spec.candidate_columns, role=f"tail:{spec.tail_event_type}", source=source, lookup=lookup)
            base_row = {
                "model": scenario.model,
                "quantile": scenario.quantile,
                "tail_event_type": spec.tail_event_type,
                "target_family": spec.target_family,
                "direction": spec.direction,
                "tail_side": spec.tail_side,
                "column_used": col or "",
                "candidate_columns": "; ".join(spec.candidate_columns),
            }
            if col is None:
                rows.append(unavailable_row(scenario, spec, column_used=None, data_status="missing_tail_column"))
                continue
            if time_col is None:
                rows.append(unavailable_row(scenario, spec, column_used=col, data_status="missing_time_column"))
                continue
            if model_daily.empty:
                rows.append(unavailable_row(scenario, spec, column_used=col, data_status="missing_daily_regret"))
                continue
            values = pd.to_numeric(hourly[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            q = 0.05 if spec.tail_side == "low" else 0.95
            threshold = float(values.quantile(q)) if values.notna().any() else math.nan
            if not math.isfinite(threshold):
                rows.append(unavailable_row(scenario, spec, column_used=col, data_status="all_tail_values_missing"))
                continue
            mask = values.le(threshold) if spec.tail_side == "low" else values.ge(threshold)
            days = pd.to_datetime(hourly.loc[mask, time_col], errors="coerce").dt.date.astype(str).dropna().unique().tolist()
            tail_daily = model_daily[model_daily["date"].astype(str).isin(days)]
            tail_regret = pd.to_numeric(tail_daily.get("daily_regret_eur", pd.Series(dtype=float)), errors="coerce").sum() if not tail_daily.empty else 0.0
            rows.append(
                {
                    **base_row,
                    "threshold_used": threshold,
                    "n_periods": int(mask.sum()),
                    "n_tail_days": len(set(days)),
                    "model_profit_in_tail_periods": pd.to_numeric(tail_daily.get("realized_model_profit_eur", pd.Series(dtype=float)), errors="coerce").sum() if not tail_daily.empty else math.nan,
                    "benchmark_profit_in_tail_periods": pd.to_numeric(tail_daily.get("benchmark_profit_eur", pd.Series(dtype=float)), errors="coerce").sum() if not tail_daily.empty else math.nan,
                    "tail_regret_eur": tail_regret,
                    "tail_regret_share_of_total_regret": tail_regret / total_regret if math.isfinite(total_regret) and total_regret != 0 else math.nan,
                    "non_tail_regret_eur": total_regret - tail_regret if math.isfinite(total_regret) else math.nan,
                    "data_status": "available_daily_overlap",
                }
            )
    return pd.DataFrame(rows)


def build_top_regret_events(scenarios: list[Scenario], lookup: ColumnLookup) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = pd.DataFrame(
        [
            {"likely_driver": "tail_price_event", "rule": "absolute realized price is in a tail column and event regret is positive"},
            {"likely_driver": "activation_volume_surprise", "rule": "activation volume forecast/realization error columns are present and large"},
            {"likely_driver": "activation_price_surprise", "rule": "activation price forecast/realization error columns are present and large"},
            {"likely_driver": "non_executed_bid", "rule": "non-executed or rejected bid columns are nonzero"},
            {"likely_driver": "penalty_or_infeasibility", "rule": "fallback, infeasible, shortfall, missed, penalty or violation flags are present"},
            {"likely_driver": "residual_unknown", "rule": "no transparent rule matched"},
        ]
    )
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario.is_benchmark or scenario.hourly.empty:
            continue
        hourly = scenario.hourly
        model_col = discover_column(hourly.columns, ["model_profit_eur", "net_revenue_eur", "hourly_model_profit_eur"], role="top_events:model_profit", source=str(scenario.scenario_dir), lookup=lookup)
        bench_col = discover_column(hourly.columns, ["benchmark_profit_eur", "rhpf_profit_eur", "oracle_profit_eur"], role="top_events:benchmark_profit", source=str(scenario.scenario_dir), lookup=lookup)
        time_col = _find_time_column(hourly)
        if not model_col or not bench_col or not time_col:
            rows.append({"model": scenario.model, "quantile": scenario.quantile, "likely_driver": "unavailable", "data_status": "missing_event_level_profit_columns"})
            continue
        d = hourly[[time_col, model_col, bench_col]].copy()
        d["regret_eur"] = pd.to_numeric(d[bench_col], errors="coerce") - pd.to_numeric(d[model_col], errors="coerce")
        d = d.sort_values("regret_eur", ascending=False).head(20)
        for _, r in d.iterrows():
            rows.append(
                {
                    "timestamp": r[time_col],
                    "model": scenario.model,
                    "quantile": scenario.quantile,
                    "market": "",
                    "model_action": "",
                    "benchmark_action": "",
                    "realized_price": math.nan,
                    "forecast_price": math.nan,
                    "realized_activation_rate": math.nan,
                    "forecast_activation_rate": math.nan,
                    "model_profit_eur": _safe_float(r[model_col]),
                    "benchmark_profit_eur": _safe_float(r[bench_col]),
                    "regret_eur": _safe_float(r["regret_eur"]),
                    "likely_driver": "residual_unknown",
                    "data_status": "available",
                }
            )
    return pd.DataFrame(rows), rules


def _latex_escape(value: Any) -> str:
    text = "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
        .replace("$", r"\$")
    )


def write_latex_table(df: pd.DataFrame, path: Path, *, caption: str, label: str, max_rows: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    view = df.head(max_rows).copy()
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + "l" * max(1, len(view.columns)) + "}",
        r"\toprule",
        " & ".join(_latex_escape(c) for c in view.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in view.iterrows():
        cells: list[str] = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                cells.append("" if not math.isfinite(float(value)) else f"{float(value):,.4f}")
            else:
                cells.append(_latex_escape(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", rf"\caption{{{_latex_escape(caption)}}}", rf"\label{{{_latex_escape(label)}}}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_annualized_regret_latex(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "Model",
        "Quantile",
        "Annualized net profit",
        "RHPF annualized profit",
        "Regret vs RHPF",
        "Regret share",
        "Model/RHPF (%)",
    ]
    view = df[columns].copy() if set(columns).issubset(df.columns) else pd.DataFrame(columns=columns)

    def eur(value: Any) -> str:
        x = _safe_float(value)
        return "" if not math.isfinite(x) else f"{x:,.0f}"

    def pct(value: Any) -> str:
        x = _safe_float(value)
        return "" if not math.isfinite(x) else f"{x:,.1f}\\%"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Quantile & Annualized net profit & RHPF annualized profit & Regret vs RHPF & Regret share & Model/RHPF (\%) \\",
        r"\midrule",
    ]
    for _, row in view.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["Model"]),
                    _latex_escape(row["Quantile"]),
                    eur(row["Annualized net profit"]),
                    eur(row["RHPF annualized profit"]),
                    eur(row["Regret vs RHPF"]),
                    pct(row["Regret share"]),
                    pct(row["Model/RHPF (%)"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Annualized net profit and regret relative to RHPF. Regret share reports the percentage of RHPF annualized profit not achieved by the model.}",
            r"\label{tab:annualized_regret_vs_rhpf}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _plot_tail_share(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    apply_geo_style()
    path.parent.mkdir(parents=True, exist_ok=True)
    d = df.copy()
    if d.empty or "tail_regret_share_of_total_regret" not in d.columns:
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        ax.text(0.5, 0.5, "Tail regret share unavailable", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    d = d[pd.to_numeric(d["tail_regret_share_of_total_regret"], errors="coerce").notna()].copy()
    if not d.empty:
        d["_abs_share"] = pd.to_numeric(d["tail_regret_share_of_total_regret"], errors="coerce").abs()
        d = d.sort_values("_abs_share", ascending=False).head(30).sort_values(["model", "quantile", "target_family", "direction", "tail_side"])
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.28 * len(d))))
    if d.empty:
        ax.text(0.5, 0.5, "Tail regret share unavailable", ha="center", va="center")
        ax.axis("off")
    else:
        labels = d["model"].astype(str) + " " + d["quantile"].astype(str) + " | " + d["tail_event_type"].astype(str)
        colors = [get_model_color(str(m).lower()) if str(m).upper() != "RLQR" else get_model_color("linear") for m in d["model"]]
        ax.barh(labels, pd.to_numeric(d["tail_regret_share_of_total_regret"], errors="coerce") * 100.0, color=colors)
        ax.set_xlabel("Share of total regret during tail-event days (%)")
        ax.set_title("Regret Occurring on Tail-Event Days")
        ax.tick_params(axis="y", labelsize=7)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_cumulative_regret(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    apply_geo_style()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    if df.empty:
        ax.text(0.5, 0.5, "Daily regret unavailable", ha="center", va="center")
        ax.axis("off")
    else:
        for (model, quantile), group in df.groupby(["model", "quantile"], sort=False):
            g = group.copy()
            g["date"] = pd.to_datetime(g["date"], errors="coerce")
            color_key = "linear" if model == "RLQR" else str(model).lower()
            ax.plot(g["date"], pd.to_numeric(g["cumulative_regret_eur"], errors="coerce"), label=f"{model} {quantile}", color=get_model_color(color_key), linewidth=1.8)
        ax.axhline(0.0, color=THESIS_PALETTE["neutral_dark"], linewidth=0.8)
        ax.set_ylabel("Cumulative regret (EUR)")
        ax.set_title("Cumulative Regret Relative to Benchmark")
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_debug(out_root: Path, lookup: ColumnLookup, scenarios: list[Scenario], top_event_rules: pd.DataFrame) -> None:
    debug = out_root / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    discovered = {
        "scenario_count": len(scenarios),
        "scenarios": [
            {
                "folder": s.folder,
                "model": s.model,
                "quantile": s.quantile,
                "strategy": s.strategy,
                "is_benchmark": s.is_benchmark,
                "scenario_dir": str(s.scenario_dir),
                "files": {k: str(v) for k, v in sorted(s.files.items())},
            }
            for s in scenarios
        ],
    }
    (debug / "discovered_files.json").write_text(json.dumps(discovered, indent=2), encoding="utf-8")
    pd.DataFrame(lookup.mapping_rows).to_json(debug / "column_mapping.json", orient="records", indent=2)
    pd.DataFrame(lookup.missing_rows).to_csv(debug / "missing_columns.csv", index=False)
    pd.DataFrame(lookup.daily_alignment_rows).to_csv(debug / "daily_alignment.csv", index=False)
    top_event_rules.to_csv(debug / "top_event_driver_rules.csv", index=False)
    (debug / "diagnostic_warnings.txt").write_text("\n".join(lookup.warnings) + ("\n" if lookup.warnings else ""), encoding="utf-8")


def write_summary(
    out_root: Path,
    *,
    run_name: str,
    scenarios: list[Scenario],
    bridge: pd.DataFrame,
    lookup: ColumnLookup,
    benchmark: str,
) -> None:
    lines = [
        f"# Regret Driver Diagnostics: {run_name}",
        "",
        "These diagnostics are descriptive and not causal. A driver label means the regret occurred during or was associated with that condition; it does not prove structural causality without counterfactual reruns.",
        "",
        "## Discovery",
        f"- Scenarios discovered: {len(scenarios)}",
        f"- Models: {', '.join(sorted({s.model for s in scenarios if not s.is_benchmark})) or 'none'}",
        f"- Quantiles: {', '.join(sorted({s.quantile for s in scenarios if not s.is_benchmark})) or 'none'}",
        f"- Benchmark requested: {benchmark}",
        "",
        "## Regret Bridge",
    ]
    if bridge.empty:
        lines.append("- Regret bridge unavailable.")
    else:
        view = bridge.sort_values("total_regret_eur", ascending=False, na_position="last").head(10)
        for _, row in view.iterrows():
            lines.append(
                f"- {row.get('model')} {row.get('quantile')}: total regret {row.get('total_regret_eur'):.2f} EUR; "
                f"realized {row.get('realized_profit_eur'):.2f} EUR; benchmark {row.get('benchmark_profit_eur'):.2f} EUR; "
                f"benchmark source `{row.get('benchmark_column_used')}`."
            )
    lines.extend(
        [
            "",
            "## Missing Data Limitations",
            f"- Missing column mappings recorded: {len(lookup.missing_rows)}",
            f"- Warnings recorded: {len(lookup.warnings)}",
            "",
            "Use `debug/missing_columns.csv` and `debug/column_mapping.json` before interpreting unavailable components.",
        ]
    )
    (out_root / "regret_driver_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")
    out_root = Path(args.out_dir) / run_dir.name
    for name in ["tables", "figures", "latex", "debug"]:
        (out_root / name).mkdir(parents=True, exist_ok=True)
    models = {m.strip().lower() for m in str(args.models).split(",") if m.strip()}
    quantiles = {q.strip().lower() for q in str(args.quantiles).split(",") if q.strip()}
    lookup = ColumnLookup()
    scenarios = discover_scenarios(run_dir, models=models, quantiles=quantiles)
    if not scenarios:
        raise FileNotFoundError(f"No scenario outputs discovered under {run_dir}. Expected files like */multi/*/backtest_summary.json.")
    print(f"[DISCOVERY] run={run_dir.name}")
    print(f"[DISCOVERY] scenarios={len(scenarios)}")
    print(f"[DISCOVERY] models={sorted({s.model for s in scenarios if not s.is_benchmark})}")
    print(f"[DISCOVERY] quantiles={sorted({s.quantile for s in scenarios if not s.is_benchmark})}")
    print(f"[DISCOVERY] strategies={sorted({s.strategy for s in scenarios})}")

    bridge = build_regret_bridge(scenarios, benchmark=str(args.benchmark), lookup=lookup, strict=bool(args.strict))
    annualized_regret = build_annualized_regret_table(scenarios, benchmark=str(args.benchmark), lookup=lookup)
    pnl = build_pnl_by_market(scenarios, lookup)
    planned = build_planned_vs_realized(scenarios, lookup)
    execution = build_bid_execution(scenarios, lookup)
    activation = build_activation_surprise(scenarios, lookup)
    penalty = build_penalty_infeasibility(scenarios, lookup)
    daily = build_daily_regret(scenarios, benchmark=str(args.benchmark), lookup=lookup)
    tail = build_tail_event_regret(daily, scenarios, lookup)
    top_events, top_event_rules = build_top_regret_events(scenarios, lookup)

    tables = out_root / "tables"
    latex = out_root / "latex"
    figures = out_root / "figures"
    _write_csv(bridge, tables / "regret_bridge.csv")
    _write_csv(
        annualized_regret[
            [
                "Model",
                "Quantile",
                "Annualized net profit",
                "RHPF annualized profit",
                "Regret vs RHPF",
                "Regret share",
                "Model/RHPF (%)",
            ]
        ],
        tables / "annualized_regret_vs_rhpf.csv",
    )
    _write_csv(pnl, tables / "pnl_by_market.csv")
    _write_csv(planned, tables / "planned_vs_realized_by_component.csv")
    _write_csv(execution, tables / "bid_execution_diagnostics.csv")
    _write_csv(activation, tables / "activation_surprise_diagnostics.csv")
    _write_csv(penalty, tables / "penalty_infeasibility_diagnostics.csv")
    _write_csv(tail, tables / "tail_event_regret.csv")
    _write_csv(daily, tables / "daily_regret.csv")
    _write_csv(top_events, tables / "top_regret_events.csv")
    write_latex_table(bridge, latex / "regret_bridge.tex", caption="Regret bridge by model and quantile.", label="tab:regret_bridge")
    write_annualized_regret_latex(annualized_regret, latex / "annualized_regret_vs_rhpf.tex")
    write_latex_table(pnl, latex / "pnl_by_market.tex", caption="Realized PnL decomposition by market.", label="tab:pnl_by_market")
    write_latex_table(planned, latex / "planned_vs_realized_by_component.tex", caption="Planned versus realized component gaps.", label="tab:planned_vs_realized_by_component")
    write_latex_table(tail, latex / "tail_event_regret.tex", caption="Regret occurring during tail periods.", label="tab:tail_event_regret")
    write_latex_table(top_events, latex / "top_regret_events.tex", caption="Top event-level regret rows where available.", label="tab:top_regret_events")
    _plot_tail_share(tail, figures / "tail_regret_share.png")
    _plot_cumulative_regret(daily, figures / "cumulative_regret.png")
    _write_debug(out_root, lookup, scenarios, top_event_rules)
    write_summary(out_root, run_name=run_dir.name, scenarios=scenarios, bridge=bridge, lookup=lookup, benchmark=str(args.benchmark))
    print(f"[OK] Wrote regret-driver diagnostics to {out_root}")
    return out_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--benchmark", default="rhpf", choices=["rhpf", "naive"])
    parser.add_argument("--models", default="xgb,tft,linear")
    parser.add_argument("--quantiles", default="p10,p30,p50,p70,p90")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
