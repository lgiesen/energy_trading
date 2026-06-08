#!/usr/bin/env python3
"""
Analyze one battery backtest simulation run and write an archive report.

Usage:
  python3 evaluation/analyze_backtest_run.py artifacts/simulation_runs/xgb_da_mid
  python3 evaluation/analyze_backtest_run.py artifacts/simulation_runs/xgb_da_short
  python3 evaluation/analyze_backtest_run.py artifacts/simulation_runs/xgb_da_mid_no_id
  python3 evaluation/analyze_backtest_run.py artifacts/simulation_runs/xgb_bcm_mid
  python3 evaluation/analyze_backtest_run.py artifacts/simulation_runs/xgb_bcm_mid_p70
  python3 evaluation/analyze_backtest_run.py artifacts/simulation_runs/xgb_bem_mid
  python3 evaluation/analyze_backtest_run.py artifacts/simulation_runs/xgb_multi_mid
  python3 evaluation/analyze_backtest_run.py artifacts/simulation_runs/xgb_afrr_mid

Outputs:
  archive/backtesting_analysis/<run_name>_analysis.md
  archive/backtesting_analysis/<run_name>_analysis.json

This script is read-only with respect to simulation artifacts.
It does not run simulations, backtests or optimizers.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DEFAULT_ARTIFACT_ROOT = Path("artifacts/simulation_runs")
DEFAULT_OUT_DIR = Path("archive/backtesting_analysis")


SUMMARY_KEYS_MAIN = [
    "strategy",
    "trading_strategy",
    "scenario",
    "model",
    "split",
    "timeframe_utc",
    "timeframe_total_days",
    "simulation_valid",
    "thesis_reportable",
    "invalid_reason",
    "realized_total_pnl_eur",
    "realized_pnl_excl_terminal_eur",
    "predicted_total_pnl_eur",
    "predicted_pnl_excl_terminal_eur",
    "naive_total_pnl_eur",
    "naive_realized_pnl_eur",
    "rolling_perfect_foresight_same_rules_total_pnl_eur",
    "comparable_rolling_perfect_foresight_same_rules_market_pnl_eur",
    "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur",
    "global_pf_available",
    "global_pf_verified_upper_bound",
    "global_pf_solver_status",
    "global_pf_failure_reason",
    "fallback_used",
    "fallback_mode_counts",
    "optimization_error_code_counts",
    "first_infeasible_timestamp_utc",
    "first_fallback_timestamp_utc",
    "first_terminal_soc_conflict_timestamp_utc",
    "final_soc_mwh",
    "target_final_soc_mwh",
    "final_soc_shortfall_mwh",
    "final_soc_surplus_mwh",
    "final_soc_physical_check_pass",
    "final_soc_economic_repair_check_pass",
    "id_recourse_mode",
    "id_recourse_events_total",
    "id_recourse_events_by_reason",
    "terminal_soc_recovery_id_mwh",
    "terminal_soc_repair_cost_eur",
]


DEBUG_FILES = [
    "optimization_failure_debug.csv",
    "optimization_infeasibility_attribution.csv",
    "hard_final_soc_infeasibility_debug.csv",
    "reserve_commitment_debug.csv",
    "solver_failure_diagnostics.csv",
    "milp_event_log.csv",
    "state_machine_audit.csv",
    "performance_metric_reconciliation_debug.csv",
]


def safe_name(value: str) -> str:
    value = value.strip().replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)


def resolve_run_path(arg: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    candidate = DEFAULT_ARTIFACT_ROOT / arg
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find run path '{arg}' or '{candidate}'. "
        "Pass either a full artifact path or a folder name under artifacts/simulation_runs."
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"__read_error__": str(exc)}


def read_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return None


def read_parquet_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def find_scenario_dirs(root: Path) -> list[Path]:
    if (root / "backtest_summary.json").exists():
        return [root]
    return sorted(p.parent for p in root.rglob("backtest_summary.json"))


def num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def col_exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def sum_col(df: pd.DataFrame, col: str) -> float:
    return float(num_series(df, col).sum()) if col in df.columns else 0.0


def max_col(df: pd.DataFrame, col: str) -> float:
    return float(num_series(df, col).max()) if col in df.columns else 0.0


def min_col(df: pd.DataFrame, col: str) -> float:
    return float(num_series(df, col).min()) if col in df.columns else 0.0


def weighted_average(value_eur: float, volume_mwh: float) -> float | None:
    if abs(volume_mwh) < 1e-12:
        return None
    return value_eur / volume_mwh


def fmt_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.6g}"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def df_to_md(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "_No rows._"
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
    try:
        return d.to_markdown(index=False)
    except Exception:
        return "```text\n" + d.to_string(index=False) + "\n```"


def dict_table(d: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{"field": k, "value": fmt_value(v)} for k, v in d.items()])


def value_counts_table(series: pd.Series, name: str) -> pd.DataFrame:
    counts = series.fillna("").astype(str).value_counts(dropna=False)
    return pd.DataFrame([{name: k, "count": int(v)} for k, v in counts.items()])


def select_existing(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def nonzero_numeric_sums(df: pd.DataFrame, pattern: str | None = None, max_items: int = 120) -> pd.DataFrame:
    rows = []
    regex = re.compile(pattern) if pattern else None
    for c in df.columns:
        if regex and not regex.search(c):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if not s.notna().any():
            continue
        filled = s.fillna(0.0)
        sm = float(filled.sum())
        mx = float(filled.max())
        mn = float(filled.min())
        nonnull = int(s.notna().sum())
        if abs(sm) > 1e-9 or abs(mx) > 1e-9 or abs(mn) > 1e-9:
            rows.append(
                {
                    "column": c,
                    "nonnull": nonnull,
                    "sum": sm,
                    "min": mn,
                    "max": mx,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_sum"] = out["sum"].abs()
    out = out.sort_values(["abs_sum", "column"], ascending=[False, True]).drop(columns=["abs_sum"])
    return out.head(max_items)


def top_or_bottom_hours(
    df: pd.DataFrame,
    sort_col: str,
    ascending: bool,
    cols: list[str],
    n: int = 30,
) -> pd.DataFrame:
    if sort_col not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d[sort_col] = pd.to_numeric(d[sort_col], errors="coerce")
    d = d[d[sort_col].notna()]
    if d.empty:
        return pd.DataFrame()
    cols = select_existing(d, cols)
    return d.sort_values(sort_col, ascending=ascending)[cols].head(n)


def scenario_label(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return path.name


def load_strategy_overview(root: Path) -> pd.DataFrame | None:
    for name in ["strategy_overview.csv", "quantile_sweep_summary.csv", "performance_metrics_all_scenarios.csv"]:
        p = root / name
        if p.exists():
            df = read_csv_safe(p)
            if df is not None:
                return df
    return None


def analyze_da(df: pd.DataFrame) -> dict[str, Any]:
    buy = sum_col(df, "real_da_buy_mwh")
    sell = sum_col(df, "real_da_sell_mwh")
    revenue = sum_col(df, "real_revenue_da_eur")
    cost = sum_col(df, "real_cost_da_eur")
    gross = revenue - cost

    return {
        "real_da_buy_mwh": buy,
        "real_da_sell_mwh": sell,
        "real_da_revenue_eur": revenue,
        "real_da_cost_eur": cost,
        "real_da_gross_eur": gross,
        "avg_real_buy_price_eur_mwh": weighted_average(cost, buy),
        "avg_real_sell_price_eur_mwh": weighted_average(revenue, sell),
        "both_buy_sell_hours": int(((num_series(df, "real_da_buy_mwh") > 1e-9) & (num_series(df, "real_da_sell_mwh") > 1e-9)).sum()),
        "real_degradation_cost_eur": sum_col(df, "real_degradation_cost_eur"),
        "real_transaction_cost_eur": sum_col(df, "real_transaction_cost_eur"),
        "real_aux_cost_eur": sum_col(df, "real_aux_cost_eur"),
        "real_auxiliary_cost_eur": sum_col(df, "real_auxiliary_cost_eur"),
        "real_pnl_id_eur": sum_col(df, "real_pnl_id_eur"),
        "real_pnl_eur": sum_col(df, "real_pnl_eur"),
        "pred_pnl_eur": sum_col(df, "pred_pnl_eur"),
    }


def analyze_aux_deg(df: pd.DataFrame) -> dict[str, Any]:
    grid_throughput = (
        num_series(df, "real_da_buy_mwh")
        + num_series(df, "real_da_sell_mwh")
        + num_series(df, "real_id_buy_mwh")
        + num_series(df, "real_id_sell_mwh")
        + num_series(df, "real_bcm_linked_pos_activation_mwh")
        + num_series(df, "real_bcm_linked_neg_activation_mwh")
        + num_series(df, "real_bem_only_pos_activation_mwh")
        + num_series(df, "real_bem_only_neg_activation_mwh")
        + num_series(df, "real_delivered_activation_pos_mwh")
        + num_series(df, "real_delivered_activation_neg_mwh")
    )

    internal_cols = [
        "real_da_charge_internal_mwh",
        "real_da_discharge_internal_mwh",
        "real_id_charge_internal_mwh",
        "real_id_discharge_internal_mwh",
        "real_act_pos_internal_mwh",
        "real_act_neg_internal_mwh",
        "real_activation_pos_internal_mwh",
        "real_activation_neg_internal_mwh",
    ]
    internal_throughput = pd.Series(0.0, index=df.index)
    for c in internal_cols:
        internal_throughput = internal_throughput + num_series(df, c)

    deg = sum_col(df, "real_degradation_cost_eur")
    aux_energy = sum_col(df, "real_aux_energy_mwh")
    aux_cost = sum_col(df, "real_aux_cost_eur")
    if abs(aux_cost) < 1e-12 and "real_auxiliary_cost_eur" in df.columns:
        aux_cost = sum_col(df, "real_auxiliary_cost_eur")

    return {
        "grid_throughput_proxy_mwh": float(grid_throughput.sum()),
        "internal_throughput_exported_mwh": float(internal_throughput.sum()),
        "degradation_cost_eur": deg,
        "deg_per_grid_mwh_proxy": weighted_average(deg, float(grid_throughput.sum())),
        "deg_per_internal_mwh_exported": weighted_average(deg, float(internal_throughput.sum())),
        "aux_energy_mwh": aux_energy,
        "aux_cost_eur": aux_cost,
        "aux_avg_price_eur_mwh": weighted_average(aux_cost, aux_energy),
        "aux_hours": int((num_series(df, "real_aux_energy_mwh") > 1e-9).sum()),
        "max_hourly_aux_mwh": max_col(df, "real_aux_energy_mwh"),
    }


def analyze_bcm(df: pd.DataFrame) -> dict[str, Any]:
    prefixes = ["real", "pred", "naive", "perfect_foresight", "pf", "global_perfect_foresight"]
    out: dict[str, Any] = {}
    for p in prefixes:
        for direction in ["pos", "neg"]:
            for kind in ["submitted", "locked", "executed", "awarded"]:
                candidates = [
                    f"{p}_{kind}_bcm_capacity_{direction}_mw",
                    f"{p}_{kind}_afrr_{direction}_mw",
                    f"{p}_{kind}_reserve_{direction}_mw",
                    f"{p}_afrr_cap_{direction}_awarded_mw",
                    f"{p}_awarded_capacity_{direction}_mw",
                ]
                out[f"{p}_{kind}_{direction}_capacity_sum"] = sum(sum_col(df, c) for c in candidates if c in df.columns)
        for metric in [
            "revenue_capacity_eur",
            "bcm_capacity_revenue_eur",
            "bcm_linked_activation_revenue_eur",
            "revenue_activation_eur",
            "bcm_total_revenue_eur",
        ]:
            out[f"{p}_{metric}"] = sum_col(df, f"{p}_{metric}")

        for c in [
            f"{p}_settlement_cap_bid_price_pos_eur_mw",
            f"{p}_settlement_cap_bid_price_neg_eur_mw",
            f"{p}_bcm_capacity_bid_price_pos_eur_per_mw_h",
            f"{p}_bcm_capacity_bid_price_neg_eur_per_mw_h",
        ]:
            if c in df.columns:
                out[f"{c}_max"] = max_col(df, c)
                out[f"{c}_sum"] = sum_col(df, c)

    return out


def collect_warnings(summary: dict[str, Any], df: pd.DataFrame | None, scenario_dir: Path) -> list[str]:
    warnings: list[str] = []

    invalid = str(summary.get("invalid_reason", "") or "")
    if invalid:
        warnings.append(f"Run invalid/non-reportable: {invalid}")

    if float(summary.get("fallback_used", 0) or 0) > 0:
        warnings.append("fallback_used > 0: accepted path used fallback/repair logic.")

    if float(summary.get("final_soc_shortfall_mwh", 0) or 0) > 1e-6:
        warnings.append(f"Final SoC shortfall: {summary.get('final_soc_shortfall_mwh')} MWh.")

    if float(summary.get("global_pf_available", 1) or 0) == 0:
        warnings.append("Global PF unavailable; do not use GHPF as upper bound.")

    if df is not None and not df.empty:
        real_locked = (
            sum_col(df, "real_locked_bcm_capacity_pos_mw")
            + sum_col(df, "real_locked_bcm_capacity_neg_mw")
            + sum_col(df, "real_executed_bcm_capacity_pos_mw")
            + sum_col(df, "real_executed_bcm_capacity_neg_mw")
        )
        real_cap_rev = sum_col(df, "real_revenue_capacity_eur") + sum_col(df, "real_bcm_capacity_revenue_eur")
        if real_locked > 1e-9 and abs(real_cap_rev) < 1e-9:
            warnings.append("BCM capacity appears locked/executed but capacity revenue is zero; check capacity bid price propagation.")

        aux_energy = sum_col(df, "real_aux_energy_mwh")
        aux_cost = sum_col(df, "real_aux_cost_eur") + sum_col(df, "real_auxiliary_cost_eur")
        if aux_energy > 1e-9 and abs(aux_cost) < 1e-9:
            warnings.append("Aux energy > 0 but aux cost is zero or missing; check column names and aux pricing.")

        deg = sum_col(df, "real_degradation_cost_eur")
        grid = analyze_aux_deg(df)["grid_throughput_proxy_mwh"]
        internal = analyze_aux_deg(df)["internal_throughput_exported_mwh"]
        if deg > 1e-9 and grid and internal == 0:
            warnings.append("Degradation exists but exported internal throughput is zero; add/verify internal-throughput diagnostics.")

        if "da_precommit_selected_incumbent" in df.columns:
            cand = num_series(df, "da_precommit_candidate_selection_pnl_eur")
            inc = num_series(df, "da_precommit_incumbent_selection_pnl_eur")
            selected = df["da_precommit_selected_incumbent"].fillna("").astype(str)
            bad = ((cand < inc - 1e-6) & (selected != "no_trade") & (selected != "current")).sum()
            if int(bad) > 0:
                warnings.append(f"DA precommit: {int(bad)} rows where candidate was worse than incumbent but not rejected.")

    return warnings


def analyze_scenario(scenario_dir: Path, root: Path, max_rows: int) -> tuple[str, dict[str, Any]]:
    summary_path = scenario_dir / "backtest_summary.json"
    hourly_path = scenario_dir / "backtest_hourly.parquet"
    plan_history_path = scenario_dir / "backtest_plan_history.parquet"

    summary = read_json(summary_path)
    hourly = read_parquet_safe(hourly_path)
    plan_history = read_parquet_safe(plan_history_path)

    scenario_name = scenario_label(scenario_dir, root)
    json_out: dict[str, Any] = {
        "scenario": scenario_name,
        "scenario_dir": str(scenario_dir),
        "summary_path": str(summary_path),
        "hourly_path": str(hourly_path) if hourly_path.exists() else None,
        "plan_history_path": str(plan_history_path) if plan_history_path.exists() else None,
        "summary_main": {k: summary.get(k) for k in SUMMARY_KEYS_MAIN if k in summary},
        "warnings": collect_warnings(summary, hourly, scenario_dir),
    }

    parts: list[str] = []
    parts.append(f"## Scenario: `{scenario_name}`\n")

    if json_out["warnings"]:
        parts.append("### Warnings\n")
        for w in json_out["warnings"]:
            parts.append(f"- {w}")
        parts.append("")

    main = {k: summary.get(k) for k in SUMMARY_KEYS_MAIN if k in summary}
    parts.append("### Summary fields\n")
    parts.append(df_to_md(dict_table(main), max_rows=max_rows))
    parts.append("")

    if hourly is None:
        parts.append("### Hourly file\n")
        parts.append("_No `backtest_hourly.parquet` found or readable._\n")
        return "\n".join(parts), json_out

    json_out["hourly_rows"] = int(len(hourly))
    json_out["hourly_columns"] = int(len(hourly.columns))
    if "timestamp_utc" in hourly.columns:
        json_out["hourly_min_timestamp_utc"] = str(hourly["timestamp_utc"].min())
        json_out["hourly_max_timestamp_utc"] = str(hourly["timestamp_utc"].max())

    parts.append("### Hourly file overview\n")
    parts.append(df_to_md(dict_table({
        "rows": len(hourly),
        "columns": len(hourly.columns),
        "min_timestamp_utc": hourly["timestamp_utc"].min() if "timestamp_utc" in hourly.columns else "",
        "max_timestamp_utc": hourly["timestamp_utc"].max() if "timestamp_utc" in hourly.columns else "",
    })))
    parts.append("")

    da = analyze_da(hourly)
    json_out["da"] = da
    parts.append("### DA analysis\n")
    parts.append(df_to_md(dict_table(da)))
    parts.append("")

    aux_deg = analyze_aux_deg(hourly)
    json_out["aux_degradation"] = aux_deg
    parts.append("### Auxiliary and degradation sanity checks\n")
    parts.append(df_to_md(dict_table(aux_deg)))
    parts.append("")

    bcm = analyze_bcm(hourly)
    json_out["bcm_afrr"] = bcm
    bcm_nonzero = {k: v for k, v in bcm.items() if isinstance(v, (int, float)) and abs(float(v)) > 1e-9}
    parts.append("### BCM / aFRR capacity and activation analysis\n")
    parts.append(df_to_md(dict_table(bcm_nonzero), max_rows=max_rows))
    parts.append("")

    parts.append("### Main nonzero monetary columns\n")
    money_pattern = r"(pnl|revenue|cost|penalty|terminal|transaction|degradation|aux).*(_eur|eur$)"
    parts.append(df_to_md(nonzero_numeric_sums(hourly, money_pattern), max_rows=max_rows))
    parts.append("")

    parts.append("### Main nonzero energy / power / volume columns\n")
    volume_pattern = r"(mwh|_mw$|capacity|charge|discharge|buy|sell|activation)"
    parts.append(df_to_md(nonzero_numeric_sums(hourly, volume_pattern), max_rows=max_rows))
    parts.append("")

    if "optimization_error_code" in hourly.columns:
        parts.append("### Optimization error codes in hourly file\n")
        parts.append(df_to_md(value_counts_table(hourly["optimization_error_code"], "optimization_error_code")))
        parts.append("")

    if "optimizer_fallback_used" in hourly.columns:
        bad = hourly[
            (num_series(hourly, "optimizer_fallback_used") > 0)
            | (~hourly.get("optimization_error_code", pd.Series("ok", index=hourly.index)).fillna("ok").isin(["ok", "ok_deterministic_noop"]))
        ]
        cols = select_existing(hourly, [
            "timestamp_utc",
            "optimization_error_code",
            "optimizer_fallback_used",
            "fallback_mode",
            "infeasibility_driver",
            "real_soc_start_mwh",
            "real_soc_mwh",
            "real_da_buy_mwh",
            "real_da_sell_mwh",
            "real_id_buy_mwh",
            "real_id_sell_mwh",
            "real_id_recourse_reason",
            "real_pnl_eur",
            "pred_pnl_eur",
        ])
        parts.append("### Non-OK / fallback hourly rows\n")
        parts.append(df_to_md(bad[cols] if cols else bad, max_rows=max_rows))
        parts.append("")

    if "real_pnl_eur" in hourly.columns:
        worst_cols = select_existing(hourly, [
            "timestamp_utc",
            "pred_da_price",
            "target_da_price",
            "real_soc_start_mwh",
            "real_soc_mwh",
            "real_da_buy_mwh",
            "real_da_sell_mwh",
            "real_cost_da_eur",
            "real_revenue_da_eur",
            "real_degradation_cost_eur",
            "real_transaction_cost_eur",
            "real_aux_cost_eur",
            "real_pnl_id_eur",
            "real_id_buy_mwh",
            "real_id_sell_mwh",
            "real_id_recourse_reason",
            "real_pnl_eur",
            "pred_pnl_eur",
        ])
        parts.append("### Worst realized PnL hours\n")
        parts.append(df_to_md(top_or_bottom_hours(hourly, "real_pnl_eur", True, worst_cols, n=max_rows), max_rows=max_rows))
        parts.append("")

    if "pred_pnl_eur" in hourly.columns:
        pred_cols = select_existing(hourly, [
            "timestamp_utc",
            "pred_da_price",
            "target_da_price",
            "real_da_buy_mwh",
            "real_da_sell_mwh",
            "pred_pnl_eur",
            "real_pnl_eur",
            "real_soc_start_mwh",
            "real_soc_mwh",
        ])
        parts.append("### Worst predicted PnL hours\n")
        parts.append(df_to_md(top_or_bottom_hours(hourly, "pred_pnl_eur", True, pred_cols, n=max_rows), max_rows=max_rows))
        parts.append("")

    # DA incumbent guard diagnostics
    if "da_precommit_selected_incumbent" in hourly.columns:
        parts.append("### DA precommit / incumbent guard diagnostics\n")
        parts.append("#### Selected incumbent counts\n")
        parts.append(df_to_md(value_counts_table(hourly["da_precommit_selected_incumbent"], "selected_incumbent")))
        parts.append("")
        if "da_precommit_selection_reason" in hourly.columns:
            parts.append("#### Selection reason counts\n")
            parts.append(df_to_md(value_counts_table(hourly["da_precommit_selection_reason"], "selection_reason")))
            parts.append("")
        cols = select_existing(hourly, [
            "timestamp_utc",
            "da_precommit_selected_incumbent",
            "da_precommit_selection_reason",
            "da_precommit_candidate_selection_pnl_eur",
            "da_precommit_incumbent_selection_pnl_eur",
            "da_precommit_candidate_minus_incumbent_eur",
            "da_precommit_no_trade_feasible",
            "da_precommit_candidate_feasible",
            "da_precommit_selection_pnl_basis",
            "real_da_buy_mwh",
            "real_da_sell_mwh",
            "real_pnl_eur",
            "pred_pnl_eur",
        ])
        mask = hourly[cols].notna().any(axis=1) if cols else pd.Series(False, index=hourly.index)
        parts.append("#### DA precommit rows\n")
        parts.append(df_to_md(hourly.loc[mask, cols], max_rows=max_rows))
        parts.append("")

    # BCM zero reasons and retry reasons
    reason_cols = [
        "bcm_precommit_zero_reason",
        "bcm_zero_reason",
        "retry_reduction_reason",
        "naive_bcm_precommit_zero_reason",
        "pf_bcm_precommit_zero_reason",
        "perfect_foresight_bcm_precommit_zero_reason",
    ]
    existing_reason_cols = [c for c in reason_cols if c in hourly.columns]
    if existing_reason_cols:
        parts.append("### BCM/precommit zero and retry reasons\n")
        for c in existing_reason_cols:
            parts.append(f"#### `{c}`\n")
            parts.append(df_to_md(value_counts_table(hourly[c], c), max_rows=max_rows))
            parts.append("")
        cols = select_existing(hourly, [
            "timestamp_utc",
            "bcm_precommit_zero_reason",
            "bcm_zero_reason",
            "retry_reduction_reason",
            "reserve_retry_factor",
            "bcm_precommit_terminal_soc_shortfall_mwh",
            "bcm_precommit_margin_after_retry_mwh",
            "real_locked_bcm_capacity_pos_mw",
            "real_locked_bcm_capacity_neg_mw",
            "real_revenue_capacity_eur",
            "real_bcm_linked_activation_revenue_eur",
        ])
        parts.append("#### BCM diagnostic rows with reasons\n")
        if cols:
            m = hourly[existing_reason_cols].fillna("").astype(str).apply(lambda x: x.str.len() > 0).any(axis=1)
            parts.append(df_to_md(hourly.loc[m, cols], max_rows=max_rows))
        else:
            parts.append("_No matching columns._")
        parts.append("")

    # PF / perfect foresight action columns
    pf_cols = [c for c in hourly.columns if c.startswith(("pf_", "perfect_foresight_", "global_perfect_foresight_"))]
    pf_nonzero = nonzero_numeric_sums(hourly[pf_cols], None) if pf_cols else pd.DataFrame()
    parts.append("### Perfect foresight / benchmark action and settlement columns\n")
    parts.append(df_to_md(pf_nonzero, max_rows=max_rows))
    parts.append("")

    # Terminal / final SoC
    terminal_keys = {k: v for k, v in summary.items() if any(x in k.lower() for x in [
        "final_soc", "terminal", "repair", "recourse", "shortfall", "soc_"
    ])}
    if terminal_keys:
        parts.append("### Terminal SoC / repair / recourse summary fields\n")
        parts.append(df_to_md(dict_table(terminal_keys), max_rows=max_rows))
        parts.append("")

    tail_cols = select_existing(hourly, [
        "timestamp_utc",
        "optimization_error_code",
        "optimizer_fallback_used",
        "real_soc_start_mwh",
        "real_soc_mwh",
        "real_da_buy_mwh",
        "real_da_sell_mwh",
        "real_id_buy_mwh",
        "real_id_sell_mwh",
        "real_id_recourse_reason",
        "real_aux_energy_mwh",
        "real_aux_cost_eur",
        "real_pnl_eur",
    ])
    if tail_cols:
        parts.append("### Last 24 hourly rows\n")
        parts.append(df_to_md(hourly[tail_cols].tail(24), max_rows=24))
        parts.append("")

    # Plan history EV/objective diagnostics
    if plan_history is not None:
        parts.append("### Plan history EV/objective diagnostics\n")
        ev_cols = [c for c in plan_history.columns if any(x in c.lower() for x in [
            "ev_", "objective", "terminal", "coef", "candidate", "incumbent", "replay"
        ])]
        if ev_cols:
            parts.append("#### Nonzero EV/objective plan-history columns\n")
            parts.append(df_to_md(nonzero_numeric_sums(plan_history[ev_cols], None), max_rows=max_rows))
            parts.append("")
            show_cols = select_existing(plan_history, [
                "timestamp_utc",
                "ev_da_charge_eur",
                "ev_da_discharge_eur",
                "ev_terminal_soc_credit_eur",
                "ev_objective_rebuild_eur",
                "objective_value_eur",
                "da_precommit_candidate_predicted_pnl_eur",
                "da_precommit_incumbent_predicted_pnl_eur",
                "da_precommit_selected_incumbent",
            ])
            if show_cols:
                parts.append("#### Tail of selected plan-history diagnostics\n")
                parts.append(df_to_md(plan_history[show_cols].tail(max_rows), max_rows=max_rows))
                parts.append("")
        else:
            parts.append("_No EV/objective plan-history columns found._\n")

    # Debug files
    parts.append("### Debug files\n")
    for name in DEBUG_FILES:
        p = scenario_dir / name
        if not p.exists():
            continue
        dbg = read_csv_safe(p)
        parts.append(f"#### `{name}`\n")
        if dbg is None:
            parts.append("_Exists, but could not read._\n")
            continue
        parts.append(f"Rows: `{len(dbg)}`, columns: `{len(dbg.columns)}`\n")
        if "optimization_error_code" in dbg.columns:
            parts.append("Error-code counts:\n")
            parts.append(df_to_md(value_counts_table(dbg["optimization_error_code"], "optimization_error_code")))
            parts.append("")
        if "infeasibility_driver" in dbg.columns:
            parts.append("Infeasibility-driver counts:\n")
            parts.append(df_to_md(value_counts_table(dbg["infeasibility_driver"], "infeasibility_driver")))
            parts.append("")
        if "suspected_infeasibility_driver" in dbg.columns:
            parts.append("Suspected-driver counts:\n")
            parts.append(df_to_md(value_counts_table(dbg["suspected_infeasibility_driver"], "suspected_driver")))
            parts.append("")
        parts.append("First rows:\n")
        parts.append(df_to_md(dbg, max_rows=max_rows))
        parts.append("")

    return "\n".join(parts), json_out


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze battery backtest simulation artifacts.")
    parser.add_argument("run", help="Run name under artifacts/simulation_runs or full path to run/scenario directory.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory. Default: archive/backtesting_analysis")
    parser.add_argument("--output-name", default=None, help="Optional output filename stem without extension.")
    parser.add_argument("--max-rows", type=int, default=40, help="Max rows per displayed table.")
    args = parser.parse_args()

    try:
        root = resolve_run_path(args.run)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_stem = args.output_name or safe_name(root.name)
    md_path = out_dir / f"{output_stem}_analysis.md"
    json_path = out_dir / f"{output_stem}_analysis.json"

    scenario_dirs = find_scenario_dirs(root)
    if not scenario_dirs:
        print(f"ERROR: No scenario folders with backtest_summary.json found under {root}", file=sys.stderr)
        return 3

    report_parts: list[str] = []
    machine: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_arg": args.run,
        "root": str(root),
        "scenario_count": len(scenario_dirs),
        "scenarios": [],
    }

    report_parts.append(f"# Backtest analysis: `{root.name}`\n")
    report_parts.append(f"Generated at UTC: `{machine['generated_at_utc']}`\n")
    report_parts.append(f"Root: `{root}`\n")
    report_parts.append(f"Scenario count: `{len(scenario_dirs)}`\n")

    overview = load_strategy_overview(root if root.is_dir() else root.parent)
    if overview is not None:
        report_parts.append("## Run-level overview file\n")
        report_parts.append(df_to_md(overview, max_rows=args.max_rows))
        report_parts.append("")

    for scenario_dir in scenario_dirs:
        text, data = analyze_scenario(scenario_dir, root, args.max_rows)
        report_parts.append(text)
        machine["scenarios"].append(data)

    md_path.write_text("\n".join(report_parts), encoding="utf-8")
    json_path.write_text(json.dumps(machine, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Wrote Markdown report: {md_path}")
    print(f"[OK] Wrote JSON sidecar:    {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())