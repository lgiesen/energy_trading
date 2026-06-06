from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ACCOUNTING_TOL_EUR = 1e-6
PNL_HIERARCHY_TOL_EUR = 1e-4
PREDICTED_PLANNED_PNL_ALIAS_TOL_EUR = 1e-6


def _is_missing_required_value(v: object) -> bool:
    if isinstance(v, list):
        return False
    if isinstance(v, dict):
        return False
    return bool(pd.isna(v))


def _read_summary(path: Path) -> dict[str, object]:
    try:
        return _normalize_predicted_pnl_aliases(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _normalize_predicted_pnl_aliases(summary: dict[str, object]) -> dict[str, object]:
    out = dict(summary)
    predicted_present = "predicted_total_pnl_eur" in out
    planned_present = "planned_total_pnl_eur" in out
    if not predicted_present and planned_present:
        out["predicted_total_pnl_eur"] = out["planned_total_pnl_eur"]
        out["predicted_total_pnl_eur_source"] = "legacy_planned_total_pnl_eur"
        predicted_present = True
    if predicted_present and not planned_present:
        out["planned_total_pnl_eur"] = out["predicted_total_pnl_eur"]
        out["planned_total_pnl_eur_is_legacy_alias"] = 1.0
        return out
    if predicted_present and planned_present:
        pred = _safe_float(out.get("predicted_total_pnl_eur"))
        planned = _safe_float(out.get("planned_total_pnl_eur"))
        if pd.notna(pred) and pd.notna(planned) and abs(float(pred) - float(planned)) > PREDICTED_PLANNED_PNL_ALIAS_TOL_EUR:
            out["predicted_planned_pnl_alias_consistency_ok"] = 0.0
            out["predicted_planned_pnl_alias_error_eur"] = float(pred) - float(planned)
        else:
            out["predicted_planned_pnl_alias_consistency_ok"] = 1.0
        out["planned_total_pnl_eur"] = out["predicted_total_pnl_eur"]
        out["planned_total_pnl_eur_is_legacy_alias"] = 1.0
    return out


def _safe_float(v: object, default: float = float("nan")) -> float:
    try:
        out = float(pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0])
    except Exception:
        return default
    return out if pd.notna(out) else default


def _read_first_record(path_csv: Path, path_json: Path) -> dict[str, object]:
    if path_csv.exists():
        try:
            df = pd.read_csv(path_csv)
            if not df.empty:
                return dict(df.iloc[0].to_dict())
        except Exception:
            return {}
    if path_json.exists():
        try:
            obj = json.loads(path_json.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if isinstance(obj, list) and obj:
            return dict(obj[0]) if isinstance(obj[0], dict) else {}
        if isinstance(obj, dict):
            for key in ("records", "data", "rows"):
                val = obj.get(key)
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    return dict(val[0])
            return dict(obj)
    return {}


def _read_hourly(scenario_dir: Path) -> pd.DataFrame:
    parquet_path = scenario_dir / "backtest_hourly.parquet"
    csv_path = scenario_dir / "backtest_hourly.csv"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            return pd.DataFrame()
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _max_hourly_numeric(hourly: pd.DataFrame, column: str) -> float:
    if hourly.empty or column not in hourly.columns:
        return float("nan")
    vals = pd.to_numeric(hourly[column], errors="coerce").dropna()
    return float(vals.max()) if not vals.empty else float("nan")


def _is_verified_global_upper_bound(summary: dict[str, object]) -> float:
    for key in (
        "global_pf_verified_upper_bound",
        "global_hindsight_perfect_foresight_is_global_upper_bound",
        "global_perfect_foresight_is_upper_bound",
    ):
        if key in summary:
            return 1.0 if _safe_float(summary.get(key), 0.0) >= 0.5 else 0.0
    status = str(summary.get("global_perfect_foresight_validation_status", "")).strip().lower()
    if "verified" in status and "unverified" not in status and "disabled" not in status:
        return 1.0
    return 0.0


def _scenario_invalid_reason(existing: object, added: list[str]) -> str:
    parts = [p.strip() for p in str(existing or "").split(",") if p.strip()]
    for reason in added:
        if reason and reason not in parts:
            parts.append(reason)
    return ",".join(parts)


def _build_pnl_validation_report(
    df: pd.DataFrame,
    *,
    pnl_tolerance: float = PNL_HIERARCHY_TOL_EUR,
    accounting_tolerance: float = ACCOUNTING_TOL_EUR,
    require_pnl_hierarchy: bool = False,
    require_afrr_decomposition: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, rec in df.iterrows():
        scenario_dir = Path(str(rec.get("scenario_path", "")))
        summary = _read_summary(scenario_dir / "backtest_summary.json")
        perf = _read_first_record(scenario_dir / "performance_metrics.csv", scenario_dir / "performance_metrics.json")
        hourly = _read_hourly(scenario_dir)

        realized = _safe_float(summary.get("realized_total_pnl_eur", rec.get("realized_total_pnl_eur")))
        rolling = _safe_float(
            summary.get(
                "rolling_perfect_foresight_same_rules_total_pnl_eur",
                rec.get("rolling_perfect_foresight_same_rules_total_pnl_eur"),
            )
        )
        global_pf = _safe_float(
            summary.get(
                "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur",
                rec.get("global_hindsight_perfect_foresight_upper_bound_total_pnl_eur"),
            )
        )
        global_available = 1.0 if _safe_float(summary.get("global_perfect_foresight_available", 0.0), 0.0) >= 0.5 else 0.0
        global_verified = _is_verified_global_upper_bound(summary)

        invalid_reasons: list[str] = []
        realized_minus_rolling = realized - rolling if pd.notna(realized) and pd.notna(rolling) else float("nan")
        rolling_minus_global = rolling - global_pf if pd.notna(rolling) and pd.notna(global_pf) else float("nan")
        realized_minus_global = realized - global_pf if pd.notna(realized) and pd.notna(global_pf) else float("nan")

        realized_le_rolling = bool(pd.notna(realized_minus_rolling) and realized_minus_rolling <= pnl_tolerance)
        global_checks_available = bool(global_available >= 0.5 and global_verified >= 0.5 and pd.notna(global_pf))
        rolling_le_global = True
        realized_le_global = True
        if require_pnl_hierarchy:
            if not realized_le_rolling:
                invalid_reasons.append("realized_exceeds_rolling_perfect_foresight")
            if global_available < 0.5:
                invalid_reasons.append("global_pf_unavailable")
            if global_available >= 0.5 and global_verified < 0.5:
                invalid_reasons.append("global_pf_unverified")
            if global_checks_available:
                rolling_le_global = bool(pd.notna(rolling_minus_global) and rolling_minus_global <= pnl_tolerance)
                realized_le_global = bool(pd.notna(realized_minus_global) and realized_minus_global <= pnl_tolerance)
                if not rolling_le_global:
                    invalid_reasons.append("rolling_pf_exceeds_global_perfect_foresight")
                if not realized_le_global:
                    invalid_reasons.append("realized_exceeds_global_perfect_foresight")
        pnl_hierarchy_pass = bool(realized_le_rolling and (not require_pnl_hierarchy or (global_checks_available and rolling_le_global and realized_le_global)))

        afrr = _safe_float(perf.get("afrr_total_net_revenue_eur"))
        bcm = _safe_float(perf.get("bcm_strategy_total_revenue_eur", perf.get("bcm_total_revenue_eur")))
        bem = _safe_float(perf.get("bem_net_revenue_eur", perf.get("bem_total_revenue_eur")))
        afrr_error = afrr - bcm - bem if pd.notna(afrr) and pd.notna(bcm) and pd.notna(bem) else float("nan")
        afrr_decomposition_pass = bool(pd.notna(afrr_error) and abs(afrr_error) <= accounting_tolerance)
        if require_afrr_decomposition and not afrr_decomposition_pass:
            invalid_reasons.append("afrr_decomposition_mismatch")

        max_activation_split_error = _safe_float(summary.get("activation_split_reconciliation_error_max"))
        if not hourly.empty and {"real_revenue_activation_eur", "real_bcm_linked_activation_revenue_eur", "real_bem_only_activation_revenue_eur"}.issubset(hourly.columns):
            split_err = (
                pd.to_numeric(hourly["real_revenue_activation_eur"], errors="coerce").fillna(0.0)
                - pd.to_numeric(hourly["real_bcm_linked_activation_revenue_eur"], errors="coerce").fillna(0.0)
                - pd.to_numeric(hourly["real_bem_only_activation_revenue_eur"], errors="coerce").fillna(0.0)
            ).abs()
            max_activation_split_error = float(split_err.max()) if len(split_err) else 0.0
        activation_split_pass = bool(pd.notna(max_activation_split_error) and max_activation_split_error <= accounting_tolerance)
        if require_afrr_decomposition and not activation_split_pass:
            invalid_reasons.append("activation_split_mismatch")

        rows.append(
            {
                "scenario_path": str(scenario_dir),
                "model": summary.get("model_key", rec.get("model", "")),
                "strategy": summary.get("trading_strategy", rec.get("trading_strategy", "")),
                "quantile_pair": rec.get("scenario", scenario_dir.name),
                "simulation_valid": rec.get("simulation_valid"),
                "thesis_reportable": rec.get("thesis_reportable"),
                "realized_total_pnl_eur": realized,
                "rolling_pf_eur": rolling,
                "global_hindsight_pf_eur": global_pf,
                "realized_minus_rolling_pf_eur": realized_minus_rolling,
                "rolling_pf_minus_global_pf_eur": rolling_minus_global,
                "realized_minus_global_pf_eur": realized_minus_global,
                "global_pf_available": global_available,
                "global_pf_verified_upper_bound": global_verified,
                "pnl_hierarchy_pass": float(pnl_hierarchy_pass),
                "afrr_total_net_revenue_eur": afrr,
                "bcm_strategy_total_revenue_eur": bcm,
                "bem_net_revenue_eur": bem,
                "afrr_decomposition_error_eur": afrr_error,
                "afrr_decomposition_pass": float(afrr_decomposition_pass),
                "activation_split_pass": float(activation_split_pass),
                "max_activation_split_error_eur": max_activation_split_error,
                "invalid_reason_added": ",".join(invalid_reasons),
                "invalid_reason_with_added": _scenario_invalid_reason(rec.get("invalid_reason", ""), invalid_reasons),
            }
        )
    return pd.DataFrame(rows)


def _collect(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for summary_path in root.rglob("backtest_summary.json"):
        s = _read_summary(summary_path)
        if not s:
            continue
        scenario_dir = summary_path.parent
        hourly = _read_hourly(scenario_dir)
        scenario_name = str(s.get("scenario", scenario_dir.name))
        strategy_name = str(s.get("trading_strategy", scenario_dir.parent.name if scenario_dir.parent != root else ""))
        model_name = str(s.get("model_key", s.get("model", "")))
        rows.append(
            {
                "summary_file_path": str(summary_path),
                "scenario_path": str(scenario_dir),
                "model": model_name,
                "strategy": strategy_name,
                "quantile_pair": scenario_name,
                "scenario": scenario_name,
                "trading_strategy": strategy_name,
                "simulation_valid": s.get("simulation_valid"),
                "thesis_reportable": s.get("thesis_reportable"),
                "invalid_reason": s.get("invalid_reason"),
                "fallback_used": s.get("fallback_used"),
                "fallback_mode_counts": s.get("fallback_mode_counts"),
                "optimization_error_code_counts": s.get("optimization_error_code_counts"),
                "id_mode": s.get("id_mode"),
                "id_recourse_mode": s.get("id_recourse_mode"),
                "id_economic_enabled": s.get("id_economic_enabled"),
                "id_technical_repair_enabled": s.get("id_technical_repair_enabled"),
                "total_id_revenue_eur": s.get("total_id_revenue_eur"),
                "total_id_cost_eur": s.get("total_id_cost_eur"),
                "total_id_pnl_eur": s.get("total_id_pnl_eur"),
                "id_repair_mwh_total": s.get("id_repair_mwh_total"),
                "id_repair_cost_eur_total": s.get("id_repair_cost_eur_total"),
                "id_economic_mwh_total": s.get("id_economic_mwh_total"),
                "id_economic_pnl_eur_total": s.get("id_economic_pnl_eur_total"),
                "id_technical_repair_pnl_eur_total": s.get("id_technical_repair_pnl_eur_total"),
                "simulation_schema_version": s.get("simulation_schema_version"),
                "required_summary_fields_version": s.get("required_summary_fields_version"),
                "code_run_started_at_utc": s.get("code_run_started_at_utc"),
                "command_line_args": s.get("command_line_args"),
                "output_was_cleaned": s.get("output_was_cleaned"),
                "infeasible_debug_dump_count": s.get("infeasible_debug_dump_count"),
                "accepted_path_infeasible_debug_dump_count": s.get("accepted_path_infeasible_debug_dump_count"),
                "candidate_infeasible_debug_dump_count": s.get("candidate_infeasible_debug_dump_count"),
                "infeasible_debug_dump_paths": s.get("infeasible_debug_dump_paths"),
                "infeasible_debug_dump_timestamps": s.get("infeasible_debug_dump_timestamps"),
                "summary_fields_defaulted": s.get("summary_fields_defaulted"),
                "required_fields_defaulted": s.get("required_fields_defaulted"),
                "required_fields_computed": s.get("required_fields_computed"),
                "required_fields_missing": s.get("required_fields_missing"),
                "critical_required_fields_defaulted": s.get("critical_required_fields_defaulted"),
                "optional_fields_defaulted": s.get("optional_fields_defaulted"),
                "required_fields_check_pass": s.get("required_fields_check_pass"),
                "headroom_violation_count": s.get("headroom_violation_count"),
                "missed_capacity_pos_mw": s.get("missed_capacity_pos_mw"),
                "missed_capacity_neg_mw": s.get("missed_capacity_neg_mw"),
                "pnl_reconciliation_error_max_eur": s.get("pnl_reconciliation_error_max_eur"),
                "activation_split_reconciliation_error_max": s.get("activation_split_reconciliation_error_max"),
                "precommit_clamp_applied_count": s.get("precommit_clamp_applied_count"),
                "final_soc_check_pass": s.get("final_soc_check_pass"),
                "final_soc_actual_mwh": s.get("final_soc_mwh", s.get("final_real_soc_mwh")),
                "final_soc_target_mwh": s.get("target_final_soc_mwh", s.get("final_soc_min_target_mwh")),
                "terminal_soc_repair_cost_eur": s.get("terminal_soc_repair_cost_eur"),
                "terminal_soc_net_adjustment_eur": s.get("terminal_soc_net_adjustment_eur", s.get("terminal_soc_adjustment_eur")),
                "benchmark_same_rules_gate_consistent": s.get("benchmark_same_rules_gate_consistent"),
                "global_perfect_foresight_available": s.get("global_perfect_foresight_available"),
                "global_pf_available": s.get("global_pf_available", s.get("global_perfect_foresight_available")),
                "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur": s.get("global_hindsight_perfect_foresight_upper_bound_total_pnl_eur"),
                "global_pf_verified_upper_bound": s.get("global_pf_verified_upper_bound"),
                "global_pf_same_market_rules": s.get("global_pf_same_market_rules"),
                "global_pf_solver_available": s.get("global_pf_solver_available"),
                "global_pf_solver_feasible": s.get("global_pf_solver_feasible"),
                "global_pf_solver_rejection_reason": s.get("global_pf_solver_rejection_reason"),
                "global_pf_best_feasible_lower_bound_name": s.get("global_pf_best_feasible_lower_bound_name"),
                "global_pf_best_feasible_lower_bound_eur": s.get("global_pf_best_feasible_lower_bound_eur"),
                "global_pf_realized_path_incumbent_eur": s.get("global_pf_realized_path_incumbent_eur"),
                "global_pf_solution_eur": s.get("global_pf_solution_eur"),
                "global_pf_minus_realized_incumbent_eur": s.get("global_pf_minus_realized_incumbent_eur"),
                "global_pf_component_gap_json": s.get("global_pf_component_gap_json"),
                "global_pf_failure_reason": s.get("global_pf_failure_reason"),
                "global_pf_solver_status": s.get("global_pf_solver_status"),
                "realized_exceeds_global_perfect_foresight": s.get("realized_exceeds_global_perfect_foresight"),
                "global_perfect_foresight_dominance_check_pass": s.get("global_perfect_foresight_dominance_check_pass"),
                "global_perfect_foresight_validation_status": s.get("global_perfect_foresight_validation_status"),
                "realized_total_pnl_eur": s.get("realized_total_pnl_eur"),
                "predicted_total_pnl_eur": s.get("predicted_total_pnl_eur"),
                "planned_total_pnl_eur": s.get("planned_total_pnl_eur"),
                "planned_total_pnl_eur_is_legacy_alias": s.get("planned_total_pnl_eur_is_legacy_alias"),
                "predicted_total_pnl_eur_source": s.get("predicted_total_pnl_eur_source"),
                "predicted_planned_pnl_alias_consistency_ok": s.get("predicted_planned_pnl_alias_consistency_ok"),
                "predicted_planned_pnl_alias_error_eur": s.get("predicted_planned_pnl_alias_error_eur"),
                "rolling_perfect_foresight_same_rules_total_pnl_eur": s.get(
                    "rolling_perfect_foresight_same_rules_total_pnl_eur",
                    s.get("perfect_foresight_total_pnl_eur"),
                ),
                "perfect_foresight_total_pnl_eur": s.get(
                    "perfect_foresight_total_pnl_eur",
                    None,
                ),
                "max_real_power_stack_charge_mw": _max_hourly_numeric(hourly, "real_power_stack_charge_mw"),
                "max_real_power_stack_discharge_mw": _max_hourly_numeric(hourly, "real_power_stack_discharge_mw"),
                "max_real_power_violation_charge_mw": _max_hourly_numeric(hourly, "real_power_violation_charge_mw"),
                "max_real_power_violation_discharge_mw": _max_hourly_numeric(hourly, "real_power_violation_discharge_mw"),
            }
        )
    return pd.DataFrame(rows)


REQUIRED_SUMMARY_FIELDS = [
    "simulation_schema_version",
    "required_summary_fields_version",
    "code_run_started_at_utc",
    "command_line_args",
    "output_was_cleaned",
    "simulation_valid",
    "thesis_reportable",
    "invalid_reason",
    "fallback_used",
    "fallback_mode_counts",
    "optimization_error_code_counts",
    "infeasible_debug_dump_count",
    "accepted_path_infeasible_debug_dump_count",
    "candidate_infeasible_debug_dump_count",
    "infeasible_debug_dump_paths",
    "infeasible_debug_dump_timestamps",
    "summary_fields_defaulted",
    "required_fields_defaulted",
    "required_fields_computed",
    "precommit_clamp_applied_count",
    "activation_split_reconciliation_error_max",
    "final_soc_check_pass",
    "benchmark_same_rules_gate_consistent",
    "global_perfect_foresight_dominance_check_pass",
    "global_perfect_foresight_validation_status",
]


def _invalid_by_reason(df: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    if "invalid_reason" not in df.columns:
        return out
    for txt in df["invalid_reason"].fillna("").astype(str):
        for reason in [r.strip() for r in txt.split(",") if r.strip()]:
            out[reason] = out.get(reason, 0) + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate simulation output directories from backtest_summary.json files.")
    ap.add_argument("output_dir", nargs="?", default="", help="Root simulation output directory (contains strategy/scenario subfolders).")
    ap.add_argument("--root", default="", help="Root simulation output directory. Alias for positional output_dir.")
    ap.add_argument("--out-csv", default="", help="Optional CSV output path for per-scenario validation table.")
    ap.add_argument("--out-json", default="", help="Optional JSON output path for aggregate validation stats.")
    ap.add_argument("--pnl-report-csv", default="", help="Optional CSV path for strict PnL/aFRR validation report.")
    ap.add_argument("--pnl-report-json", default="", help="Optional JSON path for strict PnL/aFRR validation report.")
    ap.add_argument("--require-reportable", action="store_true", help="Fail unless all scenarios are simulation_valid and thesis_reportable.")
    ap.add_argument("--require-pnl-hierarchy", action="store_true", help="Fail on realized/PF/global-PF PnL hierarchy violations.")
    ap.add_argument("--require-afrr-decomposition", action="store_true", help="Fail on aFRR BCM/BEM revenue decomposition mismatches.")
    ap.add_argument("--pnl-tolerance-eur", type=float, default=PNL_HIERARCHY_TOL_EUR, help="Tolerance for cumulative PnL hierarchy checks.")
    ap.add_argument("--accounting-tolerance-eur", type=float, default=ACCOUNTING_TOL_EUR, help="Tolerance for exact accounting/aFRR split checks.")
    ap.add_argument(
        "--allow-stale",
        action="store_true",
        help="Diagnostic mode: do not fail process on stale/missing required fields.",
    )
    args = ap.parse_args()

    output_dir = args.root or args.output_dir
    if not output_dir:
        ap.error("Provide output_dir or --root.")
    root = Path(output_dir)
    if not root.exists():
        raise FileNotFoundError(f"Output dir not found: {root}")

    df = _collect(root)
    if df.empty:
        print(f"[WARN] No backtest_summary.json files found under: {root}")
        return

    for c in ("simulation_valid", "thesis_reportable"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in (
        "fallback_used",
        "headroom_violation_count",
        "missed_capacity_pos_mw",
        "missed_capacity_neg_mw",
        "pnl_reconciliation_error_max_eur",
        "activation_split_reconciliation_error_max",
        "final_soc_check_pass",
        "final_soc_actual_mwh",
        "final_soc_target_mwh",
        "terminal_soc_repair_cost_eur",
        "benchmark_same_rules_gate_consistent",
        "infeasible_debug_dump_count",
        "accepted_path_infeasible_debug_dump_count",
        "candidate_infeasible_debug_dump_count",
        "output_was_cleaned",
        "global_perfect_foresight_available",
        "global_pf_available",
        "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur",
        "global_pf_verified_upper_bound",
        "global_pf_same_market_rules",
        "global_pf_solver_available",
        "global_pf_solver_feasible",
        "global_pf_solver_rejection_reason",
        "global_pf_best_feasible_lower_bound_name",
        "global_pf_best_feasible_lower_bound_eur",
        "global_pf_realized_path_incumbent_eur",
        "global_pf_solution_eur",
        "global_pf_minus_realized_incumbent_eur",
        "realized_exceeds_global_perfect_foresight",
        "global_perfect_foresight_dominance_check_pass",
        "max_real_power_stack_charge_mw",
        "max_real_power_stack_discharge_mw",
        "max_real_power_violation_charge_mw",
        "max_real_power_violation_discharge_mw",
        "id_economic_enabled",
        "id_technical_repair_enabled",
        "total_id_revenue_eur",
        "total_id_cost_eur",
        "total_id_pnl_eur",
        "id_repair_mwh_total",
        "id_repair_cost_eur_total",
        "id_economic_mwh_total",
        "id_economic_pnl_eur_total",
        "id_technical_repair_pnl_eur_total",
    ):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def _has_non_ok_opt_codes(v: object) -> bool:
        try:
            d = json.loads(v) if isinstance(v, str) else (v if isinstance(v, dict) else {})
        except Exception:
            return True
        if not isinstance(d, dict):
            return True
        for k, cnt in d.items():
            key = str(k).strip().lower()
            if key not in {"ok", "none", ""} and float(cnt) > 0.0:
                return True
        return False

    non_ok_codes = (
        df.get("optimization_error_code_counts", pd.Series("{}", index=df.index))
        .apply(_has_non_ok_opt_codes)
        .astype(bool)
    )

    def _defaulted_required_count(v: object) -> int:
        try:
            arr = json.loads(v) if isinstance(v, str) else (v if isinstance(v, list) else [])
        except Exception:
            return 999999
        if not isinstance(arr, list):
            return 999999
        return int(len(arr))

    legacy_required_defaulted_count = (
        df.get("required_fields_defaulted", pd.Series("[]", index=df.index))
        .apply(_defaulted_required_count)
        .astype(int)
    )
    required_missing_count = (
        df.get("required_fields_missing", pd.Series("[]", index=df.index))
        .apply(_defaulted_required_count)
        .astype(int)
    )
    critical_defaulted_count = (
        df.get("critical_required_fields_defaulted", pd.Series("[]", index=df.index))
        .apply(_defaulted_required_count)
        .astype(int)
    )
    optional_defaulted_count = (
        df.get("optional_fields_defaulted", pd.Series("[]", index=df.index))
        .apply(_defaulted_required_count)
        .astype(int)
    )
    required_fields_check_pass = (
        (required_missing_count <= 0)
        & (critical_defaulted_count <= 0)
    )
    # Backward compatibility for legacy summaries that only expose required_fields_defaulted.
    legacy_mask = (
        ~df.get("required_fields_missing", pd.Series(index=df.index)).notna()
        & ~df.get("critical_required_fields_defaulted", pd.Series(index=df.index)).notna()
    )
    required_missing_count = required_missing_count.mask(legacy_mask, legacy_required_defaulted_count)
    critical_defaulted_count = critical_defaulted_count.mask(legacy_mask, 0)
    optional_defaulted_count = optional_defaulted_count.mask(legacy_mask, 0)
    required_fields_check_pass = required_fields_check_pass.mask(legacy_mask, legacy_required_defaulted_count <= 0)
    df["required_fields_missing"] = required_missing_count.astype(int)
    df["critical_required_fields_defaulted"] = critical_defaulted_count.astype(int)
    df["optional_fields_defaulted"] = optional_defaulted_count.astype(int)
    df["required_fields_check_pass"] = required_fields_check_pass.astype(float)

    # Parser-side thesis validity check (strict).
    thesis_rule = (
        (df.get("simulation_valid", 0.0).fillna(0.0) >= 0.5)
        & (df.get("thesis_reportable", 0.0).fillna(0.0) >= 0.5)
        & (df.get("fallback_used", 0.0).fillna(0.0) <= 0.5)
        & (df.get("headroom_violation_count", 0.0).fillna(0.0) <= 1e-9)
        & (df.get("missed_capacity_pos_mw", 0.0).fillna(0.0) <= 1e-9)
        & (df.get("missed_capacity_neg_mw", 0.0).fillna(0.0) <= 1e-9)
        & (df.get("pnl_reconciliation_error_max_eur", 1e9).fillna(1e9) <= 1e-2)
        & (df.get("activation_split_reconciliation_error_max", 1e9).fillna(1e9) <= 1e-2)
        & (df.get("final_soc_check_pass", 0.0).fillna(0.0) >= 0.5)
        & (df.get("benchmark_same_rules_gate_consistent", 0.0).fillna(0.0) >= 0.5)
        & (df.get("global_perfect_foresight_dominance_check_pass", 0.0).fillna(0.0) >= 0.5)
        & (df.get("accepted_path_infeasible_debug_dump_count", 0.0).fillna(0.0) <= 0.5)
        & (df.get("max_real_power_violation_charge_mw", 0.0).fillna(0.0) <= 1e-9)
        & (df.get("max_real_power_violation_discharge_mw", 0.0).fillna(0.0) <= 1e-9)
        & (required_fields_check_pass)
        & (~non_ok_codes)
    )
    strategy_series = (
        df.get("trading_strategy", pd.Series("", index=df.index))
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"da_only": "da", "afrr_only": "afrr", "bcm_only": "bcm", "bem_only": "bem"})
    )
    id_mode_series = df.get("id_mode", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
    id_recourse_mode_series = df.get("id_recourse_mode", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
    id_abs_mwh = (
        df.get("total_id_revenue_eur", 0.0).fillna(0.0).abs()
        + df.get("total_id_cost_eur", 0.0).fillna(0.0).abs()
    )
    baseline_mask = strategy_series.isin({"da", "afrr", "bcm", "bem"})
    da_only_mask = strategy_series.eq("da")
    # Baseline contamination guard:
    # - da: no ID activity at all.
    # - other baselines: no economic ID mode; technical-repair-only allowed.
    invalid_id_policy = (
        (da_only_mask & (id_abs_mwh > 1e-9))
        | (baseline_mask & id_mode_series.eq("economic"))
        | (id_recourse_mode_series.eq("disabled") & (id_abs_mwh > 1e-9))
        | (id_recourse_mode_series.eq("afrr_obligation_only") & da_only_mask & (id_abs_mwh > 1e-9))
    )
    thesis_rule = thesis_rule & (~invalid_id_policy)
    # If shortfall exists, explicit repair must be present.
    if "final_soc_actual_mwh" in df.columns and "final_soc_target_mwh" in df.columns:
        shortfall = (df["final_soc_target_mwh"].fillna(0.0) - df["final_soc_actual_mwh"].fillna(0.0)).clip(lower=0.0)
        repair_ok = (shortfall <= 1e-9) | (df.get("terminal_soc_repair_cost_eur", 0.0).fillna(0.0) > 0.0)
        thesis_rule &= repair_ok
    debug_dump_check_pass = (
        (df.get("accepted_path_infeasible_debug_dump_count", 0.0).fillna(0.0) <= 0.5)
        & (df.get("infeasible_debug_dump_count", 0.0).fillna(0.0) >= 0.0)
    )
    df["debug_dump_check_pass"] = debug_dump_check_pass.astype(float)
    df["parser_thesis_valid"] = thesis_rule.astype(float)

    pnl_report_df = _build_pnl_validation_report(
        df,
        pnl_tolerance=float(args.pnl_tolerance_eur),
        accounting_tolerance=float(args.accounting_tolerance_eur),
        require_pnl_hierarchy=bool(args.require_pnl_hierarchy),
        require_afrr_decomposition=bool(args.require_afrr_decomposition),
    )
    if not pnl_report_df.empty:
        if bool(args.require_pnl_hierarchy):
            hierarchy_ok = pnl_report_df["pnl_hierarchy_pass"].astype(float) >= 0.5
            thesis_rule &= hierarchy_ok.to_numpy(dtype=bool)
        if bool(args.require_afrr_decomposition):
            afrr_ok = (
                (pnl_report_df["afrr_decomposition_pass"].astype(float) >= 0.5)
                & (pnl_report_df["activation_split_pass"].astype(float) >= 0.5)
            )
            thesis_rule &= afrr_ok.to_numpy(dtype=bool)
        df["parser_thesis_valid"] = thesis_rule.astype(float)

    total = int(len(df))
    valid = int((df["simulation_valid"] >= 0.5).sum()) if "simulation_valid" in df.columns else 0
    thesis = int((df["thesis_reportable"] >= 0.5).sum()) if "thesis_reportable" in df.columns else 0
    invalid = int(total - valid)
    invalid_rate = float(100.0 * invalid / total) if total else float("nan")
    invalid_rows = df.loc[(df["simulation_valid"] < 0.5) | (df["parser_thesis_valid"] < 0.5)].copy()
    invalid_quantile_rows = df.loc[(df["simulation_valid"] < 0.5) | (df["thesis_reportable"] < 0.5)].copy()
    by_reason = _invalid_by_reason(invalid_rows)
    invalid_by_quantile = (
        invalid_quantile_rows.groupby("scenario", dropna=False).size().sort_values(ascending=False).to_dict()
        if not invalid_quantile_rows.empty and "scenario" in invalid_quantile_rows.columns
        else {}
    )
    stale_rows = []
    missing_field_errors = []
    for _, r in df.iterrows():
        missing = [f for f in REQUIRED_SUMMARY_FIELDS if (f not in r.index) or _is_missing_required_value(r.get(f))]
        if missing:
            stale_rows.append({"scenario_path": r.get("scenario_path", ""), "missing_fields": missing})
            missing_field_errors.append(f"{r.get('scenario_path','')}: missing {missing}")
    required_fields_ok = (len(stale_rows) == 0) and bool((df["required_fields_check_pass"] >= 0.5).all())
    if not required_fields_ok:
        by_reason["missing_required_fields"] = len(stale_rows)
        if args.allow_stale:
            thesis = 0
            if "thesis_reportable" in df.columns:
                df["thesis_reportable"] = 0.0
    if (required_missing_count > 0).any():
        by_reason["required_fields_missing"] = int((required_missing_count > 0).sum())
    if (critical_defaulted_count > 0).any():
        by_reason["critical_required_fields_defaulted"] = int((critical_defaulted_count > 0).sum())
    if (optional_defaulted_count > 0).any():
        by_reason["optional_fields_defaulted"] = int((optional_defaulted_count > 0).sum())
    if not pnl_report_df.empty:
        for txt in pnl_report_df.get("invalid_reason_added", pd.Series("", index=pnl_report_df.index)).fillna("").astype(str):
            for reason in [r.strip() for r in txt.split(",") if r.strip()]:
                by_reason[reason] = by_reason.get(reason, 0) + 1

    consistency_errors: list[str] = []
    if total != (valid + invalid):
        consistency_errors.append(
            f"total_scenarios mismatch: total={total}, valid+invalid={valid + invalid}"
        )
    invalid_by_quantile_total = int(sum(int(v) for v in invalid_by_quantile.values()))
    if invalid_by_quantile_total != invalid:
        consistency_errors.append(
            "invalid_by_quantile mismatch: "
            f"sum={invalid_by_quantile_total}, invalid_scenarios={invalid}"
        )

    print(df.to_string(index=False))
    print()
    print(
        json.dumps(
            {
                "total_scenarios": total,
                "valid_scenarios": valid,
                "invalid_scenarios": invalid,
                "thesis_reportable_scenarios": thesis,
                "invalid_rate_pct": invalid_rate,
                "invalid_by_reason": by_reason,
                "invalid_by_quantile": invalid_by_quantile,
                "stale_scenarios": stale_rows,
                "stale_scenario_count": len(stale_rows),
                "required_fields_ok": required_fields_ok,
                "required_fields_check_pass_rate": float(df["required_fields_check_pass"].mean()) if len(df) else float("nan"),
                "consistency_errors": consistency_errors,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if stale_rows:
        print("[WARN] stale/old scenarios detected (missing required summary fields).")
        for e in missing_field_errors:
            print("[WARN]", e)

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(
                {
                    "total_scenarios": total,
                    "valid_scenarios": valid,
                    "invalid_scenarios": invalid,
                    "thesis_reportable_scenarios": thesis,
                    "invalid_rate_pct": invalid_rate,
                    "invalid_by_reason": by_reason,
                    "invalid_by_quantile": invalid_by_quantile,
                    "stale_scenarios": stale_rows,
                    "stale_scenario_count": len(stale_rows),
                    "required_fields_ok": required_fields_ok,
                    "required_fields_check_pass_rate": float(df["required_fields_check_pass"].mean()) if len(df) else float("nan"),
                    "consistency_errors": consistency_errors,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    pnl_csv = Path(args.pnl_report_csv) if args.pnl_report_csv else root / "pnl_validation_report.csv"
    pnl_json = Path(args.pnl_report_json) if args.pnl_report_json else root / "pnl_validation_report.json"
    if not pnl_report_df.empty:
        pnl_csv.parent.mkdir(parents=True, exist_ok=True)
        pnl_report_df.to_csv(pnl_csv, index=False)
        pnl_json.parent.mkdir(parents=True, exist_ok=True)
        pnl_json.write_text(
            json.dumps(pnl_report_df.to_dict(orient="records"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if consistency_errors:
        raise SystemExit(
            "Validation consistency failure: " + "; ".join(consistency_errors)
        )
    if stale_rows and not args.allow_stale:
        missing_flat = sorted({f for row in stale_rows for f in row.get("missing_fields", [])})
        raise SystemExit(
            "Missing required fields: "
            f"{missing_flat}. This indicates stale artifacts or outdated runner. Re-run with --clean-output."
        )
    strict_failures: list[str] = []
    if args.require_reportable:
        bad = df.loc[
            (df.get("simulation_valid", 0.0).fillna(0.0) < 0.5)
            | (df.get("thesis_reportable", 0.0).fillna(0.0) < 0.5)
            | (df.get("parser_thesis_valid", 0.0).fillna(0.0) < 0.5)
        ]
        if not bad.empty:
            strict_failures.append(f"non_reportable_scenarios={len(bad)}")
    if args.require_pnl_hierarchy and not pnl_report_df.empty:
        bad = pnl_report_df.loc[pnl_report_df["pnl_hierarchy_pass"].astype(float) < 0.5]
        if not bad.empty:
            reasons = sorted(
                {
                    r.strip()
                    for txt in bad["invalid_reason_added"].fillna("").astype(str)
                    for r in txt.split(",")
                    if r.strip()
                }
            )
            strict_failures.append(f"pnl_hierarchy_failed={len(bad)} reasons={reasons}")
    if args.require_afrr_decomposition and not pnl_report_df.empty:
        bad = pnl_report_df.loc[
            (pnl_report_df["afrr_decomposition_pass"].astype(float) < 0.5)
            | (pnl_report_df["activation_split_pass"].astype(float) < 0.5)
        ]
        if not bad.empty:
            reasons = sorted(
                {
                    r.strip()
                    for txt in bad["invalid_reason_added"].fillna("").astype(str)
                    for r in txt.split(",")
                    if r.strip()
                }
            )
            strict_failures.append(f"afrr_decomposition_failed={len(bad)} reasons={reasons}")
    if strict_failures:
        raise SystemExit("Strict simulation validation failed: " + "; ".join(strict_failures))


if __name__ == "__main__":
    main()
