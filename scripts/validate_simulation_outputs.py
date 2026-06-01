from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _is_missing_required_value(v: object) -> bool:
    if isinstance(v, list):
        return False
    if isinstance(v, dict):
        return False
    return bool(pd.isna(v))


def _read_summary(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _collect(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for summary_path in root.rglob("backtest_summary.json"):
        s = _read_summary(summary_path)
        if not s:
            continue
        scenario_dir = summary_path.parent
        rows.append(
            {
                "scenario_path": str(scenario_dir),
                "scenario": scenario_dir.name,
                "trading_strategy": scenario_dir.parent.name if scenario_dir.parent != root else "",
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
                "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur": s.get("global_hindsight_perfect_foresight_upper_bound_total_pnl_eur"),
                "realized_exceeds_global_perfect_foresight": s.get("realized_exceeds_global_perfect_foresight"),
                "global_perfect_foresight_dominance_check_pass": s.get("global_perfect_foresight_dominance_check_pass"),
                "global_perfect_foresight_validation_status": s.get("global_perfect_foresight_validation_status"),
                "realized_total_pnl_eur": s.get("realized_total_pnl_eur"),
                "rolling_perfect_foresight_same_rules_total_pnl_eur": s.get(
                    "rolling_perfect_foresight_same_rules_total_pnl_eur",
                    s.get("perfect_foresight_total_pnl_eur"),
                ),
                "perfect_foresight_total_pnl_eur": s.get(
                    "perfect_foresight_total_pnl_eur",
                    None,
                ),
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
    ap.add_argument("output_dir", help="Root simulation output directory (contains strategy/scenario subfolders).")
    ap.add_argument("--out-csv", default="", help="Optional CSV output path for per-scenario validation table.")
    ap.add_argument("--out-json", default="", help="Optional JSON output path for aggregate validation stats.")
    ap.add_argument(
        "--allow-stale",
        action="store_true",
        help="Diagnostic mode: do not fail process on stale/missing required fields.",
    )
    args = ap.parse_args()

    root = Path(args.output_dir)
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
        "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur",
        "realized_exceeds_global_perfect_foresight",
        "global_perfect_foresight_dominance_check_pass",
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
        & (required_fields_check_pass)
        & (~non_ok_codes)
    )
    strategy_series = df.get("trading_strategy", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
    id_mode_series = df.get("id_mode", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
    id_recourse_mode_series = df.get("id_recourse_mode", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
    id_abs_mwh = (
        df.get("total_id_revenue_eur", 0.0).fillna(0.0).abs()
        + df.get("total_id_cost_eur", 0.0).fillna(0.0).abs()
    )
    baseline_mask = strategy_series.isin({"da_only", "afrr_only", "bcm_only", "bem_only"})
    da_only_mask = strategy_series.eq("da_only")
    # Baseline contamination guard:
    # - da_only: no ID activity at all.
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


if __name__ == "__main__":
    main()
