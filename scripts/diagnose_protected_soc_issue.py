#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _safe_obj(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype=object)


def _load_scenarios(run_dir: Path) -> list[tuple[str, Path]]:
    scenarios: list[tuple[str, Path]] = []
    if not run_dir.exists():
        return scenarios
    for strategy_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        for scen_dir in sorted([p for p in strategy_dir.iterdir() if p.is_dir()]):
            if (scen_dir / "backtest_summary.json").exists():
                scenarios.append((scen_dir.name, scen_dir))
    return scenarios


def _extract_summary_fields(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "simulation_valid": float(summary.get("simulation_valid", np.nan)),
        "thesis_reportable": float(summary.get("thesis_reportable", np.nan)),
        "invalid_reason": str(summary.get("invalid_reason", "")),
        "protected_soc_violation_count": float(summary.get("protected_soc_violation_count", np.nan)),
        "protected_soc_violation_max_mwh": float(summary.get("protected_soc_violation_max_mwh", np.nan)),
        "fallback_used": float(summary.get("fallback_used", np.nan)),
        "optimization_error_code_counts": str(summary.get("optimization_error_code_counts", "")),
        "accepted_path_infeasible_debug_dump_count": float(summary.get("accepted_path_infeasible_debug_dump_count", np.nan)),
        "reserve_feasibility_repair_used": float(summary.get("reserve_feasibility_repair_used", np.nan)),
        "final_soc_actual_mwh": float(summary.get("final_soc_actual_mwh", np.nan)),
        "final_soc_target_mwh": float(summary.get("final_soc_target_mwh", np.nan)),
        "final_soc_physical_check_pass": float(summary.get("final_soc_physical_check_pass", np.nan)),
        "final_soc_economic_repair_check_pass": float(summary.get("final_soc_economic_repair_check_pass", np.nan)),
        "terminal_soc_repair_included_in_pnl": float(summary.get("terminal_soc_repair_included_in_pnl", np.nan)),
    }


def _find_hourly_file(scen_dir: Path) -> Path | None:
    for name in ["backtest_hourly.parquet", "realized_ledger.parquet", "executed_ledger.parquet"]:
        p = scen_dir / name
        if p.exists():
            return p
    return None


def _normalize_hourly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts_col = "timestamp_utc" if "timestamp_utc" in out.columns else None
    if ts_col is None:
        cand = [c for c in out.columns if "timestamp" in c.lower()]
        if cand:
            ts_col = cand[0]
    if ts_col is None:
        out["timestamp_utc"] = pd.NaT
    else:
        out["timestamp_utc"] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")

    out["protected_soc_violation_pos_mwh"] = _safe_num(out, "real_protected_soc_violation_pos_mwh", default=np.nan)
    if out["protected_soc_violation_pos_mwh"].isna().all():
        out["protected_soc_violation_pos_mwh"] = _safe_num(out, "protected_soc_violation_pos_mwh", default=0.0)

    out["protected_soc_violation_neg_mwh"] = _safe_num(out, "real_protected_soc_violation_neg_mwh", default=np.nan)
    if out["protected_soc_violation_neg_mwh"].isna().all():
        out["protected_soc_violation_neg_mwh"] = _safe_num(out, "protected_soc_violation_neg_mwh", default=0.0)

    out["soc_start_mwh"] = _safe_num(out, "real_soc_start_mwh", default=np.nan)
    if out["soc_start_mwh"].isna().all():
        out["soc_start_mwh"] = _safe_num(out, "soc_start_lp_mwh", default=np.nan)
    out["soc_end_mwh"] = _safe_num(out, "real_soc_mwh", default=np.nan)
    if out["soc_end_mwh"].isna().all():
        out["soc_end_mwh"] = _safe_num(out, "soc_mwh", default=np.nan)

    out["protected_soc_min_mwh"] = _safe_num(out, "real_protected_soc_min_mwh", default=np.nan)
    if out["protected_soc_min_mwh"].isna().all():
        out["protected_soc_min_mwh"] = _safe_num(out, "protected_soc_min_mwh", default=np.nan)
    out["protected_soc_max_mwh"] = _safe_num(out, "real_protected_soc_max_mwh", default=np.nan)
    if out["protected_soc_max_mwh"].isna().all():
        out["protected_soc_max_mwh"] = _safe_num(out, "protected_soc_max_mwh", default=np.nan)

    return out


def _classify_driver(row: pd.Series) -> tuple[str, str]:
    pos_safe = row.get("bem_only_safe_pos_mw_from_reported_envelope", np.nan)
    neg_safe = row.get("bem_only_safe_neg_mw_from_reported_envelope", np.nan)
    pos_sub = row.get("bem_only_submitted_pos_mw", 0.0)
    neg_sub = row.get("bem_only_submitted_neg_mw", 0.0)
    pos_exc = bool(row.get("bem_only_pos_exceeds_safe", False))
    neg_exc = bool(row.get("bem_only_neg_exceeds_safe", False))
    stale = bool(row.get("stale_or_missing_column_risk", False))
    pmin_diff = abs(float(row.get("protected_min_diff_mwh", 0.0)))
    pmax_diff = abs(float(row.get("protected_max_diff_mwh", 0.0)))
    da_exp = bool(row.get("da_or_activation_explains_violation", False))
    aux_exp = bool(row.get("aux_explains_violation", False))
    lock_exp = bool(row.get("locked_reserve_explains_violation", False))

    if stale:
        return "stale_or_missing_column", "required hourly fields missing/non-finite"
    if pmin_diff > 1e-6 or pmax_diff > 1e-6:
        return "protected_soc_audit_formula_mismatch", f"recompute mismatch min={pmin_diff:.6f} max={pmax_diff:.6f}"
    if pos_exc and neg_exc and pos_sub > 0 and neg_sub > 0:
        return "bem_only_both_sides_without_joint_headroom", f"submitted pos={pos_sub:.3f}>safe={pos_safe:.3f}, neg={neg_sub:.3f}>safe={neg_safe:.3f}"
    if pos_exc:
        return "bem_only_pos_submitted_without_headroom", f"submitted pos={pos_sub:.3f}>safe={pos_safe:.3f}"
    if neg_exc:
        return "bem_only_neg_submitted_without_headroom", f"submitted neg={neg_sub:.3f}>safe={neg_safe:.3f}"
    if da_exp:
        return "da_dispatch_consumed_protected_soc", "DA/BEM/activation flows explain margin breach"
    if aux_exp:
        return "aux_loss_margin_gap", "auxiliary energy contributes to margin breach"
    if lock_exp:
        return "bcm_linked_activation_consumed_protected_soc", "locked reserve obligations alone consume envelope"
    return "unknown", "no single dominant source proven from available fields"


def diagnose_run(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = _load_scenarios(run_dir)

    overview_rows: list[dict[str, Any]] = []
    viol_rows: list[dict[str, Any]] = []
    recompute_rows: list[dict[str, Any]] = []
    attrib_rows: list[dict[str, Any]] = []

    for scenario, scen_dir in scenarios:
        summary_path = scen_dir / "backtest_summary.json"
        summary = _read_json(summary_path)
        ov = {"scenario": scenario, **_extract_summary_fields(summary)}
        overview_rows.append(ov)

        hourly_path = _find_hourly_file(scen_dir)
        if hourly_path is None:
            continue
        h = pd.read_parquet(hourly_path)
        h = _normalize_hourly(h)

        vmask = (_safe_num(h, "protected_soc_violation_pos_mwh", 0.0) > 1e-12) | (_safe_num(h, "protected_soc_violation_neg_mwh", 0.0) > 1e-12)
        hv = h.loc[vmask].copy()
        if hv.empty:
            continue

        args = summary.get("command_line_args", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        eta_in = float(summary.get("eta_in", np.nan))
        eta_out = float(summary.get("eta_out", np.nan))
        if not np.isfinite(eta_in):
            eta_in = 0.94
        if not np.isfinite(eta_out):
            eta_out = 0.94
        reserve_headroom_safety_mwh = float(summary.get("reserve_headroom_safety_mwh", np.nan))
        if not np.isfinite(reserve_headroom_safety_mwh):
            reserve_headroom_safety_mwh = float(args.get("reserve_headroom_safety_mwh", 0.0)) if isinstance(args, dict) else 0.0
        reserve_activation_headroom_h = float(summary.get("reserve_activation_headroom_h", np.nan))
        if not np.isfinite(reserve_activation_headroom_h):
            reserve_activation_headroom_h = float(args.get("reserve_activation_headroom_h", 0.5)) if isinstance(args, dict) else 0.5
        bem_activation_headroom_h = float(summary.get("bem_activation_headroom_h", np.nan))
        if not np.isfinite(bem_activation_headroom_h):
            bem_activation_headroom_h = float(args.get("bem_activation_headroom_h", 0.5)) if isinstance(args, dict) else 0.5

        for _, r in hv.iterrows():
            pos_v = float(r.get("protected_soc_violation_pos_mwh", 0.0))
            neg_v = float(r.get("protected_soc_violation_neg_mwh", 0.0))
            side = "both" if (pos_v > 1e-12 and neg_v > 1e-12) else ("pos" if pos_v > 1e-12 else ("neg" if neg_v > 1e-12 else "unknown"))

            da_buy_mwh = float(r.get("real_da_buy_mwh", r.get("da_buy_mwh", 0.0)))
            da_sell_mwh = float(r.get("real_da_sell_mwh", r.get("da_sell_mwh", 0.0)))
            da_buy_acc = float(r.get("real_da_buy_accepted", r.get("da_buy_accepted", 0.0)))
            da_sell_acc = float(r.get("real_da_sell_accepted", r.get("da_sell_accepted", 0.0)))
            bem_sub_pos = float(r.get("real_bem_only_submitted_pos_mw", r.get("bem_only_submitted_pos_mw", 0.0)))
            bem_sub_neg = float(r.get("real_bem_only_submitted_neg_mw", r.get("bem_only_submitted_neg_mw", 0.0)))
            bem_exe_pos = float(r.get("real_bem_only_executed_pos_mw", r.get("bem_only_executed_pos_mw", 0.0)))
            bem_exe_neg = float(r.get("real_bem_only_executed_neg_mw", r.get("bem_only_executed_neg_mw", 0.0)))
            bcm_act_pos = float(r.get("real_executed_reserve_pos_mw", r.get("executed_reserve_pos_mw", 0.0)))
            bcm_act_neg = float(r.get("real_executed_reserve_neg_mw", r.get("executed_reserve_neg_mw", 0.0)))
            lock_pos = float(r.get("fixed_reserve_obligation_pos_mw", 0.0))
            lock_neg = float(r.get("fixed_reserve_obligation_neg_mw", 0.0))
            aux_e = float(r.get("real_aux_energy_mwh", r.get("aux_energy_mwh", 0.0)))
            opt_code = str(r.get("optimization_error_code", "ok"))
            fallback_mode = str(r.get("optimization_fallback", "none"))

            soc_min_mwh = float(summary.get("soc_min_mwh", np.nan))
            soc_max_mwh = float(summary.get("soc_max_mwh", np.nan))
            if not np.isfinite(soc_min_mwh):
                cap = float(summary.get("battery_capacity_mwh", 20.0))
                soc_min_mwh = float(summary.get("soc_min", 0.1)) * cap if float(summary.get("soc_min", np.nan)) <= 1.0 else float(summary.get("soc_min", 2.0))
            if not np.isfinite(soc_max_mwh):
                cap = float(summary.get("battery_capacity_mwh", 20.0))
                soc_max_mwh = float(summary.get("soc_max", 0.9)) * cap if float(summary.get("soc_max", np.nan)) <= 1.0 else float(summary.get("soc_max", 18.0))

            soc_start = float(r.get("soc_start_mwh", np.nan))
            reported_min = float(r.get("protected_soc_min_mwh", np.nan))
            reported_max = float(r.get("protected_soc_max_mwh", np.nan))

            recomputed_min = soc_min_mwh + lock_pos * reserve_activation_headroom_h / max(eta_out, 1e-12) + reserve_headroom_safety_mwh
            recomputed_max = soc_max_mwh - lock_neg * reserve_activation_headroom_h * eta_in - reserve_headroom_safety_mwh
            min_diff = float(reported_min - recomputed_min) if np.isfinite(reported_min) else np.nan
            max_diff = float(reported_max - recomputed_max) if np.isfinite(reported_max) else np.nan
            missing_inputs = []
            for name, val in [
                ("soc_start_mwh", soc_start),
                ("reported_protected_soc_min_mwh", reported_min),
                ("reported_protected_soc_max_mwh", reported_max),
            ]:
                if not np.isfinite(val):
                    missing_inputs.append(name)
            recompute_match = (abs(min_diff) <= 1e-6 and abs(max_diff) <= 1e-6) if (np.isfinite(min_diff) and np.isfinite(max_diff)) else False

            margin_to_min = soc_start - reported_min if (np.isfinite(soc_start) and np.isfinite(reported_min)) else np.nan
            margin_to_max = reported_max - soc_start if (np.isfinite(soc_start) and np.isfinite(reported_max)) else np.nan

            safe_pos = max(0.0, (soc_start - reported_min) * max(eta_out, 1e-12) / max(bem_activation_headroom_h, 1e-12)) if (np.isfinite(soc_start) and np.isfinite(reported_min)) else np.nan
            safe_neg = max(0.0, (reported_max - soc_start) / max(eta_in * bem_activation_headroom_h, 1e-12)) if (np.isfinite(soc_start) and np.isfinite(reported_max)) else np.nan
            pos_exceeds = bool(np.isfinite(safe_pos) and bem_sub_pos > safe_pos + 1e-9)
            neg_exceeds = bool(np.isfinite(safe_neg) and bem_sub_neg > safe_neg + 1e-9)

            da_or_activation = bool((da_buy_mwh > 1e-9) or (da_sell_mwh > 1e-9) or (bcm_act_pos > 1e-9) or (bcm_act_neg > 1e-9))
            aux_explains = bool(aux_e > 1e-9 and (pos_v > 1e-9 or neg_v > 1e-9))
            lock_explains = bool((lock_pos > 1e-9 or lock_neg > 1e-9) and (pos_v > 1e-9 or neg_v > 1e-9))
            stale_risk = bool(len(missing_inputs) > 0)

            base = {
                "scenario": scenario,
                "timestamp_utc": r.get("timestamp_utc"),
                "violation_side": side,
                "protected_soc_violation_pos_mwh": pos_v,
                "protected_soc_violation_neg_mwh": neg_v,
                "protected_soc_violation_total_mwh": pos_v + neg_v,
                "soc_start_mwh": soc_start,
                "soc_end_mwh": float(r.get("soc_end_mwh", np.nan)),
                "protected_soc_min_mwh": reported_min,
                "protected_soc_max_mwh": reported_max,
                "soc_min_mwh": soc_min_mwh,
                "soc_max_mwh": soc_max_mwh,
                "da_buy_mwh": da_buy_mwh,
                "da_sell_mwh": da_sell_mwh,
                "da_buy_accepted": da_buy_acc,
                "da_sell_accepted": da_sell_acc,
                "bem_only_submitted_pos_mw": bem_sub_pos,
                "bem_only_submitted_neg_mw": bem_sub_neg,
                "bem_only_executed_pos_mw": bem_exe_pos,
                "bem_only_executed_neg_mw": bem_exe_neg,
                "bcm_linked_activation_pos_mw": bcm_act_pos,
                "bcm_linked_activation_neg_mw": bcm_act_neg,
                "locked_reserve_pos_mw": lock_pos,
                "locked_reserve_neg_mw": lock_neg,
                "aux_energy_mwh": aux_e,
                "optimization_error_code": opt_code,
                "fallback_mode": fallback_mode,
            }
            viol_rows.append(base)

            recompute_rows.append(
                {
                    "scenario": scenario,
                    "timestamp_utc": r.get("timestamp_utc"),
                    "soc_start_mwh": soc_start,
                    "reported_protected_soc_min_mwh": reported_min,
                    "reported_protected_soc_max_mwh": reported_max,
                    "recomputed_protected_soc_min_mwh": recomputed_min,
                    "recomputed_protected_soc_max_mwh": recomputed_max,
                    "protected_min_diff_mwh": min_diff,
                    "protected_max_diff_mwh": max_diff,
                    "locked_reserve_pos_mw": lock_pos,
                    "locked_reserve_neg_mw": lock_neg,
                    "reserve_activation_headroom_h": reserve_activation_headroom_h,
                    "reserve_headroom_safety_mwh": reserve_headroom_safety_mwh,
                    "eta_in": eta_in,
                    "eta_out": eta_out,
                    "formula_version": "v1_locked_reserve_start_hour",
                    "recomputation_matches_reported": bool(recompute_match),
                    "missing_required_inputs": ",".join(missing_inputs),
                }
            )

            attrib = {
                "scenario": scenario,
                "timestamp_utc": r.get("timestamp_utc"),
                "violation_side": side,
                "soc_start_mwh": soc_start,
                "protected_soc_min_mwh": reported_min,
                "protected_soc_max_mwh": reported_max,
                "margin_to_min_mwh": margin_to_min,
                "margin_to_max_mwh": margin_to_max,
                "bem_only_submitted_pos_mw": bem_sub_pos,
                "bem_only_submitted_neg_mw": bem_sub_neg,
                "bem_only_executed_pos_mw": bem_exe_pos,
                "bem_only_executed_neg_mw": bem_exe_neg,
                "bem_activation_headroom_h": bem_activation_headroom_h,
                "bem_only_required_headroom_pos_mwh": bem_sub_pos * bem_activation_headroom_h / max(eta_out, 1e-12),
                "bem_only_required_headroom_neg_mwh": bem_sub_neg * bem_activation_headroom_h * eta_in,
                "bem_only_safe_pos_mw_from_reported_envelope": safe_pos,
                "bem_only_safe_neg_mw_from_reported_envelope": safe_neg,
                "bem_only_pos_exceeds_safe": pos_exceeds,
                "bem_only_neg_exceeds_safe": neg_exceeds,
                "da_buy_mwh": da_buy_mwh,
                "da_sell_mwh": da_sell_mwh,
                "bcm_linked_activation_pos_mw": bcm_act_pos,
                "bcm_linked_activation_neg_mw": bcm_act_neg,
                "aux_energy_mwh": aux_e,
                "da_or_activation_explains_violation": da_or_activation,
                "aux_explains_violation": aux_explains,
                "locked_reserve_explains_violation": lock_explains,
                "stale_or_missing_column_risk": stale_risk,
                "protected_min_diff_mwh": min_diff,
                "protected_max_diff_mwh": max_diff,
            }
            drv, detail = _classify_driver(pd.Series(attrib))
            attrib["suspected_driver"] = drv
            attrib["suspected_driver_detail"] = detail
            attrib_rows.append(attrib)

    ov_df = pd.DataFrame(overview_rows)
    viol_df = pd.DataFrame(viol_rows)
    rec_df = pd.DataFrame(recompute_rows)
    att_df = pd.DataFrame(attrib_rows)

    ov_df.to_csv(out_dir / "scenario_protected_soc_overview.csv", index=False)
    viol_df.to_csv(out_dir / "protected_soc_violation_rows.csv", index=False)
    rec_df.to_csv(out_dir / "protected_soc_recomputation.csv", index=False)
    att_df.to_csv(out_dir / "protected_soc_attribution.csv", index=False)

    decision_lines = [
        "# Working Version Decision",
        "",
        f"Run dir: `{run_dir}`",
        "",
    ]
    if ov_df.empty:
        decision_lines.append("No scenarios found with required summary files.")
    else:
        invalid = ov_df[(pd.to_numeric(ov_df.get("simulation_valid", 1.0), errors="coerce") < 0.5) | (pd.to_numeric(ov_df.get("thesis_reportable", 1.0), errors="coerce") < 0.5)]
        decision_lines.append(f"Total scenarios: {len(ov_df)}")
        decision_lines.append(f"Invalid/non-reportable scenarios: {len(invalid)}")
        if not att_df.empty:
            counts = att_df["suspected_driver"].value_counts(dropna=False)
            decision_lines.append("")
            decision_lines.append("## Suspected drivers")
            for k, v in counts.items():
                decision_lines.append(f"- {k}: {v}")
        if not att_df.empty:
            bem_proven = bool((att_df["suspected_driver"].astype(str).str.contains("bem_only")).any())
            decision_lines.append("")
            decision_lines.append(f"BEM-only proven as cause: {'yes' if bem_proven else 'no'}")
            must_fix = len(invalid) > 0
            decision_lines.append(f"Issue must be fixed for working thesis version: {'yes' if must_fix else 'no'}")
            if bem_proven:
                decision_lines.append("Minimum correct fix: hard cap BEM-only submitted MW by safe MW from the same reported protected envelope and start-of-hour SoC.")
            else:
                mismatch = bool((att_df["suspected_driver"] == "protected_soc_audit_formula_mismatch").any())
                if mismatch:
                    decision_lines.append("Minimum correct fix: align audit and submission formulas/timestamps; do not change optimizer behavior.")
                else:
                    decision_lines.append("Minimum correct fix not proven from current data; next step is enriched hourly trace for missing fields.")

    (out_dir / "working_version_decision.md").write_text("\n".join(decision_lines), encoding="utf-8")

    return {
        "scenarios": len(ov_df),
        "violation_rows": len(viol_df),
        "attribution_rows": len(att_df),
        "out_dir": str(out_dir),
    }


def compare_runs(run_a: Path, run_b: Path, out_dir: Path) -> None:
    def _load_overview(run_dir: Path) -> pd.DataFrame:
        p = out_dir / f"_{run_dir.name}_overview_cache.csv"
        if p.exists():
            return pd.read_csv(p)
        tmp = out_dir / f"_tmp_{run_dir.name}"
        res = diagnose_run(run_dir, tmp)
        ov = pd.read_csv(tmp / "scenario_protected_soc_overview.csv") if (tmp / "scenario_protected_soc_overview.csv").exists() else pd.DataFrame()
        if not ov.empty:
            ov.to_csv(p, index=False)
        return ov

    def _load_violation(run_dir: Path) -> pd.DataFrame:
        tmp = out_dir / f"_tmp_{run_dir.name}"
        p = tmp / "protected_soc_violation_rows.csv"
        if p.exists():
            return pd.read_csv(p)
        return pd.DataFrame()

    ov_a = _load_overview(run_a)
    ov_b = _load_overview(run_b)
    vv_a = _load_violation(run_a)
    vv_b = _load_violation(run_b)

    rows = []
    for run_name, ov, vv, disabled in [
        (run_a.name, ov_a, vv_a, 0),
        (run_b.name, ov_b, vv_b, 1 if "no_new_bcm" in run_b.name else 0),
    ]:
        if ov.empty:
            continue
        for _, s in ov.iterrows():
            sc = s.get("scenario")
            scv = vv[vv.get("scenario", "") == sc] if not vv.empty else pd.DataFrame()
            rows.append(
                {
                    "scenario": sc,
                    "run_name": run_name,
                    "protected_soc_violation_count": float(s.get("protected_soc_violation_count", np.nan)),
                    "protected_soc_violation_max_mwh": float(s.get("protected_soc_violation_max_mwh", np.nan)),
                    "bem_only_submitted_pos_mw_sum": float(pd.to_numeric(scv.get("bem_only_submitted_pos_mw", 0.0), errors="coerce").fillna(0.0).sum()) if not scv.empty else 0.0,
                    "bem_only_submitted_neg_mw_sum": float(pd.to_numeric(scv.get("bem_only_submitted_neg_mw", 0.0), errors="coerce").fillna(0.0).sum()) if not scv.empty else 0.0,
                    "bem_only_executed_pos_mw_sum": float(pd.to_numeric(scv.get("bem_only_executed_pos_mw", 0.0), errors="coerce").fillna(0.0).sum()) if not scv.empty else 0.0,
                    "bem_only_executed_neg_mw_sum": float(pd.to_numeric(scv.get("bem_only_executed_neg_mw", 0.0), errors="coerce").fillna(0.0).sum()) if not scv.empty else 0.0,
                    "locked_reserve_pos_mw_sum": float(pd.to_numeric(scv.get("locked_reserve_pos_mw", 0.0), errors="coerce").fillna(0.0).sum()) if not scv.empty else 0.0,
                    "locked_reserve_neg_mw_sum": float(pd.to_numeric(scv.get("locked_reserve_neg_mw", 0.0), errors="coerce").fillna(0.0).sum()) if not scv.empty else 0.0,
                    "new_bcm_disabled": float(disabled),
                    "invalid_reason": s.get("invalid_reason", ""),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "diagnostic_comparison_conservative_vs_no_new_bcm.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose protected SoC issues from existing simulation artifacts.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--compare-run-dir", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    res = diagnose_run(run_dir, out_dir)

    if args.compare_run_dir:
        run_b = Path(args.compare_run_dir).resolve()
        if run_b.exists():
            compare_runs(run_dir, run_b, out_dir)

    print("[OK] protected_soc diagnostics written")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
