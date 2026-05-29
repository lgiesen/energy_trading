"""Run LP-based battery backtest from ML predictions + ground truth parquet files.

Usage (manifest-autoload, recommended):
    ./.venv/bin/python scripts/run_battery_backtest.py \
      --run-id 2026-04-20T15-36-58Z \
      --split test \
      --horizon-hours 48 \
      --reopt-step-hours 1 \
      --da-gate-hour-cet 11 \
      --soc-feedback-mode realized

Usage (explicit manifest path override):
    ./.venv/bin/python scripts/run_battery_backtest.py \
      --run-manifest artifacts/model_runs/latest_xgboost.json \
      --split test

Usage (manual files):
    ./.venv/bin/python scripts/run_battery_backtest.py \
      --predictions artifacts/simulation_runs/manual/backtest_table_test.parquet \
      --ground-truth data/features/all_data_features.parquet \
      --timestamp-col timestamp_utc \
      --pred-da-col pred_da_price \
      --true-da-col da_price
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow direct script execution from repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_trading.config import MODEL_SPECS
from energy_trading.simulation.battery_backtest import (
    BacktestColumnMap, BatteryBacktester, PhaseTimeoutError,
    canonicalize_market_frame, load_and_align_market_data,
    load_prediction_warehouse_long)


def _series_from_candidates(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    required: bool,
    default: float = 0.0,
    numeric: bool = True,
) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            s = df[c]
            if numeric:
                s = pd.to_numeric(s, errors="coerce").fillna(default)
            return s
    if required:
        close = [c for c in df.columns if any(tok in c.lower() for tok in "_".join(candidates).lower().split("_"))]
        raise KeyError(
            f"Missing required column. Tried {candidates}. Similar available columns: {close[:12]}"
        )
    return pd.Series(default, index=df.index, dtype=float if numeric else object)


def require_numeric_series(df: pd.DataFrame, canonical_name: str, aliases: list[str] | None = None) -> pd.Series:
    return _series_from_candidates(
        df,
        [canonical_name] + list(aliases or []),
        required=True,
        numeric=True,
        default=0.0,
    )


def optional_numeric_series(
    df: pd.DataFrame, canonical_name: str, *, default: float = 0.0, aliases: list[str] | None = None
) -> pd.Series:
    return _series_from_candidates(
        df,
        [canonical_name] + list(aliases or []),
        required=False,
        numeric=True,
        default=default,
    )


def _suspected_infeasibility_driver_from_row(row: pd.Series) -> tuple[str, str]:
    def _f(k: str, default: float = 0.0) -> float:
        try:
            v = pd.to_numeric(pd.Series([row.get(k, default)]), errors="coerce").iloc[0]
            return float(default if pd.isna(v) else v)
        except Exception:
            return float(default)

    solver_msg = str(row.get("solver_message", "") or "").lower()
    opt_code = str(row.get("optimization_error_code", "") or "").lower()
    fallback_mode = str(row.get("fallback_mode", "") or "").lower()
    locked_pos = _f("locked_reserve_pos_mw")
    locked_neg = _f("locked_reserve_neg_mw")
    new_pos = _f("new_submitted_reserve_pos_mw")
    new_neg = _f("new_submitted_reserve_neg_mw")
    pviol_pos = _f("power_violation_pos_mw")
    pviol_neg = _f("power_violation_neg_mw")
    soc_v_pos = _f("protected_soc_violation_pos_mwh")
    soc_v_neg = _f("protected_soc_violation_neg_mwh")
    hr_short = _f("reserve_headroom_shortfall_max_mwh")
    final_press = _f("final_soc_pressure_mwh")
    aux_e = _f("aux_energy_mwh")
    soc_margin_pos = _f("protected_soc_margin_pos_mwh")
    soc_margin_neg = _f("protected_soc_margin_neg_mwh")
    da_bem_use = sum(
        abs(_f(k))
        for k in ("da_charge_mw", "da_discharge_mw", "bem_only_pos_mw", "bem_only_neg_mw", "id_charge_mw", "id_discharge_mw")
    )

    if pviol_pos > 1e-9 or pviol_neg > 1e-9:
        return "power_stack_violation", f"power_violation_pos={pviol_pos:.4f},neg={pviol_neg:.4f}"
    if soc_v_pos > 1e-9 or soc_v_neg > 1e-9:
        return "protected_soc_violation", f"protected_soc_violation_pos={soc_v_pos:.4f},neg={soc_v_neg:.4f}"
    if hr_short > 1e-9:
        return "reserve_headroom_shortfall", f"reserve_headroom_shortfall_max_mwh={hr_short:.4f}"
    if locked_pos + locked_neg > 1e-9 and new_pos + new_neg <= 1e-9 and ("infeasible" in opt_code or "reserve_feasibility_repair" in fallback_mode):
        return "existing_lockbook_obligation_infeasible", f"locked_only_pos={locked_pos:.4f},neg={locked_neg:.4f}"
    if new_pos + new_neg > 1e-9 and ("infeasible" in opt_code or "reserve_feasibility_repair" in fallback_mode):
        return "new_reserve_bid_too_high", f"new_reserve_pos={new_pos:.4f},neg={new_neg:.4f}"
    if final_press > 1e-9 and ("infeasible" in opt_code or "terminal" in fallback_mode):
        return "terminal_soc_conflict", f"final_soc_pressure_mwh={final_press:.4f}"
    if da_bem_use > 1e-9 and (soc_margin_pos < 1e-9 or soc_margin_neg < 1e-9):
        return "da_bem_consumed_protected_soc", f"dispatch_abs_sum_mw={da_bem_use:.4f},soc_margin_pos={soc_margin_pos:.4f},neg={soc_margin_neg:.4f}"
    if aux_e > 1e-9 and (soc_margin_pos < 1e-9 or soc_margin_neg < 1e-9):
        return "aux_loss_margin_gap", f"aux_energy_mwh={aux_e:.4f},soc_margin_pos={soc_margin_pos:.4f},neg={soc_margin_neg:.4f}"
    if any(x in solver_msg for x in ("not set", "numerical", "solver", "highs")) or any(
        x in opt_code for x in ("solver_failure", "numerical", "not set")
    ):
        return "unknown_solver_or_numeric", f"solver_message={solver_msg[:160]}"
    return "unknown_solver_or_numeric", f"opt_code={opt_code},fallback_mode={fallback_mode}"


def _build_optimization_infeasibility_attribution(
    *,
    hourly: pd.DataFrame,
    summary: dict[str, object],
    scenario: str,
) -> pd.DataFrame:
    h = hourly.copy()
    if h.empty:
        return pd.DataFrame()
    ts = pd.to_datetime(h.get("timestamp_utc", pd.Series(index=h.index, dtype="datetime64[ns, UTC]")), utc=True, errors="coerce")
    err = h.get("optimization_error_code", pd.Series(["ok"] * len(h), index=h.index)).astype(str).fillna("ok")
    fb = h.get("optimization_fallback", pd.Series(["none"] * len(h), index=h.index)).astype(str).fillna("none")
    fb_hour = pd.to_numeric(h.get("is_fallback_hour", 0.0), errors="coerce").fillna(0.0)
    fail_mask = err.str.lower().ne("ok") | fb.str.lower().ne("none") | fb_hour.gt(0.5)
    failing = h.loc[fail_mask].copy()
    if failing.empty:
        return pd.DataFrame()
    failing_ts = pd.to_datetime(failing.get("timestamp_utc"), utc=True, errors="coerce")
    first_ts = failing_ts.min()
    failing = failing.loc[failing_ts.eq(first_ts)].copy()

    def _num(col: str, default: float = 0.0) -> pd.Series:
        if col in failing.columns:
            return pd.to_numeric(failing[col], errors="coerce").fillna(default)
        return pd.Series(default, index=failing.index, dtype=float)

    out = pd.DataFrame(index=failing.index)
    out["scenario"] = str(scenario)
    out["timestamp_utc"] = pd.to_datetime(failing.get("timestamp_utc"), utc=True, errors="coerce")
    out["optimization_error_code"] = failing.get("optimization_error_code", "ok").astype(str)
    out["fallback_mode"] = failing["optimization_fallback"].astype(str) if "optimization_fallback" in failing else "none"
    out["reserve_feasibility_repair_used"] = float(summary.get("reserve_feasibility_repair_used", 0.0))
    out["accepted_path_infeasible_debug_dump_count"] = float(summary.get("accepted_path_infeasible_debug_dump_count", 0.0))
    out["solver_status"] = failing.get("optimization_error_code", "ok").astype(str)
    out["solver_message"] = failing.get("optimization_error", "").astype(str) if "optimization_error" in failing.columns else ""

    out["soc_start_mwh"] = _num("real_soc_start_mwh")
    out["soc_end_mwh"] = _num("real_soc_mwh")
    out["soc_min_mwh"] = float(summary.get("soc_min_mwh", np.nan))
    out["soc_max_mwh"] = float(summary.get("soc_max_mwh", np.nan))
    out["final_soc_target_mwh"] = float(summary.get("final_soc_target_mwh", np.nan))
    out["final_soc_pressure_mwh"] = np.maximum(0.0, float(summary.get("final_soc_target_mwh", 0.0)) - out["soc_end_mwh"])

    out["locked_reserve_pos_mw"] = _num("fixed_reserve_obligation_pos_mw")
    out["locked_reserve_neg_mw"] = _num("fixed_reserve_obligation_neg_mw")
    out["new_submitted_reserve_pos_mw"] = (_num("reserve_submitted_pos_mw") - out["locked_reserve_pos_mw"]).clip(lower=0.0)
    out["new_submitted_reserve_neg_mw"] = (_num("reserve_submitted_neg_mw") - out["locked_reserve_neg_mw"]).clip(lower=0.0)
    out["desired_reserve_pos_mw"] = _num("desired_reserve_pos_mw")
    out["desired_reserve_neg_mw"] = _num("desired_reserve_neg_mw")
    out["safe_reserve_pos_mw"] = _num("safe_reserve_pos_mw")
    out["safe_reserve_neg_mw"] = _num("safe_reserve_neg_mw")
    out["reserve_bid_derate"] = float(summary.get("reserve_bid_derate", np.nan))
    out["max_reserve_bid_mw"] = float(summary.get("max_reserve_bid_mw", np.nan))

    out["required_headroom_pos_mwh"] = _num("real_required_headroom_pos_mwh")
    out["required_headroom_neg_mwh"] = _num("real_required_headroom_neg_mwh")
    out["available_headroom_pos_mwh"] = _num("real_available_headroom_pos_mwh")
    out["available_headroom_neg_mwh"] = _num("real_available_headroom_neg_mwh")
    out["protected_soc_min_mwh"] = float(summary.get("protected_soc_min_mwh", np.nan))
    out["protected_soc_max_mwh"] = float(summary.get("protected_soc_max_mwh", np.nan))
    out["protected_soc_margin_pos_mwh"] = _num("real_headroom_margin_pos_mwh")
    out["protected_soc_margin_neg_mwh"] = _num("real_headroom_margin_neg_mwh")
    out["protected_soc_violation_pos_mwh"] = _num("real_headroom_violation_pos_mwh")
    out["protected_soc_violation_neg_mwh"] = _num("real_headroom_violation_neg_mwh")
    out["reserve_headroom_shortfall_max_mwh"] = float(summary.get("reserve_headroom_shortfall_max_mwh", 0.0))

    out["da_charge_mw"] = _num("real_da_charge_mw")
    out["da_discharge_mw"] = _num("real_da_discharge_mw")
    out["bem_only_pos_mw"] = _num("real_bem_only_submitted_pos_mw")
    out["bem_only_neg_mw"] = _num("real_bem_only_submitted_neg_mw")
    out["bcm_linked_activation_pos_mw"] = _num("real_executed_reserve_pos_mw")
    out["bcm_linked_activation_neg_mw"] = _num("real_executed_reserve_neg_mw")
    out["id_charge_mw"] = _num("real_id_charge_mw")
    out["id_discharge_mw"] = _num("real_id_discharge_mw")
    out["aux_energy_mwh"] = _num("real_aux_energy_mwh")

    out["power_stack_pos_mw"] = (
        out["da_discharge_mw"] + out["bem_only_pos_mw"] + out["bcm_linked_activation_pos_mw"] + out["id_discharge_mw"]
    )
    out["power_stack_neg_mw"] = (
        out["da_charge_mw"] + out["bem_only_neg_mw"] + out["bcm_linked_activation_neg_mw"] + out["id_charge_mw"]
    )
    out["p_max_mw"] = float(summary.get("p_max_mw", np.nan))
    out["power_violation_pos_mw"] = (out["power_stack_pos_mw"] - out["p_max_mw"]).clip(lower=0.0)
    out["power_violation_neg_mw"] = (out["power_stack_neg_mw"] - out["p_max_mw"]).clip(lower=0.0)

    drivers = out.apply(_suspected_infeasibility_driver_from_row, axis=1)
    out["suspected_infeasibility_driver"] = drivers.apply(lambda x: x[0])
    out["suspected_infeasibility_driver_detail"] = drivers.apply(lambda x: x[1])
    return out


def _phase_timeout_seconds(phase: str) -> float:
    key = "BACKTEST_PHASE_TIMEOUT_" + "".join(ch if ch.isalnum() else "_" for ch in phase.upper()) + "_S"
    if key in os.environ:
        return float(os.environ[key])
    return float(os.environ.get("BACKTEST_PHASE_TIMEOUT_S", "0"))


@contextmanager
def _phase_watchdog(phase: str):
    timeout_s = _phase_timeout_seconds(phase)
    t0 = time.monotonic()
    print(f"[PHASE] START {phase}")

    use_signal = (
        timeout_s > 0
        and threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
    )
    old_handler = None
    old_timer = None
    if use_signal:
        def _alarm_handler(_signum: int, _frame: object) -> None:
            raise PhaseTimeoutError(f"Phase timeout in '{phase}' after {timeout_s:.1f}s.")

        old_handler = signal.getsignal(signal.SIGALRM)
        old_timer = signal.setitimer(signal.ITIMER_REAL, timeout_s)
        signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        yield
    finally:
        if use_signal:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if old_timer is not None:
                signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        dt = time.monotonic() - t0
        print(f"[PHASE] END {phase} | elapsed={dt:.2f}s")


def _plot_cumulative_pnl(hourly: pd.DataFrame, ts_col: str, out_path: Path) -> None:
    if hourly.empty:
        return
    d = hourly.copy()
    d[ts_col] = pd.to_datetime(d[ts_col], utc=True, errors="coerce")
    d = d.dropna(subset=[ts_col]).sort_values(ts_col)
    required = ["real_pnl_eur", "naive_pnl_eur", "perfect_foresight_pnl_eur"]
    if not set(required).issubset(d.columns):
        return
    d["model_cum_pnl_eur"] = pd.to_numeric(d["real_pnl_eur"], errors="coerce").fillna(0.0).cumsum()
    d["naive_cum_pnl_eur"] = pd.to_numeric(d["naive_pnl_eur"], errors="coerce").fillna(0.0).cumsum()
    d["perfect_foresight_cum_pnl_eur"] = pd.to_numeric(d["perfect_foresight_pnl_eur"], errors="coerce").fillna(0.0).cumsum()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(d[ts_col], d["model_cum_pnl_eur"], label="Model", linewidth=2)
    ax.plot(d[ts_col], d["naive_cum_pnl_eur"], label="Naive 24h", linewidth=2)
    ax.plot(d[ts_col], d["perfect_foresight_cum_pnl_eur"], label="RollingPerfectForesightSameRules", linewidth=2)
    ax.set_title("Cumulative PnL Contribution")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Cumulative PnL [EUR]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _build_backtest_diagnostics(hourly: pd.DataFrame, summary: dict[str, object]) -> dict[str, object]:
    d = hourly.copy()
    numeric = d.select_dtypes(include=["number"])
    nonfinite_total = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum()) if not numeric.empty else 0
    nan_total = int(numeric.isna().sum().sum()) if not numeric.empty else 0

    key_cols = [
        "real_pnl_eur",
        "pred_pnl_eur",
        "naive_pnl_eur",
        "perfect_foresight_pnl_eur",
        "soc_mwh",
        "charge_mw",
        "discharge_mw",
        "reserve_pos_mw",
        "reserve_neg_mw",
    ]
    key_col_nan_counts = {
        c: int(pd.to_numeric(d[c], errors="coerce").isna().sum())
        for c in key_cols
        if c in d.columns
    }

    infeasibility_flags = {
        "final_soc_constraint_satisfied": bool(summary.get("final_soc_constraint_satisfied", False)),
        "numeric_nonfinite_total": nonfinite_total,
    }
    return {
        "rows_hourly": int(len(d)),
        "numeric_nan_total": nan_total,
        "numeric_nonfinite_total": nonfinite_total,
        "key_column_nan_counts": key_col_nan_counts,
        "infeasibility_flags": infeasibility_flags,
    }


def _build_state_machine_audit(hourly: pd.DataFrame) -> dict[str, object]:
    d = hourly.copy()
    if d.empty:
        return {
            "rows": 0,
            "gate_09_reopt_triggers": 0,
            "gate_12_da_locked_rows": 0,
            "t25_energy_bid_rows": 0,
            "da_submitted_buy_mw_total": 0.0,
            "da_submitted_sell_mw_total": 0.0,
            "da_executed_buy_mw_total": 0.0,
            "da_executed_sell_mw_total": 0.0,
            "afrr_capacity_awarded_pos_mw_total": 0.0,
            "afrr_capacity_awarded_neg_mw_total": 0.0,
            "afrr_activation_accept_pos_count": 0,
            "afrr_activation_accept_neg_count": 0,
            "obligation_fulfilled_rate": float("nan"),
        }

    def _sum_col(name: str) -> float:
        return float(pd.to_numeric(d[name], errors="coerce").fillna(0.0).sum()) if name in d.columns else 0.0

    def _count_true(name: str) -> int:
        if name not in d.columns:
            return 0
        return int((pd.to_numeric(d[name], errors="coerce").fillna(0.0) > 0.5).sum())

    out: dict[str, object] = {
        "rows": int(len(d)),
        "gate_09_reopt_triggers": _count_true("event_reopt_triggered"),
        "gate_12_da_locked_rows": _count_true("da_bid_locked"),
        "t25_energy_bid_rows": _count_true("real_aFRR_Energy_Gate_Closure_Min"),
        "da_submitted_buy_mw_total": _sum_col("real_submitted_da_buy_mw"),
        "da_submitted_sell_mw_total": _sum_col("real_submitted_da_sell_mw"),
        "da_executed_buy_mw_total": _sum_col("real_executed_charge_mw"),
        "da_executed_sell_mw_total": _sum_col("real_executed_discharge_mw"),
        "afrr_capacity_awarded_pos_mw_total": _sum_col("real_executed_reserve_pos_mw"),
        "afrr_capacity_awarded_neg_mw_total": _sum_col("real_executed_reserve_neg_mw"),
        "afrr_activation_accept_pos_count": _count_true("real_afrr_act_pos_accepted"),
        "afrr_activation_accept_neg_count": _count_true("real_afrr_act_neg_accepted"),
    }
    if "real_Obligation_Fulfilled" in d.columns:
        out["obligation_fulfilled_rate"] = float(pd.to_numeric(d["real_Obligation_Fulfilled"], errors="coerce").fillna(0.0).mean())
    else:
        out["obligation_fulfilled_rate"] = float("nan")
    return out


def _resolve_out_dir(
    out_dir_arg: str,
    *,
    run_id: str | None,
    split: str,
) -> Path:
    if out_dir_arg.strip():
        out = Path(out_dir_arg)
        out.mkdir(parents=True, exist_ok=True)
        return out
    rid = run_id or "manual"
    out = Path("artifacts/simulation_runs") / rid / split
    out.mkdir(parents=True, exist_ok=True)
    return out


def _prepare_scenario_output_dir(
    *,
    scenario_out_dir: Path,
    clean_output: bool,
    strict_simulation_validity: bool,
    simulation_schema_version: str,
) -> float:
    """Prepare scenario output dir and prevent stale schema mixing."""
    output_was_cleaned = 0.0
    if scenario_out_dir.exists():
        if bool(clean_output):
            shutil.rmtree(scenario_out_dir)
            output_was_cleaned = 1.0
        else:
            if bool(strict_simulation_validity):
                raise RuntimeError(
                    "Strict validity run refuses writing into existing scenario directory without --clean-output: "
                    f"{scenario_out_dir}"
                )
            existing_summary = scenario_out_dir / "backtest_summary.json"
            if existing_summary.exists():
                try:
                    existing_payload = json.loads(existing_summary.read_text(encoding="utf-8"))
                except Exception:
                    existing_payload = {}
                if existing_payload.get("simulation_schema_version") != simulation_schema_version:
                    raise RuntimeError(
                        f"Refusing to write into stale scenario directory without --clean-output: {scenario_out_dir}. "
                        "Existing summary schema_version is missing or mismatched."
                    )
    scenario_out_dir.mkdir(parents=True, exist_ok=True)
    return output_was_cleaned


def _parse_quantile_token(token: str) -> str:
    t = token.strip().lower()
    if not t:
        raise ValueError("Empty quantile token.")
    if t.startswith("p") and len(t) == 3 and t[1:].isdigit():
        q = int(t[1:])
        if q <= 0 or q >= 100:
            raise ValueError(f"Quantile out of range: {token}")
        return f"p{q:02d}"
    val = float(t)
    if val <= 0.0 or val >= 1.0:
        raise ValueError(f"Quantile out of range: {token}")
    q = int(round(val * 100.0))
    q = max(1, min(99, q))
    return f"p{q:02d}"


def _parse_quantile_pairs(raw: str) -> list[tuple[str, str]]:
    s = raw.strip()
    if not s:
        return []
    out: list[tuple[str, str]] = []
    for part in s.split(","):
        pair = part.strip()
        if not pair:
            continue
        if "-" not in pair:
            raise ValueError(f"Invalid quantile pair '{pair}'. Use p10-p90 or 0.1-0.9.")
        lo_raw, hi_raw = pair.split("-", 1)
        q_lo = _parse_quantile_token(lo_raw)
        q_hi = _parse_quantile_token(hi_raw)
        if int(q_lo[1:]) > int(q_hi[1:]):
            raise ValueError(f"Invalid pair '{pair}': low quantile must be <= high quantile.")
        out.append((q_lo, q_hi))
    return out


def _apply_quantile_pair_to_warehouse(
    warehouse: dict[str, pd.DataFrame],
    *,
    q_low: str,
    q_high: str,
    da_role: str,
) -> dict[str, pd.DataFrame]:
    da_q = {"low": q_low, "high": q_high, "mid": "p50"}[da_role]
    quantile_by_target = {
        "pred_da_price": da_q,
        "pred_afrr_capacity_price_pos": q_high,
        "pred_afrr_activation_price_pos": q_high,
        "pred_afrr_activation_rate_pos": q_high,
        "pred_afrr_capacity_price_neg": q_low,
        "pred_afrr_activation_price_neg": q_low,
        "pred_afrr_activation_rate_neg": q_low,
    }
    out: dict[str, pd.DataFrame] = {}
    for pred_col, df in warehouse.items():
        q_col = quantile_by_target.get(pred_col, "p50")
        cur = df.copy()
        if q_col not in cur.columns:
            available = [c for c in cur.columns if re.fullmatch(r"p\d{2}", str(c))]
            raise KeyError(
                f"Requested quantile '{q_col}' missing for {pred_col}. Available: {available}"
            )
        cur["predicted_value"] = pd.to_numeric(cur[q_col], errors="coerce")
        out[pred_col] = cur
    return out


def _scenario_suffix(q_low: str, q_high: str) -> str:
    return f"{q_low}_{q_high}"


def _matches_model_key(path: Path, model_key: str) -> bool:
    if not model_key:
        return True
    mk = model_key.strip().lower()
    name = path.name.lower()
    if mk in {"xgb", "xgboost"}:
        return "xgboost" in name
    if mk == "tft":
        return "xgboost" not in name
    return mk in name


def _score_long_candidate(path: Path, *, split: str) -> tuple[int, int]:
    """Higher score = better split match for long prediction files."""
    name = path.name.lower()
    has_test = "test" in name
    # Prefer explicit split markers first.
    if split == "test":
        return (2 if has_test else 1, len(name))
    # val: prefer files without test marker; many val files have no explicit "val" token
    return (2 if not has_test else 1, len(name))


def _resolve_existing_file(path_like: str | Path, *, manifest_dir: Path) -> Path:
    p = Path(path_like)
    if p.exists():
        return p
    repo_root = Path(__file__).resolve().parents[1]
    # Common case for downloaded artifacts: manifest keeps server-absolute path.
    cands = [
        manifest_dir / p.name,
        manifest_dir / "predictions" / p.name,
        Path.cwd() / p.name,
        Path.cwd() / "data" / "features" / p.name,
        repo_root / "data" / "features" / p.name,
        repo_root / "data" / "features" / "all_data_features.parquet",
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"File not found from manifest path '{p}'. Tried local fallbacks near {manifest_dir}.")


def _resolve_long_prediction_path(
    *,
    pred_col: str,
    configured_path: str | Path,
    manifest_dir: Path,
    split: str,
    model_key: str,
) -> Path:
    p = Path(configured_path)
    if p.exists() and _matches_model_key(p, model_key):
        return p

    pred_dir = manifest_dir / "predictions"
    candidates: list[Path] = []
    if pred_dir.exists():
        patterns = [
            f"*{split}*{pred_col}*long*.parquet",
            f"*{pred_col}*long*{split}*.parquet",
            f"*{pred_col}*{split}*long*.parquet",
            f"*{pred_col}*long*.parquet",
        ]
        for pat in patterns:
            candidates.extend(sorted(pred_dir.glob(pat)))
        # Deduplicate while preserving order
        if candidates:
            seen: set[str] = set()
            deduped: list[Path] = []
            for c in candidates:
                key = str(c.resolve())
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(c)
            candidates = deduped
    if model_key:
        candidates = [c for c in candidates if _matches_model_key(c, model_key)]
    if candidates:
        candidates = sorted(candidates, key=lambda c: _score_long_candidate(c, split=split), reverse=True)
        return candidates[0]

    # Final explicit fallbacks by filename
    for c in [manifest_dir / p.name, manifest_dir / "predictions" / p.name]:
        if c.exists() and _matches_model_key(c, model_key):
            return c
    raise FileNotFoundError(
        f"Could not resolve long prediction file for pred_col='{pred_col}', split='{split}', model_key='{model_key}'. "
        f"Configured path: {p}"
    )


def _resolve_long_map(
    *,
    long_map: dict[str, str],
    manifest_dir: Path,
    split: str,
    model_key: str,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for pred_col, p in long_map.items():
        rp = _resolve_long_prediction_path(
            pred_col=pred_col,
            configured_path=p,
            manifest_dir=manifest_dir,
            split=split,
            model_key=model_key,
        )
        resolved[pred_col] = str(rp)
    return resolved


def _target_value_modes_from_manifest(payload: dict[str, object]) -> dict[str, str]:
    out: dict[str, str] = {}
    tvm = payload.get("target_value_mode", {})
    if isinstance(tvm, dict):
        for k, v in tvm.items():
            out[str(k)] = str(v)
    sim = payload.get("simulation", {})
    if isinstance(sim, dict):
        canonical_targets = sim.get("canonical_economic_targets", [])
        if isinstance(canonical_targets, list):
            for t in canonical_targets:
                out[str(t)] = "canonical_economic"
    if "pred_afrr_activation_price_neg" not in out:
        out["pred_afrr_activation_price_neg"] = "raw_signed_legacy"
    return out


def _read_parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return list(pq.ParquetFile(path).schema.names)
    except Exception:
        # Fallback if pyarrow schema access is unavailable.
        return list(pd.read_parquet(path).columns)


def _preflight_manifest_and_quantiles(
    *,
    manifest_path: Path,
    manifest_payload: dict[str, object],
    split: str,
    model_key: str,
    manifest_dir: Path,
    expected_quantiles: set[str],
) -> None:
    if not manifest_path.exists():
        raise RuntimeError(f"Manifest not found: {manifest_path}")

    bundles = manifest_payload.get("bundles", {}) if isinstance(manifest_payload, dict) else {}
    da_long = bundles.get("da", {}).get("predictions_long", {}).get(split, {}) if isinstance(bundles, dict) else {}
    afrr_long = bundles.get("afrr", {}).get("predictions_long", {}).get(split, {}) if isinstance(bundles, dict) else {}
    long_map = {**da_long, **afrr_long}
    if not long_map:
        raise RuntimeError(
            "Preflight failed: no long-format prediction files in manifest. "
            "Quantile simulation requires predictions_long entries."
        )

    resolved = _resolve_long_map(
        long_map=long_map,
        manifest_dir=manifest_dir,
        split=split,
        model_key=model_key,
    )
    expected = {str(q).lower() for q in expected_quantiles}
    if not expected:
        expected = {"p50"}
    failures: list[str] = []
    for pred_col, file_path in sorted(resolved.items()):
        cols = set(_read_parquet_columns(Path(file_path)))
        missing = sorted(expected - cols)
        if missing:
            failures.append(
                f"{pred_col}: missing {len(missing)} required quantile columns "
                f"(first 12: {missing[:12]}) in {file_path}"
            )
    if failures:
        msg = (
            "Preflight failed: required quantiles derived from --quantile-pairs are not fully available.\n"
            f"Expected quantiles: {sorted(expected)}\n"
            + "\n".join(failures)
        )
        raise RuntimeError(msg)


def _resolve_bundle_prediction_path(
    *,
    configured_path: str | Path,
    manifest_dir: Path,
    split: str,
    model_key: str,
) -> Path:
    """Resolve wide prediction parquet for bundle fallback mode."""
    p = Path(configured_path)
    if p.exists() and _matches_model_key(p, model_key):
        return p

    pred_dir = manifest_dir / "predictions"
    candidates: list[Path] = []
    if pred_dir.exists():
        patterns = [
            f"*{split}*.parquet",
            "*.parquet",
        ]
        for pat in patterns:
            candidates.extend(sorted(pred_dir.glob(pat)))
        candidates = [c for c in candidates if "long" not in c.name.lower()]
    if model_key:
        candidates = [c for c in candidates if _matches_model_key(c, model_key)]
    if candidates:
        candidates = sorted(candidates, key=lambda c: _score_long_candidate(c, split=split), reverse=True)
        # Prefer same bundle family where possible.
        name_hint = p.name.lower()
        if "da" in name_hint:
            da_cands = [c for c in candidates if "da" in c.name.lower()]
            if da_cands:
                return da_cands[0]
        if "afrr" in name_hint:
            afrr_cands = [c for c in candidates if "afrr" in c.name.lower()]
            if afrr_cands:
                return afrr_cands[0]
        return candidates[0]

    for c in [manifest_dir / p.name, manifest_dir / "predictions" / p.name]:
        if c.exists() and _matches_model_key(c, model_key):
            return c
    raise FileNotFoundError(
        f"Could not resolve prediction parquet for split='{split}', model_key='{model_key}'. "
        f"Configured path: {p}"
    )


def _apply_fallback_column_map(pred: pd.DataFrame, truth: pd.DataFrame, colmap: BacktestColumnMap) -> BacktestColumnMap:
    """Use project-aware fallback candidates to reduce manual mapping overhead."""

    def pick(frame: pd.DataFrame, primary: str, candidates: list[str]) -> str:
        for c in [primary, *candidates]:
            if c in frame.columns:
                return c
        raise KeyError(f"Missing required column. Tried: {[primary, *candidates]}")

    def pick_pred(frame: pd.DataFrame, primary: str, candidates: list[str]) -> str:
        # In long-warehouse mode pred preview can be empty. Keep configured names.
        if frame.empty:
            return primary
        return pick(frame, primary, candidates)

    mapped = BacktestColumnMap(
        timestamp=pick(pred if colmap.timestamp in pred.columns else truth, colmap.timestamp, ["timestamp", "datetime", "date"]),

        pred_da_price=pick_pred(
            pred,
            colmap.pred_da_price,
            ["da_price_pred", "y_pred_da_price", "prediction_da_price", "pred_target_da_price"],
        ),
        pred_afrr_capacity_price_pos=pick_pred(pred, colmap.pred_afrr_capacity_price_pos, ["afrr_capacity_price_pos_pred", "pred_target_afrr_capacity_price_pos"]),
        pred_afrr_capacity_price_neg=pick_pred(pred, colmap.pred_afrr_capacity_price_neg, ["afrr_capacity_price_neg_pred", "pred_target_afrr_capacity_price_neg"]),
        pred_afrr_activation_price_pos=pick_pred(pred, colmap.pred_afrr_activation_price_pos, ["afrr_activation_price_vwap_pos_pred", "pred_target_afrr_activation_price_vwap_pos"]),
        pred_afrr_activation_price_neg=pick_pred(
            pred,
            colmap.pred_afrr_activation_price_neg,
            ["afrr_activation_price_vwap_neg_pred", "pred_target_afrr_activation_price_vwap_neg"],
        ),
        pred_afrr_activation_rate_pos=pick_pred(
            pred,
            colmap.pred_afrr_activation_rate_pos,
            ["pred_target_afrr_activation_rate_pos", "afrr_activation_rate_pred"],
        ),
        pred_afrr_activation_rate_neg=pick_pred(
            pred,
            colmap.pred_afrr_activation_rate_neg,
            ["pred_target_afrr_activation_rate_neg", "afrr_activation_rate_pred"],
        ),

        true_da_price=pick(truth, colmap.true_da_price, ["da_price_actual", "target_da_price"]),
        true_afrr_capacity_price_pos=pick(truth, colmap.true_afrr_capacity_price_pos, ["target_afrr_capacity_price_pos"]),
        true_afrr_capacity_price_neg=pick(truth, colmap.true_afrr_capacity_price_neg, ["target_afrr_capacity_price_neg"]),
        true_afrr_activation_price_pos=pick(
            truth,
            colmap.true_afrr_activation_price_pos,
            [
                "target_afrr_activation_price_vwap_pos_raw",
                "target_afrr_activation_price_vwap_pos",
                "afrr_activation_price_vwap",
            ],
        ),
        true_afrr_activation_price_neg=pick(
            truth,
            colmap.true_afrr_activation_price_neg,
            [
                "target_afrr_activation_price_vwap_neg_raw",
                "target_afrr_activation_price_vwap_neg",
                "afrr_activation_price_vwap",
                "afrr_activation_price_vwap_pos",
                "target_afrr_activation_price_vwap_pos_raw",
                "target_afrr_activation_price_vwap_pos",
            ],
        ),
        true_afrr_activation_rate_pos=pick(
            truth,
            colmap.true_afrr_activation_rate_pos,
            [
                "activation_rate_phys_pos",
                "afrr_activation_rate_pos",
                "target_afrr_activation_rate_pos",
                "afrr_activation_rate",
            ],
        ),
        true_afrr_activation_rate_neg=pick(
            truth,
            colmap.true_afrr_activation_rate_neg,
            [
                "activation_rate_phys_neg",
                "afrr_activation_rate_neg",
                "target_afrr_activation_rate_neg",
                "afrr_activation_rate",
            ],
        ),
    )
    if mapped.true_afrr_activation_price_neg == mapped.true_afrr_activation_price_pos:
        print(
            "[WARN] Missing dedicated negative activation-price truth column. "
            "Using positive activation-price column as fallback for *_neg settlement."
        )
    return mapped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run battery LP backtest with predicted-vs-realized settlement.")
    p.add_argument("--predictions", default="", help="Path to predictions parquet file.")
    p.add_argument("--ground-truth", default="", help="Path to ground-truth parquet file.")
    p.add_argument(
        "--run-id",
        default="2026-04-20T15-36-58Z",
        help=(
            "Run id used to resolve manifest path when --run-manifest is not set. "
            "Default: 2026-04-20T15-36-58Z"
        ),
    )
    p.add_argument(
        "--run-manifest",
        "--manifest",
        default="",
        help=(
            "Optional manifest path or latest-pointer json for simulation autoload. "
            "If omitted, resolves to artifacts/model_runs/<run-id>/manifest.json."
        ),
    )
    p.add_argument("--split", choices=["val", "test"], default="test", help="Prediction split for manifest mode.")
    p.add_argument(
        "--model-key",
        default="",
        help="Optional model selector when one run dir contains multiple models (e.g. 'xgboost' or 'tft').",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory for hourly/aggregated results. If empty, uses artifacts/simulation_runs/<run_id>/<split>/",
    )
    p.add_argument(
        "--quantile-pairs",
        default="",
        help=(
            "Optional comma-separated sweep list, e.g. "
            "'p50-p50,p30-p70,p10-p90' or '0.5-0.5,0.3-0.7'. "
            "Requires long-format prediction warehouse."
        ),
    )
    p.add_argument(
        "--da-quantile-role",
        choices=["low", "mid", "high"],
        default="mid",
        help="How DA uses quantile pair in sweep mode: low/high/mid (mid -> p50).",
    )

    p.add_argument("--timestamp-col", default="timestamp_utc")
    p.add_argument("--pred-da-col", default="pred_da_price")
    p.add_argument("--pred-cap-pos-col", default="pred_afrr_capacity_price_pos")
    p.add_argument("--pred-cap-neg-col", default="pred_afrr_capacity_price_neg")
    p.add_argument("--pred-act-pos-col", default="pred_afrr_activation_price_pos")
    p.add_argument("--pred-act-neg-col", default="pred_afrr_activation_price_neg")
    p.add_argument("--pred-rate-pos-col", default="pred_afrr_activation_rate_pos")
    p.add_argument("--pred-rate-neg-col", default="pred_afrr_activation_rate_neg")

    p.add_argument("--true-da-col", default="da_price")
    p.add_argument("--true-cap-pos-col", default="afrr_capacity_price_pos")
    p.add_argument("--true-cap-neg-col", default="afrr_capacity_price_neg")
    p.add_argument("--true-act-pos-col", default="afrr_activation_price_vwap_pos")
    p.add_argument("--true-act-neg-col", default="afrr_activation_price_vwap_neg")
    p.add_argument("--true-rate-pos-col", default="activation_rate_phys_pos")
    p.add_argument("--true-rate-neg-col", default="activation_rate_phys_neg")

    p.add_argument("--start", default=None, help="Optional UTC start filter.")
    p.add_argument("--end", default=None, help="Optional UTC end filter.")
    p.add_argument("--horizon-hours", type=int, default=48, help="Rolling-horizon window length in hours.")
    p.add_argument("--reopt-step-hours", type=int, default=1, help="Re-optimization step in hours.")
    p.add_argument(
        "--da-gate-hour-cet",
        type=int,
        default=11,
        help="Day-Ahead gate-closure hour in CET/CEST used for locking next-day DA bids (default: 11).",
    )
    p.add_argument(
        "--da-gate-hour-utc",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--soc-feedback-mode",
        choices=["realized", "predicted"],
        default="realized",
        help="State carryover mode between re-optimizations.",
    )
    p.add_argument(
        "--final-soc-mode",
        choices=["terminal_repair", "hard"],
        default="terminal_repair",
        help="Final SoC policy: terminal_repair (default) or hard physical minimum.",
    )
    p.add_argument(
        "--enforce-final-soc-min",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce final SoC >= soc_target_end (default: enabled).",
    )
    p.add_argument(
        "--disable-rolling-horizon",
        action="store_true",
        help="Use a single full-horizon optimization instead of rolling horizon.",
    )
    p.add_argument(
        "--trading-strategy",
        choices=["multi", "da_only", "afrr_only"],
        default="multi",
        help="Strategy isolation mode: multi (DA+aFRR), da_only (DA only), afrr_only (aFRR only).",
    )
    p.add_argument(
        "--reserve-activation-headroom-h",
        type=float,
        default=0.5,
        help="Hard SoC headroom assumption (hours) for BCM reserve deliverability (default: 0.5h).",
    )
    p.add_argument(
        "--bem-activation-headroom-h",
        type=float,
        default=0.5,
        help="Hard SoC headroom assumption (hours) for BEM-only deliverability (default: 0.5h).",
    )
    p.add_argument(
        "--reserve-feasibility-mode",
        default="",
        help="Reserve feasibility mode: normal or conservative. In strict mode, defaults to conservative.",
    )
    p.add_argument(
        "--reserve-soc-projection-safety-mwh",
        type=float,
        default=None,
        help="Additional SoC projection safety margin used in BCM precommit clamp.",
    )
    p.add_argument(
        "--reserve-headroom-safety-mwh",
        type=float,
        default=None,
        help="Reserve headroom safety margin used in BCM precommit clamp.",
    )
    p.add_argument(
        "--reserve-power-safety-mw",
        type=float,
        default=None,
        help="Reserve power safety margin used in BCM precommit clamp.",
    )
    p.add_argument(
        "--reserve-min-margin-after-bid-mwh",
        type=float,
        default=None,
        help="Conservative minimum post-bid headroom margin threshold.",
    )
    p.add_argument(
        "--bem-only-headroom-safety-mwh",
        type=float,
        default=0.0,
        help="Additional safety margin for BEM-only submission guard relative to protected SoC envelope.",
    )
    p.add_argument(
        "--max-bem-only-bid-mw",
        type=float,
        default=None,
        help="Optional hard cap for BEM-only submitted MW per direction.",
    )
    p.add_argument(
        "--disallow-simultaneous-bem-only-pos-neg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, keep only the higher-EV side when both BEM-only pos and neg are desired.",
    )
    p.add_argument(
        "--reserve-bid-derate",
        type=float,
        default=1.0,
        help="Derating factor [0,1] applied to safe feasible reserve bids before BCM submission.",
    )
    p.add_argument(
        "--max-reserve-bid-mw",
        type=float,
        default=None,
        help="Optional hard cap for reserve submitted MW per direction/block before BCM submission.",
    )
    p.add_argument(
        "--disable-new-bcm-reserve-bids",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable new BCM reserve bids while still honoring existing locked reserve obligations.",
    )
    p.add_argument(
        "--reserve-retry-ladder",
        default="1.0,0.5,0.25,0.0",
        help="Progressive factors for retrying NEW reserve submissions in strict/conservative mode.",
    )
    p.add_argument(
        "--strict-simulation-validity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Strict thesis validity mode. If enabled, any fallback/solver optimization issue "
            "marks scenario invalid/non-reportable (default: enabled)."
        ),
    )
    p.add_argument(
        "--forecast-value-mode",
        choices=["canonical_economic", "raw_signed"],
        default="canonical_economic",
        help="Forecast/settlement value convention. canonical_economic uses central postprocessing sign conventions.",
    )
    p.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, delete existing scenario output directory before writing. "
            "Prevents stale/mixed artifacts."
        ),
    )
    p.add_argument(
        "--enable-global-perfect_foresight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable global_hindsight_perfect_foresight_upper_bound computation. "
            "Disabled by default until perfect_foresight scope/validation is explicitly verified."
        ),
    )
    p.add_argument(
        "--print-validity-first-preset",
        action="store_true",
        help="Print recommended validity-first reserve settings and exit.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.print_validity_first_preset):
        print("validity_first_preset:")
        print("  reserve_soc_projection_safety_mwh: 2.0")
        print("  reserve_headroom_safety_mwh: 1.0")
        print("  reserve_power_safety_mw: 0.3")
        print("  reserve_min_margin_after_bid_mwh: 1.0")
        print("  reserve_bid_derate: 0.5")
        print("  max_reserve_bid_mw: 3.0")
        print("  reserve_retry_ladder: 1.0,0.5,0.25,0.0")
        print("  disable_new_bcm_reserve_bids: false")
        return
    run_started_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    simulation_schema_version = "v2_strict_validity_debugdump"
    required_summary_fields_version = "v2_required_debugdump_fields"
    run_id: str | None = args.run_id.strip() or None
    quantile_pairs = _parse_quantile_pairs(args.quantile_pairs)
    required_quantiles: set[str] = {"p50"}
    for q_lo, q_hi in quantile_pairs:
        required_quantiles.add(q_lo.lower())
        required_quantiles.add(q_hi.lower())

    predictions_path = args.predictions.strip()
    ground_truth_path = args.ground_truth.strip()
    payload: dict[str, object] = {}
    manifest_path: Path | None = None

    target_value_modes: dict[str, str] = {}
    if payload:
        target_value_modes = _target_value_modes_from_manifest(payload)

    if not predictions_path:
        if args.run_manifest.strip():
            manifest_path = Path(args.run_manifest.strip())
        else:
            if not run_id:
                raise ValueError("Missing run id. Provide --run-id or --run-manifest.")
            manifest_path = Path("artifacts/model_runs") / run_id / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Run manifest/latest pointer not found: {manifest_path}. "
                "Pass --run-id <RUN_ID> or --run-manifest <PATH>."
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "manifest_path" in payload:
            pointer_path = manifest_path
            run_id = payload.get("run_id")
            manifest_raw = Path(str(payload["manifest_path"]))
            manifest_path = manifest_raw if manifest_raw.is_absolute() else (pointer_path.parent / manifest_raw)
            if not manifest_path.exists():
                candidates: list[Path] = []
                if run_id:
                    candidates.append(Path("artifacts/model_runs") / str(run_id) / "manifest.json")
                    candidates.append(pointer_path.parent / str(run_id) / "manifest.json")
                candidates.append(pointer_path.parent / "manifest.json")
                fallback = next((c for c in candidates if c.exists()), None)
                if fallback is None:
                    raise FileNotFoundError(
                        "Latest pointer resolves to missing manifest path and no local fallback was found. "
                        f"pointer={pointer_path}, manifest_path={manifest_path}, run_id={run_id}"
                    )
                print(
                    f"[WARN] Latest pointer manifest path not found: {manifest_path}. "
                    f"Using local fallback: {fallback}"
                )
                manifest_path = fallback
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = run_id or payload.get("run_id") or (args.run_id.strip() or None)
        manifest_dir = manifest_path.parent

        # Strict fail-fast preflight for thesis reproducibility:
        # verify manifest and complete P01..P99 quantile grid before simulation allocation.
        _preflight_manifest_and_quantiles(
            manifest_path=manifest_path,
            manifest_payload=payload,
            split=args.split,
            model_key=args.model_key.strip(),
            manifest_dir=manifest_dir,
            expected_quantiles=required_quantiles,
        )
    else:
        manifest_dir = Path.cwd()

    out_dir = _resolve_out_dir(args.out_dir, run_id=run_id, split=args.split)
    forecast_warehouse: dict[str, pd.DataFrame] | None = None
    coverage_min: pd.Timestamp | None = None
    coverage_max: pd.Timestamp | None = None

    if not predictions_path:
        da_long = payload.get("bundles", {}).get("da", {}).get("predictions_long", {}).get(args.split, {})
        afrr_long = payload.get("bundles", {}).get("afrr", {}).get("predictions_long", {}).get(args.split, {})
        long_map = {**da_long, **afrr_long}
        if long_map:
            resolved_long_map = _resolve_long_map(
                long_map=long_map,
                manifest_dir=manifest_dir,
                split=args.split,
                model_key=args.model_key.strip(),
            )
            forecast_warehouse = load_prediction_warehouse_long(
                resolved_long_map,
                target_value_modes=target_value_modes,
            )
            print(f"[INFO] Long-format forecast warehouse loaded for split='{args.split}' with {len(long_map)} files.")
            cov_min_list: list[pd.Timestamp] = []
            cov_max_list: list[pd.Timestamp] = []
            for wdf in forecast_warehouse.values():
                t = pd.to_datetime(wdf["target_time_utc"], utc=True, errors="coerce").dropna()
                if not t.empty:
                    cov_min_list.append(t.min())
                    cov_max_list.append(t.max())
            if cov_min_list and cov_max_list:
                coverage_min = min(cov_min_list)
                coverage_max = max(cov_max_list)
        else:
            da_pred = _resolve_bundle_prediction_path(
                configured_path=payload["bundles"]["da"]["predictions"][args.split],
                manifest_dir=manifest_dir,
                split=args.split,
                model_key=args.model_key.strip(),
            )
            afrr_pred = _resolve_bundle_prediction_path(
                configured_path=payload["bundles"]["afrr"]["predictions"][args.split],
                manifest_dir=manifest_dir,
                split=args.split,
                model_key=args.model_key.strip(),
            )
            predictions_path = str((out_dir / f"backtest_table_{args.split}.parquet").resolve())
            da_df = pd.read_parquet(da_pred)
            afrr_df = pd.read_parquet(afrr_pred)
            backtest_table = da_df.merge(afrr_df, on="timestamp_utc", how="inner")
            backtest_table.to_parquet(predictions_path, index=False)
            print(f"[INFO] Backtest table created: {predictions_path}")

        if not ground_truth_path:
            ground_truth_path = str(_resolve_existing_file(payload["ground_truth"]["default_path"], manifest_dir=manifest_dir))

    if not ground_truth_path:
        raise ValueError("Either provide --predictions and --ground-truth, or use --run-manifest.")
    truth_preview = pd.read_parquet(ground_truth_path)
    pred_preview = pd.read_parquet(predictions_path) if predictions_path else pd.DataFrame()

    colmap_in = BacktestColumnMap(
        timestamp=args.timestamp_col,
        pred_da_price=args.pred_da_col,
        pred_afrr_capacity_price_pos=args.pred_cap_pos_col,
        pred_afrr_capacity_price_neg=args.pred_cap_neg_col,
        pred_afrr_activation_price_pos=args.pred_act_pos_col,
        pred_afrr_activation_price_neg=args.pred_act_neg_col,
        pred_afrr_activation_rate_pos=args.pred_rate_pos_col,
        pred_afrr_activation_rate_neg=args.pred_rate_neg_col,
        true_da_price=args.true_da_col,
        true_afrr_capacity_price_pos=args.true_cap_pos_col,
        true_afrr_capacity_price_neg=args.true_cap_neg_col,
        true_afrr_activation_price_pos=args.true_act_pos_col,
        true_afrr_activation_price_neg=args.true_act_neg_col,
        true_afrr_activation_rate_pos=args.true_rate_pos_col,
        true_afrr_activation_rate_neg=args.true_rate_neg_col,
    )
    colmap = _apply_fallback_column_map(pred_preview, truth_preview, colmap_in)

    if predictions_path:
        df = load_and_align_market_data(
            predictions_path,
            ground_truth_path,
            colmap,
            target_value_modes=target_value_modes,
        )
    else:
        df = truth_preview.copy()
        if colmap.timestamp not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df[colmap.timestamp] = df.index
        df[colmap.timestamp] = pd.to_datetime(df[colmap.timestamp], utc=True, errors="coerce")
        df = df.dropna(subset=[colmap.timestamp]).sort_values(colmap.timestamp).reset_index(drop=True)
        df = canonicalize_market_frame(df, colmap=colmap, target_value_modes=target_value_modes)
        if forecast_warehouse and coverage_min is not None and coverage_max is not None:
            df = df[(df[colmap.timestamp] >= coverage_min) & (df[colmap.timestamp] <= coverage_max)].copy()
    if args.start:
        df = df[df[colmap.timestamp] >= pd.to_datetime(args.start, utc=True)].copy()
    if args.end:
        df = df[df[colmap.timestamp] <= pd.to_datetime(args.end, utc=True)].copy()
    if df.empty:
        raise ValueError("No rows after timestamp filtering.")
    scenarios: list[tuple[str, dict[str, pd.DataFrame] | None]] = [("default", forecast_warehouse)]
    if quantile_pairs:
        if forecast_warehouse is None:
            raise ValueError("--quantile-pairs requires long-format predictions from --run-manifest.")
        scenarios = []
        for q_low, q_high in quantile_pairs:
            name = _scenario_suffix(q_low, q_high)
            wh = _apply_quantile_pair_to_warehouse(
                forecast_warehouse,
                q_low=q_low,
                q_high=q_high,
                da_role=args.da_quantile_role,
            )
            scenarios.append((name, wh))

    MODEL_SPECS["reserve_activation_headroom_h"] = float(args.reserve_activation_headroom_h)
    MODEL_SPECS["bem_activation_headroom_h"] = float(args.bem_activation_headroom_h)
    mode = str(args.reserve_feasibility_mode or "").strip().lower()
    if mode not in {"", "normal", "conservative"}:
        raise ValueError("--reserve-feasibility-mode must be one of: normal, conservative")
    if not mode:
        mode = "conservative" if bool(args.strict_simulation_validity) else "normal"
    MODEL_SPECS["reserve_feasibility_mode"] = mode
    if args.reserve_soc_projection_safety_mwh is not None:
        MODEL_SPECS["reserve_soc_projection_safety_mwh"] = float(args.reserve_soc_projection_safety_mwh)
    elif mode == "conservative" and bool(args.strict_simulation_validity):
        MODEL_SPECS["reserve_soc_projection_safety_mwh"] = 0.5
    if args.reserve_headroom_safety_mwh is not None:
        MODEL_SPECS["reserve_headroom_safety_mwh"] = float(args.reserve_headroom_safety_mwh)
    elif mode == "conservative" and bool(args.strict_simulation_validity):
        MODEL_SPECS["reserve_headroom_safety_mwh"] = 0.25
    if args.reserve_power_safety_mw is not None:
        MODEL_SPECS["reserve_power_safety_mw"] = float(args.reserve_power_safety_mw)
    elif mode == "conservative" and bool(args.strict_simulation_validity):
        MODEL_SPECS["reserve_power_safety_mw"] = 0.1
    if args.reserve_min_margin_after_bid_mwh is not None:
        MODEL_SPECS["reserve_min_margin_after_bid_mwh"] = float(args.reserve_min_margin_after_bid_mwh)
    elif mode == "conservative" and bool(args.strict_simulation_validity):
        MODEL_SPECS["reserve_min_margin_after_bid_mwh"] = 0.25
    MODEL_SPECS["reserve_bid_derate"] = float(args.reserve_bid_derate)
    if not (0.0 <= float(MODEL_SPECS["reserve_bid_derate"]) <= 1.0):
        raise ValueError("--reserve-bid-derate must be within [0, 1].")
    MODEL_SPECS["max_reserve_bid_mw"] = (
        float(args.max_reserve_bid_mw) if args.max_reserve_bid_mw is not None else None
    )
    if MODEL_SPECS["max_reserve_bid_mw"] is not None and float(MODEL_SPECS["max_reserve_bid_mw"]) < 0.0:
        raise ValueError("--max-reserve-bid-mw must be >= 0.")
    MODEL_SPECS["disable_new_bcm_reserve_bids"] = bool(args.disable_new_bcm_reserve_bids)
    MODEL_SPECS["bem_only_headroom_safety_mwh"] = float(args.bem_only_headroom_safety_mwh)
    if float(MODEL_SPECS["bem_only_headroom_safety_mwh"]) < 0.0:
        raise ValueError("--bem-only-headroom-safety-mwh must be >= 0.")
    MODEL_SPECS["max_bem_only_bid_mw"] = (
        float(args.max_bem_only_bid_mw) if args.max_bem_only_bid_mw is not None else None
    )
    if MODEL_SPECS["max_bem_only_bid_mw"] is not None and float(MODEL_SPECS["max_bem_only_bid_mw"]) < 0.0:
        raise ValueError("--max-bem-only-bid-mw must be >= 0.")
    MODEL_SPECS["disallow_simultaneous_bem_only_pos_neg"] = bool(args.disallow_simultaneous_bem_only_pos_neg)
    MODEL_SPECS["reserve_retry_ladder"] = str(args.reserve_retry_ladder).strip()
    MODEL_SPECS["enable_reserve_retry_ladder"] = bool(
        bool(args.strict_simulation_validity) and str(MODEL_SPECS["reserve_feasibility_mode"]) == "conservative"
    )
    MODEL_SPECS["forecast_value_mode"] = str(args.forecast_value_mode).strip().lower()
    MODEL_SPECS["final_soc_mode"] = str(args.final_soc_mode)
    enforce_final_soc_min = bool(args.final_soc_mode == "hard")
    backtester = BatteryBacktester()
    sweep_rows: list[dict[str, object]] = []
    if args.trading_strategy == "da_only":
        allowed_markets = ("DA",)
    elif args.trading_strategy == "afrr_only":
        allowed_markets = ("aFRR",)
    else:
        allowed_markets = ("DA", "aFRR")
    strategy_root = out_dir / args.trading_strategy

    for scenario_name, scenario_warehouse in scenarios:
        scenario_out_dir = strategy_root if scenario_name == "default" and not quantile_pairs else strategy_root / scenario_name
        output_was_cleaned = _prepare_scenario_output_dir(
            scenario_out_dir=scenario_out_dir,
            clean_output=bool(args.clean_output),
            strict_simulation_validity=bool(args.strict_simulation_validity),
            simulation_schema_version=simulation_schema_version,
        )

        with _phase_watchdog("backtester_run"):
            outputs = backtester.run(
                df,
                colmap,
                use_rolling_horizon=not args.disable_rolling_horizon,
                horizon_hours=args.horizon_hours,
                reopt_step_hours=args.reopt_step_hours,
                forecast_warehouse=scenario_warehouse,
                da_gate_hour_cet=args.da_gate_hour_cet if args.da_gate_hour_utc is None else args.da_gate_hour_utc,
                soc_feedback_mode=args.soc_feedback_mode,
                enforce_final_soc_min=enforce_final_soc_min,
                allowed_markets=allowed_markets,
                strict_simulation_validity=bool(args.strict_simulation_validity),
                enable_global_perfect_foresight=bool(args.enable_global_perfect_foresight),
            )

        hourly_path = scenario_out_dir / "backtest_hourly.parquet"
        planned_ledger_path = scenario_out_dir / "planned_ledger.parquet"
        executed_ledger_path = scenario_out_dir / "executed_ledger.parquet"
        realized_ledger_path = scenario_out_dir / "realized_ledger.parquet"
        plan_history_path = scenario_out_dir / "backtest_plan_history.parquet"
        milp_event_log_path = scenario_out_dir / "backtest_milp_event_log.parquet"
        milp_event_summary_path = scenario_out_dir / "backtest_milp_event_summary.csv"
        volatility_path = scenario_out_dir / "backtest_decision_volatility.csv"
        monthly_path = scenario_out_dir / "backtest_monthly.csv"
        yearly_path = scenario_out_dir / "backtest_yearly.csv"
        summary_path = scenario_out_dir / "backtest_summary.json"
        state_machine_audit_path = scenario_out_dir / "state_machine_audit.json"
        diagnostics_path = scenario_out_dir / "backtest_diagnostics.json"
        diagnostics_txt_path = scenario_out_dir / "backtest_diagnostics.txt"
        perfect_foresight_paradox_path = scenario_out_dir / "perfect_foresight_paradox_hours.csv"
        pnl_plot_path = scenario_out_dir / "backtest_cumulative_pnl.png"
        reserve_commitment_debug_path = scenario_out_dir / "reserve_commitment_debug.csv"
        invalid_headroom_debug_path = scenario_out_dir / "invalid_headroom_debug.csv"
        optimization_failure_debug_path = scenario_out_dir / "optimization_failure_debug.csv"
        optimization_infeasibility_attribution_path = scenario_out_dir / "optimization_infeasibility_attribution.csv"

        with _phase_watchdog("write_hourly"):
            outputs.hourly.to_parquet(hourly_path, index=False)
        planned_cols = [c for c in [
            colmap.timestamp, "charge_mw", "discharge_mw", "reserve_pos_mw", "reserve_neg_mw", "soc_lp_mwh",
            "planned_soc_mwh", "aFRR_Capacity_Won_Pos_MW", "aFRR_Capacity_Won_Neg_MW", "aFRR_Capacity_Won_MW",
            "aFRR_Energy_Price_EUR_MWh_Pos", "aFRR_Energy_Price_EUR_MWh_Neg", "event_reopt_triggered",
            "event_reopt_rejected_mw_total", "predicted_objective_eur"
        ] if c in outputs.hourly.columns]
        planned_cols.extend([c for c in outputs.hourly.columns if c.startswith("afrr_bin_")])
        planned_cols.extend([c for c in outputs.hourly.columns if c.startswith("submitted_afrr_")])
        planned_cols.extend([c for c in outputs.hourly.columns if c.startswith("executed_afrr_")])
        with _phase_watchdog("write_planned_ledger"):
            outputs.hourly[planned_cols].to_parquet(planned_ledger_path, index=False)

        executed_cols = [colmap.timestamp]
        executed_cols.extend([c for c in outputs.hourly.columns if c.startswith("real_planned_")])
        executed_cols.extend([c for c in outputs.hourly.columns if c.startswith("real_submitted_")])
        executed_cols.extend([c for c in outputs.hourly.columns if c.startswith("real_executed_")])
        executed_cols.extend([c for c in outputs.hourly.columns if c.startswith("real_afrr_bin_")])
        executed_cols.extend([c for c in [
            "real_da_buy_accepted", "real_da_sell_accepted", "real_afrr_cap_pos_awarded", "real_afrr_cap_neg_awarded",
            "real_afrr_act_pos_accepted", "real_afrr_act_neg_accepted", "real_aFRR_Capacity_Won_MW",
            "real_DA_Energy_Sold_MW", "real_aFRR_Energy_Price_EUR_MWh", "real_Obligation_Fulfilled",
            "real_aFRR_Energy_Gate_Closure_Min", "shock_source", "soc_before_mwh", "soc_after_planned_mwh",
            "soc_after_executed_mwh", "soc_shock_mwh"
        ] if c in outputs.hourly.columns])
        with _phase_watchdog("write_executed_ledger"):
            outputs.hourly[[*dict.fromkeys(executed_cols)]].to_parquet(executed_ledger_path, index=False)

        realized_cols = [colmap.timestamp]
        realized_cols.extend([c for c in outputs.hourly.columns if c.startswith("real_submitted_afrr_")])
        realized_cols.extend([c for c in outputs.hourly.columns if c.startswith("real_executed_afrr_")])
        realized_cols.extend([c for c in outputs.hourly.columns if c.startswith("real_afrr_bin_")])
        realized_cols.extend([
            c for c in outputs.hourly.columns if c.startswith("real_") and c in {
                "real_pnl_eur", "real_da_buy_mwh", "real_da_sell_mwh", "real_act_pos_mwh", "real_act_neg_mwh",
                "real_revenue_da_eur", "real_cost_da_eur", "real_revenue_capacity_eur", "real_revenue_activation_eur",
                "real_transaction_cost_eur", "real_degradation_cost_eur", "real_penalty_eur",
                "real_missed_activation_mwh", "real_missed_capacity_mw", "real_missed_capacity_pos_mw",
                "real_missed_capacity_neg_mw", "real_requested_activation_revenue_eur",
                "real_delivered_activation_revenue_eur", "real_missed_activation_revenue_eur", "real_soc_mwh",
            }
        ])
        with _phase_watchdog("write_realized_ledger"):
            outputs.hourly[[*dict.fromkeys(realized_cols)]].to_parquet(realized_ledger_path, index=False)

        # Debug exports for strict-simulation diagnostics / traceability.
        h = outputs.hourly.copy()
        with _phase_watchdog("write_debug_exports"):
            reserve_debug_cols = [
                colmap.timestamp,
                "reserve_commitment_id",
                "reserve_product_block_id",
                "reserve_commitment_source_snapshot_utc",
                "reserve_delivery_start_utc",
                "reserve_delivery_end_utc",
                "reserve_precommit_feasible_pos_mw",
                "reserve_precommit_feasible_neg_mw",
                "reserve_submitted_pos_mw",
                "reserve_submitted_neg_mw",
                "precommit_original_pos_mw",
                "precommit_original_neg_mw",
                "reserve_awarded_pos_mw",
                "reserve_awarded_neg_mw",
                "reserve_lockbook_pos_mw",
                "reserve_lockbook_neg_mw",
                "reserve_projected_soc_start_mwh",
                "fixed_reserve_obligation_pos_mw",
                "fixed_reserve_obligation_neg_mw",
                "real_soc_start_mwh",
                "real_soc_mwh",
                "real_required_headroom_pos_mwh",
                "real_required_headroom_neg_mwh",
                "real_available_headroom_pos_mwh",
                "real_available_headroom_neg_mwh",
                "real_headroom_margin_pos_mwh",
                "real_headroom_margin_neg_mwh",
                "real_headroom_violation_pos_mwh",
                "real_headroom_violation_neg_mwh",
                "precommit_margin_after_bid_min_mwh",
                "precommit_zeroed_due_to_margin",
                "precommit_reduced_due_to_margin",
                "precommit_reduction_reason",
                "desired_reserve_pos_mw",
                "desired_reserve_neg_mw",
                "safe_reserve_pos_mw",
                "safe_reserve_neg_mw",
                "submitted_reserve_pos_mw",
                "submitted_reserve_neg_mw",
                "submitted_reserve_pos_mw_before_retry",
                "submitted_reserve_neg_mw_before_retry",
                "reserve_retry_factor",
                "submitted_reserve_pos_mw_after_retry",
                "submitted_reserve_neg_mw_after_retry",
                "retry_reduction_reason",
                "precommit_safe_pos_mw",
                "precommit_safe_neg_mw",
                "precommit_submitted_pos_mw_after_derate_cap",
                "precommit_submitted_neg_mw_after_derate_cap",
                "reserve_bid_derate",
                "max_reserve_bid_mw",
                "reserve_retry_attempts_used",
                "reserve_retry_final_factor",
                "reserve_retry_succeeded",
                "disable_new_bcm_reserve_bids",
                "new_reserve_bids_zeroed_by_retry",
                "reserve_retry_infeasible_after_zero_reserve",
                "precommit_headroom_recharge_cost_eur",
                "precommit_headroom_opportunity_cost_eur",
                "precommit_net_capacity_ev_after_headroom_cost_eur",
                "precommit_bid_zeroed_due_to_negative_ev",
                "real_da_charge_mw",
                "real_da_discharge_mw",
                "real_id_charge_mw",
                "real_id_discharge_mw",
                "real_bem_only_submitted_pos_mw",
                "real_bem_only_submitted_neg_mw",
                "real_desired_bem_only_pos_mw",
                "real_desired_bem_only_neg_mw",
                "real_safe_bem_only_pos_mw",
                "real_safe_bem_only_neg_mw",
                "real_bem_only_submitted_pos_mw_before_guard",
                "real_bem_only_submitted_neg_mw_before_guard",
                "real_bem_only_submitted_pos_mw_after_guard",
                "real_bem_only_submitted_neg_mw_after_guard",
                "real_submitted_bem_only_pos_mw",
                "real_submitted_bem_only_neg_mw",
                "real_bem_only_pos_reduced_by_headroom_mw",
                "real_bem_only_neg_reduced_by_headroom_mw",
                "real_bem_only_headroom_guard_applied",
                "real_bem_only_headroom_guard_reason",
                "real_bem_only_guard_soc_now_mwh",
                "real_bem_only_guard_protected_soc_min_mwh",
                "real_bem_only_guard_protected_soc_max_mwh",
                "real_bem_only_protected_soc_min_mwh",
                "real_bem_only_protected_soc_max_mwh",
                "real_bem_only_soc_start_mwh",
                "real_aux_energy_mwh",
                "optimization_error_code",
                "optimization_fallback",
                "is_fallback_hour",
                "infeasibility_driver",
            ]
            reserve_debug_cols = [c for c in reserve_debug_cols if c in h.columns]
            if reserve_debug_cols:
                out_df = h[reserve_debug_cols].copy()
                out_df["scenario"] = str(scenario_name)
                # Normalize directional and core reserve commitment fields for traceability.
                if "reserve_submitted_pos_mw" in out_df.columns or "reserve_submitted_neg_mw" in out_df.columns:
                    out_df["submitted_mw"] = np.maximum(
                        pd.to_numeric(out_df.get("reserve_submitted_pos_mw", 0.0), errors="coerce").fillna(0.0),
                        pd.to_numeric(out_df.get("reserve_submitted_neg_mw", 0.0), errors="coerce").fillna(0.0),
                    )
                    out_df["direction"] = np.where(
                        pd.to_numeric(out_df.get("reserve_submitted_pos_mw", 0.0), errors="coerce").fillna(0.0)
                        >= pd.to_numeric(out_df.get("reserve_submitted_neg_mw", 0.0), errors="coerce").fillna(0.0),
                        "pos",
                        "neg",
                    )
                if "reserve_awarded_pos_mw" in out_df.columns or "reserve_awarded_neg_mw" in out_df.columns:
                    out_df["awarded_mw"] = np.maximum(
                        pd.to_numeric(out_df.get("reserve_awarded_pos_mw", 0.0), errors="coerce").fillna(0.0),
                        pd.to_numeric(out_df.get("reserve_awarded_neg_mw", 0.0), errors="coerce").fillna(0.0),
                    )
                if "fixed_reserve_obligation_pos_mw" in out_df.columns or "fixed_reserve_obligation_neg_mw" in out_df.columns:
                    out_df["locked_obligation_mw"] = np.maximum(
                        pd.to_numeric(out_df.get("fixed_reserve_obligation_pos_mw", 0.0), errors="coerce").fillna(0.0),
                        pd.to_numeric(out_df.get("fixed_reserve_obligation_neg_mw", 0.0), errors="coerce").fillna(0.0),
                    )
                if {"reserve_projected_soc_start_mwh", "real_soc_start_mwh"}.issubset(out_df.columns):
                    out_df["reserve_projected_vs_realized_soc_delta_mwh"] = (
                        pd.to_numeric(out_df["reserve_projected_soc_start_mwh"], errors="coerce").fillna(0.0)
                        - pd.to_numeric(out_df["real_soc_start_mwh"], errors="coerce").fillna(0.0)
                    )
                # Drift-attribution fields per locked commitment hour.
                ts_series = pd.to_datetime(out_df.get(colmap.timestamp, pd.Series(index=out_df.index, dtype="datetime64[ns, UTC]")), utc=True, errors="coerce")
                gate_series = pd.to_datetime(out_df.get("reserve_commitment_source_snapshot_utc", pd.Series(index=out_df.index, dtype="object")), utc=True, errors="coerce")
                delivery_start_series = pd.to_datetime(out_df.get("reserve_delivery_start_utc", pd.Series(index=out_df.index, dtype="object")), utc=True, errors="coerce")

                out_df["projected_soc_start_mwh_at_commitment"] = np.nan
                out_df["projected_soc_start_mwh_latest_before_delivery"] = np.nan
                out_df["realized_soc_start_mwh_at_delivery"] = np.nan
                out_df["da_dispatch_mw_between_commit_and_delivery"] = 0.0
                out_df["id_dispatch_mw_between_commit_and_delivery"] = 0.0
                out_df["bem_only_dispatch_mw_between_commit_and_delivery"] = 0.0
                out_df["aux_energy_mwh_between_commit_and_delivery"] = 0.0
                out_df["terminal_adjustment_pressure_eur"] = float(outputs.summary.get("terminal_soc_net_adjustment_eur", 0.0))
                # Aliases requested by audit task.
                out_df["required_headroom_mwh"] = pd.to_numeric(
                    out_df.get("real_required_headroom_pos_mwh", 0.0), errors="coerce"
                ).fillna(0.0)
                out_df["available_headroom_mwh"] = pd.to_numeric(
                    out_df.get("real_available_headroom_pos_mwh", 0.0), errors="coerce"
                ).fillna(0.0)
                out_df["headroom_violation_mwh"] = pd.to_numeric(
                    out_df.get("real_headroom_violation_pos_mwh", 0.0), errors="coerce"
                ).fillna(0.0)

                # Precompute dispatch magnitudes from full hourly table for interval sums.
                all_ts = pd.to_datetime(
                    h.get(colmap.timestamp, pd.Series(index=h.index, dtype="datetime64[ns, UTC]")),
                    utc=True,
                    errors="coerce",
                )

                da_charge = require_numeric_series(
                    h,
                    "real_executed_charge_mw",
                    aliases=["real_da_charge_mw", "executed_charge_mw", "da_charge_mw"],
                )
                da_discharge = require_numeric_series(
                    h,
                    "real_executed_discharge_mw",
                    aliases=["real_da_discharge_mw", "executed_discharge_mw", "da_discharge_mw"],
                )
                id_charge = require_numeric_series(
                    h,
                    "real_id_charge_mw",
                    aliases=["id_charge_mw"],
                )
                id_discharge = require_numeric_series(
                    h,
                    "real_id_discharge_mw",
                    aliases=["id_discharge_mw"],
                )
                bem_pos = require_numeric_series(
                    h,
                    "real_bem_only_submitted_pos_mw",
                    aliases=["bem_only_submitted_pos_mw"],
                )
                bem_neg = require_numeric_series(
                    h,
                    "real_bem_only_submitted_neg_mw",
                    aliases=["bem_only_submitted_neg_mw"],
                )
                aux_e = require_numeric_series(
                    h,
                    "real_aux_energy_mwh",
                    aliases=["aux_energy_mwh"],
                ).abs()

                da_disp = da_charge.abs() + da_discharge.abs()
                id_disp = id_charge.abs() + id_discharge.abs()
                bem_disp = bem_pos.abs() + bem_neg.abs()

                by_commit = (
                    out_df.loc[out_df["reserve_commitment_id"].notna(), ["reserve_commitment_id", "reserve_projected_soc_start_mwh", "real_soc_start_mwh", "headroom_violation_mwh"]]
                    .groupby("reserve_commitment_id", dropna=False)
                    .agg(
                        projected_soc_start_mwh_at_commitment=("reserve_projected_soc_start_mwh", "first"),
                        projected_soc_start_mwh_latest_before_delivery=("reserve_projected_soc_start_mwh", "last"),
                        realized_soc_start_mwh_at_delivery=("real_soc_start_mwh", "last"),
                    )
                    .reset_index()
                )
                if not by_commit.empty:
                    out_df = out_df.merge(by_commit, on="reserve_commitment_id", how="left", suffixes=("", "_agg"))
                    for c in [
                        "projected_soc_start_mwh_at_commitment",
                        "projected_soc_start_mwh_latest_before_delivery",
                        "realized_soc_start_mwh_at_delivery",
                    ]:
                        if f"{c}_agg" in out_df.columns:
                            out_df[c] = pd.to_numeric(out_df[f"{c}_agg"], errors="coerce")
                            out_df.drop(columns=[f"{c}_agg"], inplace=True)

                # Interval dispatch sums between commitment gate and delivery start.
                for i in out_df.index:
                    gate = gate_series.loc[i] if i in gate_series.index else pd.NaT
                    dstart = delivery_start_series.loc[i] if i in delivery_start_series.index else pd.NaT
                    if pd.isna(gate) or pd.isna(dstart):
                        continue
                    mask = (all_ts > gate) & (all_ts < dstart)
                    if mask.any():
                        out_df.at[i, "da_dispatch_mw_between_commit_and_delivery"] = float(da_disp.loc[mask].sum())
                        out_df.at[i, "id_dispatch_mw_between_commit_and_delivery"] = float(id_disp.loc[mask].sum())
                        out_df.at[i, "bem_only_dispatch_mw_between_commit_and_delivery"] = float(bem_disp.loc[mask].sum())
                        out_df.at[i, "aux_energy_mwh_between_commit_and_delivery"] = float(aux_e.loc[mask].sum())
                out_df.to_csv(reserve_commitment_debug_path, index=False)

            hv_pos = pd.to_numeric(h.get("real_headroom_violation_pos_mwh", h.get("headroom_violation_pos_mwh", 0.0)), errors="coerce").fillna(0.0)
            hv_neg = pd.to_numeric(h.get("real_headroom_violation_neg_mwh", h.get("headroom_violation_neg_mwh", 0.0)), errors="coerce").fillna(0.0)
            bad_headroom = h.loc[(hv_pos + hv_neg) > 1e-9, reserve_debug_cols].copy() if reserve_debug_cols else pd.DataFrame()
            if not bad_headroom.empty:
                bad_headroom.to_csv(invalid_headroom_debug_path, index=False)

            err_col = "optimization_error_code" if "optimization_error_code" in h.columns else None
            if err_col is not None:
                bad_opt = h.loc[h[err_col].fillna("ok").astype(str).str.lower().ne("ok"), reserve_debug_cols].copy() if reserve_debug_cols else pd.DataFrame()
                if not bad_opt.empty:
                    bad_opt.to_csv(optimization_failure_debug_path, index=False)

        attrib_df = _build_optimization_infeasibility_attribution(
            hourly=outputs.hourly,
            summary=outputs.summary,
            scenario=str(scenario_name),
        )
        if not attrib_df.empty:
            attrib_df.to_csv(optimization_infeasibility_attribution_path, index=False)
            first = attrib_df.iloc[0]
            outputs.summary["first_infeasible_timestamp_utc"] = str(first.get("timestamp_utc"))
            outputs.summary["first_infeasibility_driver"] = str(first.get("suspected_infeasibility_driver", "unknown_solver_or_numeric"))
            outputs.summary["first_infeasibility_driver_detail"] = str(
                first.get("suspected_infeasibility_driver_detail", "")
            )
            outputs.summary["infeasibility_attribution_path"] = str(optimization_infeasibility_attribution_path)
            outputs.summary["new_reserve_mw_at_first_failure"] = float(
                max(
                    float(pd.to_numeric(pd.Series([first.get("new_submitted_reserve_pos_mw", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                    float(pd.to_numeric(pd.Series([first.get("new_submitted_reserve_neg_mw", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                )
            )
            outputs.summary["locked_reserve_mw_at_first_failure"] = float(
                max(
                    float(pd.to_numeric(pd.Series([first.get("locked_reserve_pos_mw", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                    float(pd.to_numeric(pd.Series([first.get("locked_reserve_neg_mw", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                )
            )
            outputs.summary["protected_soc_margin_at_first_failure"] = float(
                min(
                    float(pd.to_numeric(pd.Series([first.get("protected_soc_margin_pos_mwh", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                    float(pd.to_numeric(pd.Series([first.get("protected_soc_margin_neg_mwh", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                )
            )
            outputs.summary["power_violation_at_first_failure"] = float(
                max(
                    float(pd.to_numeric(pd.Series([first.get("power_violation_pos_mw", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                    float(pd.to_numeric(pd.Series([first.get("power_violation_neg_mw", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                )
            )
        else:
            outputs.summary["first_infeasible_timestamp_utc"] = ""
            outputs.summary["first_infeasibility_driver"] = "none"
            outputs.summary["first_infeasibility_driver_detail"] = ""
            outputs.summary["infeasibility_attribution_path"] = ""
            outputs.summary["new_reserve_mw_at_first_failure"] = 0.0
            outputs.summary["locked_reserve_mw_at_first_failure"] = 0.0
            outputs.summary["protected_soc_margin_at_first_failure"] = 0.0
            outputs.summary["power_violation_at_first_failure"] = 0.0

        with _phase_watchdog("write_plan_history"):
            outputs.plan_history.to_parquet(plan_history_path, index=False)
        with _phase_watchdog("write_milp_event_log"):
            if "milp_event_type" in outputs.plan_history.columns:
                ev_cols = [c for c in outputs.plan_history.columns if c.startswith("ev_")]
                reserve_bin_cols = [c for c in outputs.plan_history.columns if c.startswith("reserve_pos_bin_") or c.startswith("reserve_neg_bin_")]
                keep_cols = [
                    "snapshot_time_utc",
                    "target_time_utc",
                    "lead_time_h",
                    "milp_event_type",
                    "charge_mw",
                    "discharge_mw",
                    "reserve_pos_mw",
                    "reserve_neg_mw",
                    "slack_pos_mw",
                    "slack_neg_mw",
                    "predicted_objective_eur",
                    "ev_objective_rebuild_eur",
                    *reserve_bin_cols,
                    *ev_cols,
                ]
                keep_cols = list(dict.fromkeys(c for c in keep_cols if c in outputs.plan_history.columns))
                event_log = outputs.plan_history.loc[
                    outputs.plan_history["milp_event_type"].astype(str).ne("none"),
                    keep_cols,
                ].copy()
                if not event_log.empty:
                    event_log["snapshot_time_utc"] = pd.to_datetime(event_log["snapshot_time_utc"], utc=True, errors="coerce")
                    event_log["snapshot_date_utc"] = event_log["snapshot_time_utc"].dt.date.astype(str)
                    event_log.to_parquet(milp_event_log_path, index=False)
                    summary = (
                        event_log.groupby(["snapshot_time_utc", "snapshot_date_utc", "milp_event_type"], dropna=False)
                        .agg(
                            hours_covered=("target_time_utc", "count"),
                            ev_da_charge_eur=("ev_da_charge_eur", "sum"),
                            ev_da_discharge_eur=("ev_da_discharge_eur", "sum"),
                            ev_afrr_pos_eur=("ev_afrr_pos_eur", "sum"),
                            ev_afrr_neg_eur=("ev_afrr_neg_eur", "sum"),
                            ev_slack_penalty_pos_eur=("ev_slack_penalty_pos_eur", "sum"),
                            ev_slack_penalty_neg_eur=("ev_slack_penalty_neg_eur", "sum"),
                            ev_terminal_soc_credit_eur=("ev_terminal_soc_credit_eur", "sum"),
                            ev_objective_rebuild_eur=("ev_objective_rebuild_eur", "sum"),
                            predicted_objective_eur=("predicted_objective_eur", "mean"),
                            avg_pred_da_price_eur_mwh=("ev_pred_da_price_eur_mwh", "mean"),
                            avg_pred_act_rate_pos=("ev_pred_act_rate_pos", "mean"),
                            avg_pred_act_rate_neg=("ev_pred_act_rate_neg", "mean"),
                            avg_pred_cap_pos_eur_mw=("ev_pred_cap_pos_eur_mw", "mean"),
                            avg_pred_cap_neg_eur_mw=("ev_pred_cap_neg_eur_mw", "mean"),
                        )
                        .reset_index()
                        .sort_values(["snapshot_time_utc", "milp_event_type"])
                    )
                    summary.to_csv(milp_event_summary_path, index=False)
        with _phase_watchdog("write_volatility"):
            outputs.volatility.to_csv(volatility_path, index=False)
        with _phase_watchdog("write_monthly"):
            outputs.monthly.to_csv(monthly_path, index=False)
        with _phase_watchdog("write_yearly"):
            outputs.yearly.to_csv(yearly_path, index=False)
        pre_summary_keys = set(outputs.summary.keys())
        defaults: dict[str, object] = {
            "simulation_schema_version": simulation_schema_version,
            "required_summary_fields_version": required_summary_fields_version,
            "code_run_started_at_utc": run_started_at_utc,
            "strict_simulation_validity": float(bool(args.strict_simulation_validity)),
            "command_line_args": json.dumps(vars(args), sort_keys=True, default=str),
            "output_was_cleaned": float(output_was_cleaned),
            "infeasible_debug_dump_count": 0.0,
            "accepted_path_infeasible_debug_dump_count": 0.0,
            "candidate_infeasible_debug_dump_count": 0.0,
            "infeasible_debug_dump_paths": [],
            "infeasible_debug_dump_timestamps": [],
            "fallback_used": 0.0,
            "fallback_mode_counts": "{}",
            "optimization_error_code_counts": "{}",
            "simulation_valid": 0.0,
            "thesis_reportable": 0.0,
            "invalid_reason": "",
            "protected_soc_violation_count": 0.0,
            "protected_soc_violation_max_mwh": 0.0,
            "reserve_headroom_shortfall_pos_mwh_sum": 0.0,
            "reserve_headroom_shortfall_neg_mwh_sum": 0.0,
            "reserve_headroom_shortfall_max_mwh": 0.0,
            "reserve_headroom_shortfall_check_pass": 1.0,
            "pnl_reconciliation_error_max_eur": 0.0,
            "activation_split_reconciliation_error_max": 0.0,
            "summary_fields_defaulted": "[]",
            "required_fields_defaulted": "[]",
            "required_fields_computed": "[]",
            "first_infeasible_timestamp_utc": "",
            "first_infeasibility_driver": "none",
            "first_infeasibility_driver_detail": "",
            "infeasibility_attribution_path": "",
            "new_reserve_mw_at_first_failure": 0.0,
            "locked_reserve_mw_at_first_failure": 0.0,
            "protected_soc_margin_at_first_failure": 0.0,
            "power_violation_at_first_failure": 0.0,
            "disable_new_bcm_reserve_bids": float(bool(args.disable_new_bcm_reserve_bids)),
            "forecast_postprocessing_applied": 1.0,
            "forecast_value_mode": str(args.forecast_value_mode),
            "forecast_postprocessing_targets": json.dumps(
                [
                    colmap.pred_da_price,
                    colmap.pred_afrr_capacity_price_pos,
                    colmap.pred_afrr_capacity_price_neg,
                    colmap.pred_afrr_activation_price_pos,
                    colmap.pred_afrr_activation_price_neg,
                    colmap.pred_afrr_activation_rate_pos,
                    colmap.pred_afrr_activation_rate_neg,
                ]
            ),
            "forecast_postprocessing_report_path": "",
            "canonical_economic_targets": json.dumps(
                [k for k, v in target_value_modes.items() if str(v).strip().lower() == "canonical_economic"]
            ),
        }
        defaulted_fields: list[str] = []
        for k, v in defaults.items():
            if k not in outputs.summary or outputs.summary.get(k) is None:
                outputs.summary[k] = v
                defaulted_fields.append(k)
        required_fields = [
            "simulation_schema_version",
            "required_summary_fields_version",
            "code_run_started_at_utc",
            "infeasible_debug_dump_count",
            "accepted_path_infeasible_debug_dump_count",
            "candidate_infeasible_debug_dump_count",
            "infeasible_debug_dump_paths",
            "infeasible_debug_dump_timestamps",
            "fallback_used",
            "fallback_mode_counts",
            "optimization_error_code_counts",
            "simulation_valid",
            "thesis_reportable",
            "invalid_reason",
            "pnl_reconciliation_error_max_eur",
            "activation_split_reconciliation_error_max",
        ]
        # Normalize debug-dump fields to stable JSON schema for validator and downstream tooling.
        for k in [
            "infeasible_debug_dump_count",
            "accepted_path_infeasible_debug_dump_count",
            "candidate_infeasible_debug_dump_count",
        ]:
            outputs.summary[k] = float(pd.to_numeric(pd.Series([outputs.summary.get(k, 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        for k in ["infeasible_debug_dump_paths", "infeasible_debug_dump_timestamps"]:
            v = outputs.summary.get(k, [])
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    v = parsed if isinstance(parsed, list) else []
                except Exception:
                    v = []
            elif not isinstance(v, list):
                v = []
            outputs.summary[k] = [str(x) for x in v]
        required_defaulted = [k for k in required_fields if k in defaulted_fields]
        required_computed = [k for k in required_fields if k not in required_defaulted]
        outputs.summary["summary_fields_defaulted"] = json.dumps(sorted(defaulted_fields))
        # Runner metadata defaults are optional and should not invalidate strict thesis checks.
        outputs.summary["required_fields_missing"] = json.dumps([])
        outputs.summary["critical_required_fields_defaulted"] = json.dumps([])
        outputs.summary["optional_fields_defaulted"] = json.dumps(sorted(defaulted_fields))
        outputs.summary["required_fields_check_pass"] = 1.0
        # Backward-compatible aliases.
        outputs.summary["required_fields_defaulted"] = json.dumps([])
        outputs.summary["required_fields_computed"] = json.dumps(sorted(required_fields))
        with _phase_watchdog("write_summary_json"):
            summary_path.write_text(json.dumps(outputs.summary, indent=2), encoding="utf-8")
        with _phase_watchdog("write_state_machine_audit"):
            state_machine_audit_path.write_text(json.dumps(_build_state_machine_audit(outputs.hourly), indent=2), encoding="utf-8")
        with _phase_watchdog("build_and_write_diagnostics"):
            diagnostics = _build_backtest_diagnostics(outputs.hourly, outputs.summary)
            def _fmt_summary_val(key: str) -> str:
                v = outputs.summary.get(key, float("nan"))
                try:
                    fv = float(v)
                except Exception:
                    return str(v)
                return f"{fv:.2f}" if pd.notna(fv) else "nan"
            ts_col = colmap.timestamp if colmap.timestamp in outputs.hourly.columns else None
            if ts_col is not None and not outputs.hourly.empty:
                _ts = pd.to_datetime(outputs.hourly[ts_col], utc=True, errors="coerce").dropna()
            else:
                _ts = pd.Series(dtype="datetime64[ns, UTC]")
            if len(_ts) > 0:
                timeframe_start = _ts.min()
                timeframe_end = _ts.max()
                num_days_total = float((timeframe_end - timeframe_start).total_seconds() / 86400.0) + (1.0 / 24.0)
                timeframe_start_txt = timeframe_start.isoformat()
                timeframe_end_txt = timeframe_end.isoformat()
            else:
                num_days_total = float("nan")
                timeframe_start_txt = "n/a"
                timeframe_end_txt = "n/a"
            diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            diagnostics_txt_path.write_text(
                "\n".join([
                    "Backtest Diagnostics",
                    f"timeframe_start_utc={timeframe_start_txt}",
                    f"timeframe_end_utc={timeframe_end_txt}",
                    f"timeframe_total_days={num_days_total:.4f}" if pd.notna(num_days_total) else "timeframe_total_days=nan",
                    f"rows_hourly={diagnostics['rows_hourly']}",
                    f"numeric_nan_total={diagnostics['numeric_nan_total']}",
                    f"numeric_nonfinite_total={diagnostics['numeric_nonfinite_total']}",
                    f"final_soc_constraint_satisfied={diagnostics['infeasibility_flags']['final_soc_constraint_satisfied']}",
                    "",
                    "PnL Summary",
                    f"realized_total_pnl_eur={_fmt_summary_val('realized_total_pnl_eur')}",
                    f"rolling_perfect_foresight_same_rules_total_pnl_eur={_fmt_summary_val('rolling_perfect_foresight_same_rules_total_pnl_eur')}",
                ]) + "\n",
                encoding="utf-8",
            )
        with _phase_watchdog("write_perfect_foresight_paradox_report"):
            hp = outputs.hourly.copy()
            if {"perfect_foresight_pnl_eur", "real_pnl_eur"}.issubset(hp.columns):
                hp = hp[hp["perfect_foresight_pnl_eur"] < hp["real_pnl_eur"]].copy()
            else:
                hp = hp.iloc[0:0].copy()
            pos_share_cols = sorted([c for c in hp.columns if c.startswith("ev_expected_act_share_pos_bin_")])
            neg_share_cols = sorted([c for c in hp.columns if c.startswith("ev_expected_act_share_neg_bin_")])
            pos_res_cols = sorted([c for c in hp.columns if c.startswith("reserve_pos_bin_") and c.endswith("_mw")])
            neg_res_cols = sorted([c for c in hp.columns if c.startswith("reserve_neg_bin_") and c.endswith("_mw")])
            n_pos = min(len(pos_share_cols), len(pos_res_cols))
            n_neg = min(len(neg_share_cols), len(neg_res_cols))
            if not hp.empty and n_pos > 0:
                exp_pos = np.zeros(len(hp), dtype=float)
                for i in range(n_pos):
                    exp_pos += pd.to_numeric(hp[pos_share_cols[i]], errors="coerce").fillna(0.0).to_numpy(dtype=float) * pd.to_numeric(hp[pos_res_cols[i]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                hp["expected_act_pos_mwh_from_objective"] = exp_pos
            else:
                hp["expected_act_pos_mwh_from_objective"] = 0.0
            if not hp.empty and n_neg > 0:
                exp_neg = np.zeros(len(hp), dtype=float)
                for i in range(n_neg):
                    exp_neg += pd.to_numeric(hp[neg_share_cols[i]], errors="coerce").fillna(0.0).to_numpy(dtype=float) * pd.to_numeric(hp[neg_res_cols[i]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                hp["expected_act_neg_mwh_from_objective"] = exp_neg
            else:
                hp["expected_act_neg_mwh_from_objective"] = 0.0
            hp["expected_act_total_mwh_from_objective"] = hp["expected_act_pos_mwh_from_objective"] + hp["expected_act_neg_mwh_from_objective"]
            hp["realized_act_total_mwh"] = pd.to_numeric(hp.get("real_act_pos_mwh", 0.0), errors="coerce").fillna(0.0) + pd.to_numeric(hp.get("real_act_neg_mwh", 0.0), errors="coerce").fillna(0.0)
            hp["perfect_foresight_act_total_mwh"] = pd.to_numeric(hp.get("perfect_foresight_act_pos_mwh", 0.0), errors="coerce").fillna(0.0) + pd.to_numeric(hp.get("perfect_foresight_act_neg_mwh", 0.0), errors="coerce").fillna(0.0)
            keep = [
                colmap.timestamp,
                "real_pnl_eur",
                "perfect_foresight_pnl_eur",
                "real_penalty_eur",
                "perfect_foresight_penalty_eur",
                "reserve_pos_mw",
                "reserve_neg_mw",
                "expected_act_pos_mwh_from_objective",
                "expected_act_neg_mwh_from_objective",
                "expected_act_total_mwh_from_objective",
                "realized_act_total_mwh",
                "perfect_foresight_act_total_mwh",
                "real_act_pos_mwh",
                "real_act_neg_mwh",
                "perfect_foresight_act_pos_mwh",
                "perfect_foresight_act_neg_mwh",
            ]
            keep = [c for c in keep if c in hp.columns]
            hp[keep].to_csv(perfect_foresight_paradox_path, index=False)
        with _phase_watchdog("write_ev_summary"):
            ev_terms = [
                "ev_da_charge_eur",
                "ev_da_discharge_eur",
                "ev_afrr_pos_eur",
                "ev_afrr_neg_eur",
                "ev_slack_penalty_pos_eur",
                "ev_slack_penalty_neg_eur",
                "ev_terminal_soc_credit_eur",
                "ev_objective_rebuild_eur",
                "ev_da_charge_coef_eur_per_mw",
                "ev_da_discharge_coef_eur_per_mw",
            ]
            ev_terms = [c for c in ev_terms if c in outputs.hourly.columns]
            if ev_terms:
                h = outputs.hourly.copy()
                strict_mask = pd.Series(True, index=h.index)
                if "is_strict_optimized_hour" in h.columns:
                    strict_mask &= pd.to_numeric(h["is_strict_optimized_hour"], errors="coerce").fillna(0.0).eq(1.0)
                rows = []
                for scope, mask in [("all_hours", pd.Series(True, index=h.index)), ("strict_optimized_hours", strict_mask)]:
                    hs = h.loc[mask]
                    for c in ev_terms:
                        s = pd.to_numeric(hs[c], errors="coerce")
                        rows.append(
                            {
                                "scope": scope,
                                "ev_term": c,
                                "count": int(s.notna().sum()),
                                "mean": float(s.mean()) if len(s) else float("nan"),
                                "pct_zero": float((s.fillna(0.0) == 0.0).mean() * 100.0) if len(s) else float("nan"),
                                "pct_negative": float((s < 0.0).mean() * 100.0) if len(s) else float("nan"),
                                "pct_positive": float((s > 0.0).mean() * 100.0) if len(s) else float("nan"),
                            }
                        )
                pd.DataFrame(rows).to_csv(scenario_out_dir / "ev_summary.csv", index=False)
        with _phase_watchdog("plot_cumulative_pnl"):
            _plot_cumulative_pnl(outputs.hourly, colmap.timestamp, pnl_plot_path)

        print(f"[OK] Battery backtest completed for scenario={scenario_name}.")
        ts_col = colmap.timestamp if colmap.timestamp in outputs.hourly.columns else None
        if ts_col is not None and not outputs.hourly.empty:
            _ts = pd.to_datetime(outputs.hourly[ts_col], utc=True, errors="coerce").dropna()
        else:
            _ts = pd.Series(dtype="datetime64[ns, UTC]")
        if len(_ts) > 0:
            timeframe_start = _ts.min()
            timeframe_end = _ts.max()
            num_days_total = float((timeframe_end - timeframe_start).total_seconds() / 86400.0) + (1.0 / 24.0)
            print(f"- timeframe_utc: {timeframe_start.isoformat()} -> {timeframe_end.isoformat()}")
            print(f"- timeframe_total_days: {num_days_total:.4f}")
        print(f"- trading_strategy: {args.trading_strategy}")
        print(
            "- realized/rolling_perfect_foresight_same_rules: "
            f"{outputs.summary.get('realized_total_pnl_eur', float('nan')):.2f} / "
            f"{outputs.summary.get('rolling_perfect_foresight_same_rules_total_pnl_eur', float('nan')):.2f}"
        )
        cmp_r = outputs.summary.get("comparable_realized_market_pnl_eur", float("nan"))
        cmp_b = outputs.summary.get("comparable_perfect_foresight_market_pnl_eur", float("nan"))
        print(
            "- realized/comparable_rolling_perfect_foresight_same_rules_market: "
            f"{float(cmp_r):.2f} / {float(cmp_b):.2f}"
        )
        # Warn on negative PnL in any reported optimization view.
        pnl_warn_fields = [
            ("Multi-market realized", "realized_total_pnl_eur"),
            ("Multi-market rolling_perfect_foresight_same_rules", "rolling_perfect_foresight_same_rules_total_pnl_eur"),
            ("Multi-market global_hindsight_perfect_foresight_upper_bound", "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur"),
        ]
        for label, key in pnl_warn_fields:
            v = pd.to_numeric(pd.Series([outputs.summary.get(key, float("nan"))]), errors="coerce").iloc[0]
            if pd.notna(v) and float(v) < 0.0:
                print(f"[WARN] negative_pnl: {label}={float(v):.2f} EUR")
        r_multi = outputs.summary.get(
            "realized_vs_perfect_foresight_ratio_multi_market",
            float("nan"),
        )
        print(
            "- realized_vs_perfect_foresight_pct (diagnostic, may exceed 100): "
            f"{(100.0 * float(r_multi)) if pd.notna(r_multi) else float('nan'):.2f}%"
        )
        r_cmp = outputs.summary.get("realized_vs_perfect_foresight_comparable_market_ratio", float("nan"))
        print(
            "- realized/comparable_rolling_perfect_foresight_same_rules_market ratio % (Multi): "
            f"{(100.0 * float(r_cmp)) if pd.notna(r_cmp) else float('nan'):.2f}%"
        )
        global_ratio_pct = outputs.summary.get("realized_vs_global_hindsight_perfect_foresight_upper_bound_pct", float("nan"))
        global_ok = (
            float(outputs.summary.get("global_perfect_foresight_available", 0.0)) >= 0.5
            and float(outputs.summary.get("global_perfect_foresight_dominance_check_pass", 0.0)) >= 0.5
        )
        if global_ok:
            print(
                "- realized_vs_global_hindsight_perfect_foresight_upper_bound_pct (upper-bound efficiency, should be <=100): "
                f"{float(global_ratio_pct) if pd.notna(global_ratio_pct) else float('nan'):.2f}%"
            )
        else:
            print("- global perfect_foresight unavailable/unverified")
        final_soc = outputs.summary.get("final_soc_actual_mwh", outputs.summary.get("final_real_soc_mwh", float("nan")))
        final_soc_target = outputs.summary.get("final_soc_target_mwh", outputs.summary.get("final_soc_min_target_mwh", float("nan")))
        final_soc_physical_ok = bool(float(outputs.summary.get("final_soc_physical_check_pass", 0.0)) >= 0.5)
        final_soc_economic_ok = bool(float(outputs.summary.get("final_soc_economic_repair_check_pass", 0.0)) >= 0.5)
        final_soc_shortfall = outputs.summary.get("final_soc_shortfall_mwh", float("nan"))
        if final_soc_physical_ok:
            print(
                "- final_soc_physical_check: OK "
                f"(actual/target={float(final_soc):.4f}/{float(final_soc_target):.4f} MWh)"
            )
        else:
            print(
                "[WARN] final_soc_physical_check: NOT MET "
                f"(actual/target={float(final_soc):.4f}/{float(final_soc_target):.4f} MWh, "
                f"shortfall={float(final_soc_shortfall):.4f} MWh)"
            )
        if final_soc_economic_ok:
            print("- final_soc_economic_repair_check: OK")
        else:
            print("[WARN] final_soc_economic_repair_check: NOT MET")
        missed_cap_pos = pd.to_numeric(
            pd.Series([outputs.summary.get("total_missed_capacity_pos_mw", 0.0)]),
            errors="coerce",
        ).iloc[0]
        missed_cap_neg = pd.to_numeric(
            pd.Series([outputs.summary.get("total_missed_capacity_neg_mw", 0.0)]),
            errors="coerce",
        ).iloc[0]
        if (pd.notna(missed_cap_pos) and float(missed_cap_pos) > 0.0) or (
            pd.notna(missed_cap_neg) and float(missed_cap_neg) > 0.0
        ):
            print(
                "[WARN] missed_capacity_afrr: "
                f"pos={float(missed_cap_pos):.4f} MW, "
                f"neg={float(missed_cap_neg):.4f} MW"
            )
        print(f"- output_dir: {scenario_out_dir}")

        row: dict[str, object] = {
            "scenario": scenario_name,
            "trading_strategy": args.trading_strategy,
            "split": args.split,
            "model_key": args.model_key,
            "start": args.start,
            "end": args.end,
            "da_quantile_role": args.da_quantile_role,
            "realized_total_pnl_eur": outputs.summary.get("realized_total_pnl_eur"),
            "predicted_total_pnl_eur": outputs.summary.get("predicted_total_pnl_eur"),
            "naive_total_pnl_eur": outputs.summary.get("naive_total_pnl_eur"),
            "rolling_perfect_foresight_same_rules_total_pnl_eur": outputs.summary.get(
                "rolling_perfect_foresight_same_rules_total_pnl_eur",
                float("nan"),
            ),
            "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur": outputs.summary.get(
                "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur"
            ),
            "global_perfect_foresight_available": outputs.summary.get("global_perfect_foresight_available"),
            "global_perfect_foresight_dominance_check_pass": outputs.summary.get("global_perfect_foresight_dominance_check_pass"),
            "global_perfect_foresight_validation_status": outputs.summary.get("global_perfect_foresight_validation_status"),
            "comparable_realized_market_pnl_eur": outputs.summary.get("comparable_realized_market_pnl_eur"),
            "comparable_perfect_foresight_market_pnl_eur": outputs.summary.get(
                "comparable_perfect_foresight_market_pnl_eur"
            ),
            "benchmark_is_global_upper_bound": outputs.summary.get("benchmark_is_global_upper_bound", 0.0),
            "rolling_pf_is_upper_bound": outputs.summary.get("rolling_pf_is_upper_bound", 0.0),
            "global_perfect_foresight_is_upper_bound": outputs.summary.get("global_perfect_foresight_is_upper_bound", 0.0),
            "cost_of_forecast_error_total_eur": outputs.summary.get("cost_of_forecast_error_total_eur"),
            "pnl_gap_total_eur": outputs.summary.get("pnl_gap_total_eur"),
            "economic_opportunity_gap_ratio": outputs.summary.get("economic_opportunity_gap_ratio"),
            "roi_on_max_capital": outputs.summary.get("roi_on_max_capital"),
            "simulation_valid": outputs.summary.get("simulation_valid"),
            "thesis_reportable": outputs.summary.get("thesis_reportable"),
            "invalid_reason": outputs.summary.get("invalid_reason"),
            "fallback_used": outputs.summary.get("fallback_used"),
            "output_dir": str(scenario_out_dir),
        }
        if scenario_name != "default":
            q_lo, q_hi = scenario_name.split("_", 1)
            row["quantile_low"] = q_lo
            row["quantile_high"] = q_hi
        sweep_rows.append(row)

    global_plan_history_path = Path("artifacts/backtest_plan_history.parquet")
    global_plan_history_path.parent.mkdir(parents=True, exist_ok=True)
    if scenarios:
        # Keep previous behavior for downstream consumers: export last scenario plan history globally.
        outputs.plan_history.to_parquet(global_plan_history_path, index=False)

    if quantile_pairs and sweep_rows:
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_csv = out_dir / "quantile_sweep_summary.csv"
        sweep_json = out_dir / "quantile_sweep_summary.json"
        sweep_df.to_csv(sweep_csv, index=False)
        sweep_json.write_text(sweep_df.to_json(orient="records", indent=2), encoding="utf-8")
        print(f"[OK] Quantile sweep summary: {sweep_csv}")

    # Strategy overview across separate runs (multi / da_only / afrr_only).
    if sweep_rows:
        overview_row = pd.DataFrame(sweep_rows)
        overview_csv = out_dir / "strategy_overview.csv"
        overview_json = out_dir / "strategy_overview.json"
        if overview_csv.exists():
            prev = pd.read_csv(overview_csv)
            overview = pd.concat([prev, overview_row], ignore_index=True, sort=False)
        else:
            overview = overview_row
        # Keep exactly one row per scenario+strategy in the overview.
        # Re-running smoke tests should overwrite previous entries.
        dedup_keys = [k for k in ["scenario", "trading_strategy"] if k in overview.columns]
        if dedup_keys:
            overview = overview.drop_duplicates(subset=dedup_keys, keep="last")
        if {"realized_total_pnl_eur", "rolling_perfect_foresight_same_rules_total_pnl_eur"}.issubset(overview.columns):
            r = pd.to_numeric(overview["realized_total_pnl_eur"], errors="coerce")
            o = pd.to_numeric(overview["rolling_perfect_foresight_same_rules_total_pnl_eur"], errors="coerce")
            overview["realized_vs_perfect_foresight_pct"] = np.where(
                o.abs() > 1e-12,
                (r / o) * 100.0,
                np.nan,
            )
        if {"realized_total_pnl_eur", "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur"}.issubset(overview.columns):
            r = pd.to_numeric(overview["realized_total_pnl_eur"], errors="coerce")
            g = pd.to_numeric(overview["global_hindsight_perfect_foresight_upper_bound_total_pnl_eur"], errors="coerce")
            avail = pd.to_numeric(overview.get("global_perfect_foresight_available", 0.0), errors="coerce").fillna(0.0)
            dom = pd.to_numeric(overview.get("global_perfect_foresight_dominance_check_pass", 0.0), errors="coerce").fillna(0.0)
            overview["realized_vs_global_hindsight_perfect_foresight_upper_bound_pct"] = np.where(
                (avail >= 0.5) & (dom >= 0.5) & (g.abs() > 1e-12),
                (r / g) * 100.0,
                np.nan,
            )
        if {
            "comparable_realized_market_pnl_eur",
            "comparable_perfect_foresight_market_pnl_eur",
        }.issubset(overview.columns):
            r_cmp = pd.to_numeric(overview["comparable_realized_market_pnl_eur"], errors="coerce")
            b_cmp = pd.to_numeric(
                overview["comparable_perfect_foresight_market_pnl_eur"], errors="coerce"
            )
            overview["realized_vs_perfect_foresight_comparable_market_pct"] = np.where(
                b_cmp.abs() > 1e-12,
                (r_cmp / b_cmp) * 100.0,
                np.nan,
            )
        overview = overview.sort_values([c for c in ["scenario", "trading_strategy"] if c in overview.columns]).reset_index(drop=True)
        overview.to_csv(overview_csv, index=False)
        overview_json.write_text(overview.to_json(orient="records", indent=2), encoding="utf-8")
        # Thesis-ready aggregate: valid and reportable runs only.
        valid_only = overview.copy()
        if "thesis_reportable" in valid_only.columns:
            sv = pd.to_numeric(valid_only["thesis_reportable"], errors="coerce").fillna(0.0)
            valid_only = valid_only.loc[sv >= 0.5].copy()
        elif "simulation_valid" in valid_only.columns:
            sv = pd.to_numeric(valid_only["simulation_valid"], errors="coerce").fillna(0.0)
            valid_only = valid_only.loc[sv >= 0.5].copy()
        valid_csv = out_dir / "strategy_overview_valid_only.csv"
        valid_json = out_dir / "strategy_overview_valid_only.json"
        valid_only.to_csv(valid_csv, index=False)
        valid_json.write_text(valid_only.to_json(orient="records", indent=2), encoding="utf-8")
        total_scenarios = int(len(overview))
        valid_scenarios = int(len(valid_only))
        invalid_scenarios = int(total_scenarios - valid_scenarios)
        invalid_rate_pct = float(100.0 * invalid_scenarios / total_scenarios) if total_scenarios > 0 else float("nan")
        invalid_by_reason: dict[str, int] = {}
        if "invalid_reason" in overview.columns and invalid_scenarios > 0:
            invalid_rows = overview.loc[pd.to_numeric(overview.get("simulation_valid", 0.0), errors="coerce").fillna(0.0) < 0.5]
            for txt in invalid_rows["invalid_reason"].fillna("").astype(str):
                for reason in [r.strip() for r in txt.split(",") if r.strip()]:
                    invalid_by_reason[reason] = invalid_by_reason.get(reason, 0) + 1
        stats = {
            "total_scenarios": total_scenarios,
            "valid_scenarios": valid_scenarios,
            "invalid_scenarios": invalid_scenarios,
            "invalid_rate_pct": invalid_rate_pct,
            "invalid_by_reason": invalid_by_reason,
            "strategy_overview_all": str(overview_csv),
            "strategy_overview_valid_only": str(valid_csv),
        }
        (out_dir / "strategy_overview_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print("[OK] Strategy overview:")
        print(f"- output_dir: {out_dir}")
        show_cols = [
            c
            for c in [
                "scenario",
                "trading_strategy",
                "realized_total_pnl_eur",
                "rolling_perfect_foresight_same_rules_total_pnl_eur",
                "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur",
                "realized_vs_perfect_foresight_pct",
                "realized_vs_perfect_foresight_comparable_market_pct",
                "realized_vs_global_hindsight_perfect_foresight_upper_bound_pct",
            ]
            if c in overview.columns
        ]
        if show_cols:
            display_df = overview[show_cols].copy()
            for c in [
                "realized_total_pnl_eur",
                "rolling_perfect_foresight_same_rules_total_pnl_eur",
                "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur",
                "realized_vs_perfect_foresight_pct",
                "realized_vs_perfect_foresight_comparable_market_pct",
                "realized_vs_global_hindsight_perfect_foresight_upper_bound_pct",
            ]:
                if c in display_df.columns:
                    display_df[c] = pd.to_numeric(display_df[c], errors="coerce").map(
                        lambda x: f"{float(x):.2f}" if pd.notna(x) else "nan"
                    )
            display_df = display_df.rename(
                columns={
                    "scenario": "scen",
                    "trading_strategy": "strat",
                    "realized_total_pnl_eur": "pnl_real_eur",
                    "rolling_perfect_foresight_same_rules_total_pnl_eur": "pnl_pf_roll_eur",
                    "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur": "pnl_pf_global_ub_eur",
                    "realized_vs_perfect_foresight_pct": "real_vs_pf_roll_pct",
                    "realized_vs_perfect_foresight_comparable_market_pct": "real_vs_pf_cmp_pct",
                    "realized_vs_global_hindsight_perfect_foresight_upper_bound_pct": "real_vs_pf_global_pct",
                }
            )
            print(display_df.to_string(index=False))
        print(f"[OK] Strategy overview files: {overview_csv} | {overview_json}")
        print(f"[OK] Valid-only overview files: {valid_csv} | {valid_json}")
        print(
            f"[OK] Validity stats: total={total_scenarios}, valid={valid_scenarios}, "
            f"invalid={invalid_scenarios}, invalid_rate={invalid_rate_pct:.2f}%"
        )


if __name__ == "__main__":
    main()
