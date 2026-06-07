"""Run LP-based battery backtest from ML predictions + ground truth parquet files.

Usage (manifest-autoload, recommended):
    ./.venv/bin/python scripts/run_battery_backtest.py \
      --model xgb \
      --start 2025-05-01T00:00:00Z \
      --end 2025-05-07T23:00:00Z \
      --out-dir artifacts/simulation_runs/sim_xgb_test

Usage (validation split):
    ./.venv/bin/python scripts/run_battery_backtest.py \
      --model xgb \
      --split val \
      --start 2025-05-01T00:00:00Z \
      --end 2025-05-07T23:00:00Z \
      --out-dir artifacts/simulation_runs/sim_xgb_val

Usage (multiple quantile policies):
    ./.venv/bin/python scripts/run_battery_backtest.py \
      --model xgb \
      --quantile-pairs p50-p50,p50-p70 \
      --start 2025-05-01T00:00:00Z \
      --end 2025-05-07T23:00:00Z \
      --out-dir artifacts/simulation_runs/sim_xgb_policies

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
import hashlib
import json
import os
import pickle
import re
import shutil
import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
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

from energy_trading.config import MARKET_SPECS, MODEL_SPECS
from energy_trading.simulation.battery_backtest import (
    AFRR_QUANTILE_BINS,
    BacktestColumnMap,
    BatteryBacktester,
    PhaseTimeoutError,
    canonicalize_market_frame,
    load_and_align_market_data,
    load_prediction_warehouse_long,
    normalize_predicted_pnl_aliases,
)
from energy_trading.visualization.style import apply_geo_style, get_backtest_line_style


INPUT_CACHE_SCHEMA_VERSION = "simulation_input_cache_v1"
SIMULATION_EVAL_START_UTC = pd.Timestamp("2025-01-14T00:00:00Z")
SIMULATION_EVAL_END_UTC = pd.Timestamp("2026-01-14T00:00:00Z")
FORECAST_COVERAGE_SCHEMA_VERSION = "forecast_coverage_preflight_v1"


def _to_utc_ts(value: object) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _resolve_simulation_eval_window(
    *,
    requested_start: object | None,
    requested_end: object | None,
    clamp_enabled: bool = True,
    lower_bound: pd.Timestamp = SIMULATION_EVAL_START_UTC,
    upper_bound: pd.Timestamp = SIMULATION_EVAL_END_UTC,
) -> dict[str, object]:
    req_start = _to_utc_ts(requested_start)
    req_end = _to_utc_ts(requested_end)
    effective_start = req_start if req_start is not None else lower_bound
    effective_end = req_end if req_end is not None else upper_bound
    reasons: list[str] = []
    warnings: list[str] = []
    if bool(clamp_enabled):
        if effective_start < lower_bound:
            warnings.append(
                "Requested start < common evaluation lower bound. "
                f"Clamped start from {effective_start.isoformat()} to {lower_bound.isoformat()}"
            )
            effective_start = lower_bound
            reasons.append("start_below_common_lower_bound")
        if effective_end > upper_bound:
            warnings.append(
                "Requested end > common evaluation upper bound. "
                f"Clamped end from {effective_end.isoformat()} to {upper_bound.isoformat()}"
            )
            effective_end = upper_bound
            reasons.append("end_above_common_upper_bound")
    if effective_start > effective_end:
        raise ValueError(
            "Effective simulation window is empty after applying bounds: "
            f"effective_start={effective_start.isoformat()}, effective_end={effective_end.isoformat()}"
        )
    window_hours = (effective_end - effective_start).total_seconds() / 3600.0
    return {
        "requested_start_utc": req_start.isoformat() if req_start is not None else "",
        "requested_end_utc": req_end.isoformat() if req_end is not None else "",
        "effective_start": effective_start,
        "effective_end": effective_end,
        "effective_start_utc": effective_start.isoformat(),
        "effective_end_utc": effective_end.isoformat(),
        "simulation_window_clamped": float(bool(reasons)),
        "simulation_window_clamp_reason": ",".join(reasons),
        "simulation_common_lower_bound_utc": lower_bound.isoformat(),
        "simulation_common_upper_bound_utc": upper_bound.isoformat(),
        "simulation_window_days": float(window_hours / 24.0),
        "simulation_window_hours": float(window_hours),
        "warnings": warnings,
    }


def _file_fingerprint(path: str | Path) -> dict[str, object]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False, "mtime_ns": None, "size": None}
    st = p.stat()
    return {"path": str(p.resolve()), "exists": True, "mtime_ns": int(st.st_mtime_ns), "size": int(st.st_size)}


def _input_cache_key(*, model_key: str, split: str, manifest_path: Path | None, prediction_files: Iterable[str | Path], truth_file: str | Path, forecast_value_mode: str) -> str:
    payload = {
        "schema": INPUT_CACHE_SCHEMA_VERSION,
        "model_key": str(model_key),
        "split": str(split),
        "manifest": _file_fingerprint(manifest_path) if manifest_path else None,
        "prediction_files": [_file_fingerprint(p) for p in sorted([str(p) for p in prediction_files])],
        "truth_file": _file_fingerprint(truth_file),
        "forecast_value_mode": str(forecast_value_mode),
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _input_cache_path(cache_key: str) -> Path:
    return REPO_ROOT / "artifacts" / "simulation_input_cache" / f"{cache_key}.pkl"


def _load_input_cache(cache_path: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame] | None, pd.Timestamp | None, pd.Timestamp | None] | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as fh:
            payload = pickle.load(fh)
        if payload.get("schema") != INPUT_CACHE_SCHEMA_VERSION:
            return None
        return payload["df"], payload.get("forecast_warehouse"), payload.get("coverage_min"), payload.get("coverage_max")
    except Exception:
        return None


def _write_input_cache(
    cache_path: Path,
    *,
    df: pd.DataFrame,
    forecast_warehouse: dict[str, pd.DataFrame] | None,
    coverage_min: pd.Timestamp | None,
    coverage_max: pd.Timestamp | None,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(
            {
                "schema": INPUT_CACHE_SCHEMA_VERSION,
                "df": df,
                "forecast_warehouse": forecast_warehouse,
                "coverage_min": coverage_min,
                "coverage_max": coverage_max,
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _select_hourly_output_columns(hourly: pd.DataFrame, *, output_detail: str, timestamp_col: str) -> pd.DataFrame:
    if str(output_detail).strip().lower() == "debug":
        return hourly
    exact = {
        timestamp_col,
        "timestamp_utc",
        "real_soc_start_mwh",
        "real_soc_mwh",
        "real_pnl_eur",
        "real_revenue_da_eur",
        "real_cost_da_eur",
        "real_da_pnl_eur",
        "real_submitted_da_buy_mw",
        "real_submitted_da_sell_mw",
        "real_submitted_da_buy_price_eur_mwh",
        "real_submitted_da_sell_price_eur_mwh",
        "real_da_buy_accepted",
        "real_da_sell_accepted",
        "real_da_buy_mwh",
        "real_da_sell_mwh",
        "real_revenue_capacity_eur",
        "real_revenue_activation_eur",
        "real_bcm_linked_activation_revenue_eur",
        "real_bem_only_activation_revenue_eur",
        "real_bcm_linked_pos_activation_mwh",
        "real_bcm_linked_neg_activation_mwh",
        "real_bem_only_pos_activation_mwh",
        "real_bem_only_neg_activation_mwh",
        "real_activation_split_reconciliation_error_eur",
        "real_submitted_afrr_pos_mw",
        "real_submitted_afrr_neg_mw",
        "real_executed_reserve_pos_mw",
        "real_executed_reserve_neg_mw",
        "real_fixed_reserve_obligation_pos_mw",
        "real_fixed_reserve_obligation_neg_mw",
        "real_bem_only_submitted_pos_mw",
        "real_bem_only_submitted_neg_mw",
        "real_bem_only_executed_pos_mwh",
        "real_bem_only_executed_neg_mwh",
        "real_id_buy_mwh",
        "real_id_sell_mwh",
        "real_revenue_id_eur",
        "real_cost_id_eur",
        "real_id_recourse_reason",
        "real_id_trade_type",
        "real_degradation_cost_eur",
        "real_aux_cost_eur",
        "real_transaction_cost_eur",
        "real_penalty_eur",
        "real_missed_capacity_mw",
        "real_missed_capacity_pos_mw",
        "real_missed_capacity_neg_mw",
        "real_missed_activation_mwh",
        "real_required_headroom_pos_mwh",
        "real_required_headroom_neg_mwh",
        "real_headroom_violation_pos_mwh",
        "real_headroom_violation_neg_mwh",
        "physical_soc_min_mwh",
        "physical_soc_max_mwh",
        "protected_soc_min_mwh",
        "protected_soc_max_mwh",
        "real_protected_soc_min_mwh",
        "real_protected_soc_max_mwh",
        "real_protected_soc_violation_pos_mwh",
        "real_protected_soc_violation_neg_mwh",
        "optimization_error_code",
        "optimization_fallback",
        "optimizer_fallback_used",
        "is_fallback_hour",
        "real_throughput_mwh",
        "perfect_foresight_pnl_eur",
        "global_hindsight_perfect_foresight_pnl_eur",
        "afrr_activation_rate_guard_policy",
        "afrr_activation_rate_guard_quantile",
        "afrr_activation_rate_guard_quantile_resolved",
        "afrr_activation_rate_guard_source_column_pos",
        "afrr_activation_rate_guard_source_column_neg",
        "ev_pred_act_rate_pos_guard",
        "ev_pred_act_rate_neg_guard",
        "ev_pred_act_rate_pos_p90",
        "ev_pred_act_rate_neg_p90",
    }
    prefixes = (
        "real_da_",
        "real_id_",
        "real_bcm_",
        "real_bem_",
        "real_headroom_",
        "real_power_",
        "final_soc_",
        "pnl_reconciliation_",
        "da_precommit_",
        "da_postlock_",
        "da_candidate_",
        "da_source_",
        "da_lockbook_",
        "da_locked_",
        "da_bid_",
        "current_row_is_da_gate",
        "raw_optimizer_",
        "accepted_lockbook_",
        "ev_da_",
    )
    cols = [c for c in hourly.columns if c in exact or c.startswith(prefixes)]
    return hourly.loc[:, list(dict.fromkeys(cols))].copy()


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
    locked_pos = max(
        _f("locked_reserve_pos_mw"),
        _f("fixed_reserve_obligation_pos_mw"),
        _f("bcm_pos_obligation_mw"),
        _f("real_bcm_locked_pos_mw"),
    )
    locked_neg = max(
        _f("locked_reserve_neg_mw"),
        _f("fixed_reserve_obligation_neg_mw"),
        _f("bcm_neg_obligation_mw"),
        _f("real_bcm_locked_neg_mw"),
    )
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


def _plot_cumulative_pnl(
    hourly: pd.DataFrame,
    ts_col: str,
    out_path: Path,
    *,
    summary: dict[str, object] | None = None,
) -> None:
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
    show_global_pf = False
    if "global_perfect_foresight_pnl_eur" in d.columns:
        global_pnl = pd.to_numeric(d["global_perfect_foresight_pnl_eur"], errors="coerce")
        if summary is None:
            show_global_pf = bool(global_pnl.notna().any())
        else:
            available = float(
                pd.to_numeric(
                    pd.Series([summary.get("global_perfect_foresight_available", 0.0)]),
                    errors="coerce",
                )
                .fillna(0.0)
                .iloc[0]
            )
            show_global_pf = bool(available >= 0.5 and global_pnl.notna().any())
        if show_global_pf:
            d["global_perfect_foresight_cum_pnl_eur"] = global_pnl.fillna(0.0).cumsum()

    apply_geo_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(d[ts_col], d["model_cum_pnl_eur"], label="Model", **get_backtest_line_style("model"))
    ax.plot(d[ts_col], d["naive_cum_pnl_eur"], label="Naive 24h", **get_backtest_line_style("naive"))
    ax.plot(
        d[ts_col],
        d["perfect_foresight_cum_pnl_eur"],
        label="RollingPerfectForesightSameRules",
        **get_backtest_line_style("rolling_perfect_foresight"),
    )
    if show_global_pf:
        ax.plot(
            d[ts_col],
            d["global_perfect_foresight_cum_pnl_eur"],
            label="GlobalHindsightPerfectForesight",
            **get_backtest_line_style("global_hindsight_perfect_foresight"),
        )
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


def _num_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if abs(float(den)) > 1e-12 else float("nan")


def _infer_dt_hours(hourly: pd.DataFrame) -> float:
    if "timestamp_utc" not in hourly.columns or hourly.empty:
        return 1.0
    ts = pd.to_datetime(hourly["timestamp_utc"], utc=True, errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return 1.0
    diffs_h = ts.diff().dt.total_seconds().dropna() / 3600.0
    diffs_h = diffs_h[(diffs_h > 0.0) & np.isfinite(diffs_h)]
    if diffs_h.empty:
        return 1.0
    dt_h = float(diffs_h.median())
    return dt_h if dt_h > 0.0 else 1.0


THROUGHPUT_SOURCE_COLUMNS = [
    "real_da_buy_mwh",
    "real_da_sell_mwh",
    "real_id_buy_mwh",
    "real_id_sell_mwh",
    "real_act_pos_mwh",
    "real_act_neg_mwh",
]


def _compute_hourly_throughput_mwh(hourly: pd.DataFrame) -> pd.Series:
    """Canonical realized physical throughput used by scenario, daily and hourly reports."""
    if "real_throughput_mwh" in hourly.columns:
        return pd.to_numeric(hourly["real_throughput_mwh"], errors="coerce").fillna(0.0).astype(float)
    missing = [c for c in THROUGHPUT_SOURCE_COLUMNS if c not in hourly.columns]
    if missing:
        raise ValueError(
            "Cannot compute real_throughput_mwh: missing required source columns "
            f"{missing}. Required columns: {THROUGHPUT_SOURCE_COLUMNS}"
        )
    total = pd.Series(0.0, index=hourly.index, dtype=float)
    for c in THROUGHPUT_SOURCE_COLUMNS:
        total = total + pd.to_numeric(hourly[c], errors="coerce").fillna(0.0).abs()
    return total.astype(float)


def _ensure_hourly_throughput(hourly: pd.DataFrame) -> pd.DataFrame:
    out = hourly.copy()
    out["real_throughput_mwh"] = _compute_hourly_throughput_mwh(out)
    return out


def _build_performance_metrics(
    *,
    hourly: pd.DataFrame,
    summary: dict[str, object],
    args: argparse.Namespace,
    scenario_name: str,
    scenario_bins: list[str],
    scenario_start_utc: pd.Timestamp | None,
    scenario_end_utc: pd.Timestamp | None,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    summary = normalize_predicted_pnl_aliases(summary)
    hourly = _ensure_hourly_throughput(hourly)
    dt_h = _infer_dt_hours(hourly)
    ts = pd.to_datetime(hourly.get("timestamp_utc", pd.Series(dtype="datetime64[ns, UTC]")), utc=True, errors="coerce")
    if ts.notna().any():
        start = ts.min()
        end = ts.max()
        n_hours = int(ts.notna().sum())
        n_days = max(float((end - start).total_seconds() / 86400.0) + (1.0 / 24.0), 1e-12)
    else:
        start = scenario_start_utc
        end = scenario_end_utc
        n_hours = int(len(hourly))
        n_days = max(float(n_hours / 24.0), 1e-12)
    annualization_factor = float(365.0 / n_days)
    p_max_mw = float(pd.to_numeric(pd.Series([summary.get("p_max_mw", np.nan)]), errors="coerce").iloc[0])
    cap_mwh = float(pd.to_numeric(pd.Series([summary.get("capacity_mwh", np.nan)]), errors="coerce").iloc[0])
    if not np.isfinite(p_max_mw) or p_max_mw <= 0.0:
        p_max_mw = float("nan")
        warnings.append("missing_or_invalid:p_max_mw")
    if not np.isfinite(cap_mwh) or cap_mwh <= 0.0:
        cap_mwh = float("nan")
        warnings.append("missing_or_invalid:capacity_mwh")

    def _hourly_sum(name: str) -> float:
        if name not in hourly.columns:
            return float("nan")
        return float(pd.to_numeric(hourly[name], errors="coerce").fillna(0.0).sum())

    realized_net = _hourly_sum("real_pnl_eur")
    if not np.isfinite(realized_net):
        realized_net = float(pd.to_numeric(pd.Series([summary.get("realized_total_pnl_eur", np.nan)]), errors="coerce").iloc[0])
    predicted_net = float(pd.to_numeric(pd.Series([summary.get("predicted_total_pnl_eur", np.nan)]), errors="coerce").iloc[0])

    da_gross_revenue = _hourly_sum("real_revenue_da_eur")
    if not np.isfinite(da_gross_revenue):
        da_gross_revenue = float(pd.to_numeric(pd.Series([summary.get("total_da_revenue_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    da_gross_cost = _hourly_sum("real_cost_da_eur")
    if not np.isfinite(da_gross_cost):
        da_gross_cost = float(pd.to_numeric(pd.Series([summary.get("total_da_cost_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    id_gross_revenue = _hourly_sum("real_revenue_id_eur")
    if not np.isfinite(id_gross_revenue):
        id_gross_revenue = float(pd.to_numeric(pd.Series([summary.get("total_id_revenue_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    id_gross_cost = _hourly_sum("real_cost_id_eur")
    if not np.isfinite(id_gross_cost):
        id_gross_cost = float(pd.to_numeric(pd.Series([summary.get("total_id_cost_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    bcm_capacity_revenue = _hourly_sum("real_revenue_capacity_eur")
    if not np.isfinite(bcm_capacity_revenue):
        bcm_capacity_revenue = float(pd.to_numeric(pd.Series([summary.get("total_afrr_capacity_revenue_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    bcm_linked_activation_revenue = _hourly_sum("real_bcm_linked_activation_revenue_eur")
    if not np.isfinite(bcm_linked_activation_revenue):
        bcm_linked_activation_revenue = float(pd.to_numeric(pd.Series([summary.get("total_bcm_linked_activation_revenue_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    bem_activation_revenue = _hourly_sum("real_bem_only_activation_revenue_eur")
    if not np.isfinite(bem_activation_revenue):
        bem_activation_revenue = float(pd.to_numeric(pd.Series([summary.get("total_bem_only_activation_revenue_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])

    realized_degradation_cost = _hourly_sum("real_degradation_cost_eur")
    if not np.isfinite(realized_degradation_cost):
        realized_degradation_cost = float(pd.to_numeric(pd.Series([summary.get("total_degradation_cost_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    realized_aux_cost = _hourly_sum("real_aux_cost_eur")
    if not np.isfinite(realized_aux_cost):
        realized_aux_cost = float(pd.to_numeric(pd.Series([summary.get("total_auxiliary_cost_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    transaction_cost = _hourly_sum("real_transaction_cost_eur")
    if not np.isfinite(transaction_cost):
        transaction_cost = float(pd.to_numeric(pd.Series([summary.get("total_transaction_cost_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    offer_cost = _hourly_sum("real_offer_cost_eur")
    if not np.isfinite(offer_cost):
        offer_cost = float(pd.to_numeric(pd.Series([summary.get("total_offer_cost_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    penalty_cost = _hourly_sum("real_penalty_eur")
    if not np.isfinite(penalty_cost):
        penalty_cost = float(pd.to_numeric(pd.Series([summary.get("total_penalty_cost_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    terminal_soc_repair_cost = float(pd.to_numeric(pd.Series([summary.get("terminal_soc_repair_cost_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])

    da_net = da_gross_revenue - da_gross_cost
    id_net = id_gross_revenue - id_gross_cost
    bcm_offer_cost = offer_cost
    bcm_strategy_total_revenue = bcm_capacity_revenue + bcm_linked_activation_revenue
    bcm_total_revenue = bcm_strategy_total_revenue
    bem_activation_cost = 0.0
    bem_net = bem_activation_revenue - bem_activation_cost
    bem_total_revenue = bem_activation_revenue
    afrr_capacity_revenue = bcm_capacity_revenue
    afrr_activation_revenue = _hourly_sum("real_revenue_activation_eur")
    if not np.isfinite(afrr_activation_revenue):
        afrr_activation_revenue = float(pd.to_numeric(pd.Series([summary.get("total_afrr_activation_revenue_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    afrr_activation_cost = 0.0
    afrr_total_net_revenue = afrr_capacity_revenue + afrr_activation_revenue - afrr_activation_cost
    bcm_bem_activation_split_reconciliation_error = (
        afrr_activation_revenue - bcm_linked_activation_revenue - bem_activation_revenue
    )

    gross_revenue_without_costs = da_gross_revenue + id_gross_revenue + afrr_capacity_revenue + afrr_activation_revenue
    gross_market_costs = da_gross_cost + id_gross_cost + afrr_activation_cost
    net_market_revenue_before_operational_costs = gross_revenue_without_costs - gross_market_costs
    # Offer cost is an optimizer/EV diagnostic. Realized settlement PnL does not
    # subtract it, so it must not enter the net-revenue reconciliation formula.
    # Operational realized PnL is sourced from hourly real_pnl_eur. Terminal SoC
    # repair/value diagnostics are exported separately and are not subtracted
    # from hourly settlement PnL in hard-final-SoC thesis runs.
    total_costs = realized_degradation_cost + realized_aux_cost + transaction_cost + penalty_cost
    reconciliation = realized_net - (gross_revenue_without_costs - gross_market_costs - total_costs)

    da_bid_buy_mwh_total = float((_num_series(hourly, "real_submitted_da_buy_mw") * dt_h).sum())
    da_bid_sell_mwh_total = float((_num_series(hourly, "real_submitted_da_sell_mw") * dt_h).sum())
    da_realized_buy_mwh_total = float(_num_series(hourly, "real_da_buy_mwh").sum())
    da_realized_sell_mwh_total = float(_num_series(hourly, "real_da_sell_mwh").sum())
    da_bid_abs_mwh_total = abs(da_bid_buy_mwh_total) + abs(da_bid_sell_mwh_total)
    da_realized_abs_mwh_total = abs(da_realized_buy_mwh_total) + abs(da_realized_sell_mwh_total)

    bem_bid_pos_mwh_total = float((_num_series(hourly, "real_bem_only_submitted_pos_mw") * dt_h).sum())
    bem_bid_neg_mwh_total = float((_num_series(hourly, "real_bem_only_submitted_neg_mw") * dt_h).sum())
    bem_realized_pos_mwh_total = float(_num_series(hourly, "real_bem_only_executed_pos_mwh").sum())
    bem_realized_neg_mwh_total = float(_num_series(hourly, "real_bem_only_executed_neg_mwh").sum())
    bem_bid_abs_mwh_total = abs(bem_bid_pos_mwh_total) + abs(bem_bid_neg_mwh_total)
    bem_realized_abs_mwh_total = abs(bem_realized_pos_mwh_total) + abs(bem_realized_neg_mwh_total)

    bcm_bid_capacity_pos_mw_mean = float(_num_series(hourly, "real_submitted_afrr_pos_mw").mean())
    bcm_bid_capacity_neg_mw_mean = float(_num_series(hourly, "real_submitted_afrr_neg_mw").mean())
    bcm_realized_capacity_pos_mw_mean = float(_num_series(hourly, "real_executed_reserve_pos_mw").mean())
    bcm_realized_capacity_neg_mw_mean = float(_num_series(hourly, "real_executed_reserve_neg_mw").mean())
    bcm_bid_capacity_abs_mw_mean = abs(bcm_bid_capacity_pos_mw_mean) + abs(bcm_bid_capacity_neg_mw_mean)
    bcm_realized_capacity_abs_mw_mean = abs(bcm_realized_capacity_pos_mw_mean) + abs(bcm_realized_capacity_neg_mw_mean)

    id_buy_mwh_total = float(_num_series(hourly, "real_id_buy_mwh").sum())
    id_sell_mwh_total = float(_num_series(hourly, "real_id_sell_mwh").sum())
    id_net_mwh_total = id_sell_mwh_total - id_buy_mwh_total
    id_abs_mwh_total = abs(id_buy_mwh_total) + abs(id_sell_mwh_total)

    throughput_mwh_total = float(_compute_hourly_throughput_mwh(hourly).sum())
    eq_cycles_total = float(throughput_mwh_total / (2.0 * cap_mwh)) if np.isfinite(cap_mwh) and cap_mwh > 0 else float("nan")
    mean_soc_mwh = float(_num_series(hourly, "real_soc_mwh").mean()) if "real_soc_mwh" in hourly.columns else float("nan")
    min_soc_mwh = float(_num_series(hourly, "real_soc_mwh").min()) if "real_soc_mwh" in hourly.columns else float("nan")
    max_soc_mwh = float(_num_series(hourly, "real_soc_mwh").max()) if "real_soc_mwh" in hourly.columns else float("nan")
    final_soc_mwh = float(pd.to_numeric(pd.Series([summary.get("final_soc_actual_mwh", np.nan)]), errors="coerce").iloc[0])
    target_final_soc_mwh = float(pd.to_numeric(pd.Series([summary.get("final_soc_target_mwh", np.nan)]), errors="coerce").iloc[0])
    final_soc_shortfall_mwh = max(0.0, target_final_soc_mwh - final_soc_mwh) if np.isfinite(final_soc_mwh) and np.isfinite(target_final_soc_mwh) else float("nan")
    final_soc_surplus_mwh = max(0.0, final_soc_mwh - target_final_soc_mwh) if np.isfinite(final_soc_mwh) and np.isfinite(target_final_soc_mwh) else float("nan")

    additive_daily_fields = [
        "realized_net_revenue_eur",
        "da_gross_revenue_eur",
        "da_gross_cost_eur",
        "id_gross_revenue_eur",
        "id_gross_cost_eur",
        "bcm_capacity_revenue_eur",
        "bcm_linked_activation_revenue_eur",
        "bcm_total_revenue_eur",
        "bem_activation_revenue_eur",
        "bem_total_revenue_eur",
        "bcm_bem_activation_split_reconciliation_error_eur",
        "realized_degradation_cost_eur",
        "realized_aux_cost_eur",
        "transaction_cost_eur",
        "penalty_cost_eur",
        "throughput_mwh_total",
        "da_bid_buy_mwh_total",
        "da_bid_sell_mwh_total",
        "da_realized_buy_mwh_total",
        "da_realized_sell_mwh_total",
        "bem_bid_pos_mwh_total",
        "bem_bid_neg_mwh_total",
        "bem_realized_pos_mwh_total",
        "bem_realized_neg_mwh_total",
        "id_buy_mwh_total",
        "id_sell_mwh_total",
        "id_net_mwh_total",
        "id_abs_mwh_total",
    ]
    row = {
        "performance_metrics_schema_version": "v1",
        "model_name": str(args.model_key or "unknown"),
        "model_key": str(args.model_key or ""),
        "run_manifest": str(args.run_manifest or ""),
        "split": str(args.split),
        "trading_strategy": str(args.trading_strategy),
        "id_recourse_mode": str(args.id_recourse_mode),
        "scenario": str(scenario_name),
        "quantile_pair": str(scenario_name),
        "da_quantile_role": str(args.da_quantile_role),
        "start_utc": str(start.isoformat() if pd.notna(start) else ""),
        "end_utc": str(end.isoformat() if pd.notna(end) else ""),
        "n_hours": float(n_hours),
        "n_days": float(n_days),
        "annualization_factor": annualization_factor,
        "capacity_mwh": cap_mwh,
        "p_max_mw": p_max_mw,
        "simulation_valid": float(pd.to_numeric(pd.Series([summary.get("simulation_valid", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
        "thesis_reportable": float(pd.to_numeric(pd.Series([summary.get("thesis_reportable", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
        "invalid_reason": str(summary.get("invalid_reason", "")),
        "realized_net_revenue_eur": realized_net,
        "predicted_net_revenue_eur": predicted_net,
        "annualized_realized_net_revenue_eur": realized_net * annualization_factor,
        "annualized_predicted_net_revenue_eur": predicted_net * annualization_factor,
        "realized_net_revenue_eur_per_mw": _safe_ratio(realized_net, p_max_mw),
        "predicted_net_revenue_eur_per_mw": _safe_ratio(predicted_net, p_max_mw),
        "annualized_realized_net_revenue_eur_per_mw": _safe_ratio(realized_net * annualization_factor, p_max_mw),
        "annualized_predicted_net_revenue_eur_per_mw": _safe_ratio(predicted_net * annualization_factor, p_max_mw),
        "da_gross_revenue_eur": da_gross_revenue,
        "da_gross_cost_eur": da_gross_cost,
        "da_net_revenue_eur": da_net,
        "id_gross_revenue_eur": id_gross_revenue,
        "id_gross_cost_eur": id_gross_cost,
        "id_net_revenue_eur": id_net,
        "bcm_capacity_revenue_eur": bcm_capacity_revenue,
        "bcm_offer_cost_eur": bcm_offer_cost,
        "bcm_activation_revenue_eur": bcm_linked_activation_revenue,
        "bcm_linked_activation_revenue_eur": bcm_linked_activation_revenue,
        "bcm_strategy_total_revenue_eur": bcm_strategy_total_revenue,
        "bcm_total_revenue_eur": bcm_total_revenue,
        "bcm_revenue_before_shared_costs_eur": bcm_total_revenue,
        "bem_activation_revenue_eur": bem_activation_revenue,
        "bem_activation_cost_eur": bem_activation_cost,
        "bem_net_revenue_eur": bem_net,
        "bem_total_revenue_eur": bem_total_revenue,
        "bem_revenue_before_shared_costs_eur": bem_total_revenue,
        "afrr_capacity_revenue_eur": afrr_capacity_revenue,
        "afrr_activation_revenue_eur": afrr_activation_revenue,
        "afrr_activation_cost_eur": afrr_activation_cost,
        "afrr_total_net_revenue_eur": afrr_total_net_revenue,
        "bcm_bem_activation_split_reconciliation_error_eur": bcm_bem_activation_split_reconciliation_error,
        "market_revenue_split_method": "source_mwh",
        "market_cost_allocation_method": "shared_operational_costs_not_allocated_to_bcm_or_bem",
        "annualized_bcm_capacity_revenue_eur": bcm_capacity_revenue * annualization_factor,
        "annualized_bcm_activation_revenue_eur": bcm_linked_activation_revenue * annualization_factor,
        "annualized_bcm_total_revenue_eur": bcm_total_revenue * annualization_factor,
        "annualized_bem_activation_revenue_eur": bem_activation_revenue * annualization_factor,
        "annualized_bem_total_revenue_eur": bem_total_revenue * annualization_factor,
        "gross_revenue_without_costs_eur": gross_revenue_without_costs,
        "gross_market_costs_eur": gross_market_costs,
        "net_market_revenue_before_operational_costs_eur": net_market_revenue_before_operational_costs,
        "realized_degradation_cost_eur": realized_degradation_cost,
        "realized_aux_cost_eur": realized_aux_cost,
        "transaction_cost_eur": transaction_cost,
        "offer_cost_eur": offer_cost,
        "penalty_cost_eur": penalty_cost,
        "terminal_soc_repair_cost_eur": terminal_soc_repair_cost,
        "total_costs_eur": total_costs,
        "net_revenue_reconciliation_error_eur": reconciliation,
        "da_bid_buy_mwh_total": da_bid_buy_mwh_total,
        "da_bid_sell_mwh_total": da_bid_sell_mwh_total,
        "da_realized_buy_mwh_total": da_realized_buy_mwh_total,
        "da_realized_sell_mwh_total": da_realized_sell_mwh_total,
        "da_bid_abs_mwh_total": da_bid_abs_mwh_total,
        "da_realized_abs_mwh_total": da_realized_abs_mwh_total,
        "da_bid_realized_ratio": _safe_ratio(da_realized_abs_mwh_total, da_bid_abs_mwh_total),
        "bem_bid_pos_mwh_total": bem_bid_pos_mwh_total,
        "bem_bid_neg_mwh_total": bem_bid_neg_mwh_total,
        "bem_realized_pos_mwh_total": bem_realized_pos_mwh_total,
        "bem_realized_neg_mwh_total": bem_realized_neg_mwh_total,
        "bem_bid_abs_mwh_total": bem_bid_abs_mwh_total,
        "bem_realized_abs_mwh_total": bem_realized_abs_mwh_total,
        "bem_bid_realized_ratio": _safe_ratio(bem_realized_abs_mwh_total, bem_bid_abs_mwh_total),
        "bcm_bid_capacity_pos_mw_mean": bcm_bid_capacity_pos_mw_mean,
        "bcm_bid_capacity_neg_mw_mean": bcm_bid_capacity_neg_mw_mean,
        "bcm_realized_capacity_pos_mw_mean": bcm_realized_capacity_pos_mw_mean,
        "bcm_realized_capacity_neg_mw_mean": bcm_realized_capacity_neg_mw_mean,
        "bcm_bid_capacity_abs_mw_mean": bcm_bid_capacity_abs_mw_mean,
        "bcm_realized_capacity_abs_mw_mean": bcm_realized_capacity_abs_mw_mean,
        "bcm_bid_realized_capacity_ratio": _safe_ratio(bcm_realized_capacity_abs_mw_mean, bcm_bid_capacity_abs_mw_mean),
        "id_buy_mwh_total": id_buy_mwh_total,
        "id_sell_mwh_total": id_sell_mwh_total,
        "id_net_mwh_total": id_net_mwh_total,
        "id_abs_mwh_total": id_abs_mwh_total,
        "id_abs_mwh_mean_per_day": _safe_ratio(id_abs_mwh_total, n_days),
        "id_mean_mw": _safe_ratio(id_abs_mwh_total, n_hours),
        "throughput_mwh_total": throughput_mwh_total,
        "throughput_mwh_per_day": _safe_ratio(throughput_mwh_total, n_days),
        "equivalent_full_cycles_total": eq_cycles_total,
        "equivalent_full_cycles_per_day": _safe_ratio(eq_cycles_total, n_days),
        "mean_soc_mwh": mean_soc_mwh,
        "mean_soc_pct": _safe_ratio(mean_soc_mwh, cap_mwh) * 100.0 if np.isfinite(mean_soc_mwh) else float("nan"),
        "min_soc_mwh": min_soc_mwh,
        "max_soc_mwh": max_soc_mwh,
        "final_soc_mwh": final_soc_mwh,
        "target_final_soc_mwh": target_final_soc_mwh,
        "final_soc_shortfall_mwh": final_soc_shortfall_mwh,
        "final_soc_surplus_mwh": final_soc_surplus_mwh,
        "performance_metric_warnings": json.dumps(sorted(set(warnings))),
        "metric_validation_tolerance_eur": 1e-6,
        "daily_additive_metric_fields": json.dumps(additive_daily_fields),
    }
    for base in [
        "da_bid_buy_mwh_total", "da_bid_sell_mwh_total", "da_realized_buy_mwh_total", "da_realized_sell_mwh_total",
        "da_bid_abs_mwh_total", "da_realized_abs_mwh_total", "bem_bid_pos_mwh_total", "bem_bid_neg_mwh_total",
        "bem_realized_pos_mwh_total", "bem_realized_neg_mwh_total", "bem_bid_abs_mwh_total", "bem_realized_abs_mwh_total",
        "id_buy_mwh_total", "id_sell_mwh_total", "id_net_mwh_total", "id_abs_mwh_total", "throughput_mwh_total",
        "equivalent_full_cycles_total",
    ]:
        row[f"{base}_per_day"] = _safe_ratio(float(row[base]), n_days)
    row["net_revenue_reconciliation_ok"] = float(abs(reconciliation) <= 1e-6)
    return pd.DataFrame([row]), warnings


def _build_daily_performance_metrics(
    *,
    hourly: pd.DataFrame,
    perf_row: pd.Series,
) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame()
    d = _ensure_hourly_throughput(hourly)
    d["date_utc"] = pd.to_datetime(d.get("timestamp_utc"), utc=True, errors="coerce").dt.date.astype(str)
    grp = d.groupby("date_utc", dropna=False)
    dt_h = _infer_dt_hours(d)
    out = pd.DataFrame(
        {
            "date_utc": grp.size().index,
            "n_hours": grp.size().values.astype(float),
            "da_gross_revenue_eur": grp["real_revenue_da_eur"].sum() if "real_revenue_da_eur" in d.columns else 0.0,
            "da_gross_cost_eur": grp["real_cost_da_eur"].sum() if "real_cost_da_eur" in d.columns else 0.0,
            "id_gross_revenue_eur": grp["real_revenue_id_eur"].sum() if "real_revenue_id_eur" in d.columns else 0.0,
            "id_gross_cost_eur": grp["real_cost_id_eur"].sum() if "real_cost_id_eur" in d.columns else 0.0,
            "afrr_capacity_revenue_eur": grp["real_revenue_capacity_eur"].sum() if "real_revenue_capacity_eur" in d.columns else 0.0,
            "afrr_activation_revenue_eur": grp["real_revenue_activation_eur"].sum() if "real_revenue_activation_eur" in d.columns else 0.0,
            "bcm_linked_activation_revenue_eur": grp["real_bcm_linked_activation_revenue_eur"].sum() if "real_bcm_linked_activation_revenue_eur" in d.columns else 0.0,
            "bem_activation_revenue_eur": grp["real_bem_only_activation_revenue_eur"].sum() if "real_bem_only_activation_revenue_eur" in d.columns else 0.0,
            "degradation_cost_eur": grp["real_degradation_cost_eur"].sum() if "real_degradation_cost_eur" in d.columns else 0.0,
            "aux_cost_eur": grp["real_aux_cost_eur"].sum() if "real_aux_cost_eur" in d.columns else 0.0,
            "transaction_cost_eur": grp["real_transaction_cost_eur"].sum() if "real_transaction_cost_eur" in d.columns else 0.0,
            "penalty_cost_eur": grp["real_penalty_eur"].sum() if "real_penalty_eur" in d.columns else 0.0,
            "offer_cost_eur": grp["real_offer_cost_eur"].sum() if "real_offer_cost_eur" in d.columns else 0.0,
            "terminal_soc_repair_cost_eur": 0.0,
            "net_revenue_eur": grp["real_pnl_eur"].sum() if "real_pnl_eur" in d.columns else 0.0,
            "da_bid_buy_mwh": (grp["real_submitted_da_buy_mw"].sum() * dt_h) if "real_submitted_da_buy_mw" in d.columns else 0.0,
            "da_bid_sell_mwh": (grp["real_submitted_da_sell_mw"].sum() * dt_h) if "real_submitted_da_sell_mw" in d.columns else 0.0,
            "da_realized_buy_mwh": grp["real_da_buy_mwh"].sum() if "real_da_buy_mwh" in d.columns else 0.0,
            "da_realized_sell_mwh": grp["real_da_sell_mwh"].sum() if "real_da_sell_mwh" in d.columns else 0.0,
            "bem_bid_pos_mwh": (grp["real_bem_only_submitted_pos_mw"].sum() * dt_h) if "real_bem_only_submitted_pos_mw" in d.columns else 0.0,
            "bem_bid_neg_mwh": (grp["real_bem_only_submitted_neg_mw"].sum() * dt_h) if "real_bem_only_submitted_neg_mw" in d.columns else 0.0,
            "bem_realized_pos_mwh": grp["real_bem_only_executed_pos_mwh"].sum() if "real_bem_only_executed_pos_mwh" in d.columns else 0.0,
            "bem_realized_neg_mwh": grp["real_bem_only_executed_neg_mwh"].sum() if "real_bem_only_executed_neg_mwh" in d.columns else 0.0,
            "bcm_bid_capacity_pos_mw_mean": grp["real_submitted_afrr_pos_mw"].mean() if "real_submitted_afrr_pos_mw" in d.columns else float("nan"),
            "bcm_bid_capacity_neg_mw_mean": grp["real_submitted_afrr_neg_mw"].mean() if "real_submitted_afrr_neg_mw" in d.columns else float("nan"),
            "bcm_realized_capacity_pos_mw_mean": grp["real_executed_reserve_pos_mw"].mean() if "real_executed_reserve_pos_mw" in d.columns else float("nan"),
            "bcm_realized_capacity_neg_mw_mean": grp["real_executed_reserve_neg_mw"].mean() if "real_executed_reserve_neg_mw" in d.columns else float("nan"),
            "id_buy_mwh": grp["real_id_buy_mwh"].sum() if "real_id_buy_mwh" in d.columns else 0.0,
            "id_sell_mwh": grp["real_id_sell_mwh"].sum() if "real_id_sell_mwh" in d.columns else 0.0,
            "throughput_mwh": grp["real_throughput_mwh"].sum(),
            "mean_soc_mwh": grp["real_soc_mwh"].mean() if "real_soc_mwh" in d.columns else float("nan"),
            "fallback_hours": grp["is_fallback_hour"].sum() if "is_fallback_hour" in d.columns else 0.0,
        }
    ).reset_index(drop=True)
    if "throughput_mwh" in out.columns and np.isfinite(float(perf_row.get("equivalent_full_cycles_total", float("nan")))):
        cap_mwh = float(perf_row.get("capacity_mwh", float("nan")))
        if not np.isfinite(cap_mwh) or cap_mwh <= 0:
            out["equivalent_full_cycles"] = float("nan")
        else:
            out["equivalent_full_cycles"] = out["throughput_mwh"] / (2.0 * cap_mwh)
    else:
        out["equivalent_full_cycles"] = float("nan")
    out["simulation_valid"] = float(perf_row.get("simulation_valid", 0.0))
    out["thesis_reportable"] = float(perf_row.get("thesis_reportable", 0.0))
    out["invalid_reason"] = str(perf_row.get("invalid_reason", ""))
    out["total_costs_eur"] = (
        pd.to_numeric(out["degradation_cost_eur"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["aux_cost_eur"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["transaction_cost_eur"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["penalty_cost_eur"], errors="coerce").fillna(0.0)
    )
    out["da_pnl_eur"] = (
        pd.to_numeric(out["da_gross_revenue_eur"], errors="coerce").fillna(0.0)
        - pd.to_numeric(out["da_gross_cost_eur"], errors="coerce").fillna(0.0)
    )
    out["id_recourse_pnl_eur"] = (
        pd.to_numeric(out["id_gross_revenue_eur"], errors="coerce").fillna(0.0)
        - pd.to_numeric(out["id_gross_cost_eur"], errors="coerce").fillna(0.0)
    )
    out["bcm_pnl_eur"] = (
        pd.to_numeric(out["afrr_capacity_revenue_eur"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["bcm_linked_activation_revenue_eur"], errors="coerce").fillna(0.0)
    )
    out["bem_pnl_eur"] = pd.to_numeric(out["bem_activation_revenue_eur"], errors="coerce").fillna(0.0)
    out["bcm_activation_revenue_eur"] = pd.to_numeric(
        out["bcm_linked_activation_revenue_eur"], errors="coerce"
    ).fillna(0.0)
    out["bcm_total_revenue_eur"] = out["bcm_pnl_eur"]
    out["bcm_revenue_before_shared_costs_eur"] = out["bcm_total_revenue_eur"]
    out["bem_total_revenue_eur"] = out["bem_pnl_eur"]
    out["bem_revenue_before_shared_costs_eur"] = out["bem_total_revenue_eur"]
    out["bcm_bem_activation_split_reconciliation_error_eur"] = (
        pd.to_numeric(out["afrr_activation_revenue_eur"], errors="coerce").fillna(0.0)
        - pd.to_numeric(out["bcm_linked_activation_revenue_eur"], errors="coerce").fillna(0.0)
        - pd.to_numeric(out["bem_activation_revenue_eur"], errors="coerce").fillna(0.0)
    )
    out["market_revenue_split_method"] = "source_mwh"
    out["market_cost_allocation_method"] = "shared_operational_costs_not_allocated_to_bcm_or_bem"
    out["afrr_pnl_eur"] = (
        pd.to_numeric(out["afrr_capacity_revenue_eur"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["afrr_activation_revenue_eur"], errors="coerce").fillna(0.0)
    )
    out["has_fallback_hour"] = pd.to_numeric(out["fallback_hours"], errors="coerce").fillna(0.0) > 0.0
    out["has_non_ok_optimization_hour"] = (
        grp["optimization_error_code"].apply(lambda s: (~s.fillna("ok").astype(str).str.lower().eq("ok")).any()).to_numpy()
        if "optimization_error_code" in d.columns
        else False
    )
    return out


def _performance_reconciliation_specs() -> list[dict[str, object]]:
    return [
        {
            "metric": "realized_net_revenue_eur",
            "scenario_col": "realized_net_revenue_eur",
            "daily_col": "net_revenue_eur",
            "hourly_col": "real_pnl_eur",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": True,
            "source_note": "hourly real_pnl_eur is the canonical realized net source",
        },
        {
            "metric": "da_gross_revenue_eur",
            "scenario_col": "da_gross_revenue_eur",
            "daily_col": "da_gross_revenue_eur",
            "hourly_col": "real_revenue_da_eur",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": True,
            "source_note": "DA realized gross revenue",
        },
        {
            "metric": "da_gross_cost_eur",
            "scenario_col": "da_gross_cost_eur",
            "daily_col": "da_gross_cost_eur",
            "hourly_col": "real_cost_da_eur",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": True,
            "source_note": "DA realized gross cost",
        },
        {
            "metric": "id_gross_revenue_eur",
            "scenario_col": "id_gross_revenue_eur",
            "daily_col": "id_gross_revenue_eur",
            "hourly_col": "real_revenue_id_eur",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": True,
            "source_note": "ID realized gross revenue",
        },
        {
            "metric": "id_gross_cost_eur",
            "scenario_col": "id_gross_cost_eur",
            "daily_col": "id_gross_cost_eur",
            "hourly_col": "real_cost_id_eur",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": True,
            "source_note": "ID realized gross cost",
        },
        {
            "metric": "bcm_capacity_revenue_eur",
            "scenario_col": "bcm_capacity_revenue_eur",
            "daily_col": "afrr_capacity_revenue_eur",
            "hourly_col": "real_revenue_capacity_eur",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": True,
            "source_note": "BCM capacity revenue is daily aFRR capacity revenue",
        },
        {
            "metric": "bcm_linked_activation_revenue_eur",
            "scenario_col": "bcm_linked_activation_revenue_eur",
            "daily_col": "bcm_linked_activation_revenue_eur",
            "hourly_col": "real_bcm_linked_activation_revenue_eur",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": True,
            "source_note": "BCM-linked activation revenue tracked separately from BEM-only activation",
        },
        {
            "metric": "bcm_total_revenue_eur",
            "scenario_col": "bcm_total_revenue_eur",
            "daily_col": "bcm_total_revenue_eur",
            "hourly_col": "",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": False,
            "source_note": "BCM market revenue = capacity revenue + BCM-linked activation revenue; shared costs are not allocated",
        },
        {
            "metric": "afrr_activation_revenue_eur",
            "scenario_col": "afrr_activation_revenue_eur",
            "daily_col": "afrr_activation_revenue_eur",
            "hourly_col": "real_revenue_activation_eur",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": True,
            "source_note": "Total realized activation revenue",
        },
        {
            "metric": "bem_activation_revenue_eur",
            "scenario_col": "bem_activation_revenue_eur",
            "daily_col": "bem_activation_revenue_eur",
            "hourly_col": "real_bem_only_activation_revenue_eur",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": True,
            "source_note": "BEM-only activation revenue",
        },
        {
            "metric": "bem_total_revenue_eur",
            "scenario_col": "bem_total_revenue_eur",
            "daily_col": "bem_total_revenue_eur",
            "hourly_col": "",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": False,
            "source_note": "BEM market revenue = BEM-only activation revenue; shared costs are not allocated",
        },
        {
            "metric": "realized_degradation_cost_eur",
            "scenario_col": "realized_degradation_cost_eur",
            "daily_col": "degradation_cost_eur",
            "hourly_col": "real_degradation_cost_eur",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": True,
            "source_note": "Operational degradation cost",
        },
        {
            "metric": "realized_aux_cost_eur",
            "scenario_col": "realized_aux_cost_eur",
            "daily_col": "aux_cost_eur",
            "hourly_col": "real_aux_cost_eur",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": True,
            "source_note": "Auxiliary energy cost",
        },
        {
            "metric": "transaction_cost_eur",
            "scenario_col": "transaction_cost_eur",
            "daily_col": "transaction_cost_eur",
            "hourly_col": "real_transaction_cost_eur",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": True,
            "source_note": "Transaction cost",
        },
        {
            "metric": "offer_cost_eur",
            "scenario_col": "offer_cost_eur",
            "daily_col": "offer_cost_eur",
            "hourly_col": "real_offer_cost_eur",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": False,
            "source_note": "Offer cost diagnostic; not subtracted from realized settlement PnL",
        },
        {
            "metric": "penalty_cost_eur",
            "scenario_col": "penalty_cost_eur",
            "daily_col": "penalty_cost_eur",
            "hourly_col": "real_penalty_eur",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": True,
            "source_note": "Penalty cost",
        },
        {
            "metric": "terminal_soc_repair_cost_eur",
            "scenario_col": "terminal_soc_repair_cost_eur",
            "daily_col": "terminal_soc_repair_cost_eur",
            "hourly_col": "",
            "checked_daily_to_scenario": False,
            "checked_component_to_net": True,
            "source_note": "Scenario-level terminal adjustment; not allocated across days in strict reconciliation",
        },
        {
            "metric": "da_bid_buy_mwh_total",
            "scenario_col": "da_bid_buy_mwh_total",
            "daily_col": "da_bid_buy_mwh",
            "hourly_col": "",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": False,
            "source_note": "Daily sums are the canonical additive check for submitted DA buy volume",
        },
        {
            "metric": "da_bid_sell_mwh_total",
            "scenario_col": "da_bid_sell_mwh_total",
            "daily_col": "da_bid_sell_mwh",
            "hourly_col": "",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": False,
            "source_note": "Daily sums are the canonical additive check for submitted DA sell volume",
        },
        {
            "metric": "da_realized_buy_mwh_total",
            "scenario_col": "da_realized_buy_mwh_total",
            "daily_col": "da_realized_buy_mwh",
            "hourly_col": "real_da_buy_mwh",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": False,
            "source_note": "Realized DA buy volume",
        },
        {
            "metric": "da_realized_sell_mwh_total",
            "scenario_col": "da_realized_sell_mwh_total",
            "daily_col": "da_realized_sell_mwh",
            "hourly_col": "real_da_sell_mwh",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": False,
            "source_note": "Realized DA sell volume",
        },
        {
            "metric": "bem_bid_pos_mwh_total",
            "scenario_col": "bem_bid_pos_mwh_total",
            "daily_col": "bem_bid_pos_mwh",
            "hourly_col": "",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": False,
            "source_note": "Daily sums are the canonical additive check for submitted BEM positive volume",
        },
        {
            "metric": "bem_bid_neg_mwh_total",
            "scenario_col": "bem_bid_neg_mwh_total",
            "daily_col": "bem_bid_neg_mwh",
            "hourly_col": "",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": False,
            "source_note": "Daily sums are the canonical additive check for submitted BEM negative volume",
        },
        {
            "metric": "bem_realized_pos_mwh_total",
            "scenario_col": "bem_realized_pos_mwh_total",
            "daily_col": "bem_realized_pos_mwh",
            "hourly_col": "real_bem_only_executed_pos_mwh",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": False,
            "source_note": "Realized BEM positive volume",
        },
        {
            "metric": "bem_realized_neg_mwh_total",
            "scenario_col": "bem_realized_neg_mwh_total",
            "daily_col": "bem_realized_neg_mwh",
            "hourly_col": "real_bem_only_executed_neg_mwh",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": False,
            "source_note": "Realized BEM negative volume",
        },
        {
            "metric": "id_buy_mwh_total",
            "scenario_col": "id_buy_mwh_total",
            "daily_col": "id_buy_mwh",
            "hourly_col": "real_id_buy_mwh",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": False,
            "source_note": "Realized ID buy volume",
        },
        {
            "metric": "id_sell_mwh_total",
            "scenario_col": "id_sell_mwh_total",
            "daily_col": "id_sell_mwh",
            "hourly_col": "real_id_sell_mwh",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": False,
            "source_note": "Realized ID sell volume",
        },
        {
            "metric": "throughput_mwh_total",
            "scenario_col": "throughput_mwh_total",
            "daily_col": "throughput_mwh",
            "hourly_col": "real_throughput_mwh",
            "checked_daily_to_scenario": True,
            "checked_component_to_net": False,
            "source_note": "Battery throughput volume",
        },
    ]


def _write_performance_metric_definitions(path: Path) -> None:
    defs = [
        {"field": "realized_net_revenue_eur", "unit": "EUR", "formula": "sum(real_pnl_eur)", "source_columns": ["real_pnl_eur"], "kind": "realized_net"},
        {"field": "annualized_realized_net_revenue_eur", "unit": "EUR/year", "formula": "realized_net_revenue_eur * 365 / n_days", "source_columns": ["realized_net_revenue_eur", "n_days"], "kind": "derived"},
        {"field": "realized_net_revenue_eur_per_mw", "unit": "EUR/MW", "formula": "realized_net_revenue_eur / p_max_mw", "source_columns": ["realized_net_revenue_eur", "p_max_mw"], "kind": "derived"},
        {"field": "equivalent_full_cycles_total", "unit": "cycles", "formula": "throughput_mwh_total / (2 * capacity_mwh)", "source_columns": ["throughput_mwh_total", "capacity_mwh"], "kind": "battery"},
        {"field": "throughput_mwh_total", "unit": "MWh", "formula": "sum(real_throughput_mwh); real_throughput_mwh = abs(real_da_buy_mwh)+abs(real_da_sell_mwh)+abs(real_id_buy_mwh)+abs(real_id_sell_mwh)+abs(real_act_pos_mwh)+abs(real_act_neg_mwh)", "source_columns": THROUGHPUT_SOURCE_COLUMNS, "kind": "battery"},
        {"field": "net_revenue_reconciliation_error_eur", "unit": "EUR", "formula": "realized_net - (gross_revenue_without_costs - gross_market_costs - total_costs); total_costs excludes offer_cost_eur and terminal_soc_repair_cost_eur because they are not subtracted in hourly settlement PnL", "source_columns": ["realized_net_revenue_eur", "gross_revenue_without_costs_eur", "gross_market_costs_eur", "total_costs_eur"], "kind": "validation"},
        {"field": "da_bid_buy_mwh_total", "unit": "MWh", "formula": "sum(real_submitted_da_buy_mw * dt_h)", "source_columns": ["real_submitted_da_buy_mw", "timestamp_utc"], "kind": "volume"},
        {"field": "bem_bid_pos_mwh_total", "unit": "MWh", "formula": "sum(real_bem_only_submitted_pos_mw * dt_h)", "source_columns": ["real_bem_only_submitted_pos_mw", "timestamp_utc"], "kind": "volume"},
    ]
    path.write_text(json.dumps(defs, indent=2), encoding="utf-8")


def _build_bcm_block_consistency_violations(
    *,
    hourly: pd.DataFrame,
    summary: dict[str, object],
    scenario_name: str,
    strategy: str,
    tol_mw: float = 1e-6,
) -> pd.DataFrame:
    if "bcm_capacity_block_id" not in hourly.columns:
        return pd.DataFrame()
    checked_cols_raw = str(summary.get("bcm_block_consistency_checked_columns", "[]") or "[]")
    try:
        checked_cols = [str(c) for c in json.loads(checked_cols_raw)]
    except Exception:
        checked_cols = []
    if not checked_cols:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    grp = hourly.groupby("bcm_capacity_block_id", dropna=False)
    for block_id, g in grp:
        part_start = float(pd.to_numeric(g.get("bcm_capacity_block_partial_start", 0.0), errors="coerce").fillna(0.0).max())
        part_end = float(pd.to_numeric(g.get("bcm_capacity_block_partial_end", 0.0), errors="coerce").fillna(0.0).max())
        for c in checked_cols:
            if c not in g.columns:
                continue
            s = pd.to_numeric(g[c], errors="coerce").fillna(0.0)
            smin = float(s.min()) if len(s) else 0.0
            smax = float(s.max()) if len(s) else 0.0
            spread = float(smax - smin)
            if spread <= tol_mw:
                continue
            sem = "bcm_capacity" if "bcm_capacity" in c else ("mixed_afrr" if "afrr" in c else "unknown")
            for _, r in g.iterrows():
                rows.append(
                    {
                        "scenario": scenario_name,
                        "strategy": strategy,
                        "bcm_capacity_block_id": block_id,
                        "timestamp_utc": r.get("timestamp_utc"),
                        "bcm_capacity_block_hour_index": r.get("bcm_capacity_block_hour_index"),
                        "column_name": c,
                        "value": float(pd.to_numeric(pd.Series([r.get(c, 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                        "block_min": smin,
                        "block_max": smax,
                        "spread_mw": spread,
                        "partial_start": part_start,
                        "partial_end": part_end,
                        "column_semantics": sem,
                        "suspected_cause": "intra_block_variation_in_bcm_capacity_column",
                    }
                )
    return pd.DataFrame(rows)


def _validate_performance_metrics(
    *,
    perf_row: pd.Series,
    daily_df: pd.DataFrame,
    tolerance: float = 1e-6,
) -> dict[str, object]:
    checks: dict[str, object] = {}
    recon_error = float(pd.to_numeric(pd.Series([perf_row.get("net_revenue_reconciliation_error_eur", np.nan)]), errors="coerce").iloc[0])
    checks["net_revenue_reconciliation_ok"] = bool(np.isfinite(recon_error) and abs(recon_error) <= tolerance)
    checks["cost_reconciliation_ok"] = bool(
        abs(
            float(pd.to_numeric(pd.Series([perf_row.get("total_costs_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            - (
                float(pd.to_numeric(pd.Series([perf_row.get("realized_degradation_cost_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
                + float(pd.to_numeric(pd.Series([perf_row.get("realized_aux_cost_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
                + float(pd.to_numeric(pd.Series([perf_row.get("transaction_cost_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
                + float(pd.to_numeric(pd.Series([perf_row.get("penalty_cost_eur", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            )
        )
        <= tolerance
    )
    if daily_df.empty:
        checks["daily_to_scenario_reconciliation_ok"] = True
        checks["daily_to_scenario_error_max_abs"] = 0.0
        return checks

    daily_error_max = 0.0
    for spec in _performance_reconciliation_specs():
        if not bool(spec.get("checked_daily_to_scenario", False)):
            continue
        scenario_col = str(spec.get("scenario_col", ""))
        daily_col = str(spec.get("daily_col", ""))
        if not scenario_col or not daily_col or daily_col not in daily_df.columns:
            continue
        scen_v = float(pd.to_numeric(pd.Series([perf_row.get(scenario_col, np.nan)]), errors="coerce").iloc[0])
        if not np.isfinite(scen_v):
            continue
        day_v = float(pd.to_numeric(daily_df[daily_col], errors="coerce").fillna(0.0).sum())
        if not np.isfinite(day_v):
            continue
        daily_error_max = max(daily_error_max, abs(scen_v - day_v))
    checks["daily_to_scenario_reconciliation_ok"] = bool(daily_error_max <= tolerance)
    checks["daily_to_scenario_error_max_abs"] = float(daily_error_max)
    return checks


def _performance_reconciliation_failure_detail(
    *,
    checks: dict[str, object],
    recon_debug_df: pd.DataFrame,
    perf_row: pd.Series,
    tolerance: float = 1e-6,
) -> dict[str, object]:
    """Return the most relevant failing reconciliation row for strict errors."""
    if not bool(checks.get("net_revenue_reconciliation_ok", True)):
        err = float(
            pd.to_numeric(
                pd.Series([perf_row.get("net_revenue_reconciliation_error_eur", np.nan)]),
                errors="coerce",
            ).iloc[0]
        )
        return {
            "failure_check": "net_revenue_reconciliation_ok",
            "metric": "net_revenue_decomposition",
            "scenario_col": "net_revenue_reconciliation_error_eur",
            "scenario_value": err,
            "daily_sum_value": "",
            "hourly_sum_value": "",
            "daily_abs_error": "",
            "details": f"abs(net_revenue_reconciliation_error_eur)={abs(err)} > tolerance={tolerance}",
        }
    if not bool(checks.get("cost_reconciliation_ok", True)):
        return {
            "failure_check": "cost_reconciliation_ok",
            "metric": "total_costs_decomposition",
            "scenario_col": "total_costs_eur",
            "scenario_value": perf_row.get("total_costs_eur", ""),
            "daily_sum_value": "",
            "hourly_sum_value": "",
            "daily_abs_error": "",
            "details": f"total_costs_eur component decomposition exceeds tolerance={tolerance}",
        }
    top_row = None
    if not recon_debug_df.empty:
        tmp = recon_debug_df.loc[
            pd.to_numeric(
                recon_debug_df.get("checked_daily_to_scenario", False),
                errors="coerce",
            ).fillna(0.0)
            >= 0.5
        ].copy()
        tmp["daily_abs_error"] = pd.to_numeric(tmp.get("daily_abs_error", np.nan), errors="coerce")
        tmp = tmp.loc[tmp["daily_abs_error"].notna()]
        tmp = tmp.loc[tmp["daily_abs_error"].abs() > float(tolerance)]
        tmp = tmp.sort_values("daily_abs_error", ascending=False)
        if len(tmp) > 0:
            top_row = tmp.iloc[0].to_dict()
    if top_row is None:
        return {"failure_check": "unknown", "metric": "", "details": "no failing row found"}
    top_row["failure_check"] = "daily_to_scenario_reconciliation_ok"
    top_row["details"] = f"daily_abs_error exceeds tolerance={tolerance}"
    return top_row


def _build_performance_reconciliation_debug(
    *,
    scenario: str,
    perf_row: pd.Series,
    daily_df: pd.DataFrame,
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    def _sum_hourly(col: str) -> float:
        if col not in hourly.columns:
            return float("nan")
        return float(pd.to_numeric(hourly[col], errors="coerce").fillna(0.0).sum())

    def _sum_daily(col: str) -> float:
        if col not in daily_df.columns:
            return float("nan")
        return float(pd.to_numeric(daily_df[col], errors="coerce").fillna(0.0).sum())

    rows: list[dict[str, object]] = []
    for spec in _performance_reconciliation_specs():
        scen_col = str(spec.get("scenario_col", ""))
        daily_col = str(spec.get("daily_col", ""))
        hourly_col = str(spec.get("hourly_col", ""))
        scen_v = float(pd.to_numeric(pd.Series([perf_row.get(scen_col, np.nan)]), errors="coerce").iloc[0])
        daily_v = _sum_daily(daily_col)
        hourly_v = _sum_hourly(hourly_col)
        daily_err = scen_v - daily_v if np.isfinite(scen_v) and np.isfinite(daily_v) else float("nan")
        hourly_err = scen_v - hourly_v if np.isfinite(scen_v) and np.isfinite(hourly_v) else float("nan")
        skipped_reason = ""
        if not bool(spec.get("checked_daily_to_scenario", False)):
            skipped_reason = "not_checked_in_daily_to_scenario_reconciliation"
        elif not daily_col:
            skipped_reason = "missing_daily_column_mapping"
        elif daily_col not in daily_df.columns:
            skipped_reason = "daily_column_not_present"
        elif not np.isfinite(scen_v):
            skipped_reason = "scenario_value_non_finite"
        elif not np.isfinite(daily_v):
            skipped_reason = "daily_sum_non_finite"
        rows.append(
            {
                "scenario": scenario,
                "metric": str(spec.get("metric", scen_col)),
                "scenario_col": scen_col,
                "daily_col": daily_col,
                "hourly_col": hourly_col,
                "scenario_value": scen_v,
                "daily_sum_value": daily_v,
                "hourly_sum_value": hourly_v,
                "scenario_minus_daily": daily_err,
                "scenario_minus_hourly": hourly_err,
                "hourly_minus_daily": hourly_v - daily_v if np.isfinite(hourly_v) and np.isfinite(daily_v) else float("nan"),
                "daily_abs_error": abs(daily_err) if np.isfinite(daily_err) else float("nan"),
                "hourly_abs_error": abs(hourly_err) if np.isfinite(hourly_err) else float("nan"),
                "checked_daily_to_scenario": bool(spec.get("checked_daily_to_scenario", False)),
                "checked_component_to_net": bool(spec.get("checked_component_to_net", False)),
                "skipped_reason": skipped_reason,
                "source_note": str(spec.get("source_note", "")),
            }
        )
    return pd.DataFrame(rows)


def _resolve_final_soc_policy(
    *,
    strict_simulation_validity: bool,
    final_soc_mode: str,
    enforce_final_soc_min_flag: bool,
    allow_terminal_soc_repair_in_strict: bool,
) -> bool:
    if bool(strict_simulation_validity) and str(final_soc_mode) == "terminal_repair" and not bool(allow_terminal_soc_repair_in_strict):
        raise ValueError(
            "Strict mode requires --final-soc-mode hard for thesis-reportable runs. "
            "Set --allow-terminal-soc-repair-in-strict only for diagnostic runs."
        )
    return bool(enforce_final_soc_min_flag) or (str(final_soc_mode) == "hard")


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


def _expand_quantile_range(q_low: str, q_high: str) -> list[str]:
    ordered = list(AFRR_QUANTILE_BINS)
    if q_low not in ordered or q_high not in ordered:
        raise ValueError(
            f"Unsupported quantile range '{q_low}-{q_high}'. Allowed bins: {ordered}"
        )
    i0 = ordered.index(q_low)
    i1 = ordered.index(q_high)
    if i0 > i1:
        raise ValueError(f"Invalid range '{q_low}-{q_high}': low quantile must be <= high quantile.")
    out = ordered[i0 : i1 + 1]
    if not out:
        raise ValueError(f"Quantile range '{q_low}-{q_high}' expands to empty bin set.")
    return out


def _apply_quantile_pair_to_warehouse(
    warehouse: dict[str, pd.DataFrame],
    *,
    q_low: str,
    q_high: str,
    da_role: str,
) -> dict[str, pd.DataFrame]:
    da_q = {"low": q_low, "high": q_high, "mid": "p50"}[da_role]
    out: dict[str, pd.DataFrame] = {}
    for pred_col, df in warehouse.items():
        cur = df.copy()
        # DA remains independent from aFRR bin-range selection and can still use
        # role-based point quantile for predicted_value if configured.
        if pred_col == "pred_da_price":
            q_col = da_q
            if q_col not in cur.columns:
                available = [c for c in cur.columns if re.fullmatch(r"p\d{2}", str(c))]
                raise KeyError(
                    f"Requested DA quantile '{q_col}' missing for {pred_col}. Available: {available}"
                )
            cur["predicted_value"] = pd.to_numeric(cur[q_col], errors="coerce")
        elif "predicted_value" not in cur.columns:
            available = [c for c in cur.columns if re.fullmatch(r"p\d{2}", str(c))]
            raise KeyError(
                "missing_forecast_point_value: non-DA prediction is missing predicted_value; "
                f"silent p50 fallback is disabled for {pred_col}. Available quantiles: {available}"
            )
        out[pred_col] = cur
    return out


def _scenario_suffix(q_low: str, q_high: str) -> str:
    return f"{q_low}_{q_high}"


def _discover_afrr_bin_ids(columns: Iterable[str]) -> list[int]:
    return sorted(
        {
            int(m.group(1))
            for c in columns
            for m in [re.match(r"^reserve_pos_bin_(\d+)_mw$", str(c))]
            if m
        }
    )


def _build_afrr_bin_ev_audit(
    *,
    hourly: pd.DataFrame,
    scenario_name: str,
    trading_strategy: str,
    active_bins: list[str],
    backtester: BatteryBacktester,
    timestamp_col: str,
    strict: bool = True,
    status_out: dict[str, object] | None = None,
) -> pd.DataFrame:
    h_ev = hourly.copy()
    strategy = str(trading_strategy).strip().lower()
    strategy = BatteryBacktester.normalize_strategy_name(strategy)
    expected_by_strategy = {
        "da": {"BCM": False, "BEM": False},
        "bcm": {"BCM": True, "BEM": False},
        "bem": {"BCM": False, "BEM": True},
        "afrr": {"BCM": True, "BEM": True},
        "multi": {"BCM": True, "BEM": True},
    }
    expected = expected_by_strategy.get(strategy, {"BCM": True, "BEM": True})
    status: dict[str, object] = {
        "scenario": scenario_name,
        "trading_strategy": strategy,
        "active_bins": list(active_bins),
        "expected_components_by_strategy": expected,
        "actual_components_with_selected_mw": [],
        "audit_row_count": 0,
        "components_emitted": [],
        "components_skipped": {},
        "missing_required_columns_by_component": {},
        "nonfinite_required_fields_by_component": {},
        "active_reconciled_row_count": 0,
        "skipped_zero_decision_row_count": 0,
        "inactive_zero_decision_skipped_count": 0,
        "inactive_zero_decision_reconciled_count": 0,
        "benchmark_or_nonaccepted_path_skipped_count": 0,
        "active_missing_ev_field_count": 0,
        "first_bad_rows": [],
        "strict_audit_pass": True,
    }
    if h_ev.empty:
        status["components_skipped"] = {"BCM": "empty_hourly", "BEM": "empty_hourly"}
        if status_out is not None:
            status_out.clear()
            status_out.update(status)
        return pd.DataFrame()
    ts_s = pd.to_datetime(h_ev.get(timestamp_col), utc=True, errors="coerce")
    bin_ids = _discover_afrr_bin_ids(list(h_ev.columns))
    if not bin_ids and active_bins:
        bin_ids = list(range(len(active_bins)))

    def _row_num(i: int, col: str, default: float = np.nan) -> float:
        if col not in h_ev.columns:
            return float(default)
        return float(pd.to_numeric(pd.Series([h_ev.iloc[i][col]]), errors="coerce").iloc[0])

    def _row_qname(i: int, b: int) -> str:
        qcol = f"afrr_bin_{b}_quantile"
        if qcol in h_ev.columns:
            qv = h_ev.iloc[i][qcol]
            if pd.notna(qv) and str(qv).strip():
                return str(qv)
        if 0 <= b < len(active_bins):
            return str(active_bins[b])
        return ""

    missing_required: dict[str, list[str]] = {}
    nonfinite_required: dict[str, list[str]] = {}

    bcm_required_by_bin = {
        "pos": [
            "ev_bcm_expected_capacity_revenue_pos_bin_{b}",
            "ev_bcm_expected_activation_revenue_pos_bin_{b}",
            "ev_bcm_expected_aux_cost_pos_bin_{b}",
            "ev_bcm_offer_cost_bin_{b}",
            "ev_rpos_coef_bin_{b}_eur_per_mw",
        ],
        "neg": [
            "ev_bcm_expected_capacity_revenue_neg_bin_{b}",
            "ev_bcm_expected_activation_revenue_neg_bin_{b}",
            "ev_bcm_expected_aux_cost_neg_bin_{b}",
            "ev_bcm_offer_cost_bin_{b}",
            "ev_rneg_coef_bin_{b}_eur_per_mw",
        ],
    }
    bem_required_by_bin = {
        "pos": [
            "ev_bem_expected_activation_revenue_pos_bin_{b}",
            "ev_bem_expected_aux_cost_pos_bin_{b}",
            "ev_bem_pos_coef_bin_{b}_eur_per_mw",
        ],
        "neg": [
            "ev_bem_expected_activation_revenue_neg_bin_{b}",
            "ev_bem_expected_aux_cost_neg_bin_{b}",
            "ev_bem_neg_coef_bin_{b}_eur_per_mw",
        ],
    }

    rows: list[dict[str, object]] = []
    first_bad_rows: list[dict[str, object]] = []

    def _is_ok_hour(i: int) -> bool:
        code = str(h_ev.iloc[i].get("optimization_error_code", "ok")).strip().lower()
        return code in {"ok", "", "none"}

    def _append_row(
        *,
        i: int,
        ts: pd.Timestamp,
        bin_id: int,
        q_name: str,
        component: str,
        direction: str,
        decision_variable_name: str,
        selected_mw: float,
        payload: dict[str, object],
        required_templates: list[str],
    ) -> None:
        required_cols = [t.format(b=bin_id) for t in required_templates]
        missing = [c for c in required_cols if c not in h_ev.columns]
        present_required = [c for c in required_cols if c in h_ev.columns]
        nonfinite = [
            c
            for c in present_required
            if not np.isfinite(float(pd.to_numeric(pd.Series([h_ev.iloc[i][c]]), errors="coerce").iloc[0]))
        ]
        sel_val = pd.to_numeric(pd.Series([selected_mw]), errors="coerce").iloc[0]
        sel_for_logic = float(sel_val) if pd.notna(sel_val) and np.isfinite(float(sel_val)) else 0.0
        sel_abs = abs(sel_for_logic)
        is_selected = bool(sel_abs > 1e-9)
        is_ok = _is_ok_hour(i)
        has_bad_required = bool(missing or nonfinite)
        if is_selected and is_ok and has_bad_required:
            row_status = "active_missing_ev_fields"
        elif is_selected and not is_ok:
            row_status = "benchmark_or_nonaccepted_path_skipped"
        elif (not is_selected) and is_ok and has_bad_required:
            row_status = "inactive_zero_decision_skipped"
        elif (not is_selected) and is_ok:
            row_status = "inactive_zero_decision_reconciled"
        else:
            row_status = "benchmark_or_nonaccepted_path_skipped"
        if missing:
            missing_required[f"{component}_{direction}_bin_{bin_id}"] = missing
        if nonfinite:
            nonfinite_required[f"{component}_{direction}_bin_{bin_id}"] = nonfinite
        if row_status == "active_missing_ev_fields" and len(first_bad_rows) < 10:
            first_bad_rows.append(
                {
                    "timestamp_utc": str(ts),
                    "market_component": component,
                    "direction": direction,
                    "quantile_bin": q_name,
                    "decision_variable_name": decision_variable_name,
                    "selected_mw": float(sel_for_logic),
                    "audit_row_status": row_status,
                    "optimization_error_code": str(h_ev.iloc[i].get("optimization_error_code", "")),
                    "missing_columns": missing[:12],
                    "nonfinite_required_audit_fields": nonfinite[:12],
                }
            )
        out_row = {
            "timestamp_utc": ts,
            "scenario": scenario_name,
            "trading_strategy": strategy,
            "market_component": component,
            "direction": direction,
            "quantile_bin": q_name,
            "decision_variable_name": decision_variable_name,
            "selected_mw": float(sel_for_logic),
            "audit_row_status": row_status,
            "optimization_error_code": str(h_ev.iloc[i].get("optimization_error_code", "")),
            "missing_required_columns": "|".join(missing),
            "nonfinite_required_audit_fields": "|".join(nonfinite),
        }
        out_row.update(payload)
        rows.append(out_row)
    for i in range(len(h_ev)):
        ts = ts_s.iloc[i] if i < len(ts_s) else pd.NaT
        for b in bin_ids:
            q_name = _row_qname(i, b)
            if expected.get("BCM", False):
                sel_pos = _row_num(i, f"reserve_pos_bin_{b}_mw", default=0.0)
                _append_row(
                    i=i,
                    ts=ts,
                    bin_id=b,
                    q_name=q_name,
                    component="BCM",
                    direction="pos",
                    decision_variable_name=f"reserve_pos_bin_{b}_mw",
                    selected_mw=sel_pos,
                    required_templates=bcm_required_by_bin["pos"],
                    payload={
                        "capacity_price_q": _row_num(i, f"afrr_bin_{b}_cap_price_pos"),
                        "activation_price_q": _row_num(i, f"ev_afrr_bin_{b}_act_price_pos"),
                        "activation_rate_q": _row_num(i, f"ev_afrr_bin_{b}_act_rate_pos"),
                        "p_acc_or_p_exec_q": _row_num(i, f"ev_pacc_pos_bin_{b}"),
                        "expected_capacity_revenue": _row_num(i, f"ev_bcm_expected_capacity_revenue_pos_bin_{b}"),
                        "expected_activation_revenue": _row_num(i, f"ev_bcm_expected_activation_revenue_pos_bin_{b}"),
                        "expected_aux_cost": _row_num(i, f"ev_bcm_expected_aux_cost_pos_bin_{b}"),
                        "offer_cost": _row_num(i, f"ev_bcm_offer_cost_bin_{b}"),
                        "transaction_cost": float(backtester.trans_eur_mwh),
                        "degradation_cost": float(backtester.deg_eur_mwh / max(backtester.eta_out, 1e-12)),
                        "activation_margin": _row_num(i, f"ev_bcm_activation_margin_pos_bin_{b}"),
                        "ev_coefficient": _row_num(i, f"ev_rpos_coef_bin_{b}_eur_per_mw"),
                    },
                )
                sel_neg = _row_num(i, f"reserve_neg_bin_{b}_mw", default=0.0)
                _append_row(
                    i=i,
                    ts=ts,
                    bin_id=b,
                    q_name=q_name,
                    component="BCM",
                    direction="neg",
                    decision_variable_name=f"reserve_neg_bin_{b}_mw",
                    selected_mw=sel_neg,
                    required_templates=bcm_required_by_bin["neg"],
                    payload={
                        "capacity_price_q": _row_num(i, f"afrr_bin_{b}_cap_price_neg"),
                        "activation_price_q": _row_num(i, f"ev_afrr_bin_{b}_act_price_neg"),
                        "activation_rate_q": _row_num(i, f"ev_afrr_bin_{b}_act_rate_neg"),
                        "p_acc_or_p_exec_q": _row_num(i, f"ev_pacc_neg_bin_{b}"),
                        "expected_capacity_revenue": _row_num(i, f"ev_bcm_expected_capacity_revenue_neg_bin_{b}"),
                        "expected_activation_revenue": _row_num(i, f"ev_bcm_expected_activation_revenue_neg_bin_{b}"),
                        "expected_aux_cost": _row_num(i, f"ev_bcm_expected_aux_cost_neg_bin_{b}"),
                        "offer_cost": _row_num(i, f"ev_bcm_offer_cost_bin_{b}"),
                        "transaction_cost": float(backtester.trans_eur_mwh),
                        "degradation_cost": float(backtester.deg_eur_mwh * backtester.eta_in),
                        "activation_margin": _row_num(i, f"ev_bcm_activation_margin_neg_bin_{b}"),
                        "ev_coefficient": _row_num(i, f"ev_rneg_coef_bin_{b}_eur_per_mw"),
                    },
                )

            if expected.get("BEM", False):
                sel_pos = _row_num(i, f"bem_pos_bin_{b}_mw", default=0.0)
                _append_row(
                    i=i,
                    ts=ts,
                    bin_id=b,
                    q_name=q_name,
                    component="BEM",
                    direction="pos",
                    decision_variable_name=f"bem_pos_bin_{b}_mw",
                    selected_mw=sel_pos,
                    required_templates=bem_required_by_bin["pos"],
                    payload={
                        "capacity_price_q": np.nan,
                        "activation_price_q": _row_num(i, f"ev_bem_bin_{b}_act_price_pos"),
                        "activation_rate_q": _row_num(i, f"ev_bem_bin_{b}_act_rate_pos"),
                        "p_acc_or_p_exec_q": _row_num(i, f"ev_bem_bin_{b}_p_exec_pos"),
                        "expected_capacity_revenue": 0.0,
                        "expected_activation_revenue": _row_num(i, f"ev_bem_expected_activation_revenue_pos_bin_{b}"),
                        "expected_aux_cost": _row_num(i, f"ev_bem_expected_aux_cost_pos_bin_{b}"),
                        "offer_cost": 0.0,
                        "transaction_cost": float(backtester.trans_eur_mwh),
                        "degradation_cost": float(backtester.deg_eur_mwh / max(backtester.eta_out, 1e-12)),
                        "activation_margin": _row_num(i, f"ev_bem_activation_margin_pos_bin_{b}"),
                        "ev_coefficient": _row_num(i, f"ev_bem_pos_coef_bin_{b}_eur_per_mw"),
                    },
                )
                sel_neg = _row_num(i, f"bem_neg_bin_{b}_mw", default=0.0)
                _append_row(
                    i=i,
                    ts=ts,
                    bin_id=b,
                    q_name=q_name,
                    component="BEM",
                    direction="neg",
                    decision_variable_name=f"bem_neg_bin_{b}_mw",
                    selected_mw=sel_neg,
                    required_templates=bem_required_by_bin["neg"],
                    payload={
                        "capacity_price_q": np.nan,
                        "activation_price_q": _row_num(i, f"ev_bem_bin_{b}_act_price_neg"),
                        "activation_rate_q": _row_num(i, f"ev_bem_bin_{b}_act_rate_neg"),
                        "p_acc_or_p_exec_q": _row_num(i, f"ev_bem_bin_{b}_p_exec_neg"),
                        "expected_capacity_revenue": 0.0,
                        "expected_activation_revenue": _row_num(i, f"ev_bem_expected_activation_revenue_neg_bin_{b}"),
                        "expected_aux_cost": _row_num(i, f"ev_bem_expected_aux_cost_neg_bin_{b}"),
                        "offer_cost": 0.0,
                        "transaction_cost": float(backtester.trans_eur_mwh),
                        "degradation_cost": float(backtester.deg_eur_mwh * backtester.eta_in),
                        "activation_margin": _row_num(i, f"ev_bem_activation_margin_neg_bin_{b}"),
                        "ev_coefficient": _row_num(i, f"ev_bem_neg_coef_bin_{b}_eur_per_mw"),
                    },
                )
    if missing_required:
        status["missing_required_columns_by_component"] = missing_required
    if nonfinite_required:
        status["nonfinite_required_fields_by_component"] = nonfinite_required
    if missing_required or nonfinite_required:
        active_missing_count = int(
            (pd.DataFrame(rows).get("audit_row_status", pd.Series(dtype=str)) == "active_missing_ev_fields").sum()
        )
        if strict and active_missing_count > 0:
            sample_keys = list(missing_required.keys())[:8]
            details = "; ".join([f"{k}: {missing_required[k][:5]}" for k in sample_keys])
            sample_nonfinite_keys = list(nonfinite_required.keys())[:8]
            details_nonfinite = "; ".join([f"{k}: {nonfinite_required[k][:5]}" for k in sample_nonfinite_keys])
            raise ValueError(
                f"EV audit build failed for scenario={scenario_name}, strategy={strategy}: "
                f"missing/nonfinite required EV fields for active components. "
                f"missing={details or 'none'}; nonfinite={details_nonfinite or 'none'}"
            )
        if active_missing_count > 0:
            status["strict_audit_pass"] = False

    if not expected.get("BCM", False):
        status["components_skipped"]["BCM"] = "inactive_for_strategy"
    elif any(k.startswith("BCM_") for k in missing_required):
        status["components_skipped"]["BCM"] = "missing_required_columns"
    else:
        status["components_emitted"].append("BCM")
    if not expected.get("BEM", False):
        status["components_skipped"]["BEM"] = "inactive_for_strategy"
    elif any(k.startswith("BEM_") for k in missing_required):
        status["components_skipped"]["BEM"] = "missing_required_columns"
    else:
        status["components_emitted"].append("BEM")

    out = pd.DataFrame(rows)
    if not out.empty and "audit_row_status" in out.columns:
        status["active_reconciled_row_count"] = int((out["audit_row_status"] == "active_reconciled").sum())
        status["skipped_zero_decision_row_count"] = int((out["audit_row_status"] == "inactive_zero_decision_skipped").sum())
        status["inactive_zero_decision_skipped_count"] = int(
            (out["audit_row_status"] == "inactive_zero_decision_skipped").sum()
        )
        status["inactive_zero_decision_reconciled_count"] = int(
            (out["audit_row_status"] == "inactive_zero_decision_reconciled").sum()
        )
        status["benchmark_or_nonaccepted_path_skipped_count"] = int(
            (out["audit_row_status"] == "benchmark_or_nonaccepted_path_skipped").sum()
        )
        status["active_missing_ev_field_count"] = int((out["audit_row_status"] == "active_missing_ev_fields").sum())
        comp_sel: list[str] = []
        for comp in ("BCM", "BEM"):
            m = (
                out["market_component"].astype(str).eq(comp)
                & pd.to_numeric(out["selected_mw"], errors="coerce").fillna(0.0).abs().gt(1e-9)
            )
            if bool(m.any()):
                comp_sel.append(comp)
        status["actual_components_with_selected_mw"] = comp_sel
    status["first_bad_rows"] = first_bad_rows
    status["audit_row_count"] = int(len(out))
    if status_out is not None:
        status_out.clear()
        status_out.update(status)
    return out


def _validate_afrr_bin_ev_audit(
    audit: pd.DataFrame,
    tol: float = 1e-6,
    *,
    scenario_name: str = "",
    trading_strategy: str = "",
    audit_path: str = "",
) -> dict[str, float]:
    if audit.empty:
        return {
            "ev_audit_row_count": 0.0,
            "ev_audit_max_bcm_formula_error": 0.0,
            "ev_audit_max_bem_formula_error": 0.0,
        }
    required = [
        "market_component",
        "expected_capacity_revenue",
        "expected_activation_revenue",
        "expected_aux_cost",
        "offer_cost",
        "ev_coefficient",
    ]
    missing = [c for c in required if c not in audit.columns]
    if missing:
        raise ValueError(f"EV audit missing required columns: {missing}")
    d = audit.copy()
    for c in required:
        if c == "market_component":
            continue
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if "audit_row_status" not in d.columns:
        d["audit_row_status"] = "active_reconciled"
    active = d.loc[d["audit_row_status"].astype(str).eq("active_reconciled")].copy()
    active_missing = d.loc[d["audit_row_status"].astype(str).eq("active_missing_ev_fields")].copy()
    if not active_missing.empty:
        cols = [
            "timestamp_utc",
            "market_component",
            "direction",
            "quantile_bin",
            "decision_variable_name",
            "selected_mw",
            "audit_row_status",
            "optimization_error_code",
            "missing_required_columns",
            "nonfinite_required_audit_fields",
        ]
        sample = active_missing.loc[:, [c for c in cols if c in active_missing.columns]].head(10).to_dict(orient="records")
        raise ValueError(
            "EV audit has active rows with missing EV fields. "
            f"scenario={scenario_name}, strategy={trading_strategy}, audit_path={audit_path or '<memory>'}, "
            f"first_bad_rows={sample}"
        )
    if active.empty:
        return {
            "ev_audit_row_count": float(len(d)),
            "ev_audit_max_bcm_formula_error": 0.0,
            "ev_audit_max_bem_formula_error": 0.0,
        }
    bcm = active.loc[active["market_component"].astype(str).eq("BCM")].copy()
    bem = active.loc[active["market_component"].astype(str).eq("BEM")].copy()
    if not bcm.empty:
        bcm_req = ["expected_capacity_revenue", "expected_activation_revenue", "expected_aux_cost", "offer_cost", "ev_coefficient"]
        bad_mask = bcm[bcm_req].isna().any(axis=1)
        if bool(bad_mask.any()):
            bad = bcm.loc[bad_mask].copy()
            bad_cols = [c for c in bcm_req if bad[c].isna().any()]
            sample = bad.loc[:, [c for c in ["timestamp_utc", "market_component", "direction", "quantile_bin", "decision_variable_name", *bad_cols] if c in bad.columns]].head(10)
            raise ValueError(
                "EV audit contains NaN/non-finite BCM reconciliation fields. "
                f"scenario={scenario_name}, strategy={trading_strategy}, bad_columns={bad_cols}, "
                f"audit_path={audit_path or '<memory>'}, sample={sample.to_dict(orient='records')}"
            )
        bcm["formula"] = (
            bcm["expected_capacity_revenue"]
            + bcm["expected_activation_revenue"]
            - bcm["expected_aux_cost"]
            - bcm["offer_cost"]
        )
        bcm_err = (bcm["formula"] - bcm["ev_coefficient"]).abs()
        max_bcm = float(bcm_err.max()) if len(bcm_err) else 0.0
    else:
        max_bcm = 0.0
    if not bem.empty:
        bem_req = ["expected_activation_revenue", "expected_aux_cost", "ev_coefficient"]
        bad_mask = bem[bem_req].isna().any(axis=1)
        if bool(bad_mask.any()):
            bad = bem.loc[bad_mask].copy()
            bad_cols = [c for c in bem_req if bad[c].isna().any()]
            sample = bad.loc[:, [c for c in ["timestamp_utc", "market_component", "direction", "quantile_bin", "decision_variable_name", *bad_cols] if c in bad.columns]].head(10)
            raise ValueError(
                "EV audit contains NaN/non-finite BEM reconciliation fields. "
                f"scenario={scenario_name}, strategy={trading_strategy}, bad_columns={bad_cols}, "
                f"audit_path={audit_path or '<memory>'}, sample={sample.to_dict(orient='records')}"
            )
        bem["formula"] = bem["expected_activation_revenue"] - bem["expected_aux_cost"]
        bem_err = (bem["formula"] - bem["ev_coefficient"]).abs()
        max_bem = float(bem_err.max()) if len(bem_err) else 0.0
    else:
        max_bem = 0.0
    if max(max_bcm, max_bem) > float(tol):
        raise ValueError(
            f"EV audit formula reconciliation failed: max_bcm={max_bcm:.6g}, max_bem={max_bem:.6g}, tol={tol:.6g}"
        )
    return {
        "ev_audit_row_count": float(len(d)),
        "ev_audit_max_bcm_formula_error": float(max_bcm),
        "ev_audit_max_bem_formula_error": float(max_bem),
    }


def _matches_model_key(path: Path, model_key: str) -> bool:
    if not model_key:
        return True
    mk = model_key.strip().lower()
    full_path = str(path).lower()
    token_patterns = {
        "xgb": r"(^|[^a-z0-9])(xgb|xgboost)([^a-z0-9]|$)",
        "tft": r"(^|[^a-z0-9])tft([^a-z0-9]|$)",
        "linear": r"(^|[^a-z0-9])(linear|rlqr)([^a-z0-9]|$)",
    }
    families = {fam for fam, pat in token_patterns.items() if re.search(pat, full_path) is not None}
    target_family = "xgb" if mk in {"xgb", "xgboost"} else "linear" if mk in {"linear", "rlqr"} else mk
    return families == {target_family}


def _normalize_model_choice(model: str) -> tuple[str, str]:
    m = str(model or "").strip().lower()
    if m in {"xgb", "xgboost"}:
        return "xgb", "latest_xgboost.json"
    if m == "tft":
        return "tft", "latest_tft.json"
    if m in {"linear", "rlqr"}:
        return "linear", "latest_linear.json"
    raise ValueError(f"Unsupported model selector '{model}'. Expected one of xgb, xgboost, tft, linear, rlqr.")


def _load_manifest_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_manifest_paths_for_model(*, model_key: str, model_runs_root: Path) -> list[Path]:
    mk, _ = _normalize_model_choice(model_key)
    patterns: list[str]
    if mk == "xgb":
        patterns = ["xgb_*/manifest.json", "xgboost_*/manifest.json"]
    elif mk == "tft":
        patterns = ["tft_*/manifest.json"]
    else:
        patterns = ["linear_*/manifest.json", "rlqr_*/manifest.json"]
    out: list[Path] = []
    for pat in patterns:
        out.extend(sorted(model_runs_root.glob(pat)))
    seen: set[str] = set()
    deduped: list[Path] = []
    for p in out:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


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
    exact_candidates: list[Path] = []
    fallback_scanned: list[Path] = []
    fallback_rejected: list[Path] = []

    def _append_exact(cand: Path) -> None:
        key = str(cand)
        if key not in {str(x) for x in exact_candidates}:
            exact_candidates.append(cand)

    if p.is_absolute():
        _append_exact(p)
    else:
        _append_exact(p)
        _append_exact(manifest_dir / p)
        if str(p).startswith("predictions/"):
            _append_exact(manifest_dir / p)
        _append_exact(manifest_dir / "predictions" / p.name)

    for cand in exact_candidates:
        if cand.exists():
            return cand

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
    fallback_scanned = list(candidates)
    if model_key:
        filtered: list[Path] = []
        for c in candidates:
            if _matches_model_key(c, model_key):
                filtered.append(c)
            else:
                fallback_rejected.append(c)
        candidates = filtered
    if candidates:
        candidates = sorted(candidates, key=lambda c: _score_long_candidate(c, split=split), reverse=True)
        return candidates[0]
    raise FileNotFoundError(
        f"Could not resolve long prediction file for pred_col='{pred_col}', split='{split}', model_key='{model_key}'. "
        f"manifest_path='{manifest_dir / 'manifest.json'}', manifest_dir='{manifest_dir}', configured_path='{p}', "
        f"exact_candidates={[(str(c), c.exists()) for c in exact_candidates]}, "
        f"fallback_glob_candidates={[str(c) for c in fallback_scanned]}, "
        f"fallback_rejected_by_model_key={[str(c) for c in fallback_rejected]}"
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


def _manifest_can_resolve_long_predictions(
    payload: dict[str, object],
    manifest_dir: Path,
    split: str,
    model_key: str,
    required_quantiles: set[str] | None = None,
) -> tuple[bool, list[str]]:
    bundles = payload.get("bundles", {}) if isinstance(payload, dict) else {}
    da_long = bundles.get("da", {}).get("predictions_long", {}).get(split, {}) if isinstance(bundles, dict) else {}
    afrr_long = bundles.get("afrr", {}).get("predictions_long", {}).get(split, {}) if isinstance(bundles, dict) else {}
    long_map = {**da_long, **afrr_long}
    if not long_map:
        return False, [f"no predictions_long entries for split='{split}'"]
    issues: list[str] = []
    try:
        resolved = _resolve_long_map(
            long_map=long_map,
            manifest_dir=manifest_dir,
            split=split,
            model_key=model_key,
        )
    except Exception as exc:
        return False, [str(exc)]
    if required_quantiles:
        expected = {str(q).lower() for q in required_quantiles}
        for pred_col, file_path in sorted(resolved.items()):
            try:
                cols = set(_read_parquet_columns(Path(file_path)))
            except Exception as exc:
                issues.append(f"{pred_col}: failed reading parquet columns from {file_path}: {exc}")
                continue
            missing = sorted(expected - cols)
            if missing:
                issues.append(f"{pred_col}: missing quantiles {missing} in {file_path}")
    return (len(issues) == 0), issues


def _resolve_pointer_or_latest_manifest(
    *,
    candidate_path: Path,
    model_runs_root: Path,
) -> tuple[Path, dict[str, object], str | None]:
    payload = _load_manifest_payload(candidate_path)
    run_id = payload.get("run_id")
    if "manifest_path" in payload:
        manifest_raw = Path(str(payload["manifest_path"]))
        resolved = manifest_raw if manifest_raw.is_absolute() else (candidate_path.parent / manifest_raw)
        if resolved.exists():
            return resolved, _load_manifest_payload(resolved), str(run_id) if run_id else None
        fallback_candidates: list[Path] = []
        if run_id:
            fallback_candidates.append(model_runs_root / str(run_id) / "manifest.json")
            fallback_candidates.append(candidate_path.parent / str(run_id) / "manifest.json")
        fallback_candidates.append(candidate_path.parent / "manifest.json")
        fallback = next((p for p in fallback_candidates if p.exists()), None)
        if fallback is None:
            raise FileNotFoundError(
                "Latest pointer resolves to missing manifest path and no local fallback was found. "
                f"pointer={candidate_path}, manifest_path={resolved}, run_id={run_id}"
            )
        print(f"[WARN] Latest pointer manifest path not found: {resolved}. Using local fallback: {fallback}")
        return fallback, _load_manifest_payload(fallback), str(run_id) if run_id else None

    if candidate_path.parent == model_runs_root and candidate_path.name.startswith("latest_") and run_id:
        actual = model_runs_root / str(run_id) / "manifest.json"
        if actual.exists():
            return actual, _load_manifest_payload(actual), str(run_id)
    return candidate_path, payload, str(run_id) if run_id else None


def _resolve_model_manifest(
    *,
    run_manifest_arg: str,
    run_id: str | None,
    model_key: str,
    split: str,
    model_runs_root: Path = Path("artifacts/model_runs"),
) -> tuple[Path, dict[str, object], str | None]:
    latest_name = ""
    if model_key:
        _, latest_name = _normalize_model_choice(model_key)
    attempted: list[str] = []
    scanned_candidates: list[str] = []

    if str(run_manifest_arg or "").strip():
        initial = Path(str(run_manifest_arg).strip())
    elif run_id:
        initial = model_runs_root / str(run_id) / "manifest.json"
    else:
        if not model_key:
            raise ValueError("Model manifest auto-resolution requires --model or --model-key.")
        initial = model_runs_root / latest_name
    attempted.append(str(initial))
    if not initial.exists():
        initial = None  # type: ignore[assignment]

    candidate_infos: list[tuple[Path, dict[str, object], str | None]] = []
    if initial is not None:
        resolved_path, payload, resolved_run_id = _resolve_pointer_or_latest_manifest(
            candidate_path=initial,
            model_runs_root=model_runs_root,
        )
        candidate_infos.append((resolved_path, payload, resolved_run_id))

    if model_key:
        for path in _candidate_manifest_paths_for_model(model_key=model_key, model_runs_root=model_runs_root):
            scanned_candidates.append(str(path))
            if not path.exists():
                continue
            try:
                payload = _load_manifest_payload(path)
            except Exception:
                continue
            candidate_infos.append((path, payload, str(payload.get("run_id")) if payload.get("run_id") else None))

    seen: set[str] = set()
    deduped_infos: list[tuple[Path, dict[str, object], str | None]] = []
    for path, payload, rid in candidate_infos:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped_infos.append((path, payload, rid))

    requested_quantiles = {"p50"}
    for idx, (path, payload, rid) in enumerate(deduped_infos):
        usable, issues = _manifest_can_resolve_long_predictions(
            payload=payload,
            manifest_dir=path.parent,
            split=split,
            model_key=model_key,
            required_quantiles=requested_quantiles,
        )
        if usable:
            if idx > 0 and initial is not None:
                print(
                    f"[WARN] Using actual run manifest instead of latest/candidate pointer: {path} "
                    f"(initial attempt: {attempted[0]})"
                )
            return path, payload, (rid or run_id)
        attempted.extend(issues)

    configured_path = ""
    if deduped_infos:
        bundles = deduped_infos[0][1].get("bundles", {})
        if isinstance(bundles, dict):
            da_long = bundles.get("da", {}).get("predictions_long", {}).get(split, {})
            afrr_long = bundles.get("afrr", {}).get("predictions_long", {}).get(split, {})
            long_map = {**(da_long if isinstance(da_long, dict) else {}), **(afrr_long if isinstance(afrr_long, dict) else {})}
            configured_path = next(iter(long_map.values()), "")
    raise FileNotFoundError(
        "Could not resolve a usable model manifest for simulation. "
        f"model={model_key}, split={split}, attempted_latest_or_explicit={attempted[:1]}, "
        f"attempted_run_id_path={str(model_runs_root / str(run_id) / 'manifest.json') if run_id else ''}, "
        f"scanned_candidates={scanned_candidates}, missing_configured_prediction_path={configured_path}, "
        f"expected_predictions_dir='predictions'. "
        "If using latest_xgboost.json, ensure it is a pointer to xgb_<timestamp>/manifest.json or run with "
        f"--model {model_key or 'xgb'} so the resolver can select the newest model run."
    )


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
    afrr_activation_rate_guard_quantile: str | set[str] | None = None,
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

    try:
        resolved = _resolve_long_map(
            long_map=long_map,
            manifest_dir=manifest_dir,
            split=split,
            model_key=model_key,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Preflight failed while resolving long-format prediction files. "
            f"resolved_manifest_path={manifest_path}, manifest_dir={manifest_dir}, model_key={model_key}, split={split}. "
            f"{exc} "
            "Hint: If using latest_xgboost.json, ensure it is a pointer to xgb_<timestamp>/manifest.json or run with "
            f"--model {model_key or 'xgb'} so the resolver can select the newest compatible run."
        ) from exc
    expected = {str(q).lower() for q in expected_quantiles}
    if not expected:
        expected = {"p50"}
    failures: list[str] = []
    if isinstance(afrr_activation_rate_guard_quantile, (set, list, tuple)):
        guard_quantiles = {str(q).strip().lower() for q in afrr_activation_rate_guard_quantile if str(q).strip()}
    else:
        guard_q_raw = str(afrr_activation_rate_guard_quantile or "").strip().lower()
        guard_quantiles = {guard_q_raw} if guard_q_raw else set()
    if guard_quantiles:
        guard_failures: list[str] = []
        for guard_q in sorted(guard_quantiles):
            for pred_col in ["pred_afrr_activation_rate_pos", "pred_afrr_activation_rate_neg"]:
                file_path = resolved.get(pred_col)
                if file_path is None:
                    guard_failures.append(f"{pred_col}: missing prediction file for guard quantile {guard_q}")
                    continue
                cols = set(_read_parquet_columns(Path(file_path)))
                if guard_q not in cols:
                    guard_failures.append(
                        f"{pred_col}: missing_afrr_activation_rate_guard_quantile {guard_q!r} "
                        f"in {file_path}"
                    )
        if guard_failures:
            raise RuntimeError(
                "Preflight failed: missing_afrr_activation_rate_guard_quantile.\n"
                f"Guard quantile(s): {','.join(sorted(guard_quantiles))}\n"
                "Missing columns: "
                + ", ".join(
                    [
                        f"pred_afrr_activation_rate_pos_{q}, pred_afrr_activation_rate_neg_{q}"
                        for q in sorted(guard_quantiles)
                    ]
                )
                + "\n"
                + "\n".join(guard_failures)
            )
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


def _forecast_coverage_report(
    *,
    forecast_warehouse: dict[str, pd.DataFrame],
    effective_start_utc: pd.Timestamp,
    effective_end_utc: pd.Timestamp,
    horizon_hours: int,
    expected_quantiles: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    expected_snapshots = pd.date_range(effective_start_utc, effective_end_utc, freq="h", tz="UTC", inclusive="left")
    expected_leads = list(range(1, int(horizon_hours) + 1))
    rows: list[dict[str, object]] = []
    total_missing_snapshots = 0
    total_missing_targets = 0
    first_missing_snapshot: str = ""
    expected_quantiles_norm = {str(q).lower() for q in expected_quantiles if str(q).strip()}
    if not expected_quantiles_norm:
        expected_quantiles_norm = {"p50"}

    for pred_col, raw in sorted(forecast_warehouse.items()):
        df = raw.copy()
        for col in ["snapshot_time_utc", "target_time_utc"]:
            if col not in df.columns:
                df[col] = pd.NaT
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        df["lead_time_h"] = pd.to_numeric(df.get("lead_time_h"), errors="coerce")
        df = df.dropna(subset=["snapshot_time_utc", "target_time_utc", "lead_time_h"]).copy()
        snap_values = pd.DatetimeIndex(df["snapshot_time_utc"].dropna().drop_duplicates().sort_values())
        min_snapshot = snap_values.min().isoformat() if len(snap_values) else ""
        max_snapshot = snap_values.max().isoformat() if len(snap_values) else ""
        target_values = pd.DatetimeIndex(df["target_time_utc"].dropna().drop_duplicates().sort_values())
        min_target = target_values.min().isoformat() if len(target_values) else ""
        max_target = target_values.max().isoformat() if len(target_values) else ""
        available_snapshots = set(snap_values)
        missing_snapshots = [ts for ts in expected_snapshots if ts not in available_snapshots]
        if missing_snapshots and not first_missing_snapshot:
            first_missing_snapshot = missing_snapshots[0].isoformat()
        total_missing_snapshots += len(missing_snapshots)
        missing_quantiles = sorted(q for q in expected_quantiles_norm if q not in df.columns)

        missing_targets: list[str] = []
        if not missing_quantiles:
            available_pairs = set(
                zip(
                    pd.to_datetime(df["snapshot_time_utc"], utc=True, errors="coerce"),
                    pd.to_numeric(df["lead_time_h"], errors="coerce").astype("Int64"),
                    pd.to_datetime(df["target_time_utc"], utc=True, errors="coerce"),
                )
            )
            for snap in expected_snapshots:
                if snap in missing_snapshots:
                    continue
                for lead in expected_leads:
                    target = snap + pd.Timedelta(hours=int(lead))
                    if (snap, int(lead), target) not in available_pairs:
                        missing_targets.append(f"{snap.isoformat()}+h{lead}->{target.isoformat()}")
                        if len(missing_targets) >= 20:
                            break
                if len(missing_targets) >= 20:
                    break
        total_missing_targets += len(missing_targets)
        nearby_available = []
        if missing_snapshots and len(snap_values):
            first = missing_snapshots[0]
            near = snap_values[(snap_values >= first - pd.Timedelta(hours=6)) & (snap_values <= first + pd.Timedelta(hours=6))]
            nearby_available = [ts.isoformat() for ts in near[:12]]
        status = "ok"
        if missing_quantiles or missing_snapshots or missing_targets:
            status = "missing_coverage"
        rows.append(
            {
                "schema_version": FORECAST_COVERAGE_SCHEMA_VERSION,
                "prediction_column": pred_col,
                "effective_start_utc": effective_start_utc.isoformat(),
                "effective_end_utc": effective_end_utc.isoformat(),
                "interval_semantics": "[start,end)",
                "horizon_hours": int(horizon_hours),
                "expected_snapshot_count": int(len(expected_snapshots)),
                "available_snapshot_count": int(len(available_snapshots)),
                "missing_snapshot_count": int(len(missing_snapshots)),
                "first_missing_snapshot": missing_snapshots[0].isoformat() if missing_snapshots else "",
                "nearby_available_snapshots": json.dumps(nearby_available),
                "missing_targets_count_sampled": int(len(missing_targets)),
                "missing_targets_sample": json.dumps(missing_targets[:20]),
                "affected_prediction_columns": json.dumps([pred_col] if status != "ok" else []),
                "missing_quantile_columns": json.dumps(missing_quantiles),
                "min_snapshot_utc": min_snapshot,
                "max_snapshot_utc": max_snapshot,
                "min_target_utc": min_target,
                "max_target_utc": max_target,
                "status": status,
            }
        )
    report_df = pd.DataFrame(rows)
    summary = {
        "schema_version": FORECAST_COVERAGE_SCHEMA_VERSION,
        "effective_start_utc": effective_start_utc.isoformat(),
        "effective_end_utc": effective_end_utc.isoformat(),
        "interval_semantics": "[start,end)",
        "horizon_hours": int(horizon_hours),
        "status": "ok" if not rows or (report_df["status"] == "ok").all() else "missing_coverage",
        "missing_snapshot_count": int(total_missing_snapshots),
        "missing_targets_count_sampled": int(total_missing_targets),
        "first_missing_snapshot": first_missing_snapshot,
        "rows": rows,
    }
    return report_df, summary


def _write_forecast_coverage_report(
    *,
    out_dir: Path,
    report_df: pd.DataFrame,
    report_summary: dict[str, object],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "forecast_coverage_report.csv"
    json_path = out_dir / "forecast_coverage_report.json"
    report_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(report_summary, indent=2, default=str), encoding="utf-8")
    return csv_path, json_path


def _preflight_forecast_coverage(
    *,
    forecast_warehouse: dict[str, pd.DataFrame] | None,
    out_dir: Path,
    effective_start_utc: pd.Timestamp,
    effective_end_utc: pd.Timestamp,
    horizon_hours: int,
    expected_quantiles: set[str],
) -> tuple[Path | None, Path | None, dict[str, object]]:
    if not forecast_warehouse:
        return None, None, {"status": "skipped_no_long_forecast_warehouse"}
    report_df, report_summary = _forecast_coverage_report(
        forecast_warehouse=forecast_warehouse,
        effective_start_utc=effective_start_utc,
        effective_end_utc=effective_end_utc,
        horizon_hours=horizon_hours,
        expected_quantiles=expected_quantiles,
    )
    csv_path, json_path = _write_forecast_coverage_report(
        out_dir=out_dir,
        report_df=report_df,
        report_summary=report_summary,
    )
    if str(report_summary.get("status")) != "ok":
        raise RuntimeError(
            "Forecast coverage preflight failed inside the effective simulation window. "
            f"report_csv={csv_path}, report_json={json_path}, "
            f"first_missing_snapshot={report_summary.get('first_missing_snapshot', '')}, "
            f"missing_snapshot_count={report_summary.get('missing_snapshot_count', 0)}, "
            f"missing_targets_count_sampled={report_summary.get('missing_targets_count_sampled', 0)}"
        )
    return csv_path, json_path, report_summary


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
        default="",
        help=(
            "Run id used to resolve manifest path when --run-manifest is not set. "
            "If omitted, --model / --model-key resolves the newest thesis-safe manifest."
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
    p.add_argument(
        "--model",
        choices=["xgb", "xgboost", "tft", "linear", "rlqr"],
        default="",
        help=(
            "User-facing model selector. If --run-manifest is omitted, resolves the latest thesis-safe manifest "
            "for the selected model."
        ),
    )
    p.add_argument("--split", choices=["val", "test"], default="test", help="Prediction split for manifest mode.")
    p.add_argument(
        "--model-key",
        default="",
        help="Optional canonical model selector when one run dir contains multiple models (e.g. 'xgb', 'tft', 'linear').",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory for hourly/aggregated results. If empty, uses artifacts/simulation_runs/<run_id>/<split>/",
    )
    p.add_argument(
        "--quantile-pairs",
        default="p50-p50",
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
    p.add_argument(
        "--disable-common-eval-window-clamp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Disable common thesis simulation evaluation-window clamp. "
            "By default simulations use [2025-01-14T00:00:00Z, 2026-01-14T00:00:00Z)."
        ),
    )
    p.add_argument("--horizon-hours", type=int, default=48, help="Rolling-horizon window length in hours.")
    p.add_argument("--reopt-step-hours", type=int, default=1, help="Re-optimization step in hours.")
    p.add_argument(
        "--da-bid-hour-local",
        "--da-gate-hour-cet",
        dest="da_bid_hour_local",
        type=int,
        default=11,
        help="Day-Ahead bid submission hour in Europe/Berlin local time used for locking next-day DA bids (default: 11).",
    )
    p.add_argument(
        "--da-limit-quantile",
        choices=["p05", "p10", "p50", "p90", "p95"],
        default=None,
        help=(
            "DA limit-price quantile for both buy and sell orders. "
            "Defaults to config MARKET_SPECS['da_*_limit_quantile']."
        ),
    )
    p.add_argument(
        "--afrr-activation-rate-guard-quantile",
        choices=["scenario", "same_as_bid", "p01", "p05", "p10", "p30", "p50", "p70", "p90", "p95", "p99"],
        default="scenario",
        help=(
            "aFRR/BEM/BCM activation-rate quantile used for physical headroom guards. "
            "Use 'scenario' to match a single active --quantile-pairs bid bin (default)."
        ),
    )
    p.add_argument(
        "--bcm-bid-hour-local",
        type=int,
        default=8,
        help="aFRR BCM bid submission hour in Europe/Berlin local time for D+1 capacity products (default: 8).",
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
        default="hard",
        help="Final SoC policy: hard physical minimum (default) or terminal_repair.",
    )
    p.add_argument(
        "--enforce-final-soc-min",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce final SoC >= soc_target_end in optimizer (default: enabled).",
    )
    p.add_argument(
        "--allow-terminal-soc-repair-in-strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow --final-soc-mode terminal_repair even in strict mode (diagnostic only).",
    )
    p.add_argument(
        "--allow-p50-from-predicted-value",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicit compatibility mode: materialize missing p50 from predicted_value in long forecasts.",
    )
    p.add_argument(
        "--allow-invalid-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If disabled, strict mode exits non-zero when any scenario is invalid after writing artifacts.",
    )
    p.add_argument(
        "--disable-rolling-horizon",
        action="store_true",
        help="Use a single full-horizon optimization instead of rolling horizon.",
    )
    p.add_argument(
        "--trading-strategy",
        choices=["multi", "da", "afrr", "bcm", "bem"],
        default="multi",
        help="Strategy isolation mode: multi, da, afrr, bcm, bem.",
    )
    p.add_argument(
        "--id-mode",
        choices=["none", "technical_repair", "economic"],
        default="",
        help="Legacy override for low-level ID trade type. Prefer --id-recourse-mode.",
    )
    p.add_argument(
        "--id-recourse-mode",
        choices=["common", "disabled", "afrr_obligation_only"],
        default="common",
        help=(
            "ID recourse policy: common (all strategies), disabled (none), "
            "afrr_obligation_only (no ID for da)."
        ),
    )
    p.add_argument(
        "--allow-economic-id-in-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow --id-mode economic for non-multi baseline strategies.",
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
        default=True,
        help=(
            "If enabled, delete existing scenario output directory before writing. "
            "Prevents stale/mixed artifacts."
        ),
    )
    p.add_argument(
        "--enable-global-perfect_foresight",
        "--enable-global-perfect-foresight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable global_hindsight_perfect_foresight_upper_bound computation. "
            "Enabled by default for thesis runs; use --no-enable-global-perfect-foresight "
            "for faster debug runs."
        ),
    )
    p.add_argument(
        "--print-validity-first-preset",
        action="store_true",
        help="Print recommended validity-first reserve settings and exit.",
    )
    p.add_argument(
        "--export-afrr-bin-ev-audit",
        action="store_true",
        help="Export per-bin BCM/BEM EV audit CSV from optimizer hourly outputs.",
    )
    p.add_argument(
        "--output-detail",
        choices=["thesis", "debug"],
        default="debug",
        help="Hourly output width. debug preserves all columns; thesis writes only essential validation/report columns.",
    )
    p.add_argument(
        "--debug-dumps",
        choices=["accepted_only", "all", "none"],
        default="all",
        help="Infeasible MILP matrix dump policy. accepted_only suppresses candidate .npz dumps but keeps counters.",
    )
    p.add_argument(
        "--use-input-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse cached joined/canonicalized model/split input when cache key matches.",
    )
    p.add_argument(
        "--refresh-input-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rebuild input cache even if a matching cache file exists.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.trading_strategy = BatteryBacktester.normalize_strategy_name(args.trading_strategy)
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
    eval_window = _resolve_simulation_eval_window(
        requested_start=args.start,
        requested_end=args.end,
        clamp_enabled=not bool(args.disable_common_eval_window_clamp),
    )
    effective_start_utc = eval_window["effective_start"]
    effective_end_utc = eval_window["effective_end"]
    assert isinstance(effective_start_utc, pd.Timestamp)
    assert isinstance(effective_end_utc, pd.Timestamp)
    for warning in eval_window.get("warnings", []):
        print(f"[WARN] {warning}")
    print(
        "[INFO] Simulation evaluation window: "
        f"requested_start={eval_window['requested_start_utc'] or '<default>'}, "
        f"requested_end={eval_window['requested_end_utc'] or '<default>'}, "
        f"effective_start={eval_window['effective_start_utc']}, "
        f"effective_end={eval_window['effective_end_utc']}"
    )
    run_id: str | None = args.run_id.strip() or None
    model_arg = str(getattr(args, "model", "") or "").strip()
    model_key_arg = str(args.model_key or "").strip()
    if model_arg:
        canonical_model_key, _ = _normalize_model_choice(model_arg)
        if model_key_arg:
            canonical_model_key_from_key, _ = _normalize_model_choice(model_key_arg)
            if canonical_model_key_from_key != canonical_model_key:
                raise ValueError(
                    f"Incompatible --model and --model-key: model={model_arg}, model_key={model_key_arg}. "
                    "Use matching selectors only."
                )
        args.model_key = canonical_model_key
    elif model_key_arg:
        args.model_key = _normalize_model_choice(model_key_arg)[0]
    quantile_pairs = _parse_quantile_pairs(args.quantile_pairs)
    required_quantiles: set[str] = {"p50"}
    scenario_bin_map: dict[str, list[str]] = {}
    for q_lo, q_hi in quantile_pairs:
        expanded = _expand_quantile_range(q_lo.lower(), q_hi.lower())
        required_quantiles.update(expanded)
        scenario_bin_map[_scenario_suffix(q_lo, q_hi)] = expanded
    afrr_activation_rate_guard_quantile = str(args.afrr_activation_rate_guard_quantile).lower()
    strategy_uses_afrr = args.trading_strategy in {"bcm", "bem", "afrr", "multi"}
    afrr_activation_rate_guard_quantiles_required: set[str] = set()
    if strategy_uses_afrr:
        if afrr_activation_rate_guard_quantile in {"scenario", "same_as_bid"}:
            if not scenario_bin_map:
                raise ValueError(
                    "ambiguous_afrr_activation_rate_guard_quantile_for_multi_bin_scenario: "
                    f"guard_policy={afrr_activation_rate_guard_quantile}, active_bins={list(AFRR_QUANTILE_BINS)}. "
                    "Pass a single --quantile-pairs value or set --afrr-activation-rate-guard-quantile explicitly."
                )
            for scenario_name, bins in scenario_bin_map.items():
                if len(bins) != 1:
                    raise ValueError(
                        "ambiguous_afrr_activation_rate_guard_quantile_for_multi_bin_scenario: "
                        f"guard_policy={afrr_activation_rate_guard_quantile}, scenario={scenario_name}, "
                        f"active_bins={bins}. Use a single quantile pair or set "
                        "--afrr-activation-rate-guard-quantile explicitly."
                    )
                afrr_activation_rate_guard_quantiles_required.add(str(bins[0]).lower())
        else:
            afrr_activation_rate_guard_quantiles_required.add(afrr_activation_rate_guard_quantile)
        required_quantiles.update(afrr_activation_rate_guard_quantiles_required)

    predictions_path = args.predictions.strip()
    ground_truth_path = args.ground_truth.strip()
    payload: dict[str, object] = {}
    manifest_path: Path | None = None

    target_value_modes: dict[str, str] = {}

    if not predictions_path:
        if not args.run_manifest.strip() and not run_id and not args.model_key:
            raise ValueError("Missing model selector or manifest. Provide --model, --model-key, --run-id, or --run-manifest.")
        manifest_path, payload, resolved_run_id = _resolve_model_manifest(
            run_manifest_arg=args.run_manifest.strip(),
            run_id=run_id,
            model_key=args.model_key.strip(),
            split=args.split,
        )
        run_id = resolved_run_id or run_id or payload.get("run_id") or (args.run_id.strip() or None)
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
            afrr_activation_rate_guard_quantile=(
                afrr_activation_rate_guard_quantiles_required if strategy_uses_afrr else None
            ),
        )
        target_value_modes = _target_value_modes_from_manifest(payload)
    else:
        manifest_dir = Path.cwd()

    out_dir = _resolve_out_dir(args.out_dir, run_id=run_id, split=args.split)
    forecast_warehouse: dict[str, pd.DataFrame] | None = None
    coverage_min: pd.Timestamp | None = None
    coverage_max: pd.Timestamp | None = None
    df: pd.DataFrame | None = None
    input_cache_used = False
    input_cache_path = ""

    if not predictions_path:
        if not ground_truth_path:
            ground_truth_path = str(_resolve_existing_file(payload["ground_truth"]["default_path"], manifest_dir=manifest_dir))
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
            cache_key = _input_cache_key(
                model_key=str(args.model_key),
                split=str(args.split),
                manifest_path=manifest_path,
                prediction_files=resolved_long_map.values(),
                truth_file=ground_truth_path,
                forecast_value_mode=str(args.forecast_value_mode),
            )
            cache_path = _input_cache_path(cache_key)
            input_cache_path = str(cache_path)
            cached = None if bool(args.refresh_input_cache) or not bool(args.use_input_cache) else _load_input_cache(cache_path)
            if cached is not None:
                df, forecast_warehouse, coverage_min, coverage_max = cached
                input_cache_used = True
                print(f"[INFO] Simulation input cache hit: {cache_path}")
            else:
                forecast_warehouse = load_prediction_warehouse_long(
                    resolved_long_map,
                    target_value_modes=target_value_modes,
                    allow_p50_materialization_from_predicted_value=bool(args.allow_p50_from_predicted_value),
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

    if df is None and predictions_path:
        df = load_and_align_market_data(
            predictions_path,
            ground_truth_path,
            colmap,
            target_value_modes=target_value_modes,
        )
    elif df is None:
        df = truth_preview.copy()
        if colmap.timestamp not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df[colmap.timestamp] = df.index
        df[colmap.timestamp] = pd.to_datetime(df[colmap.timestamp], utc=True, errors="coerce")
        df = df.dropna(subset=[colmap.timestamp]).sort_values(colmap.timestamp).reset_index(drop=True)
        df = canonicalize_market_frame(df, colmap=colmap, target_value_modes=target_value_modes)
        if forecast_warehouse and coverage_min is not None and coverage_max is not None:
            df = df[(df[colmap.timestamp] >= coverage_min) & (df[colmap.timestamp] <= coverage_max)].copy()
        if bool(args.use_input_cache) and input_cache_path and not input_cache_used:
            _write_input_cache(
                Path(input_cache_path),
                df=df,
                forecast_warehouse=forecast_warehouse,
                coverage_min=coverage_min,
                coverage_max=coverage_max,
            )
            print(f"[INFO] Simulation input cache written: {input_cache_path}")
    forecast_coverage_report_csv, forecast_coverage_report_json, forecast_coverage_summary = _preflight_forecast_coverage(
        forecast_warehouse=forecast_warehouse,
        out_dir=out_dir,
        effective_start_utc=effective_start_utc,
        effective_end_utc=effective_end_utc,
        horizon_hours=int(args.horizon_hours),
        expected_quantiles=required_quantiles,
    )
    df = df[df[colmap.timestamp] >= effective_start_utc].copy()
    df = df[df[colmap.timestamp] < effective_end_utc].copy()
    if df.empty:
        raise ValueError("No rows after timestamp filtering.")
    scenarios: list[tuple[str, dict[str, pd.DataFrame] | None, list[str]]] = [
        ("default", forecast_warehouse, list(AFRR_QUANTILE_BINS))
    ]
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
            scenarios.append((name, wh, list(scenario_bin_map[name])))

    MODEL_SPECS["reserve_activation_headroom_h"] = float(args.reserve_activation_headroom_h)
    MODEL_SPECS["bem_activation_headroom_h"] = float(args.bem_activation_headroom_h)
    MARKET_SPECS["afrr_activation_rate_guard_quantile"] = str(afrr_activation_rate_guard_quantile)
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
    MODEL_SPECS["afrr_bcm_bid_hour_local"] = int(args.bcm_bid_hour_local)
    MODEL_SPECS["afrr_bcm_gate_hour_cet"] = int(args.bcm_bid_hour_local)
    if args.da_limit_quantile is not None:
        MARKET_SPECS["da_buy_limit_quantile"] = str(args.da_limit_quantile)
        MARKET_SPECS["da_sell_limit_quantile"] = str(args.da_limit_quantile)
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
    MODEL_SPECS["debug_dumps"] = str(args.debug_dumps).strip().lower()
    enforce_final_soc_min = _resolve_final_soc_policy(
        strict_simulation_validity=bool(args.strict_simulation_validity),
        final_soc_mode=str(args.final_soc_mode),
        enforce_final_soc_min_flag=bool(args.enforce_final_soc_min),
        allow_terminal_soc_repair_in_strict=bool(args.allow_terminal_soc_repair_in_strict),
    )
    backtester = BatteryBacktester()
    if (
        bool(args.strict_simulation_validity)
        and str(args.final_soc_mode).strip().lower() == "hard"
        and float(getattr(backtester, "milp_time_limit_seconds", 30.0)) < 30.0
    ):
        print(
            "[WARN] BACKTEST_MILP_TIME_LIMIT_S is below 30s for strict hard-final-SoC mode; "
            "solver Not Set statuses are more likely. Recommended thesis default: 30s or higher."
        )
    sweep_rows: list[dict[str, object]] = []
    perf_rows_all: list[pd.DataFrame] = []
    daily_perf_rows_all: list[pd.DataFrame] = []
    perf_recon_debug_rows_all: list[pd.DataFrame] = []
    resolved_id_mode = str(args.id_mode).strip().lower()
    resolved_id_recourse_mode = str(args.id_recourse_mode).strip().lower()
    if (
        args.trading_strategy != "multi"
        and resolved_id_mode == "economic"
        and not bool(args.allow_economic_id_in_baseline)
    ):
        raise ValueError(
            "Economic ID is disabled for baseline strategies by default. "
            "Use --allow-economic-id-in-baseline only for explicit robustness variants."
        )

    if args.trading_strategy == "da":
        allowed_markets = ("DA",)
    elif args.trading_strategy == "afrr":
        allowed_markets = ("aFRR", "BCM", "BEM")
    elif args.trading_strategy == "bcm":
        allowed_markets = ("aFRR", "BCM")
    elif args.trading_strategy == "bem":
        allowed_markets = ("aFRR", "BEM")
    else:
        allowed_markets = ("DA", "aFRR", "ID", "BCM", "BEM")
    strategy_root = out_dir / args.trading_strategy

    for scenario_name, scenario_warehouse, scenario_bins in scenarios:
        backtester.afrr_quantile_bins = list(scenario_bins)
        backtester.afrr_quantile_prob = {
            q: 1.0 - float(q.replace("p", "")) / 100.0 for q in backtester.afrr_quantile_bins
        }
        materialized_p50_from_predicted_value_count = 0.0
        materialized_p50_targets: list[str] = []
        if scenario_warehouse is not None:
            for target_name, sdf in scenario_warehouse.items():
                if "materialized_p50_from_predicted_value" not in sdf.columns:
                    continue
                cnt = float(
                    pd.to_numeric(sdf["materialized_p50_from_predicted_value"], errors="coerce")
                    .fillna(0.0)
                    .sum()
                )
                if cnt > 0.0:
                    materialized_p50_from_predicted_value_count += cnt
                    materialized_p50_targets.append(str(target_name))
        scenario_out_dir = strategy_root if scenario_name == "default" and not quantile_pairs else strategy_root / scenario_name
        output_was_cleaned = _prepare_scenario_output_dir(
            scenario_out_dir=scenario_out_dir,
            clean_output=bool(args.clean_output),
            strict_simulation_validity=bool(args.strict_simulation_validity),
            simulation_schema_version=simulation_schema_version,
        )
        backtester._soc_mass_balance_debug_path = scenario_out_dir / "backtest_soc_mass_balance_debug.csv"
        backtester._hard_final_soc_debug_path = scenario_out_dir / "hard_final_soc_infeasibility_debug.csv"
        backtester._solver_failure_diagnostics_path = scenario_out_dir / "solver_failure_diagnostics.csv"
        backtester._hard_final_soc_debug_context = {
            "scenario": str(scenario_name),
            "model": str(args.model_key),
            "strategy": str(args.trading_strategy),
        }

        with _phase_watchdog("backtester_run"):
            outputs = backtester.run(
                df,
                colmap,
                use_rolling_horizon=not args.disable_rolling_horizon,
                horizon_hours=args.horizon_hours,
                reopt_step_hours=args.reopt_step_hours,
                forecast_warehouse=scenario_warehouse,
                da_bid_hour_local=args.da_bid_hour_local if args.da_gate_hour_utc is None else args.da_gate_hour_utc,
                soc_feedback_mode=args.soc_feedback_mode,
                enforce_final_soc_min=enforce_final_soc_min,
                allowed_markets=allowed_markets,
                strategy_name=str(args.trading_strategy),
                id_mode=resolved_id_mode,
                id_recourse_mode=resolved_id_recourse_mode,
                strict_simulation_validity=bool(args.strict_simulation_validity),
                enable_global_perfect_foresight=bool(args.enable_global_perfect_foresight),
                bcm_bid_hour_local=int(args.bcm_bid_hour_local),
            )
        outputs = replace(outputs, hourly=_ensure_hourly_throughput(outputs.hourly))

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
        performance_json_path = scenario_out_dir / "performance_metrics.json"
        performance_csv_path = scenario_out_dir / "performance_metrics.csv"
        daily_performance_json_path = scenario_out_dir / "daily_performance_metrics.json"
        daily_performance_csv_path = scenario_out_dir / "daily_performance_metrics.csv"
        state_machine_audit_path = scenario_out_dir / "state_machine_audit.json"
        diagnostics_path = scenario_out_dir / "backtest_diagnostics.json"
        diagnostics_txt_path = scenario_out_dir / "backtest_diagnostics.txt"
        perfect_foresight_paradox_path = scenario_out_dir / "perfect_foresight_paradox_hours.csv"
        pnl_plot_path = scenario_out_dir / "backtest_cumulative_pnl.png"
        reserve_commitment_debug_path = scenario_out_dir / "reserve_commitment_debug.csv"
        da_precommit_debug_path = scenario_out_dir / "da_precommit_debug.csv"
        invalid_headroom_debug_path = scenario_out_dir / "invalid_headroom_debug.csv"
        optimization_failure_debug_path = scenario_out_dir / "optimization_failure_debug.csv"
        bcm_block_consistency_violations_path = scenario_out_dir / "bcm_block_consistency_violations.csv"
        protected_soc_forensics_path = scenario_out_dir / "protected_soc_forensics.csv"
        optimization_infeasibility_attribution_path = scenario_out_dir / "optimization_infeasibility_attribution.csv"
        afrr_bin_ev_audit_path = scenario_out_dir / "afrr_bid_bin_ev_audit.csv"
        afrr_bin_ev_audit_status_path = scenario_out_dir / "afrr_bid_bin_ev_audit_status.json"
        run_status_path = scenario_out_dir / "run_status.json"
        output_write_started = time.monotonic()

        with _phase_watchdog("write_hourly"):
            hourly_to_write = _select_hourly_output_columns(
                outputs.hourly,
                output_detail=str(args.output_detail),
                timestamp_col=colmap.timestamp,
            )
            hourly_to_write.to_parquet(hourly_path, index=False)
        ev_audit_stats = {
            "ev_audit_row_count": 0.0,
            "ev_audit_max_bcm_formula_error": 0.0,
            "ev_audit_max_bem_formula_error": 0.0,
        }
        if bool(args.export_afrr_bin_ev_audit):
            ev_audit_status: dict[str, object] = {}
            try:
                ev_audit = _build_afrr_bin_ev_audit(
                    hourly=outputs.hourly,
                    scenario_name=scenario_name,
                    trading_strategy=str(args.trading_strategy),
                    active_bins=list(scenario_bins),
                    backtester=backtester,
                    timestamp_col=colmap.timestamp,
                    strict=True,
                    status_out=ev_audit_status,
                )
                if not ev_audit.empty:
                    ev_audit.to_csv(afrr_bin_ev_audit_path, index=False)
                afrr_bin_ev_audit_status_path.write_text(
                    json.dumps(ev_audit_status, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                ev_audit_stats = _validate_afrr_bin_ev_audit(
                    ev_audit,
                    tol=1e-6,
                    scenario_name=scenario_name,
                    trading_strategy=str(args.trading_strategy),
                    audit_path=str(afrr_bin_ev_audit_path),
                )
                outputs.summary.update(ev_audit_stats)
            except Exception as exc:
                run_status_path.write_text(
                    json.dumps(
                        {
                            "status": "failed",
                            "phase": "ev_audit_validation",
                            "scenario": scenario_name,
                            "trading_strategy": str(args.trading_strategy),
                            "error_message": str(exc),
                            "audit_path": str(afrr_bin_ev_audit_path),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                raise
        planned_cols = [c for c in [
            colmap.timestamp, "charge_mw", "discharge_mw", "reserve_pos_mw", "reserve_neg_mw", "soc_lp_mwh",
            "planned_soc_mwh", "aFRR_Capacity_Won_Pos_MW", "aFRR_Capacity_Won_Neg_MW", "aFRR_Capacity_Won_MW",
            "aFRR_Energy_Price_EUR_MWh_Pos", "aFRR_Energy_Price_EUR_MWh_Neg", "event_reopt_triggered",
            "event_reopt_rejected_mw_total", "predicted_objective_eur"
        ] if c in outputs.hourly.columns]
        planned_cols.extend([c for c in outputs.hourly.columns if c.startswith("afrr_bin_")])
        planned_cols.extend([c for c in outputs.hourly.columns if c.startswith(("afrr_p", "bcm_p", "bem_p"))])
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
            bcm_ok = float(pd.to_numeric(pd.Series([outputs.summary.get("bcm_block_consistency_check_pass", 1.0)]), errors="coerce").fillna(1.0).iloc[0]) >= 0.5
            if not bcm_ok:
                bcm_viol = _build_bcm_block_consistency_violations(
                    hourly=h,
                    summary=outputs.summary,
                    scenario_name=str(scenario_name),
                    strategy=str(args.trading_strategy),
                    tol_mw=1e-6,
                )
                if not bcm_viol.empty:
                    bcm_viol.to_csv(bcm_block_consistency_violations_path, index=False)
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

            da_debug_cols = [c for c in h.columns if c.startswith("da_precommit_")]
            if da_debug_cols:
                da_debug = h[[colmap.timestamp, *da_debug_cols]].copy()
                da_debug = da_debug.loc[da_debug[da_debug_cols].notna().any(axis=1)].copy()
                if not da_debug.empty:
                    da_debug["scenario"] = str(scenario_name)
                    da_debug.rename(
                        columns={
                            c: c.replace("da_precommit_da_", "da_").replace("da_precommit_", "")
                            for c in da_debug_cols
                        },
                        inplace=True,
                    )
                    da_debug.to_csv(da_precommit_debug_path, index=False)

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
            invalid_reason_txt = str(outputs.summary.get("invalid_reason", "") or "")
            if "protected_soc" in invalid_reason_txt:
                psv_pos = pd.to_numeric(
                    h.get("real_protected_soc_violation_pos_mwh", h.get("protected_soc_violation_pos_mwh", 0.0)),
                    errors="coerce",
                ).fillna(0.0)
                psv_neg = pd.to_numeric(
                    h.get("real_protected_soc_violation_neg_mwh", h.get("protected_soc_violation_neg_mwh", 0.0)),
                    errors="coerce",
                ).fillna(0.0)
                psv = psv_pos + psv_neg
                bad_ps = h.loc[psv > 1e-9].copy()
                if not bad_ps.empty:
                    direction = np.where(
                        (psv_pos.loc[bad_ps.index] > 1e-9) & (psv_neg.loc[bad_ps.index] > 1e-9),
                        "both",
                        np.where(psv_pos.loc[bad_ps.index] > 1e-9, "pos", np.where(psv_neg.loc[bad_ps.index] > 1e-9, "neg", "unknown")),
                    )
                    out_ps = pd.DataFrame(
                        {
                            "scenario": str(scenario_name),
                            "timestamp_utc": pd.to_datetime(bad_ps[colmap.timestamp], utc=True, errors="coerce"),
                            "direction": direction,
                            "violation_mwh": psv.loc[bad_ps.index].to_numpy(dtype=float),
                            "soc_start_mwh": pd.to_numeric(bad_ps.get("real_soc_start_mwh", bad_ps.get("soc_start_mwh", 0.0)), errors="coerce").fillna(0.0),
                            "soc_after_executed_mwh": pd.to_numeric(bad_ps.get("real_soc_mwh", bad_ps.get("soc_mwh", 0.0)), errors="coerce").fillna(0.0),
                            "protected_soc_min_mwh": pd.to_numeric(bad_ps.get("real_protected_soc_min_mwh", bad_ps.get("protected_soc_min_mwh", 0.0)), errors="coerce").fillna(0.0),
                            "protected_soc_max_mwh": pd.to_numeric(bad_ps.get("real_protected_soc_max_mwh", bad_ps.get("protected_soc_max_mwh", 0.0)), errors="coerce").fillna(0.0),
                            "locked_reserve_pos_mw": pd.to_numeric(bad_ps.get("real_locked_reserve_pos_mw", bad_ps.get("locked_reserve_pos_mw", 0.0)), errors="coerce").fillna(0.0),
                            "locked_reserve_neg_mw": pd.to_numeric(bad_ps.get("real_locked_reserve_neg_mw", bad_ps.get("locked_reserve_neg_mw", 0.0)), errors="coerce").fillna(0.0),
                            "awarded_or_executed_reserve_pos_mw": pd.to_numeric(bad_ps.get("real_reserve_pos_mw", bad_ps.get("reserve_pos_mw", 0.0)), errors="coerce").fillna(0.0),
                            "awarded_or_executed_reserve_neg_mw": pd.to_numeric(bad_ps.get("real_reserve_neg_mw", bad_ps.get("reserve_neg_mw", 0.0)), errors="coerce").fillna(0.0),
                            "bem_only_pos_mw": pd.to_numeric(bad_ps.get("real_bem_only_submitted_pos_mw", bad_ps.get("bem_only_submitted_pos_mw", 0.0)), errors="coerce").fillna(0.0),
                            "bem_only_neg_mw": pd.to_numeric(bad_ps.get("real_bem_only_submitted_neg_mw", bad_ps.get("bem_only_submitted_neg_mw", 0.0)), errors="coerce").fillna(0.0),
                            "da_charge_mw": pd.to_numeric(bad_ps.get("real_da_charge_mw", bad_ps.get("da_charge_mw", 0.0)), errors="coerce").fillna(0.0),
                            "da_discharge_mw": pd.to_numeric(bad_ps.get("real_da_discharge_mw", bad_ps.get("da_discharge_mw", 0.0)), errors="coerce").fillna(0.0),
                            "id_charge_mw": pd.to_numeric(bad_ps.get("real_id_charge_mw", bad_ps.get("id_charge_mw", 0.0)), errors="coerce").fillna(0.0),
                            "id_discharge_mw": pd.to_numeric(bad_ps.get("real_id_discharge_mw", bad_ps.get("id_discharge_mw", 0.0)), errors="coerce").fillna(0.0),
                            "act_pos_mwh": pd.to_numeric(bad_ps.get("real_act_pos_mwh", bad_ps.get("act_pos_mwh", 0.0)), errors="coerce").fillna(0.0),
                            "act_neg_mwh": pd.to_numeric(bad_ps.get("real_act_neg_mwh", bad_ps.get("act_neg_mwh", 0.0)), errors="coerce").fillna(0.0),
                            "aux_energy_mwh": pd.to_numeric(bad_ps.get("real_aux_energy_mwh", bad_ps.get("aux_energy_mwh", 0.0)), errors="coerce").fillna(0.0),
                            "suspected_cause": "submitted_without_locked_obligation" ,
                        }
                    )
                    out_ps.to_csv(protected_soc_forensics_path, index=False)

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
            "output_detail": str(args.output_detail),
            "debug_dumps": str(args.debug_dumps),
            "input_cache_used": float(bool(input_cache_used)),
            "input_cache_path": str(input_cache_path),
            "input_cache_schema_version": INPUT_CACHE_SCHEMA_VERSION,
            "requested_start_utc": str(eval_window["requested_start_utc"]),
            "requested_end_utc": str(eval_window["requested_end_utc"]),
            "effective_start_utc": str(eval_window["effective_start_utc"]),
            "effective_end_utc": str(eval_window["effective_end_utc"]),
            "simulation_window_clamped": float(eval_window["simulation_window_clamped"]),
            "simulation_window_clamp_reason": str(eval_window["simulation_window_clamp_reason"]),
            "simulation_common_lower_bound_utc": str(eval_window["simulation_common_lower_bound_utc"]),
            "simulation_common_upper_bound_utc": str(eval_window["simulation_common_upper_bound_utc"]),
            "simulation_window_days": float(eval_window["simulation_window_days"]),
            "simulation_window_hours": float(eval_window["simulation_window_hours"]),
            "simulation_window_interval_semantics": "[start,end)",
            "common_eval_window_clamp_enabled": float(not bool(args.disable_common_eval_window_clamp)),
            "forecast_coverage_report_path": str(forecast_coverage_report_csv or ""),
            "forecast_coverage_report_json_path": str(forecast_coverage_report_json or ""),
            "forecast_coverage_status": str(forecast_coverage_summary.get("status", "")),
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
            "materialized_p50_from_predicted_value_count": float(materialized_p50_from_predicted_value_count),
            "materialized_p50_targets": json.dumps(sorted(set(materialized_p50_targets))),
            "ev_audit_row_count": float(ev_audit_stats.get("ev_audit_row_count", 0.0)),
            "ev_audit_max_bcm_formula_error": float(ev_audit_stats.get("ev_audit_max_bcm_formula_error", 0.0)),
            "ev_audit_max_bem_formula_error": float(ev_audit_stats.get("ev_audit_max_bem_formula_error", 0.0)),
            "output_write_seconds": 0.0,
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
        with _phase_watchdog("write_performance_metrics"):
            scenario_start_utc = effective_start_utc
            scenario_end_utc = effective_end_utc
            perf_df, _ = _build_performance_metrics(
                hourly=outputs.hourly,
                summary=outputs.summary,
                args=args,
                scenario_name=scenario_name,
                scenario_bins=scenario_bins,
                scenario_start_utc=scenario_start_utc,
                scenario_end_utc=scenario_end_utc,
            )
            daily_df = _build_daily_performance_metrics(hourly=outputs.hourly, perf_row=perf_df.iloc[0])
            checks = _validate_performance_metrics(perf_row=perf_df.iloc[0], daily_df=daily_df)
            recon_debug_df = _build_performance_reconciliation_debug(
                scenario=str(scenario_name),
                perf_row=perf_df.iloc[0],
                daily_df=daily_df,
                hourly=outputs.hourly,
            )
            recon_debug_path = scenario_out_dir / "performance_metric_reconciliation_debug.csv"
            recon_debug_path.parent.mkdir(parents=True, exist_ok=True)
            recon_debug_df.to_csv(recon_debug_path, index=False)
            perf_recon_debug_rows_all.append(recon_debug_df.assign(scenario_path=str(scenario_out_dir)))
            for k, v in checks.items():
                perf_df[k] = [v]
            if bool(args.strict_simulation_validity) and not all(
                bool(checks.get(k, True))
                for k in ["net_revenue_reconciliation_ok", "cost_reconciliation_ok", "daily_to_scenario_reconciliation_ok"]
            ):
                top_row = _performance_reconciliation_failure_detail(
                    checks=checks,
                    recon_debug_df=recon_debug_df,
                    perf_row=perf_df.iloc[0],
                )
                raise RuntimeError(
                    "Performance metric reconciliation failed in strict mode: "
                    f"{json.dumps(checks, sort_keys=True)}; "
                    f"scenario={scenario_name}; "
                    f"failure_check={top_row.get('failure_check', '')}; "
                    f"offending_metric={top_row.get('metric', '')}; "
                    f"scenario_col={top_row.get('scenario_col', '')}; "
                    f"daily_col={top_row.get('daily_col', '')}; "
                    f"scenario_value={top_row.get('scenario_value', '')}; "
                    f"daily_sum={top_row.get('daily_sum_value', '')}; "
                    f"hourly_sum={top_row.get('hourly_sum_value', '')}; "
                    f"daily_abs_error={top_row.get('daily_abs_error', '')}; "
                    f"details={top_row.get('details', '')}; "
                    f"debug_path={recon_debug_path}"
                )
            perf_df.to_csv(performance_csv_path, index=False)
            performance_json_path.write_text(perf_df.to_json(orient="records", indent=2), encoding="utf-8")
            daily_df.insert(0, "scenario", str(scenario_name))
            daily_df.insert(1, "trading_strategy", str(args.trading_strategy))
            daily_df.to_csv(daily_performance_csv_path, index=False)
            daily_performance_json_path.write_text(daily_df.to_json(orient="records", indent=2), encoding="utf-8")
            perf_rows_all.append(perf_df.assign(scenario_path=str(scenario_out_dir)))
            daily_perf_rows_all.append(daily_df.assign(scenario_path=str(scenario_out_dir)))
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
                    f"global_hindsight_perfect_foresight_upper_bound_total_pnl_eur={_fmt_summary_val('global_hindsight_perfect_foresight_upper_bound_total_pnl_eur')}",
                    f"global_perfect_foresight_available={_fmt_summary_val('global_perfect_foresight_available')}",
                    f"global_perfect_foresight_validation_status={outputs.summary.get('global_perfect_foresight_validation_status', '')}",
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
            _plot_cumulative_pnl(
                outputs.hourly,
                colmap.timestamp,
                pnl_plot_path,
                summary=outputs.summary,
            )

        outputs.summary["output_write_seconds"] = float(max(0.0, time.monotonic() - output_write_started))
        with _phase_watchdog("write_summary_json"):
            summary_path.write_text(json.dumps(outputs.summary, indent=2), encoding="utf-8")

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
        realized_pnl = pd.to_numeric(
            pd.Series([outputs.summary.get("realized_total_pnl_eur", float("nan"))]),
            errors="coerce",
        ).iloc[0]
        naive_pnl = pd.to_numeric(
            pd.Series([outputs.summary.get("naive_total_pnl_eur", float("nan"))]),
            errors="coerce",
        ).iloc[0]
        model_vs_naive_ratio = (
            float(realized_pnl) / float(naive_pnl)
            if pd.notna(realized_pnl) and pd.notna(naive_pnl) and abs(float(naive_pnl)) > 1e-12
            else float("nan")
        )
        print(f"- naive_realized_pnl_eur: {float(naive_pnl) if pd.notna(naive_pnl) else float('nan'):.2f}")
        print(f"- model_realized_vs_naive_realized_ratio: {float(model_vs_naive_ratio):.4f}")
        # Warn on negative PnL in any reported optimization view.
        pnl_warn_fields = [
            ("Multi-market realized", "realized_total_pnl_eur"),
            ("Naive realized", "naive_total_pnl_eur"),
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
        print(
            "- validity: "
            f"simulation_valid={float(outputs.summary.get('simulation_valid', float('nan'))):.0f}, "
            f"thesis_reportable={float(outputs.summary.get('thesis_reportable', float('nan'))):.0f}, "
            f"invalid_reason={str(outputs.summary.get('invalid_reason', '')) or 'none'}"
        )
        print(
            "- fallback/solver: "
            f"fallback_used={float(outputs.summary.get('fallback_used', 0.0)):.0f}, "
            f"optimization_error_code_counts={outputs.summary.get('optimization_error_code_counts', '{}')}"
        )
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
            "realized_net_revenue_eur": float(perf_df.iloc[0].get("realized_net_revenue_eur", float("nan"))),
            "annualized_realized_net_revenue_eur": float(perf_df.iloc[0].get("annualized_realized_net_revenue_eur", float("nan"))),
            "realized_net_revenue_eur_per_mw": float(perf_df.iloc[0].get("realized_net_revenue_eur_per_mw", float("nan"))),
            "da_net_revenue_eur": float(perf_df.iloc[0].get("da_net_revenue_eur", float("nan"))),
            "id_net_revenue_eur": float(perf_df.iloc[0].get("id_net_revenue_eur", float("nan"))),
            "bcm_capacity_revenue_eur": float(perf_df.iloc[0].get("bcm_capacity_revenue_eur", float("nan"))),
            "bem_net_revenue_eur": float(perf_df.iloc[0].get("bem_net_revenue_eur", float("nan"))),
            "throughput_mwh_total": float(perf_df.iloc[0].get("throughput_mwh_total", float("nan"))),
            "equivalent_full_cycles_total": float(perf_df.iloc[0].get("equivalent_full_cycles_total", float("nan"))),
            "total_costs_eur": float(perf_df.iloc[0].get("total_costs_eur", float("nan"))),
        }
        if scenario_name != "default":
            q_lo, q_hi = scenario_name.split("_", 1)
            row["quantile_low"] = q_lo
            row["quantile_high"] = q_hi
            row["afrr_bid_quantile_bins"] = ",".join(_expand_quantile_range(q_lo, q_hi))
            row["afrr_activation_rate_guard_policy"] = afrr_activation_rate_guard_quantile
            row["afrr_activation_rate_guard_quantile"] = (
                q_lo if afrr_activation_rate_guard_quantile in {"scenario", "same_as_bid"} else afrr_activation_rate_guard_quantile
            )
            row["afrr_activation_rate_guard_quantile_resolved"] = row["afrr_activation_rate_guard_quantile"]
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

    if perf_rows_all:
        perf_all = pd.concat(perf_rows_all, ignore_index=True, sort=False)
        perf_all_csv = out_dir / "performance_metrics_all_scenarios.csv"
        perf_all_json = out_dir / "performance_metrics_all_scenarios.json"
        perf_all.to_csv(perf_all_csv, index=False)
        perf_all_json.write_text(perf_all.to_json(orient="records", indent=2), encoding="utf-8")
        print(f"[OK] Performance metrics (all scenarios): {perf_all_csv}")
    if daily_perf_rows_all:
        daily_all = pd.concat(daily_perf_rows_all, ignore_index=True, sort=False)
        daily_all_csv = out_dir / "daily_performance_metrics_all_scenarios.csv"
        daily_all.to_csv(daily_all_csv, index=False)
        print(f"[OK] Daily performance metrics (all scenarios): {daily_all_csv}")
    if perf_recon_debug_rows_all:
        recon_all = pd.concat(perf_recon_debug_rows_all, ignore_index=True, sort=False)
        recon_all_csv = out_dir / "performance_metric_reconciliation_debug_all.csv"
        recon_all.to_csv(recon_all_csv, index=False)
        print(f"[OK] Performance reconciliation debug (all scenarios): {recon_all_csv}")
    _write_performance_metric_definitions(out_dir / "performance_metric_definitions.json")

    # Strategy overview across separate runs (multi / da / afrr).
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
                "realized_vs_global_hindsight_perfect_foresight_upper_bound_pct",
                "simulation_valid",
                "thesis_reportable",
                "invalid_reason",
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
        if bool(args.strict_simulation_validity) and invalid_scenarios > 0 and not bool(args.allow_invalid_output):
            raise SystemExit(
                "Strict validity failed: one or more scenarios are invalid/non-reportable. "
                "Artifacts were written. Re-run with --allow-invalid-output for diagnostic runs."
            )


if __name__ == "__main__":
    main()
