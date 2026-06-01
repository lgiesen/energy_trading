"""LP-based battery backtesting for DA + aFRR value stacking.

Causality guard:
- Dispatch is optimized only on prediction columns.
- Financial settlement is computed only on ground-truth columns.

This ensures forecast-time decisions stay causally valid while PnL reflects
realized market outcomes.

Programmatic usage:
    from energy_trading.simulation.battery_backtest import (
        BatteryBacktester, BacktestColumnMap, load_and_align_market_data
    )

    colmap = BacktestColumnMap()
    df = load_and_align_market_data(predictions_path, ground_truth_path, colmap)
    out = BatteryBacktester().run(
        df,
        colmap,
        use_rolling_horizon=True,
        horizon_hours=48,
        reopt_step_hours=1,
        da_gate_hour_cet=11,
        soc_feedback_mode="realized",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
import json
import os
import signal
import threading
import time
import logging
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from energy_trading.config import BATTERY_SPECS, FINANCIAL_PARAMS, MARKET_SPECS, MODEL_SPECS
from energy_trading.evaluation.forecast_postprocessing import (
    canonicalize_prediction_frame,
    canonicalize_truth_series,
)
from energy_trading.simulation.bid_builder import AFRRCapacityBid, BidBuilder, BidPricingPolicy
from energy_trading.simulation.market_clearing import AFRRCapacityClearingResult, MarketClearingEngine


@dataclass(frozen=True)
class BacktestColumnMap:
    """Column mapping for prediction and ground-truth inputs."""

    timestamp: str = "timestamp_utc"

    pred_da_price: str = "pred_da_price"
    pred_afrr_capacity_price_pos: str = "pred_afrr_capacity_price_pos"
    pred_afrr_capacity_price_neg: str = "pred_afrr_capacity_price_neg"
    pred_afrr_activation_price_pos: str = "pred_afrr_activation_price_pos"
    pred_afrr_activation_price_neg: str = "pred_afrr_activation_price_neg"
    pred_afrr_activation_rate_pos: str = "pred_afrr_activation_rate_pos"
    pred_afrr_activation_rate_neg: str = "pred_afrr_activation_rate_neg"

    true_da_price: str = "da_price"
    true_afrr_capacity_price_pos: str = "afrr_capacity_price_pos"
    true_afrr_capacity_price_neg: str = "afrr_capacity_price_neg"
    true_afrr_activation_price_pos: str = "afrr_activation_price_vwap_pos"
    true_afrr_activation_price_neg: str = "afrr_activation_price_vwap_neg"
    true_afrr_activation_rate_pos: str = "activation_rate_phys_pos"
    true_afrr_activation_rate_neg: str = "activation_rate_phys_neg"


@dataclass(frozen=True)
class BacktestOutputs:
    """Container for full simulation outputs."""

    hourly: pd.DataFrame
    monthly: pd.DataFrame
    yearly: pd.DataFrame
    plan_history: pd.DataFrame
    volatility: pd.DataFrame
    summary: dict[str, float]
    isolated_hourly: dict[str, pd.DataFrame] | None = None


@dataclass(frozen=True)
class StrategyPermissions:
    allow_da: bool
    id_mode: str
    allow_bcm: bool
    allow_bcm_activation_obligations: bool
    allow_bem_only: bool

    @property
    def allow_id(self) -> bool:
        return self.id_mode in {"technical_repair", "economic"}

    @property
    def allow_id_technical_repair(self) -> bool:
        return self.id_mode in {"technical_repair", "economic"}

    @property
    def allow_id_economic(self) -> bool:
        return self.id_mode == "economic"


ID_RECOURSE_MODES = {"common", "disabled", "afrr_obligation_only"}


CANONICAL_PREDICTION_COLUMNS = [
    "pred_da_price",
    "pred_afrr_capacity_price_pos",
    "pred_afrr_capacity_price_neg",
    "pred_afrr_activation_price_pos",
    "pred_afrr_activation_price_neg",
    "pred_afrr_activation_rate_pos",
    "pred_afrr_activation_rate_neg",
]
# Full quantile surface used by model exports.
QUANTILE_COLUMNS = ["p01", "p05", "p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90", "p95", "p99"]
# Dynamic quantile bidding bins (ordered from most aggressive to most conservative).
AFRR_QUANTILE_BINS = ["p01", "p05", "p10", "p30", "p50", "p70", "p90", "p95", "p99"]

# Central schema registry for settlement validation.
# These are logical keys resolved via BacktestColumnMap at runtime.
CRITICAL_PRED_COL_KEYS = (
    "pred_da_price",
    "pred_afrr_capacity_price_pos",
    "pred_afrr_capacity_price_neg",
    "pred_afrr_activation_price_pos",
    "pred_afrr_activation_price_neg",
    "pred_afrr_activation_rate_pos",
    "pred_afrr_activation_rate_neg",
)

CRITICAL_TRUE_COL_KEYS = (
    "true_da_price",
    "true_afrr_capacity_price_pos",
    "true_afrr_capacity_price_neg",
    "true_afrr_activation_price_pos",
    "true_afrr_activation_price_neg",
    "true_afrr_activation_rate_pos",
    "true_afrr_activation_rate_neg",
)

# Dispatch/settlement schema registry.
DISPATCH_DECISION_COLS = (
    "plan_charge_mw",
    "plan_discharge_mw",
    "plan_reserve_pos_mw",
    "plan_reserve_neg_mw",
    "plan_bem_only_pos_mw",
    "plan_bem_only_neg_mw",
    "id_charge_mw",
    "id_discharge_mw",
    "pending_id_charge_mw",
    "pending_id_discharge_mw",
    "is_precleared",
    "aFRR_Capacity_Won_Pos_MW",
    "aFRR_Capacity_Won_Neg_MW",
    "aFRR_Capacity_Won_MW",
    "aFRR_Energy_Price_EUR_MWh_Pos",
    "aFRR_Energy_Price_EUR_MWh_Neg",
    "soc_before_mwh",
    "soc_after_planned_mwh",
    "soc_after_executed_mwh",
)

DISPATCH_METADATA_COLS = (
    "shock_source",
    "da_buy_reason",
    "da_sell_reason",
)

SETTLEMENT_NUMERIC_COLS = (
    "plan_charge_mw",
    "plan_discharge_mw",
    "plan_reserve_pos_mw",
    "plan_reserve_neg_mw",
    "plan_bem_only_pos_mw",
    "plan_bem_only_neg_mw",
    "submitted_da_buy_mw",
    "submitted_da_sell_mw",
    "submitted_da_buy_price_eur_mwh",
    "submitted_da_sell_price_eur_mwh",
    "submitted_afrr_pos_mw",
    "submitted_afrr_neg_mw",
    "desired_bem_only_pos_mw",
    "desired_bem_only_neg_mw",
    "safe_bem_only_pos_mw",
    "safe_bem_only_neg_mw",
    "bem_only_submitted_pos_mw_before_guard",
    "bem_only_submitted_neg_mw_before_guard",
    "bem_only_submitted_pos_mw_after_guard",
    "bem_only_submitted_neg_mw_after_guard",
    "bem_only_submitted_pos_mw",
    "bem_only_submitted_neg_mw",
    "bem_only_pos_reduced_by_headroom_mw",
    "bem_only_neg_reduced_by_headroom_mw",
    "bem_only_headroom_guard_applied",
    "bem_only_guard_soc_now_mwh",
    "bem_only_guard_protected_soc_min_mwh",
    "bem_only_guard_protected_soc_max_mwh",
    "bem_only_protected_soc_min_mwh",
    "bem_only_protected_soc_max_mwh",
    "bem_only_soc_start_mwh",
    "bem_only_pos_neg_exclusivity_applied",
    "bem_only_disabled_by_config",
    "max_bem_only_bid_mw",
    "executed_charge_mw",
    "executed_discharge_mw",
    "executed_reserve_pos_mw",
    "executed_reserve_neg_mw",
    "bem_only_executed_pos_mw",
    "bem_only_executed_neg_mw",
    "bem_only_executed_pos_mwh",
    "bem_only_executed_neg_mwh",
    "settlement_cap_bid_price_pos_eur_mw",
    "settlement_cap_bid_price_neg_eur_mw",
    "executed_rate_pos",
    "executed_rate_neg",
    "da_buy_accepted",
    "da_sell_accepted",
    "afrr_cap_pos_awarded",
    "afrr_cap_neg_awarded",
    "afrr_act_pos_accepted",
    "afrr_act_neg_accepted",
    "da_price_taker_mode",
    "id_economic_enabled",
    "id_technical_repair_enabled",
    "id_allowed",
    "id_buy_price_eur_mwh",
    "id_sell_price_eur_mwh",
    "id_net_mwh",
    "id_net_pnl_eur",
    "id_repair_mwh",
    "id_repair_cost_eur",
    "id_economic_mwh",
    "id_economic_pnl_eur",
    "id_technical_repair_pnl_eur",
    "pnl_id_eur",
    "aFRR_Capacity_Won_MW",
    "DA_Energy_Sold_MW",
    "aFRR_Energy_Price_EUR_MWh",
    "Obligation_Fulfilled",
    "aFRR_Energy_Gate_Closure_Min",
)

SETTLEMENT_METADATA_COLS = (
    "da_buy_reason",
    "da_sell_reason",
    "bem_only_headroom_guard_reason",
    "id_mode",
    "id_recourse_mode",
    "id_trade_type",
    "id_repair_reason",
    "id_recourse_reason",
    "pending_id_recourse_reason",
)

SETTLEMENT_CLEARING_COLS = SETTLEMENT_NUMERIC_COLS + SETTLEMENT_METADATA_COLS

SETTLEMENT_RENAME_MAP_REAL = {
    # Canonicalize realized execution primitives into stable physical names.
    "real_executed_charge_mw": "real_charge_mw",
    "real_executed_discharge_mw": "real_discharge_mw",
    "real_executed_reserve_pos_mw": "real_reserve_pos_mw",
    "real_executed_reserve_neg_mw": "real_reserve_neg_mw",
    "aux_power_mw": "real_aux_power_mw",
    "aux_energy_mwh": "real_aux_energy_mwh",
    "aux_state": "real_aux_state",
    "soc_for_capacity_mwh": "real_soc_for_capacity_mwh",
}


class PhaseTimeoutError(RuntimeError):
    """Raised when a timed backtest phase exceeds configured timeout."""


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


def _coalesce_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of the candidate columns exist: {list(candidates)}")
    return None


def load_and_align_market_data(
    predictions_path: str | Path,
    ground_truth_path: str | Path,
    colmap: BacktestColumnMap,
    *,
    target_value_modes: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load prediction and truth parquet files, align them by timestamp."""
    pred = pd.read_parquet(predictions_path)
    truth = pd.read_parquet(ground_truth_path)

    for name, frame in (("predictions", pred), ("ground_truth", truth)):
        if colmap.timestamp not in frame.columns:
            if isinstance(frame.index, pd.DatetimeIndex):
                frame[colmap.timestamp] = frame.index
            else:
                raise KeyError(f"{name} missing timestamp column '{colmap.timestamp}'")
        frame[colmap.timestamp] = pd.to_datetime(frame[colmap.timestamp], utc=True, errors="coerce")
        frame.dropna(subset=[colmap.timestamp], inplace=True)

    pred = pred.drop_duplicates(subset=[colmap.timestamp]).copy()
    truth = truth.drop_duplicates(subset=[colmap.timestamp]).copy()

    merged = pred.merge(truth, on=colmap.timestamp, how="inner", suffixes=("", "_gt"))
    merged = merged.sort_values(colmap.timestamp).reset_index(drop=True)
    if merged.empty:
        raise ValueError("No overlapping timestamps between predictions and ground truth.")
    return canonicalize_market_frame(merged, colmap=colmap, target_value_modes=target_value_modes)


def load_prediction_warehouse_long(
    prediction_files: dict[str, str | Path],
    *,
    target_value_modes: dict[str, str] | None = None,
    allow_p50_materialization_from_predicted_value: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load long-format forecast warehouse files keyed by canonical prediction column."""
    warehouse: dict[str, pd.DataFrame] = {}
    for pred_col, path in prediction_files.items():
        if pred_col not in CANONICAL_PREDICTION_COLUMNS:
            continue
        df = pd.read_parquet(path)
        required = {"snapshot_time_utc", "target_time_utc", "lead_time_h", "predicted_value"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Long prediction file for {pred_col} is missing columns: {sorted(missing)}")
        available_quantiles = [c for c in QUANTILE_COLUMNS if c in df.columns]
        out = df[[*list(required), *available_quantiles]].copy()
        out["snapshot_time_utc"] = pd.to_datetime(out["snapshot_time_utc"], utc=True, errors="coerce")
        out["target_time_utc"] = pd.to_datetime(out["target_time_utc"], utc=True, errors="coerce")
        out["lead_time_h"] = pd.to_numeric(out["lead_time_h"], errors="coerce").astype("Int64")
        out["predicted_value"] = pd.to_numeric(out["predicted_value"], errors="coerce")
        for qc in available_quantiles:
            out[qc] = pd.to_numeric(out[qc], errors="coerce")
        materialized_p50 = False
        if "p50" not in out.columns:
            if not bool(allow_p50_materialization_from_predicted_value):
                raise KeyError(
                    f"Long prediction file for {pred_col} is missing required quantile column 'p50'. "
                    "Enable explicit compatibility mode allow_p50_materialization_from_predicted_value "
                    "to map predicted_value -> p50."
                )
            out["p50"] = pd.to_numeric(out["predicted_value"], errors="coerce")
            materialized_p50 = True
            available_quantiles = [c for c in QUANTILE_COLUMNS if c in out.columns]
        out, _ = canonicalize_prediction_frame(
            out,
            target_name=pred_col,
            quantile_cols=available_quantiles,
            predicted_value_col="predicted_value",
            target_value_mode=(target_value_modes or {}).get(pred_col),
        )
        out["materialized_p50_from_predicted_value"] = 1.0 if materialized_p50 else 0.0
        out = out.dropna(subset=["snapshot_time_utc", "target_time_utc", "lead_time_h"]).copy()
        out = out.sort_values(["snapshot_time_utc", "lead_time_h", "target_time_utc"]).reset_index(drop=True)
        warehouse[pred_col] = out
    if not warehouse:
        raise ValueError("No valid long-format prediction files loaded.")
    return warehouse


def canonicalize_market_frame(
    df: pd.DataFrame, *, colmap: BacktestColumnMap, target_value_modes: dict[str, str] | None = None
) -> pd.DataFrame:
    out = df.copy()
    target_map = {
        "pred_da_price": (colmap.pred_da_price, colmap.true_da_price),
        "pred_afrr_capacity_price_pos": (colmap.pred_afrr_capacity_price_pos, colmap.true_afrr_capacity_price_pos),
        "pred_afrr_capacity_price_neg": (colmap.pred_afrr_capacity_price_neg, colmap.true_afrr_capacity_price_neg),
        "pred_afrr_activation_price_pos": (colmap.pred_afrr_activation_price_pos, colmap.true_afrr_activation_price_pos),
        "pred_afrr_activation_price_neg": (colmap.pred_afrr_activation_price_neg, colmap.true_afrr_activation_price_neg),
        "pred_afrr_activation_rate_pos": (colmap.pred_afrr_activation_rate_pos, colmap.true_afrr_activation_rate_pos),
        "pred_afrr_activation_rate_neg": (colmap.pred_afrr_activation_rate_neg, colmap.true_afrr_activation_rate_neg),
    }
    for target_name, (pred_col, true_col) in target_map.items():
        if pred_col in out.columns:
            out[pred_col] = canonicalize_truth_series(
                out[pred_col],
                target_name=target_name,
                target_value_mode=(target_value_modes or {}).get(target_name),
            )
        q_cols = [f"{pred_col}_{q}" for q in QUANTILE_COLUMNS if f"{pred_col}_{q}" in out.columns]
        if q_cols:
            temp = out[q_cols].copy()
            temp.columns = [c.rsplit("_", 1)[-1] for c in q_cols]
            temp, _ = canonicalize_prediction_frame(
                temp,
                target_name=target_name,
                quantile_cols=list(temp.columns),
                predicted_value_col="predicted_value",
                target_value_mode=(target_value_modes or {}).get(target_name),
            )
            for src_col, q_col in zip(q_cols, temp.columns, strict=False):
                out[src_col] = pd.to_numeric(temp[q_col], errors="coerce")
        if true_col in out.columns:
            out[true_col] = canonicalize_truth_series(
                out[true_col],
                target_name=target_name,
                target_value_mode=(target_value_modes or {}).get(target_name),
            )
    return out


class BatteryBacktester:
    """Linear-programming dispatch optimizer + realized settlement engine."""

    def __init__(self) -> None:
        dt = float(MODEL_SPECS["time_step_hours"])
        self.dt_h = dt

        self.cap_mwh = float(BATTERY_SPECS["capacity_mwh"])
        self.p_max_mw = float(BATTERY_SPECS["power_mw"])
        self.eta_in = float(BATTERY_SPECS["efficiency_in"])
        self.eta_out = float(BATTERY_SPECS["efficiency_out"])

        self.soc_min = float(BATTERY_SPECS["soc_min"]) * self.cap_mwh
        self.soc_max = float(BATTERY_SPECS["soc_max"]) * self.cap_mwh
        self.soc_init = float(BATTERY_SPECS["initial_soc"]) * self.cap_mwh
        self.soc_target_end = float(BATTERY_SPECS["soc_target_end"]) * self.cap_mwh

        self.aux_mode = str(BATTERY_SPECS.get("aux_power_mode", "state_dependent")).strip().lower()
        self.aux_peak_mw = float(BATTERY_SPECS.get("aux_power_peak_mw", BATTERY_SPECS.get("aux_power_mw", 0.0)))
        self.aux_off_mw = max(0.0, self.aux_peak_mw * float(BATTERY_SPECS.get("aux_power_off_duty", 0.0)))
        self.aux_standby_mw = max(0.0, self.aux_peak_mw * float(BATTERY_SPECS.get("aux_power_standby_duty", 0.25)))
        self.aux_trading_mw = max(0.0, self.aux_peak_mw * float(BATTERY_SPECS.get("aux_power_trading_duty", 0.75)))
        self.aux_afrr_active_mw = max(
            0.0, self.aux_peak_mw * float(BATTERY_SPECS.get("aux_power_afrr_active_duty", 1.0))
        )
        # Legacy constant auxiliary load energy-equivalent (used only in constant mode paths).
        self.aux_mwh = float(BATTERY_SPECS.get("aux_power_mw", 0.0)) * self.dt_h
        aux_override = os.environ.get("BACKTEST_AUX_POWER_MW_OVERRIDE", "").strip()
        if aux_override:
            try:
                aux_ovr = float(aux_override)
                self.aux_peak_mw = aux_ovr
                self.aux_off_mw = 0.0
                self.aux_standby_mw = 0.25 * aux_ovr
                self.aux_trading_mw = 0.75 * aux_ovr
                self.aux_afrr_active_mw = aux_ovr
                self.aux_mwh = aux_ovr * self.dt_h
                print(f"[INFO] Using aux power override: {float(aux_override):.6f} MW")
            except ValueError:
                pass
        self.deg_eur_mwh = float(BATTERY_SPECS["degradation_cost"])
        self.trans_eur_mwh = float(FINANCIAL_PARAMS["transaction_cost_eur_per_mwh"])
        # Fixed offer/availability reserve cost (EUR per offered MW per hour).
        # This is incurred when reserve is offered, independent of clearing.
        self.afrr_offer_cost_eur_mw_h = float(FINANCIAL_PARAMS.get("afrr_offer_cost_eur_mw_h", 0.0))
        self.initial_cash = float(FINANCIAL_PARAMS["initial_cash"])
        # Penalty used for non-delivery / imbalance settlement when awarded aFRR
        # cannot be physically delivered due to SoC constraints.
        self.imbalance_penalty_eur_mwh = float(FINANCIAL_PARAMS.get("imbalance_penalty_eur_mwh", 500.0))
        self.afrr_penalty_aufschlag_eur_mwh = float(FINANCIAL_PARAMS.get("afrr_penalty_aufschlag_eur_mwh", 30.0))
        self.afrr_penalty_aufschlag_eur_mw_h = float(FINANCIAL_PARAMS.get("afrr_penalty_aufschlag_eur_mw_h", 3.0))
        self.afrr_penalty_default_marginal_energy_price_eur_mwh = float(
            FINANCIAL_PARAMS.get("afrr_penalty_default_marginal_energy_price_eur_mwh", 150.0)
        )
        self.afrr_penalty_default_avg_capacity_price_product_eur_mw_h = float(
            FINANCIAL_PARAMS.get("afrr_penalty_default_avg_capacity_price_product_eur_mw_h", 12.5)
        )
        # Replacement for hard end-SoC floor: soft shortfall penalty in objective.
        self.final_soc_shortfall_penalty_eur_per_mwh = float(
            os.environ.get(
                "BACKTEST_FINAL_SOC_SHORTFALL_PENALTY_EUR_PER_MWH",
                FINANCIAL_PARAMS.get("final_soc_shortfall_penalty_eur_per_mwh", 100000.0),
            )
        )

        self.bid_power_max_mw = float(MARKET_SPECS.get("bid_power_max_mw", self.p_max_mw))
        self.reserve_max_mw = min(self.p_max_mw, self.bid_power_max_mw)
        self.reserve_product_duration_h = float(MODEL_SPECS.get("reserve_product_duration_h", 4.0))
        self.min_activation_headroom_fraction = float(
            MODEL_SPECS.get("min_activation_headroom_fraction", 0.25)
        )
        self.min_activation_headroom_fraction = max(0.0, min(1.0, self.min_activation_headroom_fraction))
        self.reserve_activation_headroom_h = max(0.0, float(MODEL_SPECS.get("reserve_activation_headroom_h", 0.5)))
        self.bem_activation_headroom_h = max(0.0, float(MODEL_SPECS.get("bem_activation_headroom_h", 0.5)))
        self.reserve_feasibility_mode = str(MODEL_SPECS.get("reserve_feasibility_mode", "normal")).strip().lower()
        self.reserve_headroom_safety_mwh = max(0.0, float(MODEL_SPECS.get("reserve_headroom_safety_mwh", 0.1)))
        self.reserve_soc_projection_safety_mwh = max(
            0.0, float(MODEL_SPECS.get("reserve_soc_projection_safety_mwh", 0.1))
        )
        self.reserve_power_safety_mw = max(0.0, float(MODEL_SPECS.get("reserve_power_safety_mw", 0.05)))
        self.reserve_min_margin_after_bid_mwh = max(
            0.0, float(MODEL_SPECS.get("reserve_min_margin_after_bid_mwh", 0.25))
        )
        self.reserve_bid_derate = float(MODEL_SPECS.get("reserve_bid_derate", 1.0))
        self.reserve_bid_derate = max(0.0, min(1.0, self.reserve_bid_derate))
        _max_reserve_bid_mw_cfg = MODEL_SPECS.get("max_reserve_bid_mw", None)
        self.max_reserve_bid_mw = (
            None if _max_reserve_bid_mw_cfg is None else max(0.0, float(_max_reserve_bid_mw_cfg))
        )
        self.bem_only_headroom_safety_mwh = max(
            0.0,
            float(MODEL_SPECS.get("bem_only_headroom_safety_mwh", 0.0)),
        )
        _max_bem_only_bid_mw_cfg = MODEL_SPECS.get("max_bem_only_bid_mw", None)
        self.max_bem_only_bid_mw = (
            None if _max_bem_only_bid_mw_cfg is None else max(0.0, float(_max_bem_only_bid_mw_cfg))
        )
        self.disable_bem_only = bool(MODEL_SPECS.get("disable_bem_only", False))
        self.disallow_simultaneous_bem_only_pos_neg = bool(
            MODEL_SPECS.get("disallow_simultaneous_bem_only_pos_neg", False)
        )
        self.disable_new_bcm_reserve_bids = bool(MODEL_SPECS.get("disable_new_bcm_reserve_bids", False))
        self.enable_reserve_retry_ladder = bool(MODEL_SPECS.get("enable_reserve_retry_ladder", False))
        self.reserve_retry_ladder = self._parse_reserve_retry_ladder(
            MODEL_SPECS.get("reserve_retry_ladder", "1.0,0.5,0.25,0.0")
        )
        self.final_soc_mode = str(MODEL_SPECS.get("final_soc_mode", "terminal_repair")).strip().lower()
        if self.final_soc_mode not in {"terminal_repair", "hard"}:
            self.final_soc_mode = "terminal_repair"
        # Single source of truth for aFRR BCM gate hour (CET/CEST local clock).
        # The perfect_foresight branch must use the same gate hour.
        self.afrr_bcm_gate_hour_cet = int(MODEL_SPECS.get("afrr_bcm_gate_hour_cet", 8))
        self.da_bid_granularity_mw = float(MARKET_SPECS.get("da_bid_granularity", 0.1))
        self.afrr_bid_granularity_mw = float(MARKET_SPECS.get("afrr_bid_granularity", 1.0))
        # Backward-compatible alias used by some helper paths.
        self.afrr_step_mw = self.afrr_bid_granularity_mw
        if self.da_bid_granularity_mw <= 0 or self.afrr_bid_granularity_mw <= 0:
            raise ValueError("Bid granularities must be > 0.")
        self.da_min_bid_size_mw = float(MARKET_SPECS.get("da_min_bid_size", self.da_bid_granularity_mw))
        self.afrr_min_bid_size_mw = float(MARKET_SPECS.get("afrr_min_bid_size", self.afrr_bid_granularity_mw))
        # Dynamic quantile bidding bins:
        # reserve decision bins are tied to model quantile outputs instead of static price levels.
        configured_bins = MODEL_SPECS.get("afrr_quantile_bins")
        if isinstance(configured_bins, (list, tuple)) and configured_bins:
            requested = [str(x).lower() for x in configured_bins]
            allowed = set(AFRR_QUANTILE_BINS)
            invalid = [q for q in requested if q not in allowed]
            if invalid:
                raise ValueError(
                    f"Invalid MODEL_SPECS['afrr_quantile_bins'] entries: {invalid}. Allowed: {AFRR_QUANTILE_BINS}"
                )
            self.afrr_quantile_bins = list(dict.fromkeys(requested))
        else:
            self.afrr_quantile_bins = list(AFRR_QUANTILE_BINS)
        self.afrr_quantile_prob = {q: 1.0 - float(q.replace("p", "")) / 100.0 for q in self.afrr_quantile_bins}
        self.da_execution_mode = str(MARKET_SPECS.get("da_execution_mode", "price_taker"))
        self.da_bid_fail_fast_debug = bool(MARKET_SPECS.get("da_bid_fail_fast_debug", False))
        self.da_link_to_awarded_afrr = bool(MARKET_SPECS.get("da_link_to_awarded_afrr", True))
        self.bid_pricing_policy = BidPricingPolicy(
            cap_risk_lambda=float(MARKET_SPECS.get("afrr_capacity_bid_risk_lambda", 0.2)),
            act_risk_lambda=float(MARKET_SPECS.get("afrr_activation_bid_risk_lambda", 0.2)),
            da_buy_limit_price_eur_mwh=float(MARKET_SPECS.get("da_buy_limit_price_eur_mwh", 3000.0)),
            da_sell_limit_price_eur_mwh=float(MARKET_SPECS.get("da_sell_limit_price_eur_mwh", -500.0)),
        )
        self.bid_builder = BidBuilder(
            pricing_policy=self.bid_pricing_policy,
            da_step_mw=self.da_bid_granularity_mw,
            afrr_step_mw=self.afrr_bid_granularity_mw,
            da_min_bid_size_mw=self.da_min_bid_size_mw,
            afrr_min_bid_size_mw=self.afrr_min_bid_size_mw,
            eta_in=self.eta_in,
            eta_out=self.eta_out,
            degradation_cost_eur_mwh=self.deg_eur_mwh,
            transaction_cost_eur_mwh=self.trans_eur_mwh,
            da_mode=self.da_execution_mode,
            da_arb_mode=str(MARKET_SPECS.get("da_arbitrage_mode", "limit")),
            da_buy_limit_offset_eur_mwh=float(MARKET_SPECS.get("da_buy_limit_offset_eur_mwh", 0.0)),
            da_sell_limit_offset_eur_mwh=float(MARKET_SPECS.get("da_sell_limit_offset_eur_mwh", 0.0)),
            da_buy_limit_quantile=str(MARKET_SPECS.get("da_buy_limit_quantile", "p90")),
            da_sell_limit_quantile=str(MARKET_SPECS.get("da_sell_limit_quantile", "p10")),
            afrr_energy_bid_strategy=str(MARKET_SPECS.get("afrr_energy_bid_strategy", "forecast")),
            link_da_to_awarded_afrr=self.da_link_to_awarded_afrr,
        )
        self.market_clearing_engine = MarketClearingEngine(
            da_mode_default=self.da_execution_mode,
        )
        self.id_rescue_spread_eur_mwh = float(MARKET_SPECS.get("id_rescue_spread_eur_mwh", 30.0))
        self.forecast_value_mode = str(MODEL_SPECS.get("forecast_value_mode", "canonical_economic")).strip().lower()
        if self.forecast_value_mode not in {"canonical_economic", "raw_signed"}:
            self.forecast_value_mode = "canonical_economic"
        self._neg_activation_sign_diagnostic_emitted = False
        self.id_buy_price_cap_eur_mwh = float(MARKET_SPECS.get("id_buy_price_cap_eur_mwh", 3000.0))
        self.id_sell_price_floor_eur_mwh = float(MARKET_SPECS.get("id_sell_price_floor_eur_mwh", -500.0))
        # MILP runtime controls:
        # - default execution is bounded and robust for rolling simulation
        # - diagnostic mode can be enabled to inspect HiGHS solver behavior
        #   (presolve, branching, degeneracy / collinearity indicators).
        self.milp_diagnostic_mode = os.environ.get("BACKTEST_MILP_DIAG", "0") == "1"
        self.milp_time_limit_seconds = float(os.environ.get("BACKTEST_MILP_TIME_LIMIT_S", "10.0"))
        self.milp_rel_gap = float(os.environ.get("BACKTEST_MILP_REL_GAP", "1e-4"))
        self.milp_diag_time_limit_seconds = float(os.environ.get("BACKTEST_MILP_DIAG_TIME_LIMIT_S", "60.0"))
        # Per-run infeasibility diagnostics written by _write_infeasible_debug_dump.
        self._infeasible_debug_dumps: list[dict[str, str]] = []
        self._strategy_permissions = StrategyPermissions(
            allow_da=True,
            id_mode="economic",
            allow_bcm=True,
            allow_bcm_activation_obligations=True,
            allow_bem_only=True,
        )
        self._id_recourse_mode = "common"

    @staticmethod
    def strategy_permissions_from_name(strategy: str) -> StrategyPermissions:
        s = str(strategy).strip().lower()
        if s == "multi":
            return StrategyPermissions(True, "economic", True, True, True)
        if s == "da_only":
            return StrategyPermissions(True, "none", False, False, False)
        if s == "afrr_only":
            return StrategyPermissions(False, "technical_repair", True, True, True)
        if s == "bcm_only":
            return StrategyPermissions(False, "technical_repair", True, True, False)
        if s == "bem_only":
            return StrategyPermissions(False, "technical_repair", False, False, True)
        raise ValueError(f"Unknown trading strategy: {strategy}")

    @classmethod
    def strategy_permissions_from_allowed_markets(
        cls, allowed_markets: list[str] | tuple[str, ...] | set[str]
    ) -> StrategyPermissions:
        allowed = {str(m).strip().lower() for m in allowed_markets}
        allow_da = "da" in allowed
        allow_id = "id" in allowed
        if "afrr" not in allowed:
            allow_bcm = False
            allow_bem = False
        else:
            # Backward compatibility: ("aFRR",) implies both BCM and BEM are allowed.
            explicit_bcm = "bcm" in allowed
            explicit_bem = "bem" in allowed
            if explicit_bcm or explicit_bem:
                allow_bcm = explicit_bcm
                allow_bem = explicit_bem
            else:
                allow_bcm = True
                allow_bem = True
        return StrategyPermissions(
            allow_da=bool(allow_da),
            id_mode=("economic" if bool(allow_id) else "none"),
            allow_bcm=bool(allow_bcm),
            allow_bcm_activation_obligations=bool(allow_bcm),
            allow_bem_only=bool(allow_bem),
        )

    @staticmethod
    def _normalize_id_mode(id_mode: str | None) -> str:
        mode = str(id_mode or "").strip().lower()
        if not mode:
            return ""
        if mode not in {"none", "technical_repair", "economic"}:
            raise ValueError(f"Unknown id_mode '{id_mode}'. Expected one of: none, technical_repair, economic.")
        return mode

    @staticmethod
    def _normalize_id_recourse_mode(id_recourse_mode: str | None) -> str:
        mode = str(id_recourse_mode or "").strip().lower()
        if not mode:
            return "common"
        if mode not in ID_RECOURSE_MODES:
            raise ValueError(
                f"Unknown id_recourse_mode '{id_recourse_mode}'. Expected one of: common, disabled, afrr_obligation_only."
            )
        return mode

    @classmethod
    def _apply_id_recourse_policy(
        cls,
        *,
        strategy_name: str | None,
        base_permissions: StrategyPermissions,
        id_recourse_mode: str,
    ) -> StrategyPermissions:
        s = str(strategy_name or "").strip().lower()
        mode = cls._normalize_id_recourse_mode(id_recourse_mode)
        if mode == "disabled":
            id_mode = "none"
        elif mode == "common":
            id_mode = "technical_repair"
        else:  # afrr_obligation_only
            id_mode = "none" if s == "da_only" else "technical_repair"
        return StrategyPermissions(
            allow_da=base_permissions.allow_da,
            id_mode=id_mode,
            allow_bcm=base_permissions.allow_bcm,
            allow_bcm_activation_obligations=base_permissions.allow_bcm_activation_obligations,
            allow_bem_only=base_permissions.allow_bem_only,
        )

    @classmethod
    def resolve_strategy_permissions(
        cls,
        *,
        strategy_name: str | None,
        allowed_markets: list[str] | tuple[str, ...] | set[str],
        id_mode: str | None = None,
        id_recourse_mode: str | None = None,
    ) -> StrategyPermissions:
        perms = (
            cls.strategy_permissions_from_name(strategy_name)
            if strategy_name is not None
            else cls.strategy_permissions_from_allowed_markets(allowed_markets)
        )
        policy_mode = cls._normalize_id_recourse_mode(id_recourse_mode)
        perms = cls._apply_id_recourse_policy(
            strategy_name=strategy_name,
            base_permissions=perms,
            id_recourse_mode=policy_mode,
        )
        override = cls._normalize_id_mode(id_mode)
        if not override:
            return perms
        return StrategyPermissions(
            allow_da=perms.allow_da,
            id_mode=override,
            allow_bcm=perms.allow_bcm,
            allow_bcm_activation_obligations=perms.allow_bcm_activation_obligations,
            allow_bem_only=perms.allow_bem_only,
        )

    @staticmethod
    def _parse_reserve_retry_ladder(raw: object) -> list[float]:
        if isinstance(raw, (list, tuple)):
            vals = [float(x) for x in raw]
        else:
            txt = str(raw).strip()
            vals = [float(x.strip()) for x in txt.split(",") if x.strip()] if txt else []
        vals = [max(0.0, min(1.0, float(v))) for v in vals if np.isfinite(float(v))]
        if not vals:
            vals = [1.0]
        vals = list(dict.fromkeys(vals))
        if 1.0 not in vals:
            vals.insert(0, 1.0)
        return vals

    def _state_aux_power_mw(
        self,
        *,
        charge_mw: float,
        discharge_mw: float,
        reserve_pos_mw: float,
        reserve_neg_mw: float,
        act_pos_rate: float,
        act_neg_rate: float,
        id_charge_mw: float = 0.0,
        id_discharge_mw: float = 0.0,
    ) -> tuple[float, str]:
        """Return state-dependent auxiliary power and discrete operating mode."""
        if self.aux_mode != "state_dependent":
            return float(BATTERY_SPECS.get("aux_power_mw", 0.0)), "CONSTANT"
        eps = 1e-9
        reserve_committed = (reserve_pos_mw > eps) or (reserve_neg_mw > eps)
        activated = ((reserve_pos_mw > eps) and (act_pos_rate > eps)) or ((reserve_neg_mw > eps) and (act_neg_rate > eps))
        trading = (charge_mw > eps) or (discharge_mw > eps) or (id_charge_mw > eps) or (id_discharge_mw > eps)
        if activated:
            return self.aux_afrr_active_mw, "aFRR_ACTIVE"
        if trading:
            return self.aux_trading_mw, "TRADING"
        if reserve_committed:
            return self.aux_standby_mw, "STANDBY"
        return self.aux_off_mw, "OFF"

    @staticmethod
    def _clip_rate(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if np.isnan(arr).any():
            raise ValueError("NaN detected in activation-rate input. Fail-fast is enabled; fix upstream data mapping.")
        return np.clip(arr, 0.0, 1.0)

    def _validate_critical_data(
        self,
        frame: pd.DataFrame,
        colmap: BacktestColumnMap,
        *,
        run_mode: str = "advanced_ml",
        require_pacc_bins: bool = False,
    ) -> None:
        """Fail-fast gate for critical financial inputs before clearing/optimization."""
        mode = str(run_mode).strip().lower()
        if mode not in {"advanced_ml", "naive", "perfect_foresight"}:
            raise ValueError(f"Unknown run_mode '{run_mode}'. Expected one of: advanced_ml, naive, perfect_foresight.")

        if mode == "advanced_ml":
            critical_cols = [
                colmap.pred_da_price,
                colmap.pred_afrr_capacity_price_pos,
                colmap.pred_afrr_capacity_price_neg,
                colmap.pred_afrr_activation_price_pos,
                colmap.pred_afrr_activation_price_neg,
                colmap.pred_afrr_activation_rate_pos,
                colmap.pred_afrr_activation_rate_neg,
            ]
            critical_cols.extend(
                [
                    f"{colmap.pred_da_price}_p05",
                    f"{colmap.pred_da_price}_p10",
                    f"{colmap.pred_da_price}_p90",
                    f"{colmap.pred_da_price}_p95",
                ]
            )
        elif mode == "naive":
            critical_cols = [
                colmap.pred_da_price,
                colmap.pred_afrr_capacity_price_pos,
                colmap.pred_afrr_capacity_price_neg,
                colmap.pred_afrr_activation_price_pos,
                colmap.pred_afrr_activation_price_neg,
                colmap.pred_afrr_activation_rate_pos,
                colmap.pred_afrr_activation_rate_neg,
            ]
        else:  # perfect_foresight
            critical_cols = [
                colmap.true_da_price,
                colmap.true_afrr_capacity_price_pos,
                colmap.true_afrr_capacity_price_neg,
                colmap.true_afrr_activation_price_pos,
                colmap.true_afrr_activation_price_neg,
                colmap.true_afrr_activation_rate_pos,
                colmap.true_afrr_activation_rate_neg,
            ]

        # p_acc bins are critical once created for optimization.
        if require_pacc_bins:
            for b in range(len(self.afrr_quantile_bins)):
                critical_cols.append(f"pacc_pos_bin_{b}")
                critical_cols.append(f"pacc_neg_bin_{b}")

        missing_cols = [c for c in critical_cols if c not in frame.columns]
        if missing_cols:
            raise ValueError(
                "Critical data gate failed: missing required columns: " + ", ".join(missing_cols[:20])
            )

        bad_msgs: list[str] = []
        for c in critical_cols:
            s = pd.to_numeric(frame[c], errors="coerce")
            bad = s.isna() | ~np.isfinite(s.to_numpy(dtype=float))
            if bool(bad.any()):
                idx = frame.index[bad].tolist()[:5]
                bad_msgs.append(f"{c}: bad={int(bad.sum())}/{len(s)} sample_index={idx}")
        if bad_msgs:
            raise ValueError("Critical data gate failed due to null/non-finite values. " + " | ".join(bad_msgs[:20]))

    @staticmethod
    def _finite_numeric_series(
        frame: pd.DataFrame,
        primary_col: str,
        *,
        fallback_cols: list[str] | None = None,
        default: float = 0.0,
        allow_temporal_fill: bool = True,
        strict_non_null: bool = False,
    ) -> pd.Series:
        """Return a finite numeric series using ordered fallbacks and safe fills."""
        if primary_col in frame.columns:
            base = frame.loc[:, primary_col]
            # Duplicate column names can return a DataFrame; use first occurrence.
            if isinstance(base, pd.DataFrame):
                base = base.iloc[:, 0]
        else:
            base = pd.Series(np.nan, index=frame.index, dtype="float64")

        vals = pd.to_numeric(base, errors="coerce")
        if isinstance(vals, np.ndarray):
            vals = pd.Series(vals, index=frame.index)
        elif not isinstance(vals, pd.Series):
            vals = pd.Series([vals] * len(frame), index=frame.index)
        vals = vals.astype("float64")
        for c in (fallback_cols or []):
            if c in frame.columns:
                fb = frame.loc[:, c]
                if isinstance(fb, pd.DataFrame):
                    fb = fb.iloc[:, 0]
                fb = pd.to_numeric(fb, errors="coerce")
                if isinstance(fb, np.ndarray):
                    fb = pd.Series(fb, index=frame.index)
                elif not isinstance(fb, pd.Series):
                    fb = pd.Series([fb] * len(frame), index=frame.index)
                vals = vals.fillna(fb.astype("float64"))
        vals = vals.replace([np.inf, -np.inf], np.nan)
        if allow_temporal_fill:
            vals = vals.ffill().bfill().fillna(float(default))
        elif not strict_non_null:
            vals = vals.fillna(float(default))
        if strict_non_null and vals.isna().any():
            miss_n = int(vals.isna().sum())
            raise ValueError(
                f"NaN detected in critical optimizer input '{primary_col}' after fallback resolution "
                f"(missing={miss_n}/{len(vals)}). Fix data pipeline."
            )
        return vals

    def _normalize_da_bid(self, charge_mw: float, discharge_mw: float) -> tuple[float, float]:
        """Project DA bid pair to a single net direction to avoid binary lock conflicts."""
        ch = max(0.0, float(charge_mw))
        dis = max(0.0, float(discharge_mw))
        net = dis - ch
        step = self.da_bid_granularity_mw
        # Enforce DA market granularity on lockbook bids.
        def q(x: float) -> float:
            if x <= 0.0:
                return 0.0
            # Floor to the nearest valid step so we never exceed power limits.
            units = int(np.floor((x / step) + 1e-12))
            return min(self.p_max_mw, float(units * step))
        if net >= 0.0:
            return 0.0, q(min(self.p_max_mw, net))
        return q(min(self.p_max_mw, -net)), 0.0

    @staticmethod
    def _assert_valid_time_index(df: pd.DataFrame, timestamp_col: str) -> pd.Series:
        """Validate timestamp column for optimization/backtesting invariants."""
        if timestamp_col not in df.columns:
            raise KeyError(f"Missing required timestamp column: {timestamp_col}")
        ts = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
        if ts.isna().any():
            raise ValueError(f"Timestamp column '{timestamp_col}' contains invalid/NaT values.")
        if not ts.is_monotonic_increasing:
            raise ValueError(f"Timestamp column '{timestamp_col}' must be sorted ascending.")
        if ts.duplicated().any():
            raise ValueError(f"Timestamp column '{timestamp_col}' contains duplicates.")
        return ts

    def _variable_slices(self, n: int, n_bins: int) -> dict[str, slice]:
        # Hourly decisions:
        # - charge/discharge,
        # - bin-specific reserve offers for pos/neg activation,
        # - binary is_charging flag,
        # - soc state,
        # - soft-constraint slacks for reserve infeasibility,
        # - soft terminal SoC shortfall slack (replacement for hard final SoC floor).
        n_r = n * n_bins
        s = 0
        out: dict[str, slice] = {}
        out["ch"] = slice(s, s + n); s += n
        out["dis"] = slice(s, s + n); s += n
        out["rpos_bin"] = slice(s, s + n_r); s += n_r
        out["rneg_bin"] = slice(s, s + n_r); s += n_r
        out["bem_pos_bin"] = slice(s, s + n_r); s += n_r
        out["bem_neg_bin"] = slice(s, s + n_r); s += n_r
        out["u"] = slice(s, s + n); s += n
        out["soc"] = slice(s, s + (n + 1)); s += (n + 1)
        out["slack_pos"] = slice(s, s + n); s += n
        out["slack_neg"] = slice(s, s + n); s += n
        out["slack_final_soc"] = slice(s, s + 1); s += 1
        # Emergency physical SoC slacks (n+1 states, including terminal state).
        out["slack_soc_min"] = slice(s, s + (n + 1)); s += (n + 1)
        out["slack_soc_max"] = slice(s, s + (n + 1)); s += (n + 1)
        # Deliverability slacks (softened 4h reserve-energy/headroom constraints).
        out["slack_deliver_pos"] = slice(s, s + n); s += n
        out["slack_deliver_neg"] = slice(s, s + n); s += n
        # Obligation slacks (softened lockbook equality/feasibility constraints).
        out["slack_obligation_pos"] = slice(s, s + n); s += n
        out["slack_obligation_neg"] = slice(s, s + n); s += n
        return out

    def _guarded_merge(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        *,
        on: str = "timestamp_utc",
        how: str = "left",
        validate_row_count: bool = True,
    ) -> pd.DataFrame:
        """Defensive merge with timezone normalization, key-uniqueness and suffix guards."""
        if on not in left_df.columns or on not in right_df.columns:
            raise ValueError(f"_guarded_merge requires join key '{on}' in both frames.")

        left = left_df.copy()
        right = right_df.copy()
        left[on] = pd.to_datetime(left[on], utc=True, errors="coerce")
        right[on] = pd.to_datetime(right[on], utc=True, errors="coerce")
        left = left.dropna(subset=[on])
        right = right.dropna(subset=[on])

        if not pd.Index(left[on]).is_unique:
            raise ValueError(f"_guarded_merge left key '{on}' is not unique.")
        if not pd.Index(right[on]).is_unique:
            raise ValueError(f"_guarded_merge right key '{on}' is not unique.")

        expected_rows = len(left)
        merged = pd.merge(left, right, on=on, how=how)

        if validate_row_count and how == "left" and len(merged) != expected_rows:
            raise ValueError(
                f"_guarded_merge row count mismatch for left join! expected={expected_rows}, got={len(merged)}"
            )

        collision_cols = [c for c in merged.columns if c.endswith("_x") or c.endswith("_y")]
        if collision_cols:
            raise ValueError(
                "_guarded_merge detected overlapping columns (suffix collision): "
                + ", ".join(collision_cols[:20])
            )
        return merged

    def _milp_options(self) -> dict[str, float | bool]:
        """Build SciPy/HiGHS MILP options for stable rolling optimization."""
        if self.milp_diagnostic_mode:
            # Verbose diagnostics with longer cap for bottleneck analysis.
            return {
                "disp": True,
                "time_limit": max(1e-3, float(self.milp_diag_time_limit_seconds)),
                "mip_rel_gap": max(0.0, float(self.milp_rel_gap)),
            }
        return {
            "disp": False,
            "time_limit": max(1e-3, float(self.milp_time_limit_seconds)),
            "mip_rel_gap": max(0.0, float(self.milp_rel_gap)),
        }

    @staticmethod
    def _is_timeout_result(sol: object) -> bool:
        """Detect MILP timeout status across SciPy result formats/messages."""
        status = getattr(sol, "status", None)
        msg = str(getattr(sol, "message", "")).lower()
        # SciPy/HiGHS commonly uses status=1 for iteration/time limit.
        return status == 1 or ("time limit" in msg) or ("timelimit" in msg)

    def _write_infeasible_debug_dump(
        self,
        *,
        df: pd.DataFrame,
        colmap: BacktestColumnMap,
        c: np.ndarray,
        a_ub: np.ndarray | None,
        b_ub: np.ndarray | None,
        a_eq: np.ndarray | None,
        b_eq: np.ndarray | None,
        lb: np.ndarray,
        ub: np.ndarray,
        message: str,
        solve_context: dict[str, object] | None = None,
    ) -> None:
        """Persist MILP matrices for post-mortem infeasibility debugging."""
        try:
            out_dir = Path("artifacts/simulation_debug")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts0 = pd.to_datetime(df[colmap.timestamp].iloc[0], utc=True, errors="coerce")
            tag = ts0.strftime("%Y%m%dT%H%M%SZ") if pd.notna(ts0) else "unknown_ts"
            path = out_dir / f"infeasible_debug_{tag}.npz"
            np.savez_compressed(
                path,
                c=c,
                a_ub=(a_ub if a_ub is not None else np.empty((0, len(c)))),
                b_ub=(b_ub if b_ub is not None else np.empty((0,))),
                a_eq=(a_eq if a_eq is not None else np.empty((0, len(c)))),
                b_eq=(b_eq if b_eq is not None else np.empty((0,))),
                lb=lb,
                ub=ub,
                message=np.array([str(message)]),
                solve_context=np.array([json.dumps(solve_context or {}, sort_keys=True)]),
                variable_names=np.array([], dtype=str),
                constraint_names=np.array([], dtype=str),
            )
            self._infeasible_debug_dumps.append(
                {
                    "path": str(path),
                    "timestamp_utc": str(tag),
                    "solve_context": json.dumps(solve_context or {}, sort_keys=True),
                }
            )
            print(f"[DIAG] wrote infeasible debug dump: {path}")
        except Exception as exc:
            print(f"[WARN] failed to write infeasible debug dump: {exc}")

    @staticmethod
    def _classify_infeasible_debug_dumps(
        dumps: list[dict[str, str]],
        hourly: pd.DataFrame,
        *,
        timestamp_col: str,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Return (accepted_path_dumps, candidate_dumps)."""
        if not dumps:
            return [], []
        accepted_mask = pd.Series(False, index=hourly.index)
        if "real_optimization_error_code" in hourly.columns:
            ec = hourly["real_optimization_error_code"].fillna("ok").astype(str).str.strip().str.lower()
            accepted_mask |= ~ec.isin(["ok", "none", ""])
        elif "optimization_error_code" in hourly.columns:
            ec = hourly["optimization_error_code"].fillna("ok").astype(str).str.strip().str.lower()
            accepted_mask |= ~ec.isin(["ok", "none", ""])
        if "real_optimization_fallback" in hourly.columns:
            fb = hourly["real_optimization_fallback"].fillna("none").astype(str).str.strip().str.lower()
            accepted_mask |= fb.ne("none")
        elif "optimization_fallback" in hourly.columns:
            fb = hourly["optimization_fallback"].fillna("none").astype(str).str.strip().str.lower()
            accepted_mask |= fb.ne("none")
        if "optimizer_fallback_used" in hourly.columns:
            accepted_mask |= pd.to_numeric(hourly["optimizer_fallback_used"], errors="coerce").fillna(0.0).gt(0.5)
        accepted_ts = set(
            pd.to_datetime(hourly.loc[accepted_mask, timestamp_col], utc=True, errors="coerce")
            .dropna()
            .tolist()
        )
        accepted: list[dict[str, str]] = []
        candidate: list[dict[str, str]] = []
        for d in dumps:
            ctx_raw = str(d.get("solve_context", "")).strip()
            ctx_ts = ""
            if ctx_raw:
                try:
                    ctx_ts = str(json.loads(ctx_raw).get("timestamp_utc", "")).strip()
                except Exception:
                    ctx_ts = ""
            ts_raw = ctx_ts or str(d.get("timestamp_utc", "")).strip()
            ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
            if pd.notna(ts) and ts in accepted_ts:
                accepted.append(d)
            else:
                candidate.append(d)
        return accepted, candidate

    @staticmethod
    def _quantile_map_from_row(row: pd.Series) -> dict[float, float]:
        qmap: dict[float, float] = {}
        for q in range(10, 100, 10):
            col = f"p{q:02d}"
            if col not in row.index:
                continue
            val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.notna(val):
                qmap[q / 100.0] = float(val)
        return qmap

    def optimize_dispatch(
        self,
        df: pd.DataFrame,
        colmap: BacktestColumnMap,
        *,
        soc_start: float | None = None,
        soc_end_target: float | None = None,
        soc_end_min_target: float | None = None,
        fixed_da_dispatch: dict[pd.Timestamp, tuple[float, float]] | None = None,
        fixed_reserve_obligation: dict[pd.Timestamp, tuple[float, float]] | None = None,
        deterministic_reserve_settlement: bool = False,
        allowed_markets: list[str] | tuple[str, ...] | set[str] = ("DA", "aFRR"),
        strict_input_validation: bool = False,
    ) -> pd.DataFrame:
        """Solve LP on predicted market signals to obtain hourly dispatch.

        aFRR reserve offers are optimized over bid-price bins using expected value.
        The LP decouples:
        - p_acc_cap[j,t]: probability that capacity bid in bin j clears.
        - r_act[t]: expected activation-rate conditional on awarded capacity.
        """
        if df.empty:
            raise ValueError("optimize_dispatch received empty dataframe.")
        self._assert_valid_time_index(df, colmap.timestamp)
        n = len(df)
        n_bins = int(len(self.afrr_quantile_bins))
        if n_bins <= 0:
            raise ValueError("afrr_quantile_bins must contain at least one bid bin.")
        sl = self._variable_slices(n, n_bins=n_bins)
        n_vars = int(sl["slack_obligation_neg"].stop)
        if sl["ch"].start != 0 or sl["slack_neg"].stop <= 0:
            raise ValueError("Invalid variable slice definition for MILP.")
        perms = self.strategy_permissions_from_allowed_markets(allowed_markets)
        self._strategy_permissions = perms
        da_enabled = bool(perms.allow_da)
        afrr_enabled = bool(perms.allow_bcm or perms.allow_bem_only)

        p_da = self._finite_numeric_series(
            df,
            colmap.pred_da_price,
            fallback_cols=[],
            default=0.0,
            allow_temporal_fill=not strict_input_validation,
            strict_non_null=strict_input_validation,
        ).to_numpy(dtype=float)
        p_cap_pos = self._finite_numeric_series(
            df,
            colmap.pred_afrr_capacity_price_pos,
            fallback_cols=[colmap.pred_afrr_capacity_price_neg],
            default=0.0,
            allow_temporal_fill=not strict_input_validation,
            strict_non_null=strict_input_validation,
        ).to_numpy(dtype=float)
        p_cap_neg = self._finite_numeric_series(
            df,
            colmap.pred_afrr_capacity_price_neg,
            fallback_cols=[colmap.pred_afrr_capacity_price_pos],
            default=0.0,
            allow_temporal_fill=not strict_input_validation,
            strict_non_null=strict_input_validation,
        ).to_numpy(dtype=float)
        # Fallback acceptance-rate priors (used when no quantile-derived p_acc bins are available).
        r_act_pos_base = self._clip_rate(
            self._finite_numeric_series(
                df,
                colmap.pred_afrr_activation_rate_pos,
                fallback_cols=[colmap.pred_afrr_activation_rate_neg],
                default=0.0,
                allow_temporal_fill=not strict_input_validation,
                strict_non_null=strict_input_validation,
            ).to_numpy(dtype=float)
        )
        r_act_neg_base = self._clip_rate(
            self._finite_numeric_series(
                df,
                colmap.pred_afrr_activation_rate_neg,
                fallback_cols=[colmap.pred_afrr_activation_rate_pos],
                default=0.0,
                allow_temporal_fill=not strict_input_validation,
                strict_non_null=strict_input_validation,
            ).to_numpy(dtype=float)
        )
        # p90 activation-rate chance-constraint rates (fallback to point rates).
        r_act_pos_p90 = self._clip_rate(
            self._finite_numeric_series(
                df,
                f"{colmap.pred_afrr_activation_rate_pos}_p90",
                fallback_cols=[
                    colmap.pred_afrr_activation_rate_pos,
                    f"{colmap.pred_afrr_activation_rate_neg}_p90",
                    colmap.pred_afrr_activation_rate_neg,
                ],
                default=0.0,
                allow_temporal_fill=not strict_input_validation,
                strict_non_null=strict_input_validation,
            ).to_numpy(dtype=float)
        )
        r_act_neg_p90 = self._clip_rate(
            self._finite_numeric_series(
                df,
                f"{colmap.pred_afrr_activation_rate_neg}_p90",
                fallback_cols=[
                    colmap.pred_afrr_activation_rate_neg,
                    f"{colmap.pred_afrr_activation_rate_pos}_p90",
                    colmap.pred_afrr_activation_rate_pos,
                ],
                default=0.0,
                allow_temporal_fill=not strict_input_validation,
                strict_non_null=strict_input_validation,
            ).to_numpy(dtype=float)
        )
        act_price_pos = self._finite_numeric_series(
            df,
            colmap.pred_afrr_activation_price_pos,
            fallback_cols=[],
            default=0.0,
            allow_temporal_fill=not strict_input_validation,
            strict_non_null=strict_input_validation,
        ).to_numpy(dtype=float)
        act_price_neg = self._finite_numeric_series(
            df,
            colmap.pred_afrr_activation_price_neg,
            fallback_cols=[],
            default=0.0,
            allow_temporal_fill=not strict_input_validation,
            strict_non_null=strict_input_validation,
        ).to_numpy(dtype=float)

        act_price_pos_by_bin = np.zeros((n, n_bins), dtype=float)
        act_price_neg_by_bin = np.zeros((n, n_bins), dtype=float)
        act_rate_pos_by_bin = np.zeros((n, n_bins), dtype=float)
        act_rate_neg_by_bin = np.zeros((n, n_bins), dtype=float)
        missing_bin_cols: list[str] = []
        for b, qcol in enumerate(self.afrr_quantile_bins):
            c_app = f"{colmap.pred_afrr_activation_price_pos}_{qcol}"
            c_apn = f"{colmap.pred_afrr_activation_price_neg}_{qcol}"
            c_arp = f"{colmap.pred_afrr_activation_rate_pos}_{qcol}"
            c_arn = f"{colmap.pred_afrr_activation_rate_neg}_{qcol}"
            if c_app in df.columns:
                act_price_pos_by_bin[:, b] = pd.to_numeric(df[c_app], errors="coerce").to_numpy(dtype=float)
            else:
                missing_bin_cols.append(c_app)
                act_price_pos_by_bin[:, b] = np.nan
            if c_apn in df.columns:
                act_price_neg_by_bin[:, b] = pd.to_numeric(df[c_apn], errors="coerce").to_numpy(dtype=float)
            else:
                missing_bin_cols.append(c_apn)
                act_price_neg_by_bin[:, b] = np.nan
            if c_arp in df.columns:
                act_rate_pos_by_bin[:, b] = self._clip_rate(pd.to_numeric(df[c_arp], errors="coerce").to_numpy(dtype=float))
            else:
                missing_bin_cols.append(c_arp)
                act_rate_pos_by_bin[:, b] = np.nan
            if c_arn in df.columns:
                act_rate_neg_by_bin[:, b] = self._clip_rate(pd.to_numeric(df[c_arn], errors="coerce").to_numpy(dtype=float))
            else:
                missing_bin_cols.append(c_arn)
                act_rate_neg_by_bin[:, b] = np.nan
        if strict_input_validation and missing_bin_cols:
            uniq = sorted(set(missing_bin_cols))
            raise ValueError(
                "Missing required aFRR quantile-bin inputs in strict mode. "
                f"missing_columns={uniq[:40]}"
            )
        # Non-strict fallback remains explicit and deterministic.
        for b in range(n_bins):
            bad_pos_price = ~np.isfinite(act_price_pos_by_bin[:, b])
            bad_neg_price = ~np.isfinite(act_price_neg_by_bin[:, b])
            bad_pos_rate = ~np.isfinite(act_rate_pos_by_bin[:, b])
            bad_neg_rate = ~np.isfinite(act_rate_neg_by_bin[:, b])
            if bad_pos_price.any():
                act_price_pos_by_bin[bad_pos_price, b] = act_price_pos[bad_pos_price]
            if bad_neg_price.any():
                act_price_neg_by_bin[bad_neg_price, b] = act_price_neg[bad_neg_price]
            if bad_pos_rate.any():
                act_rate_pos_by_bin[bad_pos_rate, b] = r_act_pos_base[bad_pos_rate]
            if bad_neg_rate.any():
                act_rate_neg_by_bin[bad_neg_rate, b] = r_act_neg_base[bad_neg_rate]

        # Fail-fast critical-input validation before MILP formulation.
        critical_inputs = {
            "pred_da_price": p_da,
            "pred_afrr_capacity_price_pos": p_cap_pos,
            "pred_afrr_capacity_price_neg": p_cap_neg,
            "pred_afrr_activation_price_pos": act_price_pos,
            "pred_afrr_activation_price_neg": act_price_neg,
            "pred_afrr_activation_rate_pos": r_act_pos_base,
            "pred_afrr_activation_rate_neg": r_act_neg_base,
            "pred_afrr_activation_rate_pos_p90": r_act_pos_p90,
            "pred_afrr_activation_rate_neg_p90": r_act_neg_p90,
        }
        bad_inputs = [name for name, arr in critical_inputs.items() if not np.isfinite(arr).all()]
        if strict_input_validation and bad_inputs:
            raise ValueError(
                "NaN/inf detected in critical optimizer inputs before MILP build. "
                f"Fix data pipeline. bad_columns={bad_inputs}"
            )

        # Dynamic quantile bidding:
        # - regular mode: bin price = predicted quantile value, p_acc = 1-q
        # - deterministic/perfect_foresight mode: collapse uncertainty, use point capacity
        #   prices for every bin and set p_acc=1.0.
        p_acc_cap_pos = np.zeros((n, n_bins), dtype=float)
        p_acc_cap_neg = np.zeros((n, n_bins), dtype=float)
        cap_price_pos_by_bin = np.zeros((n, n_bins), dtype=float)
        cap_price_neg_by_bin = np.zeros((n, n_bins), dtype=float)
        pacc_pos_fallback_used = np.zeros(n, dtype=float)
        pacc_neg_fallback_used = np.zeros(n, dtype=float)
        if deterministic_reserve_settlement:
            base_cap_pos = self._finite_numeric_series(
                df,
                colmap.pred_afrr_capacity_price_pos,
                fallback_cols=[colmap.pred_afrr_capacity_price_neg],
                default=0.0,
                strict_non_null=True,
                allow_temporal_fill=False,
            ).to_numpy(dtype=float)
            base_cap_neg = self._finite_numeric_series(
                df,
                colmap.pred_afrr_capacity_price_neg,
                fallback_cols=[colmap.pred_afrr_capacity_price_pos],
                default=0.0,
                strict_non_null=True,
                allow_temporal_fill=False,
            ).to_numpy(dtype=float)
            for b in range(n_bins):
                p_acc_cap_pos[:, b] = 1.0
                p_acc_cap_neg[:, b] = 1.0
                cap_price_pos_by_bin[:, b] = base_cap_pos
                cap_price_neg_by_bin[:, b] = base_cap_neg
        else:
            fallback_decay = 0.7
            pos_quant_cols = [f"{colmap.pred_afrr_capacity_price_pos}_{q}" for q in self.afrr_quantile_bins]
            neg_quant_cols = [f"{colmap.pred_afrr_capacity_price_neg}_{q}" for q in self.afrr_quantile_bins]
            pos_quant_struct_missing = any(c not in df.columns for c in pos_quant_cols)
            neg_quant_struct_missing = any(c not in df.columns for c in neg_quant_cols)
            if strict_input_validation and (pos_quant_struct_missing or neg_quant_struct_missing):
                missing_cap_cols: list[str] = []
                if pos_quant_struct_missing:
                    missing_cap_cols.extend([c for c in pos_quant_cols if c not in df.columns])
                if neg_quant_struct_missing:
                    missing_cap_cols.extend([c for c in neg_quant_cols if c not in df.columns])
                raise ValueError(
                    "Missing required aFRR capacity quantile-bin inputs in strict mode. "
                    f"missing_columns={sorted(set(missing_cap_cols))}"
                )

            for b, qcol in enumerate(self.afrr_quantile_bins):
                q_level = float(qcol.replace("p", "")) / 100.0
                pacc_const = max(0.0, min(1.0, 1.0 - q_level))
                p_acc_cap_pos[:, b] = pacc_const
                p_acc_cap_neg[:, b] = pacc_const
                c_pos = f"{colmap.pred_afrr_capacity_price_pos}_{qcol}"
                c_neg = f"{colmap.pred_afrr_capacity_price_neg}_{qcol}"
                if c_pos in df.columns:
                    cap_price_pos_by_bin[:, b] = pd.to_numeric(df[c_pos], errors="coerce").to_numpy(dtype=float)
                else:
                    cap_price_pos_by_bin[:, b] = np.nan
                if c_neg in df.columns:
                    cap_price_neg_by_bin[:, b] = pd.to_numeric(df[c_neg], errors="coerce").to_numpy(dtype=float)
                else:
                    cap_price_neg_by_bin[:, b] = np.nan

            mid_pos = self._finite_numeric_series(
                df,
                f"{colmap.pred_afrr_capacity_price_pos}_p50",
                fallback_cols=[colmap.pred_afrr_capacity_price_pos, f"{colmap.pred_afrr_capacity_price_neg}_p50", colmap.pred_afrr_capacity_price_neg],
                default=0.0,
            ).to_numpy(dtype=float)
            mid_neg = self._finite_numeric_series(
                df,
                f"{colmap.pred_afrr_capacity_price_neg}_p50",
                fallback_cols=[colmap.pred_afrr_capacity_price_neg, f"{colmap.pred_afrr_capacity_price_pos}_p50", colmap.pred_afrr_capacity_price_pos],
                default=0.0,
            ).to_numpy(dtype=float)
            mid_bin = self.afrr_quantile_bins.index("p50") if "p50" in self.afrr_quantile_bins else n_bins // 2
            for t in range(n):
                have_pos = np.isfinite(cap_price_pos_by_bin[t, :]).all()
                have_neg = np.isfinite(cap_price_neg_by_bin[t, :]).all()
                if pos_quant_struct_missing:
                    pacc_pos_fallback_used[t] = 1.0
                    cap_price_pos_by_bin[t, :] = mid_pos[t]
                    for b in range(n_bins):
                        p_acc_cap_pos[t, b] = 0.5 * (fallback_decay ** abs(b - mid_bin))
                elif not have_pos:
                    raise ValueError(
                        "NaN detected in aFRR positive quantile capacity prices although quantile columns "
                        "exist. Fallback is only allowed for structurally missing quantile columns."
                    )
                if neg_quant_struct_missing:
                    pacc_neg_fallback_used[t] = 1.0
                    cap_price_neg_by_bin[t, :] = mid_neg[t]
                    for b in range(n_bins):
                        p_acc_cap_neg[t, b] = 0.5 * (fallback_decay ** abs(b - mid_bin))
                elif not have_neg:
                    raise ValueError(
                        "NaN detected in aFRR negative quantile capacity prices although quantile columns "
                        "exist. Fallback is only allowed for structurally missing quantile columns."
                    )

        c = np.zeros(n_vars, dtype=float)

        da_step = self.da_bid_granularity_mw
        afrr_step = self.afrr_bid_granularity_mw

        # Objective (maximize predicted margin, scipy.milp minimizes => negate coefficients).
        # Keep DA/aFRR coefficients as pure hourly marginal cashflows.
        # Intertemporal energy value is represented via SoC state transition +
        # terminal SoC credit (added separately below), not inside these marginals.
        # DA auxiliary-energy cost per dispatched MW:
        # Hours_trading_equiv = throughput_mwh / power_mw
        # For 1 MW decision variable over dt_h: throughput_mwh = 1 * dt_h.
        # => hours_equiv_per_mw = dt_h / P_max
        # Energy_trading_per_mw = hours_equiv_per_mw * aux_peak_mw * duty_trading
        # Cost_per_mw = Energy_trading_per_mw * DA_price
        da_hours_trading_equiv_per_mw = self.dt_h / max(self.p_max_mw, 1e-12)
        da_aux_energy_trading_per_mw_mwh = da_hours_trading_equiv_per_mw * self.aux_trading_mw
        da_aux_eur_per_mw = da_aux_energy_trading_per_mw_mwh * p_da
        # Financial DA cashflows are settled on grid-side volumes.
        # Efficiency is therefore not applied to DA price/transaction terms here;
        # it enters through SoC physics and internal-throughput degradation conversion.
        ch_coef = (
            -p_da
            - self.trans_eur_mwh
            - (self.deg_eur_mwh * self.eta_in)
            - da_aux_eur_per_mw
        )
        dis_coef = (
            p_da
            - self.trans_eur_mwh
            - (self.deg_eur_mwh / max(self.eta_out, 1e-12))
            - da_aux_eur_per_mw
        )
        c[sl["ch"]] = -(ch_coef * da_step)
        c[sl["dis"]] = -(dis_coef * da_step)
        rpos_coef_by_bin = np.zeros((n, n_bins), dtype=float)
        rneg_coef_by_bin = np.zeros((n, n_bins), dtype=float)
        bem_pos_coef_by_bin = np.zeros((n, n_bins), dtype=float)
        bem_neg_coef_by_bin = np.zeros((n, n_bins), dtype=float)
        bcm_expected_capacity_revenue_pos_by_bin = np.zeros((n, n_bins), dtype=float)
        bcm_expected_capacity_revenue_neg_by_bin = np.zeros((n, n_bins), dtype=float)
        bcm_expected_activation_revenue_pos_by_bin = np.zeros((n, n_bins), dtype=float)
        bcm_expected_activation_revenue_neg_by_bin = np.zeros((n, n_bins), dtype=float)
        bcm_expected_aux_cost_pos_by_bin = np.zeros((n, n_bins), dtype=float)
        bcm_expected_aux_cost_neg_by_bin = np.zeros((n, n_bins), dtype=float)
        bcm_offer_cost_by_bin = np.zeros((n, n_bins), dtype=float)
        bcm_activation_margin_pos_by_bin = np.zeros((n, n_bins), dtype=float)
        bcm_activation_margin_neg_by_bin = np.zeros((n, n_bins), dtype=float)
        bem_expected_activation_revenue_pos_by_bin = np.zeros((n, n_bins), dtype=float)
        bem_expected_activation_revenue_neg_by_bin = np.zeros((n, n_bins), dtype=float)
        bem_expected_aux_cost_pos_by_bin = np.zeros((n, n_bins), dtype=float)
        bem_expected_aux_cost_neg_by_bin = np.zeros((n, n_bins), dtype=float)
        bem_activation_margin_pos_by_bin = np.zeros((n, n_bins), dtype=float)
        bem_activation_margin_neg_by_bin = np.zeros((n, n_bins), dtype=float)
        for b in range(n_bins):
            # Correct EV split:
            # 1) offer-side fixed availability cost (no p_acc multiplier)
            # 2) activation/throughput terms scaled by expected activated share
            #    exp_act = p_acc * expected_activation_rate
            exp_act_pos = p_acc_cap_pos[:, b] * act_rate_pos_by_bin[:, b]
            exp_act_neg = p_acc_cap_neg[:, b] * act_rate_neg_by_bin[:, b]
            offer_cost = self.afrr_offer_cost_eur_mw_h * self.dt_h
            # Expected aFRR auxiliary energy over reserve product window:
            # hours_afrr_active = window_h * p_acc * activation_rate
            # hours_afrr_standby = window_h * (p_acc - p_acc*activation_rate)
            # reserve-MW scaling enters via per-MW equivalent hours:
            # hours_equiv_per_mw = window_h / P_max
            window_h = float(self.reserve_product_duration_h)
            afrr_hours_equiv_per_mw = window_h / max(self.p_max_mw, 1e-12)
            standby_pos = np.maximum(0.0, p_acc_cap_pos[:, b] - exp_act_pos)
            standby_neg = np.maximum(0.0, p_acc_cap_neg[:, b] - exp_act_neg)
            afrr_aux_cost_pos = (
                afrr_hours_equiv_per_mw
                * (
                    exp_act_pos * self.aux_afrr_active_mw
                    + standby_pos * self.aux_standby_mw
                )
                * p_da
            )
            afrr_aux_cost_neg = (
                afrr_hours_equiv_per_mw
                * (
                    exp_act_neg * self.aux_afrr_active_mw
                    + standby_neg * self.aux_standby_mw
                )
                * p_da
            )
            bcm_cap_rev_pos = p_acc_cap_pos[:, b] * cap_price_pos_by_bin[:, b]
            bcm_cap_rev_neg = p_acc_cap_neg[:, b] * cap_price_neg_by_bin[:, b]
            bcm_act_margin_pos = (
                act_price_pos_by_bin[:, b]
                - self.trans_eur_mwh
                - (self.deg_eur_mwh / max(self.eta_out, 1e-12))
            )
            bcm_act_margin_neg = (
                -act_price_neg_by_bin[:, b]
                - self.trans_eur_mwh
                - (self.deg_eur_mwh * self.eta_in)
            )
            bcm_act_rev_pos = exp_act_pos * bcm_act_margin_pos
            bcm_act_rev_neg = exp_act_neg * bcm_act_margin_neg
            # aFRR EV terms:
            # - capacity remuneration is expected via p_acc
            # - activation remuneration/cost is expected via p_acc * act_rate
            # - market price terms remain grid-side (no eta scaling)
            # - degradation is internal-throughput based (eta conversion kept)
            rpos_coef = (
                bcm_cap_rev_pos
                - offer_cost
                + bcm_act_rev_pos
                - afrr_aux_cost_pos
            )
            rneg_coef = (
                bcm_cap_rev_neg
                - offer_cost
                + bcm_act_rev_neg
                - afrr_aux_cost_neg
            )
            s_pos = sl["rpos_bin"].start + b * n
            s_neg = sl["rneg_bin"].start + b * n
            rpos_coef_by_bin[:, b] = rpos_coef
            rneg_coef_by_bin[:, b] = rneg_coef
            c[s_pos : s_pos + n] = -(rpos_coef * afrr_step)
            c[s_neg : s_neg + n] = -(rneg_coef * afrr_step)
            bem_exec_prob_pos = p_acc_cap_pos[:, b]
            bem_exec_prob_neg = p_acc_cap_neg[:, b]
            bem_act_margin_pos = (
                act_price_pos_by_bin[:, b]
                - self.trans_eur_mwh
                - (self.deg_eur_mwh / max(self.eta_out, 1e-12))
            )
            bem_act_margin_neg = (
                -act_price_neg_by_bin[:, b]
                - self.trans_eur_mwh
                - (self.deg_eur_mwh * self.eta_in)
            )
            bem_pos_coef = (
                bem_exec_prob_pos
                * act_rate_pos_by_bin[:, b]
                * bem_act_margin_pos
            )
            bem_neg_coef = (
                bem_exec_prob_neg
                * act_rate_neg_by_bin[:, b]
                * bem_act_margin_neg
            )
            # Expected BEM active auxiliary energy cost (no standby component).
            bem_aux_hours_equiv_per_mw = self.dt_h / max(self.p_max_mw, 1e-12)
            bem_aux_cost_pos = (
                bem_aux_hours_equiv_per_mw
                * bem_exec_prob_pos
                * act_rate_pos_by_bin[:, b]
                * self.aux_afrr_active_mw
                * p_da
            )
            bem_aux_cost_neg = (
                bem_aux_hours_equiv_per_mw
                * bem_exec_prob_neg
                * act_rate_neg_by_bin[:, b]
                * self.aux_afrr_active_mw
                * p_da
            )
            bem_pos_coef -= bem_aux_cost_pos
            bem_neg_coef -= bem_aux_cost_neg
            bcm_expected_capacity_revenue_pos_by_bin[:, b] = bcm_cap_rev_pos
            bcm_expected_capacity_revenue_neg_by_bin[:, b] = bcm_cap_rev_neg
            bcm_expected_activation_revenue_pos_by_bin[:, b] = bcm_act_rev_pos
            bcm_expected_activation_revenue_neg_by_bin[:, b] = bcm_act_rev_neg
            bcm_expected_aux_cost_pos_by_bin[:, b] = afrr_aux_cost_pos
            bcm_expected_aux_cost_neg_by_bin[:, b] = afrr_aux_cost_neg
            bcm_offer_cost_by_bin[:, b] = np.full(n, offer_cost, dtype=float)
            bcm_activation_margin_pos_by_bin[:, b] = bcm_act_margin_pos
            bcm_activation_margin_neg_by_bin[:, b] = bcm_act_margin_neg
            bem_expected_activation_revenue_pos_by_bin[:, b] = (
                bem_exec_prob_pos * act_rate_pos_by_bin[:, b] * bem_act_margin_pos
            )
            bem_expected_activation_revenue_neg_by_bin[:, b] = (
                bem_exec_prob_neg * act_rate_neg_by_bin[:, b] * bem_act_margin_neg
            )
            bem_expected_aux_cost_pos_by_bin[:, b] = bem_aux_cost_pos
            bem_expected_aux_cost_neg_by_bin[:, b] = bem_aux_cost_neg
            bem_activation_margin_pos_by_bin[:, b] = bem_act_margin_pos
            bem_activation_margin_neg_by_bin[:, b] = bem_act_margin_neg
            bem_pos_coef_by_bin[:, b] = bem_pos_coef
            bem_neg_coef_by_bin[:, b] = bem_neg_coef
            s_bem_pos = sl["bem_pos_bin"].start + b * n
            s_bem_neg = sl["bem_neg_bin"].start + b * n
            c[s_bem_pos : s_bem_pos + n] = -(bem_pos_coef * afrr_step)
            c[s_bem_neg : s_bem_neg + n] = -(bem_neg_coef * afrr_step)
        # Soft chance-constraint penalties (continuous non-delivery slack).
        # Make slack prohibitively expensive relative to capacity revenues so
        # solver prioritizes physical feasibility over speculative awards.
        cap_price_scale = float(
            max(
                1.0,
                float(np.nanmax(p_cap_pos)) if len(p_cap_pos) else 1.0,
                float(np.nanmax(p_cap_neg)) if len(p_cap_neg) else 1.0,
                float(np.nanmax(cap_price_pos_by_bin)) if cap_price_pos_by_bin.size else 1.0,
                float(np.nanmax(cap_price_neg_by_bin)) if cap_price_neg_by_bin.size else 1.0,
            )
        )
        strict_slack_lambda = max(
            self.imbalance_penalty_eur_mwh * self.dt_h,
            (cap_price_scale + 1.0) ** 2 * self.dt_h,
        )
        c[sl["slack_pos"]] = strict_slack_lambda
        c[sl["slack_neg"]] = strict_slack_lambda
        # In thesis hard mode, terminal SoC is a hard feasibility constraint and
        # must not be softened via slack penalties.
        if str(self.final_soc_mode) == "hard":
            c[sl["slack_final_soc"]] = 0.0
        else:
            c[sl["slack_final_soc"]] = max(0.0, float(self.final_soc_shortfall_penalty_eur_per_mwh))
        # Emergency physical SoC slacks: make violations possible but extremely expensive.
        emergency_soc_slack_lambda = max(
            10_000.0,
            float(self.final_soc_shortfall_penalty_eur_per_mwh),
            strict_slack_lambda * 10.0,
        )
        c[sl["slack_soc_min"]] = emergency_soc_slack_lambda
        c[sl["slack_soc_max"]] = emergency_soc_slack_lambda
        # Structural-infeasibility guards:
        # - deliverability slacks soften 4h reserve-energy/headroom hard bounds
        # - obligation slacks soften lockbook reserve obligations
        penalty_deliverability = max(5_000.0, strict_slack_lambda)
        penalty_obligation = max(10_000.0, strict_slack_lambda * 2.0)
        c[sl["slack_deliver_pos"]] = penalty_deliverability
        c[sl["slack_deliver_neg"]] = penalty_deliverability
        c[sl["slack_obligation_pos"]] = penalty_obligation
        c[sl["slack_obligation_neg"]] = penalty_obligation
        # Terminal SoC opportunity value (anti end-of-horizon dumping):
        # Reward terminal inventory with a forward-looking DA proxy from the
        # current horizon (mean predicted DA price), scaled by export efficiency.
        # scipy.milp minimizes, so we add a negative coefficient on terminal SoC.
        da_ref = pd.Series(p_da, dtype="float64").dropna()
        ref_da_price = max(0.0, float(da_ref.mean()) if not da_ref.empty else 0.0)
        terminal_soc_discount = float(MODEL_SPECS.get("terminal_soc_value_discount", 0.8))
        terminal_soc_discount = max(0.0, min(1.0, terminal_soc_discount))
        terminal_soc_value_eur_per_mwh = terminal_soc_discount * self.eta_out * ref_da_price
        c[sl["soc"].start + n] = -terminal_soc_value_eur_per_mwh
        if not np.isfinite(c).all():
            bad = np.where(~np.isfinite(c))[0]
            raise ValueError(
                "Non-finite objective coefficients detected in MILP objective vector "
                f"(count={len(bad)}). Check prediction coverage/fallback mapping."
            )

        a_eq = []
        b_eq = []

        # SoC dynamics.
        for t in range(n):
            row = np.zeros(n_vars, dtype=float)
            row[sl["soc"].start + t + 1] = 1.0
            row[sl["soc"].start + t] = -1.0
            row[sl["ch"].start + t] = -self.eta_in * da_step
            row[sl["dis"].start + t] = (1.0 / self.eta_out) * da_step
            row[sl["u"].start + t] = self.dt_h
            for b in range(n_bins):
                # Base SoC drift on expected awarded-and-activated reserve.
                exp_pos = p_acc_cap_pos[t, b] * act_rate_pos_by_bin[t, b]
                exp_neg = p_acc_cap_neg[t, b] * act_rate_neg_by_bin[t, b]
                row[sl["rpos_bin"].start + b * n + t] = (exp_pos / self.eta_out) * afrr_step
                row[sl["rneg_bin"].start + b * n + t] = -self.eta_in * exp_neg * afrr_step
            for b in range(n_bins):
                row[sl["bem_pos_bin"].start + b * n + t] = (
                    act_rate_pos_by_bin[t, b] / self.eta_out
                ) * afrr_step
                row[sl["bem_neg_bin"].start + b * n + t] = (
                    -self.eta_in * act_rate_neg_by_bin[t, b]
                ) * afrr_step
            a_eq.append(row)
            b_eq.append(0.0)

        # Initial and terminal SoC.
        row_init = np.zeros(n_vars, dtype=float)
        row_init[sl["soc"].start] = 1.0
        a_eq.append(row_init)
        b_eq.append(self.soc_init if soc_start is None else soc_start)

        # Do not enforce hard terminal SoC equality; route any caller-provided
        # terminal target into the soft minimum target to preserve feasibility.
        if soc_end_target is not None:
            soc_end_min_target = (
                float(soc_end_target)
                if soc_end_min_target is None
                else max(float(soc_end_min_target), float(soc_end_target))
            )

        a_ub = []
        b_ub = []

        # Power limits and conservative reserve coupling.
        ts_index = pd.to_datetime(df[colmap.timestamp], utc=True, errors="coerce").reset_index(drop=True)
        for t in range(n):
            # charge + reserve_neg <= Pmax
            row = np.zeros(n_vars, dtype=float)
            row[sl["ch"].start + t] = da_step
            for b in range(n_bins):
                row[sl["rneg_bin"].start + b * n + t] = afrr_step
            for b in range(n_bins):
                row[sl["bem_neg_bin"].start + b * n + t] = afrr_step
            a_ub.append(row)
            b_ub.append(self.p_max_mw)

            # discharge + reserve_pos <= Pmax
            row = np.zeros(n_vars, dtype=float)
            row[sl["dis"].start + t] = da_step
            for b in range(n_bins):
                row[sl["rpos_bin"].start + b * n + t] = afrr_step
            for b in range(n_bins):
                row[sl["bem_pos_bin"].start + b * n + t] = afrr_step
            a_ub.append(row)
            b_ub.append(self.p_max_mw)

            # reserve_pos + reserve_neg <= reserve_max
            row = np.zeros(n_vars, dtype=float)
            for b in range(n_bins):
                row[sl["rpos_bin"].start + b * n + t] = afrr_step
                row[sl["rneg_bin"].start + b * n + t] = afrr_step
            a_ub.append(row)
            b_ub.append(self.reserve_max_mw)

            # Aux lower-bound proxies by operating regime (LP-friendly, no binaries):
            # u_t >= standby when reserve is committed (scaled by committed reserve fraction)
            # u_t >= trading when DA dispatch is active (scaled by power fraction)
            # u_t >= aFRR-active when expected activated reserve is high (scaled)
            if self.aux_mode == "state_dependent":
                reserve_scale = max(self.reserve_max_mw, 1e-9)
                power_scale = max(self.p_max_mw, 1e-9)
                row = np.zeros(n_vars, dtype=float)
                row[sl["u"].start + t] = -1.0
                for b in range(n_bins):
                    row[sl["rpos_bin"].start + b * n + t] = (self.aux_standby_mw / reserve_scale) * afrr_step
                    row[sl["rneg_bin"].start + b * n + t] = (self.aux_standby_mw / reserve_scale) * afrr_step
                a_ub.append(row)
                b_ub.append(0.0)

                row = np.zeros(n_vars, dtype=float)
                row[sl["u"].start + t] = -1.0
                row[sl["ch"].start + t] = (self.aux_trading_mw / power_scale) * da_step
                row[sl["dis"].start + t] = (self.aux_trading_mw / power_scale) * da_step
                a_ub.append(row)
                b_ub.append(0.0)

                row = np.zeros(n_vars, dtype=float)
                row[sl["u"].start + t] = -1.0
                for b in range(n_bins):
                    exp_pos = p_acc_cap_pos[t, b] * act_rate_pos_by_bin[t, b]
                    exp_neg = p_acc_cap_neg[t, b] * act_rate_neg_by_bin[t, b]
                    row[sl["rpos_bin"].start + b * n + t] = (self.aux_afrr_active_mw / power_scale) * exp_pos * afrr_step
                    row[sl["rneg_bin"].start + b * n + t] = (self.aux_afrr_active_mw / power_scale) * exp_neg * afrr_step
                a_ub.append(row)
                b_ub.append(0.0)

            # p90 chance-constraint safety bounds:
            # enforce sufficient SoC footroom/headroom for p90 activation-rate
            # on expected awarded reserve (capacity acceptance by bin).
            # pos reserve needs discharge energy from SoC:
            # enforce a minimum 15-min-equivalent activation headroom floor via
            # max(r_act_p90, min_activation_headroom_fraction).
            # soc_t - sum_b(p_acc_cap_pos[t,b] * max(r_act_pos_p90[t], r_floor) * reserve_bin[b,t] * dt / eta_out) >= soc_min
            row = np.zeros(n_vars, dtype=float)
            row[sl["soc"].start + t] = -1.0
            for b in range(n_bins):
                chance_pos = p_acc_cap_pos[t, b] * max(
                    float(r_act_pos_p90[t]),
                    float(self.min_activation_headroom_fraction),
                )
                row[sl["rpos_bin"].start + b * n + t] = (
                    chance_pos * self.dt_h / max(self.eta_out, 1e-12)
                ) * afrr_step
            row[sl["slack_pos"].start + t] = -1.0
            a_ub.append(row)
            b_ub.append(-self.soc_min)

            # neg reserve needs charging headroom in SoC:
            # same activation headroom floor as positive direction.
            # soc_t + sum_b(p_acc_cap_neg[t,b] * max(r_act_neg_p90[t], r_floor) * reserve_bin[b,t] * dt * eta_in) <= soc_max
            row = np.zeros(n_vars, dtype=float)
            row[sl["soc"].start + t] = 1.0
            for b in range(n_bins):
                chance_neg = p_acc_cap_neg[t, b] * max(
                    float(r_act_neg_p90[t]),
                    float(self.min_activation_headroom_fraction),
                )
                row[sl["rneg_bin"].start + b * n + t] = (chance_neg * self.dt_h * self.eta_in) * afrr_step
            row[sl["slack_neg"].start + t] = -1.0
            a_ub.append(row)
            b_ub.append(self.soc_max)

            # Hard physical deliverability bounds for awarded reserve under
            # full-activation headroom assumption (configurable, default 0.5h).
            # pos reserve needs footroom above soc_min (discharge capability).
            row = np.zeros(n_vars, dtype=float)
            row[sl["soc"].start + t] = -1.0
            for b in range(n_bins):
                row[sl["rpos_bin"].start + b * n + t] = (
                    self.reserve_activation_headroom_h / max(self.eta_out, 1e-12)
                ) * afrr_step
            a_ub.append(row)
            b_ub.append(-self.soc_min)

            # neg reserve needs headroom below soc_max (charge capability).
            row = np.zeros(n_vars, dtype=float)
            row[sl["soc"].start + t] = 1.0
            for b in range(n_bins):
                row[sl["rneg_bin"].start + b * n + t] = (
                    self.reserve_activation_headroom_h * self.eta_in
                ) * afrr_step
            a_ub.append(row)
            b_ub.append(self.soc_max)

            # Hard physical deliverability bounds for explicit BEM-only bids
            # under full-activation headroom assumption.
            row = np.zeros(n_vars, dtype=float)
            row[sl["soc"].start + t] = -1.0
            for b in range(n_bins):
                row[sl["bem_pos_bin"].start + b * n + t] = (
                    self.bem_activation_headroom_h / max(self.eta_out, 1e-12)
                ) * afrr_step
            a_ub.append(row)
            b_ub.append(-self.soc_min)

            row = np.zeros(n_vars, dtype=float)
            row[sl["soc"].start + t] = 1.0
            for b in range(n_bins):
                row[sl["bem_neg_bin"].start + b * n + t] = (
                    self.bem_activation_headroom_h * self.eta_in
                ) * afrr_step
            a_ub.append(row)
            b_ub.append(self.soc_max)

            # Note: DA charge/discharge mutual exclusivity has been relaxed by request.
            # Simultaneous charge/discharge is therefore allowed as long as all
            # other physical/power constraints remain satisfied.

            # DA gate-closure lock: fixed day-ahead charge/discharge bids.
            if fixed_da_dispatch and da_enabled:
                ts = ts_index.iloc[t]
                if pd.notna(ts) and ts in fixed_da_dispatch:
                    ch_fix, dis_fix = self._normalize_da_bid(*fixed_da_dispatch[ts])

                    row = np.zeros(n_vars, dtype=float)
                    row[sl["ch"].start + t] = da_step
                    a_eq.append(row)
                    b_eq.append(float(ch_fix))

                    row = np.zeros(n_vars, dtype=float)
                    row[sl["dis"].start + t] = da_step
                    a_eq.append(row)
                    b_eq.append(float(dis_fix))
            # Reserve lockbook obligation: when capacity has already been
            # awarded, force optimization to carry that commitment explicitly.
            if fixed_reserve_obligation and afrr_enabled:
                ts = ts_index.iloc[t]
                if pd.notna(ts) and ts in fixed_reserve_obligation:
                    ob_pos, ob_neg = fixed_reserve_obligation[ts]
                    ob_pos_q = self.bid_builder._qfloor(
                        max(0.0, float(ob_pos)),
                        self.afrr_bid_granularity_mw,
                    )
                    ob_neg_q = self.bid_builder._qfloor(
                        max(0.0, float(ob_neg)),
                        self.afrr_bid_granularity_mw,
                    )
                    # Hard lockbook obligations (exact):
                    # sum(reserve_pos_bins) == obligated_pos
                    # sum(reserve_neg_bins) == obligated_neg
                    # First add >= in <= form: -sum(reserve_*) <= -obligated_*
                    row = np.zeros(n_vars, dtype=float)
                    for b in range(n_bins):
                        row[sl["rpos_bin"].start + b * n + t] = -afrr_step
                    a_ub.append(row)
                    b_ub.append(-float(ob_pos_q))

                    row = np.zeros(n_vars, dtype=float)
                    for b in range(n_bins):
                        row[sl["rneg_bin"].start + b * n + t] = -afrr_step
                    a_ub.append(row)
                    b_ub.append(-float(ob_neg_q))

                    # Then add <= bounds to enforce equality.
                    row = np.zeros(n_vars, dtype=float)
                    for b in range(n_bins):
                        row[sl["rpos_bin"].start + b * n + t] = afrr_step
                    a_ub.append(row)
                    b_ub.append(float(ob_pos_q))

                    row = np.zeros(n_vars, dtype=float)
                    for b in range(n_bins):
                        row[sl["rneg_bin"].start + b * n + t] = afrr_step
                    a_ub.append(row)
                    b_ub.append(float(ob_neg_q))

                    # Locked-reserve trajectory guard:
                    # preserve delivery-start SoC headroom for fixed obligations
                    # with the same strict start-of-hour convention as audit.
                    req_pos_mwh = (
                        float(ob_pos_q) * float(self.reserve_activation_headroom_h) / max(float(self.eta_out), 1e-12)
                        + float(self.reserve_headroom_safety_mwh)
                        + float(self.reserve_soc_projection_safety_mwh)
                    )
                    req_neg_mwh = (
                        float(ob_neg_q) * float(self.reserve_activation_headroom_h) * float(self.eta_in)
                        + float(self.reserve_headroom_safety_mwh)
                        + float(self.reserve_soc_projection_safety_mwh)
                    )
                    if req_pos_mwh > 0.0:
                        # soc_start_lp[t] >= soc_min + req_pos_mwh
                        row = np.zeros(n_vars, dtype=float)
                        row[sl["soc"].start + t] = -1.0
                        a_ub.append(row)
                        b_ub.append(-(float(self.soc_min) + req_pos_mwh))
                    if req_neg_mwh > 0.0:
                        # soc_start_lp[t] <= soc_max - req_neg_mwh
                        row = np.zeros(n_vars, dtype=float)
                        row[sl["soc"].start + t] = 1.0
                        a_ub.append(row)
                        b_ub.append(float(self.soc_max) - req_neg_mwh)

        # Soft physical SoC bounds (including terminal state):
        # soc_t + slack_soc_min_t >= soc_min
        # soc_t - slack_soc_max_t <= soc_max
        for t in range(n + 1):
            row = np.zeros(n_vars, dtype=float)
            row[sl["soc"].start + t] = -1.0
            row[sl["slack_soc_min"].start + t] = -1.0
            a_ub.append(row)
            b_ub.append(-self.soc_min)

            row = np.zeros(n_vars, dtype=float)
            row[sl["soc"].start + t] = 1.0
            row[sl["slack_soc_max"].start + t] = -1.0
            a_ub.append(row)
            b_ub.append(self.soc_max)

        if soc_end_min_target is not None:
            if str(self.final_soc_mode) == "hard":
                # Hard terminal SoC floor for thesis-reportable strict runs.
                # No slack-based repair is allowed in this mode.
                row = np.zeros(n_vars, dtype=float)
                row[sl["soc"].start + n] = -1.0
                a_ub.append(row)
                b_ub.append(-float(soc_end_min_target))
            else:
                # Diagnostic terminal-repair mode (soft floor via slack).
                row = np.zeros(n_vars, dtype=float)
                row[sl["soc"].start + n] = -1.0
                row[sl["slack_final_soc"].start] = -1.0
                a_ub.append(row)
                b_ub.append(-float(soc_end_min_target))

        lb = np.zeros(n_vars, dtype=float)
        ub = np.full(n_vars, np.inf, dtype=float)
        da_units_max = int(np.floor(self.p_max_mw / da_step + 1e-9)) if da_enabled else 0
        afrr_units_max = int(np.floor(self.reserve_max_mw / afrr_step + 1e-9)) if afrr_enabled else 0

        # Integer decision blocks.
        ub[sl["ch"]] = da_units_max
        ub[sl["dis"]] = da_units_max
        ub[sl["rpos_bin"]] = afrr_units_max if perms.allow_bcm else 0.0
        ub[sl["rneg_bin"]] = afrr_units_max if perms.allow_bcm else 0.0
        ub[sl["bem_pos_bin"]] = afrr_units_max if perms.allow_bem_only else 0.0
        ub[sl["bem_neg_bin"]] = afrr_units_max if perms.allow_bem_only else 0.0
        if self.aux_mode == "state_dependent":
            lb[sl["u"]] = max(0.0, self.aux_off_mw)
            ub[sl["u"]] = self.aux_peak_mw
        else:
            aux_const_mw = float(BATTERY_SPECS.get("aux_power_mw", 0.0))
            lb[sl["u"]] = aux_const_mw
            ub[sl["u"]] = aux_const_mw

        # State and slack blocks.
        # Keep SoC unbounded in LP variable bounds; enforce soft physical limits via
        # emergency slacks in explicit constraints below.
        lb[sl["soc"]] = -np.inf
        ub[sl["soc"]] = np.inf
        lb[sl["slack_pos"]] = 0.0
        lb[sl["slack_neg"]] = 0.0
        lb[sl["slack_final_soc"]] = 0.0
        lb[sl["slack_soc_min"]] = 0.0
        lb[sl["slack_soc_max"]] = 0.0
        lb[sl["slack_deliver_pos"]] = 0.0
        lb[sl["slack_deliver_neg"]] = 0.0
        lb[sl["slack_obligation_pos"]] = 0.0
        lb[sl["slack_obligation_neg"]] = 0.0

        # Defensive check: prevent MILP vector-shape mismatches.
        if not (len(c) == len(lb) == len(ub) == n_vars):
            raise ValueError(
                "MILP shape mismatch: "
                f"len(c)={len(c)} len(lb)={len(lb)} len(ub)={len(ub)} n_vars={n_vars}"
            )

        constraints = []
        if a_ub:
            a_ub_arr = np.array(a_ub)
            constraints.append(
                LinearConstraint(a_ub_arr, -np.inf * np.ones(a_ub_arr.shape[0]), np.array(b_ub))
            )
        if a_eq:
            a_eq_arr = np.array(a_eq)
            b_eq_arr = np.array(b_eq)
            constraints.append(LinearConstraint(a_eq_arr, b_eq_arr, b_eq_arr))

        # LP relaxation by design: all dispatch/reserve/auxiliary variables continuous.
        # This avoids NP-hard branch-and-bound behavior from integer decisions and
        # improves solve robustness for perfect_foresight benchmarking.
        integrality = np.zeros(n_vars, dtype=int)

        milp_options = self._milp_options()
        sol = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options=milp_options,
        )
        # Graceful timeout handling: continue with best incumbent feasible solution.
        timeout_with_incumbent = self._is_timeout_result(sol) and (sol.x is not None)
        if (not sol.success) and (not timeout_with_incumbent):
            msg = str(getattr(sol, "message", ""))
            if "infeasible" in msg.lower():
                self._write_infeasible_debug_dump(
                    df=df,
                    colmap=colmap,
                    c=c,
                    a_ub=(a_ub_arr if a_ub else None),
                    b_ub=(np.array(b_ub) if a_ub else None),
                    a_eq=(a_eq_arr if a_eq else None),
                    b_eq=(b_eq_arr if a_eq else None),
                    lb=lb,
                    ub=ub,
                    message=msg,
                    solve_context={
                        "timestamp_utc": (
                            pd.to_datetime(df[colmap.timestamp].iloc[0], utc=True, errors="coerce").isoformat()
                            if len(df) > 0
                            else ""
                        ),
                        "attempt_type": "optimize_dispatch_primary",
                        "final_accepted_path": False,
                    },
                )
            raise RuntimeError(f"MIP optimization failed: {sol.message}")
        if sol.x is None or len(sol.x) != n_vars:
            raise RuntimeError(
                "MILP solver returned invalid solution vector shape: "
                f"got={0 if sol.x is None else len(sol.x)} expected={n_vars}"
            )
        if not np.isfinite(sol.x).all():
            raise RuntimeError("MILP solver returned non-finite decision values.")
        if timeout_with_incumbent:
            print(
                "[WARN] MILP hit time limit; proceeding with incumbent feasible solution "
                f"(status={getattr(sol, 'status', 'n/a')}, msg='{getattr(sol, 'message', '')}')."
            )

        x = sol.x
        rpos_bin = x[sl["rpos_bin"]].reshape(n_bins, n).T
        rneg_bin = x[sl["rneg_bin"]].reshape(n_bins, n).T
        bem_pos_bin = x[sl["bem_pos_bin"]].reshape(n_bins, n).T
        bem_neg_bin = x[sl["bem_neg_bin"]].reshape(n_bins, n).T
        out = df[[colmap.timestamp]].copy()
        out["charge_mw"] = x[sl["ch"]] * da_step
        out["discharge_mw"] = x[sl["dis"]] * da_step
        out["reserve_pos_mw"] = rpos_bin.sum(axis=1) * afrr_step
        out["reserve_neg_mw"] = rneg_bin.sum(axis=1) * afrr_step
        out["bem_only_pos_mw"] = bem_pos_bin.sum(axis=1) * afrr_step
        out["bem_only_neg_mw"] = bem_neg_bin.sum(axis=1) * afrr_step
        out["aux_power_mw"] = x[sl["u"]]
        reserve_pos_bin_mw = rpos_bin * afrr_step
        reserve_neg_bin_mw = rneg_bin * afrr_step
        charge_mw = out["charge_mw"].to_numpy(dtype=float)
        discharge_mw = out["discharge_mw"].to_numpy(dtype=float)
        soc_lp = x[sl["soc"].start + 1 : sl["soc"].start + n + 1]
        soc_start_lp = x[sl["soc"].start : sl["soc"].start + n]
        final_soc_shortfall_mwh = (
            float(x[sl["slack_final_soc"].start])
            if ("slack_final_soc" in sl and str(self.final_soc_mode) != "hard")
            else 0.0
        )
        slack_pos = x[sl["slack_pos"]]
        slack_neg = x[sl["slack_neg"]]
        slack_soc_min = x[sl["slack_soc_min"]]
        slack_soc_max = x[sl["slack_soc_max"]]
        ev_da_charge_eur = ch_coef * charge_mw
        ev_da_discharge_eur = dis_coef * discharge_mw
        ev_afrr_pos_eur = (rpos_coef_by_bin * reserve_pos_bin_mw).sum(axis=1)
        ev_afrr_neg_eur = (rneg_coef_by_bin * reserve_neg_bin_mw).sum(axis=1)
        bem_pos_bin_mw = bem_pos_bin * afrr_step
        bem_neg_bin_mw = bem_neg_bin * afrr_step
        ev_bem_only_pos_eur = (bem_pos_coef_by_bin * bem_pos_bin_mw).sum(axis=1)
        ev_bem_only_neg_eur = (bem_neg_coef_by_bin * bem_neg_bin_mw).sum(axis=1)
        ev_slack_penalty_pos_eur = c[sl["slack_pos"]] * slack_pos
        ev_slack_penalty_neg_eur = c[sl["slack_neg"]] * slack_neg
        ev_terminal_soc_credit_eur = np.zeros(n, dtype=float)
        if n > 0:
            ev_terminal_soc_credit_eur[-1] = terminal_soc_value_eur_per_mwh * float(soc_lp[-1])
        ev_objective_rebuild_eur = (
            ev_da_charge_eur
            + ev_da_discharge_eur
            + ev_afrr_pos_eur
            + ev_afrr_neg_eur
            + ev_bem_only_pos_eur
            + ev_bem_only_neg_eur
            - ev_slack_penalty_pos_eur
            - ev_slack_penalty_neg_eur
            + ev_terminal_soc_credit_eur
        )
        extra_cols: dict[str, np.ndarray] = {
            "is_charging": (charge_mw > 1e-9).astype(float),
            "soc_lp_mwh": soc_lp,
            "soc_start_lp_mwh": soc_start_lp,
            "slack_pos_mw": slack_pos,
            "slack_neg_mw": slack_neg,
            "slack_soc_min_mwh": slack_soc_min[1:],
            "slack_soc_max_mwh": slack_soc_max[1:],
            "ev_pred_da_price_eur_mwh": p_da,
            "ev_pred_cap_pos_eur_mw": p_cap_pos,
            "ev_pred_cap_neg_eur_mw": p_cap_neg,
            "ev_pred_act_price_pos_eur_mwh": act_price_pos,
            "ev_pred_act_price_neg_eur_mwh": act_price_neg,
            "ev_pred_act_rate_pos": r_act_pos_base,
            "ev_pred_act_rate_neg": r_act_neg_base,
            "ev_pred_act_rate_pos_p90": r_act_pos_p90,
            "ev_pred_act_rate_neg_p90": r_act_neg_p90,
            "ev_da_charge_coef_eur_per_mw": ch_coef,
            "ev_da_discharge_coef_eur_per_mw": dis_coef,
            "ev_da_charge_eur": ev_da_charge_eur,
            "ev_da_discharge_eur": ev_da_discharge_eur,
            "ev_afrr_pos_eur": ev_afrr_pos_eur,
            "ev_afrr_neg_eur": ev_afrr_neg_eur,
            "ev_bem_only_pos_eur": ev_bem_only_pos_eur,
            "ev_bem_only_neg_eur": ev_bem_only_neg_eur,
            "ev_slack_penalty_pos_eur": ev_slack_penalty_pos_eur,
            "ev_slack_penalty_neg_eur": ev_slack_penalty_neg_eur,
            "ev_terminal_soc_credit_eur": ev_terminal_soc_credit_eur,
            "ev_objective_rebuild_eur": ev_objective_rebuild_eur,
            "final_soc_shortfall_mwh": np.full(n, final_soc_shortfall_mwh, dtype=float),
            "ev_pacc_pos_fallback_used": pacc_pos_fallback_used,
            "ev_pacc_neg_fallback_used": pacc_neg_fallback_used,
        }
        # Audit-grade hard-headroom diagnostics (optimizer-side, per hour).
        reserve_pos_mw = out["reserve_pos_mw"].to_numpy(dtype=float)
        reserve_neg_mw = out["reserve_neg_mw"].to_numpy(dtype=float)
        bem_pos_mw = out["bem_only_pos_mw"].to_numpy(dtype=float)
        bem_neg_mw = out["bem_only_neg_mw"].to_numpy(dtype=float)
        req_pos_mwh = (
            (reserve_pos_mw * self.reserve_activation_headroom_h + bem_pos_mw * self.bem_activation_headroom_h)
            / max(self.eta_out, 1e-12)
        )
        req_neg_mwh = (
            (reserve_neg_mw * self.reserve_activation_headroom_h + bem_neg_mw * self.bem_activation_headroom_h)
            * self.eta_in
        )
        avail_pos_mwh = np.maximum(0.0, soc_start_lp - self.soc_min)
        avail_neg_mwh = np.maximum(0.0, self.soc_max - soc_start_lp)
        headroom_margin_pos = avail_pos_mwh - req_pos_mwh
        headroom_margin_neg = avail_neg_mwh - req_neg_mwh
        power_stack_pos_mw = discharge_mw + reserve_pos_mw + bem_pos_mw
        power_stack_neg_mw = charge_mw + reserve_neg_mw + bem_neg_mw
        extra_cols.update(
            {
                "required_headroom_pos_mwh": req_pos_mwh,
                "required_headroom_neg_mwh": req_neg_mwh,
                "available_headroom_pos_mwh": avail_pos_mwh,
                "available_headroom_neg_mwh": avail_neg_mwh,
                "headroom_margin_pos_mwh": headroom_margin_pos,
                "headroom_margin_neg_mwh": headroom_margin_neg,
                "headroom_violation_pos_mwh": np.maximum(0.0, -headroom_margin_pos),
                "headroom_violation_neg_mwh": np.maximum(0.0, -headroom_margin_neg),
                "power_stack_pos_mw": power_stack_pos_mw,
                "power_stack_neg_mw": power_stack_neg_mw,
                "power_margin_pos_mw": self.p_max_mw - power_stack_pos_mw,
                "power_margin_neg_mw": self.p_max_mw - power_stack_neg_mw,
                "power_violation_pos_mw": np.maximum(0.0, power_stack_pos_mw - self.p_max_mw),
                "power_violation_neg_mw": np.maximum(0.0, power_stack_neg_mw - self.p_max_mw),
                "reserve_activation_headroom_h": np.full(n, self.reserve_activation_headroom_h, dtype=float),
                "bem_activation_headroom_h": np.full(n, self.bem_activation_headroom_h, dtype=float),
            }
        )
        if "optimizer_required_input_imputed_count" in df.columns:
            extra_cols["optimizer_required_input_imputed_count"] = pd.to_numeric(
                df["optimizer_required_input_imputed_count"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)
        else:
            extra_cols["optimizer_required_input_imputed_count"] = np.zeros(n, dtype=float)
        if "optimizer_required_input_imputed_any" in df.columns:
            extra_cols["optimizer_required_input_imputed_any"] = pd.to_numeric(
                df["optimizer_required_input_imputed_any"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)
        else:
            extra_cols["optimizer_required_input_imputed_any"] = np.zeros(n, dtype=float)
        if "optimizer_fallback_used" in df.columns:
            extra_cols["optimizer_fallback_used"] = pd.to_numeric(
                df["optimizer_fallback_used"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)
        else:
            extra_cols["optimizer_fallback_used"] = np.zeros(n, dtype=float)
        for b in range(n_bins):
            extra_cols[f"afrr_bin_{b}_quantile"] = np.full(n, str(self.afrr_quantile_bins[b]), dtype=object)
            extra_cols[f"afrr_bin_{b}_cap_price_pos"] = cap_price_pos_by_bin[:, b]
            extra_cols[f"afrr_bin_{b}_cap_price_neg"] = cap_price_neg_by_bin[:, b]
            extra_cols[f"reserve_pos_bin_{b}_mw"] = reserve_pos_bin_mw[:, b]
            extra_cols[f"reserve_neg_bin_{b}_mw"] = reserve_neg_bin_mw[:, b]
            extra_cols[f"bem_pos_bin_{b}_mw"] = bem_pos_bin_mw[:, b]
            extra_cols[f"bem_neg_bin_{b}_mw"] = bem_neg_bin_mw[:, b]
            extra_cols[f"ev_pacc_pos_bin_{b}"] = p_acc_cap_pos[:, b]
            extra_cols[f"ev_pacc_neg_bin_{b}"] = p_acc_cap_neg[:, b]
            extra_cols[f"ev_expected_act_share_pos_bin_{b}"] = p_acc_cap_pos[:, b] * act_rate_pos_by_bin[:, b]
            extra_cols[f"ev_expected_act_share_neg_bin_{b}"] = p_acc_cap_neg[:, b] * act_rate_neg_by_bin[:, b]
            extra_cols[f"ev_afrr_bin_{b}_act_price_pos"] = act_price_pos_by_bin[:, b]
            extra_cols[f"ev_afrr_bin_{b}_act_price_neg"] = act_price_neg_by_bin[:, b]
            extra_cols[f"ev_afrr_bin_{b}_act_rate_pos"] = act_rate_pos_by_bin[:, b]
            extra_cols[f"ev_afrr_bin_{b}_act_rate_neg"] = act_rate_neg_by_bin[:, b]
            extra_cols[f"ev_bem_bin_{b}_act_price_pos"] = act_price_pos_by_bin[:, b]
            extra_cols[f"ev_bem_bin_{b}_act_price_neg"] = act_price_neg_by_bin[:, b]
            extra_cols[f"ev_bem_bin_{b}_act_rate_pos"] = act_rate_pos_by_bin[:, b]
            extra_cols[f"ev_bem_bin_{b}_act_rate_neg"] = act_rate_neg_by_bin[:, b]
            extra_cols[f"ev_bem_bin_{b}_p_exec_pos"] = p_acc_cap_pos[:, b]
            extra_cols[f"ev_bem_bin_{b}_p_exec_neg"] = p_acc_cap_neg[:, b]
            extra_cols[f"ev_bem_pos_coef_bin_{b}_eur_per_mw"] = bem_pos_coef_by_bin[:, b]
            extra_cols[f"ev_bem_neg_coef_bin_{b}_eur_per_mw"] = bem_neg_coef_by_bin[:, b]
            extra_cols[f"ev_rpos_coef_bin_{b}_eur_per_mw"] = rpos_coef_by_bin[:, b]
            extra_cols[f"ev_rneg_coef_bin_{b}_eur_per_mw"] = rneg_coef_by_bin[:, b]
            extra_cols[f"ev_bcm_expected_capacity_revenue_pos_bin_{b}"] = bcm_expected_capacity_revenue_pos_by_bin[:, b]
            extra_cols[f"ev_bcm_expected_capacity_revenue_neg_bin_{b}"] = bcm_expected_capacity_revenue_neg_by_bin[:, b]
            extra_cols[f"ev_bcm_expected_activation_revenue_pos_bin_{b}"] = bcm_expected_activation_revenue_pos_by_bin[:, b]
            extra_cols[f"ev_bcm_expected_activation_revenue_neg_bin_{b}"] = bcm_expected_activation_revenue_neg_by_bin[:, b]
            extra_cols[f"ev_bcm_expected_aux_cost_pos_bin_{b}"] = bcm_expected_aux_cost_pos_by_bin[:, b]
            extra_cols[f"ev_bcm_expected_aux_cost_neg_bin_{b}"] = bcm_expected_aux_cost_neg_by_bin[:, b]
            extra_cols[f"ev_bcm_offer_cost_bin_{b}"] = bcm_offer_cost_by_bin[:, b]
            extra_cols[f"ev_bcm_activation_margin_pos_bin_{b}"] = bcm_activation_margin_pos_by_bin[:, b]
            extra_cols[f"ev_bcm_activation_margin_neg_bin_{b}"] = bcm_activation_margin_neg_by_bin[:, b]
            extra_cols[f"ev_bem_expected_activation_revenue_pos_bin_{b}"] = bem_expected_activation_revenue_pos_by_bin[:, b]
            extra_cols[f"ev_bem_expected_activation_revenue_neg_bin_{b}"] = bem_expected_activation_revenue_neg_by_bin[:, b]
            extra_cols[f"ev_bem_expected_aux_cost_pos_bin_{b}"] = bem_expected_aux_cost_pos_by_bin[:, b]
            extra_cols[f"ev_bem_expected_aux_cost_neg_bin_{b}"] = bem_expected_aux_cost_neg_by_bin[:, b]
            extra_cols[f"ev_bem_activation_margin_pos_bin_{b}"] = bem_activation_margin_pos_by_bin[:, b]
            extra_cols[f"ev_bem_activation_margin_neg_bin_{b}"] = bem_activation_margin_neg_by_bin[:, b]
        out = pd.concat([out, pd.DataFrame(extra_cols, index=out.index)], axis=1)
        out["predicted_objective_eur"] = -sol.fun
        return out

    @staticmethod
    def _calculate_soc_delta(
        *,
        charge_mw: float,
        discharge_mw: float,
        id_charge_mw: float,
        id_discharge_mw: float,
        act_pos_mwh: float,
        act_neg_mwh: float,
        aux_mwh: float,
        battery_specs: dict[str, float],
        dt_h: float,
    ) -> float:
        """Canonical battery physics delta on SoC (single source of truth)."""
        eta_in = float(battery_specs["eta_in"])
        eta_out = float(battery_specs["eta_out"])
        # Convert grid-side activations to internal battery-energy basis.
        act_pos_internal = float(act_pos_mwh) / max(eta_out, 1e-12)
        act_neg_internal = float(act_neg_mwh) * eta_in
        # Internal charge/discharge energies.
        in_internal = (float(charge_mw) + float(id_charge_mw)) * float(dt_h) + act_neg_internal
        out_internal = (float(discharge_mw) + float(id_discharge_mw)) * float(dt_h) + act_pos_internal
        return eta_in * in_internal - out_internal / max(eta_out, 1e-12) - float(aux_mwh)

    def _settle_one_hour(
        self,
        soc: float,
        charge: float,
        discharge: float,
        reserve_pos: float,
        reserve_neg: float,
        da_price: float,
        cap_pos: float,
        cap_neg: float,
        act_pos_price: float,
        act_neg_price: float,
        act_pos_rate: float,
        act_neg_rate: float,
        cap_bid_pos: float | None = None,
        cap_bid_neg: float | None = None,
        *,
        da_charge_mw: float | None = None,
        da_discharge_mw: float | None = None,
        id_charge_mw: float = 0.0,
        id_discharge_mw: float = 0.0,
        marginal_energy_price_pos_eur_mwh: float | None = None,
        marginal_energy_price_neg_eur_mwh: float | None = None,
        average_capacity_price_product_eur_mw_h: float | None = None,
        aufschlag_eur_mwh: float | None = None,
        aufschlag_eur_mw_h: float | None = None,
        id_recourse_reason_hint: str = "none",
    ) -> tuple[float, dict[str, float]]:
        """
        3-Layer ID Rescue settlement:
        (1) Cashflow separation between DA and synthetic ID trades,
        (2) post-rescue capacity penalty check using SoC after ID rebalance,
        (3) one-hour execution lag handled by caller via pending ID setpoints.

        The synthetic ID market approximates Intraday Continuous urgency near
        delivery (e.g., T-5m closure proxy) with asymmetric liquidity premium:
        buy at (DA + 30 EUR/MWh), sell at (DA - 30 EUR/MWh). This avoids free
        lunches from costless rescue.
        """
        # Backward compatibility: if explicit DA setpoints are not provided, use
        # legacy `charge`/`discharge` arguments as DA dispatch.
        da_charge = float(charge if da_charge_mw is None else da_charge_mw)
        da_discharge = float(discharge if da_discharge_mw is None else da_discharge_mw)
        id_charge = max(0.0, float(id_charge_mw))
        id_discharge = max(0.0, float(id_discharge_mw))
        # Enforce instantaneous converter power constraints on combined DA+ID.
        id_charge = min(
            id_charge,
            max(0.0, self.p_max_mw - max(0.0, da_charge) - max(0.0, reserve_neg)),
        )
        id_discharge = min(
            id_discharge,
            max(0.0, self.p_max_mw - max(0.0, da_discharge) - max(0.0, reserve_pos)),
        )
        # Reserve-aware ID rescue is only allowed in strategies that enable ID.
        if self._strategy_permissions.allow_id:
            req_soc_min_for_pos = (
                self.soc_min
                + (max(0.0, reserve_pos) * self.reserve_activation_headroom_h) / max(self.eta_out, 1e-12)
                + float(self.reserve_headroom_safety_mwh)
            )
            req_soc_max_for_neg = (
                self.soc_max
                - (max(0.0, reserve_neg) * self.reserve_activation_headroom_h) * self.eta_in
                - float(self.reserve_headroom_safety_mwh)
            )
            soc_after_id = soc + self.eta_in * (id_charge * self.dt_h) - (id_discharge * self.dt_h) / max(self.eta_out, 1e-12)
            if soc_after_id < req_soc_min_for_pos:
                need_soc_up = req_soc_min_for_pos - soc_after_id
                add_id_charge_mw = (need_soc_up / max(self.eta_in, 1e-12)) / max(self.dt_h, 1e-12)
                add_id_charge_mw = max(
                    0.0,
                    min(
                        add_id_charge_mw,
                        self.p_max_mw - max(0.0, da_charge) - max(0.0, reserve_neg) - id_charge,
                    ),
                )
                id_charge += add_id_charge_mw
                id_discharge = 0.0
            elif soc_after_id > req_soc_max_for_neg:
                need_soc_down = soc_after_id - req_soc_max_for_neg
                add_id_discharge_mw = (need_soc_down * max(self.eta_out, 1e-12)) / max(self.dt_h, 1e-12)
                add_id_discharge_mw = max(
                    0.0,
                    min(
                        add_id_discharge_mw,
                        self.p_max_mw - max(0.0, da_discharge) - max(0.0, reserve_pos) - id_discharge,
                    ),
                )
                id_discharge += add_id_discharge_mw
                id_charge = 0.0

        # Requested internal energies for this hour.
        act_pos_internal_req = max(0.0, act_pos_rate) * reserve_pos * self.dt_h
        act_neg_internal_req = max(0.0, act_neg_rate) * reserve_neg * self.dt_h
        act_pos_internal = float(act_pos_internal_req)
        act_neg_internal = float(act_neg_internal_req)
        da_ch_internal = da_charge * self.dt_h
        da_dis_internal = da_discharge * self.dt_h
        id_ch_internal = id_charge * self.dt_h
        id_dis_internal = id_discharge * self.dt_h

        in_internal = da_ch_internal + id_ch_internal + act_neg_internal
        out_internal = da_dis_internal + id_dis_internal + act_pos_internal
        aux_power_mw, aux_state = self._state_aux_power_mw(
            charge_mw=da_charge,
            discharge_mw=da_discharge,
            reserve_pos_mw=reserve_pos,
            reserve_neg_mw=reserve_neg,
            act_pos_rate=act_pos_rate,
            act_neg_rate=act_neg_rate,
            id_charge_mw=id_charge,
            id_discharge_mw=id_discharge,
        )
        aux_mwh = aux_power_mw * self.dt_h
        aux_cost_eur = aux_mwh * float(da_price)

        # Keep SoC feasible by scaling in/out streams if needed.
        delta = self._calculate_soc_delta(
            charge_mw=da_charge,
            discharge_mw=da_discharge,
            id_charge_mw=id_charge,
            id_discharge_mw=id_discharge,
            act_pos_mwh=act_pos_internal * self.eta_out,
            act_neg_mwh=act_neg_internal / max(self.eta_in, 1e-12),
            aux_mwh=aux_mwh,
            battery_specs={"eta_in": self.eta_in, "eta_out": self.eta_out},
            dt_h=self.dt_h,
        )
        min_delta = self.soc_min - soc
        max_delta = self.soc_max - soc

        if delta < min_delta and out_internal > 0:
            excess = min_delta - delta
            scale = max(0.0, 1.0 - (excess * self.eta_out) / max(out_internal, 1e-12))
            out_internal *= scale
            da_dis_internal *= scale
            id_dis_internal *= scale
            act_pos_internal *= scale
            delta = self._calculate_soc_delta(
                charge_mw=da_ch_internal / max(self.dt_h, 1e-12),
                discharge_mw=da_dis_internal / max(self.dt_h, 1e-12),
                id_charge_mw=id_ch_internal / max(self.dt_h, 1e-12),
                id_discharge_mw=id_dis_internal / max(self.dt_h, 1e-12),
                act_pos_mwh=act_pos_internal * self.eta_out,
                act_neg_mwh=act_neg_internal / max(self.eta_in, 1e-12),
                aux_mwh=aux_mwh,
                battery_specs={"eta_in": self.eta_in, "eta_out": self.eta_out},
                dt_h=self.dt_h,
            )
        if delta > max_delta and in_internal > 0:
            excess = delta - max_delta
            scale = max(0.0, 1.0 - excess / max(self.eta_in * in_internal, 1e-12))
            in_internal *= scale
            da_ch_internal *= scale
            id_ch_internal *= scale
            act_neg_internal *= scale
            delta = self._calculate_soc_delta(
                charge_mw=da_ch_internal / max(self.dt_h, 1e-12),
                discharge_mw=da_dis_internal / max(self.dt_h, 1e-12),
                id_charge_mw=id_ch_internal / max(self.dt_h, 1e-12),
                id_discharge_mw=id_dis_internal / max(self.dt_h, 1e-12),
                act_pos_mwh=act_pos_internal * self.eta_out,
                act_neg_mwh=act_neg_internal / max(self.eta_in, 1e-12),
                aux_mwh=aux_mwh,
                battery_specs={"eta_in": self.eta_in, "eta_out": self.eta_out},
                dt_h=self.dt_h,
            )

        soc_next = float(np.clip(soc + delta, self.soc_min, self.soc_max))

        # Requested-vs-delivered activation energy for strict non-delivery accounting.
        act_pos_grid_req = act_pos_internal_req * self.eta_out
        act_neg_grid_req = act_neg_internal_req / self.eta_in

        # Settlement cashflows.
        da_buy_grid = da_ch_internal / self.eta_in
        da_sell_grid = da_dis_internal * self.eta_out
        id_buy_mwh = id_ch_internal / self.eta_in
        id_sell_mwh = id_dis_internal * self.eta_out
        id_repair_mwh = id_buy_mwh + id_sell_mwh
        act_pos_grid = act_pos_internal * self.eta_out
        act_neg_grid = act_neg_internal / self.eta_in

        rev_da = da_sell_grid * da_price
        cost_da = da_buy_grid * da_price
        # Synthetic ID rescue prices with EPEX technical caps.
        id_buy_price_eur_mwh = min(self.id_buy_price_cap_eur_mwh, float(da_price) + self.id_rescue_spread_eur_mwh)
        id_sell_price_eur_mwh = max(self.id_sell_price_floor_eur_mwh, float(da_price) - self.id_rescue_spread_eur_mwh)
        if float(id_buy_price_eur_mwh) < float(id_sell_price_eur_mwh) - 1e-12:
            raise ValueError(
                "Invalid ID price configuration: id_buy_price < id_sell_price. "
                f"buy={id_buy_price_eur_mwh:.6f}, sell={id_sell_price_eur_mwh:.6f}, "
                f"da={float(da_price):.6f}, spread={float(self.id_rescue_spread_eur_mwh):.6f}, "
                f"cap={float(self.id_buy_price_cap_eur_mwh):.6f}, floor={float(self.id_sell_price_floor_eur_mwh):.6f}"
            )
        cost_id_eur = id_buy_mwh * id_buy_price_eur_mwh
        revenue_id_eur = id_sell_mwh * id_sell_price_eur_mwh
        id_trade_mwh = id_buy_mwh + id_sell_mwh
        id_slippage_cost_eur = id_trade_mwh * self.id_rescue_spread_eur_mwh
        id_repair_reason = "none"
        id_trade_type = "none"
        if id_trade_mwh > 1e-12:
            if self._strategy_permissions.id_mode == "economic":
                # Current ID implementation is reserve/SoC rescue logic.
                # Keep explicit tagging to avoid silent baseline contamination.
                id_trade_type = "technical_repair"
                id_repair_reason = "soc_or_obligation_repair"
            elif self._strategy_permissions.id_mode == "technical_repair":
                id_trade_type = "technical_repair"
                id_repair_reason = str(id_recourse_reason_hint or "soc_or_obligation_repair")

        # Capacity non-delivery check on post-rescue SoC (ID already applied).
        soc_for_capacity = soc + self.eta_in * id_ch_internal - id_dis_internal / self.eta_out
        soc_for_capacity = float(np.clip(soc_for_capacity, self.soc_min, self.soc_max))
        # Inverter-aware capacity support: both SoC headroom/footroom and
        # residual converter power after DA+ID setpoints must allow reserve.
        soc_pos_limit_mw = max(
            0.0,
            (soc_for_capacity - self.soc_min)
            * self.eta_out
            / max(self.reserve_activation_headroom_h, 1e-12),
        )
        soc_neg_limit_mw = max(
            0.0,
            (self.soc_max - soc_for_capacity)
            / max(self.eta_in * self.reserve_activation_headroom_h, 1e-12),
        )
        inverter_pos_limit_mw = max(
            0.0, self.p_max_mw - max(0.0, da_discharge) - max(0.0, id_discharge)
        )
        inverter_neg_limit_mw = max(
            0.0, self.p_max_mw - max(0.0, da_charge) - max(0.0, id_charge)
        )
        max_pos_capacity_supported_mw = min(soc_pos_limit_mw, inverter_pos_limit_mw)
        max_neg_capacity_supported_mw = min(soc_neg_limit_mw, inverter_neg_limit_mw)
        missed_capacity_pos_mw = max(0.0, reserve_pos - max_pos_capacity_supported_mw)
        missed_capacity_neg_mw = max(0.0, reserve_neg - max_neg_capacity_supported_mw)
        delivered_capacity_pos_mw = max(0.0, reserve_pos - missed_capacity_pos_mw)
        delivered_capacity_neg_mw = max(0.0, reserve_neg - missed_capacity_neg_mw)

        # Capacity remuneration must follow pay-as-bid:
        # awarded_MW * submitted_capacity_bid_price * dt_h.
        cap_pos_settlement = float(cap_pos if cap_bid_pos is None else cap_bid_pos)
        cap_neg_settlement = float(cap_neg if cap_bid_neg is None else cap_bid_neg)
        # Capacity remuneration is paid for awarded capacity (market award),
        # not for physically activated energy. Non-delivery is handled via penalties.
        rev_cap = (
            reserve_pos * cap_pos_settlement * self.dt_h
            + reserve_neg * cap_neg_settlement * self.dt_h
        )

        # Activation revenue is paid on delivered activation energy. We still avoid
        # double counting by not adding synthetic internal energy replacement costs
        # here; replenishment economics are reflected through subsequent DA/ID trades.
        # NEG (downward) price is stored with market-side sign. Provider-side
        # settlement cashflow is therefore the negative of that signed price.
        if self.forecast_value_mode == "canonical_economic":
            requested_activation_revenue_eur = act_pos_grid_req * act_pos_price + act_neg_grid_req * act_neg_price
            delivered_activation_revenue_eur = act_pos_grid * act_pos_price + act_neg_grid * act_neg_price
        else:
            requested_activation_revenue_eur = act_pos_grid_req * act_pos_price - act_neg_grid_req * act_neg_price
            delivered_activation_revenue_eur = act_pos_grid * act_pos_price - act_neg_grid * act_neg_price
        missed_activation_revenue_eur = max(0.0, requested_activation_revenue_eur - delivered_activation_revenue_eur)
        rev_act = float(delivered_activation_revenue_eur)
        if not self._neg_activation_sign_diagnostic_emitted:
            zero_volume_revenue = 0.0 * float(act_neg_price)
            if abs(zero_volume_revenue) > 1e-12:
                raise AssertionError(
                    "NEG activation sign diagnostic failed: volume=0 must imply zero NEG activation revenue."
                )
            neg_cashflow = (
                float(act_neg_grid) * float(act_neg_price)
                if self.forecast_value_mode == "canonical_economic"
                else -float(act_neg_grid) * float(act_neg_price)
            )
            if (
                self.forecast_value_mode == "canonical_economic"
                and float(act_neg_grid) > 1e-12
                and float(act_neg_price) > 0.0
                and neg_cashflow <= 0.0
            ):
                raise AssertionError(
                    "NEG activation sign diagnostic failed: canonical mode expects positive NEG activation value to yield positive revenue."
                )
            if (
                self.forecast_value_mode == "raw_signed"
                and float(act_neg_grid) > 1e-12
                and float(act_neg_price) < 0.0
                and neg_cashflow <= 0.0
            ):
                raise AssertionError(
                    "NEG activation sign diagnostic failed: raw_signed mode expects negative NEG price to yield positive revenue."
                )
            print(
                "[DIAG] NEG activation sign check: "
                f"volume_mwh={float(act_neg_grid):.6f}, price_eur_mwh={float(act_neg_price):.6f}, "
                f"provider_cashflow_eur={neg_cashflow:.6f}, mode={self.forecast_value_mode}"
            )
            self._neg_activation_sign_diagnostic_emitted = True
        missed_activation_mwh = max(0.0, act_pos_grid_req - act_pos_grid) + max(0.0, act_neg_grid_req - act_neg_grid)
        missed_capacity_mw = missed_capacity_pos_mw + missed_capacity_neg_mw

        # Transaction-cost accounting on total grid-side throughput:
        # C_tx = c_tx_fee * (E_DA_total + E_ID_total + abs(E_act)).
        # Here, abs(E_act) is represented by the non-negative directional components
        # act_pos_grid + act_neg_grid.
        e_da_total = da_buy_grid + da_sell_grid
        e_id_total = id_buy_mwh + id_sell_mwh
        e_act_abs = act_pos_grid + act_neg_grid
        trans_cost = self.trans_eur_mwh * (e_da_total + e_id_total + e_act_abs)
        degr_cost = self.deg_eur_mwh * (
            da_ch_internal + da_dis_internal + id_ch_internal + id_dis_internal + act_pos_internal + act_neg_internal
        )

        # Activation non-delivery penalty:
        # requested-minus-delivered activation volume settled at activation price.
        missed_activation_pos_mwh = max(0.0, act_pos_grid_req - act_pos_grid)
        missed_activation_neg_mwh = max(0.0, act_neg_grid_req - act_neg_grid)
        penalty_activation_basis_pos_eur_mwh = abs(float(act_pos_price))
        penalty_activation_basis_neg_eur_mwh = abs(float(act_neg_price))
        penalty_activation_pos_eur = missed_activation_pos_mwh * penalty_activation_basis_pos_eur_mwh
        penalty_activation_neg_eur = missed_activation_neg_mwh * penalty_activation_basis_neg_eur_mwh
        penalty_activation_eur = penalty_activation_pos_eur + penalty_activation_neg_eur

        # Capacity non-delivery penalty removed by request.
        # Keep missed-capacity diagnostics, but do not charge an economic penalty.
        penalty_capacity_basis_pos_eur_mw_h = 0.0
        penalty_capacity_basis_neg_eur_mw_h = 0.0
        penalty_capacity_pos_eur = 0.0
        penalty_capacity_neg_eur = 0.0
        penalty_capacity_eur = 0.0
        penalty_eur = penalty_activation_eur + penalty_capacity_eur

        # Cashflow excludes non-cash degradation accounting.
        net_cashflow_eur = (
            rev_da
            - cost_da
            + rev_cap
            + rev_act
            + revenue_id_eur
            - cost_id_eur
            - trans_cost
            - aux_cost_eur
            - penalty_eur
        )
        pnl = (
            rev_da
            - cost_da
            + rev_cap
            + rev_act
            + revenue_id_eur
            - cost_id_eur
            - trans_cost
            - aux_cost_eur
            - degr_cost
            - penalty_eur
        )

        metrics = {
            "soc_mwh": soc_next,
            "da_buy_mwh": da_buy_grid,
            "da_sell_mwh": da_sell_grid,
            "act_pos_mwh": act_pos_grid,
            "act_neg_mwh": act_neg_grid,
            "revenue_da_eur": rev_da,
            "cost_da_eur": cost_da,
            "revenue_capacity_eur": rev_cap,
            "revenue_activation_eur": rev_act,
            "transaction_cost_eur": trans_cost,
            "degradation_cost_eur": degr_cost,
            "missed_activation_mwh": missed_activation_mwh,
            "missed_activation_pos_mwh": missed_activation_pos_mwh,
            "missed_activation_neg_mwh": missed_activation_neg_mwh,
            "missed_capacity_mw": missed_capacity_mw,
            "missed_capacity_pos_mw": missed_capacity_pos_mw,
            "missed_capacity_neg_mw": missed_capacity_neg_mw,
            "requested_activation_revenue_eur": requested_activation_revenue_eur,
            "delivered_activation_revenue_eur": delivered_activation_revenue_eur,
            "missed_activation_revenue_eur": missed_activation_revenue_eur,
            "id_charge_mw": id_charge,
            "id_discharge_mw": id_discharge,
            "id_buy_mwh": id_buy_mwh,
            "id_sell_mwh": id_sell_mwh,
            "id_net_mwh": float(id_sell_mwh - id_buy_mwh),
            "id_buy_price_eur_mwh": float(id_buy_price_eur_mwh),
            "id_sell_price_eur_mwh": float(id_sell_price_eur_mwh),
            "revenue_id_eur": revenue_id_eur,
            "cost_id_eur": cost_id_eur,
            "id_revenue_eur": revenue_id_eur,
            "id_cost_eur": cost_id_eur,
            "id_net_pnl_eur": float(revenue_id_eur - cost_id_eur),
            "id_slippage_cost_eur": id_slippage_cost_eur,
            "id_recourse_mode": str(getattr(self, "_id_recourse_mode", "common")),
            "id_allowed": float(self._strategy_permissions.allow_id),
            "id_mode": str(self._strategy_permissions.id_mode),
            "id_economic_enabled": float(self._strategy_permissions.allow_id_economic),
            "id_technical_repair_enabled": float(self._strategy_permissions.allow_id_technical_repair),
            "id_trade_type": str(id_trade_type),
            "id_repair_reason": str(id_repair_reason),
            "id_recourse_reason": str(id_repair_reason),
            "id_repair_mwh": float(id_repair_mwh if id_trade_type == "technical_repair" else 0.0),
            "id_repair_cost_eur": float((cost_id_eur - revenue_id_eur) if id_trade_type == "technical_repair" else 0.0),
            "id_economic_mwh": float(id_trade_mwh if id_trade_type == "economic" else 0.0),
            "id_economic_pnl_eur": float((revenue_id_eur - cost_id_eur) if id_trade_type == "economic" else 0.0),
            "id_technical_repair_pnl_eur": float((revenue_id_eur - cost_id_eur) if id_trade_type == "technical_repair" else 0.0),
            "pnl_id_eur": float(revenue_id_eur - cost_id_eur),
            "aux_power_mw": aux_power_mw,
            "aux_energy_mwh": aux_mwh,
            "aux_cost_eur": aux_cost_eur,
            "aux_state": aux_state,
            "soc_for_capacity_mwh": soc_for_capacity,
            "requested_activation_pos_mwh": act_pos_grid_req,
            "requested_activation_neg_mwh": act_neg_grid_req,
            "awarded_capacity_pos_mw": reserve_pos,
            "awarded_capacity_neg_mw": reserve_neg,
            "physically_deliverable_capacity_pos_mw": max_pos_capacity_supported_mw,
            "physically_deliverable_capacity_neg_mw": max_neg_capacity_supported_mw,
            "delivered_capacity_pos_mw": delivered_capacity_pos_mw,
            "delivered_capacity_neg_mw": delivered_capacity_neg_mw,
            "delivered_activation_pos_mwh": act_pos_grid,
            "delivered_activation_neg_mwh": act_neg_grid,
            "penalty_activation_pos_eur": penalty_activation_pos_eur,
            "penalty_activation_neg_eur": penalty_activation_neg_eur,
            "penalty_activation_eur": penalty_activation_eur,
            "penalty_activation_basis_pos_eur_mwh": penalty_activation_basis_pos_eur_mwh,
            "penalty_activation_basis_neg_eur_mwh": penalty_activation_basis_neg_eur_mwh,
            "penalty_capacity_pos_eur": penalty_capacity_pos_eur,
            "penalty_capacity_neg_eur": penalty_capacity_neg_eur,
            "penalty_capacity_eur": penalty_capacity_eur,
            "penalty_capacity_basis_pos_eur_mw_h": penalty_capacity_basis_pos_eur_mw_h,
            "penalty_capacity_basis_neg_eur_mw_h": penalty_capacity_basis_neg_eur_mw_h,
            "penalty_eur": penalty_eur,
            "net_cashflow_eur": net_cashflow_eur,
            "pnl_eur": pnl,
        }
        return soc_next, metrics

    def _split_activation_revenue_components(
        self,
        *,
        delivered_pos_mwh: float,
        delivered_neg_mwh: float,
        bem_only_pos_mwh: float,
        bem_only_neg_mwh: float,
        act_pos_price_eur_mwh: float,
        act_neg_price_eur_mwh: float,
    ) -> dict[str, float]:
        """Split activation revenue into BCM-linked and BEM-only components."""
        d_pos = max(0.0, float(delivered_pos_mwh))
        d_neg = max(0.0, float(delivered_neg_mwh))
        b_pos = max(0.0, float(bem_only_pos_mwh))
        b_neg = max(0.0, float(bem_only_neg_mwh))
        p_pos = float(act_pos_price_eur_mwh)
        p_neg = float(act_neg_price_eur_mwh)

        # Clamp BEM-only delivered share to total delivered activation.
        b_pos = min(b_pos, d_pos)
        b_neg = min(b_neg, d_neg)
        c_pos = max(0.0, d_pos - b_pos)
        c_neg = max(0.0, d_neg - b_neg)

        bem_pos_rev = b_pos * p_pos
        bcm_pos_rev = c_pos * p_pos
        if self.forecast_value_mode == "canonical_economic":
            bem_neg_rev = b_neg * p_neg
            bcm_neg_rev = c_neg * p_neg
        else:
            bem_neg_rev = -b_neg * p_neg
            bcm_neg_rev = -c_neg * p_neg

        return {
            "bem_only_pos_activation_revenue_eur": float(bem_pos_rev),
            "bem_only_neg_activation_revenue_eur": float(bem_neg_rev),
            "bem_only_activation_revenue_eur": float(bem_pos_rev + bem_neg_rev),
            "bcm_linked_pos_activation_revenue_eur": float(bcm_pos_rev),
            "bcm_linked_neg_activation_revenue_eur": float(bcm_neg_rev),
            "bcm_linked_activation_revenue_eur": float(bcm_pos_rev + bcm_neg_rev),
            "activation_revenue_reconciled_eur": float(bem_pos_rev + bem_neg_rev + bcm_pos_rev + bcm_neg_rev),
        }

    def _plan_id_rescue_for_next_hour(
        self,
        *,
        soc_next: float,
        reserve_pos_next_mw: float,
        reserve_neg_next_mw: float,
        da_charge_next_mw: float,
        da_discharge_next_mw: float,
    ) -> tuple[float, float, str]:
        """Compute ID rescue setpoints decided at t for execution in t+1."""
        id_charge = 0.0
        id_discharge = 0.0
        reason = "none"
        req_soc_min_for_pos = (
            self.soc_min
            + (max(0.0, reserve_pos_next_mw) * self.reserve_activation_headroom_h) / max(self.eta_out, 1e-12)
            + float(self.reserve_headroom_safety_mwh)
        )
        req_soc_max_for_neg = (
            self.soc_max
            - (max(0.0, reserve_neg_next_mw) * self.reserve_activation_headroom_h) * self.eta_in
            - float(self.reserve_headroom_safety_mwh)
        )
        if soc_next < req_soc_min_for_pos:
            soc_gap_up = req_soc_min_for_pos - soc_next
            needed_internal_charge_mwh = soc_gap_up / max(self.eta_in, 1e-12)
            needed_charge_mw = needed_internal_charge_mwh / max(self.dt_h, 1e-12)
            id_charge = max(0.0, min(needed_charge_mw, self.p_max_mw - max(0.0, da_charge_next_mw)))
            reason = "afrr_headroom_repair" if reserve_pos_next_mw > 1e-12 else "soc_min_repair"
        elif soc_next > req_soc_max_for_neg:
            soc_gap_down = soc_next - req_soc_max_for_neg
            needed_internal_discharge_mwh = soc_gap_down * self.eta_out
            needed_discharge_mw = needed_internal_discharge_mwh / max(self.dt_h, 1e-12)
            id_discharge = max(0.0, min(needed_discharge_mw, self.p_max_mw - max(0.0, da_discharge_next_mw)))
            reason = "afrr_headroom_repair" if reserve_neg_next_mw > 1e-12 else "soc_max_repair"
        return float(id_charge), float(id_discharge), str(reason)

    def _apply_bem_only_submission_guard(
        self,
        *,
        desired_bem_only_pos_mw: float,
        desired_bem_only_neg_mw: float,
        soc_start_mwh: float,
        locked_reserve_pos_mw: float,
        locked_reserve_neg_mw: float,
        pred_act_pos: float,
        pred_act_neg: float,
    ) -> dict[str, float | str]:
        desired_pos = max(0.0, float(desired_bem_only_pos_mw))
        desired_neg = max(0.0, float(desired_bem_only_neg_mw))
        soc_start = float(soc_start_mwh)
        locked_pos = max(0.0, float(locked_reserve_pos_mw))
        locked_neg = max(0.0, float(locked_reserve_neg_mw))
        protected_soc_min_mwh = float(self.soc_min)
        protected_soc_max_mwh = float(self.soc_max)
        if locked_pos > 0.0 or locked_neg > 0.0:
            protected_soc_min_mwh = float(
                self.soc_min
                + locked_pos * self.reserve_activation_headroom_h / max(self.eta_out, 1e-12)
            )
            protected_soc_max_mwh = float(
                self.soc_max
                - locked_neg * self.reserve_activation_headroom_h * self.eta_in
            )
        safe_pos = max(
            0.0,
            (soc_start - protected_soc_min_mwh - self.bem_only_headroom_safety_mwh)
            * max(self.eta_out, 1e-12)
            / max(self.bem_activation_headroom_h, 1e-12),
        )
        safe_neg = max(
            0.0,
            (protected_soc_max_mwh - soc_start - self.bem_only_headroom_safety_mwh)
            / max(self.eta_in * self.bem_activation_headroom_h, 1e-12),
        )
        guard_reason = "none"
        exclusivity_applied = 0.0
        if self.disable_bem_only:
            desired_pos = 0.0
            desired_neg = 0.0
            guard_reason = "disabled_by_config"
        if self.disallow_simultaneous_bem_only_pos_neg and desired_pos > 1e-12 and desired_neg > 1e-12:
            pos_ev = float(pred_act_pos) * desired_pos
            neg_ev = -float(pred_act_neg) * desired_neg
            if pos_ev >= neg_ev:
                desired_neg = 0.0
            else:
                desired_pos = 0.0
            exclusivity_applied = 1.0
            guard_reason = "pos_neg_exclusivity"

        submitted_pos = min(desired_pos, safe_pos)
        submitted_neg = min(desired_neg, safe_neg)
        if self.max_bem_only_bid_mw is not None:
            submitted_pos = min(submitted_pos, float(self.max_bem_only_bid_mw))
            submitted_neg = min(submitted_neg, float(self.max_bem_only_bid_mw))
        if submitted_pos <= 1e-9:
            submitted_pos = 0.0
        if submitted_neg <= 1e-9:
            submitted_neg = 0.0
        if guard_reason == "none" and (
            abs(submitted_pos - desired_pos) > 1e-9 or abs(submitted_neg - desired_neg) > 1e-9
        ):
            guard_reason = "protected_soc_headroom_cap"
        guard_applied = float(guard_reason != "none")
        return {
            "desired_bem_only_pos_mw": float(desired_bem_only_pos_mw),
            "desired_bem_only_neg_mw": float(desired_bem_only_neg_mw),
            "safe_bem_only_pos_mw": float(safe_pos),
            "safe_bem_only_neg_mw": float(safe_neg),
            "bem_only_submitted_pos_mw_before_guard": float(desired_bem_only_pos_mw),
            "bem_only_submitted_neg_mw_before_guard": float(desired_bem_only_neg_mw),
            "bem_only_submitted_pos_mw_after_guard": float(submitted_pos),
            "bem_only_submitted_neg_mw_after_guard": float(submitted_neg),
            "submitted_bem_only_pos_mw": float(submitted_pos),
            "submitted_bem_only_neg_mw": float(submitted_neg),
            "bem_only_pos_reduced_by_headroom_mw": float(max(0.0, desired_bem_only_pos_mw - submitted_pos)),
            "bem_only_neg_reduced_by_headroom_mw": float(max(0.0, desired_bem_only_neg_mw - submitted_neg)),
            "bem_only_headroom_guard_applied": guard_applied,
            "bem_only_headroom_guard_reason": str(guard_reason),
            "bem_only_protected_soc_min_mwh": float(protected_soc_min_mwh),
            "bem_only_protected_soc_max_mwh": float(protected_soc_max_mwh),
            "bem_only_soc_start_mwh": float(soc_start),
            "bem_only_guard_soc_now_mwh": float(soc_start),
            "bem_only_guard_protected_soc_min_mwh": float(protected_soc_min_mwh),
            "bem_only_guard_protected_soc_max_mwh": float(protected_soc_max_mwh),
            "bem_only_pos_neg_exclusivity_applied": float(exclusivity_applied),
            "bem_only_disabled_by_config": float(self.disable_bem_only),
            "max_bem_only_bid_mw": float(self.max_bem_only_bid_mw) if self.max_bem_only_bid_mw is not None else float("nan"),
        }

    def _compute_obligation_driven_protected_soc_bounds(
        self,
        *,
        soc_start_mwh: float,
        required_headroom_pos_mwh: float,
        required_headroom_neg_mwh: float,
        locked_reserve_pos_mw: float,
        locked_reserve_neg_mw: float,
        committed_bem_pos_mw: float = 0.0,
        committed_bem_neg_mw: float = 0.0,
        reserve_pos_mw: float = 0.0,
        reserve_neg_mw: float = 0.0,
    ) -> dict[str, float]:
        physical_soc_min_mwh = float(self.soc_min)
        physical_soc_max_mwh = float(self.soc_max)
        req_pos = max(0.0, float(required_headroom_pos_mwh))
        req_neg = max(0.0, float(required_headroom_neg_mwh))
        locked_pos = max(0.0, float(locked_reserve_pos_mw))
        locked_neg = max(0.0, float(locked_reserve_neg_mw))
        # Strict thesis validity: protected envelope is obligation-driven.
        # Submitted/current-hour reserve or BEM-only volumes do not create a
        # future deliverability obligation by themselves.
        ob_pos_active = float((locked_pos > 1e-9) or (req_pos > 1e-9))
        ob_neg_active = float((locked_neg > 1e-9) or (req_neg > 1e-9))
        buffer_pos = float(
            (self.reserve_headroom_safety_mwh + self.reserve_soc_projection_safety_mwh)
            if ob_pos_active > 0.5
            else 0.0
        )
        buffer_neg = float(
            (self.reserve_headroom_safety_mwh + self.reserve_soc_projection_safety_mwh)
            if ob_neg_active > 0.5
            else 0.0
        )
        protected_soc_min_mwh = float(physical_soc_min_mwh + (req_pos if ob_pos_active > 0.5 else 0.0) + buffer_pos)
        protected_soc_max_mwh = float(physical_soc_max_mwh - (req_neg if ob_neg_active > 0.5 else 0.0) - buffer_neg)
        soc_start = float(soc_start_mwh)
        protected_violation_pos = float(max(0.0, protected_soc_min_mwh - soc_start))
        protected_violation_neg = float(max(0.0, soc_start - protected_soc_max_mwh))
        physical_violation_pos = float(max(0.0, physical_soc_min_mwh - soc_start))
        physical_violation_neg = float(max(0.0, soc_start - physical_soc_max_mwh))
        violation_without_obligation = float(
            ((protected_violation_pos + protected_violation_neg) > 1e-9)
            and (ob_pos_active < 0.5)
            and (ob_neg_active < 0.5)
        )
        return {
            "physical_soc_min_mwh": physical_soc_min_mwh,
            "physical_soc_max_mwh": physical_soc_max_mwh,
            "obligation_headroom_pos_active": float(ob_pos_active),
            "obligation_headroom_neg_active": float(ob_neg_active),
            "protected_soc_min_mwh": float(protected_soc_min_mwh),
            "protected_soc_max_mwh": float(protected_soc_max_mwh),
            "protected_soc_buffer_pos_mwh": float(buffer_pos),
            "protected_soc_buffer_neg_mwh": float(buffer_neg),
            "protected_soc_violation_pos_mwh": float(protected_violation_pos),
            "protected_soc_violation_neg_mwh": float(protected_violation_neg),
            "protected_soc_violation_without_obligation": float(violation_without_obligation),
            "physical_soc_violation_pos_mwh": float(physical_violation_pos),
            "physical_soc_violation_neg_mwh": float(physical_violation_neg),
        }

    def _apply_market_clearing(
        self,
        *,
        target_time_utc: pd.Timestamp | None = None,
        is_perfect_foresight: bool = False,
        planned_charge_mw: float,
        planned_discharge_mw: float,
        planned_reserve_pos_mw: float,
        planned_reserve_neg_mw: float,
        planned_bem_only_pos_mw: float = 0.0,
        planned_bem_only_neg_mw: float = 0.0,
        pred_da_price: float,
        pred_da_price_p05: float | None = None,
        pred_da_price_p10: float | None = None,
        pred_da_price_p90: float | None = None,
        pred_da_price_p95: float | None = None,
        true_da_price: float,
        pred_cap_pos: float,
        true_cap_pos: float,
        pred_cap_neg: float,
        true_cap_neg: float,
        pred_act_pos: float,
        true_act_pos: float,
        pred_act_neg: float,
        true_act_neg: float,
        true_rate_pos: float,
        true_rate_neg: float,
        pred_rate_pos: float | None = None,
        pred_rate_neg: float | None = None,
        soc_now: float | None = None,
        pred_act_pos_q10: float | None = None,
        pred_act_pos_q50: float | None = None,
        pred_act_pos_q90: float | None = None,
        pred_act_neg_q10: float | None = None,
        pred_act_neg_q50: float | None = None,
        pred_act_neg_q90: float | None = None,
        obligation_pos_mw: float = 0.0,
        obligation_neg_mw: float = 0.0,
        obligation_energy_pos: float | None = None,
        obligation_energy_neg: float | None = None,
        planned_reserve_pos_bins_mw: list[float] | None = None,
        planned_reserve_neg_bins_mw: list[float] | None = None,
        pred_cap_pos_bins_eur_mw: list[float] | None = None,
        pred_cap_neg_bins_eur_mw: list[float] | None = None,
    ) -> dict[str, float | str]:
        """Sequential market clearing: aFRR capacity -> DA -> aFRR activation."""
        perms = self._strategy_permissions
        allow_da = bool(perms.allow_da)
        allow_bcm = bool(perms.allow_bcm)
        allow_bem = bool(perms.allow_bem_only)
        ch_plan, dis_plan = self._normalize_da_bid(planned_charge_mw, planned_discharge_mw)
        res_pos_plan = max(0.0, float(planned_reserve_pos_mw))
        res_neg_plan = max(0.0, float(planned_reserve_neg_mw))
        bem_pos_plan = max(0.0, float(planned_bem_only_pos_mw))
        bem_neg_plan = max(0.0, float(planned_bem_only_neg_mw))
        if not allow_da:
            ch_plan = 0.0
            dis_plan = 0.0
        if not allow_bcm:
            res_pos_plan = 0.0
            res_neg_plan = 0.0
        if not allow_bem:
            bem_pos_plan = 0.0
            bem_neg_plan = 0.0
        ts = pd.to_datetime(target_time_utc, utc=True, errors="coerce")
        if pd.isna(ts):
            ts = pd.Timestamp("1970-01-01T00:00:00Z")

        ob_pos = max(0.0, float(obligation_pos_mw)) if allow_bcm else 0.0
        ob_neg = max(0.0, float(obligation_neg_mw)) if allow_bcm else 0.0
        soc_ref = float(self.soc_init if soc_now is None else soc_now)
        n_bins = int(len(self.afrr_quantile_bins))

        if self.da_bid_fail_fast_debug:
            # Debug-mode hard guard: DA limit bids must not be built from missing or
            # degenerate quantile inputs.
            buy_q_name = str(getattr(self.bid_builder, "da_buy_limit_quantile", "p90")).lower()
            sell_q_name = str(getattr(self.bid_builder, "da_sell_limit_quantile", "p10")).lower()
            da_q_map = {
                "p05": pred_da_price_p05,
                "p10": pred_da_price_p10,
                "p90": pred_da_price_p90,
                "p95": pred_da_price_p95,
            }
            if ch_plan > 0.0:
                buy_q_val = da_q_map.get(buy_q_name)
                if buy_q_val is None or not np.isfinite(float(buy_q_val)):
                    raise RuntimeError(
                        f"DA fail-fast: missing/invalid buy quantile '{buy_q_name}' at {ts}; "
                        f"planned_charge_mw={ch_plan:.4f}, pred_da_price={float(pred_da_price):.6f}"
                    )
                if abs(float(buy_q_val)) <= 1e-12:
                    raise RuntimeError(
                        f"DA fail-fast: zero buy quantile '{buy_q_name}' at {ts}; "
                        f"planned_charge_mw={ch_plan:.4f}, pred_da_price={float(pred_da_price):.6f}"
                    )
            if dis_plan > 0.0:
                sell_q_val = da_q_map.get(sell_q_name)
                if sell_q_val is None or not np.isfinite(float(sell_q_val)):
                    raise RuntimeError(
                        f"DA fail-fast: missing/invalid sell quantile '{sell_q_name}' at {ts}; "
                        f"planned_discharge_mw={dis_plan:.4f}, pred_da_price={float(pred_da_price):.6f}"
                    )
                if abs(float(sell_q_val)) <= 1e-12:
                    raise RuntimeError(
                        f"DA fail-fast: zero sell quantile '{sell_q_name}' at {ts}; "
                        f"planned_discharge_mw={dis_plan:.4f}, pred_da_price={float(pred_da_price):.6f}"
                    )

        def _qlevel(q: str) -> float:
            try:
                return float(q.replace("p", "")) / 100.0
            except Exception:
                return float("nan")

        # Per-bin transparency payload (filled later with submitted/awarded/activated values).
        bin_payload: dict[str, float | str] = {}
        for b, q in enumerate(self.afrr_quantile_bins):
            q_level = _qlevel(str(q))
            bin_payload[f"afrr_bin_{b}_quantile"] = str(q)
            bin_payload[f"afrr_bin_{b}_quantile_level"] = float(q_level)
            bin_payload[f"submitted_afrr_pos_bin_{b}_mw"] = 0.0
            bin_payload[f"submitted_afrr_neg_bin_{b}_mw"] = 0.0
            bin_payload[f"submitted_afrr_pos_bin_{b}_price_eur_mw"] = 0.0
            bin_payload[f"submitted_afrr_neg_bin_{b}_price_eur_mw"] = 0.0
            bin_payload[f"executed_afrr_cap_pos_bin_{b}_mw"] = 0.0
            bin_payload[f"executed_afrr_cap_neg_bin_{b}_mw"] = 0.0
            bin_payload[f"executed_afrr_cap_pos_bin_{b}_price_eur_mw"] = 0.0
            bin_payload[f"executed_afrr_cap_neg_bin_{b}_price_eur_mw"] = 0.0
            bin_payload[f"executed_afrr_act_pos_bin_{b}_mw"] = 0.0
            bin_payload[f"executed_afrr_act_neg_bin_{b}_mw"] = 0.0
            bin_payload[f"executed_afrr_act_pos_bin_{b}_price_eur_mwh"] = 0.0
            bin_payload[f"executed_afrr_act_neg_bin_{b}_price_eur_mwh"] = 0.0
        if ob_pos > 0.0 or ob_neg > 0.0:
            # Capacity already cleared in aFRR BCM: use mandatory obligation.
            cap_bids = []
            if ob_pos > 0.0:
                cap_bids.append(
                    AFRRCapacityBid(
                        ts=ts,
                        side="pos",
                        quantity_mw=ob_pos,
                        capacity_price_eur_mw=float(pred_cap_pos),
                        energy_price_eur_mwh=self.bid_builder.dynamic_afrr_energy_price(
                            side="pos",
                            pred_act_price=float(pred_act_pos if not (obligation_energy_pos is not None and np.isfinite(obligation_energy_pos)) else obligation_energy_pos),
                            soc_now_mwh=soc_ref,
                            soc_min_mwh=self.soc_min,
                            soc_max_mwh=self.soc_max,
                            obligation_mw=ob_pos,
                            delivery_duration_h=self.dt_h,
                            q10=float(pred_act_pos_q10) if pred_act_pos_q10 is not None and np.isfinite(pred_act_pos_q10) else None,
                            q50=float(pred_act_pos_q50) if pred_act_pos_q50 is not None and np.isfinite(pred_act_pos_q50) else None,
                            q90=float(pred_act_pos_q90) if pred_act_pos_q90 is not None and np.isfinite(pred_act_pos_q90) else None,
                            is_perfect_foresight=is_perfect_foresight,
                            true_act_price=float(true_act_pos),
                        ),
                    )
                )
            if ob_neg > 0.0:
                cap_bids.append(
                    AFRRCapacityBid(
                        ts=ts,
                        side="neg",
                        quantity_mw=ob_neg,
                        capacity_price_eur_mw=float(pred_cap_neg),
                        energy_price_eur_mwh=self.bid_builder.dynamic_afrr_energy_price(
                            side="neg",
                            pred_act_price=float(pred_act_neg if not (obligation_energy_neg is not None and np.isfinite(obligation_energy_neg)) else obligation_energy_neg),
                            soc_now_mwh=soc_ref,
                            soc_min_mwh=self.soc_min,
                            soc_max_mwh=self.soc_max,
                            obligation_mw=ob_neg,
                            delivery_duration_h=self.dt_h,
                            q10=float(pred_act_neg_q10) if pred_act_neg_q10 is not None and np.isfinite(pred_act_neg_q10) else None,
                            q50=float(pred_act_neg_q50) if pred_act_neg_q50 is not None and np.isfinite(pred_act_neg_q50) else None,
                            q90=float(pred_act_neg_q90) if pred_act_neg_q90 is not None and np.isfinite(pred_act_neg_q90) else None,
                            is_perfect_foresight=is_perfect_foresight,
                            true_act_price=float(true_act_neg),
                        ),
                    )
                )
            cap_res = self.market_clearing_engine.clear_afrr_capacity(
                cap_bids,
                true_cap_pos=float(true_cap_pos),
                true_cap_neg=float(true_cap_neg),
            )
            # Enforce already-awarded obligations independently of spot capacity check.
            cap_res = type(cap_res)(
                submitted_pos_mw=ob_pos,
                submitted_neg_mw=ob_neg,
                awarded_pos_mw=ob_pos,
                awarded_neg_mw=ob_neg,
                pos_awarded=ob_pos > 0.0,
                neg_awarded=ob_neg > 0.0,
            )
        else:
            # No BCM obligation for this delivery hour.
            # Explicit BEM-only participation uses independent planned BEM volumes.
            bem_guard = self._apply_bem_only_submission_guard(
                desired_bem_only_pos_mw=float(bem_pos_plan),
                desired_bem_only_neg_mw=float(bem_neg_plan),
                soc_start_mwh=float(soc_ref),
                locked_reserve_pos_mw=float(ob_pos),
                locked_reserve_neg_mw=float(ob_neg),
                pred_act_pos=float(pred_act_pos),
                pred_act_neg=float(pred_act_neg),
            )
            cap_bids: list[AFRRCapacityBid] = []
            q_pos = self.bid_builder._qfloor(float(bem_guard["submitted_bem_only_pos_mw"]), self.afrr_step_mw)
            q_neg = self.bid_builder._qfloor(float(bem_guard["submitted_bem_only_neg_mw"]), self.afrr_step_mw)
            if 0.0 < q_pos < self.afrr_min_bid_size_mw:
                q_pos = 0.0
            if 0.0 < q_neg < self.afrr_min_bid_size_mw:
                q_neg = 0.0
            bin_payload["submitted_afrr_pos_bin_0_mw"] = float(q_pos)
            bin_payload["submitted_afrr_neg_bin_0_mw"] = float(q_neg)
            bin_payload["submitted_afrr_pos_bin_0_price_eur_mw"] = 0.0
            bin_payload["submitted_afrr_neg_bin_0_price_eur_mw"] = 0.0
            if q_pos > 0.0:
                cap_bids.append(
                    AFRRCapacityBid(
                        ts=ts,
                        side="pos",
                        quantity_mw=float(q_pos),
                        capacity_price_eur_mw=0.0,
                        energy_price_eur_mwh=float(pred_act_pos),
                    )
                )
            if q_neg > 0.0:
                cap_bids.append(
                    AFRRCapacityBid(
                        ts=ts,
                        side="neg",
                        quantity_mw=float(q_neg),
                        capacity_price_eur_mw=0.0,
                        energy_price_eur_mwh=float(pred_act_neg),
                    )
                )
            submitted_pos = float(sum(float(b.quantity_mw) for b in cap_bids if b.side == "pos"))
            submitted_neg = float(sum(float(b.quantity_mw) for b in cap_bids if b.side == "neg"))
            cap_res = AFRRCapacityClearingResult(
                submitted_pos_mw=submitted_pos,
                submitted_neg_mw=submitted_neg,
                awarded_pos_mw=0.0,
                awarded_neg_mw=0.0,
                pos_awarded=False,
                neg_awarded=False,
            )
            # T-25 dynamic energy update prior to activation merit-order check.
            cap_bids = [
                AFRRCapacityBid(
                    ts=b.ts,
                    side=b.side,
                    quantity_mw=b.quantity_mw,
                    capacity_price_eur_mw=b.capacity_price_eur_mw,
                    energy_price_eur_mwh=self.bid_builder.dynamic_afrr_energy_price(
                        side=b.side,
                        pred_act_price=float(pred_act_pos if b.side == "pos" else pred_act_neg),
                        soc_now_mwh=soc_ref,
                        soc_min_mwh=self.soc_min,
                        soc_max_mwh=self.soc_max,
                        obligation_mw=float(b.quantity_mw),
                        delivery_duration_h=self.dt_h,
                        q10=float(pred_act_pos_q10) if (b.side == "pos" and pred_act_pos_q10 is not None and np.isfinite(pred_act_pos_q10)) else (float(pred_act_neg_q10) if (b.side == "neg" and pred_act_neg_q10 is not None and np.isfinite(pred_act_neg_q10)) else None),
                        q50=float(pred_act_pos_q50) if (b.side == "pos" and pred_act_pos_q50 is not None and np.isfinite(pred_act_pos_q50)) else (float(pred_act_neg_q50) if (b.side == "neg" and pred_act_neg_q50 is not None and np.isfinite(pred_act_neg_q50)) else None),
                        q90=float(pred_act_pos_q90) if (b.side == "pos" and pred_act_pos_q90 is not None and np.isfinite(pred_act_pos_q90)) else (float(pred_act_neg_q90) if (b.side == "neg" and pred_act_neg_q90 is not None and np.isfinite(pred_act_neg_q90)) else None),
                        is_perfect_foresight=is_perfect_foresight,
                        true_act_price=float(true_act_pos if b.side == "pos" else true_act_neg),
                    ),
                )
                for b in cap_bids
            ]
        if ob_pos > 0.0 or ob_neg > 0.0:
            bem_guard = self._apply_bem_only_submission_guard(
                desired_bem_only_pos_mw=0.0,
                desired_bem_only_neg_mw=0.0,
                soc_start_mwh=float(soc_ref),
                locked_reserve_pos_mw=float(ob_pos),
                locked_reserve_neg_mw=float(ob_neg),
                pred_act_pos=float(pred_act_pos),
                pred_act_neg=float(pred_act_neg),
            )
        if allow_da:
            da_bids = self.bid_builder.build_da_bids_from_plan(
                ts=ts,
                planned_charge_mw=ch_plan,
                planned_discharge_mw=dis_plan,
                obligation_pos_mw=ob_pos if ob_pos > 0.0 else cap_res.awarded_pos_mw,
                obligation_neg_mw=ob_neg if ob_neg > 0.0 else cap_res.awarded_neg_mw,
                pred_da_price=float(pred_da_price),
                pred_da_price_p05=float(pred_da_price_p05) if pred_da_price_p05 is not None and np.isfinite(pred_da_price_p05) else None,
                pred_da_price_p10=float(pred_da_price_p10) if pred_da_price_p10 is not None and np.isfinite(pred_da_price_p10) else None,
                pred_da_price_p90=float(pred_da_price_p90) if pred_da_price_p90 is not None and np.isfinite(pred_da_price_p90) else None,
                pred_da_price_p95=float(pred_da_price_p95) if pred_da_price_p95 is not None and np.isfinite(pred_da_price_p95) else None,
                is_perfect_foresight=is_perfect_foresight,
            )
        else:
            da_bids = []
        if ob_pos > 0.0 or ob_neg > 0.0:
            # Phase 2 DA restriction: available power after aFRR capacity lock.
            available_da = max(0.0, self.p_max_mw - max(ob_pos, ob_neg))
            capped_da_bids = []
            for b in da_bids:
                q = min(float(b.quantity_mw), available_da)
                if q <= 0.0:
                    continue
                capped_da_bids.append(type(b)(ts=b.ts, side=b.side, quantity_mw=q, price_eur_mwh=b.price_eur_mwh, mode=b.mode, reason=b.reason))
            da_bids = capped_da_bids
        da_res = self.market_clearing_engine.clear_da(
            da_bids,
            true_da_price=float(true_da_price),
        )
        submitted_da_buy_prices = [float(b.price_eur_mwh) for b in da_bids if b.side == "buy" and float(b.quantity_mw) > 0.0]
        submitted_da_sell_prices = [float(b.price_eur_mwh) for b in da_bids if b.side == "sell" and float(b.quantity_mw) > 0.0]
        submitted_da_buy_price = float(np.mean(submitted_da_buy_prices)) if submitted_da_buy_prices else float("nan")
        submitted_da_sell_price = float(np.mean(submitted_da_sell_prices)) if submitted_da_sell_prices else float("nan")
        # aFRR BEM (balancing energy market):
        # - with BCM obligation: activation eligibility comes from awarded BCM capacity
        # - without BCM obligation: activation eligibility comes from submitted BEM volume
        if ob_pos > 0.0 or ob_neg > 0.0:
            act_cap_res = cap_res
        else:
            act_cap_res = AFRRCapacityClearingResult(
                submitted_pos_mw=float(cap_res.submitted_pos_mw),
                submitted_neg_mw=float(cap_res.submitted_neg_mw),
                awarded_pos_mw=float(cap_res.submitted_pos_mw),
                awarded_neg_mw=float(cap_res.submitted_neg_mw),
                pos_awarded=float(cap_res.submitted_pos_mw) > 1e-12,
                neg_awarded=float(cap_res.submitted_neg_mw) > 1e-12,
            )
        act_res = self.market_clearing_engine.clear_afrr_activation(
            cap_bids,
            act_cap_res,
            true_act_pos=float(true_act_pos),
            true_act_neg=float(true_act_neg),
            true_rate_pos=float(true_rate_pos),
            true_rate_neg=float(true_rate_neg),
        )
        # Pay-as-bid settlement price for awarded capacity (weighted by awarded MW).
        def _weighted_awarded_bid_price(
            *,
            side: str,
            awarded_mw: float,
            true_cap_price: float,
        ) -> float:
            aw = max(0.0, float(awarded_mw))
            if aw <= 1e-12:
                return 0.0
            side_bids = [b for b in cap_bids if b.side == side and float(b.quantity_mw) > 0.0]
            if not side_bids:
                return 0.0
            accepted = [
                b for b in side_bids if float(b.capacity_price_eur_mw) <= float(true_cap_price) + 1e-12
            ]
            accepted_qty = float(sum(float(b.quantity_mw) for b in accepted))
            if accepted_qty + 1e-12 >= aw and accepted_qty > 1e-12:
                num = float(sum(float(b.quantity_mw) * float(b.capacity_price_eur_mw) for b in accepted))
                return float(num / max(accepted_qty, 1e-12))
            # Fallback for forced obligation awards (already-awarded lockbooks):
            # settle with submitted bid price of the obligated bids.
            sub_qty = float(sum(float(b.quantity_mw) for b in side_bids))
            num = float(sum(float(b.quantity_mw) * float(b.capacity_price_eur_mw) for b in side_bids))
            return float(num / max(sub_qty, 1e-12))

        ch_exec = float(da_res.executed_buy_mw)
        dis_exec = float(da_res.executed_sell_mw)
        if ob_pos > 0.0 or ob_neg > 0.0:
            res_pos_exec = float(cap_res.awarded_pos_mw)
            res_neg_exec = float(cap_res.awarded_neg_mw)
        else:
            res_pos_exec = float(act_cap_res.awarded_pos_mw) if bool(act_res.pos_accepted) else 0.0
            res_neg_exec = float(act_cap_res.awarded_neg_mw) if bool(act_res.neg_accepted) else 0.0
        # Capacity settlement must be pay-as-bid on awarded capacity.
        cap_bid_pos_settlement = (
            _weighted_awarded_bid_price(
                side="pos",
                awarded_mw=res_pos_exec,
                true_cap_price=float(true_cap_pos),
            )
            if ((ob_pos > 0.0 or ob_neg > 0.0) and res_pos_exec > 1e-12)
            else 0.0
        )
        cap_bid_neg_settlement = (
            _weighted_awarded_bid_price(
                side="neg",
                awarded_mw=res_neg_exec,
                true_cap_price=float(true_cap_neg),
            )
            if ((ob_pos > 0.0 or ob_neg > 0.0) and res_neg_exec > 1e-12)
            else 0.0
        )
        rate_pos_exec = float(act_res.executed_rate_pos)
        rate_neg_exec = float(act_res.executed_rate_neg)
        # Fill per-bin executed settlement transparency (capacity + activated energy).
        for b in range(n_bins):
            s_pos = float(bin_payload.get(f"submitted_afrr_pos_bin_{b}_mw", 0.0))
            s_neg = float(bin_payload.get(f"submitted_afrr_neg_bin_{b}_mw", 0.0))
            p_pos = float(bin_payload.get(f"submitted_afrr_pos_bin_{b}_price_eur_mw", 0.0))
            p_neg = float(bin_payload.get(f"submitted_afrr_neg_bin_{b}_price_eur_mw", 0.0))
            if ob_pos > 0.0 or ob_neg > 0.0:
                aw_pos = s_pos if (s_pos > 0.0 and p_pos <= float(true_cap_pos) + 1e-12) else 0.0
                aw_neg = s_neg if (s_neg > 0.0 and p_neg <= float(true_cap_neg) + 1e-12) else 0.0
            else:
                aw_pos = 0.0
                aw_neg = 0.0
            bin_payload[f"executed_afrr_cap_pos_bin_{b}_mw"] = float(aw_pos)
            bin_payload[f"executed_afrr_cap_neg_bin_{b}_mw"] = float(aw_neg)
            bin_payload[f"executed_afrr_cap_pos_bin_{b}_price_eur_mw"] = float(p_pos if aw_pos > 0.0 else 0.0)
            bin_payload[f"executed_afrr_cap_neg_bin_{b}_price_eur_mw"] = float(p_neg if aw_neg > 0.0 else 0.0)
            if ob_pos > 0.0 or ob_neg > 0.0:
                act_base_pos = aw_pos
                act_base_neg = aw_neg
            else:
                act_base_pos = s_pos
                act_base_neg = s_neg
            act_mw_pos = float(act_base_pos * rate_pos_exec if act_res.pos_accepted else 0.0)
            act_mw_neg = float(act_base_neg * rate_neg_exec if act_res.neg_accepted else 0.0)
            bin_payload[f"executed_afrr_act_pos_bin_{b}_mw"] = float(act_mw_pos)
            bin_payload[f"executed_afrr_act_neg_bin_{b}_mw"] = float(act_mw_neg)
            bin_payload[f"executed_afrr_act_pos_bin_{b}_price_eur_mwh"] = float(true_act_pos if act_mw_pos > 0.0 else 0.0)
            bin_payload[f"executed_afrr_act_neg_bin_{b}_price_eur_mwh"] = float(true_act_neg if act_mw_neg > 0.0 else 0.0)
        da_buy_accepted = bool(da_res.buy_accepted)
        da_sell_accepted = bool(da_res.sell_accepted)
        cap_pos_awarded = bool(cap_res.pos_awarded)
        cap_neg_awarded = bool(cap_res.neg_awarded)
        act_pos_accepted = bool(act_res.pos_accepted)
        act_neg_accepted = bool(act_res.neg_accepted)
        is_bcm_obligation_hour = bool((ob_pos > 0.0) or (ob_neg > 0.0))
        # Approximate BEM-only mode: outside BCM obligation hours, submitted
        # reserve volume is reused as activation-eligible BEM volume.
        bem_only_submitted_pos_mw = float(cap_res.submitted_pos_mw) if not is_bcm_obligation_hour else 0.0
        bem_only_submitted_neg_mw = float(cap_res.submitted_neg_mw) if not is_bcm_obligation_hour else 0.0
        bem_only_executed_pos_mw = float(res_pos_exec) if (not is_bcm_obligation_hour and act_pos_accepted) else 0.0
        bem_only_executed_neg_mw = float(res_neg_exec) if (not is_bcm_obligation_hour and act_neg_accepted) else 0.0
        bem_only_executed_pos_mwh = float(bem_only_executed_pos_mw * rate_pos_exec * self.dt_h)
        bem_only_executed_neg_mwh = float(bem_only_executed_neg_mw * rate_neg_exec * self.dt_h)
        energy_prices = [
            float(v)
            for v in (obligation_energy_pos, obligation_energy_neg)
            if v is not None and np.isfinite(v)
        ]
        mean_energy_price = float(np.mean(energy_prices)) if energy_prices else float("nan")

        return {
            "plan_charge_mw": ch_plan,
            "plan_discharge_mw": dis_plan,
            "plan_reserve_pos_mw": res_pos_plan,
            "plan_reserve_neg_mw": res_neg_plan,
            "desired_bem_only_pos_mw": float(bem_guard.get("desired_bem_only_pos_mw", 0.0)),
            "desired_bem_only_neg_mw": float(bem_guard.get("desired_bem_only_neg_mw", 0.0)),
            "safe_bem_only_pos_mw": float(bem_guard.get("safe_bem_only_pos_mw", 0.0)),
            "safe_bem_only_neg_mw": float(bem_guard.get("safe_bem_only_neg_mw", 0.0)),
            "bem_only_submitted_pos_mw_before_guard": float(
                bem_guard.get("bem_only_submitted_pos_mw_before_guard", 0.0)
            ),
            "bem_only_submitted_neg_mw_before_guard": float(
                bem_guard.get("bem_only_submitted_neg_mw_before_guard", 0.0)
            ),
            "bem_only_submitted_pos_mw_after_guard": float(
                bem_guard.get("bem_only_submitted_pos_mw_after_guard", 0.0)
            ),
            "bem_only_submitted_neg_mw_after_guard": float(
                bem_guard.get("bem_only_submitted_neg_mw_after_guard", 0.0)
            ),
            "submitted_bem_only_pos_mw": float(bem_guard.get("submitted_bem_only_pos_mw", 0.0)),
            "submitted_bem_only_neg_mw": float(bem_guard.get("submitted_bem_only_neg_mw", 0.0)),
            "bem_only_pos_reduced_by_headroom_mw": float(bem_guard.get("bem_only_pos_reduced_by_headroom_mw", 0.0)),
            "bem_only_neg_reduced_by_headroom_mw": float(bem_guard.get("bem_only_neg_reduced_by_headroom_mw", 0.0)),
            "bem_only_headroom_guard_applied": float(bem_guard.get("bem_only_headroom_guard_applied", 0.0)),
            "bem_only_headroom_guard_reason": str(bem_guard.get("bem_only_headroom_guard_reason", "none")),
            "bem_only_protected_soc_min_mwh": float(bem_guard.get("bem_only_protected_soc_min_mwh", self.soc_min)),
            "bem_only_protected_soc_max_mwh": float(bem_guard.get("bem_only_protected_soc_max_mwh", self.soc_max)),
            "bem_only_soc_start_mwh": float(bem_guard.get("bem_only_soc_start_mwh", soc_ref)),
            "bem_only_guard_soc_now_mwh": float(bem_guard.get("bem_only_guard_soc_now_mwh", soc_ref)),
            "bem_only_guard_protected_soc_min_mwh": float(
                bem_guard.get("bem_only_guard_protected_soc_min_mwh", self.soc_min)
            ),
            "bem_only_guard_protected_soc_max_mwh": float(
                bem_guard.get("bem_only_guard_protected_soc_max_mwh", self.soc_max)
            ),
            "bem_only_pos_neg_exclusivity_applied": float(bem_guard.get("bem_only_pos_neg_exclusivity_applied", 0.0)),
            "bem_only_disabled_by_config": float(bem_guard.get("bem_only_disabled_by_config", 0.0)),
            "max_bem_only_bid_mw": float(bem_guard.get("max_bem_only_bid_mw", float("nan"))),
            "submitted_da_buy_mw": float(da_res.submitted_buy_mw),
            "submitted_da_sell_mw": float(da_res.submitted_sell_mw),
            "submitted_da_buy_price_eur_mwh": submitted_da_buy_price,
            "submitted_da_sell_price_eur_mwh": submitted_da_sell_price,
            "submitted_afrr_pos_mw": float(cap_res.submitted_pos_mw),
            "submitted_afrr_neg_mw": float(cap_res.submitted_neg_mw),
            "afrr_bcm_auction_cleared": float((ob_pos > 0.0) or (ob_neg > 0.0)),
            "fixed_reserve_obligation_pos_mw": float(ob_pos),
            "fixed_reserve_obligation_neg_mw": float(ob_neg),
            "afrr_bem_auction_open": 1.0,
            "afrr_bem_submitted_pos_mw": float(cap_res.submitted_pos_mw),
            "afrr_bem_submitted_neg_mw": float(cap_res.submitted_neg_mw),
            "bem_only_submitted_pos_mw": float(bem_only_submitted_pos_mw),
            "bem_only_submitted_neg_mw": float(bem_only_submitted_neg_mw),
            "bem_only_executed_pos_mw": float(bem_only_executed_pos_mw),
            "bem_only_executed_neg_mw": float(bem_only_executed_neg_mw),
            "bem_only_executed_pos_mwh": float(bem_only_executed_pos_mwh),
            "bem_only_executed_neg_mwh": float(bem_only_executed_neg_mwh),
            "executed_charge_mw": ch_exec,
            "executed_discharge_mw": dis_exec,
            "executed_reserve_pos_mw": res_pos_exec,
            "executed_reserve_neg_mw": res_neg_exec,
            "settlement_cap_bid_price_pos_eur_mw": float(cap_bid_pos_settlement),
            "settlement_cap_bid_price_neg_eur_mw": float(cap_bid_neg_settlement),
            "executed_rate_pos": rate_pos_exec,
            "executed_rate_neg": rate_neg_exec,
            "da_buy_accepted": float(da_buy_accepted),
            "da_sell_accepted": float(da_sell_accepted),
            "afrr_cap_pos_awarded": float(cap_pos_awarded),
            "afrr_cap_neg_awarded": float(cap_neg_awarded),
            "afrr_act_pos_accepted": float(act_pos_accepted),
            "afrr_act_neg_accepted": float(act_neg_accepted),
            "da_price_taker_mode": float(self.da_execution_mode == "price_taker"),
            "da_buy_reason": str(da_res.reason_buy) if da_res.reason_buy else "none",
            "da_sell_reason": str(da_res.reason_sell) if da_res.reason_sell else "none",
            "aFRR_Capacity_Won_MW": float(max(ob_pos, ob_neg, res_pos_exec, res_neg_exec)),
            "DA_Energy_Sold_MW": float(dis_exec),
            "aFRR_Energy_Price_EUR_MWh": mean_energy_price,
            "Obligation_Fulfilled": float((ob_pos <= cap_res.submitted_pos_mw + 1e-9) and (ob_neg <= cap_res.submitted_neg_mw + 1e-9)),
            "aFRR_Energy_Gate_Closure_Min": 25.0,
            **bin_payload,
        }

    @staticmethod
    def _is_gate_hour_cet(ts_utc: pd.Timestamp, hour_cet: int) -> bool:
        if pd.isna(ts_utc):
            return False
        ts_cet = ts_utc.tz_convert("Europe/Berlin")
        return int(ts_cet.hour) == int(hour_cet) and int(ts_cet.minute) == 0

    def _update_afrr_capacity_lockbooks_from_snapshot(
        self,
        *,
        snapshot_ts: pd.Timestamp,
        snapshot_plan: pd.DataFrame,
        source: pd.DataFrame,
        colmap: BacktestColumnMap,
        lock_pos: dict[pd.Timestamp, float],
        lock_neg: dict[pd.Timestamp, float],
        lock_energy_pos: dict[pd.Timestamp, float],
        lock_energy_neg: dict[pd.Timestamp, float],
        lock_source_snapshot_utc: dict[pd.Timestamp, pd.Timestamp] | None = None,
        precommit_audit_by_ts: dict[str, dict[pd.Timestamp, float | str]] | None = None,
        is_perfect_foresight: bool = False,
    ) -> dict[str, float]:
        if lock_source_snapshot_utc is None:
            lock_source_snapshot_utc = {}
        # aFRR BCM (balancing capacity market) gate closure:
        # single configured gate hour for both realized/model and benchmark paths.
        afrr_bcm_gate_hour_cet = int(self.afrr_bcm_gate_hour_cet)
        if not self._is_gate_hour_cet(snapshot_ts, afrr_bcm_gate_hour_cet):
            return {"triggered": 0.0, "rejected_mw_total": 0.0}
        if snapshot_plan.empty:
            return {"triggered": 0.0, "rejected_mw_total": 0.0}
        tgt = snapshot_plan.copy()
        tgt["target_time_utc"] = pd.to_datetime(tgt["target_time_utc"], utc=True, errors="coerce")
        tgt = tgt.dropna(subset=["target_time_utc"]).copy()
        if tgt.empty:
            return {"triggered": 0.0, "rejected_mw_total": 0.0}
        tgt["target_time_cet"] = tgt["target_time_utc"].dt.tz_convert("Europe/Berlin")
        next_day_cet = (snapshot_ts.tz_convert("Europe/Berlin") + pd.Timedelta(days=1)).normalize()
        end_day_cet = next_day_cet + pd.Timedelta(days=1)
        day_rows = tgt[(tgt["target_time_cet"] >= next_day_cet) & (tgt["target_time_cet"] < end_day_cet)].copy()
        if day_rows.empty:
            return {"triggered": 0.0, "rejected_mw_total": 0.0}
        rejected_total = 0.0

        for block_start in range(0, 24, 4):
            block_end = block_start + 4
            blk = day_rows[
                (day_rows["target_time_cet"].dt.hour >= block_start)
                & (day_rows["target_time_cet"].dt.hour < block_end)
            ].copy()
            if blk.empty:
                continue

            offered_pos = self.bid_builder._qfloor(float(pd.to_numeric(blk["reserve_pos_mw"], errors="coerce").fillna(0.0).mean()), self.afrr_bid_granularity_mw)
            offered_neg = self.bid_builder._qfloor(float(pd.to_numeric(blk["reserve_neg_mw"], errors="coerce").fillna(0.0).mean()), self.afrr_bid_granularity_mw)
            desired_reserve_pos_mw = float(offered_pos)
            desired_reserve_neg_mw = float(offered_neg)
            precommit_clamp_reason = "none"
            precommit_reduction_reason = "none"
            precommit_applied = 0.0
            precommit_feasible_pos = float(offered_pos)
            precommit_feasible_neg = float(offered_neg)
            precommit_orig_pos = float(offered_pos)
            precommit_orig_neg = float(offered_neg)
            precommit_headroom_margin_min_mwh = np.nan
            precommit_power_margin_min_mw = np.nan
            precommit_soc_margin_min_mwh = np.nan
            precommit_aux_loss_margin_mwh = np.nan
            precommit_lockbook_ob_pos = 0.0
            precommit_lockbook_ob_neg = 0.0
            precommit_margin_after_bid_min_mwh = np.nan
            precommit_zeroed_due_to_margin = 0.0
            precommit_reduced_due_to_margin = 0.0
            precommit_safe_pos_mw = float(offered_pos)
            precommit_safe_neg_mw = float(offered_neg)
            precommit_submitted_pos_mw_after_derate_cap = float(offered_pos)
            precommit_submitted_neg_mw_after_derate_cap = float(offered_neg)
            submitted_reserve_pos_mw_before_retry = float(offered_pos)
            submitted_reserve_neg_mw_before_retry = float(offered_neg)
            reserve_retry_factor = 1.0
            submitted_reserve_pos_mw_after_retry = float(offered_pos)
            submitted_reserve_neg_mw_after_retry = float(offered_neg)
            retry_reduction_reason = "none"
            precommit_headroom_recharge_cost_eur = 0.0
            precommit_headroom_opportunity_cost_eur = 0.0
            precommit_net_capacity_ev_after_headroom_cost_eur = 0.0
            precommit_bid_zeroed_due_to_negative_ev = 0.0

            # Pre-commitment feasibility clamp for BCM capacity offers:
            # ensure offered reserve is physically supportable across the
            # delivery block under configured headroom assumptions.
            if "soc_start_lp_mwh" in blk.columns:
                def _blk_series(name: str, fallback: float = 0.0) -> pd.Series:
                    if name in blk.columns:
                        return pd.to_numeric(blk[name], errors="coerce").fillna(fallback)
                    return pd.Series(fallback, index=blk.index, dtype=float)

                soc_start = pd.to_numeric(blk["soc_start_lp_mwh"], errors="coerce").fillna(float(self.soc_init))
                dis_mw = _blk_series("discharge_mw", 0.0)
                ch_mw = _blk_series("charge_mw", 0.0)
                bem_pos = _blk_series("bem_only_pos_mw", 0.0)
                bem_neg = _blk_series("bem_only_neg_mw", 0.0)
                id_dis = _blk_series("id_discharge_mw", 0.0)
                if "id_discharge_mw" not in blk.columns:
                    id_dis = _blk_series("pending_id_discharge_mw", 0.0)
                id_ch = _blk_series("id_charge_mw", 0.0)
                if "id_charge_mw" not in blk.columns:
                    id_ch = _blk_series("pending_id_charge_mw", 0.0)
                aux_mwh = _blk_series("aux_power_mw", 0.0) * float(self.dt_h)
                ts_blk = pd.to_datetime(blk["target_time_utc"], utc=True, errors="coerce")
                projected_soc_by_ts = {
                    pd.to_datetime(ts, utc=True): float(sv)
                    for ts, sv in zip(ts_blk, soc_start.to_numpy(dtype=float), strict=False)
                    if pd.notna(ts)
                }
                ob_pos_existing = ts_blk.map(lambda ts: float(lock_pos.get(ts, 0.0)) if pd.notna(ts) else 0.0).astype(float)
                ob_neg_existing = ts_blk.map(lambda ts: float(lock_neg.get(ts, 0.0)) if pd.notna(ts) else 0.0).astype(float)

                avail_pos_mwh = (soc_start - float(self.soc_min)).clip(lower=0.0) - aux_mwh - (
                    bem_pos * self.bem_activation_headroom_h / max(self.eta_out, 1e-12)
                ) - float(self.reserve_headroom_safety_mwh) - float(self.reserve_soc_projection_safety_mwh)
                avail_neg_mwh = (float(self.soc_max) - soc_start).clip(lower=0.0) - (
                    bem_neg * self.bem_activation_headroom_h * self.eta_in
                ) - float(self.reserve_headroom_safety_mwh) - float(self.reserve_soc_projection_safety_mwh)
                soc_feas_pos_mw = (
                    avail_pos_mwh.clip(lower=0.0) * max(self.eta_out, 1e-12) / max(self.reserve_activation_headroom_h, 1e-12)
                )
                soc_feas_neg_mw = (
                    avail_neg_mwh.clip(lower=0.0) / max(self.eta_in * self.reserve_activation_headroom_h, 1e-12)
                )
                p_feas_pos_mw = (
                    float(self.p_max_mw)
                    - dis_mw
                    - id_dis
                    - bem_pos
                    - ob_pos_existing
                    - float(self.reserve_power_safety_mw)
                ).clip(lower=0.0)
                p_feas_neg_mw = (
                    float(self.p_max_mw)
                    - ch_mw
                    - id_ch
                    - bem_neg
                    - ob_neg_existing
                    - float(self.reserve_power_safety_mw)
                ).clip(lower=0.0)
                # Remaining SoC-supportable room after already-locked obligations.
                soc_rem_pos_mw = (soc_feas_pos_mw - ob_pos_existing).clip(lower=0.0)
                soc_rem_neg_mw = (soc_feas_neg_mw - ob_neg_existing).clip(lower=0.0)
                block_pos_cap = float(
                    np.nanmin(np.minimum(soc_rem_pos_mw.to_numpy(dtype=float), p_feas_pos_mw.to_numpy(dtype=float)))
                )
                block_neg_cap = float(
                    np.nanmin(np.minimum(soc_rem_neg_mw.to_numpy(dtype=float), p_feas_neg_mw.to_numpy(dtype=float)))
                )
                precommit_feasible_pos = float(max(0.0, block_pos_cap))
                precommit_feasible_neg = float(max(0.0, block_neg_cap))
                precommit_headroom_margin_min_mwh = float(
                    np.nanmin(np.minimum(avail_pos_mwh.to_numpy(dtype=float), avail_neg_mwh.to_numpy(dtype=float)))
                )
                precommit_power_margin_min_mw = float(
                    np.nanmin(np.minimum(p_feas_pos_mw.to_numpy(dtype=float), p_feas_neg_mw.to_numpy(dtype=float)))
                )
                precommit_soc_margin_min_mwh = precommit_headroom_margin_min_mwh
                precommit_aux_loss_margin_mwh = float(np.nanmean(aux_mwh.to_numpy(dtype=float)))
                precommit_lockbook_ob_pos = float(np.nanmax(ob_pos_existing.to_numpy(dtype=float)))
                precommit_lockbook_ob_neg = float(np.nanmax(ob_neg_existing.to_numpy(dtype=float)))
                offered_pos_before = float(offered_pos)
                offered_neg_before = float(offered_neg)
                offered_pos = min(
                    offered_pos,
                    self.bid_builder._qfloor(max(0.0, block_pos_cap), self.afrr_bid_granularity_mw),
                )
                offered_neg = min(
                    offered_neg,
                    self.bid_builder._qfloor(max(0.0, block_neg_cap), self.afrr_bid_granularity_mw),
                )
                precommit_safe_pos_mw = float(offered_pos)
                precommit_safe_neg_mw = float(offered_neg)
                # Conservative feasibility guard: avoid marginal offers near infeasibility.
                if self.reserve_feasibility_mode == "conservative":
                    pos_margin_mwh = (
                        (soc_rem_pos_mw - float(offered_pos)).to_numpy(dtype=float)
                        * max(self.reserve_activation_headroom_h, 1e-12)
                        / max(self.eta_out, 1e-12)
                    )
                    neg_margin_mwh = (
                        (soc_rem_neg_mw - float(offered_neg)).to_numpy(dtype=float)
                        * max(self.reserve_activation_headroom_h, 1e-12)
                        * max(self.eta_in, 1e-12)
                    )
                    margins = []
                    if float(offered_pos) > 1e-9 and len(pos_margin_mwh):
                        margins.append(float(np.nanmin(pos_margin_mwh)))
                    if float(offered_neg) > 1e-9 and len(neg_margin_mwh):
                        margins.append(float(np.nanmin(neg_margin_mwh)))
                    if margins:
                        precommit_margin_after_bid_min_mwh = float(min(margins))
                        if precommit_margin_after_bid_min_mwh + 1e-12 < float(self.reserve_min_margin_after_bid_mwh):
                            margin_target = float(self.reserve_min_margin_after_bid_mwh)
                            safe_pos_cap = float(offered_pos)
                            safe_neg_cap = float(offered_neg)
                            if float(offered_pos) > 1e-9:
                                pos_safe_by_margin = (
                                    soc_rem_pos_mw
                                    - margin_target * max(self.eta_out, 1e-12) / max(self.reserve_activation_headroom_h, 1e-12)
                                ).clip(lower=0.0)
                                safe_pos_cap = float(np.nanmin(pos_safe_by_margin.to_numpy(dtype=float)))
                            if float(offered_neg) > 1e-9:
                                neg_safe_by_margin = (
                                    soc_rem_neg_mw
                                    - margin_target / max(self.eta_in * self.reserve_activation_headroom_h, 1e-12)
                                ).clip(lower=0.0)
                                safe_neg_cap = float(np.nanmin(neg_safe_by_margin.to_numpy(dtype=float)))
                            precommit_safe_pos_mw = float(
                                self.bid_builder._qfloor(max(0.0, safe_pos_cap), self.afrr_bid_granularity_mw)
                            )
                            precommit_safe_neg_mw = float(
                                self.bid_builder._qfloor(max(0.0, safe_neg_cap), self.afrr_bid_granularity_mw)
                            )
                            new_pos = float(min(float(offered_pos), precommit_safe_pos_mw))
                            new_neg = float(min(float(offered_neg), precommit_safe_neg_mw))
                            if (new_pos + 1e-9) < float(offered_pos) or (new_neg + 1e-9) < float(offered_neg):
                                precommit_reduced_due_to_margin = 1.0
                                precommit_clamp_reason = "reduced_due_to_margin"
                                precommit_reduction_reason = "margin_below_threshold"
                                precommit_applied = 1.0
                            offered_pos = new_pos
                            offered_neg = new_neg
                            if offered_pos <= 0.0 and offered_neg <= 0.0:
                                precommit_zeroed_due_to_margin = 1.0
                                precommit_clamp_reason = "zeroed_due_to_margin"
                                precommit_reduction_reason = "margin_zeroed"
                                precommit_applied = 1.0
                if (offered_pos + 1e-9) < offered_pos_before or (offered_neg + 1e-9) < offered_neg_before:
                    precommit_clamp_reason = "power_soc_aux_lockbook_clamp"
                    precommit_reduction_reason = "power_soc_aux_lockbook"
                    precommit_applied = 1.0
                safe_after_margin_pos = float(offered_pos)
                safe_after_margin_neg = float(offered_neg)
                offered_pos = float(
                    self.bid_builder._qfloor(
                        max(0.0, safe_after_margin_pos * float(self.reserve_bid_derate)),
                        self.afrr_bid_granularity_mw,
                    )
                )
                offered_neg = float(
                    self.bid_builder._qfloor(
                        max(0.0, safe_after_margin_neg * float(self.reserve_bid_derate)),
                        self.afrr_bid_granularity_mw,
                    )
                )
                if self.max_reserve_bid_mw is not None:
                    offered_pos = float(
                        min(offered_pos, self.bid_builder._qfloor(float(self.max_reserve_bid_mw), self.afrr_bid_granularity_mw))
                    )
                    offered_neg = float(
                        min(offered_neg, self.bid_builder._qfloor(float(self.max_reserve_bid_mw), self.afrr_bid_granularity_mw))
                    )
                precommit_submitted_pos_mw_after_derate_cap = float(offered_pos)
                precommit_submitted_neg_mw_after_derate_cap = float(offered_neg)
                if (offered_pos + 1e-9) < safe_after_margin_pos or (offered_neg + 1e-9) < safe_after_margin_neg:
                    precommit_clamp_reason = "reserve_derate_or_cap"
                    precommit_reduction_reason = "derate_or_cap"
                    precommit_applied = 1.0
                submitted_reserve_pos_mw_before_retry = float(offered_pos)
                submitted_reserve_neg_mw_before_retry = float(offered_neg)
                if self.disable_new_bcm_reserve_bids:
                    offered_pos = 0.0
                    offered_neg = 0.0
                    reserve_retry_factor = 0.0
                    retry_reduction_reason = "disable_new_bcm_reserve_bids"
                elif (
                    self.reserve_feasibility_mode == "conservative"
                    and bool(self.enable_reserve_retry_ladder)
                    and len(self.reserve_retry_ladder) > 1
                ):
                    # Use ladder at precommit stage to reduce new reserve submissions
                    # before they can create infeasible lockbook obligations.
                    reserve_retry_factor = float(self.reserve_retry_ladder[-1])
                    for f in self.reserve_retry_ladder:
                        trial_pos = float(
                            self.bid_builder._qfloor(
                                max(0.0, submitted_reserve_pos_mw_before_retry * float(f)),
                                self.afrr_bid_granularity_mw,
                            )
                        )
                        trial_neg = float(
                            self.bid_builder._qfloor(
                                max(0.0, submitted_reserve_neg_mw_before_retry * float(f)),
                                self.afrr_bid_granularity_mw,
                            )
                        )
                        reserve_retry_factor = float(f)
                        offered_pos = trial_pos
                        offered_neg = trial_neg
                        if trial_pos <= 0.0 and trial_neg <= 0.0:
                            break
                    if reserve_retry_factor < 0.999:
                        retry_reduction_reason = "reserve_retry_ladder"
                submitted_reserve_pos_mw_after_retry = float(offered_pos)
                submitted_reserve_neg_mw_after_retry = float(offered_neg)

                # Precommit EV check with explicit headroom maintenance cost.
                # This is a first-line bid filter only; post-award feasibility remains hard-constrained.
                src_blk = source.reindex(ts_blk)

                def _src_series(name: str) -> pd.Series:
                    if name in src_blk.columns:
                        return pd.to_numeric(src_blk[name], errors="coerce").fillna(0.0)
                    return pd.Series(0.0, index=src_blk.index, dtype=float)

                pred_cap_pos_blk = float(_src_series(colmap.pred_afrr_capacity_price_pos).mean())
                pred_cap_neg_blk = float(_src_series(colmap.pred_afrr_capacity_price_neg).mean())
                if not np.isfinite(pred_cap_pos_blk):
                    pred_cap_pos_blk = 0.0
                if not np.isfinite(pred_cap_neg_blk):
                    pred_cap_neg_blk = 0.0
                pred_da_blk = _src_series(colmap.pred_da_price)
                recharge_price_eur_mwh = float(max(0.0, float(pred_da_blk.mean()) if len(pred_da_blk) else 0.0))
                opp_price_eur_mwh = float(max(0.0, float(pred_da_blk.quantile(0.75)) if len(pred_da_blk) else recharge_price_eur_mwh))
                req_pos_mwh = (
                    float(offered_pos) * float(self.reserve_activation_headroom_h) / max(float(self.eta_out), 1e-12)
                    + float(self.reserve_headroom_safety_mwh)
                    + float(self.reserve_soc_projection_safety_mwh)
                )
                req_neg_mwh = (
                    float(offered_neg) * float(self.reserve_activation_headroom_h) * float(self.eta_in)
                    + float(self.reserve_headroom_safety_mwh)
                    + float(self.reserve_soc_projection_safety_mwh)
                )
                short_pos = np.maximum(0.0, req_pos_mwh - avail_pos_mwh.to_numpy(dtype=float))
                short_neg = np.maximum(0.0, req_neg_mwh - avail_neg_mwh.to_numpy(dtype=float))
                # Recharge/emptying cost proxy to enforce maintainable reserve headroom economics.
                precommit_headroom_recharge_cost_eur = float(
                    (
                        (short_pos / max(self.eta_in, 1e-12))
                        + (short_neg / max(self.eta_out, 1e-12))
                    ).sum()
                    * recharge_price_eur_mwh
                )
                # Opportunity cost proxy for headroom that must be kept unavailable
                # to energy arbitrage while reserve is committed.
                reserved_headroom_mwh_block = float((req_pos_mwh + req_neg_mwh) * len(blk))
                precommit_headroom_opportunity_cost_eur = float(
                    reserved_headroom_mwh_block * opp_price_eur_mwh
                )
                block_hours = float(len(blk) * self.dt_h)
                expected_capacity_ev_eur = float(
                    float(offered_pos) * pred_cap_pos_blk * block_hours
                    + float(offered_neg) * pred_cap_neg_blk * block_hours
                )
                precommit_net_capacity_ev_after_headroom_cost_eur = float(
                    expected_capacity_ev_eur
                    - precommit_headroom_recharge_cost_eur
                    - precommit_headroom_opportunity_cost_eur
                )
                if precommit_net_capacity_ev_after_headroom_cost_eur < -1e-9 and (offered_pos > 0.0 or offered_neg > 0.0):
                    offered_pos = 0.0
                    offered_neg = 0.0
                    precommit_bid_zeroed_due_to_negative_ev = 1.0
                    precommit_clamp_reason = "zeroed_due_to_negative_ev_after_headroom_cost"
                    precommit_reduction_reason = "negative_ev_after_headroom_cost"
                    precommit_applied = 1.0

            if offered_pos <= 0.0 and offered_neg <= 0.0:
                for ts in blk["target_time_utc"]:
                    lock_pos[pd.to_datetime(ts, utc=True)] = 0.0
                    lock_neg[pd.to_datetime(ts, utc=True)] = 0.0
                    lock_source_snapshot_utc[pd.to_datetime(ts, utc=True)] = pd.to_datetime(
                        snapshot_ts, utc=True, errors="coerce"
                    )
                    if precommit_audit_by_ts is not None:
                        tsu = pd.to_datetime(ts, utc=True)
                        precommit_audit_by_ts.setdefault("precommit_clamp_applied", {})[tsu] = float(precommit_applied)
                        precommit_audit_by_ts.setdefault("precommit_clamp_reason", {})[tsu] = str(precommit_clamp_reason)
                        precommit_audit_by_ts.setdefault("precommit_reduction_reason", {})[tsu] = str(precommit_reduction_reason)
                        precommit_audit_by_ts.setdefault("desired_reserve_pos_mw", {})[tsu] = float(desired_reserve_pos_mw)
                        precommit_audit_by_ts.setdefault("desired_reserve_neg_mw", {})[tsu] = float(desired_reserve_neg_mw)
                        precommit_audit_by_ts.setdefault("safe_reserve_pos_mw", {})[tsu] = float(precommit_safe_pos_mw)
                        precommit_audit_by_ts.setdefault("safe_reserve_neg_mw", {})[tsu] = float(precommit_safe_neg_mw)
                        precommit_audit_by_ts.setdefault("submitted_reserve_pos_mw", {})[tsu] = float(
                            precommit_submitted_pos_mw_after_derate_cap
                        )
                        precommit_audit_by_ts.setdefault("submitted_reserve_neg_mw", {})[tsu] = float(
                            precommit_submitted_neg_mw_after_derate_cap
                        )
                        precommit_audit_by_ts.setdefault("submitted_reserve_pos_mw_before_retry", {})[tsu] = float(
                            submitted_reserve_pos_mw_before_retry
                        )
                        precommit_audit_by_ts.setdefault("submitted_reserve_neg_mw_before_retry", {})[tsu] = float(
                            submitted_reserve_neg_mw_before_retry
                        )
                        precommit_audit_by_ts.setdefault("reserve_retry_factor", {})[tsu] = float(reserve_retry_factor)
                        precommit_audit_by_ts.setdefault("submitted_reserve_pos_mw_after_retry", {})[tsu] = float(
                            submitted_reserve_pos_mw_after_retry
                        )
                        precommit_audit_by_ts.setdefault("submitted_reserve_neg_mw_after_retry", {})[tsu] = float(
                            submitted_reserve_neg_mw_after_retry
                        )
                        precommit_audit_by_ts.setdefault("retry_reduction_reason", {})[tsu] = str(retry_reduction_reason)
                        precommit_audit_by_ts.setdefault("precommit_feasible_pos_mw", {})[tsu] = float(precommit_feasible_pos)
                        precommit_audit_by_ts.setdefault("precommit_feasible_neg_mw", {})[tsu] = float(precommit_feasible_neg)
                        precommit_audit_by_ts.setdefault("precommit_original_pos_mw", {})[tsu] = float(precommit_orig_pos)
                        precommit_audit_by_ts.setdefault("precommit_original_neg_mw", {})[tsu] = float(precommit_orig_neg)
                        precommit_audit_by_ts.setdefault("precommit_clamped_pos_mw", {})[tsu] = float(max(0.0, precommit_orig_pos - offered_pos))
                        precommit_audit_by_ts.setdefault("precommit_clamped_neg_mw", {})[tsu] = float(max(0.0, precommit_orig_neg - offered_neg))
                        precommit_audit_by_ts.setdefault("precommit_headroom_margin_min_mwh", {})[tsu] = float(precommit_headroom_margin_min_mwh)
                        precommit_audit_by_ts.setdefault("precommit_power_margin_min_mw", {})[tsu] = float(precommit_power_margin_min_mw)
                        precommit_audit_by_ts.setdefault("precommit_soc_margin_min_mwh", {})[tsu] = float(precommit_soc_margin_min_mwh)
                        precommit_audit_by_ts.setdefault("precommit_aux_loss_margin_mwh", {})[tsu] = float(precommit_aux_loss_margin_mwh)
                        precommit_audit_by_ts.setdefault("precommit_lockbook_obligation_pos_mw", {})[tsu] = float(precommit_lockbook_ob_pos)
                        precommit_audit_by_ts.setdefault("precommit_lockbook_obligation_neg_mw", {})[tsu] = float(precommit_lockbook_ob_neg)
                        precommit_audit_by_ts.setdefault("precommit_margin_after_bid_min_mwh", {})[tsu] = float(
                            precommit_margin_after_bid_min_mwh
                        )
                        precommit_audit_by_ts.setdefault("precommit_zeroed_due_to_margin", {})[tsu] = float(
                            precommit_zeroed_due_to_margin
                        )
                        precommit_audit_by_ts.setdefault("precommit_reduced_due_to_margin", {})[tsu] = float(
                            precommit_reduced_due_to_margin
                        )
                        precommit_audit_by_ts.setdefault("precommit_safe_pos_mw", {})[tsu] = float(
                            precommit_safe_pos_mw
                        )
                        precommit_audit_by_ts.setdefault("precommit_safe_neg_mw", {})[tsu] = float(
                            precommit_safe_neg_mw
                        )
                        precommit_audit_by_ts.setdefault("precommit_submitted_pos_mw_after_derate_cap", {})[tsu] = float(
                            precommit_submitted_pos_mw_after_derate_cap
                        )
                        precommit_audit_by_ts.setdefault("precommit_submitted_neg_mw_after_derate_cap", {})[tsu] = float(
                            precommit_submitted_neg_mw_after_derate_cap
                        )
                        precommit_audit_by_ts.setdefault("reserve_bid_derate", {})[tsu] = float(self.reserve_bid_derate)
                        precommit_audit_by_ts.setdefault("max_reserve_bid_mw", {})[tsu] = float(
                            self.max_reserve_bid_mw if self.max_reserve_bid_mw is not None else np.nan
                        )
                        precommit_audit_by_ts.setdefault("precommit_headroom_recharge_cost_eur", {})[tsu] = float(
                            precommit_headroom_recharge_cost_eur
                        )
                        precommit_audit_by_ts.setdefault("precommit_headroom_opportunity_cost_eur", {})[tsu] = float(
                            precommit_headroom_opportunity_cost_eur
                        )
                        precommit_audit_by_ts.setdefault(
                            "precommit_net_capacity_ev_after_headroom_cost_eur", {}
                        )[tsu] = float(precommit_net_capacity_ev_after_headroom_cost_eur)
                        precommit_audit_by_ts.setdefault("precommit_bid_zeroed_due_to_negative_ev", {})[tsu] = float(
                            precommit_bid_zeroed_due_to_negative_ev
                        )
                continue

            # Stage 1 (forecast-only): formulate submitted capacity bids.
            cap_bids, ts_idx = self._formulate_afrr_capacity_block_bids(
                blk=blk,
                source=source,
                colmap=colmap,
                snapshot_ts=snapshot_ts,
                offered_pos=offered_pos,
                offered_neg=offered_neg,
                is_perfect_foresight=is_perfect_foresight,
            )
            # Stage 2 (isolated market clearing): compare submitted bid prices
            # to realized clearing prices to determine awarded capacity.
            cap_res = self._clear_afrr_capacity_block_against_truth(
                cap_bids=cap_bids,
                ts_idx=ts_idx,
                source=source,
                colmap=colmap,
            )
            e_pos = 0.0
            e_neg = 0.0
            block_start_utc = pd.to_datetime(blk["target_time_utc"], utc=True, errors="coerce").min()
            block_end_utc = pd.to_datetime(blk["target_time_utc"], utc=True, errors="coerce").max()
            reserve_product_block_id = f"{next_day_cet.date().isoformat()}_h{block_start:02d}-{block_end:02d}"
            reserve_commitment_id = (
                f"{pd.to_datetime(snapshot_ts, utc=True, errors='coerce').isoformat()}|{reserve_product_block_id}"
            )
            for b in cap_bids:
                if b.side == "pos":
                    e_pos = float(b.energy_price_eur_mwh)
                elif b.side == "neg":
                    e_neg = float(b.energy_price_eur_mwh)
            for ts in blk["target_time_utc"]:
                tsu = pd.to_datetime(ts, utc=True)
                lock_pos[tsu] = float(cap_res.awarded_pos_mw)
                lock_neg[tsu] = float(cap_res.awarded_neg_mw)
                lock_energy_pos[tsu] = float(e_pos)
                lock_energy_neg[tsu] = float(e_neg)
                lock_source_snapshot_utc[tsu] = pd.to_datetime(snapshot_ts, utc=True, errors="coerce")
                if precommit_audit_by_ts is not None:
                    precommit_audit_by_ts.setdefault("precommit_clamp_applied", {})[tsu] = float(precommit_applied)
                    precommit_audit_by_ts.setdefault("precommit_clamp_reason", {})[tsu] = str(precommit_clamp_reason)
                    precommit_audit_by_ts.setdefault("precommit_reduction_reason", {})[tsu] = str(precommit_reduction_reason)
                    precommit_audit_by_ts.setdefault("desired_reserve_pos_mw", {})[tsu] = float(desired_reserve_pos_mw)
                    precommit_audit_by_ts.setdefault("desired_reserve_neg_mw", {})[tsu] = float(desired_reserve_neg_mw)
                    precommit_audit_by_ts.setdefault("safe_reserve_pos_mw", {})[tsu] = float(precommit_safe_pos_mw)
                    precommit_audit_by_ts.setdefault("safe_reserve_neg_mw", {})[tsu] = float(precommit_safe_neg_mw)
                    precommit_audit_by_ts.setdefault("submitted_reserve_pos_mw", {})[tsu] = float(
                        precommit_submitted_pos_mw_after_derate_cap
                    )
                    precommit_audit_by_ts.setdefault("submitted_reserve_neg_mw", {})[tsu] = float(
                        precommit_submitted_neg_mw_after_derate_cap
                    )
                    precommit_audit_by_ts.setdefault("submitted_reserve_pos_mw_before_retry", {})[tsu] = float(
                        submitted_reserve_pos_mw_before_retry
                    )
                    precommit_audit_by_ts.setdefault("submitted_reserve_neg_mw_before_retry", {})[tsu] = float(
                        submitted_reserve_neg_mw_before_retry
                    )
                    precommit_audit_by_ts.setdefault("reserve_retry_factor", {})[tsu] = float(reserve_retry_factor)
                    precommit_audit_by_ts.setdefault("submitted_reserve_pos_mw_after_retry", {})[tsu] = float(
                        submitted_reserve_pos_mw_after_retry
                    )
                    precommit_audit_by_ts.setdefault("submitted_reserve_neg_mw_after_retry", {})[tsu] = float(
                        submitted_reserve_neg_mw_after_retry
                    )
                    precommit_audit_by_ts.setdefault("retry_reduction_reason", {})[tsu] = str(retry_reduction_reason)
                    precommit_audit_by_ts.setdefault("precommit_feasible_pos_mw", {})[tsu] = float(precommit_feasible_pos)
                    precommit_audit_by_ts.setdefault("precommit_feasible_neg_mw", {})[tsu] = float(precommit_feasible_neg)
                    precommit_audit_by_ts.setdefault("precommit_original_pos_mw", {})[tsu] = float(precommit_orig_pos)
                    precommit_audit_by_ts.setdefault("precommit_original_neg_mw", {})[tsu] = float(precommit_orig_neg)
                    precommit_audit_by_ts.setdefault("precommit_clamped_pos_mw", {})[tsu] = float(max(0.0, precommit_orig_pos - offered_pos))
                    precommit_audit_by_ts.setdefault("precommit_clamped_neg_mw", {})[tsu] = float(max(0.0, precommit_orig_neg - offered_neg))
                    precommit_audit_by_ts.setdefault("precommit_headroom_margin_min_mwh", {})[tsu] = float(precommit_headroom_margin_min_mwh)
                    precommit_audit_by_ts.setdefault("precommit_power_margin_min_mw", {})[tsu] = float(precommit_power_margin_min_mw)
                    precommit_audit_by_ts.setdefault("precommit_soc_margin_min_mwh", {})[tsu] = float(precommit_soc_margin_min_mwh)
                    precommit_audit_by_ts.setdefault("precommit_aux_loss_margin_mwh", {})[tsu] = float(precommit_aux_loss_margin_mwh)
                    precommit_audit_by_ts.setdefault("precommit_lockbook_obligation_pos_mw", {})[tsu] = float(precommit_lockbook_ob_pos)
                    precommit_audit_by_ts.setdefault("precommit_lockbook_obligation_neg_mw", {})[tsu] = float(precommit_lockbook_ob_neg)
                    precommit_audit_by_ts.setdefault("precommit_margin_after_bid_min_mwh", {})[tsu] = float(
                        precommit_margin_after_bid_min_mwh
                    )
                    precommit_audit_by_ts.setdefault("precommit_zeroed_due_to_margin", {})[tsu] = float(
                        precommit_zeroed_due_to_margin
                    )
                    precommit_audit_by_ts.setdefault("precommit_reduced_due_to_margin", {})[tsu] = float(
                        precommit_reduced_due_to_margin
                    )
                    precommit_audit_by_ts.setdefault("precommit_safe_pos_mw", {})[tsu] = float(
                        precommit_safe_pos_mw
                    )
                    precommit_audit_by_ts.setdefault("precommit_safe_neg_mw", {})[tsu] = float(
                        precommit_safe_neg_mw
                    )
                    precommit_audit_by_ts.setdefault("precommit_submitted_pos_mw_after_derate_cap", {})[tsu] = float(
                        precommit_submitted_pos_mw_after_derate_cap
                    )
                    precommit_audit_by_ts.setdefault("precommit_submitted_neg_mw_after_derate_cap", {})[tsu] = float(
                        precommit_submitted_neg_mw_after_derate_cap
                    )
                    precommit_audit_by_ts.setdefault("reserve_bid_derate", {})[tsu] = float(self.reserve_bid_derate)
                    precommit_audit_by_ts.setdefault("max_reserve_bid_mw", {})[tsu] = float(
                        self.max_reserve_bid_mw if self.max_reserve_bid_mw is not None else np.nan
                    )
                    precommit_audit_by_ts.setdefault("precommit_headroom_recharge_cost_eur", {})[tsu] = float(
                        precommit_headroom_recharge_cost_eur
                    )
                    precommit_audit_by_ts.setdefault("precommit_headroom_opportunity_cost_eur", {})[tsu] = float(
                        precommit_headroom_opportunity_cost_eur
                    )
                    precommit_audit_by_ts.setdefault(
                        "precommit_net_capacity_ev_after_headroom_cost_eur", {}
                    )[tsu] = float(precommit_net_capacity_ev_after_headroom_cost_eur)
                    precommit_audit_by_ts.setdefault("precommit_bid_zeroed_due_to_negative_ev", {})[tsu] = float(
                        precommit_bid_zeroed_due_to_negative_ev
                    )
                    precommit_audit_by_ts.setdefault("reserve_commitment_id", {})[tsu] = str(reserve_commitment_id)
                    precommit_audit_by_ts.setdefault("reserve_product_block_id", {})[tsu] = str(reserve_product_block_id)
                    precommit_audit_by_ts.setdefault("reserve_commitment_source_snapshot_utc", {})[tsu] = str(
                        pd.to_datetime(snapshot_ts, utc=True, errors="coerce").isoformat()
                    )
                    precommit_audit_by_ts.setdefault("reserve_delivery_start_utc", {})[tsu] = str(
                        pd.to_datetime(block_start_utc, utc=True, errors="coerce").isoformat()
                    )
                    precommit_audit_by_ts.setdefault("reserve_delivery_end_utc", {})[tsu] = str(
                        pd.to_datetime(block_end_utc, utc=True, errors="coerce").isoformat()
                    )
                    precommit_audit_by_ts.setdefault("reserve_precommit_feasible_pos_mw", {})[tsu] = float(precommit_feasible_pos)
                    precommit_audit_by_ts.setdefault("reserve_precommit_feasible_neg_mw", {})[tsu] = float(precommit_feasible_neg)
                    precommit_audit_by_ts.setdefault("reserve_submitted_pos_mw", {})[tsu] = float(offered_pos)
                    precommit_audit_by_ts.setdefault("reserve_submitted_neg_mw", {})[tsu] = float(offered_neg)
                    precommit_audit_by_ts.setdefault("reserve_awarded_pos_mw", {})[tsu] = float(cap_res.awarded_pos_mw)
                    precommit_audit_by_ts.setdefault("reserve_awarded_neg_mw", {})[tsu] = float(cap_res.awarded_neg_mw)
                    precommit_audit_by_ts.setdefault("reserve_lockbook_pos_mw", {})[tsu] = float(lock_pos.get(tsu, 0.0))
                    precommit_audit_by_ts.setdefault("reserve_lockbook_neg_mw", {})[tsu] = float(lock_neg.get(tsu, 0.0))
                    precommit_audit_by_ts.setdefault("reserve_projected_soc_start_mwh", {})[tsu] = float(
                        projected_soc_by_ts.get(tsu, np.nan)
                    )
            rejected_total += max(0.0, offered_pos - float(cap_res.awarded_pos_mw))
            rejected_total += max(0.0, offered_neg - float(cap_res.awarded_neg_mw))
        return {"triggered": float(rejected_total > 1e-9), "rejected_mw_total": float(rejected_total)}

    def _formulate_afrr_capacity_block_bids(
        self,
        *,
        blk: pd.DataFrame,
        source: pd.DataFrame,
        colmap: BacktestColumnMap,
        snapshot_ts: pd.Timestamp,
        offered_pos: float,
        offered_neg: float,
        is_perfect_foresight: bool = False,
    ) -> tuple[list[AFRRCapacityBid], pd.Series]:
        """Build submitted aFRR capacity bids using forecast-side information only."""
        ts_idx = pd.to_datetime(blk["target_time_utc"], utc=True, errors="coerce")
        sblk = source.reindex(ts_idx).copy()

        # Forecast-side pricing inputs only (no realized capacity clearing prices).
        pred_cap_pos = float(pd.to_numeric(sblk.get(colmap.pred_afrr_capacity_price_pos), errors="coerce").mean())
        pred_cap_neg = float(pd.to_numeric(sblk.get(colmap.pred_afrr_capacity_price_neg), errors="coerce").mean())
        pred_act_pos = float(pd.to_numeric(sblk.get(colmap.pred_afrr_activation_price_pos), errors="coerce").mean())
        pred_act_neg = float(pd.to_numeric(sblk.get(colmap.pred_afrr_activation_price_neg), errors="coerce").mean())

        if not np.isfinite(pred_cap_pos):
            pred_cap_pos = 0.0
        if not np.isfinite(pred_cap_neg):
            pred_cap_neg = 0.0
        if not np.isfinite(pred_act_pos):
            pred_act_pos = 0.0
        if not np.isfinite(pred_act_neg):
            pred_act_neg = 0.0

        cap_bids = self.bid_builder.build_afrr_capacity_bids(
            ts=ts_idx.iloc[0] if len(ts_idx) else snapshot_ts,
            reserve_pos_mw=offered_pos,
            reserve_neg_mw=offered_neg,
            pred_cap_pos=pred_cap_pos,
            pred_cap_neg=pred_cap_neg,
            pred_act_pos=pred_act_pos,
            pred_act_neg=pred_act_neg,
            is_perfect_foresight=is_perfect_foresight,
        )
        return cap_bids, ts_idx

    def _clear_afrr_capacity_block_against_truth(
        self,
        *,
        cap_bids: list[AFRRCapacityBid],
        ts_idx: pd.Series,
        source: pd.DataFrame,
        colmap: BacktestColumnMap,
    ):
        """Isolated aFRR capacity auction: submitted bids vs realized clearing prices."""
        sblk = source.reindex(ts_idx).copy()
        true_cap_pos = float(pd.to_numeric(sblk.get(colmap.true_afrr_capacity_price_pos), errors="coerce").mean())
        true_cap_neg = float(pd.to_numeric(sblk.get(colmap.true_afrr_capacity_price_neg), errors="coerce").mean())
        if not np.isfinite(true_cap_pos):
            true_cap_pos = 0.0
        if not np.isfinite(true_cap_neg):
            true_cap_neg = 0.0
        return self.market_clearing_engine.clear_afrr_capacity(
            cap_bids,
            true_cap_pos=true_cap_pos,
            true_cap_neg=true_cap_neg,
        )

    def optimize_dispatch_rolling(
        self,
        df: pd.DataFrame,
        colmap: BacktestColumnMap,
        *,
        horizon_hours: int = 48,
        reopt_step_hours: int = 1,
        forecast_warehouse: dict[str, pd.DataFrame] | None = None,
        da_gate_hour_cet: int = 11,
        soc_feedback_mode: str = "realized",
        enforce_final_soc_min: bool = True,
        deterministic_reserve_settlement: bool = False,
        is_perfect_foresight: bool = False,
        allowed_markets: list[str] | tuple[str, ...] | set[str] = ("DA", "aFRR"),
        strategy_name: str | None = None,
        id_mode: str | None = None,
        id_recourse_mode: str = "common",
        run_mode: str = "advanced_ml",
        strict_simulation_validity: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Rolling-horizon LP (re-optimized repeatedly with SoC state carryover)."""
        if horizon_hours <= 0 or reopt_step_hours <= 0:
            raise ValueError("horizon_hours and reopt_step_hours must be > 0")
        if soc_feedback_mode not in {"realized", "predicted"}:
            raise ValueError("soc_feedback_mode must be one of {'realized', 'predicted'}")
        self._assert_valid_time_index(df, colmap.timestamp)
        perms = self.resolve_strategy_permissions(
            strategy_name=strategy_name,
            allowed_markets=allowed_markets,
            id_mode=id_mode,
            id_recourse_mode=id_recourse_mode,
        )
        self._id_recourse_mode = self._normalize_id_recourse_mode(id_recourse_mode)
        self._strategy_permissions = perms
        da_enabled = bool(perms.allow_da)
        afrr_enabled = bool(perms.allow_bcm or perms.allow_bem_only)
        if df.empty:
            empty_dispatch = pd.DataFrame(
                columns=[
                    colmap.timestamp,
                    "charge_mw",
                    "discharge_mw",
                    "reserve_pos_mw",
                    "reserve_neg_mw",
                    "bem_only_pos_mw",
                    "bem_only_neg_mw",
                    "soc_lp_mwh",
                    "planned_soc_mwh",
                    "executed_soc_mwh",
                    "soc_shock_mwh",
                    "soc_before_mwh",
                    "soc_after_planned_mwh",
                    "soc_after_executed_mwh",
                    "shock_source",
                    "predicted_objective_eur",
                ]
            )
            empty_plan = pd.DataFrame(
                columns=[
                    "snapshot_time_utc",
                    "target_time_utc",
                    "lead_time_h",
                    "charge_mw",
                    "discharge_mw",
                    "reserve_pos_mw",
                    "reserve_neg_mw",
                    "bem_only_pos_mw",
                    "bem_only_neg_mw",
                    "planned_charge_mw",
                    "planned_discharge_mw",
                    "planned_reserve_mw",
                    "predicted_price",
                    "da_bid_locked",
                    "is_charging",
                    "soc_lp_mwh",
                    "predicted_objective_eur",
                ]
            )
            return empty_dispatch, empty_plan

        # Pre-simulation gate is only safe without forecast warehouse.
        # With long-format warehouse mode, critical prediction columns are
        # populated per-window later and validated there.
        if forecast_warehouse is None:
            # Apply global gate only when a full prediction surface is present on df.
            # Naive/perfect_foresight helper runs may intentionally build prediction columns
            # differently (e.g., lagged construction) and are validated downstream.
            pred_surface_cols = [
                colmap.pred_da_price,
                colmap.pred_afrr_capacity_price_pos,
                colmap.pred_afrr_capacity_price_neg,
                colmap.pred_afrr_activation_price_pos,
                colmap.pred_afrr_activation_price_neg,
                colmap.pred_afrr_activation_rate_pos,
                colmap.pred_afrr_activation_rate_neg,
            ]
            if all(c in df.columns for c in pred_surface_cols):
                self._validate_critical_data(df, colmap, run_mode=run_mode, require_pacc_bins=False)

        decisions: list[pd.DataFrame] = []
        plan_history: list[pd.DataFrame] = []
        soc = float(self.soc_init)
        n = len(df)
        source = df.set_index(colmap.timestamp)
        da_lockbook: dict[pd.Timestamp, tuple[float, float]] = {}
        afrr_cap_pos_lockbook: dict[pd.Timestamp, float] = {}
        afrr_cap_neg_lockbook: dict[pd.Timestamp, float] = {}
        afrr_energy_pos_lockbook: dict[pd.Timestamp, float] = {}
        afrr_energy_neg_lockbook: dict[pd.Timestamp, float] = {}
        afrr_lock_source_snapshot_utc: dict[pd.Timestamp, pd.Timestamp] = {}
        precommit_audit_by_ts: dict[str, dict[pd.Timestamp, float | str]] = {}
        # ID rescue decided at t and executed at t+1 (one-hour execution lag).
        pending_id_charge_mw = 0.0
        pending_id_discharge_mw = 0.0
        pending_id_reason = "none"
        reopt_restart_done: set[pd.Timestamp] = set()
        asof_right_cache: dict[tuple[int, str], dict[int, pd.DataFrame]] = {}
        i = 0
        progress_start = time.monotonic()
        progress_last_log = progress_start
        progress_last_i = -1
        progress_log_interval_s = 30.0

        def _log_progress(*, force: bool = False, note: str = "") -> None:
            nonlocal progress_last_log, progress_last_i
            now = time.monotonic()
            if not force and (now - progress_last_log) < progress_log_interval_s:
                return
            elapsed = max(0.0, now - progress_start)
            done = max(0, int(i))
            total = max(1, int(n))
            pct = 100.0 * (done / total)
            if done > 0:
                avg_s_per_step = elapsed / done
                eta_s = max(0.0, (total - done) * avg_s_per_step)
            else:
                eta_s = float("inf")
            eta_txt = f"{eta_s/60.0:.1f} min" if np.isfinite(eta_s) else "n/a"
            suffix = f" | {note}" if note else ""
            print(
                f"[PROGRESS] backtest rolling step {done}/{total} ({pct:.1f}%) "
                f"| elapsed={elapsed/60.0:.1f} min | eta={eta_txt}{suffix}"
            )
            progress_last_log = now
            progress_last_i = done
        while i < n:
            if i == progress_last_i:
                _log_progress(note="re-optimizing current snapshot")
            if forecast_warehouse:
                if i >= n - 1:
                    break
                w_end = min(n, i + 1 + horizon_hours)
                window = df.iloc[i + 1 : w_end].copy()
                if window.empty:
                    break
                snapshot_ts = pd.to_datetime(df.iloc[i][colmap.timestamp], utc=True, errors="coerce")
                target_times = pd.to_datetime(window[colmap.timestamp], utc=True, errors="coerce")
                left_targets = pd.DataFrame(
                    {
                        "_ord": np.arange(len(window), dtype=int),
                        "target_time_utc": pd.to_datetime(target_times, utc=True, errors="coerce"),
                        "snapshot_time_utc": pd.to_datetime(
                            pd.Series([snapshot_ts] * len(window), index=window.index),
                            utc=True,
                            errors="coerce",
                        ).to_numpy(),
                    }
                )
                left_targets["target_time_ns"] = left_targets["target_time_utc"].astype("int64")
                left_targets["snapshot_time_ns"] = left_targets["snapshot_time_utc"].astype("int64")

                def _asof_fill_from_long(src_df: pd.DataFrame, value_col: str) -> pd.Series:
                    """For each target, pick latest snapshot <= decision snapshot via asof-merge."""
                    def _to_ns_utc(s: pd.Series) -> pd.Series:
                        ts = pd.to_datetime(s, utc=True, errors="coerce")
                        # Force a common nanosecond representation independent of
                        # source parquet timestamp resolution (us/ns).
                        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
                        return ts.astype("int64")
                    if value_col not in src_df.columns:
                        return pd.Series(np.nan, index=window.index, dtype="float64")
                    cache_key = (id(src_df), value_col)
                    right_groups = asof_right_cache.get(cache_key)
                    if right_groups is None:
                        right = src_df.loc[:, ["target_time_utc", "snapshot_time_utc", value_col]].copy()
                        right["target_time_utc"] = pd.to_datetime(right["target_time_utc"], utc=True, errors="coerce")
                        right["snapshot_time_utc"] = pd.to_datetime(right["snapshot_time_utc"], utc=True, errors="coerce")
                        right["target_time_ns"] = _to_ns_utc(right["target_time_utc"])
                        right["snapshot_time_ns"] = _to_ns_utc(right["snapshot_time_utc"])
                        right = right.dropna(subset=["target_time_utc", "snapshot_time_utc"]).sort_values(
                            ["target_time_utc", "snapshot_time_utc"]
                        )
                        if right.empty:
                            asof_right_cache[cache_key] = {}
                            return pd.Series(np.nan, index=window.index, dtype="float64")
                        right2 = right[["target_time_ns", "snapshot_time_ns", value_col]].copy()
                        right2 = right2.dropna(subset=["target_time_ns", "snapshot_time_ns"]).copy()
                        right2["target_time_ns"] = pd.to_numeric(right2["target_time_ns"], errors="coerce").astype("int64")
                        right2["snapshot_time_ns"] = pd.to_numeric(right2["snapshot_time_ns"], errors="coerce").astype("int64")
                        right_groups = {}
                        for tns, g in right2.groupby("target_time_ns", sort=False):
                            right_groups[int(tns)] = g[["snapshot_time_ns", value_col]].sort_values(
                                "snapshot_time_ns", kind="mergesort"
                            )
                        asof_right_cache[cache_key] = right_groups
                    if not right_groups:
                        return pd.Series(np.nan, index=window.index, dtype="float64")
                    left = left_targets.copy()
                    left["target_time_utc"] = pd.to_datetime(left["target_time_utc"], utc=True, errors="coerce")
                    left["snapshot_time_utc"] = pd.to_datetime(left["snapshot_time_utc"], utc=True, errors="coerce")
                    left["target_time_ns"] = _to_ns_utc(left["target_time_utc"])
                    left["snapshot_time_ns"] = _to_ns_utc(left["snapshot_time_utc"])
                    out_vals = pd.Series(np.nan, index=left.index, dtype="float64")
                    left2 = left[["_ord", "target_time_ns", "snapshot_time_ns"]].copy()
                    left2 = left2.dropna(subset=["target_time_ns", "snapshot_time_ns"]).copy()
                    if left2.empty:
                        out_vals.index = window.index
                        return out_vals
                    left2["target_time_ns"] = pd.to_numeric(left2["target_time_ns"], errors="coerce").astype("int64")
                    left2["snapshot_time_ns"] = pd.to_numeric(left2["snapshot_time_ns"], errors="coerce").astype("int64")

                    for tns, lsub in left2.groupby("target_time_ns", sort=False):
                        rsub = right_groups.get(int(tns))
                        if rsub is None or rsub.empty:
                            continue
                        lsub2 = lsub[["_ord", "snapshot_time_ns"]].sort_values("snapshot_time_ns", kind="mergesort")
                        msub = pd.merge_asof(
                            lsub2,
                            rsub,
                            on="snapshot_time_ns",
                            direction="backward",
                            allow_exact_matches=True,
                        )
                        out_vals.iloc[msub["_ord"].to_numpy(dtype=int)] = pd.to_numeric(
                            msub[value_col], errors="coerce"
                        ).to_numpy(dtype=float)
                    out_vals.index = window.index
                    return out_vals

                for pred_col in CANONICAL_PREDICTION_COLUMNS:
                    if pred_col not in window.columns:
                        window[pred_col] = np.nan
                    if pred_col not in forecast_warehouse:
                        continue
                    src_df = forecast_warehouse[pred_col]
                    filled = _asof_fill_from_long(src_df, "predicted_value")
                    window[pred_col] = filled.to_numpy(dtype=float)

                    # Fallback to wide per-target timestamp predictions when warehouse has gaps.
                    if pred_col in source.columns:
                        fallback = pd.to_numeric(window[pred_col], errors="coerce")
                        miss = fallback.isna()
                        if bool(miss.any()):
                            fb_vals = pd.to_numeric(window.loc[miss, colmap.timestamp].map(source[pred_col]), errors="coerce")
                            window.loc[miss, pred_col] = fb_vals.to_numpy()

                    # Propagate capacity-price quantiles into window columns used
                    # by optimize_dispatch(), e.g. pred_afrr_capacity_price_pos_p01.
                    # Without this mapping, downstream logic falls back to median
                    # placeholders and creates artificial symmetric coefficients.
                    if pred_col in {colmap.pred_afrr_capacity_price_pos, colmap.pred_afrr_capacity_price_neg}:
                        q_cols = [c for c in QUANTILE_COLUMNS if c in src_df.columns]
                        if q_cols:
                            for qc in q_cols:
                                filled_q = _asof_fill_from_long(src_df, qc)
                                window[f"{pred_col}_{qc}"] = filled_q.to_numpy(dtype=float)
                            for b, qcol in enumerate(self.afrr_quantile_bins):
                                q_level = float(qcol.replace("p", "")) / 100.0
                                pacc = max(0.0, min(1.0, 1.0 - q_level))
                                side = "pos" if pred_col == colmap.pred_afrr_capacity_price_pos else "neg"
                                col = f"pacc_{side}_bin_{b}"
                                window[col] = np.full(len(window), pacc, dtype=float)
                    # Propagate activation-rate quantiles into window columns
                    # for p90 chance constraints in optimize_dispatch().
                    if pred_col in {colmap.pred_afrr_activation_rate_pos, colmap.pred_afrr_activation_rate_neg}:
                        q_cols = [c for c in QUANTILE_COLUMNS if c in src_df.columns]
                        if q_cols:
                            base = colmap.pred_afrr_activation_rate_pos if pred_col == colmap.pred_afrr_activation_rate_pos else colmap.pred_afrr_activation_rate_neg
                            for qc in q_cols:
                                filled_q = _asof_fill_from_long(src_df, qc)
                                window[f"{base}_{qc}"] = filled_q.to_numpy(dtype=float)
                    # Propagate activation-price quantiles into window columns
                    # used directly by strict quantile-bin optimizer inputs.
                    if pred_col in {colmap.pred_afrr_activation_price_pos, colmap.pred_afrr_activation_price_neg}:
                        q_cols = [c for c in QUANTILE_COLUMNS if c in src_df.columns]
                        if q_cols:
                            base = (
                                colmap.pred_afrr_activation_price_pos
                                if pred_col == colmap.pred_afrr_activation_price_pos
                                else colmap.pred_afrr_activation_price_neg
                            )
                            for qc in q_cols:
                                filled_q = _asof_fill_from_long(src_df, qc)
                                window[f"{base}_{qc}"] = filled_q.to_numpy(dtype=float)
                    # Propagate DA quantile columns used by DA limit bid pricing.
                    if pred_col == colmap.pred_da_price:
                        q_cols = [c for c in QUANTILE_COLUMNS if c in src_df.columns]
                        if q_cols:
                            for qc in q_cols:
                                filled_q = _asof_fill_from_long(src_df, qc)
                                window[f"{pred_col}_{qc}"] = filled_q.to_numpy(dtype=float)

                # Ensure required prediction columns are finite even if some long files are absent.
                # Causality guard: prediction-side optimization must never fall back to truth-side columns.
                pred_fallbacks: dict[str, list[str]] = {
                    colmap.pred_da_price: [],
                    colmap.pred_afrr_capacity_price_pos: [
                        colmap.pred_afrr_capacity_price_neg,
                    ],
                    colmap.pred_afrr_capacity_price_neg: [
                        colmap.pred_afrr_capacity_price_pos,
                    ],
                    colmap.pred_afrr_activation_price_pos: [
                        colmap.pred_afrr_activation_price_neg,
                    ],
                    colmap.pred_afrr_activation_price_neg: [
                        colmap.pred_afrr_activation_price_pos,
                    ],
                    colmap.pred_afrr_activation_rate_pos: [
                        colmap.pred_afrr_activation_rate_neg,
                    ],
                    colmap.pred_afrr_activation_rate_neg: [
                        colmap.pred_afrr_activation_rate_pos,
                    ],
                }
                # Fail-fast coverage check:
                # If forecast warehouse mapping leaves missing required optimizer
                # inputs in the current window, abort immediately with context.
                # This avoids silently biasing economics via default imputations.
                # End-of-coverage guard:
                # If *all* required prediction columns are missing only in a trailing
                # suffix of the rolling window (typical at split boundary), trim that
                # suffix and continue with the feasible prefix. Internal holes remain
                # hard failures (no silent interpolation across gaps).
                req_pred_cols = list(pred_fallbacks.keys())
                req_missing_frame = pd.DataFrame(
                    {
                        c: pd.to_numeric(window[c], errors="coerce").isna()
                        for c in req_pred_cols
                    },
                    index=window.index,
                )
                all_req_missing = req_missing_frame.all(axis=1)
                if bool(all_req_missing.any()):
                    miss_pos = np.flatnonzero(all_req_missing.to_numpy(dtype=bool))
                    first_miss = int(miss_pos[0])
                    trailing_only = np.array_equal(miss_pos, np.arange(first_miss, len(window)))
                    if trailing_only:
                        window = window.iloc[:first_miss].copy()
                        if window.empty:
                            break

                missing_msgs: list[str] = []
                for pred_col, fallbacks in pred_fallbacks.items():
                    if pred_col not in window.columns:
                        window[pred_col] = np.nan
                    raw_vals = pd.to_numeric(window[pred_col], errors="coerce")
                    miss_mask = raw_vals.isna()
                    if bool(miss_mask.any()):
                        miss_n = int(miss_mask.sum())
                        miss_ts = (
                            pd.to_datetime(window.loc[miss_mask, colmap.timestamp], utc=True, errors="coerce")
                            .dropna()
                            .astype(str)
                            .head(3)
                            .tolist()
                        )
                        missing_msgs.append(
                            f"{pred_col}: missing={miss_n}/{len(window)} sample_targets={miss_ts}"
                        )
                if missing_msgs:
                    snap_txt = str(pd.to_datetime(snapshot_ts, utc=True, errors="coerce")) if forecast_warehouse else "n/a"
                    raise RuntimeError(
                        "Forecast coverage check failed before optimizer input imputation. "
                        f"snapshot={snap_txt}; details=" + " | ".join(missing_msgs)
                    )

                # Input-quality diagnostics for optimizer-side forecast/prices/rates.
                # Count how many required optimizer inputs are missing before robust fill.
                req_cols = list(pred_fallbacks.keys())
                req_missing_count = np.zeros(len(window), dtype=int)
                req_missing_any = np.zeros(len(window), dtype=int)
                for pred_col in req_cols:
                    raw = pd.to_numeric(window[pred_col], errors="coerce")
                    miss = raw.isna().to_numpy(dtype=bool)
                    req_missing_count += miss.astype(int)
                    req_missing_any = np.maximum(req_missing_any, miss.astype(int))
                window["optimizer_required_input_imputed_count"] = req_missing_count.astype(float)
                window["optimizer_required_input_imputed_any"] = req_missing_any.astype(float)

                for pred_col, fallbacks in pred_fallbacks.items():
                    window[pred_col] = self._finite_numeric_series(
                        window,
                        pred_col,
                        fallback_cols=[c for c in fallbacks if c in window.columns],
                        default=0.0,
                        allow_temporal_fill=False,
                        strict_non_null=True,
                    ).to_numpy(dtype=float)

                # Final strict gate: critical optimizer inputs must be fully valid.
                self._validate_critical_data(
                    window,
                    colmap,
                    run_mode=run_mode,
                    require_pacc_bins=True,
                )
            else:
                w_end = min(n, i + horizon_hours)
                window = df.iloc[i:w_end].copy()

            def _fill_zero_ev(plan_df: pd.DataFrame) -> pd.DataFrame:
                """Populate fallback plans with EV columns to avoid NaN diagnostics."""
                out = plan_df.copy()
                scalar_zero_cols = [
                    "aux_power_mw",
                    "is_charging",
                    "predicted_objective_eur",
                    "ev_objective_rebuild_eur",
                    "ev_da_charge_coef_eur_per_mw",
                    "ev_da_discharge_coef_eur_per_mw",
                    "ev_da_charge_eur",
                    "ev_da_discharge_eur",
                    "ev_afrr_pos_eur",
                    "ev_afrr_neg_eur",
                    "ev_slack_penalty_pos_eur",
                    "ev_slack_penalty_neg_eur",
                    "ev_terminal_soc_credit_eur",
                    "ev_pred_da_price_eur_mwh",
                    "ev_pred_cap_pos_eur_mw",
                    "ev_pred_cap_neg_eur_mw",
                    "ev_pred_act_price_pos_eur_mwh",
                    "ev_pred_act_price_neg_eur_mwh",
                    "ev_pred_act_rate_pos",
                    "ev_pred_act_rate_neg",
                    "ev_pred_act_rate_pos_p90",
                    "ev_pred_act_rate_neg_p90",
                    "slack_pos_mw",
                    "slack_neg_mw",
                    "slack_soc_min_mwh",
                    "slack_soc_max_mwh",
                    "final_soc_shortfall_mwh",
                    "optimizer_required_input_imputed_count",
                    "optimizer_required_input_imputed_any",
                    "ev_pacc_pos_fallback_used",
                    "ev_pacc_neg_fallback_used",
                    "optimizer_fallback_used",
                    "terminal_constraint_dropped",
                ]
                for c in scalar_zero_cols:
                    if c not in out.columns:
                        out[c] = 0.0
                for b in range(len(self.afrr_quantile_bins)):
                    for c in (
                        f"reserve_pos_bin_{b}_mw",
                        f"reserve_neg_bin_{b}_mw",
                        f"ev_pacc_pos_bin_{b}",
                        f"ev_pacc_neg_bin_{b}",
                        f"ev_expected_act_share_pos_bin_{b}",
                        f"ev_expected_act_share_neg_bin_{b}",
                        f"ev_rpos_coef_bin_{b}_eur_per_mw",
                        f"ev_rneg_coef_bin_{b}_eur_per_mw",
                    ):
                        if c not in out.columns:
                            out[c] = 0.0
                return out

            # Terminal SoC floor policy:
            # - Intermediate rolling windows: only enforce physical minimum SoC.
            # - Final global window: enforce configured terminal target SoC.
            # Terminal valuation in objective governs economic carry-over behavior.
            enforce_end = None
            if enforce_final_soc_min:
                enforce_end_min = self.soc_target_end if (w_end == n) else self.soc_min
            else:
                enforce_end_min = None
            optimization_fallback = "none"
            optimization_error = ""
            terminal_constraint_dropped = 0.0
            fixed_reserve_obligation: dict[pd.Timestamp, tuple[float, float]] = {}
            reserve_retry_attempts_used = 0.0
            reserve_retry_final_factor = 1.0
            reserve_retry_succeeded = 0.0
            new_reserve_bids_zeroed_by_retry = 0.0
            reserve_retry_infeasible_after_zero_reserve = 0.0
            disable_new_bcm_reserve_bids_used = float(self.disable_new_bcm_reserve_bids)
            if perms.allow_bcm:
                w_ts = pd.to_datetime(window[colmap.timestamp], utc=True, errors="coerce")
                for tsw in w_ts:
                    if pd.isna(tsw):
                        continue
                    ob_pos = float(afrr_cap_pos_lockbook.get(tsw, 0.0))
                    ob_neg = float(afrr_cap_neg_lockbook.get(tsw, 0.0))
                    if ob_pos > 0.0 or ob_neg > 0.0:
                        fixed_reserve_obligation[tsw] = (ob_pos, ob_neg)
            else:
                # Strategy purity guard: bem_only must not carry BCM lockbook obligations.
                if bool(afrr_cap_pos_lockbook) or bool(afrr_cap_neg_lockbook):
                    raise RuntimeError(
                        "Strategy contamination: bcm lockbook obligations exist while BCM is disabled."
                    )
            def _apply_retry_factor_to_new_obligations(
                obligations: dict[pd.Timestamp, tuple[float, float]],
                factor: float,
                snapshot_current: pd.Timestamp | None,
            ) -> dict[pd.Timestamp, tuple[float, float]]:
                if snapshot_current is None:
                    return dict(obligations)
                out_ob: dict[pd.Timestamp, tuple[float, float]] = {}
                for ts_ob, (ob_pos, ob_neg) in obligations.items():
                    src_snap = afrr_lock_source_snapshot_utc.get(ts_ob)
                    is_new = pd.notna(src_snap) and pd.notna(snapshot_current) and pd.Timestamp(src_snap) == pd.Timestamp(snapshot_current)
                    if is_new:
                        out_ob[ts_ob] = (
                            self.bid_builder._qfloor(max(0.0, float(ob_pos) * float(factor)), self.afrr_bid_granularity_mw),
                            self.bid_builder._qfloor(max(0.0, float(ob_neg) * float(factor)), self.afrr_bid_granularity_mw),
                        )
                    else:
                        out_ob[ts_ob] = (float(ob_pos), float(ob_neg))
                return out_ob

            snapshot_current_ts = pd.to_datetime(snapshot_ts, utc=True, errors="coerce") if forecast_warehouse else pd.to_datetime(window.iloc[0][colmap.timestamp], utc=True, errors="coerce")
            retry_ladder = [1.0]
            if self.disable_new_bcm_reserve_bids:
                retry_ladder = [0.0]
            elif (
                bool(strict_simulation_validity)
                and str(self.reserve_feasibility_mode) == "conservative"
                and bool(self.enable_reserve_retry_ladder)
            ):
                retry_ladder = list(self.reserve_retry_ladder)

            last_exc: RuntimeError | None = None
            plan = None
            for att_idx, fac in enumerate(retry_ladder):
                reserve_retry_attempts_used = float(att_idx + 1)
                reserve_retry_final_factor = float(fac)
                trial_ob = _apply_retry_factor_to_new_obligations(fixed_reserve_obligation, float(fac), snapshot_current_ts)
                try:
                    plan = self.optimize_dispatch(
                        window,
                        colmap,
                        soc_start=soc,
                        soc_end_target=enforce_end,
                        soc_end_min_target=enforce_end_min,
                        fixed_da_dispatch=da_lockbook,
                        fixed_reserve_obligation=trial_ob,
                        deterministic_reserve_settlement=deterministic_reserve_settlement,
                        allowed_markets=allowed_markets,
                        strict_input_validation=bool(forecast_warehouse is not None),
                    )
                    fixed_reserve_obligation = trial_ob
                    reserve_retry_succeeded = 1.0
                    plan["terminal_constraint_dropped"] = 0.0
                    if float(fac) <= 1e-12:
                        new_reserve_bids_zeroed_by_retry = 1.0
                    break
                except RuntimeError as exc_try:
                    last_exc = exc_try
                    if float(fac) <= 1e-12:
                        reserve_retry_infeasible_after_zero_reserve = 1.0
                    continue
            try:
                if plan is None and last_exc is not None:
                    raise last_exc
                if plan is None:
                    raise RuntimeError("optimization failed without solver exception context")
            except RuntimeError as exc:
                msg = str(exc)
                msg_l = msg.lower()
                # Recoverable solver failures (e.g. transient HiGHS "Not Set")
                # should not abort the entire backtest. Degrade this window to a
                # physically safe hold plan and continue rolling.
                recoverable_solver_failure = (
                    "highs status 0: not set" in msg_l
                    or "not set" in msg_l
                    or "numerical" in msg_l
                )
                is_infeasible = "infeasible" in msg_l
                if (not is_infeasible) and (not recoverable_solver_failure):
                    raise
                optimization_error = msg
                has_fixed_reserve_obligation = bool(
                    any((float(v[0]) > 1e-9 or float(v[1]) > 1e-9) for v in fixed_reserve_obligation.values())
                )
                def _safe_hold_with_obligations() -> pd.DataFrame:
                    ts_vals = pd.to_datetime(window[colmap.timestamp], utc=True, errors="coerce")
                    hold = pd.DataFrame(
                        {
                            colmap.timestamp: ts_vals,
                            "charge_mw": np.zeros(len(window), dtype=float),
                            "discharge_mw": np.zeros(len(window), dtype=float),
                            "reserve_pos_mw": np.zeros(len(window), dtype=float),
                            "reserve_neg_mw": np.zeros(len(window), dtype=float),
                            "bem_only_pos_mw": np.zeros(len(window), dtype=float),
                            "bem_only_neg_mw": np.zeros(len(window), dtype=float),
                            "soc_lp_mwh": np.full(len(window), float(soc), dtype=float),
                            "final_soc_shortfall_mwh": np.full(len(window), 0.0, dtype=float),
                        },
                        index=window.index,
                    )
                    # Keep already-awarded reserve obligations to avoid synthetic
                    # capacity misses caused by fallback dropping commitments.
                    if fixed_reserve_obligation:
                        ts_series = pd.to_datetime(hold[colmap.timestamp], utc=True, errors="coerce")
                        pos_vals = np.zeros(len(hold), dtype=float)
                        neg_vals = np.zeros(len(hold), dtype=float)
                        for i, tsw in enumerate(ts_series):
                            if pd.notna(tsw) and tsw in fixed_reserve_obligation:
                                ob_pos, ob_neg = fixed_reserve_obligation[tsw]
                                pos_vals[i] = self.bid_builder._qfloor(
                                    max(0.0, float(ob_pos)), self.afrr_bid_granularity_mw
                                )
                                neg_vals[i] = self.bid_builder._qfloor(
                                    max(0.0, float(ob_neg)), self.afrr_bid_granularity_mw
                                )
                        hold["reserve_pos_mw"] = pos_vals
                        hold["reserve_neg_mw"] = neg_vals
                    # Maintain SoC against auxiliary losses during fallback,
                    # especially when reserve obligations are locked.
                    rate_pos_series = pd.to_numeric(
                        window.get(colmap.pred_afrr_activation_rate_pos, 0.0), errors="coerce"
                    ).fillna(0.0).to_numpy(dtype=float)
                    rate_neg_series = pd.to_numeric(
                        window.get(colmap.pred_afrr_activation_rate_neg, 0.0), errors="coerce"
                    ).fillna(0.0).to_numpy(dtype=float)
                    ch_vals = np.zeros(len(hold), dtype=float)
                    for i in range(len(hold)):
                        aux_p, _ = self._state_aux_power_mw(
                            charge_mw=0.0,
                            discharge_mw=0.0,
                            reserve_pos_mw=float(hold["reserve_pos_mw"].iloc[i]),
                            reserve_neg_mw=float(hold["reserve_neg_mw"].iloc[i]),
                            act_pos_rate=float(rate_pos_series[i] if i < len(rate_pos_series) else 0.0),
                            act_neg_rate=float(rate_neg_series[i] if i < len(rate_neg_series) else 0.0),
                            id_charge_mw=0.0,
                            id_discharge_mw=0.0,
                        )
                        # charge needed so eta_in * charge ~= aux power
                        ch_needed = float(aux_p) / max(self.eta_in, 1e-12)
                        ch_vals[i] = max(0.0, min(self.p_max_mw, ch_needed))
                    hold["charge_mw"] = ch_vals
                    for b in range(len(self.afrr_quantile_bins)):
                        hold[f"reserve_pos_bin_{b}_mw"] = 0.0
                        hold[f"reserve_neg_bin_{b}_mw"] = 0.0
                    if len(self.afrr_quantile_bins):
                        hold["reserve_pos_bin_0_mw"] = hold["reserve_pos_mw"]
                        hold["reserve_neg_bin_0_mw"] = hold["reserve_neg_mw"]
                    return _fill_zero_ev(hold)
                def _has_headroom_violation_now() -> bool:
                    if not has_fixed_reserve_obligation:
                        return False
                    ts0 = pd.to_datetime(window.iloc[0][colmap.timestamp], utc=True, errors="coerce")
                    ob_pos, ob_neg = fixed_reserve_obligation.get(ts0, (0.0, 0.0))
                    req_pos = max(0.0, float(ob_pos)) * float(self.reserve_activation_headroom_h) / max(float(self.eta_out), 1e-12)
                    req_neg = max(0.0, float(ob_neg)) * float(self.reserve_activation_headroom_h) * float(self.eta_in)
                    avail_pos = max(0.0, float(soc) - float(self.soc_min))
                    avail_neg = max(0.0, float(self.soc_max) - float(soc))
                    return (avail_pos + 1e-9 < req_pos) or (avail_neg + 1e-9 < req_neg)
                # If terminal SoC floor conflicts with already locked reserve obligations,
                # keep reserve obligations and retry without terminal minimum first.
                allow_terminal_drop_retry = not (
                    bool(strict_simulation_validity) and str(self.final_soc_mode) == "hard"
                )
                if (
                    is_infeasible
                    and has_fixed_reserve_obligation
                    and (enforce_end_min is not None)
                    and allow_terminal_drop_retry
                ):
                    try:
                        plan = self.optimize_dispatch(
                            window,
                            colmap,
                            soc_start=soc,
                            soc_end_target=enforce_end,
                            soc_end_min_target=None,
                            fixed_da_dispatch=da_lockbook,
                            fixed_reserve_obligation=fixed_reserve_obligation,
                            deterministic_reserve_settlement=deterministic_reserve_settlement,
                            allowed_markets=allowed_markets,
                            strict_input_validation=bool(forecast_warehouse is not None),
                        )
                        terminal_constraint_dropped = 1.0
                        plan["terminal_constraint_dropped"] = 1.0
                        optimization_fallback = "none"
                        optimization_error = ""
                    except RuntimeError:
                        plan = None
                else:
                    plan = None
                if plan is not None:
                    pass
                if plan is None:
                    if recoverable_solver_failure:
                        plan = _safe_hold_with_obligations()
                        plan["optimizer_fallback_used"] = 1.0
                        plan["terminal_constraint_dropped"] = float(terminal_constraint_dropped)
                        optimization_fallback = "reserve_feasibility_repair" if has_fixed_reserve_obligation else "safe_hold_plan_under_solver_not_set"
                    elif enforce_end_min is not None:
                        # In strict hard mode we do not drop the terminal target.
                        # If infeasible, degrade to safe hold and mark invalid later.
                        plan = _safe_hold_with_obligations()
                        plan["optimizer_fallback_used"] = 1.0
                        plan["terminal_constraint_dropped"] = float(terminal_constraint_dropped)
                        if _has_headroom_violation_now():
                            optimization_fallback = "reserve_infeasible"
                        else:
                            if bool(strict_simulation_validity) and str(self.final_soc_mode) == "hard":
                                optimization_fallback = "hard_final_soc_infeasible"
                            else:
                                optimization_fallback = "reserve_feasibility_repair" if has_fixed_reserve_obligation else "safe_hold_plan_under_infeasible_soft_final_soc"
                    else:
                        # Last-resort fallback: if rolling MILP is infeasible even
                        # without terminal floor, degrade to a physically safe hold
                        # plan for this window so the simulation can proceed.
                        # This avoids full-run aborts from local pathological windows.
                        plan = _safe_hold_with_obligations()
                        plan["optimizer_fallback_used"] = 1.0
                        plan["terminal_constraint_dropped"] = float(terminal_constraint_dropped)
                        optimization_fallback = "reserve_feasibility_repair" if has_fixed_reserve_obligation else "safe_hold_plan"

            snapshot_plan = plan.copy()
            if "terminal_constraint_dropped" not in snapshot_plan.columns:
                snapshot_plan["terminal_constraint_dropped"] = float(terminal_constraint_dropped)
            snapshot_plan["snapshot_time_utc"] = snapshot_ts if forecast_warehouse else pd.to_datetime(window.iloc[0][colmap.timestamp], utc=True, errors="coerce")
            snapshot_plan["target_time_utc"] = pd.to_datetime(snapshot_plan[colmap.timestamp], utc=True, errors="coerce")
            if snapshot_plan["target_time_utc"].isna().any():
                raise ValueError("Rolling plan contains invalid target timestamps.")
            snapshot_plan["lead_time_h"] = (
                (snapshot_plan["target_time_utc"] - snapshot_plan["snapshot_time_utc"]).dt.total_seconds() // 3600
            ).astype("Int64")
            # Long-format plan warehouse fields for decision-volatility analysis.
            snapshot_plan["planned_charge_mw"] = snapshot_plan["charge_mw"]
            snapshot_plan["planned_discharge_mw"] = snapshot_plan["discharge_mw"]
            snapshot_plan["planned_reserve_mw"] = snapshot_plan["reserve_pos_mw"] + snapshot_plan["reserve_neg_mw"]
            snapshot_plan["planned_bem_only_mw"] = snapshot_plan.get("bem_only_pos_mw", 0.0) + snapshot_plan.get("bem_only_neg_mw", 0.0)
            window_price_map = pd.Series(
                self._finite_numeric_series(
                    window,
                    colmap.pred_da_price,
                    fallback_cols=[],
                    default=0.0,
                ).to_numpy(dtype=float),
                index=pd.to_datetime(window[colmap.timestamp], utc=True, errors="coerce"),
            )
            snapshot_plan["predicted_price"] = snapshot_plan["target_time_utc"].map(window_price_map)
            snapshot_plan["da_bid_locked"] = snapshot_plan["target_time_utc"].isin(set(da_lockbook.keys()))
            snapshot_plan["optimization_fallback"] = optimization_fallback
            snapshot_plan["optimizer_fallback_used"] = float(optimization_fallback != "none")
            snapshot_plan["optimization_error"] = optimization_error
            snapshot_plan["reserve_retry_attempts_used"] = float(reserve_retry_attempts_used)
            snapshot_plan["reserve_retry_final_factor"] = float(reserve_retry_final_factor)
            snapshot_plan["reserve_retry_succeeded"] = float(reserve_retry_succeeded)
            snapshot_plan["disable_new_bcm_reserve_bids"] = float(disable_new_bcm_reserve_bids_used)
            snapshot_plan["new_reserve_bids_zeroed_by_retry"] = float(new_reserve_bids_zeroed_by_retry)
            snapshot_plan["reserve_retry_infeasible_after_zero_reserve"] = float(
                reserve_retry_infeasible_after_zero_reserve
            )
            snapshot_ts_current = pd.to_datetime(snapshot_plan["snapshot_time_utc"].iloc[0], utc=True, errors="coerce")
            afrr_bcm_gate_hour_cet = int(self.afrr_bcm_gate_hour_cet)
            is_afrr_gate_now = bool(
                perms.allow_bcm and pd.notna(snapshot_ts_current) and self._is_gate_hour_cet(snapshot_ts_current, afrr_bcm_gate_hour_cet)
            )
            is_da_gate_now = bool(da_enabled and pd.notna(snapshot_ts_current) and self._is_gate_hour_cet(snapshot_ts_current, da_gate_hour_cet))
            if is_afrr_gate_now and is_da_gate_now:
                snapshot_plan["milp_event_type"] = "afrr_bcm_auction+da_auction"
            elif is_afrr_gate_now:
                snapshot_plan["milp_event_type"] = "afrr_bcm_auction"
            elif is_da_gate_now:
                snapshot_plan["milp_event_type"] = "da_auction"
            else:
                snapshot_plan["milp_event_type"] = "none"
            snapshot_for_history = snapshot_plan.copy()
            history_rename_dict = {
                "charge_mw": "plan_charge_mw",
                "discharge_mw": "plan_discharge_mw",
                "reserve_pos_mw": "plan_reserve_pos_mw",
                "reserve_neg_mw": "plan_reserve_neg_mw",
                "bem_only_pos_mw": "plan_bem_only_pos_mw",
                "bem_only_neg_mw": "plan_bem_only_neg_mw",
                "aFRR_Capacity_Won_MW": "plan_aFRR_Capacity_Won_MW",
            }
            legacy_cols = [c for c in snapshot_for_history.columns if c.startswith("planned_")]
            if legacy_cols:
                snapshot_for_history.drop(columns=legacy_cols, inplace=True)
            existing_targets = [v for v in history_rename_dict.values() if v in snapshot_for_history.columns]
            if existing_targets:
                snapshot_for_history.drop(columns=existing_targets, inplace=True)
            snapshot_for_history.rename(columns=history_rename_dict, inplace=True)
            plan_history.append(snapshot_for_history)

            # Phase 1 (D-1 configured aFRR BCM gate hour): clear aFRR capacity in 4h blocks and
            # propagate awarded obligations to delivery intervals.
            if perms.allow_bcm:
                cap_gate_stats = self._update_afrr_capacity_lockbooks_from_snapshot(
                    snapshot_ts=pd.to_datetime(snapshot_plan["snapshot_time_utc"].iloc[0], utc=True, errors="coerce"),
                    snapshot_plan=snapshot_plan,
                    source=source,
                    colmap=colmap,
                    lock_pos=afrr_cap_pos_lockbook,
                    lock_neg=afrr_cap_neg_lockbook,
                    lock_energy_pos=afrr_energy_pos_lockbook,
                    lock_energy_neg=afrr_energy_neg_lockbook,
                    lock_source_snapshot_utc=afrr_lock_source_snapshot_utc,
                    precommit_audit_by_ts=precommit_audit_by_ts,
                    is_perfect_foresight=is_perfect_foresight,
                )
            else:
                cap_gate_stats = {"triggered": 0.0, "rejected_mw_total": 0.0}
            cap_gate_triggered = bool(cap_gate_stats.get("triggered", 0.0))
            snapshot_plan["event_reopt_triggered"] = float(cap_gate_triggered)
            snapshot_plan["event_reopt_rejected_mw_total"] = float(cap_gate_stats.get("rejected_mw_total", 0.0))
            this_snapshot_ts = pd.to_datetime(snapshot_plan["snapshot_time_utc"].iloc[0], utc=True, errors="coerce")
            if cap_gate_triggered and pd.notna(this_snapshot_ts) and this_snapshot_ts not in reopt_restart_done:
                # Event-driven hard restart: invalidate current MILP plan and
                # re-optimize immediately from the same snapshot state (no chunk hack).
                reopt_restart_done.add(this_snapshot_ts)
                _log_progress(note="capacity-gate event restart")
                continue

            # Lock-in DA bids at gate closure for next UTC day (24 hours).
            if da_enabled and pd.notna(snapshot_plan["snapshot_time_utc"].iloc[0]) and self._is_gate_hour_cet(pd.to_datetime(snapshot_plan["snapshot_time_utc"].iloc[0], utc=True), da_gate_hour_cet):
                snapshot_ts_effective = pd.to_datetime(snapshot_plan["snapshot_time_utc"].iloc[0], utc=True)
                next_day = (snapshot_ts_effective + pd.Timedelta(days=1)).normalize()
                day_end = next_day + pd.Timedelta(hours=23)
                lock_rows = snapshot_plan[
                    (snapshot_plan["target_time_utc"] >= next_day) & (snapshot_plan["target_time_utc"] <= day_end)
                ]
                for r in lock_rows[[colmap.timestamp, "charge_mw", "discharge_mw"]].itertuples(index=False):
                    tsu = pd.to_datetime(r[0], utc=True)
                    ch_fix, dis_fix = self._normalize_da_bid(float(r[1]), float(r[2]))
                    obligation_mw = max(
                        float(afrr_cap_pos_lockbook.get(tsu, 0.0)),
                        float(afrr_cap_neg_lockbook.get(tsu, 0.0)),
                    )
                    available_da = max(0.0, self.p_max_mw - obligation_mw)
                    ch_fix = min(ch_fix, available_da)
                    dis_fix = min(dis_fix, available_da)
                    da_lockbook[tsu] = (ch_fix, dis_fix)

            k = min(reopt_step_hours, len(plan))
            take = plan.iloc[:k].copy()
            pred_cols = [c for c in CANONICAL_PREDICTION_COLUMNS if c in window.columns]
            # Include DA quantile columns needed later for quantile-backed
            # settlement/bid validation in fail-fast mode.
            da_quant_cols = [
                c
                for c in (
                    f"{colmap.pred_da_price}_p05",
                    f"{colmap.pred_da_price}_p10",
                    f"{colmap.pred_da_price}_p90",
                    f"{colmap.pred_da_price}_p95",
                )
                if c in window.columns
            ]
            pred_cols = pred_cols + [c for c in da_quant_cols if c not in pred_cols]
            if pred_cols:
                pred_take = window[[colmap.timestamp, *pred_cols]].iloc[:k].copy()
                # Defensive normalization: ensure same tz-aware dtype on merge key.
                take[colmap.timestamp] = pd.to_datetime(take[colmap.timestamp], utc=True, errors="coerce")
                pred_take[colmap.timestamp] = pd.to_datetime(pred_take[colmap.timestamp], utc=True, errors="coerce")
                # Merge on explicit epoch key to avoid pandas tz/unit merge mismatches.
                take["_merge_ts_key"] = pd.to_datetime(take[colmap.timestamp], utc=True, errors="coerce").astype("int64")
                pred_take["_merge_ts_key"] = pd.to_datetime(pred_take[colmap.timestamp], utc=True, errors="coerce").astype("int64")
                pred_take = pred_take.drop(columns=[colmap.timestamp], errors="ignore")
                take = take.merge(pred_take, on="_merge_ts_key", how="left").drop(columns=["_merge_ts_key"], errors="ignore")
            if is_perfect_foresight:
                # Explicit perfect_foresight overrides for settlement whitelist merge path.
                true_take = window[
                    [
                        colmap.timestamp,
                        colmap.true_da_price,
                        colmap.true_afrr_capacity_price_pos,
                        colmap.true_afrr_capacity_price_neg,
                        colmap.true_afrr_activation_price_pos,
                        colmap.true_afrr_activation_price_neg,
                        colmap.true_afrr_activation_rate_pos,
                        colmap.true_afrr_activation_rate_neg,
                    ]
                ].iloc[:k].copy()
                true_take[colmap.timestamp] = pd.to_datetime(true_take[colmap.timestamp], utc=True, errors="coerce")
                take[colmap.timestamp] = pd.to_datetime(take[colmap.timestamp], utc=True, errors="coerce")
                true_take["_merge_ts_key"] = pd.to_datetime(true_take[colmap.timestamp], utc=True, errors="coerce").astype("int64")
                take["_merge_ts_key"] = pd.to_datetime(take[colmap.timestamp], utc=True, errors="coerce").astype("int64")
                true_take = true_take.drop(columns=[colmap.timestamp], errors="ignore")
                take = take.merge(true_take, on="_merge_ts_key", how="left").drop(columns=["_merge_ts_key"], errors="ignore")
                take["perfect_foresight_override_da_price"] = pd.to_numeric(take.get(colmap.true_da_price), errors="coerce")
                take["perfect_foresight_override_cap_pos"] = pd.to_numeric(take.get(colmap.true_afrr_capacity_price_pos), errors="coerce")
                take["perfect_foresight_override_cap_neg"] = pd.to_numeric(take.get(colmap.true_afrr_capacity_price_neg), errors="coerce")
                take["perfect_foresight_override_act_pos"] = pd.to_numeric(take.get(colmap.true_afrr_activation_price_pos), errors="coerce")
                take["perfect_foresight_override_act_neg"] = pd.to_numeric(take.get(colmap.true_afrr_activation_price_neg), errors="coerce")
                take["perfect_foresight_override_rate_pos"] = pd.to_numeric(take.get(colmap.true_afrr_activation_rate_pos), errors="coerce")
                take["perfect_foresight_override_rate_neg"] = pd.to_numeric(take.get(colmap.true_afrr_activation_rate_neg), errors="coerce")
            tsu_take = pd.to_datetime(take[colmap.timestamp], utc=True, errors="coerce")
            take["aFRR_Capacity_Won_Pos_MW"] = tsu_take.map(lambda ts: float(afrr_cap_pos_lockbook.get(ts, 0.0)))
            take["aFRR_Capacity_Won_Neg_MW"] = tsu_take.map(lambda ts: float(afrr_cap_neg_lockbook.get(ts, 0.0)))
            take["aFRR_Capacity_Won_MW"] = take[["aFRR_Capacity_Won_Pos_MW", "aFRR_Capacity_Won_Neg_MW"]].max(axis=1)
            take["aFRR_Energy_Price_EUR_MWh_Pos"] = tsu_take.map(lambda ts: float(afrr_energy_pos_lockbook.get(ts, np.nan)))
            take["aFRR_Energy_Price_EUR_MWh_Neg"] = tsu_take.map(lambda ts: float(afrr_energy_neg_lockbook.get(ts, np.nan)))
            precommit_fields = [
                "precommit_clamp_applied",
                "precommit_clamp_reason",
                "precommit_feasible_pos_mw",
                "precommit_feasible_neg_mw",
                "precommit_original_pos_mw",
                "precommit_original_neg_mw",
                "precommit_clamped_pos_mw",
                "precommit_clamped_neg_mw",
                "precommit_headroom_margin_min_mwh",
                "precommit_power_margin_min_mw",
                "precommit_soc_margin_min_mwh",
                "precommit_aux_loss_margin_mwh",
                "precommit_lockbook_obligation_pos_mw",
                "precommit_lockbook_obligation_neg_mw",
                "precommit_margin_after_bid_min_mwh",
                "precommit_zeroed_due_to_margin",
                "reserve_commitment_id",
                "reserve_product_block_id",
                "reserve_commitment_source_snapshot_utc",
                "reserve_delivery_start_utc",
                "reserve_delivery_end_utc",
                "reserve_precommit_feasible_pos_mw",
                "reserve_precommit_feasible_neg_mw",
                "reserve_submitted_pos_mw",
                "reserve_submitted_neg_mw",
                "submitted_reserve_pos_mw_before_retry",
                "submitted_reserve_neg_mw_before_retry",
                "reserve_retry_factor",
                "submitted_reserve_pos_mw_after_retry",
                "submitted_reserve_neg_mw_after_retry",
                "retry_reduction_reason",
                "reserve_awarded_pos_mw",
                "reserve_awarded_neg_mw",
                "reserve_lockbook_pos_mw",
                "reserve_lockbook_neg_mw",
                "reserve_projected_soc_start_mwh",
            ]
            for f in precommit_fields:
                m = precommit_audit_by_ts.get(f, {})
                take[f] = tsu_take.map(lambda ts, mm=m: mm.get(ts, np.nan))
            take["event_reopt_triggered"] = float(cap_gate_triggered)
            take["event_reopt_rejected_mw_total"] = float(cap_gate_stats.get("rejected_mw_total", 0.0))
            take["optimization_error_code"] = str(optimization_fallback) if str(optimization_fallback) != "none" else "ok"
            take["is_fallback_hour"] = float(str(optimization_fallback) != "none")
            take["is_precleared"] = True
            if "optimizer_required_input_imputed_any" in take.columns:
                take["is_strict_optimized_hour"] = (
                    pd.to_numeric(take["is_fallback_hour"], errors="coerce").fillna(0.0).eq(0.0)
                    & pd.to_numeric(take["optimizer_required_input_imputed_any"], errors="coerce").fillna(0.0).eq(0.0)
                ).astype(float)
            else:
                take["is_strict_optimized_hour"] = pd.to_numeric(take["is_fallback_hour"], errors="coerce").fillna(0.0).eq(0.0).astype(float)
            # Propagate both planned and executed SoC. Re-planning state always
            # uses executed SoC to reflect market-clearing reality.
            window_src = window.set_index(colmap.timestamp)
            planned_soc_vals: list[float] = []
            executed_soc_vals: list[float] = []
            soc_before_vals: list[float] = []
            shock_source_vals: list[str] = []
            clearing_records: list[dict[str, float]] = []
            planned_s = float(soc)
            executed_s = float(soc)
            for step_idx, r in enumerate(take.itertuples(index=False)):
                soc_before_vals.append(float(executed_s))
                ts = getattr(r, colmap.timestamp)
                src = window_src.loc[ts] if ts in window_src.index else source.loc[ts]
                def _sf(series: pd.Series, key: str, default: float = np.nan) -> float:
                    try:
                        v = pd.to_numeric(pd.Series([series.get(key, default)]), errors="coerce").iloc[0]
                        return float(default if pd.isna(v) else v)
                    except Exception:
                        return float(default)

                charge_plan = float(getattr(r, "charge_mw"))
                discharge_plan = float(getattr(r, "discharge_mw"))
                reserve_pos_plan = float(getattr(r, "reserve_pos_mw"))
                reserve_neg_plan = float(getattr(r, "reserve_neg_mw"))
                bem_only_pos_plan = float(getattr(r, "bem_only_pos_mw", 0.0))
                bem_only_neg_plan = float(getattr(r, "bem_only_neg_mw", 0.0))

                if soc_feedback_mode == "realized":
                    da_price_plan = float(src[colmap.true_da_price])
                    cap_pos_plan = float(src[colmap.true_afrr_capacity_price_pos])
                    cap_neg_plan = float(src[colmap.true_afrr_capacity_price_neg])
                    act_pos_price_plan = float(src[colmap.true_afrr_activation_price_pos])
                    act_neg_price_plan = float(src[colmap.true_afrr_activation_price_neg])
                    act_pos_rate_plan = float(src[colmap.true_afrr_activation_rate_pos])
                    act_neg_rate_plan = float(src[colmap.true_afrr_activation_rate_neg])
                else:
                    da_price_plan = float(src[colmap.pred_da_price])
                    cap_pos_plan = float(src[colmap.pred_afrr_capacity_price_pos])
                    cap_neg_plan = float(src[colmap.pred_afrr_capacity_price_neg])
                    act_pos_price_plan = float(src[colmap.pred_afrr_activation_price_pos])
                    act_neg_price_plan = float(src[colmap.pred_afrr_activation_price_neg])
                    act_pos_rate_plan = float(src[colmap.pred_afrr_activation_rate_pos])
                    act_neg_rate_plan = float(src[colmap.pred_afrr_activation_rate_neg])

                planned_s, _ = self._settle_one_hour(
                    soc=planned_s,
                    charge=charge_plan,
                    discharge=discharge_plan,
                    da_charge_mw=charge_plan,
                    da_discharge_mw=discharge_plan,
                    id_charge_mw=float(pending_id_charge_mw),
                    id_discharge_mw=float(pending_id_discharge_mw),
                    reserve_pos=reserve_pos_plan,
                    reserve_neg=reserve_neg_plan,
                    da_price=da_price_plan,
                    cap_pos=cap_pos_plan,
                    cap_neg=cap_neg_plan,
                    act_pos_price=act_pos_price_plan,
                    act_neg_price=act_neg_price_plan,
                    act_pos_rate=act_pos_rate_plan,
                    act_neg_rate=act_neg_rate_plan,
                )
                planned_soc_vals.append(float(planned_s))

                critical_inputs = {
                    "pred_da_price": float(da_price_plan),
                    "true_da_price": _sf(src, colmap.true_da_price, np.nan),
                    "pred_cap_pos": _sf(src, colmap.pred_afrr_capacity_price_pos, np.nan),
                    "true_cap_pos": _sf(src, colmap.true_afrr_capacity_price_pos, np.nan),
                    "pred_cap_neg": _sf(src, colmap.pred_afrr_capacity_price_neg, np.nan),
                    "true_cap_neg": _sf(src, colmap.true_afrr_capacity_price_neg, np.nan),
                    "pred_act_pos": _sf(src, colmap.pred_afrr_activation_price_pos, np.nan),
                    "true_act_pos": _sf(src, colmap.true_afrr_activation_price_pos, np.nan),
                    "pred_act_neg": _sf(src, colmap.pred_afrr_activation_price_neg, np.nan),
                    "true_act_neg": _sf(src, colmap.true_afrr_activation_price_neg, np.nan),
                    "pred_rate_pos": _sf(src, colmap.pred_afrr_activation_rate_pos, np.nan),
                    "pred_rate_neg": _sf(src, colmap.pred_afrr_activation_rate_neg, np.nan),
                    "true_rate_pos": _sf(src, colmap.true_afrr_activation_rate_pos, np.nan),
                    "true_rate_neg": _sf(src, colmap.true_afrr_activation_rate_neg, np.nan),
                }
                if str(run_mode).strip().lower() == "advanced_ml":
                    critical_inputs["pred_da_price_p05"] = _sf(src, f"{colmap.pred_da_price}_p05", np.nan)
                    critical_inputs["pred_da_price_p10"] = _sf(src, f"{colmap.pred_da_price}_p10", np.nan)
                    critical_inputs["pred_da_price_p90"] = _sf(src, f"{colmap.pred_da_price}_p90", np.nan)
                    critical_inputs["pred_da_price_p95"] = _sf(src, f"{colmap.pred_da_price}_p95", np.nan)
                bad = [k for k, v in critical_inputs.items() if not np.isfinite(float(v))]
                if bad:
                    raise ValueError(
                        f"Critical clearing input missing/non-finite at {pd.to_datetime(ts, utc=True)}: "
                        + ", ".join(bad)
                    )

                cleared = self._apply_market_clearing(
                    target_time_utc=pd.to_datetime(ts, utc=True, errors="coerce"),
                    is_perfect_foresight=is_perfect_foresight,
                    planned_charge_mw=charge_plan,
                    planned_discharge_mw=discharge_plan,
                    planned_reserve_pos_mw=reserve_pos_plan,
                    planned_reserve_neg_mw=reserve_neg_plan,
                    planned_bem_only_pos_mw=bem_only_pos_plan,
                    planned_bem_only_neg_mw=bem_only_neg_plan,
                    # Never default DA prediction to 0.0 for bid pricing.
                    # Use the already-selected planning-side DA forecast instead.
                    pred_da_price=float(da_price_plan),
                    pred_da_price_p05=critical_inputs.get("pred_da_price_p05", np.nan),
                    pred_da_price_p10=critical_inputs.get("pred_da_price_p10", np.nan),
                    pred_da_price_p90=critical_inputs.get("pred_da_price_p90", np.nan),
                    pred_da_price_p95=critical_inputs.get("pred_da_price_p95", np.nan),
                    true_da_price=critical_inputs["true_da_price"],
                    pred_cap_pos=critical_inputs["pred_cap_pos"],
                    true_cap_pos=critical_inputs["true_cap_pos"],
                    pred_cap_neg=critical_inputs["pred_cap_neg"],
                    true_cap_neg=critical_inputs["true_cap_neg"],
                    pred_act_pos=critical_inputs["pred_act_pos"],
                    true_act_pos=critical_inputs["true_act_pos"],
                    pred_act_neg=critical_inputs["pred_act_neg"],
                    true_act_neg=critical_inputs["true_act_neg"],
                    true_rate_pos=critical_inputs["true_rate_pos"],
                    true_rate_neg=critical_inputs["true_rate_neg"],
                    pred_rate_pos=critical_inputs["pred_rate_pos"],
                    pred_rate_neg=critical_inputs["pred_rate_neg"],
                    soc_now=float(executed_s),
                    pred_act_pos_q10=_sf(src, "pred_afrr_activation_price_pos_p10", np.nan),
                    pred_act_pos_q50=_sf(src, "pred_afrr_activation_price_pos_p50", np.nan),
                    pred_act_pos_q90=_sf(src, "pred_afrr_activation_price_pos_p90", np.nan),
                    pred_act_neg_q10=_sf(src, "pred_afrr_activation_price_neg_p10", np.nan),
                    pred_act_neg_q50=_sf(src, "pred_afrr_activation_price_neg_p50", np.nan),
                    pred_act_neg_q90=_sf(src, "pred_afrr_activation_price_neg_p90", np.nan),
                    obligation_pos_mw=float(afrr_cap_pos_lockbook.get(pd.to_datetime(ts, utc=True), 0.0)),
                    obligation_neg_mw=float(afrr_cap_neg_lockbook.get(pd.to_datetime(ts, utc=True), 0.0)),
                    obligation_energy_pos=float(afrr_energy_pos_lockbook.get(pd.to_datetime(ts, utc=True), np.nan)),
                    obligation_energy_neg=float(afrr_energy_neg_lockbook.get(pd.to_datetime(ts, utc=True), np.nan)),
                    planned_reserve_pos_bins_mw=[
                        float(pd.to_numeric(pd.Series([getattr(r, f"reserve_pos_bin_{b}_mw", 0.0)]), errors="coerce").iloc[0])
                        for b in range(len(self.afrr_quantile_bins))
                    ],
                    planned_reserve_neg_bins_mw=[
                        float(pd.to_numeric(pd.Series([getattr(r, f"reserve_neg_bin_{b}_mw", 0.0)]), errors="coerce").iloc[0])
                        for b in range(len(self.afrr_quantile_bins))
                    ],
                    pred_cap_pos_bins_eur_mw=[
                        _sf(src, f"pred_afrr_capacity_price_pos_{q}", np.nan) for q in self.afrr_quantile_bins
                    ],
                    pred_cap_neg_bins_eur_mw=[
                        _sf(src, f"pred_afrr_capacity_price_neg_{q}", np.nan) for q in self.afrr_quantile_bins
                    ],
                )
                clearing_records.append(cleared)
                executed_s, _ = self._settle_one_hour(
                    soc=executed_s,
                    charge=float(cleared["executed_charge_mw"]),
                    discharge=float(cleared["executed_discharge_mw"]),
                    da_charge_mw=float(cleared["executed_charge_mw"]),
                    da_discharge_mw=float(cleared["executed_discharge_mw"]),
                    id_charge_mw=float(pending_id_charge_mw),
                    id_discharge_mw=float(pending_id_discharge_mw),
                    reserve_pos=float(cleared["executed_reserve_pos_mw"]),
                    reserve_neg=float(cleared["executed_reserve_neg_mw"]),
                    da_price=_sf(src, colmap.true_da_price, 0.0),
                    cap_pos=_sf(src, colmap.true_afrr_capacity_price_pos, 0.0),
                    cap_neg=_sf(src, colmap.true_afrr_capacity_price_neg, 0.0),
                    act_pos_price=_sf(src, colmap.true_afrr_activation_price_pos, 0.0),
                    act_neg_price=_sf(src, colmap.true_afrr_activation_price_neg, 0.0),
                    act_pos_rate=float(cleared["executed_rate_pos"]),
                    act_neg_rate=float(cleared["executed_rate_neg"]),
                    id_recourse_reason_hint=str(pending_id_reason),
                )
                clearing_records[-1]["pending_id_charge_mw"] = float(pending_id_charge_mw)
                clearing_records[-1]["pending_id_discharge_mw"] = float(pending_id_discharge_mw)
                clearing_records[-1]["pending_id_buy_mwh"] = float(pending_id_charge_mw * self.dt_h / max(self.eta_in, 1e-12))
                clearing_records[-1]["pending_id_sell_mwh"] = float(pending_id_discharge_mw * self.dt_h * self.eta_out)
                clearing_records[-1]["pending_id_recourse_reason"] = str(pending_id_reason)
                executed_soc_vals.append(float(executed_s))
                shock_sources: list[str] = []
                if float(cleared.get("submitted_da_buy_mw", 0.0)) > 0 and not bool(cleared.get("da_buy_accepted", 0.0)):
                    shock_sources.append("da_reject")
                if float(cleared.get("submitted_da_sell_mw", 0.0)) > 0 and not bool(cleared.get("da_sell_accepted", 0.0)):
                    shock_sources.append("da_reject")
                if float(cleared.get("submitted_afrr_pos_mw", 0.0)) > 0 and not bool(cleared.get("afrr_cap_pos_awarded", 0.0)):
                    shock_sources.append("afrr_capacity_reject")
                if float(cleared.get("submitted_afrr_neg_mw", 0.0)) > 0 and not bool(cleared.get("afrr_cap_neg_awarded", 0.0)):
                    shock_sources.append("afrr_capacity_reject")
                if bool(cleared.get("afrr_cap_pos_awarded", 0.0)) and not bool(cleared.get("afrr_act_pos_accepted", 0.0)):
                    shock_sources.append("afrr_activation_reject")
                if bool(cleared.get("afrr_cap_neg_awarded", 0.0)) and not bool(cleared.get("afrr_act_neg_accepted", 0.0)):
                    shock_sources.append("afrr_activation_reject")
                shock_source_vals.append("|".join(sorted(set(shock_sources))) if shock_sources else "none")
                # Decide ID rescue for next hour (t+1) and store as pending.
                # Uses post-settlement SoC and next-hour DA plan + aFRR obligation.
                next_idx = step_idx + 1
                if perms.allow_id and next_idx < len(take):
                    next_row = take.iloc[next_idx]
                    next_ts = pd.to_datetime(next_row[colmap.timestamp], utc=True, errors="coerce")
                    next_da_charge = float(pd.to_numeric(pd.Series([next_row.get("charge_mw", 0.0)]), errors="coerce").iloc[0])
                    next_da_discharge = float(pd.to_numeric(pd.Series([next_row.get("discharge_mw", 0.0)]), errors="coerce").iloc[0])
                    next_reserve_pos_plan = float(pd.to_numeric(pd.Series([next_row.get("reserve_pos_mw", 0.0)]), errors="coerce").iloc[0])
                    next_reserve_neg_plan = float(pd.to_numeric(pd.Series([next_row.get("reserve_neg_mw", 0.0)]), errors="coerce").iloc[0])
                    next_ob_pos = float(afrr_cap_pos_lockbook.get(next_ts, 0.0)) if pd.notna(next_ts) else 0.0
                    next_ob_neg = float(afrr_cap_neg_lockbook.get(next_ts, 0.0)) if pd.notna(next_ts) else 0.0
                    reserve_pos_next = max(next_reserve_pos_plan, next_ob_pos)
                    reserve_neg_next = max(next_reserve_neg_plan, next_ob_neg)
                    pending_id_charge_mw, pending_id_discharge_mw, pending_id_reason = self._plan_id_rescue_for_next_hour(
                        soc_next=float(executed_s),
                        reserve_pos_next_mw=reserve_pos_next,
                        reserve_neg_next_mw=reserve_neg_next,
                        da_charge_next_mw=next_da_charge,
                        da_discharge_next_mw=next_da_discharge,
                    )
                else:
                    pending_id_charge_mw = 0.0
                    pending_id_discharge_mw = 0.0
                    pending_id_reason = "none"
                soc = float(executed_s)
                i += 1
                _log_progress()
                if i >= n:
                    break

            take["planned_soc_mwh"] = pd.Series(planned_soc_vals, index=take.index, dtype="float64")
            take["executed_soc_mwh"] = pd.Series(executed_soc_vals, index=take.index, dtype="float64")
            take["soc_shock_mwh"] = take["executed_soc_mwh"] - take["planned_soc_mwh"]
            take["soc_before_mwh"] = pd.Series(soc_before_vals, index=take.index, dtype="float64")
            take["soc_after_planned_mwh"] = take["planned_soc_mwh"]
            take["soc_after_executed_mwh"] = take["executed_soc_mwh"]
            take["shock_source"] = pd.Series(shock_source_vals, index=take.index, dtype="string")
            if clearing_records:
                clr_df = pd.DataFrame(clearing_records, index=take.index)
                # Add clearing columns in one block to avoid DataFrame fragmentation
                # from repeated per-column inserts.
                overlap = [c for c in clr_df.columns if c in take.columns]
                if overlap:
                    take = take.drop(columns=overlap)
                take = pd.concat([take, clr_df], axis=1)

            # Boundary: keep internal raw names, export plan_* names to dispatch output.
            take_for_output = take.copy()
            dispatch_rename_dict = {
                "charge_mw": "plan_charge_mw",
                "discharge_mw": "plan_discharge_mw",
                "reserve_pos_mw": "plan_reserve_pos_mw",
                "reserve_neg_mw": "plan_reserve_neg_mw",
                "bem_only_pos_mw": "plan_bem_only_pos_mw",
                "bem_only_neg_mw": "plan_bem_only_neg_mw",
                "aFRR_Capacity_Won_MW": "plan_aFRR_Capacity_Won_MW",
            }
            existing_targets = [v for v in dispatch_rename_dict.values() if v in take_for_output.columns]
            if existing_targets:
                take_for_output.drop(columns=existing_targets, inplace=True)
            take_for_output.rename(columns=dispatch_rename_dict, inplace=True)
            decisions.append(take_for_output)

        if not decisions:
            empty_dispatch = pd.DataFrame(
                columns=[
                    colmap.timestamp,
                    "plan_charge_mw",
                    "plan_discharge_mw",
                    "plan_reserve_pos_mw",
                    "plan_reserve_neg_mw",
                    "id_charge_mw",
                    "id_discharge_mw",
                    "pending_id_charge_mw",
                    "pending_id_discharge_mw",
                    "is_precleared",
                    "is_charging",
                    "soc_lp_mwh",
                    "planned_soc_mwh",
                    "executed_soc_mwh",
                    "soc_shock_mwh",
                    "soc_before_mwh",
                    "soc_after_planned_mwh",
                    "soc_after_executed_mwh",
                    "shock_source",
                    "predicted_objective_eur",
                    *CANONICAL_PREDICTION_COLUMNS,
                ]
            )
            empty_plan = pd.DataFrame(
                columns=[
                    "snapshot_time_utc",
                    "target_time_utc",
                    "lead_time_h",
                    "plan_charge_mw",
                    "plan_discharge_mw",
                    "plan_reserve_pos_mw",
                    "plan_reserve_neg_mw",
                    "planned_reserve_mw",
                    "predicted_price",
                    "da_bid_locked",
                    "is_charging",
                    "soc_lp_mwh",
                    "predicted_objective_eur",
                ]
            )
            return empty_dispatch, empty_plan
        _log_progress(force=True, note="completed")
        out = pd.concat(decisions, ignore_index=True)
        out = out.sort_values(colmap.timestamp).reset_index(drop=True)
        plan_out = pd.concat(plan_history, ignore_index=True) if plan_history else pd.DataFrame()
        if not plan_out.empty:
            plan_out = plan_out.sort_values(["snapshot_time_utc", "target_time_utc"]).reset_index(drop=True)
        return out, plan_out

    def settle_dispatch(
        self,
        df: pd.DataFrame,
        dispatch: pd.DataFrame,
        colmap: BacktestColumnMap,
        *,
        predicted_settlement: bool,
        apply_market_clearing: bool = False,
        perfect_foresight_mode: bool = False,
    ) -> pd.DataFrame:
        """Settle dispatch against either predicted or realized market values."""
        # Merge guardrails: timestamp integrity on both sides.
        if colmap.timestamp not in df.columns:
            raise ValueError(f"Missing merge key '{colmap.timestamp}' in input df.")
        if colmap.timestamp not in dispatch.columns:
            raise ValueError(f"Missing merge key '{colmap.timestamp}' in dispatch df.")
        if not pd.Index(df[colmap.timestamp]).is_unique:
            raise ValueError(f"Input df merge key '{colmap.timestamp}' is not unique.")
        if not pd.Index(dispatch[colmap.timestamp]).is_unique:
            raise ValueError(f"Dispatch merge key '{colmap.timestamp}' is not unique.")
        simulation_start_time = pd.to_datetime(df[colmap.timestamp], utc=True, errors="coerce").dropna().min()

        def _assert_finite_cols(frame: pd.DataFrame, cols: list[str], *, ctx: str) -> None:
            if not frame.columns.is_unique:
                dups = pd.Index(frame.columns)[pd.Index(frame.columns).duplicated()].unique().tolist()
                raise ValueError(f"{ctx}: duplicate column labels detected: {dups}")
            missing = [c for c in cols if c not in frame.columns]
            if missing:
                raise ValueError(f"{ctx}: missing critical columns: {missing}")
            bad_parts: list[str] = []
            for c in cols:
                s = pd.to_numeric(frame[c], errors="coerce")
                bad = s.isna() | ~np.isfinite(s.to_numpy(dtype=float))
                if bool(bad.any()):
                    bad_parts.append(f"{c} bad={int(bad.sum())}/{len(s)}")
            if bad_parts:
                raise ValueError(f"{ctx}: non-finite critical values detected: " + " | ".join(bad_parts))

        if predicted_settlement:
            da_col = colmap.pred_da_price
            cap_pos_col = colmap.pred_afrr_capacity_price_pos
            cap_neg_col = colmap.pred_afrr_capacity_price_neg
            act_pos_col = colmap.pred_afrr_activation_price_pos
            act_neg_col = colmap.pred_afrr_activation_price_neg
            rate_pos_col = colmap.pred_afrr_activation_rate_pos
            rate_neg_col = colmap.pred_afrr_activation_rate_neg
            kind = "pred"
        else:
            da_col = colmap.true_da_price
            cap_pos_col = colmap.true_afrr_capacity_price_pos
            cap_neg_col = colmap.true_afrr_capacity_price_neg
            act_pos_col = colmap.true_afrr_activation_price_pos
            act_neg_col = colmap.true_afrr_activation_price_neg
            rate_pos_col = colmap.true_afrr_activation_rate_pos
            rate_neg_col = colmap.true_afrr_activation_rate_neg
            kind = "real"

        dispatch_cols = [colmap.timestamp]
        dispatch_cols += [c for c in DISPATCH_DECISION_COLS if c in dispatch.columns]
        dispatch_metadata_cols = [c for c in DISPATCH_METADATA_COLS if c in dispatch.columns]
        dispatch_clearing_cols = list(SETTLEMENT_CLEARING_COLS)
        dispatch_clearing_cols += [
            "id_charge_mw",
            "id_discharge_mw",
            "pending_id_charge_mw",
            "pending_id_discharge_mw",
        ]
        dispatch_clearing_cols.extend(
            [
                c
                for c in dispatch.columns
                if c.startswith("submitted_afrr_")
                or c.startswith("executed_afrr_")
                or c.startswith("afrr_bin_")
            ]
        )
        dispatch_cols = dispatch_cols + dispatch_metadata_cols + [c for c in dispatch_clearing_cols if c in dispatch.columns]
        # Deduplicate composed dispatch selection while preserving order.
        # This prevents left-side duplicate labels from slice requests like df[['A','A']].
        dispatch_cols = list(dict.fromkeys(dispatch_cols))
        # Backward compatibility: fallback/legacy dispatch paths may not include
        # all expected execution columns; synthesize safe defaults.
        required_dispatch_defaults: dict[str, float | bool] = {
            colmap.timestamp: np.nan,
            "plan_charge_mw": 0.0,
            "plan_discharge_mw": 0.0,
            "plan_reserve_pos_mw": 0.0,
            "plan_reserve_neg_mw": 0.0,
            "id_charge_mw": 0.0,
            "id_discharge_mw": 0.0,
            "pending_id_charge_mw": 0.0,
            "pending_id_discharge_mw": 0.0,
            "is_precleared": False,
        }
        missing_defaults = {
            c: default_val
            for c, default_val in required_dispatch_defaults.items()
            if c not in dispatch.columns
        }
        if missing_defaults:
            # Add missing default columns in one block to avoid DataFrame
            # fragmentation caused by repeated per-column inserts.
            defaults_df = pd.DataFrame(
                {c: np.full(len(dispatch), default_val) for c, default_val in missing_defaults.items()},
                index=dispatch.index,
            )
            dispatch = pd.concat([dispatch, defaults_df], axis=1)

        if predicted_settlement and all(c in dispatch.columns for c in [da_col, cap_pos_col, cap_neg_col, act_pos_col, act_neg_col, rate_pos_col, rate_neg_col]):
            selected_cols = list(
                dict.fromkeys(
                    dispatch_cols + [da_col, cap_pos_col, cap_neg_col, act_pos_col, act_neg_col, rate_pos_col, rate_neg_col]
                )
            )
            merged = dispatch[selected_cols].copy()
            expected_rows = len(dispatch)
            if len(merged) != expected_rows:
                raise ValueError(
                    f"Merge row count mismatch! Expected {expected_rows}, got {len(merged)}."
                )
        else:
            base_cols = [
                colmap.timestamp,
                da_col,
                cap_pos_col,
                cap_neg_col,
                act_pos_col,
                act_neg_col,
                rate_pos_col,
                rate_neg_col,
            ]
            # For realized settlement with clearing, also need prediction-side bid references.
            if apply_market_clearing and not predicted_settlement:
                base_cols.extend(
                    [
                        colmap.pred_da_price,
                        f"{colmap.pred_da_price}_p05",
                        f"{colmap.pred_da_price}_p10",
                        f"{colmap.pred_da_price}_p90",
                        f"{colmap.pred_da_price}_p95",
                        colmap.pred_afrr_capacity_price_pos,
                        colmap.pred_afrr_capacity_price_neg,
                        colmap.pred_afrr_activation_price_pos,
                        colmap.pred_afrr_activation_price_neg,
                        colmap.pred_afrr_activation_rate_pos,
                        colmap.pred_afrr_activation_rate_neg,
                    ]
                )
            base_cols = [c for c in dict.fromkeys(base_cols) if c in df.columns]
            market_side = df[base_cols].copy()
            # Decision-driven, strict boundary: settlement intent comes from
            # dispatch; market prices come from canonical market_side only.
            dispatch_clean = dispatch[[c for c in dispatch_cols if c in dispatch.columns]].copy()
            # Hard guard against namespace contamination from optimizer output.
            contaminating_cols = [
                c for c in dispatch_clean.columns
                if c.startswith("pred_") or c.startswith("true_")
            ]
            if contaminating_cols:
                dispatch_clean = dispatch_clean.drop(columns=contaminating_cols)
            if "is_precleared" in dispatch_clean.columns:
                dispatch_clean["is_precleared"] = dispatch_clean["is_precleared"].astype(bool)
            else:
                dispatch_clean["is_precleared"] = pd.Series(False, index=dispatch_clean.index, dtype=bool)
            # Dynamic overlap pruning: dispatch intent is source of truth.
            # If upstream mutated df with decision-like columns, drop overlaps
            # from right side to prevent duplicate labels/collisions.
            overlap = [
                c for c in market_side.columns
                if c in dispatch_clean.columns and c != colmap.timestamp
            ]
            if overlap:
                df_safe = market_side.drop(columns=overlap)
            else:
                df_safe = market_side
            # Settlement must be decision-driven: only timestamps that exist in
            # dispatch are financially settled (prevents lookback/initialization
            # rows from df entering realized reclearing with missing decision data).
            merged = self._guarded_merge(
                dispatch_clean,
                df_safe,
                on=colmap.timestamp,
                how="left",
                validate_row_count=True,
            )
            # Data-loss check: only meaningful when canonical pred price already
            # existed on df side before merge. If it did, left-merge must not
            # increase missingness.
            if colmap.pred_da_price in market_side.columns:
                before = pd.to_numeric(df[colmap.pred_da_price], errors="coerce")
                before_bad = int((before.isna() | ~np.isfinite(before.to_numpy(dtype=float))).sum())
                after = pd.to_numeric(merged[colmap.pred_da_price], errors="coerce")
                after_bad = int((after.isna() | ~np.isfinite(after.to_numpy(dtype=float))).sum())
                if after_bad > before_bad:
                    raise ValueError(
                        "Data loss! Canonical prices were overwritten or lost "
                        f"(pred_da_price bad before={before_bad}, after={after_bad})."
                    )

        # Expanded post-merge critical schema assertions.
        pred_critical_cols = [getattr(colmap, k) for k in CRITICAL_PRED_COL_KEYS]
        true_critical_cols = [getattr(colmap, k) for k in CRITICAL_TRUE_COL_KEYS]
        # Predicted inputs are strictly required for predicted settlement.
        # For realized settlement with clearing, pred_* checks are performed in the
        # per-row reclearing branch only when precleared execution columns are absent.
        if predicted_settlement:
            missing_pred = [c for c in pred_critical_cols if c not in merged.columns]
            if missing_pred:
                raise ValueError(f"post-merge predicted schema: missing critical columns: {missing_pred}")
            _assert_finite_cols(merged, pred_critical_cols, ctx="post-merge predicted schema")
        if (not predicted_settlement) or perfect_foresight_mode:
            missing_true = [c for c in true_critical_cols if c not in merged.columns]
            if missing_true:
                raise ValueError(f"post-merge realized/perfect_foresight schema: missing critical columns: {missing_true}")
            _assert_finite_cols(merged, true_critical_cols, ctx="post-merge realized/perfect_foresight schema")

        da_quant_cols_required = [
            f"{colmap.pred_da_price}_p05",
            f"{colmap.pred_da_price}_p10",
            f"{colmap.pred_da_price}_p90",
            f"{colmap.pred_da_price}_p95",
        ]
        require_da_quantiles_here = bool(self.da_bid_fail_fast_debug) or any(
            c in merged.columns for c in da_quant_cols_required
        )

        soc = self.soc_init
        financial_results: list[dict[str, float | str | pd.Timestamp]] = []
        for r in merged.itertuples(index=False):
            ts = getattr(r, colmap.timestamp)
            ts_utc = pd.to_datetime(ts, utc=True, errors="coerce")
            if pd.notna(simulation_start_time) and pd.notna(ts_utc) and ts_utc < simulation_start_time:
                # Skip initialization/lookback rows that are outside active simulation horizon.
                continue
            def _g(name: str, default: float = np.nan) -> float:
                try:
                    return float(getattr(r, name))
                except Exception:
                    return float(default)
            charge = float(getattr(r, "plan_charge_mw"))
            discharge = float(getattr(r, "plan_discharge_mw"))
            reserve_pos = float(getattr(r, "plan_reserve_pos_mw"))
            reserve_neg = float(getattr(r, "plan_reserve_neg_mw"))
            id_charge = _g("id_charge_mw", _g("pending_id_charge_mw", 0.0))
            id_discharge = _g("id_discharge_mw", _g("pending_id_discharge_mw", 0.0))
            rate_pos = float(getattr(r, rate_pos_col))
            rate_neg = float(getattr(r, rate_neg_col))

            clearing_rec: dict[str, float | str] = {}
            if apply_market_clearing and not predicted_settlement:
                row_is_precleared = bool(getattr(r, "is_precleared", False))
                if row_is_precleared and (not perfect_foresight_mode):
                    out = self._settle_realized_precleared_row(
                        row=r,
                        getf=_g,
                        charge=charge,
                        discharge=discharge,
                        reserve_pos=reserve_pos,
                        reserve_neg=reserve_neg,
                        rate_pos=rate_pos,
                        rate_neg=rate_neg,
                        id_charge=id_charge,
                        id_discharge=id_discharge,
                        dispatch_clearing_cols=dispatch_clearing_cols,
                        merged_columns=set(merged.columns),
                    )
                    charge = float(out["charge"])
                    discharge = float(out["discharge"])
                    reserve_pos = float(out["reserve_pos"])
                    reserve_neg = float(out["reserve_neg"])
                    rate_pos = float(out["rate_pos"])
                    rate_neg = float(out["rate_neg"])
                    id_charge = float(out["id_charge"])
                    id_discharge = float(out["id_discharge"])
                    clearing_rec = dict(out["clearing_rec"])
                else:
                    out = self._settle_realized_reclearing_row(
                        row=r,
                        getf=_g,
                        ts_utc=ts_utc,
                        colmap=colmap,
                        perfect_foresight_mode=perfect_foresight_mode,
                        require_da_quantiles=require_da_quantiles_here,
                        charge=charge,
                        discharge=discharge,
                        reserve_pos=reserve_pos,
                        reserve_neg=reserve_neg,
                        bem_only_pos=float(getattr(r, "plan_bem_only_pos_mw")) if hasattr(r, "plan_bem_only_pos_mw") else 0.0,
                        bem_only_neg=float(getattr(r, "plan_bem_only_neg_mw")) if hasattr(r, "plan_bem_only_neg_mw") else 0.0,
                        soc=float(soc),
                    )
                    charge = float(out["charge"])
                    discharge = float(out["discharge"])
                    reserve_pos = float(out["reserve_pos"])
                    reserve_neg = float(out["reserve_neg"])
                    rate_pos = float(out["rate_pos"])
                    rate_neg = float(out["rate_neg"])
                    clearing_rec = dict(out["clearing_rec"])
            elif predicted_settlement:
                self._settle_predicted_row(
                    row=r,
                    getf=_g,
                    ts_utc=ts_utc,
                    colmap=colmap,
                )

            cap_bid_pos_settlement = (
                float(clearing_rec.get("settlement_cap_bid_price_pos_eur_mw"))
                if "settlement_cap_bid_price_pos_eur_mw" in clearing_rec
                else None
            )
            cap_bid_neg_settlement = (
                float(clearing_rec.get("settlement_cap_bid_price_neg_eur_mw"))
                if "settlement_cap_bid_price_neg_eur_mw" in clearing_rec
                else None
            )
            soc_start_hour = float(soc)
            soc, m = self._settle_one_hour(
                soc=soc,
                charge=charge,
                discharge=discharge,
                id_charge_mw=id_charge,
                id_discharge_mw=id_discharge,
                reserve_pos=reserve_pos,
                reserve_neg=reserve_neg,
                da_price=float(getattr(r, da_col)),
                cap_pos=float(getattr(r, cap_pos_col)),
                cap_neg=float(getattr(r, cap_neg_col)),
                act_pos_price=float(getattr(r, act_pos_col)),
                act_neg_price=float(getattr(r, act_neg_col)),
                act_pos_rate=rate_pos,
                act_neg_rate=rate_neg,
                cap_bid_pos=cap_bid_pos_settlement,
                cap_bid_neg=cap_bid_neg_settlement,
            )
            # Realized headroom audit (aligned to settlement-time commitments).
            committed_bem_pos_mw = float(clearing_rec.get("bem_only_submitted_pos_mw", 0.0))
            committed_bem_neg_mw = float(clearing_rec.get("bem_only_submitted_neg_mw", 0.0))
            avail_pos_mwh = max(0.0, soc_start_hour - self.soc_min)
            avail_neg_mwh = max(0.0, self.soc_max - soc_start_hour)
            locked_reserve_pos_mw = float(clearing_rec.get("fixed_reserve_obligation_pos_mw", 0.0))
            locked_reserve_neg_mw = float(clearing_rec.get("fixed_reserve_obligation_neg_mw", 0.0))
            # Protected-SoC validity must be obligation-driven:
            # use locked/awarded reserve obligations only (not merely submitted volume).
            req_pos_mwh = (
                float(locked_reserve_pos_mw) * self.reserve_activation_headroom_h
            ) / max(self.eta_out, 1e-12)
            req_neg_mwh = (
                float(locked_reserve_neg_mw) * self.reserve_activation_headroom_h
            ) * self.eta_in
            psv = self._compute_obligation_driven_protected_soc_bounds(
                soc_start_mwh=float(soc_start_hour),
                required_headroom_pos_mwh=float(req_pos_mwh),
                required_headroom_neg_mwh=float(req_neg_mwh),
                locked_reserve_pos_mw=float(locked_reserve_pos_mw),
                locked_reserve_neg_mw=float(locked_reserve_neg_mw),
                committed_bem_pos_mw=float(committed_bem_pos_mw),
                committed_bem_neg_mw=float(committed_bem_neg_mw),
                reserve_pos_mw=0.0,
                reserve_neg_mw=0.0,
            )
            m.update(
                {
                    "soc_start_mwh": soc_start_hour,
                    "soc_min_mwh": float(self.soc_min),
                    "soc_max_mwh": float(self.soc_max),
                    "physical_soc_min_mwh": float(psv["physical_soc_min_mwh"]),
                    "physical_soc_max_mwh": float(psv["physical_soc_max_mwh"]),
                    "locked_reserve_pos_mw": float(locked_reserve_pos_mw),
                    "locked_reserve_neg_mw": float(locked_reserve_neg_mw),
                    "obligation_headroom_pos_active": float(psv["obligation_headroom_pos_active"]),
                    "obligation_headroom_neg_active": float(psv["obligation_headroom_neg_active"]),
                    "protected_soc_min_mwh": float(psv["protected_soc_min_mwh"]),
                    "protected_soc_max_mwh": float(psv["protected_soc_max_mwh"]),
                    "protected_soc_buffer_pos_mwh": float(psv["protected_soc_buffer_pos_mwh"]),
                    "protected_soc_buffer_neg_mwh": float(psv["protected_soc_buffer_neg_mwh"]),
                    "protected_soc_margin_pos_mwh": float(soc_start_hour - float(psv["protected_soc_min_mwh"])),
                    "protected_soc_margin_neg_mwh": float(float(psv["protected_soc_max_mwh"]) - soc_start_hour),
                    "protected_soc_violation_pos_mwh": float(psv["protected_soc_violation_pos_mwh"]),
                    "protected_soc_violation_neg_mwh": float(psv["protected_soc_violation_neg_mwh"]),
                    "protected_soc_violation_without_obligation": float(psv["protected_soc_violation_without_obligation"]),
                    "physical_soc_violation_pos_mwh": float(psv["physical_soc_violation_pos_mwh"]),
                    "physical_soc_violation_neg_mwh": float(psv["physical_soc_violation_neg_mwh"]),
                    "required_headroom_pos_mwh": float(req_pos_mwh),
                    "required_headroom_neg_mwh": float(req_neg_mwh),
                    "available_headroom_pos_mwh": float(avail_pos_mwh),
                    "available_headroom_neg_mwh": float(avail_neg_mwh),
                    "headroom_margin_pos_mwh": float(avail_pos_mwh - req_pos_mwh),
                    "headroom_margin_neg_mwh": float(avail_neg_mwh - req_neg_mwh),
                    "headroom_violation_pos_mwh": float(max(0.0, req_pos_mwh - avail_pos_mwh)),
                    "headroom_violation_neg_mwh": float(max(0.0, req_neg_mwh - avail_neg_mwh)),
                    "power_stack_pos_mw": float(max(0.0, discharge) + max(0.0, id_discharge) + max(0.0, reserve_pos) + max(0.0, committed_bem_pos_mw)),
                    "power_stack_neg_mw": float(max(0.0, charge) + max(0.0, id_charge) + max(0.0, reserve_neg) + max(0.0, committed_bem_neg_mw)),
                    "power_margin_pos_mw": float(
                        self.p_max_mw
                        - (max(0.0, discharge) + max(0.0, id_discharge) + max(0.0, reserve_pos) + max(0.0, committed_bem_pos_mw))
                    ),
                    "power_margin_neg_mw": float(
                        self.p_max_mw
                        - (max(0.0, charge) + max(0.0, id_charge) + max(0.0, reserve_neg) + max(0.0, committed_bem_neg_mw))
                    ),
                }
            )
            # Row-level activation split: BCM-linked vs explicit BEM-only.
            bem_only_pos_mwh = float(clearing_rec.get("bem_only_executed_pos_mwh", 0.0))
            bem_only_neg_mwh = float(clearing_rec.get("bem_only_executed_neg_mwh", 0.0))
            delivered_pos_mwh = float(m.get("delivered_activation_pos_mwh", 0.0))
            delivered_neg_mwh = float(m.get("delivered_activation_neg_mwh", 0.0))
            act_pos_price = float(getattr(r, act_pos_col))
            act_neg_price = float(getattr(r, act_neg_col))
            act_split = self._split_activation_revenue_components(
                delivered_pos_mwh=delivered_pos_mwh,
                delivered_neg_mwh=delivered_neg_mwh,
                bem_only_pos_mwh=bem_only_pos_mwh,
                bem_only_neg_mwh=bem_only_neg_mwh,
                act_pos_price_eur_mwh=act_pos_price,
                act_neg_price_eur_mwh=act_neg_price,
            )
            m.update(act_split)
            # Keep generic activation revenue internally consistent with split components.
            m["revenue_activation_eur"] = float(act_split["activation_revenue_reconciled_eur"])
            financial_results.append({colmap.timestamp: ts, **m, **clearing_rec})

        results_df = pd.DataFrame(financial_results)
        if results_df.empty:
            merged_active = merged.iloc[0:0].copy()
        else:
            active_ts = pd.to_datetime(results_df[colmap.timestamp], utc=True, errors="coerce")
            merged_ts = pd.to_datetime(merged[colmap.timestamp], utc=True, errors="coerce")
            merged_active = merged.loc[merged_ts.isin(set(active_ts.dropna().tolist()))].copy()

        overlap_cols = [
            c for c in results_df.columns
            if c != colmap.timestamp and c in merged_active.columns
        ]
        if overlap_cols:
            merged_active = merged_active.drop(columns=overlap_cols)

        out = self._guarded_merge(
            merged_active,
            results_df,
            on=colmap.timestamp,
            how="left",
            validate_row_count=True,
        )

        aux_clearing_cols = list(SETTLEMENT_CLEARING_COLS)
        aux_clearing_cols.extend(
            [
                c
                for c in out.columns
                if c.startswith("submitted_afrr_")
                or c.startswith("executed_afrr_")
                or c.startswith("afrr_bin_")
                or c.startswith("afrr_bcm_")
                or c.startswith("afrr_bem_")
                or c.startswith("bem_only_")
            ]
        )
        out.rename(
            columns={c: f"{kind}_{c}" for c in aux_clearing_cols if c in out.columns},
            inplace=True,
        )
        out.rename(columns={
            "pnl_eur": f"{kind}_pnl_eur",
            "da_buy_mwh": f"{kind}_da_buy_mwh",
            "da_sell_mwh": f"{kind}_da_sell_mwh",
            "id_charge_mw": f"{kind}_id_charge_mw",
            "id_discharge_mw": f"{kind}_id_discharge_mw",
            "id_buy_mwh": f"{kind}_id_buy_mwh",
            "id_sell_mwh": f"{kind}_id_sell_mwh",
            "act_pos_mwh": f"{kind}_act_pos_mwh",
            "act_neg_mwh": f"{kind}_act_neg_mwh",
            "revenue_da_eur": f"{kind}_revenue_da_eur",
            "cost_da_eur": f"{kind}_cost_da_eur",
            "revenue_id_eur": f"{kind}_revenue_id_eur",
            "cost_id_eur": f"{kind}_cost_id_eur",
            "revenue_capacity_eur": f"{kind}_revenue_capacity_eur",
            "revenue_activation_eur": f"{kind}_revenue_activation_eur",
            "activation_revenue_reconciled_eur": f"{kind}_activation_revenue_reconciled_eur",
            "bcm_linked_pos_activation_revenue_eur": f"{kind}_bcm_linked_pos_activation_revenue_eur",
            "bcm_linked_neg_activation_revenue_eur": f"{kind}_bcm_linked_neg_activation_revenue_eur",
            "bcm_linked_activation_revenue_eur": f"{kind}_bcm_linked_activation_revenue_eur",
            "transaction_cost_eur": f"{kind}_transaction_cost_eur",
            "degradation_cost_eur": f"{kind}_degradation_cost_eur",
            "aux_cost_eur": f"{kind}_aux_cost_eur",
            "missed_activation_mwh": f"{kind}_missed_activation_mwh",
            "missed_activation_pos_mwh": f"{kind}_missed_activation_pos_mwh",
            "missed_activation_neg_mwh": f"{kind}_missed_activation_neg_mwh",
            "requested_activation_pos_mwh": f"{kind}_requested_activation_pos_mwh",
            "requested_activation_neg_mwh": f"{kind}_requested_activation_neg_mwh",
            "delivered_activation_pos_mwh": f"{kind}_delivered_activation_pos_mwh",
            "delivered_activation_neg_mwh": f"{kind}_delivered_activation_neg_mwh",
            "missed_capacity_mw": f"{kind}_missed_capacity_mw",
            "missed_capacity_pos_mw": f"{kind}_missed_capacity_pos_mw",
            "missed_capacity_neg_mw": f"{kind}_missed_capacity_neg_mw",
            "awarded_capacity_pos_mw": f"{kind}_awarded_capacity_pos_mw",
            "awarded_capacity_neg_mw": f"{kind}_awarded_capacity_neg_mw",
            "physically_deliverable_capacity_pos_mw": f"{kind}_physically_deliverable_capacity_pos_mw",
            "physically_deliverable_capacity_neg_mw": f"{kind}_physically_deliverable_capacity_neg_mw",
            "delivered_capacity_pos_mw": f"{kind}_delivered_capacity_pos_mw",
            "delivered_capacity_neg_mw": f"{kind}_delivered_capacity_neg_mw",
            "requested_activation_revenue_eur": f"{kind}_requested_activation_revenue_eur",
            "delivered_activation_revenue_eur": f"{kind}_delivered_activation_revenue_eur",
            "missed_activation_revenue_eur": f"{kind}_missed_activation_revenue_eur",
            "penalty_activation_pos_eur": f"{kind}_penalty_activation_pos_eur",
            "penalty_activation_neg_eur": f"{kind}_penalty_activation_neg_eur",
            "penalty_activation_eur": f"{kind}_penalty_activation_eur",
            "penalty_capacity_pos_eur": f"{kind}_penalty_capacity_pos_eur",
            "penalty_capacity_neg_eur": f"{kind}_penalty_capacity_neg_eur",
            "penalty_capacity_eur": f"{kind}_penalty_capacity_eur",
            "penalty_eur": f"{kind}_penalty_eur",
            "net_cashflow_eur": f"{kind}_net_cashflow_eur",
            "soc_mwh": f"{kind}_soc_mwh",
            "soc_start_mwh": f"{kind}_soc_start_mwh",
            "soc_min_mwh": f"{kind}_soc_min_mwh",
            "soc_max_mwh": f"{kind}_soc_max_mwh",
            "locked_reserve_pos_mw": f"{kind}_locked_reserve_pos_mw",
            "locked_reserve_neg_mw": f"{kind}_locked_reserve_neg_mw",
            "protected_soc_min_mwh": f"{kind}_protected_soc_min_mwh",
            "protected_soc_max_mwh": f"{kind}_protected_soc_max_mwh",
            "protected_soc_margin_pos_mwh": f"{kind}_protected_soc_margin_pos_mwh",
            "protected_soc_margin_neg_mwh": f"{kind}_protected_soc_margin_neg_mwh",
            "protected_soc_violation_pos_mwh": f"{kind}_protected_soc_violation_pos_mwh",
            "protected_soc_violation_neg_mwh": f"{kind}_protected_soc_violation_neg_mwh",
            "required_headroom_pos_mwh": f"{kind}_required_headroom_pos_mwh",
            "required_headroom_neg_mwh": f"{kind}_required_headroom_neg_mwh",
            "available_headroom_pos_mwh": f"{kind}_available_headroom_pos_mwh",
            "available_headroom_neg_mwh": f"{kind}_available_headroom_neg_mwh",
            "headroom_margin_pos_mwh": f"{kind}_headroom_margin_pos_mwh",
            "headroom_margin_neg_mwh": f"{kind}_headroom_margin_neg_mwh",
            "headroom_violation_pos_mwh": f"{kind}_headroom_violation_pos_mwh",
            "headroom_violation_neg_mwh": f"{kind}_headroom_violation_neg_mwh",
            "power_stack_pos_mw": f"{kind}_power_stack_pos_mw",
            "power_stack_neg_mw": f"{kind}_power_stack_neg_mw",
            "power_margin_pos_mw": f"{kind}_power_margin_pos_mw",
            "power_margin_neg_mw": f"{kind}_power_margin_neg_mw",
        }, inplace=True)
        # Enforce strict realized namespace contract for settlement outcomes.
        if kind == "real":
            out.rename(columns={k: v for k, v in SETTLEMENT_RENAME_MAP_REAL.items() if k in out.columns}, inplace=True)
        return out

    def _settle_predicted_row(
        self,
        *,
        row: object,
        getf,
        ts_utc: pd.Timestamp,
        colmap: BacktestColumnMap,
    ) -> None:
        """Predicted settlement branch: schema checks are handled upstream."""
        return None

    def _settle_realized_precleared_row(
        self,
        *,
        row: object,
        getf,
        charge: float,
        discharge: float,
        reserve_pos: float,
        reserve_neg: float,
        rate_pos: float,
        rate_neg: float,
        id_charge: float,
        id_discharge: float,
        dispatch_clearing_cols: list[str],
        merged_columns: set[str],
    ) -> dict[str, object]:
        """Use precleared realized execution values; no pred_* accesses."""
        charge = getf("executed_charge_mw", charge)
        discharge = getf("executed_discharge_mw", discharge)
        reserve_pos = getf("executed_reserve_pos_mw", reserve_pos)
        reserve_neg = getf("executed_reserve_neg_mw", reserve_neg)
        rate_pos = getf("executed_rate_pos", rate_pos)
        rate_neg = getf("executed_rate_neg", rate_neg)
        id_charge = getf("id_charge_mw", getf("pending_id_charge_mw", id_charge))
        id_discharge = getf("id_discharge_mw", getf("pending_id_discharge_mw", id_discharge))
        clearing_rec: dict[str, float | str] = {}
        metadata_cols = set(SETTLEMENT_METADATA_COLS)
        for c in dispatch_clearing_cols:
            if c in merged_columns:
                if c in metadata_cols or c.endswith("_reason") or c.endswith("_quantile"):
                    clearing_rec[c] = str(getattr(row, c))
                else:
                    clearing_rec[c] = float(getattr(row, c))
        return {
            "charge": charge,
            "discharge": discharge,
            "reserve_pos": reserve_pos,
            "reserve_neg": reserve_neg,
            "rate_pos": rate_pos,
            "rate_neg": rate_neg,
            "id_charge": id_charge,
            "id_discharge": id_discharge,
            "clearing_rec": clearing_rec,
        }

    def _settle_realized_reclearing_row(
        self,
        *,
        row: object,
        getf,
        ts_utc: pd.Timestamp,
        colmap: BacktestColumnMap,
        perfect_foresight_mode: bool,
        require_da_quantiles: bool,
        charge: float,
        discharge: float,
        reserve_pos: float,
        reserve_neg: float,
        bem_only_pos: float,
        bem_only_neg: float,
        soc: float,
    ) -> dict[str, object]:
        """Re-run market-clearing for realized settlement when no precleared execution exists."""
        pred_da_price = float(
            getattr(row, "perfect_foresight_override_da_price")
            if hasattr(row, "perfect_foresight_override_da_price")
            else getattr(row, colmap.pred_da_price)
        )
        pred_cap_pos = float(
            getattr(row, "perfect_foresight_override_cap_pos")
            if hasattr(row, "perfect_foresight_override_cap_pos")
            else getattr(row, colmap.pred_afrr_capacity_price_pos)
        )
        pred_cap_neg = float(
            getattr(row, "perfect_foresight_override_cap_neg")
            if hasattr(row, "perfect_foresight_override_cap_neg")
            else getattr(row, colmap.pred_afrr_capacity_price_neg)
        )
        pred_act_pos = float(
            getattr(row, "perfect_foresight_override_act_pos")
            if hasattr(row, "perfect_foresight_override_act_pos")
            else getattr(row, colmap.pred_afrr_activation_price_pos)
        )
        pred_act_neg = float(
            getattr(row, "perfect_foresight_override_act_neg")
            if hasattr(row, "perfect_foresight_override_act_neg")
            else getattr(row, colmap.pred_afrr_activation_price_neg)
        )
        pred_rate_pos = float(
            getattr(row, "perfect_foresight_override_rate_pos")
            if hasattr(row, "perfect_foresight_override_rate_pos")
            else getattr(row, colmap.pred_afrr_activation_rate_pos)
        )
        pred_rate_neg = float(
            getattr(row, "perfect_foresight_override_rate_neg")
            if hasattr(row, "perfect_foresight_override_rate_neg")
            else getattr(row, colmap.pred_afrr_activation_rate_neg)
        )
        true_da_price = float(getattr(row, colmap.true_da_price))
        true_cap_pos = float(getattr(row, colmap.true_afrr_capacity_price_pos))
        true_cap_neg = float(getattr(row, colmap.true_afrr_capacity_price_neg))
        true_act_pos = float(getattr(row, colmap.true_afrr_activation_price_pos))
        true_act_neg = float(getattr(row, colmap.true_afrr_activation_price_neg))
        true_rate_pos = float(getattr(row, colmap.true_afrr_activation_rate_pos))
        true_rate_neg = float(getattr(row, colmap.true_afrr_activation_rate_neg))
        pred_da_price_p05 = float(getattr(row, f"{colmap.pred_da_price}_p05")) if require_da_quantiles else np.nan
        pred_da_price_p10 = float(getattr(row, f"{colmap.pred_da_price}_p10")) if require_da_quantiles else np.nan
        pred_da_price_p90 = float(getattr(row, f"{colmap.pred_da_price}_p90")) if require_da_quantiles else np.nan
        pred_da_price_p95 = float(getattr(row, f"{colmap.pred_da_price}_p95")) if require_da_quantiles else np.nan
        cleared = self._apply_market_clearing(
            target_time_utc=pd.to_datetime(ts_utc, utc=True, errors="coerce"),
            is_perfect_foresight=perfect_foresight_mode,
            planned_charge_mw=charge,
            planned_discharge_mw=discharge,
            planned_reserve_pos_mw=reserve_pos,
            planned_reserve_neg_mw=reserve_neg,
            planned_bem_only_pos_mw=bem_only_pos,
            planned_bem_only_neg_mw=bem_only_neg,
            pred_da_price=pred_da_price,
            pred_da_price_p05=pred_da_price_p05,
            pred_da_price_p10=pred_da_price_p10,
            pred_da_price_p90=pred_da_price_p90,
            pred_da_price_p95=pred_da_price_p95,
            true_da_price=true_da_price,
            pred_cap_pos=pred_cap_pos,
            true_cap_pos=true_cap_pos,
            pred_cap_neg=pred_cap_neg,
            true_cap_neg=true_cap_neg,
            pred_act_pos=pred_act_pos,
            true_act_pos=true_act_pos,
            pred_act_neg=pred_act_neg,
            true_act_neg=true_act_neg,
            true_rate_pos=true_rate_pos,
            true_rate_neg=true_rate_neg,
            pred_rate_pos=pred_rate_pos,
            pred_rate_neg=pred_rate_neg,
            soc_now=float(soc),
            pred_act_pos_q10=float(getattr(row, "pred_afrr_activation_price_pos_p10")) if hasattr(row, "pred_afrr_activation_price_pos_p10") else np.nan,
            pred_act_pos_q50=float(getattr(row, "pred_afrr_activation_price_pos_p50")) if hasattr(row, "pred_afrr_activation_price_pos_p50") else np.nan,
            pred_act_pos_q90=float(getattr(row, "pred_afrr_activation_price_pos_p90")) if hasattr(row, "pred_afrr_activation_price_pos_p90") else np.nan,
            pred_act_neg_q10=float(getattr(row, "pred_afrr_activation_price_neg_p10")) if hasattr(row, "pred_afrr_activation_price_neg_p10") else np.nan,
            pred_act_neg_q50=float(getattr(row, "pred_afrr_activation_price_neg_p50")) if hasattr(row, "pred_afrr_activation_price_neg_p50") else np.nan,
            pred_act_neg_q90=float(getattr(row, "pred_afrr_activation_price_neg_p90")) if hasattr(row, "pred_afrr_activation_price_neg_p90") else np.nan,
            obligation_pos_mw=getf("aFRR_Capacity_Won_Pos_MW", 0.0),
            obligation_neg_mw=getf("aFRR_Capacity_Won_Neg_MW", 0.0),
            obligation_energy_pos=float(getattr(row, "aFRR_Energy_Price_EUR_MWh_Pos")) if hasattr(row, "aFRR_Energy_Price_EUR_MWh_Pos") else np.nan,
            obligation_energy_neg=float(getattr(row, "aFRR_Energy_Price_EUR_MWh_Neg")) if hasattr(row, "aFRR_Energy_Price_EUR_MWh_Neg") else np.nan,
        )
        return {
            "charge": float(cleared["executed_charge_mw"]),
            "discharge": float(cleared["executed_discharge_mw"]),
            "reserve_pos": float(cleared["executed_reserve_pos_mw"]),
            "reserve_neg": float(cleared["executed_reserve_neg_mw"]),
            "rate_pos": float(cleared["executed_rate_pos"]),
            "rate_neg": float(cleared["executed_rate_neg"]),
            "clearing_rec": cleared,
        }

    def _audit_backtest_results(
        self,
        *,
        realized: pd.DataFrame,
        dispatch: pd.DataFrame,
        df_input: pd.DataFrame,
        colmap: BacktestColumnMap,
        epsilon: float = 1e-4,
    ) -> None:
        """Run strict post-run invariants on realized settlement output."""
        if realized.empty:
            raise RuntimeError("Backtest audit failed: realized settlement output is empty.")

        ts_col = colmap.timestamp
        if ts_col not in realized.columns or ts_col not in dispatch.columns or ts_col not in df_input.columns:
            raise RuntimeError(f"Backtest audit failed: missing timestamp column '{ts_col}' in audit inputs.")

        realized_ts = pd.to_datetime(realized[ts_col], utc=True, errors="coerce")
        dispatch_ts = pd.to_datetime(dispatch[ts_col], utc=True, errors="coerce")
        input_ts = pd.to_datetime(df_input[ts_col], utc=True, errors="coerce")
        if realized_ts.isna().any() or dispatch_ts.isna().any() or input_ts.isna().any():
            raise RuntimeError("Backtest audit failed: non-finite timestamps detected.")

        simulation_start_time = input_ts.min()
        expected_ts = pd.Index(dispatch_ts[dispatch_ts >= simulation_start_time]).drop_duplicates()
        got_ts = pd.Index(realized_ts).drop_duplicates()
        if len(realized) != len(expected_ts):
            raise RuntimeError(
                "Backtest audit failed: timeline row mismatch "
                f"(expected={len(expected_ts)}, got={len(realized)})."
            )
        if not expected_ts.equals(got_ts):
            raise RuntimeError("Backtest audit failed: timeline timestamps mismatch between dispatch and realized.")

        if "real_soc_mwh" not in realized.columns:
            raise RuntimeError("Backtest audit failed: missing 'real_soc_mwh' in realized output.")
        soc = pd.to_numeric(realized["real_soc_mwh"], errors="coerce")
        if soc.isna().any():
            raise RuntimeError("Backtest audit failed: non-finite values in 'real_soc_mwh'.")
        if ((soc < (self.soc_min - epsilon)) | (soc > (self.soc_max + epsilon))).any():
            raise RuntimeError("Backtest audit failed: SoC out of physical bounds.")

        req_cols = [
            "real_charge_mw",
            "real_discharge_mw",
            "real_act_pos_mwh",
            "real_act_neg_mwh",
            "real_aux_energy_mwh",
        ]
        missing_req = [c for c in req_cols if c not in realized.columns]
        if missing_req:
            raise RuntimeError(f"Backtest audit failed: missing columns for SoC mass-balance: {missing_req}")
        for c in req_cols:
            s = pd.to_numeric(realized[c], errors="coerce")
            if s.isna().any():
                raise RuntimeError(f"Backtest audit failed: non-finite values in '{c}'.")

        soc_prev = float(self.soc_init)
        for r in realized.itertuples(index=False):
            charge_mw = float(getattr(r, "real_charge_mw"))
            discharge_mw = float(getattr(r, "real_discharge_mw"))
            act_pos_grid = float(getattr(r, "real_act_pos_mwh"))
            act_neg_grid = float(getattr(r, "real_act_neg_mwh"))
            aux_mwh = float(getattr(r, "real_aux_energy_mwh"))
            id_charge_mw = float(getattr(r, "real_id_charge_mw")) if hasattr(r, "real_id_charge_mw") else 0.0
            id_discharge_mw = float(getattr(r, "real_id_discharge_mw")) if hasattr(r, "real_id_discharge_mw") else 0.0
            delta_soc = self._calculate_soc_delta(
                charge_mw=charge_mw,
                discharge_mw=discharge_mw,
                id_charge_mw=id_charge_mw,
                id_discharge_mw=id_discharge_mw,
                act_pos_mwh=act_pos_grid,
                act_neg_mwh=act_neg_grid,
                aux_mwh=aux_mwh,
                battery_specs={"eta_in": self.eta_in, "eta_out": self.eta_out},
                dt_h=self.dt_h,
            )
            theor_next = soc_prev + delta_soc
            theor_next = float(np.clip(theor_next, self.soc_min, self.soc_max))
            got_next = float(getattr(r, "real_soc_mwh"))
            if not np.isfinite(got_next) or abs(theor_next - got_next) > max(epsilon, 1e-6):
                ts = pd.to_datetime(getattr(r, ts_col), utc=True, errors="coerce")
                raise RuntimeError(
                    "Backtest audit failed: SoC mass-balance mismatch at "
                    f"{ts} (expected={theor_next:.6f}, got={got_next:.6f})."
                )
            soc_prev = got_next

        if "real_pnl_eur" not in realized.columns:
            raise RuntimeError("Backtest audit failed: missing 'real_pnl_eur'.")
        pnl = pd.to_numeric(realized["real_pnl_eur"], errors="coerce")
        if pnl.isna().any():
            raise RuntimeError("Backtest audit failed: NaN/non-finite detected in realized_total_pnl components.")

    @staticmethod
    def _sum_abs_present(frame: pd.DataFrame, cols: list[str]) -> float:
        total = 0.0
        for c in cols:
            if c in frame.columns:
                total += float(pd.to_numeric(frame[c], errors="coerce").fillna(0.0).abs().sum())
        return total

    def _validate_strategy_isolation_outputs(
        self,
        *,
        hourly: pd.DataFrame,
        allowed_markets: list[str] | tuple[str, ...] | set[str],
        tol: float = 1e-9,
    ) -> None:
        """Runtime assertions that final settled outputs respect strategy isolation."""
        perms = self.strategy_permissions_from_allowed_markets(allowed_markets)
        if perms.allow_da and (not perms.allow_bcm) and (not perms.allow_bem_only):
            afrr_cols = [
                "real_reserve_pos_mw",
                "real_reserve_neg_mw",
                "real_executed_reserve_pos_mw",
                "real_executed_reserve_neg_mw",
                "real_act_pos_mwh",
                "real_act_neg_mwh",
                "real_revenue_capacity_eur",
                "real_revenue_activation_eur",
                "real_submitted_afrr_pos_mw",
                "real_submitted_afrr_neg_mw",
            ]
            afrr_mag = self._sum_abs_present(hourly, afrr_cols)
            if afrr_mag > tol:
                raise RuntimeError(
                    "Strategy isolation check failed for da_only: non-zero aFRR activity detected "
                    f"(abs-sum={afrr_mag:.6f})."
                )
        elif (not perms.allow_da) and (perms.allow_bcm or perms.allow_bem_only):
            da_cols = [
                "real_charge_mw",
                "real_discharge_mw",
                "real_executed_charge_mw",
                "real_executed_discharge_mw",
                "real_da_buy_mwh",
                "real_da_sell_mwh",
                "real_revenue_da_eur",
                "real_cost_da_eur",
                "real_submitted_da_buy_mw",
                "real_submitted_da_sell_mw",
            ]
            da_mag = self._sum_abs_present(hourly, da_cols)
            if da_mag > tol:
                raise RuntimeError(
                    "Strategy isolation check failed for afrr_only: non-zero DA activity detected "
                    f"(abs-sum={da_mag:.6f})."
                )
        if not perms.allow_bem_only:
            bem_cols = [
                "real_bem_only_submitted_pos_mw",
                "real_bem_only_submitted_neg_mw",
                "real_bem_only_executed_pos_mw",
                "real_bem_only_executed_neg_mw",
            ]
            bem_mag = self._sum_abs_present(hourly, bem_cols)
            if bem_mag > tol:
                raise RuntimeError(
                    "Strategy isolation check failed: non-zero BEM-only activity while BEM-only is disabled "
                    f"(abs-sum={bem_mag:.6f})."
                )
        if not perms.allow_bcm:
            bcm_cols = [
                "real_revenue_capacity_eur",
                "real_afrr_cap_pos_awarded",
                "real_afrr_cap_neg_awarded",
                "real_locked_reserve_pos_mw",
                "real_locked_reserve_neg_mw",
            ]
            bcm_mag = self._sum_abs_present(hourly, bcm_cols)
            if bcm_mag > tol:
                raise RuntimeError(
                    "Strategy isolation check failed: non-zero BCM activity while BCM is disabled "
                    f"(abs-sum={bcm_mag:.6f})."
                )
        # ID policy guardrails
        id_cols = [
            "real_id_charge_mw",
            "real_id_discharge_mw",
            "real_pending_id_charge_mw",
            "real_pending_id_discharge_mw",
        ]
        id_mag = self._sum_abs_present(hourly, id_cols)
        if perms.id_mode == "none" and id_mag > tol:
            raise RuntimeError(
                "Strategy isolation check failed: non-zero ID activity while id_mode=none "
                f"(abs-sum={id_mag:.6f})."
            )
        if perms.id_mode == "technical_repair" and id_mag > tol and "real_id_trade_type" in hourly.columns:
            tags = hourly["real_id_trade_type"].fillna("").astype(str).str.strip().str.lower()
            bad = ((tags != "technical_repair") & (tags != "none")) & (
                pd.to_numeric(hourly.get("real_id_charge_mw", 0.0), errors="coerce").fillna(0.0).abs()
                + pd.to_numeric(hourly.get("real_id_discharge_mw", 0.0), errors="coerce").fillna(0.0).abs()
                > tol
            )
            if bool(bad.any()):
                raise RuntimeError(
                    "Strategy isolation check failed: non-technical ID trade type detected while id_mode=technical_repair."
                )

    def run(
        self,
        df: pd.DataFrame,
        colmap: BacktestColumnMap,
        *,
        use_rolling_horizon: bool = True,
        horizon_hours: int = 48,
        reopt_step_hours: int = 1,
        forecast_warehouse: dict[str, pd.DataFrame] | None = None,
        da_gate_hour_cet: int = 11,
        soc_feedback_mode: str = "realized",
        enforce_final_soc_min: bool = True,
        allowed_markets: list[str] | tuple[str, ...] | set[str] = ("DA", "aFRR"),
        strategy_name: str | None = None,
        id_mode: str | None = None,
        id_recourse_mode: str = "common",
        strict_simulation_validity: bool = True,
        enable_global_perfect_foresight: bool = False,
    ) -> BacktestOutputs:
        """Run optimization + predicted settlement + realized settlement."""
        if horizon_hours <= 0 or reopt_step_hours <= 0:
            raise ValueError("horizon_hours and reopt_step_hours must be > 0")
        self._infeasible_debug_dumps = []
        self._strategy_permissions = self.resolve_strategy_permissions(
            strategy_name=strategy_name,
            allowed_markets=allowed_markets,
            id_mode=id_mode,
            id_recourse_mode=id_recourse_mode,
        )
        self._id_recourse_mode = self._normalize_id_recourse_mode(id_recourse_mode)
        self._assert_valid_time_index(df, colmap.timestamp)
        def _run_isolated_path(
            *,
            path_df: pd.DataFrame,
            allowed_markets_local: tuple[str, ...],
            deterministic_local: bool,
            is_perfect_foresight_local: bool,
        ) -> tuple[float, bool, dict[str, float], pd.DataFrame]:
            allowed_local = {str(m).strip().lower() for m in allowed_markets_local}
            is_da_only = allowed_local == {"da"}
            is_afrr_only = allowed_local == {"afrr"}
            try:
                if use_rolling_horizon:
                    path_dispatch, _ = self.optimize_dispatch_rolling(
                        path_df,
                        colmap,
                        horizon_hours=horizon_hours,
                        reopt_step_hours=reopt_step_hours,
                        da_gate_hour_cet=da_gate_hour_cet,
                        soc_feedback_mode=soc_feedback_mode,
                        enforce_final_soc_min=enforce_final_soc_min,
                        deterministic_reserve_settlement=deterministic_local,
                        is_perfect_foresight=is_perfect_foresight_local,
                        allowed_markets=allowed_markets_local,
                        strategy_name=strategy_name,
                        id_mode=id_mode,
                        id_recourse_mode=id_recourse_mode,
                        run_mode=("perfect_foresight" if is_perfect_foresight_local else "advanced_ml"),
                        strict_simulation_validity=bool(strict_simulation_validity),
                    )
                else:
                    path_dispatch = self.optimize_dispatch(
                        path_df,
                        colmap,
                        soc_start=self.soc_init,
                        soc_end_target=None,
                        soc_end_min_target=self.soc_target_end if enforce_final_soc_min else None,
                        deterministic_reserve_settlement=deterministic_local,
                        allowed_markets=allowed_markets_local,
                    )
            except RuntimeError as exc:
                # Isolated market ablation can be infeasible under strict physics;
                # keep full backtest robust and report feasibility in summary.
                if "infeasible" in str(exc).lower():
                    return 0.0, False, {
                        "da_net_eur": 0.0,
                        "afrr_net_eur": 0.0,
                        "id_net_eur": 0.0,
                        "common_costs_eur": 0.0,
                        "terminal_value_eur": 0.0,
                        "pnl_excl_terminal_eur": 0.0,
                    }, pd.DataFrame()
                raise
            path_real = self.settle_dispatch(
                path_df,
                path_dispatch,
                colmap,
                predicted_settlement=False,
                apply_market_clearing=True,
                perfect_foresight_mode=is_perfect_foresight_local,
            )
            def _sum_any(frame: pd.DataFrame, *cands: str) -> float:
                for c in cands:
                    if c in frame.columns:
                        return float(pd.to_numeric(frame[c], errors="coerce").fillna(0.0).sum())
                return 0.0
            # Strategy-explicit component accounting to prevent cross-market
            # leakage in isolated ablation reports.
            if path_real.empty:
                da_net = 0.0
                afrr_net = 0.0
                id_net = 0.0
                common_costs = 0.0
            else:
                da_net = _sum_any(path_real, "revenue_da_eur", "real_revenue_da_eur") - _sum_any(path_real, "cost_da_eur", "real_cost_da_eur")
                afrr_net = _sum_any(path_real, "revenue_capacity_eur", "real_revenue_capacity_eur") + _sum_any(path_real, "revenue_activation_eur", "real_revenue_activation_eur")
                id_net = _sum_any(path_real, "revenue_id_eur", "real_revenue_id_eur") - _sum_any(path_real, "cost_id_eur", "real_cost_id_eur")
                common_costs = (
                    -_sum_any(path_real, "transaction_cost_eur", "real_transaction_cost_eur")
                    - _sum_any(path_real, "degradation_cost_eur", "real_degradation_cost_eur")
                    - _sum_any(path_real, "penalty_eur", "real_penalty_eur")
                )
            if path_real.empty:
                pnl_excl = 0.0
            elif is_da_only:
                pnl_excl = da_net + id_net + common_costs
            elif is_afrr_only:
                pnl_excl = afrr_net + id_net + common_costs
            else:
                pnl_excl = _sum_any(path_real, "pnl_eur", "real_pnl_eur")
            final_soc = float(path_real["soc_mwh"].iloc[-1]) if (not path_real.empty and "soc_mwh" in path_real.columns) else (
                float(path_real["real_soc_mwh"].iloc[-1]) if (not path_real.empty and "real_soc_mwh" in path_real.columns) else float(self.soc_init)
            )
            da_true_last_local = self._finite_numeric_series(
                path_df,
                colmap.true_da_price,
                fallback_cols=[colmap.pred_da_price],
                default=0.0,
            )
            terminal_price_local = max(
                0.0,
                float(da_true_last_local.iloc[-1]) if len(da_true_last_local) else 0.0,
            )
            # Terminal valuation is based on delta energy versus start SoC:
            # - above start SoC: positive value
            # - below start SoC: negative value (inventory deficit)
            terminal_delta_mwh_local = float(final_soc - self.soc_init)
            terminal_value_local = terminal_delta_mwh_local * self.eta_out * terminal_price_local
            # Keep isolated-path reporting strictly market-separated:
            # *_only_total_pnl_eur is reported excluding terminal carry value.
            # Terminal is exposed separately in diagnostics to avoid masking DA/aFRR
            # contribution differences when both paths end with similar SoC.
            return float(pnl_excl), True, {
                "da_net_eur": float(da_net),
                "afrr_net_eur": float(afrr_net),
                "id_net_eur": float(id_net),
                "common_costs_eur": float(common_costs),
                "terminal_value_eur": float(terminal_value_local),
                "pnl_excl_terminal_eur": float(pnl_excl),
                "pnl_incl_terminal_eur": float(pnl_excl + terminal_value_local),
            }, path_real

        plan_history = pd.DataFrame()
        if use_rolling_horizon:
            dispatch, plan_history = self.optimize_dispatch_rolling(
                df,
                colmap,
                horizon_hours=horizon_hours,
                reopt_step_hours=reopt_step_hours,
                forecast_warehouse=forecast_warehouse,
                da_gate_hour_cet=da_gate_hour_cet,
                soc_feedback_mode=soc_feedback_mode,
                enforce_final_soc_min=enforce_final_soc_min,
                deterministic_reserve_settlement=False,
                allowed_markets=allowed_markets,
                strategy_name=strategy_name,
                id_mode=id_mode,
                id_recourse_mode=id_recourse_mode,
                run_mode="advanced_ml",
                strict_simulation_validity=bool(strict_simulation_validity),
            )
        else:
            dispatch = self.optimize_dispatch(
                df,
                colmap,
                soc_start=self.soc_init,
                soc_end_target=None,
                soc_end_min_target=self.soc_target_end if enforce_final_soc_min else None,
                deterministic_reserve_settlement=False,
                allowed_markets=allowed_markets,
            )
        with _phase_watchdog("settlement_predicted"):
            pred = self.settle_dispatch(df, dispatch, colmap, predicted_settlement=True)
        with _phase_watchdog("settlement_realized"):
            real = self.settle_dispatch(
                df,
                dispatch,
                colmap,
                predicted_settlement=False,
                apply_market_clearing=True,
                perfect_foresight_mode=False,
            )
        self._audit_backtest_results(realized=real, dispatch=dispatch, df_input=df, colmap=colmap)

        # Naive-24h benchmark run: y_t_hat := y_{t-24} for predicted market columns.
        naive_df = df.copy()
        naive_pairs = [
            (colmap.pred_da_price, colmap.true_da_price),
            (colmap.pred_afrr_capacity_price_pos, colmap.true_afrr_capacity_price_pos),
            (colmap.pred_afrr_capacity_price_neg, colmap.true_afrr_capacity_price_neg),
            (colmap.pred_afrr_activation_price_pos, colmap.true_afrr_activation_price_pos),
            (colmap.pred_afrr_activation_price_neg, colmap.true_afrr_activation_price_neg),
            (colmap.pred_afrr_activation_rate_pos, colmap.true_afrr_activation_rate_pos),
            (colmap.pred_afrr_activation_rate_neg, colmap.true_afrr_activation_rate_neg),
        ]
        naive_plan_history = pd.DataFrame()
        for pred_col, true_col in naive_pairs:
            if true_col in naive_df.columns:
                lagged = pd.to_numeric(naive_df[true_col], errors="coerce").shift(24)
                # If prediction column is absent (truth-only input mode), use true
                # series as finite fallback to avoid propagating all-NaN naive preds.
                if pred_col in naive_df.columns:
                    fallback = pd.to_numeric(naive_df[pred_col], errors="coerce")
                else:
                    fallback = pd.to_numeric(naive_df[true_col], errors="coerce")
                naive_df[pred_col] = lagged.fillna(fallback)
        if use_rolling_horizon:
            naive_dispatch, naive_plan_history = self.optimize_dispatch_rolling(
                naive_df,
                colmap,
                horizon_hours=horizon_hours,
                reopt_step_hours=reopt_step_hours,
                da_gate_hour_cet=da_gate_hour_cet,
                soc_feedback_mode=soc_feedback_mode,
                enforce_final_soc_min=enforce_final_soc_min,
                deterministic_reserve_settlement=False,
                allowed_markets=allowed_markets,
                strategy_name=strategy_name,
                id_mode=id_mode,
                id_recourse_mode=id_recourse_mode,
                run_mode="naive",
                strict_simulation_validity=bool(strict_simulation_validity),
            )
        else:
            naive_dispatch = self.optimize_dispatch(
                naive_df,
                colmap,
                soc_start=self.soc_init,
                soc_end_target=None,
                soc_end_min_target=self.soc_target_end if enforce_final_soc_min else None,
                deterministic_reserve_settlement=False,
                allowed_markets=allowed_markets,
            )
        with _phase_watchdog("settlement_naive_realized"):
            naive_real = self.settle_dispatch(
                df,
                naive_dispatch,
                colmap,
                predicted_settlement=False,
                apply_market_clearing=True,
                perfect_foresight_mode=False,
            )
        naive_real = naive_real.rename(
            columns={
                c: c.replace("real_", "naive_", 1)
                for c in naive_real.columns
                if c.startswith("real_")
            }
        )

        # Rolling perfect-foresight same-rules benchmark.
        # This remains a receding-horizon diagnostic benchmark and is not a
        # global-hindsight upper bound.
        perfect_foresight_df = df.copy()
        perfect_foresight_df[colmap.pred_da_price] = perfect_foresight_df[colmap.true_da_price]
        perfect_foresight_df[colmap.pred_afrr_capacity_price_pos] = perfect_foresight_df[colmap.true_afrr_capacity_price_pos]
        perfect_foresight_df[colmap.pred_afrr_capacity_price_neg] = perfect_foresight_df[colmap.true_afrr_capacity_price_neg]
        perfect_foresight_df[colmap.pred_afrr_activation_price_pos] = perfect_foresight_df[colmap.true_afrr_activation_price_pos]
        perfect_foresight_df[colmap.pred_afrr_activation_price_neg] = perfect_foresight_df[colmap.true_afrr_activation_price_neg]
        perfect_foresight_df[colmap.pred_afrr_activation_rate_pos] = perfect_foresight_df[colmap.true_afrr_activation_rate_pos]
        perfect_foresight_df[colmap.pred_afrr_activation_rate_neg] = perfect_foresight_df[colmap.true_afrr_activation_rate_neg]
        # Collapse full quantile surfaces in perfect_foresight mode so optimization and
        # settlement observe one deterministic "true" world.
        perfect_foresight_quantile_pairs = [
            (colmap.pred_da_price, colmap.true_da_price),
            (colmap.pred_afrr_capacity_price_pos, colmap.true_afrr_capacity_price_pos),
            (colmap.pred_afrr_capacity_price_neg, colmap.true_afrr_capacity_price_neg),
            (colmap.pred_afrr_activation_price_pos, colmap.true_afrr_activation_price_pos),
            (colmap.pred_afrr_activation_price_neg, colmap.true_afrr_activation_price_neg),
            (colmap.pred_afrr_activation_rate_pos, colmap.true_afrr_activation_rate_pos),
            (colmap.pred_afrr_activation_rate_neg, colmap.true_afrr_activation_rate_neg),
        ]
        for pred_col, true_col in perfect_foresight_quantile_pairs:
            for q in QUANTILE_COLUMNS:
                q_col = f"{pred_col}_{q}"
                if q_col in perfect_foresight_df.columns:
                    perfect_foresight_df[q_col] = perfect_foresight_df[true_col]
        # Apply stricter/longer solver settings locally for benchmark branches.
        _orig_tl = float(self.milp_time_limit_seconds)
        _orig_gap = float(self.milp_rel_gap)
        self.milp_time_limit_seconds = max(_orig_tl, 300.0)
        self.milp_rel_gap = min(_orig_gap, 1e-4)
        try:
            if use_rolling_horizon:
                perfect_foresight_dispatch, _ = self.optimize_dispatch_rolling(
                    perfect_foresight_df,
                    colmap,
                    horizon_hours=horizon_hours,
                    reopt_step_hours=reopt_step_hours,
                    da_gate_hour_cet=da_gate_hour_cet,
                    soc_feedback_mode=soc_feedback_mode,
                    enforce_final_soc_min=enforce_final_soc_min,
                    deterministic_reserve_settlement=True,
                    is_perfect_foresight=True,
                    allowed_markets=allowed_markets,
                    strategy_name=strategy_name,
                    id_mode=id_mode,
                    id_recourse_mode=id_recourse_mode,
                    run_mode="perfect_foresight",
                    strict_simulation_validity=bool(strict_simulation_validity),
                )
            else:
                perfect_foresight_dispatch = self.optimize_dispatch(
                    perfect_foresight_df,
                    colmap,
                    soc_start=self.soc_init,
                    soc_end_target=None,
                    soc_end_min_target=self.soc_target_end if enforce_final_soc_min else None,
                    deterministic_reserve_settlement=True,
                    allowed_markets=allowed_markets,
                )
        finally:
            self.milp_time_limit_seconds = _orig_tl
            self.milp_rel_gap = _orig_gap
        with _phase_watchdog("settlement_perfect_foresight_realized"):
            perfect_foresight_real = self.settle_dispatch(
                perfect_foresight_df,
                perfect_foresight_dispatch,
                colmap,
                predicted_settlement=False,
                apply_market_clearing=True,
                perfect_foresight_mode=True,
            )
        perfect_foresight_real = perfect_foresight_real.rename(
            columns={
                c: c.replace("real_", "perfect_foresight_", 1)
                for c in perfect_foresight_real.columns
                if c.startswith("real_")
            }
        )

        # Comparable rolling perfect-foresight benchmark under model-market
        # rules (same gate semantics as realized path).
        _orig_tl_cmp = float(self.milp_time_limit_seconds)
        _orig_gap_cmp = float(self.milp_rel_gap)
        self.milp_time_limit_seconds = max(_orig_tl_cmp, 300.0)
        self.milp_rel_gap = min(_orig_gap_cmp, 1e-5)
        try:
            if use_rolling_horizon:
                perfect_foresight_cmp_dispatch, _ = self.optimize_dispatch_rolling(
                    perfect_foresight_df,
                    colmap,
                    horizon_hours=horizon_hours,
                    reopt_step_hours=reopt_step_hours,
                    da_gate_hour_cet=da_gate_hour_cet,
                    soc_feedback_mode=soc_feedback_mode,
                    enforce_final_soc_min=enforce_final_soc_min,
                    deterministic_reserve_settlement=False,
                    is_perfect_foresight=False,
                    allowed_markets=allowed_markets,
                    strategy_name=strategy_name,
                    id_mode=id_mode,
                    id_recourse_mode=id_recourse_mode,
                    run_mode="perfect_foresight",
                    strict_simulation_validity=bool(strict_simulation_validity),
                )
            else:
                perfect_foresight_cmp_dispatch = self.optimize_dispatch(
                    perfect_foresight_df,
                    colmap,
                    soc_start=self.soc_init,
                    soc_end_target=None,
                    soc_end_min_target=self.soc_target_end if enforce_final_soc_min else None,
                    deterministic_reserve_settlement=False,
                    allowed_markets=allowed_markets,
                )
        finally:
            self.milp_time_limit_seconds = _orig_tl_cmp
            self.milp_rel_gap = _orig_gap_cmp
        with _phase_watchdog("settlement_perfect_foresight_comparable_realized"):
            perfect_foresight_cmp_real = self.settle_dispatch(
                perfect_foresight_df,
                perfect_foresight_cmp_dispatch,
                colmap,
                predicted_settlement=False,
                apply_market_clearing=True,
                perfect_foresight_mode=True,
            )
        perfect_foresight_cmp_real = perfect_foresight_cmp_real.rename(
            columns={
                c: c.replace("perfect_foresight_", "pf_", 1) if c.startswith("perfect_foresight_") else c.replace("real_", "pf_", 1)
                for c in perfect_foresight_cmp_real.columns
                if c.startswith("real_") or c.startswith("perfect_foresight_")
            }
        )

        global_perfect_foresight_real = pd.DataFrame()
        global_perfect_foresight_validation_status = "disabled_unverified"
        global_perfect_foresight_dispatch_rows = 0.0
        global_perfect_foresight_settlement_rows = 0.0
        global_perfect_foresight_bem_only_included = 0.0
        global_perfect_foresight_pnl_reconciliation_error_eur = float("nan")
        global_perfect_foresight_available_flag = 0.0
        if enable_global_perfect_foresight:
            # Global hindsight perfect_foresight upper bound: single full-horizon optimization
            # over the full evaluation window using true values from t=0.
            _orig_tl_global = float(self.milp_time_limit_seconds)
            _orig_gap_global = float(self.milp_rel_gap)
            self.milp_time_limit_seconds = max(_orig_tl_global, 300.0)
            self.milp_rel_gap = min(_orig_gap_global, 1e-5)
            try:
                global_perfect_foresight_dispatch = self.optimize_dispatch(
                    perfect_foresight_df,
                    colmap,
                    soc_start=self.soc_init,
                    soc_end_target=None,
                    soc_end_min_target=self.soc_target_end if enforce_final_soc_min else None,
                    deterministic_reserve_settlement=True,
                    allowed_markets=allowed_markets,
                )
                # settle_dispatch expects plan_* dispatch columns. Single-shot LP
                # paths may return base columns without plan_ prefixes.
                for src_col, dst_col in [
                    ("charge_mw", "plan_charge_mw"),
                    ("discharge_mw", "plan_discharge_mw"),
                    ("reserve_pos_mw", "plan_reserve_pos_mw"),
                    ("reserve_neg_mw", "plan_reserve_neg_mw"),
                    ("bem_only_pos_mw", "plan_bem_only_pos_mw"),
                    ("bem_only_neg_mw", "plan_bem_only_neg_mw"),
                ]:
                    if dst_col not in global_perfect_foresight_dispatch.columns and src_col in global_perfect_foresight_dispatch.columns:
                        global_perfect_foresight_dispatch[dst_col] = pd.to_numeric(global_perfect_foresight_dispatch[src_col], errors="coerce").fillna(0.0)
                global_perfect_foresight_dispatch_rows = float(len(global_perfect_foresight_dispatch))
                global_perfect_foresight_bem_only_included = float(
                    ("plan_bem_only_pos_mw" in global_perfect_foresight_dispatch.columns)
                    and ("plan_bem_only_neg_mw" in global_perfect_foresight_dispatch.columns)
                )
            finally:
                self.milp_time_limit_seconds = _orig_tl_global
                self.milp_rel_gap = _orig_gap_global
            with _phase_watchdog("settlement_global_perfect_foresight_realized"):
                global_perfect_foresight_real = self.settle_dispatch(
                    perfect_foresight_df,
                    global_perfect_foresight_dispatch,
                    colmap,
                    predicted_settlement=False,
                    apply_market_clearing=True,
                    perfect_foresight_mode=True,
                )
            global_perfect_foresight_settlement_rows = float(len(global_perfect_foresight_real))
            global_perfect_foresight_real = global_perfect_foresight_real.rename(
                columns={
                    c: c.replace("real_", "global_perfect_foresight_", 1)
                    for c in global_perfect_foresight_real.columns
                    if c.startswith("real_")
                }
            )
            global_perfect_foresight_validation_status = "computed_unverified"
            # Conservative availability gate: only mark available when the
            # full-horizon branch produced non-empty dispatch/settlement and
            # horizon lengths match expected scope.
            if global_perfect_foresight_dispatch_rows > 0 and global_perfect_foresight_settlement_rows == float(len(df)):
                global_perfect_foresight_available_flag = 1.0
                global_perfect_foresight_validation_status = "available_scope_validated"
            else:
                global_perfect_foresight_available_flag = 0.0
                global_perfect_foresight_validation_status = "disabled_scope_mismatch"

        # Legacy isolated-market retrospective accounting removed.
        realized_da_only_feasible = True
        perfect_foresight_da_only_feasible = True
        realized_afrr_only_feasible = True
        perfect_foresight_afrr_only_feasible = True
        realized_da_only_total = float("nan")
        perfect_foresight_da_only_total = float("nan")
        realized_afrr_only_total = float("nan")
        perfect_foresight_afrr_only_total = float("nan")
        realized_da_only_diag: dict[str, float] = {}
        perfect_foresight_da_only_diag: dict[str, float] = {}
        realized_afrr_only_diag: dict[str, float] = {}
        perfect_foresight_afrr_only_diag: dict[str, float] = {}
        realized_da_only_hourly = pd.DataFrame()
        perfect_foresight_da_only_hourly = pd.DataFrame()
        realized_afrr_only_hourly = pd.DataFrame()
        perfect_foresight_afrr_only_hourly = pd.DataFrame()

        def _merge_unique(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
            """Merge while dropping overlapping non-key columns from right."""
            if colmap.timestamp not in left.columns or colmap.timestamp not in right.columns:
                raise ValueError(f"_merge_unique requires '{colmap.timestamp}' in both frames.")
            right_cols = [colmap.timestamp] + [
                c for c in right.columns if c != colmap.timestamp and c not in left.columns
            ]
            return self._guarded_merge(
                left,
                right[right_cols],
                on=colmap.timestamp,
                how="left",
                validate_row_count=True,
            )

        hourly = dispatch.copy()
        hourly = _merge_unique(hourly, pred)
        hourly = _merge_unique(hourly, real)
        hourly = _merge_unique(hourly, naive_real)
        hourly = _merge_unique(hourly, perfect_foresight_real)
        hourly = _merge_unique(hourly, perfect_foresight_cmp_real)
        if not global_perfect_foresight_real.empty:
            hourly = _merge_unique(hourly, global_perfect_foresight_real)
        hourly = hourly.sort_values(colmap.timestamp).reset_index(drop=True)
        self._validate_strategy_isolation_outputs(hourly=hourly, allowed_markets=allowed_markets)

        if "real_net_cashflow_eur" in hourly.columns:
            hourly["real_cashflow_eur"] = hourly["real_net_cashflow_eur"]
        else:
            hourly["real_cashflow_eur"] = hourly["real_pnl_eur"] + hourly.get("real_degradation_cost_eur", 0.0)
        if "pred_net_cashflow_eur" in hourly.columns:
            hourly["pred_cashflow_eur"] = hourly["pred_net_cashflow_eur"]
        else:
            hourly["pred_cashflow_eur"] = hourly["pred_pnl_eur"] + hourly.get("pred_degradation_cost_eur", 0.0)
        if "naive_net_cashflow_eur" in hourly.columns:
            hourly["naive_cashflow_eur"] = hourly["naive_net_cashflow_eur"]
        else:
            hourly["naive_cashflow_eur"] = hourly["naive_pnl_eur"] + hourly.get("naive_degradation_cost_eur", 0.0)
        if "perfect_foresight_net_cashflow_eur" in hourly.columns:
            hourly["perfect_foresight_cashflow_eur"] = hourly["perfect_foresight_net_cashflow_eur"]
        else:
            hourly["perfect_foresight_cashflow_eur"] = hourly["perfect_foresight_pnl_eur"] + hourly.get("perfect_foresight_degradation_cost_eur", 0.0)
        hourly["real_cum_cash_eur"] = self.initial_cash + hourly["real_cashflow_eur"].cumsum()
        hourly["pred_cum_cash_eur"] = self.initial_cash + hourly["pred_cashflow_eur"].cumsum()
        hourly["naive_cum_cash_eur"] = self.initial_cash + hourly["naive_cashflow_eur"].cumsum()
        hourly["perfect_foresight_cum_cash_eur"] = self.initial_cash + hourly["perfect_foresight_cashflow_eur"].cumsum()
        # Profit cumulatives (includes non-cash degradation by definition of *_pnl_eur).
        hourly["real_cum_pnl_eur"] = hourly["real_pnl_eur"].cumsum()
        hourly["pred_cum_pnl_eur"] = hourly["pred_pnl_eur"].cumsum()
        hourly["naive_cum_pnl_eur"] = hourly["naive_pnl_eur"].cumsum()
        hourly["perfect_foresight_cum_pnl_eur"] = hourly["perfect_foresight_pnl_eur"].cumsum()
        # User-facing naming alias.
        for old_col, new_col in [
            ("perfect_foresight_pnl_eur", "perfect_foresight_pnl_eur"),
            ("perfect_foresight_penalty_eur", "perfect_foresight_penalty_eur"),
            ("perfect_foresight_act_pos_mwh", "perfect_foresight_act_pos_mwh"),
            ("perfect_foresight_act_neg_mwh", "perfect_foresight_act_neg_mwh"),
        ]:
            if old_col in hourly.columns and new_col not in hourly.columns:
                hourly[new_col] = hourly[old_col]
        hourly["pnl_gap_eur"] = hourly["real_pnl_eur"] - hourly["pred_pnl_eur"]
        hourly["cost_of_forecast_error_eur"] = hourly["perfect_foresight_pnl_eur"] - hourly["real_pnl_eur"]
        # Per-market opportunity-cost proxy (EV-space):
        # Compare DA-only EV vs aFRR-only EV each hour and quantify the gap to the
        # best single-market alternative (non-negative baseline).
        if all(c in hourly.columns for c in ["ev_da_charge_eur", "ev_da_discharge_eur", "ev_afrr_pos_eur", "ev_afrr_neg_eur"]):
            ev_da_total = pd.to_numeric(hourly["ev_da_charge_eur"], errors="coerce").fillna(0.0) + pd.to_numeric(
                hourly["ev_da_discharge_eur"], errors="coerce"
            ).fillna(0.0)
            ev_afrr_total = pd.to_numeric(hourly["ev_afrr_pos_eur"], errors="coerce").fillna(0.0) + pd.to_numeric(
                hourly["ev_afrr_neg_eur"], errors="coerce"
            ).fillna(0.0)
            hourly["ev_da_total_eur"] = ev_da_total
            hourly["ev_afrr_total_eur"] = ev_afrr_total
            hourly["ev_best_single_market_eur"] = np.maximum(np.maximum(ev_da_total, ev_afrr_total), 0.0)
            hourly["ev_opportunity_cost_da_eur"] = hourly["ev_best_single_market_eur"] - ev_da_total
            hourly["ev_opportunity_cost_afrr_eur"] = hourly["ev_best_single_market_eur"] - ev_afrr_total

        # Lightweight merge audit.
        missing_realized_pnl = int(pd.to_numeric(hourly.get("real_pnl_eur", pd.Series(dtype=float)), errors="coerce").isna().sum())
        logging.info(
            "[MERGE_AUDIT] hourly_rows=%d missing_realized_pnl_rows=%d",
            int(len(hourly)),
            missing_realized_pnl,
        )

        # Audit aliases for awarded capacity and delivered activation.
        if "real_afrr_cap_pos_awarded_mw" not in hourly.columns:
            if "real_executed_reserve_pos_mw" in hourly.columns:
                hourly["real_afrr_cap_pos_awarded_mw"] = pd.to_numeric(hourly["real_executed_reserve_pos_mw"], errors="coerce").fillna(0.0)
            elif "real_awarded_capacity_pos_mw" in hourly.columns:
                hourly["real_afrr_cap_pos_awarded_mw"] = pd.to_numeric(hourly["real_awarded_capacity_pos_mw"], errors="coerce").fillna(0.0)
            else:
                hourly["real_afrr_cap_pos_awarded_mw"] = 0.0
        if "real_afrr_cap_neg_awarded_mw" not in hourly.columns:
            if "real_executed_reserve_neg_mw" in hourly.columns:
                hourly["real_afrr_cap_neg_awarded_mw"] = pd.to_numeric(hourly["real_executed_reserve_neg_mw"], errors="coerce").fillna(0.0)
            elif "real_awarded_capacity_neg_mw" in hourly.columns:
                hourly["real_afrr_cap_neg_awarded_mw"] = pd.to_numeric(hourly["real_awarded_capacity_neg_mw"], errors="coerce").fillna(0.0)
            else:
                hourly["real_afrr_cap_neg_awarded_mw"] = 0.0
        if "real_delivered_activation_pos_mwh" not in hourly.columns and "real_act_pos_mwh" in hourly.columns:
            hourly["real_delivered_activation_pos_mwh"] = pd.to_numeric(hourly["real_act_pos_mwh"], errors="coerce").fillna(0.0)
        if "real_delivered_activation_neg_mwh" not in hourly.columns and "real_act_neg_mwh" in hourly.columns:
            hourly["real_delivered_activation_neg_mwh"] = pd.to_numeric(hourly["real_act_neg_mwh"], errors="coerce").fillna(0.0)
        for plain, pref in (
            ("afrr_cap_pos_awarded_mw", "real_afrr_cap_pos_awarded_mw"),
            ("afrr_cap_neg_awarded_mw", "real_afrr_cap_neg_awarded_mw"),
            ("delivered_activation_pos_mwh", "real_delivered_activation_pos_mwh"),
            ("delivered_activation_neg_mwh", "real_delivered_activation_neg_mwh"),
        ):
            if plain not in hourly.columns and pref in hourly.columns:
                hourly[plain] = pd.to_numeric(hourly[pref], errors="coerce").fillna(0.0)

        # Row-level realized PnL reconciliation (explicit components).
        for c in (
            "real_revenue_da_eur",
            "real_cost_da_eur",
            "real_revenue_id_eur",
            "real_cost_id_eur",
            "real_revenue_capacity_eur",
            "real_revenue_activation_eur",
            "real_transaction_cost_eur",
            "real_degradation_cost_eur",
            "real_aux_cost_eur",
            "real_penalty_eur",
            "real_pnl_eur",
        ):
            if c not in hourly.columns:
                hourly[c] = 0.0
            hourly[c] = pd.to_numeric(hourly[c], errors="coerce").fillna(0.0)
        hourly["real_recomputed_pnl_eur"] = (
            hourly["real_revenue_da_eur"]
            - hourly["real_cost_da_eur"]
            + hourly["real_revenue_id_eur"]
            - hourly["real_cost_id_eur"]
            + hourly["real_revenue_capacity_eur"]
            + hourly["real_revenue_activation_eur"]
            - hourly["real_transaction_cost_eur"]
            - hourly["real_degradation_cost_eur"]
            - hourly["real_aux_cost_eur"]
            - hourly["real_penalty_eur"]
        )
        hourly["real_pnl_reconciliation_error_eur"] = (
            hourly["real_pnl_eur"] - hourly["real_recomputed_pnl_eur"]
        ).abs()
        for c in ("real_bcm_linked_activation_revenue_eur", "real_bem_only_activation_revenue_eur", "real_revenue_activation_eur"):
            if c not in hourly.columns:
                hourly[c] = 0.0
            hourly[c] = pd.to_numeric(hourly[c], errors="coerce").fillna(0.0)
        hourly["real_activation_split_reconciliation_error_eur"] = (
            hourly["real_revenue_activation_eur"]
            - (hourly["real_bcm_linked_activation_revenue_eur"] + hourly["real_bem_only_activation_revenue_eur"])
        ).abs()

        min_cash = float(hourly["real_cum_cash_eur"].min()) if not hourly.empty else self.initial_cash
        capital_required = max(0.0, -min_cash)

        monthly = aggregate_periodic(hourly, colmap.timestamp, freq="ME")
        yearly = aggregate_periodic(hourly, colmap.timestamp, freq="YE")
        volatility = calculate_volatility(plan_history)
        naive_volatility = calculate_volatility(naive_plan_history)

        pred_pnl_raw = float(hourly["pred_pnl_eur"].sum())
        real_pnl_raw = float(hourly["real_pnl_eur"].sum())
        naive_pnl_raw = float(hourly["naive_pnl_eur"].sum())
        perfect_foresight_pnl_raw = float(hourly["perfect_foresight_pnl_eur"].sum())
        def _sum_col_zero(col: str) -> float:
            if col not in hourly.columns:
                return 0.0
            return float(pd.to_numeric(hourly[col], errors="coerce").fillna(0.0).sum())
        real_market_revenue_excl_costs_eur = float(
            (
                pd.to_numeric(hourly["real_revenue_da_eur"], errors="coerce").fillna(0.0)
                - pd.to_numeric(hourly["real_cost_da_eur"], errors="coerce").fillna(0.0)
                + pd.to_numeric(hourly["real_revenue_capacity_eur"], errors="coerce").fillna(0.0)
                + pd.to_numeric(hourly["real_revenue_activation_eur"], errors="coerce").fillna(0.0)
                + pd.to_numeric(hourly["real_revenue_id_eur"], errors="coerce").fillna(0.0)
                - pd.to_numeric(hourly["real_cost_id_eur"], errors="coerce").fillna(0.0)
            ).sum()
        )
        real_transaction_cost_eur = _sum_col_zero("real_transaction_cost_eur")
        real_auxiliary_cost_eur = _sum_col_zero("real_aux_cost_eur")
        real_degradation_cost_eur = _sum_col_zero("real_degradation_cost_eur")
        real_penalty_cost_eur = _sum_col_zero("real_penalty_eur")
        # Optimizer-only numerical feasibility penalties (not market cashflows).
        real_numerical_slack_penalty_eur = _sum_col_zero("real_ev_slack_penalty_pos_eur") + _sum_col_zero("real_ev_slack_penalty_neg_eur")

        if not hourly.empty:
            final_pred_soc_mwh = float(hourly["pred_soc_mwh"].iloc[-1]) if "pred_soc_mwh" in hourly.columns else float(self.soc_init)
            final_real_soc_mwh = float(hourly["real_soc_mwh"].iloc[-1]) if "real_soc_mwh" in hourly.columns else float(self.soc_init)
            final_naive_soc_mwh = float(hourly["naive_soc_mwh"].iloc[-1]) if "naive_soc_mwh" in hourly.columns else float(self.soc_init)
            final_perfect_foresight_soc_mwh = float(hourly["perfect_foresight_soc_mwh"].iloc[-1]) if "perfect_foresight_soc_mwh" in hourly.columns else float(self.soc_init)
            final_global_perfect_foresight_soc_mwh = float(hourly["global_perfect_foresight_soc_mwh"].iloc[-1]) if "global_perfect_foresight_soc_mwh" in hourly.columns else float(self.soc_init)
        else:
            final_pred_soc_mwh = float(self.soc_init)
            final_real_soc_mwh = float(self.soc_init)
            final_naive_soc_mwh = float(self.soc_init)
            final_perfect_foresight_soc_mwh = float(self.soc_init)
            final_global_perfect_foresight_soc_mwh = float(self.soc_init)

        da_pred_last = self._finite_numeric_series(
            df,
            colmap.pred_da_price,
            fallback_cols=[colmap.true_da_price],
            default=0.0,
        )
        da_true_last = self._finite_numeric_series(
            df,
            colmap.true_da_price,
            fallback_cols=[colmap.pred_da_price],
            default=0.0,
        )
        terminal_price_pred_eur_mwh = max(
            0.0,
            float(da_pred_last.iloc[-1]) if len(da_pred_last) else 0.0,
        )
        terminal_price_true_eur_mwh = max(
            0.0,
            float(da_true_last.iloc[-1]) if len(da_true_last) else 0.0,
        )

        # Legacy terminal valuation is delta-to-start inventory (not absolute inventory):
        # only energy above start SoC adds value; missing energy is subtracted.
        terminal_delta_pred_mwh = (final_pred_soc_mwh - self.soc_init) * self.eta_out
        terminal_delta_real_mwh = (final_real_soc_mwh - self.soc_init) * self.eta_out
        terminal_delta_naive_mwh = (final_naive_soc_mwh - self.soc_init) * self.eta_out
        terminal_delta_perfect_foresight_mwh = (final_perfect_foresight_soc_mwh - self.soc_init) * self.eta_out
        terminal_delta_global_perfect_foresight_mwh = (final_global_perfect_foresight_soc_mwh - self.soc_init) * self.eta_out

        terminal_value_pred_eur = terminal_delta_pred_mwh * terminal_price_pred_eur_mwh
        terminal_value_real_eur = terminal_delta_real_mwh * terminal_price_true_eur_mwh
        terminal_value_naive_eur = terminal_delta_naive_mwh * terminal_price_true_eur_mwh
        terminal_value_perfect_foresight_eur = terminal_delta_perfect_foresight_mwh * terminal_price_true_eur_mwh
        terminal_value_global_perfect_foresight_eur = terminal_delta_global_perfect_foresight_mwh * terminal_price_true_eur_mwh

        # Same-rules perfect_foresight comparable market PnL (uses model-market gates with perfect foresight inputs).
        def _cmp_sum(df_cmp: pd.DataFrame, col: str) -> float:
            if col not in df_cmp.columns:
                return 0.0
            return float(pd.to_numeric(df_cmp[col], errors="coerce").fillna(0.0).sum())

        perfect_foresight_cmp_pnl_raw = _cmp_sum(perfect_foresight_cmp_real, "pf_pnl_eur") if "pf_pnl_eur" in perfect_foresight_cmp_real.columns else _cmp_sum(perfect_foresight_cmp_real, "real_pnl_eur")
        perfect_foresight_cmp_final_soc_mwh = (
            float(pd.to_numeric(perfect_foresight_cmp_real["pf_soc_mwh"], errors="coerce").dropna().iloc[-1])
            if ("pf_soc_mwh" in perfect_foresight_cmp_real.columns and not perfect_foresight_cmp_real.empty)
            else (
                float(pd.to_numeric(perfect_foresight_cmp_real["real_soc_mwh"], errors="coerce").dropna().iloc[-1])
                if ("real_soc_mwh" in perfect_foresight_cmp_real.columns and not perfect_foresight_cmp_real.empty)
                else float(self.soc_init)
            )
        )
        perfect_foresight_cmp_terminal_delta_mwh = (perfect_foresight_cmp_final_soc_mwh - self.soc_init) * self.eta_out
        perfect_foresight_cmp_terminal_value_eur = perfect_foresight_cmp_terminal_delta_mwh * terminal_price_true_eur_mwh
        perfect_foresight_comparable_total_pnl_eur = float(perfect_foresight_cmp_pnl_raw + perfect_foresight_cmp_terminal_value_eur)

        def _terminal_target_adjustment(soc_final_mwh: float, term_price_eur_mwh: float) -> dict[str, float]:
            target = float(self.soc_target_end)
            shortfall_mwh = max(0.0, target - float(soc_final_mwh))
            surplus_mwh = max(0.0, float(soc_final_mwh) - target)
            repair_cost_eur = (shortfall_mwh / max(self.eta_in, 1e-12)) * float(term_price_eur_mwh)
            liquidation_revenue_eur = (surplus_mwh * self.eta_out) * float(term_price_eur_mwh)
            return {
                "shortfall_mwh": float(shortfall_mwh),
                "surplus_mwh": float(surplus_mwh),
                "repair_cost_eur": float(repair_cost_eur),
                "liquidation_revenue_eur": float(liquidation_revenue_eur),
                "adjustment_eur": float(liquidation_revenue_eur - repair_cost_eur),
            }

        term_adj_pred = _terminal_target_adjustment(final_pred_soc_mwh, terminal_price_pred_eur_mwh)
        term_adj_real = _terminal_target_adjustment(final_real_soc_mwh, terminal_price_true_eur_mwh)
        term_adj_naive = _terminal_target_adjustment(final_naive_soc_mwh, terminal_price_true_eur_mwh)
        term_adj_perfect_foresight = _terminal_target_adjustment(final_perfect_foresight_soc_mwh, terminal_price_true_eur_mwh)
        term_adj_global_perfect_foresight = _terminal_target_adjustment(final_global_perfect_foresight_soc_mwh, terminal_price_true_eur_mwh)

        pred_pnl_total = pred_pnl_raw + term_adj_pred["adjustment_eur"]
        real_pnl_total = real_pnl_raw + term_adj_real["adjustment_eur"]
        naive_pnl_total = naive_pnl_raw + term_adj_naive["adjustment_eur"]
        perfect_foresight_pnl_total = perfect_foresight_pnl_raw + term_adj_perfect_foresight["adjustment_eur"]
        global_perfect_foresight_pnl_raw = _sum_col_zero("global_perfect_foresight_pnl_eur") if enable_global_perfect_foresight else float("nan")
        global_perfect_foresight_pnl_total = (
            global_perfect_foresight_pnl_raw + term_adj_global_perfect_foresight["adjustment_eur"]
            if enable_global_perfect_foresight else float("nan")
        )

        # Rolling perfect-foresight is diagnostic only and can be beaten.


        if np.isfinite(perfect_foresight_pnl_total) and abs(perfect_foresight_pnl_total) > 1e-9:
            opportunity_gap_ratio = float((perfect_foresight_pnl_total - real_pnl_total) / perfect_foresight_pnl_total)
        else:
            opportunity_gap_ratio = float("nan")
        summary = {
            "rows": float(len(hourly)),
            "input_row_count": float(len(df)),
            "optimized_hour_count": float(len(hourly)),
            "first_input_timestamp_utc": (
                pd.to_datetime(df[colmap.timestamp], utc=True, errors="coerce").dropna().min().isoformat()
                if (colmap.timestamp in df.columns and not df.empty)
                else ""
            ),
            "first_optimized_target_timestamp_utc": (
                pd.to_datetime(hourly[colmap.timestamp], utc=True, errors="coerce").dropna().min().isoformat()
                if (colmap.timestamp in hourly.columns and not hourly.empty)
                else ""
            ),
            "dropped_initial_rows_due_to_forecast_target_alignment": float(max(0, len(df) - len(hourly))),
            "planned_total_pnl_eur": float(pred_pnl_total),
            "predicted_total_pnl_eur": float(pred_pnl_total),
            "realized_total_pnl_eur": float(real_pnl_total),
            "naive_total_pnl_eur": float(naive_pnl_total),
            "rolling_perfect_foresight_same_rules_total_pnl_eur": float(perfect_foresight_pnl_total),
            "comparable_rolling_perfect_foresight_same_rules_market_pnl_eur": float(perfect_foresight_comparable_total_pnl_eur),
            "rolling_perfect_foresight_same_rules_is_global_upper_bound": 0.0,
            "rolling_perfect_foresight_same_rules_can_be_beaten": 1.0,
            "benchmark_type": "rolling_perfect_foresight_same_rules",
            "rolling_pf_quantile_surface_mode": "collapsed_to_truth",
            "perfect_foresight_total_pnl_eur": float(perfect_foresight_pnl_total),
            "perfect_foresight_total_pnl_eur": float(perfect_foresight_pnl_total),
            "perfect_foresight_total_pnl_eur_is_deprecated": 1.0,
            "perfect_foresight_total_pnl_eur_semantics": "rolling_perfect_foresight_same_rules",
            "perfect_foresight_total_pnl_eur_is_deprecated": 1.0,
            "perfect_foresight_total_pnl_eur_semantics": "rolling_perfect_foresight_same_rules",
            "perfect_foresight_same_rules_total_pnl_eur": float(perfect_foresight_comparable_total_pnl_eur),
            "perfect_foresight_comparable_total_pnl_eur": float(perfect_foresight_comparable_total_pnl_eur),
            "comparable_perfect_foresight_market_pnl_eur": float(perfect_foresight_comparable_total_pnl_eur),
            "perfect_foresight_is_global_upper_bound": 0.0,
            "perfect_foresight_can_be_beaten": 1.0,
            "benchmark_is_global_upper_bound": 0.0,
            "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur": float(global_perfect_foresight_pnl_total) if np.isfinite(global_perfect_foresight_pnl_total) else float("nan"),
            "global_hindsight_perfect_foresight_upper_bound_market_pnl_eur": float(global_perfect_foresight_pnl_total) if np.isfinite(global_perfect_foresight_pnl_total) else float("nan"),
            "global_hindsight_perfect_foresight_is_global_upper_bound": 0.0,
            "global_perfect_foresight_available": float(global_perfect_foresight_available_flag),
            "global_perfect_foresight_capacity_bid_semantics": "hindsight_pay_as_bid_upper_bound",
            "benchmark_is_global_upper_bound_global_perfect_foresight": 1.0,
            "predicted_pnl_excl_terminal_eur": float(pred_pnl_raw),
            "realized_pnl_excl_terminal_eur": float(real_pnl_raw),
            "naive_pnl_excl_terminal_eur": float(naive_pnl_raw),
            "perfect_foresight_pnl_excl_terminal_eur": float(perfect_foresight_pnl_raw),
            "terminal_value_predicted_eur": float(terminal_value_pred_eur),
            "terminal_value_realized_eur": float(terminal_value_real_eur),
            "terminal_value_naive_eur": float(terminal_value_naive_eur),
            "terminal_value_perfect_foresight_eur": float(terminal_value_perfect_foresight_eur),
            "terminal_value_global_perfect_foresight_eur": float(terminal_value_global_perfect_foresight_eur) if enable_global_perfect_foresight else float("nan"),
            "terminal_soc_shortfall_mwh": float(term_adj_real["shortfall_mwh"]),
            "terminal_soc_surplus_mwh": float(term_adj_real["surplus_mwh"]),
            "terminal_soc_repair_cost_eur": float(term_adj_real["repair_cost_eur"]),
            "terminal_soc_liquidation_revenue_eur": float(term_adj_real["liquidation_revenue_eur"]),
            "terminal_soc_adjustment_eur": float(term_adj_real["adjustment_eur"]),
            "terminal_soc_net_adjustment_eur": float(term_adj_real["adjustment_eur"]),
            "terminal_soc_price_eur_mwh": float(terminal_price_true_eur_mwh),
            "terminal_price_eur_mwh": float(terminal_price_true_eur_mwh),
            "terminal_price_source": "last_true_da_price",
            "terminal_delta_predicted_mwh": float(terminal_delta_pred_mwh),
            "terminal_delta_realized_mwh": float(terminal_delta_real_mwh),
            "terminal_delta_naive_mwh": float(terminal_delta_naive_mwh),
            "terminal_delta_perfect_foresight_mwh": float(terminal_delta_perfect_foresight_mwh),
            "terminal_price_predicted_eur_mwh": float(terminal_price_pred_eur_mwh),
            "terminal_price_realized_eur_mwh": float(terminal_price_true_eur_mwh),
            "realized_total_penalty_eur": float(hourly["real_penalty_eur"].sum()) if "real_penalty_eur" in hourly.columns else 0.0,
            "naive_total_penalty_eur": float(hourly["naive_penalty_eur"].sum()) if "naive_penalty_eur" in hourly.columns else 0.0,
            "perfect_foresight_total_penalty_eur": float(hourly["perfect_foresight_penalty_eur"].sum()) if "perfect_foresight_penalty_eur" in hourly.columns else 0.0,
            "pnl_gap_total_eur": float(real_pnl_total - pred_pnl_total),
            "economic_opportunity_gap_ratio": opportunity_gap_ratio,
            "cost_of_forecast_error_total_eur": float(perfect_foresight_pnl_total - real_pnl_total),
            "objective_value_eur": float(
                pd.to_numeric(hourly["predicted_objective_eur"], errors="coerce").fillna(0.0).sum()
            ) if "predicted_objective_eur" in hourly.columns else float(pred_pnl_total),
            "market_revenue_eur": float(real_market_revenue_excl_costs_eur),
            "degradation_cost_eur": float(real_degradation_cost_eur),
            "auxiliary_cost_eur": float(real_auxiliary_cost_eur),
            "transaction_cost_eur": float(real_transaction_cost_eur),
            "penalty_cost_eur": float(real_penalty_cost_eur),
            "numerical_slack_penalty_eur": float(real_numerical_slack_penalty_eur),
            # Revenue-stream aliases for downstream thesis/report consumers.
            "capacity_revenue_eur": float(pd.to_numeric(hourly.get("real_revenue_capacity_eur", 0.0), errors="coerce").sum()) if "real_revenue_capacity_eur" in hourly.columns else 0.0,
            "activation_revenue_eur": float(pd.to_numeric(hourly.get("real_revenue_activation_eur", 0.0), errors="coerce").sum()) if "real_revenue_activation_eur" in hourly.columns else 0.0,
            "da_revenue_eur": float(pd.to_numeric(hourly.get("real_revenue_da_eur", 0.0), errors="coerce").sum()) if "real_revenue_da_eur" in hourly.columns else 0.0,
            "max_capital_required_eur": float(capital_required),
            "max_drawdown_eur": float(capital_required),
            "max_hourly_cash_outflow_eur": float(max(0.0, -hourly["real_cashflow_eur"].min())),
            "avg_reserve_pos_mw": float(pd.to_numeric(hourly["real_reserve_pos_mw"], errors="coerce").mean()) if "real_reserve_pos_mw" in hourly.columns else 0.0,
            "avg_reserve_neg_mw": float(pd.to_numeric(hourly["real_reserve_neg_mw"], errors="coerce").mean()) if "real_reserve_neg_mw" in hourly.columns else 0.0,
            "avg_charge_mw": float(pd.to_numeric(hourly["real_charge_mw"], errors="coerce").mean()) if "real_charge_mw" in hourly.columns else 0.0,
            "avg_discharge_mw": float(pd.to_numeric(hourly["real_discharge_mw"], errors="coerce").mean()) if "real_discharge_mw" in hourly.columns else 0.0,
            "realized_da_only_feasible": float(realized_da_only_feasible),
            "perfect_foresight_da_only_feasible": float(perfect_foresight_da_only_feasible),
            "realized_afrr_only_feasible": float(realized_afrr_only_feasible),
            "perfect_foresight_afrr_only_feasible": float(perfect_foresight_afrr_only_feasible),
            "bem_only_mode": "approx_reuse_reserve_volume",
            "id_price_taker": 1.0,
            "id_price_mode": "synthetic_da_spread_price_taker",
            "id_spread": float(self.id_rescue_spread_eur_mwh),
            "id_price_cap": float(self.id_buy_price_cap_eur_mwh),
            "id_price_floor": float(self.id_sell_price_floor_eur_mwh),
            "strategy": ",".join(sorted({str(m).strip().lower() for m in allowed_markets})),
            "id_recourse_mode": str(getattr(self, "_id_recourse_mode", "common")),
            "id_mode": str(self._strategy_permissions.id_mode),
            "id_economic_enabled": float(self._strategy_permissions.allow_id_economic),
            "id_technical_repair_enabled": float(self._strategy_permissions.allow_id_technical_repair),
            "allow_da": float(self._strategy_permissions.allow_da),
            "allow_id": float(self._strategy_permissions.allow_id),
            "allow_bcm": float(self._strategy_permissions.allow_bcm),
            "allow_bcm_activation_obligations": float(self._strategy_permissions.allow_bcm_activation_obligations),
            "allow_bem_only": float(self._strategy_permissions.allow_bem_only),
            "reserve_activation_headroom_h": float(self.reserve_activation_headroom_h),
            "bem_activation_headroom_h": float(self.bem_activation_headroom_h),
        }
        accepted_dumps, candidate_dumps = self._classify_infeasible_debug_dumps(
            self._infeasible_debug_dumps,
            hourly,
            timestamp_col=colmap.timestamp,
        )
        summary["infeasible_debug_dump_count"] = float(len(self._infeasible_debug_dumps))
        summary["accepted_path_infeasible_debug_dump_count"] = float(len(accepted_dumps))
        summary["candidate_infeasible_debug_dump_count"] = float(len(candidate_dumps))
        summary["infeasible_debug_dump_paths"] = [
            d.get("path", "") for d in self._infeasible_debug_dumps
        ]
        summary["infeasible_debug_dump_timestamps"] = [
            d.get("timestamp_utc", "") for d in self._infeasible_debug_dumps
        ]
        accepted_dump_timestamps = [
            d.get("timestamp_utc", "") for d in accepted_dumps if str(d.get("timestamp_utc", "")).strip()
        ]
        if accepted_dump_timestamps:
            first_inf_ts = pd.to_datetime(accepted_dump_timestamps, utc=True, errors="coerce").dropna()
            summary["first_infeasible_timestamp_utc"] = (
                first_inf_ts.min().isoformat() if len(first_inf_ts) else ""
            )
        else:
            summary["first_infeasible_timestamp_utc"] = ""
        # Optimizer data-quality and fallback diagnostics.
        if "optimizer_fallback_used" in hourly.columns:
            fb = pd.to_numeric(hourly["optimizer_fallback_used"], errors="coerce").fillna(0.0)
            summary["optimizer_fallback_hours"] = float((fb > 0.0).sum())
            summary["optimizer_fallback_share_pct"] = float(100.0 * (fb > 0.0).mean())
        else:
            summary["optimizer_fallback_hours"] = 0.0
            summary["optimizer_fallback_share_pct"] = 0.0
        if "optimizer_required_input_imputed_any" in hourly.columns:
            imp_any = pd.to_numeric(hourly["optimizer_required_input_imputed_any"], errors="coerce").fillna(0.0)
            summary["optimizer_input_imputed_hours"] = float((imp_any > 0.0).sum())
            summary["optimizer_input_imputed_share_pct"] = float(100.0 * (imp_any > 0.0).mean())
        else:
            summary["optimizer_input_imputed_hours"] = 0.0
            summary["optimizer_input_imputed_share_pct"] = 0.0
        if "optimizer_required_input_imputed_count" in hourly.columns:
            imp_cnt = pd.to_numeric(hourly["optimizer_required_input_imputed_count"], errors="coerce").fillna(0.0)
            summary["optimizer_input_imputed_total_count"] = float(imp_cnt.sum())
            summary["optimizer_input_imputed_avg_count_per_hour"] = float(imp_cnt.mean())
        else:
            summary["optimizer_input_imputed_total_count"] = 0.0
            summary["optimizer_input_imputed_avg_count_per_hour"] = 0.0
        if "ev_pacc_pos_fallback_used" in hourly.columns:
            summary["pacc_pos_fallback_share_pct"] = float(
                100.0
                * pd.to_numeric(hourly["ev_pacc_pos_fallback_used"], errors="coerce")
                .fillna(0.0)
                .clip(0.0, 1.0)
                .mean()
            )
        else:
            summary["pacc_pos_fallback_share_pct"] = 0.0
        if "ev_pacc_neg_fallback_used" in hourly.columns:
            summary["pacc_neg_fallback_share_pct"] = float(
                100.0
                * pd.to_numeric(hourly["ev_pacc_neg_fallback_used"], errors="coerce")
                .fillna(0.0)
                .clip(0.0, 1.0)
                .mean()
            )
        else:
            summary["pacc_neg_fallback_share_pct"] = 0.0
        summary["stacking_value_realized_eur"] = float("nan")
        summary["stacking_value_perfect_foresight_eur"] = float("nan")
        for prefix, diag in (
            ("realized_da_only", realized_da_only_diag),
            ("perfect_foresight_da_only", perfect_foresight_da_only_diag),
            ("realized_afrr_only", realized_afrr_only_diag),
            ("perfect_foresight_afrr_only", perfect_foresight_afrr_only_diag),
        ):
            for k, v in diag.items():
                summary[f"{prefix}_{k}"] = float(v)
        def _safe_ratio(num: float, den: float) -> float:
            return float(num / den) if abs(float(den)) > 1e-12 else float("nan")
        summary["realized_vs_perfect_foresight_ratio_multi_market"] = _safe_ratio(
            float(summary["realized_total_pnl_eur"]),
            float(summary["rolling_perfect_foresight_same_rules_total_pnl_eur"]),
        )
        # EV sanity on "optimized-only" hours (exclude fallback and imputed-input hours).
        optimized_mask = pd.Series(True, index=hourly.index)
        if "optimizer_fallback_used" in hourly.columns:
            optimized_mask &= pd.to_numeric(hourly["optimizer_fallback_used"], errors="coerce").fillna(0.0).eq(0.0)
        if "optimizer_required_input_imputed_any" in hourly.columns:
            optimized_mask &= pd.to_numeric(hourly["optimizer_required_input_imputed_any"], errors="coerce").fillna(0.0).eq(0.0)
        summary["optimized_hours_only_rows"] = float(int(optimized_mask.sum()))
        for ev_col in (
            "ev_da_charge_eur",
            "ev_da_discharge_eur",
            "ev_afrr_pos_eur",
            "ev_afrr_neg_eur",
            "ev_objective_rebuild_eur",
        ):
            if ev_col in hourly.columns and int(optimized_mask.sum()) > 0:
                vals = pd.to_numeric(hourly.loc[optimized_mask, ev_col], errors="coerce")
                summary[f"optimized_hours_only_{ev_col}_mean"] = float(vals.mean())
            elif ev_col in hourly.columns:
                summary[f"optimized_hours_only_{ev_col}_mean"] = float("nan")
        # Economic building blocks for thesis-grade PnL decomposition.
        summary["total_da_revenue_eur"] = float(hourly["real_revenue_da_eur"].sum())
        summary["total_da_cost_eur"] = float(hourly["real_cost_da_eur"].sum())
        summary["total_id_revenue_eur"] = float(hourly["real_revenue_id_eur"].sum()) if "real_revenue_id_eur" in hourly.columns else 0.0
        summary["total_id_cost_eur"] = float(hourly["real_cost_id_eur"].sum()) if "real_cost_id_eur" in hourly.columns else 0.0
        summary["total_id_pnl_eur"] = float(summary["total_id_revenue_eur"] - summary["total_id_cost_eur"])
        summary["total_id_buy_mwh"] = float(pd.to_numeric(hourly.get("real_id_buy_mwh", 0.0), errors="coerce").fillna(0.0).sum())
        summary["total_id_sell_mwh"] = float(pd.to_numeric(hourly.get("real_id_sell_mwh", 0.0), errors="coerce").fillna(0.0).sum())
        summary["total_id_net_mwh"] = float(summary["total_id_sell_mwh"] - summary["total_id_buy_mwh"])
        summary["id_repair_mwh_total"] = float(
            pd.to_numeric(hourly.get("real_id_repair_mwh", 0.0), errors="coerce").fillna(0.0).sum()
        )
        summary["id_repair_cost_eur_total"] = float(
            pd.to_numeric(hourly.get("real_id_repair_cost_eur", 0.0), errors="coerce").fillna(0.0).sum()
        )
        summary["id_economic_mwh_total"] = float(
            pd.to_numeric(hourly.get("real_id_economic_mwh", 0.0), errors="coerce").fillna(0.0).sum()
        )
        summary["id_economic_pnl_eur_total"] = float(
            pd.to_numeric(hourly.get("real_id_economic_pnl_eur", 0.0), errors="coerce").fillna(0.0).sum()
        )
        summary["id_technical_repair_pnl_eur_total"] = float(
            pd.to_numeric(hourly.get("real_id_technical_repair_pnl_eur", 0.0), errors="coerce").fillna(0.0).sum()
        )
        if "real_id_recourse_reason" in hourly.columns:
            reason_counts = hourly["real_id_recourse_reason"].fillna("none").astype(str).str.strip().str.lower().value_counts().to_dict()
        else:
            reason_counts = {}
        summary["id_recourse_events_total"] = float(
            (
                pd.to_numeric(hourly.get("real_id_buy_mwh", 0.0), errors="coerce").fillna(0.0).abs()
                + pd.to_numeric(hourly.get("real_id_sell_mwh", 0.0), errors="coerce").fillna(0.0).abs()
            ).gt(1e-12).sum()
        )
        summary["id_recourse_events_by_reason"] = json.dumps(reason_counts, ensure_ascii=False, sort_keys=True)
        summary["id_recourse_share_hours"] = float(summary["id_recourse_events_total"] / max(len(hourly), 1))
        summary["realized_total_pnl_excluding_id_eur"] = float(summary["realized_total_pnl_eur"] - summary["total_id_pnl_eur"])
        summary["total_afrr_capacity_revenue_eur"] = float(hourly["real_revenue_capacity_eur"].sum())
        summary["total_afrr_activation_revenue_eur"] = float(hourly["real_revenue_activation_eur"].sum())
        summary["total_bcm_linked_activation_revenue_eur"] = float(
            pd.to_numeric(hourly.get("real_bcm_linked_activation_revenue_eur", 0.0), errors="coerce").fillna(0.0).sum()
        )
        summary["total_bem_only_activation_revenue_eur"] = float(
            pd.to_numeric(hourly.get("real_bem_only_activation_revenue_eur", 0.0), errors="coerce").fillna(0.0).sum()
        )
        summary["total_degradation_cost_eur"] = float(hourly["real_degradation_cost_eur"].sum())
        summary["total_transaction_cost_eur"] = float(hourly["real_transaction_cost_eur"].sum())
        summary["total_auxiliary_cost_eur"] = float(hourly["real_aux_cost_eur"].sum()) if "real_aux_cost_eur" in hourly.columns else 0.0
        summary["total_penalty_cost_eur"] = float(hourly["real_penalty_eur"].sum()) if "real_penalty_eur" in hourly.columns else 0.0
        summary["total_penalty_capacity_pos_eur"] = float(hourly["real_penalty_capacity_pos_eur"].sum()) if "real_penalty_capacity_pos_eur" in hourly.columns else 0.0
        summary["total_penalty_capacity_neg_eur"] = float(hourly["real_penalty_capacity_neg_eur"].sum()) if "real_penalty_capacity_neg_eur" in hourly.columns else 0.0
        summary["total_penalty_capacity_eur"] = float(hourly["real_penalty_capacity_eur"].sum()) if "real_penalty_capacity_eur" in hourly.columns else 0.0
        summary["total_penalty_activation_pos_eur"] = float(hourly["real_penalty_activation_pos_eur"].sum()) if "real_penalty_activation_pos_eur" in hourly.columns else 0.0
        summary["total_penalty_activation_neg_eur"] = float(hourly["real_penalty_activation_neg_eur"].sum()) if "real_penalty_activation_neg_eur" in hourly.columns else 0.0
        summary["total_penalty_activation_eur"] = float(hourly["real_penalty_activation_eur"].sum()) if "real_penalty_activation_eur" in hourly.columns else 0.0
        summary["total_missed_capacity_pos_mw"] = float(hourly["real_missed_capacity_pos_mw"].sum()) if "real_missed_capacity_pos_mw" in hourly.columns else 0.0
        summary["total_missed_capacity_neg_mw"] = float(hourly["real_missed_capacity_neg_mw"].sum()) if "real_missed_capacity_neg_mw" in hourly.columns else 0.0
        summary["total_missed_capacity_mw"] = float(hourly["real_missed_capacity_mw"].sum()) if "real_missed_capacity_mw" in hourly.columns else 0.0
        # Soft-constraint diagnostics from MILP (risk-taking behavior).
        summary["total_slack_pos_mw"] = float(hourly["slack_pos_mw"].sum()) if "slack_pos_mw" in hourly.columns else 0.0
        summary["total_slack_neg_mw"] = float(hourly["slack_neg_mw"].sum()) if "slack_neg_mw" in hourly.columns else 0.0
        summary["mean_slack_pos_mw"] = float(hourly["slack_pos_mw"].mean()) if "slack_pos_mw" in hourly.columns else 0.0
        summary["mean_slack_neg_mw"] = float(hourly["slack_neg_mw"].mean()) if "slack_neg_mw" in hourly.columns else 0.0
        summary["total_slack_soc_min_mwh"] = float(hourly["slack_soc_min_mwh"].sum()) if "slack_soc_min_mwh" in hourly.columns else 0.0
        summary["total_slack_soc_max_mwh"] = float(hourly["slack_soc_max_mwh"].sum()) if "slack_soc_max_mwh" in hourly.columns else 0.0

        # Backward-compatible aliases used by older reports/scripts.
        summary["total_da_energy_revenue_eur"] = summary["total_da_revenue_eur"]
        summary["total_da_energy_cost_eur"] = summary["total_da_cost_eur"]

        pnl_components_rhs = (
            summary["total_da_revenue_eur"]
            + summary["total_id_revenue_eur"]
            + summary["total_afrr_capacity_revenue_eur"]
            + summary["total_afrr_activation_revenue_eur"]
            - summary["total_da_cost_eur"]
            - summary["total_id_cost_eur"]
            - summary["total_degradation_cost_eur"]
            - summary["total_transaction_cost_eur"]
            - summary["total_auxiliary_cost_eur"]
            - summary["total_penalty_cost_eur"]
        )
        summary["realized_pnl_from_components_eur"] = float(pnl_components_rhs)
        summary["realized_pnl_from_components_plus_terminal_eur"] = float(
            pnl_components_rhs + summary["terminal_soc_adjustment_eur"]
        )
        summary["realized_pnl_balance_error_eur"] = float(
            summary["realized_total_pnl_eur"] - summary["realized_pnl_from_components_plus_terminal_eur"]
        )
        summary["realized_pnl_balance_ok"] = float(abs(summary["realized_pnl_balance_error_eur"]) <= 1e-6)
        summary["pnl_reconciliation_error_max_eur"] = float(
            pd.to_numeric(hourly.get("real_pnl_reconciliation_error_eur", 0.0), errors="coerce").fillna(0.0).max()
        )
        summary["activation_split_reconciliation_error_max"] = float(
            pd.to_numeric(hourly.get("real_activation_split_reconciliation_error_eur", 0.0), errors="coerce").fillna(0.0).max()
        )

        # Oracle component decomposition / balance check (same settlement accounting identity).
        summary["perfect_foresight_total_da_revenue_eur"] = float(hourly["perfect_foresight_revenue_da_eur"].sum())
        summary["perfect_foresight_total_da_cost_eur"] = float(hourly["perfect_foresight_cost_da_eur"].sum())
        summary["perfect_foresight_total_afrr_capacity_revenue_eur"] = float(hourly["perfect_foresight_revenue_capacity_eur"].sum())
        summary["perfect_foresight_total_afrr_activation_revenue_eur"] = float(hourly["perfect_foresight_revenue_activation_eur"].sum())
        summary["perfect_foresight_total_bcm_linked_activation_revenue_eur"] = float(
            pd.to_numeric(hourly.get("perfect_foresight_bcm_linked_activation_revenue_eur", 0.0), errors="coerce").fillna(0.0).sum()
        )
        summary["perfect_foresight_total_bem_only_activation_revenue_eur"] = float(
            pd.to_numeric(hourly.get("perfect_foresight_bem_only_activation_revenue_eur", 0.0), errors="coerce").fillna(0.0).sum()
        )
        summary["perfect_foresight_total_id_revenue_eur"] = float(hourly["perfect_foresight_revenue_id_eur"].sum()) if "perfect_foresight_revenue_id_eur" in hourly.columns else 0.0
        summary["perfect_foresight_total_id_cost_eur"] = float(hourly["perfect_foresight_cost_id_eur"].sum()) if "perfect_foresight_cost_id_eur" in hourly.columns else 0.0
        summary["perfect_foresight_total_degradation_cost_eur"] = float(hourly["perfect_foresight_degradation_cost_eur"].sum())
        summary["perfect_foresight_total_transaction_cost_eur"] = float(hourly["perfect_foresight_transaction_cost_eur"].sum())
        summary["perfect_foresight_total_auxiliary_cost_eur"] = float(hourly["perfect_foresight_aux_cost_eur"].sum()) if "perfect_foresight_aux_cost_eur" in hourly.columns else 0.0
        summary["perfect_foresight_total_penalty_cost_eur"] = float(hourly["perfect_foresight_penalty_eur"].sum()) if "perfect_foresight_penalty_eur" in hourly.columns else 0.0
        perfect_foresight_pnl_components_rhs = (
            summary["perfect_foresight_total_da_revenue_eur"]
            + summary["perfect_foresight_total_id_revenue_eur"]
            + summary["perfect_foresight_total_afrr_capacity_revenue_eur"]
            + summary["perfect_foresight_total_afrr_activation_revenue_eur"]
            - summary["perfect_foresight_total_da_cost_eur"]
            - summary["perfect_foresight_total_id_cost_eur"]
            - summary["perfect_foresight_total_degradation_cost_eur"]
            - summary["perfect_foresight_total_transaction_cost_eur"]
            - summary["perfect_foresight_total_auxiliary_cost_eur"]
            - summary["perfect_foresight_total_penalty_cost_eur"]
        )
        summary["perfect_foresight_pnl_from_components_eur"] = float(perfect_foresight_pnl_components_rhs)
        summary["perfect_foresight_pnl_from_components_plus_terminal_eur"] = float(
            perfect_foresight_pnl_components_rhs + float(term_adj_perfect_foresight["adjustment_eur"])
        )
        summary["perfect_foresight_pnl_balance_error_eur"] = float(
            summary["perfect_foresight_total_pnl_eur"] - summary["perfect_foresight_pnl_from_components_plus_terminal_eur"]
        )
        summary["perfect_foresight_pnl_balance_ok"] = float(abs(summary["perfect_foresight_pnl_balance_error_eur"]) <= 1e-6)
        summary["comparable_realized_market_pnl_eur"] = float(summary["realized_pnl_from_components_plus_terminal_eur"])
        # Comparable perfect_foresight benchmark should use same market access semantics as realized.
        summary["comparable_perfect_foresight_market_pnl_eur"] = float(summary["perfect_foresight_same_rules_total_pnl_eur"])
        # Backward-compatible alias (name retained for historical consumers).
        summary["comparable_perfect_foresight_market_pnl_eur"] = float(summary["comparable_perfect_foresight_market_pnl_eur"])
        summary["comparable_perfect_foresight_market_pnl_eur"] = float(
            summary["comparable_perfect_foresight_market_pnl_eur"]
        )
        summary["comparable_benchmark_type"] = "rolling_perfect_foresight_same_rules"
        summary["rolling_pf_is_upper_bound"] = 0.0
        summary["global_perfect_foresight_is_upper_bound"] = 0.0
        summary["global_perfect_foresight_validation_status"] = str(global_perfect_foresight_validation_status)
        summary["global_perfect_foresight_dispatch_rows"] = float(global_perfect_foresight_dispatch_rows)
        summary["global_perfect_foresight_settlement_rows"] = float(global_perfect_foresight_settlement_rows)
        summary["global_perfect_foresight_bem_only_included"] = float(global_perfect_foresight_bem_only_included)
        summary["global_perfect_foresight_market_scope"] = json.dumps(sorted({str(m).strip().lower() for m in allowed_markets}))
        summary["global_perfect_foresight_terminal_adjustment_eur"] = float(term_adj_global_perfect_foresight["adjustment_eur"]) if enable_global_perfect_foresight else float("nan")

        # Per-hour comparable path diagnostics (realized minus perfect-foresight benchmark).
        if "real_pnl_eur" in hourly.columns and "pf_pnl_eur" in hourly.columns:
            hourly["cmp_delta_pnl_eur"] = pd.to_numeric(hourly["real_pnl_eur"], errors="coerce").fillna(0.0) - pd.to_numeric(
                hourly["pf_pnl_eur"], errors="coerce"
            ).fillna(0.0)
            for comp in [
                "revenue_da_eur",
                "revenue_capacity_eur",
                "bcm_linked_activation_revenue_eur",
                "bem_only_activation_revenue_eur",
                "revenue_activation_eur",
                "degradation_cost_eur",
                "aux_cost_eur",
                "penalty_eur",
            ]:
                rc = f"real_{comp}"
                pc = f"pf_{comp}"
                if rc in hourly.columns and pc in hourly.columns:
                    hourly[f"cmp_delta_{comp}"] = pd.to_numeric(hourly[rc], errors="coerce").fillna(0.0) - pd.to_numeric(
                        hourly[pc], errors="coerce"
                    ).fillna(0.0)
            top_idx = (
                hourly["cmp_delta_pnl_eur"].nlargest(10).index
                if "cmp_delta_pnl_eur" in hourly.columns
                else pd.Index([])
            )
            summary["cmp_top10_positive_hourly_delta_pnl_eur_sum"] = float(
                pd.to_numeric(hourly.loc[top_idx, "cmp_delta_pnl_eur"], errors="coerce").fillna(0.0).sum()
            )
            summary["cmp_max_hourly_delta_pnl_eur"] = float(
                pd.to_numeric(hourly["cmp_delta_pnl_eur"], errors="coerce").fillna(0.0).max()
            )
        else:
            summary["cmp_top10_positive_hourly_delta_pnl_eur_sum"] = 0.0
            summary["cmp_max_hourly_delta_pnl_eur"] = 0.0

        # Operational KPIs
        total_grid_discharge_mwh = float(hourly["real_da_sell_mwh"].sum()) if "real_da_sell_mwh" in hourly.columns else 0.0
        total_grid_discharge_mwh += float(hourly["real_id_sell_mwh"].sum()) if "real_id_sell_mwh" in hourly.columns else 0.0
        total_grid_discharge_mwh += float(hourly["real_act_pos_mwh"].sum()) if "real_act_pos_mwh" in hourly.columns else 0.0
        total_internal_discharge_mwh = total_grid_discharge_mwh / max(self.eta_out, 1e-12)
        summary["total_equivalent_full_cycles"] = float(
            total_internal_discharge_mwh / max(self.cap_mwh, 1e-12)
        )
        submitted_afrr_mw = 0.0
        awarded_afrr_mw = 0.0
        if "real_submitted_afrr_pos_mw" in hourly.columns:
            submitted_afrr_mw += float(hourly["real_submitted_afrr_pos_mw"].sum())
        if "real_submitted_afrr_neg_mw" in hourly.columns:
            submitted_afrr_mw += float(hourly["real_submitted_afrr_neg_mw"].sum())
        if "real_executed_reserve_pos_mw" in hourly.columns:
            awarded_afrr_mw += float(hourly["real_executed_reserve_pos_mw"].sum())
        if "real_executed_reserve_neg_mw" in hourly.columns:
            awarded_afrr_mw += float(hourly["real_executed_reserve_neg_mw"].sum())
        summary["afrr_capacity_award_rate"] = float(awarded_afrr_mw / submitted_afrr_mw) if submitted_afrr_mw > 1e-12 else float("nan")
        summary["total_missed_activation_mwh"] = float(hourly["real_missed_activation_mwh"].sum()) if "real_missed_activation_mwh" in hourly.columns else 0.0
        summary["missed_activation_pos_mwh"] = float(hourly["real_missed_activation_pos_mwh"].sum()) if "real_missed_activation_pos_mwh" in hourly.columns else 0.0
        summary["missed_activation_neg_mwh"] = float(hourly["real_missed_activation_neg_mwh"].sum()) if "real_missed_activation_neg_mwh" in hourly.columns else 0.0
        summary["missed_capacity_pos_mw"] = float(hourly["real_missed_capacity_pos_mw"].sum()) if "real_missed_capacity_pos_mw" in hourly.columns else 0.0
        summary["missed_capacity_neg_mw"] = float(hourly["real_missed_capacity_neg_mw"].sum()) if "real_missed_capacity_neg_mw" in hourly.columns else 0.0
        if (
            "real_headroom_violation_pos_mwh" in hourly.columns
            or "real_headroom_violation_neg_mwh" in hourly.columns
            or "headroom_violation_pos_mwh" in hourly.columns
            or "headroom_violation_neg_mwh" in hourly.columns
        ):
            hv_pos = pd.to_numeric(
                hourly.get(
                    "real_headroom_violation_pos_mwh",
                    hourly.get("headroom_violation_pos_mwh", 0.0),
                ),
                errors="coerce",
            ).fillna(0.0)
            hv_neg = pd.to_numeric(
                hourly.get(
                    "real_headroom_violation_neg_mwh",
                    hourly.get("headroom_violation_neg_mwh", 0.0),
                ),
                errors="coerce",
            ).fillna(0.0)
            hv = hv_pos + hv_neg
            summary["headroom_violation_count"] = float((hv > 1e-9).sum())
            summary["headroom_violation_max_mwh"] = float(hv.max() if len(hv) else 0.0)
            hv_pos_series = pd.to_numeric(
                hourly.get("real_headroom_violation_pos_mwh", hourly.get("headroom_violation_pos_mwh", 0.0)),
                errors="coerce",
            ).fillna(0.0)
            hv_neg_series = pd.to_numeric(
                hourly.get("real_headroom_violation_neg_mwh", hourly.get("headroom_violation_neg_mwh", 0.0)),
                errors="coerce",
            ).fillna(0.0)
            summary["reserve_headroom_shortfall_pos_mwh_sum"] = float(hv_pos_series.sum())
            summary["reserve_headroom_shortfall_neg_mwh_sum"] = float(hv_neg_series.sum())
            summary["reserve_headroom_shortfall_max_mwh"] = float(max(hv_pos_series.max(), hv_neg_series.max()) if len(hv_pos_series) else 0.0)
            summary["reserve_headroom_shortfall_penalty_eur"] = 0.0
            summary["reserve_headroom_shortfall_check_pass"] = float(
                (float(summary["reserve_headroom_shortfall_pos_mwh_sum"]) + float(summary["reserve_headroom_shortfall_neg_mwh_sum"])) <= 1e-9
            )
        else:
            summary["headroom_violation_count"] = 0.0
            summary["headroom_violation_max_mwh"] = 0.0
            summary["reserve_headroom_shortfall_pos_mwh_sum"] = 0.0
            summary["reserve_headroom_shortfall_neg_mwh_sum"] = 0.0
            summary["reserve_headroom_shortfall_max_mwh"] = 0.0
            summary["reserve_headroom_shortfall_penalty_eur"] = 0.0
            summary["reserve_headroom_shortfall_check_pass"] = 1.0
        if (
            "real_protected_soc_violation_pos_mwh" in hourly.columns
            or "real_protected_soc_violation_neg_mwh" in hourly.columns
            or "protected_soc_violation_pos_mwh" in hourly.columns
            or "protected_soc_violation_neg_mwh" in hourly.columns
        ):
            psv_pos = pd.to_numeric(
                hourly.get(
                    "real_protected_soc_violation_pos_mwh",
                    hourly.get("protected_soc_violation_pos_mwh", 0.0),
                ),
                errors="coerce",
            ).fillna(0.0)
            psv_neg = pd.to_numeric(
                hourly.get(
                    "real_protected_soc_violation_neg_mwh",
                    hourly.get("protected_soc_violation_neg_mwh", 0.0),
                ),
                errors="coerce",
            ).fillna(0.0)
            psv = psv_pos + psv_neg
            summary["protected_soc_violation_count"] = float((psv > 1e-9).sum())
            summary["protected_soc_violation_max_mwh"] = float(psv.max() if len(psv) else 0.0)
        else:
            summary["protected_soc_violation_count"] = 0.0
            summary["protected_soc_violation_max_mwh"] = 0.0
        psv_wo = pd.to_numeric(
            hourly.get("real_protected_soc_violation_without_obligation", hourly.get("protected_soc_violation_without_obligation", 0.0)),
            errors="coerce",
        ).fillna(0.0)
        summary["protected_soc_violation_without_obligation_count"] = float((psv_wo > 0.5).sum())
        oph_pos = pd.to_numeric(
            hourly.get("real_obligation_headroom_pos_active", hourly.get("obligation_headroom_pos_active", 0.0)),
            errors="coerce",
        ).fillna(0.0)
        oph_neg = pd.to_numeric(
            hourly.get("real_obligation_headroom_neg_active", hourly.get("obligation_headroom_neg_active", 0.0)),
            errors="coerce",
        ).fillna(0.0)
        summary["protected_soc_obligation_pos_active_count"] = float((oph_pos > 0.5).sum())
        summary["protected_soc_obligation_neg_active_count"] = float((oph_neg > 0.5).sum())
        summary["protected_soc_mode"] = "obligation_driven"
        summary["protected_soc_global_buffer_applied"] = float(0.0)
        phys_pos = pd.to_numeric(
            hourly.get("real_physical_soc_violation_pos_mwh", hourly.get("physical_soc_violation_pos_mwh", 0.0)),
            errors="coerce",
        ).fillna(0.0)
        phys_neg = pd.to_numeric(
            hourly.get("real_physical_soc_violation_neg_mwh", hourly.get("physical_soc_violation_neg_mwh", 0.0)),
            errors="coerce",
        ).fillna(0.0)
        phys = phys_pos + phys_neg
        summary["physical_soc_violation_count"] = float((phys > 1e-9).sum())
        summary["physical_soc_violation_max_mwh"] = float(phys.max() if len(phys) else 0.0)
        if (
            "real_locked_reserve_pos_mw" in hourly.columns
            or "real_locked_reserve_neg_mw" in hourly.columns
            or "locked_reserve_pos_mw" in hourly.columns
            or "locked_reserve_neg_mw" in hourly.columns
        ):
            lpos = pd.to_numeric(
                hourly.get("real_locked_reserve_pos_mw", hourly.get("locked_reserve_pos_mw", 0.0)),
                errors="coerce",
            ).fillna(0.0)
            lneg = pd.to_numeric(
                hourly.get("real_locked_reserve_neg_mw", hourly.get("locked_reserve_neg_mw", 0.0)),
                errors="coerce",
            ).fillna(0.0)
            summary["locked_reserve_obligation_hours"] = float(((lpos + lneg) > 1e-9).sum())
            summary["locked_reserve_obligation_pos_mw_sum"] = float(lpos.sum())
            summary["locked_reserve_obligation_neg_mw_sum"] = float(lneg.sum())
        else:
            summary["locked_reserve_obligation_hours"] = 0.0
            summary["locked_reserve_obligation_pos_mw_sum"] = 0.0
            summary["locked_reserve_obligation_neg_mw_sum"] = 0.0
        _hourly_zero = pd.Series(0.0, index=hourly.index, dtype=float)
        pca = pd.to_numeric(
            hourly.get(
                "real_precommit_clamp_applied",
                hourly.get("precommit_clamp_applied", _hourly_zero),
            ),
            errors="coerce",
        ).fillna(0.0)
        summary["precommit_clamp_applied_count"] = float((pca > 0.5).sum())
        summary["precommit_clamped_pos_mw_sum"] = float(
            pd.to_numeric(
                hourly.get(
                    "real_precommit_clamped_pos_mw",
                    hourly.get("precommit_clamped_pos_mw", _hourly_zero),
                ),
                errors="coerce",
            )
            .fillna(0.0)
            .sum()
        )
        summary["precommit_clamped_neg_mw_sum"] = float(
            pd.to_numeric(
                hourly.get(
                    "real_precommit_clamped_neg_mw",
                    hourly.get("precommit_clamped_neg_mw", _hourly_zero),
                ),
                errors="coerce",
            )
            .fillna(0.0)
            .sum()
        )
        summary["precommit_zeroed_due_to_margin_count"] = float(
            pd.to_numeric(
                hourly.get(
                    "real_precommit_zeroed_due_to_margin",
                    hourly.get("precommit_zeroed_due_to_margin", _hourly_zero),
                ),
                errors="coerce",
            )
            .fillna(0.0)
            .gt(0.5)
            .sum()
        )
        summary["precommit_margin_after_bid_min_mwh"] = float(
            pd.to_numeric(
                hourly.get(
                    "real_precommit_margin_after_bid_min_mwh",
                    hourly.get("precommit_margin_after_bid_min_mwh", _hourly_zero),
                ),
                errors="coerce",
            ).fillna(np.nan).min()
        )
        summary["precommit_reduced_due_to_margin_count"] = float(
            pd.to_numeric(
                hourly.get(
                    "real_precommit_reduced_due_to_margin",
                    hourly.get("precommit_reduced_due_to_margin", _hourly_zero),
                ),
                errors="coerce",
            )
            .fillna(0.0)
            .gt(0.5)
            .sum()
        )
        summary["precommit_safe_pos_mw_avg"] = float(
            pd.to_numeric(
                hourly.get(
                    "real_precommit_safe_pos_mw",
                    hourly.get("precommit_safe_pos_mw", _hourly_zero),
                ),
                errors="coerce",
            ).fillna(0.0).mean()
        )
        summary["precommit_safe_neg_mw_avg"] = float(
            pd.to_numeric(
                hourly.get(
                    "real_precommit_safe_neg_mw",
                    hourly.get("precommit_safe_neg_mw", _hourly_zero),
                ),
                errors="coerce",
            ).fillna(0.0).mean()
        )
        summary["desired_reserve_pos_mw_avg"] = float(
            pd.to_numeric(
                hourly.get("real_desired_reserve_pos_mw", hourly.get("desired_reserve_pos_mw", _hourly_zero)),
                errors="coerce",
            ).fillna(0.0).mean()
        )
        summary["desired_reserve_neg_mw_avg"] = float(
            pd.to_numeric(
                hourly.get("real_desired_reserve_neg_mw", hourly.get("desired_reserve_neg_mw", _hourly_zero)),
                errors="coerce",
            ).fillna(0.0).mean()
        )
        summary["safe_reserve_pos_mw_avg"] = float(
            pd.to_numeric(
                hourly.get("real_safe_reserve_pos_mw", hourly.get("safe_reserve_pos_mw", _hourly_zero)),
                errors="coerce",
            ).fillna(0.0).mean()
        )
        summary["safe_reserve_neg_mw_avg"] = float(
            pd.to_numeric(
                hourly.get("real_safe_reserve_neg_mw", hourly.get("safe_reserve_neg_mw", _hourly_zero)),
                errors="coerce",
            ).fillna(0.0).mean()
        )
        summary["submitted_reserve_pos_mw_avg"] = float(
            pd.to_numeric(
                hourly.get("real_submitted_reserve_pos_mw", hourly.get("submitted_reserve_pos_mw", _hourly_zero)),
                errors="coerce",
            ).fillna(0.0).mean()
        )
        summary["submitted_reserve_neg_mw_avg"] = float(
            pd.to_numeric(
                hourly.get("real_submitted_reserve_neg_mw", hourly.get("submitted_reserve_neg_mw", _hourly_zero)),
                errors="coerce",
            ).fillna(0.0).mean()
        )
        summary["desired_reserve_pos_mw"] = float(summary["desired_reserve_pos_mw_avg"])
        summary["desired_reserve_neg_mw"] = float(summary["desired_reserve_neg_mw_avg"])
        summary["safe_reserve_pos_mw"] = float(summary["safe_reserve_pos_mw_avg"])
        summary["safe_reserve_neg_mw"] = float(summary["safe_reserve_neg_mw_avg"])
        summary["submitted_reserve_pos_mw"] = float(summary["submitted_reserve_pos_mw_avg"])
        summary["submitted_reserve_neg_mw"] = float(summary["submitted_reserve_neg_mw_avg"])
        summary["precommit_bid_zeroed_due_to_negative_ev_count"] = float(
            pd.to_numeric(
                hourly.get(
                    "real_precommit_bid_zeroed_due_to_negative_ev",
                    hourly.get("precommit_bid_zeroed_due_to_negative_ev", _hourly_zero),
                ),
                errors="coerce",
            )
            .fillna(0.0)
            .gt(0.5)
            .sum()
        )
        summary["precommit_headroom_recharge_cost_eur_sum"] = float(
            pd.to_numeric(
                hourly.get(
                    "real_precommit_headroom_recharge_cost_eur",
                    hourly.get("precommit_headroom_recharge_cost_eur", _hourly_zero),
                ),
                errors="coerce",
            ).fillna(0.0).sum()
        )
        summary["precommit_headroom_opportunity_cost_eur_sum"] = float(
            pd.to_numeric(
                hourly.get(
                    "real_precommit_headroom_opportunity_cost_eur",
                    hourly.get("precommit_headroom_opportunity_cost_eur", _hourly_zero),
                ),
                errors="coerce",
            ).fillna(0.0).sum()
        )
        pcr = hourly.get("real_precommit_clamp_reason", hourly.get("precommit_clamp_reason"))
        if pcr is None:
            summary["precommit_clamp_reasons"] = "{}"
            summary["precommit_reduction_reason"] = "{}"
        else:
            vc = pcr.fillna("none").astype(str).value_counts(dropna=False).to_dict()
            summary["precommit_clamp_reasons"] = json.dumps({str(k): float(v) for k, v in vc.items()}, sort_keys=True)
            prr = hourly.get("real_precommit_reduction_reason", hourly.get("precommit_reduction_reason"))
            if prr is None:
                summary["precommit_reduction_reason"] = "{}"
            else:
                vc2 = prr.fillna("none").astype(str).value_counts(dropna=False).to_dict()
                summary["precommit_reduction_reason"] = json.dumps({str(k): float(v) for k, v in vc2.items()}, sort_keys=True)
        summary["bem_only_mode"] = "explicit_optimizer"
        summary["bem_only_explicit_optimizer"] = 1.0
        summary["bem_only_disabled_by_config"] = float(self.disable_bem_only)
        summary["bem_only_headroom_safety_mwh"] = float(self.bem_only_headroom_safety_mwh)
        summary["max_bem_only_bid_mw"] = (
            float(self.max_bem_only_bid_mw) if self.max_bem_only_bid_mw is not None else float("nan")
        )
        bcm_col = "real_afrr_bcm_auction_cleared"
        if bcm_col in hourly.columns:
            bcm_mask = pd.to_numeric(hourly[bcm_col], errors="coerce").fillna(0.0) > 0.5
            bem_only_mask = ~bcm_mask
            summary["bem_only_hours"] = float(bem_only_mask.sum())
            bem_pos_bid = (
                pd.to_numeric(hourly["real_bem_only_submitted_pos_mw"], errors="coerce").fillna(0.0)
                if "real_bem_only_submitted_pos_mw" in hourly.columns
                else pd.Series(0.0, index=hourly.index)
            )
            bem_neg_bid = (
                pd.to_numeric(hourly["real_bem_only_submitted_neg_mw"], errors="coerce").fillna(0.0)
                if "real_bem_only_submitted_neg_mw" in hourly.columns
                else pd.Series(0.0, index=hourly.index)
            )
            summary["bem_only_pos_bid_hours"] = float((bem_pos_bid > 1e-12).sum())
            summary["bem_only_neg_bid_hours"] = float((bem_neg_bid > 1e-12).sum())
            summary["bem_only_pos_bid_mw_sum"] = float(bem_pos_bid.sum())
            summary["bem_only_neg_bid_mw_sum"] = float(bem_neg_bid.sum())
            pos_acc = (
                pd.to_numeric(hourly["real_afrr_act_pos_accepted"], errors="coerce").fillna(0.0) > 0.5
                if "real_afrr_act_pos_accepted" in hourly.columns
                else pd.Series(False, index=hourly.index)
            )
            neg_acc = (
                pd.to_numeric(hourly["real_afrr_act_neg_accepted"], errors="coerce").fillna(0.0) > 0.5
                if "real_afrr_act_neg_accepted" in hourly.columns
                else pd.Series(False, index=hourly.index)
            )
            summary["bem_only_pos_activation_accept_count"] = float((bem_only_mask & pos_acc).sum())
            summary["bem_only_neg_activation_accept_count"] = float((bem_only_mask & neg_acc).sum())
            pos_bin_cols = [c for c in hourly.columns if c.startswith("real_executed_afrr_act_pos_bin_") and c.endswith("_mw")]
            neg_bin_cols = [c for c in hourly.columns if c.startswith("real_executed_afrr_act_neg_bin_") and c.endswith("_mw")]
            if pos_bin_cols:
                pos_vals = pd.to_numeric(hourly.loc[bem_only_mask, pos_bin_cols].stack(), errors="coerce").fillna(0.0)
                summary["bem_only_pos_activation_mw_sum"] = float(pos_vals.sum())
            else:
                summary["bem_only_pos_activation_mw_sum"] = 0.0
            if neg_bin_cols:
                neg_vals = pd.to_numeric(hourly.loc[bem_only_mask, neg_bin_cols].stack(), errors="coerce").fillna(0.0)
                summary["bem_only_neg_activation_mw_sum"] = float(neg_vals.sum())
            else:
                summary["bem_only_neg_activation_mw_sum"] = 0.0
            bem_pos_mwh = (
                pd.to_numeric(hourly["real_bem_only_executed_pos_mwh"], errors="coerce").fillna(0.0)
                if "real_bem_only_executed_pos_mwh" in hourly.columns
                else pd.Series(0.0, index=hourly.index)
            )
            bem_neg_mwh = (
                pd.to_numeric(hourly["real_bem_only_executed_neg_mwh"], errors="coerce").fillna(0.0)
                if "real_bem_only_executed_neg_mwh" in hourly.columns
                else pd.Series(0.0, index=hourly.index)
            )
            summary["bem_only_pos_activation_mwh_sum"] = float(bem_pos_mwh.sum())
            summary["bem_only_neg_activation_mwh_sum"] = float(bem_neg_mwh.sum())
            summary["bem_only_activation_revenue_eur"] = float(
                pd.to_numeric(hourly.get("real_bem_only_activation_revenue_eur", 0.0), errors="coerce").fillna(0.0).sum()
            )
            guard_applied = pd.to_numeric(hourly.get("real_bem_only_headroom_guard_applied", 0.0), errors="coerce").fillna(0.0)
            red_pos = pd.to_numeric(hourly.get("real_bem_only_pos_reduced_by_headroom_mw", 0.0), errors="coerce").fillna(0.0)
            red_neg = pd.to_numeric(hourly.get("real_bem_only_neg_reduced_by_headroom_mw", 0.0), errors="coerce").fillna(0.0)
            summary["bem_only_headroom_guard_applied_count"] = float((guard_applied > 0.5).sum())
            summary["bem_only_pos_reduced_by_headroom_mw_sum"] = float(red_pos.sum())
            summary["bem_only_neg_reduced_by_headroom_mw_sum"] = float(red_neg.sum())
            summary["bem_only_headroom_guard_max_reduction_mw"] = float(
                max((red_pos + red_neg).max() if len(red_pos) else 0.0, 0.0)
            )
            guard_hours = pd.to_datetime(
                hourly.loc[guard_applied > 0.5, colmap.timestamp] if colmap.timestamp in hourly.columns else pd.Series([], dtype="datetime64[ns, UTC]"),
                utc=True,
                errors="coerce",
            ).dropna()
            summary["bem_only_headroom_guard_hours"] = json.dumps(
                [ts.isoformat() for ts in guard_hours.tolist()],
                sort_keys=False,
            )
        else:
            summary["bem_only_hours"] = 0.0
            summary["bem_only_pos_bid_hours"] = 0.0
            summary["bem_only_neg_bid_hours"] = 0.0
            summary["bem_only_pos_bid_mw_sum"] = 0.0
            summary["bem_only_neg_bid_mw_sum"] = 0.0
            summary["bem_only_pos_activation_accept_count"] = 0.0
            summary["bem_only_neg_activation_accept_count"] = 0.0
            summary["bem_only_pos_activation_mw_sum"] = 0.0
            summary["bem_only_neg_activation_mw_sum"] = 0.0
            summary["bem_only_pos_activation_mwh_sum"] = 0.0
            summary["bem_only_neg_activation_mwh_sum"] = 0.0
            summary["bem_only_activation_revenue_eur"] = 0.0
            summary["bem_only_headroom_guard_applied_count"] = 0.0
            summary["bem_only_pos_reduced_by_headroom_mw_sum"] = 0.0
            summary["bem_only_neg_reduced_by_headroom_mw_sum"] = 0.0
            summary["bem_only_headroom_guard_max_reduction_mw"] = 0.0
            summary["bem_only_headroom_guard_hours"] = "[]"
        # Keep ROI numerically stable by flooring denominator at 1 EUR.
        roi_denom = max(1.0, float(summary["max_capital_required_eur"]))
        summary["roi_on_max_capital"] = float(summary["realized_total_pnl_eur"] / roi_denom)
        summary["capture_ratio_vs_naive"] = (
            float(summary["realized_total_pnl_eur"] / summary["naive_total_pnl_eur"])
            if abs(float(summary["naive_total_pnl_eur"])) > 1e-12 else float("nan")
        )
        summary["capture_ratio_vs_perfect_foresight"] = (
            float(summary["realized_total_pnl_eur"] / summary["rolling_perfect_foresight_same_rules_total_pnl_eur"])
            if abs(float(summary["rolling_perfect_foresight_same_rules_total_pnl_eur"])) > 1e-12 else float("nan")
        )
        summary["capture_ratio_vs_perfect_foresight"] = float(summary["capture_ratio_vs_perfect_foresight"])
        summary["capture_ratio_vs_perfect_foresight_is_deprecated"] = 1.0
        summary["capture_ratio_vs_perfect_foresight_semantics"] = "rolling_perfect_foresight_same_rules"
        summary["capture_ratio_vs_perfect_foresight_is_deprecated"] = 1.0
        summary["capture_ratio_vs_perfect_foresight_semantics"] = "rolling_perfect_foresight_same_rules"
        summary["realized_vs_perfect_foresight_ratio_multi_market"] = float(
            summary.get("realized_vs_perfect_foresight_ratio_multi_market", float("nan"))
        )
        summary["realized_vs_perfect_foresight_ratio_multi_market"] = float(
            summary.get("realized_vs_perfect_foresight_ratio_multi_market", float("nan"))
        )
        summary["realized_vs_perfect_foresight_pct"] = float(
            100.0 * summary.get("realized_vs_perfect_foresight_ratio_multi_market", float("nan"))
        )
        summary["perfect_foresight_upper_bound_ok"] = float("nan")
        summary["perfect_foresight_upper_bound_ok"] = float("nan")
        if "realized_vs_perfect_foresight_ratio_multi_market" in summary:
            summary["realized_vs_perfect_foresight_ratio_multi_market"] = float(
                summary["realized_vs_perfect_foresight_ratio_multi_market"]
            )
        cmp_real = float(summary.get("comparable_realized_market_pnl_eur", float("nan")))
        cmp_bench = float(summary.get("comparable_perfect_foresight_market_pnl_eur", float("nan")))
        summary["realized_vs_perfect_foresight_comparable_market_ratio"] = (
            float(cmp_real / cmp_bench) if np.isfinite(cmp_real) and np.isfinite(cmp_bench) and abs(cmp_bench) > 1e-12 else float("nan")
        )
        summary["realized_vs_perfect_foresight_comparable_market_ratio"] = float(
            summary["realized_vs_perfect_foresight_comparable_market_ratio"]
        )
        global_perfect_foresight_total = float(summary.get("global_hindsight_perfect_foresight_upper_bound_total_pnl_eur", float("nan")))
        realized_total = float(summary.get("realized_total_pnl_eur", float("nan")))
        eps_global_perfect_foresight = 1e-2
        summary["realized_minus_global_perfect_foresight_eur"] = (
            float(realized_total - global_perfect_foresight_total)
            if np.isfinite(realized_total) and np.isfinite(global_perfect_foresight_total)
            else float("nan")
        )
        summary["realized_vs_global_hindsight_perfect_foresight_upper_bound_pct"] = (
            float(100.0 * realized_total / global_perfect_foresight_total)
            if (
                (summary.get("global_perfect_foresight_available", 0.0) >= 0.5)
                and np.isfinite(realized_total)
                and np.isfinite(global_perfect_foresight_total)
                and abs(global_perfect_foresight_total) > 1e-12
            )
            else float("nan")
        )
        summary["realized_exceeds_global_perfect_foresight"] = float(
            (summary.get("global_perfect_foresight_available", 0.0) >= 0.5)
            and np.isfinite(realized_total)
            and np.isfinite(global_perfect_foresight_total)
            and (realized_total - global_perfect_foresight_total) > eps_global_perfect_foresight
        )
        summary["global_perfect_foresight_dominance_check_pass"] = float(
            (summary.get("global_perfect_foresight_available", 0.0) < 0.5)
            or (summary["realized_exceeds_global_perfect_foresight"] < 0.5)
        )
        summary["global_perfect_foresight_is_upper_bound"] = float(
            (summary.get("global_perfect_foresight_available", 0.0) >= 0.5)
            and (summary.get("global_perfect_foresight_dominance_check_pass", 0.0) >= 0.5)
        )
        summary["global_hindsight_perfect_foresight_is_global_upper_bound"] = float(summary["global_perfect_foresight_is_upper_bound"])
        summary["global_perfect_foresight_pnl_reconciliation_error_eur"] = (
            float(
                summary["global_hindsight_perfect_foresight_upper_bound_total_pnl_eur"]
                - (
                    _sum_col_zero("global_perfect_foresight_revenue_da_eur")
                    + _sum_col_zero("global_perfect_foresight_revenue_id_eur")
                    + _sum_col_zero("global_perfect_foresight_revenue_capacity_eur")
                    + _sum_col_zero("global_perfect_foresight_revenue_activation_eur")
                    - _sum_col_zero("global_perfect_foresight_cost_da_eur")
                    - _sum_col_zero("global_perfect_foresight_cost_id_eur")
                    - _sum_col_zero("global_perfect_foresight_degradation_cost_eur")
                    - _sum_col_zero("global_perfect_foresight_transaction_cost_eur")
                    - _sum_col_zero("global_perfect_foresight_aux_cost_eur")
                    - _sum_col_zero("global_perfect_foresight_penalty_eur")
                    + term_adj_global_perfect_foresight["adjustment_eur"]
                )
            )
            if summary.get("global_perfect_foresight_available", 0.0) >= 0.5 else float("nan")
        )
        if "shock_source" in hourly.columns:
            ss = hourly["shock_source"].fillna("none").astype(str)
            summary["soc_shock_events_total"] = float((ss != "none").sum())
            summary["soc_shock_events_da_reject"] = float(ss.str.contains("da_reject", regex=False).sum())
            summary["soc_shock_events_afrr_capacity_reject"] = float(ss.str.contains("afrr_capacity_reject", regex=False).sum())
            summary["soc_shock_events_afrr_activation_reject"] = float(ss.str.contains("afrr_activation_reject", regex=False).sum())
        else:
            summary["soc_shock_events_total"] = 0.0
            summary["soc_shock_events_da_reject"] = 0.0
            summary["soc_shock_events_afrr_capacity_reject"] = 0.0
            summary["soc_shock_events_afrr_activation_reject"] = 0.0
        summary["initial_soc_mwh"] = float(self.soc_init)
        summary["final_real_soc_mwh"] = float(final_real_soc_mwh)
        summary["final_soc_mwh"] = float(final_real_soc_mwh)
        summary["final_soc_actual_mwh"] = float(final_real_soc_mwh)
        summary["target_final_soc_mwh"] = float(self.soc_target_end)
        summary["final_soc_min_target_mwh"] = float(self.soc_target_end)
        summary["final_soc_target_mwh"] = float(self.soc_target_end)
        summary["final_soc_constraint_satisfied"] = float(final_real_soc_mwh >= float(self.soc_target_end) - 1e-9)
        summary["final_soc_physical_check_pass"] = float(summary["final_soc_constraint_satisfied"])
        summary["final_soc_mode"] = str(self.final_soc_mode)
        summary["final_soc_exact_match_pass"] = float(abs(float(final_real_soc_mwh) - float(self.soc_target_end)) <= 1e-6)
        # Terminal inventory is explicitly settled to target.
        # In strict mode, shortfall is only accepted when explicit repair cashflow is applied.
        summary["final_soc_handling_mode"] = "terminal_settlement_to_target"
        summary["final_soc_check_mode"] = "terminal_settlement_to_target"
        summary["final_soc_shortfall_mwh"] = float(term_adj_real["shortfall_mwh"])
        summary["final_soc_slack_used_mwh"] = float(
            pd.to_numeric(
                hourly.get("real_final_soc_shortfall_mwh", hourly.get("final_soc_shortfall_mwh", 0.0)),
                errors="coerce",
            )
            .fillna(0.0)
            .max()
            if len(hourly) > 0
            else 0.0
        )
        summary["final_soc_surplus_mwh"] = float(term_adj_real["surplus_mwh"])
        summary["terminal_constraint_dropped"] = float(
            pd.to_numeric(
                hourly.get("real_terminal_constraint_dropped", hourly.get("terminal_constraint_dropped", 0.0)),
                errors="coerce",
            )
            .fillna(0.0)
            .gt(0.5)
            .any()
            if len(hourly) > 0
            else 0.0
        )
        ts_series_utc = pd.to_datetime(hourly[colmap.timestamp], utc=True, errors="coerce") if colmap.timestamp in hourly.columns else pd.Series(dtype="datetime64[ns, UTC]")
        if len(ts_series_utc) > 0 and ts_series_utc.notna().any():
            ts_max = ts_series_utc.dropna().max()
            last24_mask = ts_series_utc >= (ts_max - pd.Timedelta(hours=23))
        else:
            last24_mask = pd.Series(False, index=hourly.index, dtype=bool)
        def _sum_last24(col_name: str) -> float:
            if col_name not in hourly.columns:
                return 0.0
            return float(pd.to_numeric(hourly[col_name], errors="coerce").fillna(0.0)[last24_mask].sum())
        def _max_last24(col_name: str) -> float:
            if col_name not in hourly.columns:
                return 0.0
            s = pd.to_numeric(hourly[col_name], errors="coerce").fillna(0.0)[last24_mask]
            return float(s.max()) if len(s) else 0.0
        summary["da_charge_mwh_last_24h"] = _sum_last24("real_da_buy_mwh")
        if {"real_submitted_da_buy_mw", "real_da_buy_accepted"}.issubset(hourly.columns):
            locked_da_buy = (
                pd.to_numeric(hourly["real_submitted_da_buy_mw"], errors="coerce").fillna(0.0)
                * pd.to_numeric(hourly["real_da_buy_accepted"], errors="coerce").fillna(0.0)
                * float(self.dt_h)
            )
            summary["locked_da_buy_mwh_last_24h"] = float(locked_da_buy[last24_mask].sum())
        else:
            summary["locked_da_buy_mwh_last_24h"] = 0.0
        if {"real_submitted_da_sell_mw", "real_da_sell_accepted"}.issubset(hourly.columns):
            locked_da_sell = (
                pd.to_numeric(hourly["real_submitted_da_sell_mw"], errors="coerce").fillna(0.0)
                * pd.to_numeric(hourly["real_da_sell_accepted"], errors="coerce").fillna(0.0)
                * float(self.dt_h)
            )
            summary["locked_da_sell_mwh_last_24h"] = float(locked_da_sell[last24_mask].sum())
        else:
            summary["locked_da_sell_mwh_last_24h"] = 0.0
        summary["max_possible_charge_mwh_last_24h"] = float(last24_mask.sum()) * float(self.p_max_mw) * float(self.dt_h)
        summary["final_window_locked_reserve_pos_mw_sum"] = _sum_last24("real_fixed_reserve_obligation_pos_mw")
        summary["final_window_locked_reserve_neg_mw_sum"] = _sum_last24("real_fixed_reserve_obligation_neg_mw")
        summary["final_window_locked_reserve_pos_mw_max"] = _max_last24("real_fixed_reserve_obligation_pos_mw")
        summary["final_window_locked_reserve_neg_mw_max"] = _max_last24("real_fixed_reserve_obligation_neg_mw")
        shortfall_tol_mwh = 1e-6
        terminal_repair_included = bool(
            np.isfinite(float(summary.get("terminal_soc_net_adjustment_eur", float("nan"))))
            and np.isfinite(float(summary.get("realized_pnl_excl_terminal_eur", float("nan"))))
            and np.isfinite(float(summary.get("realized_total_pnl_eur", float("nan"))))
            and abs(
                float(summary["realized_total_pnl_eur"])
                - (
                    float(summary["realized_pnl_excl_terminal_eur"])
                    + float(summary["terminal_soc_net_adjustment_eur"])
                )
            )
            <= 1e-4
        )
        summary["terminal_soc_repair_included_in_pnl"] = float(terminal_repair_included)
        economic_repair_pass = bool(
            (float(summary["final_soc_shortfall_mwh"]) <= shortfall_tol_mwh)
            or (
                float(summary["terminal_soc_repair_cost_eur"]) > 0.0
                and terminal_repair_included
            )
        )
        summary["final_soc_economic_repair_check_pass"] = float(economic_repair_pass)
        if str(self.final_soc_mode) == "hard":
            summary["final_soc_check_pass"] = float(summary["final_soc_physical_check_pass"])
        else:
            summary["final_soc_check_pass"] = float(economic_repair_pass)
        summary["final_soc_shortfall_check_pass"] = float(summary["final_soc_shortfall_mwh"] <= 1e-6)
        if not volatility.empty:
            summary["decision_volatility_total_flips"] = float(volatility["action_flips"].sum())
            summary["decision_volatility_mean_flips_per_target"] = float(volatility["action_flips"].mean())
            summary["decision_volatility_mean_abs_revision_mw"] = float(volatility["mean_abs_revision_mw"].mean())
        else:
            summary["decision_volatility_total_flips"] = 0.0
            summary["decision_volatility_mean_flips_per_target"] = 0.0
            summary["decision_volatility_mean_abs_revision_mw"] = 0.0
        if not naive_volatility.empty:
            summary["naive_decision_volatility_total_flips"] = float(naive_volatility["action_flips"].sum())
        else:
            summary["naive_decision_volatility_total_flips"] = 0.0
        summary["decision_volatility_vs_naive_delta_flips"] = (
            float(summary["decision_volatility_total_flips"] - summary["naive_decision_volatility_total_flips"])
        )

        # Run validity flags for thesis-safe filtering.
        missed_cap_pos = float(summary.get("missed_capacity_pos_mw", 0.0))
        missed_cap_neg = float(summary.get("missed_capacity_neg_mw", 0.0))
        missed_act_pos = float(summary.get("missed_activation_pos_mwh", 0.0))
        missed_act_neg = float(summary.get("missed_activation_neg_mwh", 0.0))
        headroom_ok = float(summary.get("headroom_violation_count", 0.0)) <= 1e-9
        pnl_recon_err = float(summary.get("pnl_reconciliation_error_max_eur", 0.0))
        final_soc_ok = bool(float(summary.get("final_soc_check_pass", 0.0)) >= 0.5)
        missed_capacity_ok = (missed_cap_pos + missed_cap_neg) <= 1e-9
        missed_activation_ok = (missed_act_pos + missed_act_neg) <= 1e-9
        pnl_recon_ok = pnl_recon_err <= 1e-2
        protected_soc_effective_count = max(
            0.0,
            float(summary.get("protected_soc_violation_count", 0.0))
            - float(summary.get("protected_soc_violation_without_obligation_count", 0.0)),
        )
        protected_soc_ok = protected_soc_effective_count <= 1e-9
        physical_soc_ok = float(summary.get("physical_soc_violation_count", 0.0)) <= 1e-9
        summary["missed_capacity_check_pass"] = float(missed_capacity_ok)
        summary["missed_activation_check_pass"] = float(missed_activation_ok)
        summary["pnl_reconciliation_check_pass"] = float(pnl_recon_ok)
        summary["headroom_check_pass"] = float(headroom_ok)
        summary["protected_soc_check_pass"] = float(protected_soc_ok)
        summary["physical_soc_check_pass"] = float(physical_soc_ok)
        summary["strict_simulation_validity"] = float(bool(strict_simulation_validity))
        if "real_optimization_fallback" in hourly.columns:
            fb_counts = hourly["real_optimization_fallback"].fillna("none").astype(str).value_counts(dropna=False).to_dict()
            summary["fallback_mode_counts"] = json.dumps({str(k): float(v) for k, v in fb_counts.items()}, sort_keys=True)
        elif "optimization_fallback" in hourly.columns:
            fb_counts = hourly["optimization_fallback"].fillna("none").astype(str).value_counts(dropna=False).to_dict()
            summary["fallback_mode_counts"] = json.dumps({str(k): float(v) for k, v in fb_counts.items()}, sort_keys=True)
        else:
            # Backfill from optimization_error_code if dedicated fallback column is missing.
            if "real_optimization_error_code" in hourly.columns:
                ec = hourly["real_optimization_error_code"].fillna("ok").astype(str).str.strip().str.lower()
            elif "optimization_error_code" in hourly.columns:
                ec = hourly["optimization_error_code"].fillna("ok").astype(str).str.strip().str.lower()
            else:
                ec = pd.Series(dtype=str)
            if len(ec) > 0:
                fb_like = ec[~ec.isin(["", "ok", "none"])]
                if not fb_like.empty:
                    fb_counts = fb_like.value_counts(dropna=False).to_dict()
                    summary["fallback_mode_counts"] = json.dumps(
                        {str(k): float(v) for k, v in fb_counts.items()}, sort_keys=True
                    )
                else:
                    summary["fallback_mode_counts"] = "{}"
            else:
                summary["fallback_mode_counts"] = "{}"
        if "real_optimization_error_code" in hourly.columns:
            ec_counts = hourly["real_optimization_error_code"].fillna("ok").astype(str).value_counts(dropna=False).to_dict()
            summary["optimization_error_code_counts"] = json.dumps({str(k): float(v) for k, v in ec_counts.items()}, sort_keys=True)
        elif "optimization_error_code" in hourly.columns:
            ec_counts = hourly["optimization_error_code"].fillna("ok").astype(str).value_counts(dropna=False).to_dict()
            summary["optimization_error_code_counts"] = json.dumps({str(k): float(v) for k, v in ec_counts.items()}, sort_keys=True)
        else:
            summary["optimization_error_code_counts"] = "{}"
        fallback_used = 0.0
        if "real_optimization_fallback" in hourly.columns:
            fallback_used = float((hourly["real_optimization_fallback"].fillna("none").astype(str).str.lower() != "none").any())
        elif "optimization_fallback" in hourly.columns:
            fallback_used = float((hourly["optimization_fallback"].fillna("none").astype(str).str.lower() != "none").any())
        elif "optimizer_fallback_used" in hourly.columns:
            fallback_used = float(pd.to_numeric(hourly["optimizer_fallback_used"], errors="coerce").fillna(0.0).gt(0.5).any())
        summary["fallback_used"] = float(fallback_used)
        fallback_counts_txt = str(summary.get("fallback_mode_counts", "{}") or "{}").lower()
        reserve_feasibility_repair_used = float("reserve_feasibility_repair" in fallback_counts_txt)
        summary["reserve_feasibility_repair_used"] = float(reserve_feasibility_repair_used)
        opt_codes_series = None
        if "real_optimization_error_code" in hourly.columns:
            opt_codes_series = hourly["real_optimization_error_code"].fillna("ok").astype(str).str.lower().str.strip()
        elif "optimization_error_code" in hourly.columns:
            opt_codes_series = hourly["optimization_error_code"].fillna("ok").astype(str).str.lower().str.strip()
        optimization_non_ok = float(False)
        if opt_codes_series is not None:
            optimization_non_ok = float((~opt_codes_series.isin(["ok", "none", ""])).any())
        if float(summary.get("accepted_path_infeasible_debug_dump_count", 0.0)) > 0.5 and not str(
            summary.get("first_infeasible_timestamp_utc", "")
        ).strip():
            if opt_codes_series is not None:
                bad_idx = opt_codes_series[~opt_codes_series.isin(["ok", "none", ""])].index
                if len(bad_idx) > 0:
                    ts_bad = pd.to_datetime(hourly.loc[bad_idx, colmap.timestamp], utc=True, errors="coerce").dropna()
                    if len(ts_bad) > 0:
                        summary["first_infeasible_timestamp_utc"] = ts_bad.min().isoformat()
        reserve_infeasible_hours = 0.0
        if "optimizer_fallback_used" in hourly.columns:
            def _num_series(col: str) -> pd.Series:
                if col in hourly.columns:
                    return pd.to_numeric(hourly[col], errors="coerce").fillna(0.0)
                return pd.Series(0.0, index=hourly.index, dtype=float)

            fb = pd.to_numeric(hourly["optimizer_fallback_used"], errors="coerce").fillna(0.0)
            ob_pos = _num_series("real_fixed_reserve_obligation_pos_mw")
            ob_neg = _num_series("real_fixed_reserve_obligation_neg_mw")
            hv_pos = _num_series("real_headroom_violation_pos_mwh")
            hv_neg = _num_series("real_headroom_violation_neg_mwh")
            mc_pos = _num_series("real_missed_capacity_pos_mw")
            mc_neg = _num_series("real_missed_capacity_neg_mw")
            reserve_infeasible_mask = (
                (fb > 0.5)
                & ((ob_pos + ob_neg) > 1e-9)
                & (((hv_pos + hv_neg) > 1e-9) | ((mc_pos + mc_neg) > 1e-9))
            )
            reserve_infeasible_hours = float(reserve_infeasible_mask.sum())
        summary["reserve_infeasible_hours"] = float(reserve_infeasible_hours)
        invalid_reasons: list[str] = []
        if reserve_infeasible_hours > 0:
            invalid_reasons.append("reserve_infeasible")
        if bool(strict_simulation_validity):
            if fallback_used > 0.5:
                invalid_reasons.append("fallback_used")
            if reserve_feasibility_repair_used > 0.5:
                invalid_reasons.append("reserve_feasibility_repair")
            if float(summary.get("accepted_path_infeasible_debug_dump_count", 0.0)) > 0.5:
                invalid_reasons.append("optimization_infeasible_debug_dump")
            if optimization_non_ok > 0.5:
                if opt_codes_series is not None and opt_codes_series.str.contains("infeasible|reserve_infeasible", regex=True).any():
                    invalid_reasons.append("optimization_infeasible")
                elif opt_codes_series is not None and opt_codes_series.str.contains("solver|not set|numerical", regex=True).any():
                    invalid_reasons.append("solver_failure")
                else:
                    invalid_reasons.append("optimization_failure")
            if float(summary.get("benchmark_same_rules_gate_consistent", 1.0)) < 0.5:
                invalid_reasons.append("benchmark_gate_mismatch")
            if (
                float(summary.get("global_perfect_foresight_available", 0.0)) >= 0.5
                and float(summary.get("global_perfect_foresight_dominance_check_pass", 1.0)) < 0.5
            ):
                invalid_reasons.append("realized_exceeds_global_perfect_foresight")
        if not final_soc_ok:
            if (
                float(summary.get("final_soc_shortfall_mwh", 0.0)) > 1e-6
                and (
                    float(summary.get("terminal_soc_repair_cost_eur", 0.0)) <= 0.0
                    or float(summary.get("terminal_soc_repair_included_in_pnl", 0.0)) < 0.5
                )
            ):
                invalid_reasons.append("final_soc_unrepaired")
            else:
                invalid_reasons.append("terminal_soc")
        if not missed_capacity_ok:
            invalid_reasons.append("missed_capacity")
        if not missed_activation_ok:
            invalid_reasons.append("missed_activation")
        if not pnl_recon_ok:
            invalid_reasons.append("pnl_reconciliation")
        if not headroom_ok:
            invalid_reasons.append("headroom")
        if float(summary.get("reserve_headroom_shortfall_check_pass", 1.0)) < 0.5:
            invalid_reasons.append("reserve_headroom_shortfall")
        if not physical_soc_ok:
            invalid_reasons.append("physical_soc")
        if not protected_soc_ok:
            invalid_reasons.append("protected_soc")
        invalid_reasons = list(dict.fromkeys(invalid_reasons))
        summary["simulation_valid"] = float(len(invalid_reasons) == 0)
        summary["invalid_reason"] = ",".join(invalid_reasons)
        infeasibility_driver = "none"
        if summary["simulation_valid"] < 0.5:
            if (not headroom_ok) or ("headroom" in invalid_reasons) or ("reserve_infeasible" in invalid_reasons):
                infeasibility_driver = "reserve_headroom"
            elif ("optimization_infeasible" in invalid_reasons) or ("terminal_soc" in invalid_reasons):
                infeasibility_driver = "terminal_soc"
            elif ("solver_failure" in invalid_reasons) or ("optimization_failure" in invalid_reasons):
                infeasibility_driver = "solver_failure"
            else:
                infeasibility_driver = "unknown"
        summary["infeasibility_driver"] = str(infeasibility_driver)
        if "optimization_error_code" in hourly.columns:
            drv = np.full(len(hourly), "none", dtype=object)
            if infeasibility_driver != "none":
                mask_bad = hourly["optimization_error_code"].fillna("ok").astype(str).str.lower().ne("ok")
                drv = np.where(mask_bad.to_numpy(dtype=bool), infeasibility_driver, drv)
            hourly["infeasibility_driver"] = drv
        summary["thesis_reportable"] = float(
            (summary["simulation_valid"] >= 0.5)
            and (summary.get("fallback_used", 0.0) <= 0.5)
            and (summary.get("reserve_feasibility_repair_used", 0.0) <= 0.5)
            and (summary.get("final_soc_check_pass", 0.0) >= 0.5)
            and (summary.get("final_soc_physical_check_pass", 0.0) >= 0.5)
            and (summary.get("missed_capacity_check_pass", 0.0) >= 0.5)
            and (summary.get("missed_activation_check_pass", 0.0) >= 0.5)
            and (summary.get("headroom_check_pass", 0.0) >= 0.5)
            and (summary.get("protected_soc_check_pass", 0.0) >= 0.5)
            and (summary.get("reserve_headroom_shortfall_check_pass", 0.0) >= 0.5)
            and (summary.get("pnl_reconciliation_check_pass", 0.0) >= 0.5)
        )
        summary["reserve_soc_projection_safety_mwh"] = float(self.reserve_soc_projection_safety_mwh)
        summary["reserve_headroom_safety_mwh"] = float(self.reserve_headroom_safety_mwh)
        summary["reserve_power_safety_mw"] = float(self.reserve_power_safety_mw)
        summary["reserve_min_margin_after_bid_mwh"] = float(self.reserve_min_margin_after_bid_mwh)
        summary["reserve_bid_derate"] = float(self.reserve_bid_derate)
        summary["max_reserve_bid_mw"] = float(self.max_reserve_bid_mw) if self.max_reserve_bid_mw is not None else float("nan")
        summary["reserve_feasibility_mode"] = str(self.reserve_feasibility_mode)
        summary["reserve_retry_ladder"] = ",".join(f"{float(x):g}" for x in self.reserve_retry_ladder)
        summary["disable_new_bcm_reserve_bids"] = float(self.disable_new_bcm_reserve_bids)
        if "reserve_retry_attempts_used" in hourly.columns:
            summary["reserve_retry_attempts_used"] = float(
                pd.to_numeric(hourly["reserve_retry_attempts_used"], errors="coerce").fillna(0.0).max()
            )
            summary["reserve_retry_final_factor"] = float(
                pd.to_numeric(hourly.get("reserve_retry_final_factor", 1.0), errors="coerce").fillna(1.0).min()
            )
            summary["reserve_retry_succeeded"] = float(
                pd.to_numeric(hourly.get("reserve_retry_succeeded", 0.0), errors="coerce").fillna(0.0).max()
            )
            summary["new_reserve_bids_zeroed_by_retry"] = float(
                pd.to_numeric(hourly.get("new_reserve_bids_zeroed_by_retry", 0.0), errors="coerce").fillna(0.0).max()
            )
            summary["reserve_retry_infeasible_after_zero_reserve"] = float(
                pd.to_numeric(hourly.get("reserve_retry_infeasible_after_zero_reserve", 0.0), errors="coerce").fillna(0.0).max()
            )
        else:
            summary["reserve_retry_attempts_used"] = 0.0
            summary["reserve_retry_final_factor"] = 1.0
            summary["reserve_retry_succeeded"] = 0.0
            summary["new_reserve_bids_zeroed_by_retry"] = 0.0
            summary["reserve_retry_infeasible_after_zero_reserve"] = 0.0
        summary["afrr_bcm_gate_hour_cet_model"] = float(self.afrr_bcm_gate_hour_cet)
        summary["afrr_bcm_gate_hour_cet_benchmark"] = float(self.afrr_bcm_gate_hour_cet)
        summary["benchmark_same_rules_gate_consistent"] = 1.0
        summary["fallback_is_repair_optimization"] = 0.0
        for k, v in [
            ("activation_split_reconciliation_error_max", 0.0),
            ("precommit_clamp_applied_count", 0.0),
            ("precommit_clamped_pos_mw_sum", 0.0),
            ("precommit_clamped_neg_mw_sum", 0.0),
            ("precommit_clamp_reasons", "{}"),
            ("precommit_reduction_reason", "{}"),
            ("precommit_zeroed_due_to_margin_count", 0.0),
            ("precommit_reduced_due_to_margin_count", 0.0),
            ("desired_reserve_pos_mw_avg", 0.0),
            ("desired_reserve_neg_mw_avg", 0.0),
            ("safe_reserve_pos_mw_avg", 0.0),
            ("safe_reserve_neg_mw_avg", 0.0),
            ("submitted_reserve_pos_mw_avg", 0.0),
            ("submitted_reserve_neg_mw_avg", 0.0),
            ("desired_reserve_pos_mw", 0.0),
            ("desired_reserve_neg_mw", 0.0),
            ("safe_reserve_pos_mw", 0.0),
            ("safe_reserve_neg_mw", 0.0),
            ("submitted_reserve_pos_mw", 0.0),
            ("submitted_reserve_neg_mw", 0.0),
            ("precommit_bid_zeroed_due_to_negative_ev_count", 0.0),
            ("precommit_headroom_recharge_cost_eur_sum", 0.0),
            ("precommit_headroom_opportunity_cost_eur_sum", 0.0),
            ("fallback_mode_counts", "{}"),
            ("optimization_error_code_counts", "{}"),
            ("fallback_used", 0.0),
            ("thesis_reportable", 0.0),
            ("reserve_feasibility_repair_used", 0.0),
            ("infeasibility_driver", "none"),
            ("protected_soc_mode", "obligation_driven"),
            ("protected_soc_global_buffer_applied", 0.0),
            ("protected_soc_obligation_pos_active_count", 0.0),
            ("protected_soc_obligation_neg_active_count", 0.0),
            ("protected_soc_violation_without_obligation_count", 0.0),
            ("physical_soc_violation_count", 0.0),
            ("physical_soc_violation_max_mwh", 0.0),
            ("reserve_feasibility_mode", "normal"),
            ("reserve_min_margin_after_bid_mwh", 0.25),
            ("reserve_bid_derate", 1.0),
            ("max_reserve_bid_mw", float("nan")),
            ("reserve_retry_ladder", "1.0,0.5,0.25,0.0"),
            ("reserve_retry_attempts_used", 0.0),
            ("reserve_retry_final_factor", 1.0),
            ("reserve_retry_succeeded", 0.0),
            ("disable_new_bcm_reserve_bids", 0.0),
            ("new_reserve_bids_zeroed_by_retry", 0.0),
            ("reserve_retry_infeasible_after_zero_reserve", 0.0),
            ("afrr_bcm_gate_hour_cet_model", 8.0),
            ("afrr_bcm_gate_hour_cet_benchmark", 8.0),
            ("benchmark_same_rules_gate_consistent", 1.0),
            ("final_soc_mode", "terminal_repair"),
            ("benchmark_is_global_upper_bound", 0.0),
            ("rolling_perfect_foresight_same_rules_is_global_upper_bound", 0.0),
            ("rolling_perfect_foresight_same_rules_can_be_beaten", 1.0),
            ("benchmark_type", "rolling_perfect_foresight_same_rules"),
            ("rolling_pf_quantile_surface_mode", "unknown"),
            ("perfect_foresight_total_pnl_eur_is_deprecated", 1.0),
            ("perfect_foresight_total_pnl_eur_semantics", "rolling_perfect_foresight_same_rules"),
            ("perfect_foresight_total_pnl_eur_is_deprecated", 1.0),
            ("perfect_foresight_total_pnl_eur_semantics", "rolling_perfect_foresight_same_rules"),
            ("capture_ratio_vs_perfect_foresight_is_deprecated", 1.0),
            ("capture_ratio_vs_perfect_foresight_semantics", "rolling_perfect_foresight_same_rules"),
            ("capture_ratio_vs_perfect_foresight_is_deprecated", 1.0),
            ("capture_ratio_vs_perfect_foresight_semantics", "rolling_perfect_foresight_same_rules"),
            ("rolling_perfect_foresight_same_rules_total_pnl_eur", float("nan")),
            ("comparable_rolling_perfect_foresight_same_rules_market_pnl_eur", float("nan")),
            ("realized_vs_perfect_foresight_ratio_multi_market", float("nan")),
            ("realized_vs_perfect_foresight_pct", float("nan")),
            ("realized_vs_perfect_foresight_comparable_market_ratio", float("nan")),
            ("perfect_foresight_is_global_upper_bound", 0.0),
            ("perfect_foresight_can_be_beaten", 1.0),
            ("rolling_pf_is_upper_bound", 0.0),
            ("global_perfect_foresight_is_upper_bound", 0.0),
            ("global_perfect_foresight_available", 0.0),
            ("global_perfect_foresight_validation_status", "disabled_unverified"),
            ("global_perfect_foresight_market_scope", "[]"),
            ("global_perfect_foresight_dispatch_rows", 0.0),
            ("global_perfect_foresight_settlement_rows", 0.0),
            ("global_perfect_foresight_bem_only_included", 0.0),
            ("global_hindsight_perfect_foresight_upper_bound_total_pnl_eur", float("nan")),
            ("global_hindsight_perfect_foresight_upper_bound_market_pnl_eur", float("nan")),
            ("global_hindsight_perfect_foresight_is_global_upper_bound", 0.0),
            ("global_perfect_foresight_capacity_bid_semantics", "hindsight_pay_as_bid_upper_bound"),
            ("realized_minus_global_perfect_foresight_eur", float("nan")),
            ("realized_vs_global_hindsight_perfect_foresight_upper_bound_pct", float("nan")),
            ("realized_exceeds_global_perfect_foresight", 0.0),
            ("global_perfect_foresight_dominance_check_pass", 1.0),
            ("global_perfect_foresight_pnl_reconciliation_error_eur", float("nan")),
            ("perfect_foresight_total_pnl_eur", float("nan")),
            ("comparable_perfect_foresight_market_pnl_eur", float("nan")),
            ("final_soc_handling_mode", "terminal_settlement_to_target"),
            ("terminal_soc_net_adjustment_eur", 0.0),
            ("terminal_price_eur_mwh", 0.0),
            ("terminal_price_source", "last_true_da_price"),
            ("fallback_is_repair_optimization", 0.0),
            ("reserve_headroom_shortfall_pos_mwh_sum", 0.0),
            ("reserve_headroom_shortfall_neg_mwh_sum", 0.0),
            ("reserve_headroom_shortfall_max_mwh", 0.0),
            ("reserve_headroom_shortfall_penalty_eur", 0.0),
            ("reserve_headroom_shortfall_check_pass", 1.0),
            ("protected_soc_violation_count", 0.0),
            ("protected_soc_violation_max_mwh", 0.0),
            ("protected_soc_check_pass", 1.0),
            ("locked_reserve_obligation_hours", 0.0),
            ("locked_reserve_obligation_pos_mw_sum", 0.0),
            ("locked_reserve_obligation_neg_mw_sum", 0.0),
            ("infeasible_debug_dump_count", 0.0),
            ("accepted_path_infeasible_debug_dump_count", 0.0),
            ("candidate_infeasible_debug_dump_count", 0.0),
            ("infeasible_debug_dump_paths", []),
            ("infeasible_debug_dump_timestamps", []),
            ("first_infeasible_timestamp_utc", ""),
        ]:
            if k not in summary or summary[k] is None:
                summary[k] = v

        return BacktestOutputs(
            hourly=hourly,
            monthly=monthly,
            yearly=yearly,
            plan_history=plan_history,
            volatility=volatility,
            summary=summary,
            isolated_hourly={
                "realized_da_only": realized_da_only_hourly,
                "realized_afrr_only": realized_afrr_only_hourly,
                "perfect_foresight_da_only": perfect_foresight_da_only_hourly,
                "perfect_foresight_afrr_only": perfect_foresight_afrr_only_hourly,
            },
        )


def aggregate_periodic(hourly: pd.DataFrame, timestamp_col: str, freq: str) -> pd.DataFrame:
    """Aggregate hourly backtest outputs to monthly/yearly performance tables."""
    if hourly.empty:
        return pd.DataFrame()

    df = hourly.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    grouped = df.groupby(pd.Grouper(key=timestamp_col, freq=freq), dropna=True)

    agg_candidates = {
        "realized_pnl_eur": ("real_pnl_eur", "sum"),
        "naive_pnl_eur": ("naive_pnl_eur", "sum"),
        "perfect_foresight_pnl_eur": ("perfect_foresight_pnl_eur", "sum"),
        "predicted_pnl_eur": ("pred_pnl_eur", "sum"),
        "realized_penalty_eur": ("real_penalty_eur", "sum"),
        "naive_penalty_eur": ("naive_penalty_eur", "sum"),
        "perfect_foresight_penalty_eur": ("perfect_foresight_penalty_eur", "sum"),
        "pnl_gap_eur": ("pnl_gap_eur", "sum"),
        "cost_of_forecast_error_eur": ("cost_of_forecast_error_eur", "sum"),
        "realized_revenue_da_eur": ("real_revenue_da_eur", "sum"),
        "realized_cost_da_eur": ("real_cost_da_eur", "sum"),
        "realized_revenue_capacity_eur": ("real_revenue_capacity_eur", "sum"),
        "realized_revenue_activation_eur": ("real_revenue_activation_eur", "sum"),
        "realized_transaction_cost_eur": ("real_transaction_cost_eur", "sum"),
        "realized_degradation_cost_eur": ("real_degradation_cost_eur", "sum"),
        "avg_charge_mw": ("real_charge_mw", "mean"),
        "avg_discharge_mw": ("real_discharge_mw", "mean"),
        "avg_reserve_pos_mw": ("real_reserve_pos_mw", "mean"),
        "avg_reserve_neg_mw": ("real_reserve_neg_mw", "mean"),
        "avg_soc_shock_mwh": ("soc_shock_mwh", "mean"),
    }
    agg_spec = {k: v for k, v in agg_candidates.items() if v[0] in df.columns}
    out = grouped.agg(**agg_spec).reset_index()

    if "realized_pnl_eur" in out.columns:
        out["realized_pnl_cum_eur"] = out["realized_pnl_eur"].cumsum()
    if "naive_pnl_eur" in out.columns:
        out["naive_pnl_cum_eur"] = out["naive_pnl_eur"].cumsum()
    if "predicted_pnl_eur" in out.columns:
        out["predicted_pnl_cum_eur"] = out["predicted_pnl_eur"].cumsum()
    return out


def calculate_volatility(plan_history: pd.DataFrame) -> pd.DataFrame:
    """Compute decision-volatility metrics from rolling plan revisions."""
    cols = [
        "target_time_utc",
        "n_snapshots",
        "action_flips",
        "mean_abs_revision_mw",
        "final_action_class",
    ]
    if plan_history.empty:
        return pd.DataFrame(columns=cols)

    df = plan_history.copy()
    df["snapshot_time_utc"] = pd.to_datetime(df.get("snapshot_time_utc"), utc=True, errors="coerce")
    df["target_time_utc"] = pd.to_datetime(df.get("target_time_utc"), utc=True, errors="coerce")
    df = df.dropna(subset=["snapshot_time_utc", "target_time_utc"]).copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    for col in ("charge_mw", "discharge_mw", "reserve_pos_mw", "reserve_neg_mw"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["reserve_total_mw"] = df["reserve_pos_mw"] + df["reserve_neg_mw"]
    df["planned_action_mw"] = df["discharge_mw"] - df["charge_mw"]

    eps = 1e-6
    stack = np.column_stack(
        [
            df["charge_mw"].to_numpy(dtype=float),
            df["discharge_mw"].to_numpy(dtype=float),
            df["reserve_total_mw"].to_numpy(dtype=float),
        ]
    )
    argmax = np.argmax(stack, axis=1)
    labels = np.array(["charge", "discharge", "reserve"], dtype=object)
    max_vals = stack[np.arange(len(df)), argmax]
    df["action_class"] = np.where(max_vals > eps, labels[argmax], "idle")

    df = df.sort_values(["target_time_utc", "snapshot_time_utc"]).reset_index(drop=True)
    grp = df.groupby("target_time_utc", sort=False)
    df["prev_action_class"] = grp["action_class"].shift(1)
    df["prev_planned_action_mw"] = grp["planned_action_mw"].shift(1)
    df["action_flip"] = (
        df["prev_action_class"].notna()
        & (df["action_class"] != df["prev_action_class"])
    ).astype(int)
    df["abs_revision_mw"] = (df["planned_action_mw"] - df["prev_planned_action_mw"]).abs()

    out = grp.agg(
        n_snapshots=("snapshot_time_utc", "size"),
        action_flips=("action_flip", "sum"),
        mean_abs_revision_mw=("abs_revision_mw", "mean"),
        final_action_class=("action_class", "last"),
    ).reset_index()
    out["mean_abs_revision_mw"] = out["mean_abs_revision_mw"].fillna(0.0)
    return out.sort_values("target_time_utc").reset_index(drop=True)
