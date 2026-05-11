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
        da_gate_hour_cet=12,
        soc_feedback_mode="realized",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
import os
import signal
import threading
import time
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from energy_trading.config import BATTERY_SPECS, FINANCIAL_PARAMS, MARKET_SPECS, MODEL_SPECS
from energy_trading.models.train_xgboost_export import calculate_acceptance_probabilities
from energy_trading.simulation.bid_builder import AFRRCapacityBid, BidBuilder, BidPricingPolicy
from energy_trading.simulation.market_clearing import MarketClearingEngine


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


CANONICAL_PREDICTION_COLUMNS = [
    "pred_da_price",
    "pred_afrr_capacity_price_pos",
    "pred_afrr_capacity_price_neg",
    "pred_afrr_activation_price_pos",
    "pred_afrr_activation_price_neg",
    "pred_afrr_activation_rate_pos",
    "pred_afrr_activation_rate_neg",
]
QUANTILE_COLUMNS = [f"p{q:02d}" for q in range(10, 100, 10)]


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
    return merged


def load_prediction_warehouse_long(
    prediction_files: dict[str, str | Path],
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
        out = out.dropna(subset=["snapshot_time_utc", "target_time_utc", "lead_time_h"]).copy()
        out = out.sort_values(["snapshot_time_utc", "lead_time_h", "target_time_utc"]).reset_index(drop=True)
        warehouse[pred_col] = out
    if not warehouse:
        raise ValueError("No valid long-format prediction files loaded.")
    return warehouse


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

        self.aux_mwh = float(BATTERY_SPECS["aux_power_mw"]) * self.dt_h
        self.deg_eur_mwh = float(BATTERY_SPECS["degradation_cost"])
        self.trans_eur_mwh = float(FINANCIAL_PARAMS["transaction_cost_eur_per_mwh"])
        self.initial_cash = float(FINANCIAL_PARAMS["initial_cash"])
        # Penalty used for non-delivery / imbalance settlement when awarded aFRR
        # cannot be physically delivered due to SoC constraints.
        self.imbalance_penalty_eur_mwh = float(FINANCIAL_PARAMS.get("imbalance_penalty_eur_mwh", 500.0))

        self.bid_power_max_mw = float(MARKET_SPECS.get("bid_power_max_mw", self.p_max_mw))
        self.reserve_max_mw = min(self.p_max_mw, self.bid_power_max_mw)
        self.da_bid_granularity_mw = float(MARKET_SPECS.get("da_bid_granularity", 0.1))
        self.afrr_bid_granularity_mw = float(MARKET_SPECS.get("afrr_bid_granularity", 1.0))
        if self.da_bid_granularity_mw <= 0 or self.afrr_bid_granularity_mw <= 0:
            raise ValueError("Bid granularities must be > 0.")
        self.da_min_bid_size_mw = float(MARKET_SPECS.get("da_min_bid_size", self.da_bid_granularity_mw))
        self.afrr_min_bid_size_mw = float(MARKET_SPECS.get("afrr_min_bid_size", self.afrr_bid_granularity_mw))
        # aFRR bid-price bins for expected-value reserve optimization.
        self.afrr_bid_prices_eur_mwh = np.arange(50.0, 501.0, 50.0, dtype=float)
        self.da_execution_mode = str(MARKET_SPECS.get("da_execution_mode", "price_taker"))
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
            eta_in=self.eta_in,
            eta_out=self.eta_out,
            degradation_cost_eur_mwh=self.deg_eur_mwh,
            transaction_cost_eur_mwh=self.trans_eur_mwh,
            da_mode=self.da_execution_mode,
            da_arb_mode=str(MARKET_SPECS.get("da_arbitrage_mode", "limit")),
            afrr_energy_bid_strategy=str(MARKET_SPECS.get("afrr_energy_bid_strategy", "forecast")),
            link_da_to_awarded_afrr=self.da_link_to_awarded_afrr,
        )
        self.market_clearing_engine = MarketClearingEngine(
            da_mode_default=self.da_execution_mode,
        )
        # MILP runtime controls:
        # - default execution is bounded and robust for rolling simulation
        # - diagnostic mode can be enabled to inspect HiGHS solver behavior
        #   (presolve, branching, degeneracy / collinearity indicators).
        self.milp_diagnostic_mode = os.environ.get("BACKTEST_MILP_DIAG", "0") == "1"
        self.milp_time_limit_seconds = float(os.environ.get("BACKTEST_MILP_TIME_LIMIT_S", "10.0"))
        self.milp_rel_gap = float(os.environ.get("BACKTEST_MILP_REL_GAP", "1e-4"))
        self.milp_diag_time_limit_seconds = float(os.environ.get("BACKTEST_MILP_DIAG_TIME_LIMIT_S", "60.0"))

    @staticmethod
    def _clip_rate(x: np.ndarray) -> np.ndarray:
        return np.clip(np.nan_to_num(x, nan=0.0), 0.0, 1.0)

    @staticmethod
    def _finite_numeric_series(
        frame: pd.DataFrame,
        primary_col: str,
        *,
        fallback_cols: list[str] | None = None,
        default: float = 0.0,
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
        vals = vals.ffill().bfill().fillna(float(default))
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
        # - soft-constraint slacks for reserve infeasibility.
        n_r = n * n_bins
        base = 3 * n + 2 * n_r
        return {
            "ch": slice(0, n),
            "dis": slice(n, 2 * n),
            "rpos_bin": slice(2 * n, 2 * n + n_r),
            "rneg_bin": slice(2 * n + n_r, 2 * n + 2 * n_r),
            "u": slice(base, base + n),
            "soc": slice(base + n, base + n + (n + 1)),
            "slack_pos": slice(base + n + (n + 1), base + n + (n + 1) + n),
            "slack_neg": slice(base + n + (n + 1) + n, base + n + (n + 1) + 2 * n),
        }

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
        deterministic_reserve_settlement: bool = False,
        allowed_markets: list[str] | tuple[str, ...] | set[str] = ("DA", "aFRR"),
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
        n_bins = int(len(self.afrr_bid_prices_eur_mwh))
        if n_bins <= 0:
            raise ValueError("afrr_bid_prices_eur_mwh must contain at least one bid bin.")
        sl = self._variable_slices(n, n_bins=n_bins)
        n_vars = int(sl["slack_neg"].stop)
        if sl["ch"].start != 0 or sl["slack_neg"].stop <= 0:
            raise ValueError("Invalid variable slice definition for MILP.")
        allowed = {str(m).strip().lower() for m in allowed_markets}
        da_enabled = "da" in allowed
        afrr_enabled = "afrr" in allowed

        p_da = self._finite_numeric_series(
            df,
            colmap.pred_da_price,
            fallback_cols=[],
            default=0.0,
        ).to_numpy(dtype=float)
        p_cap_pos = self._finite_numeric_series(
            df,
            colmap.pred_afrr_capacity_price_pos,
            fallback_cols=[colmap.pred_afrr_capacity_price_neg],
            default=0.0,
        ).to_numpy(dtype=float)
        p_cap_neg = self._finite_numeric_series(
            df,
            colmap.pred_afrr_capacity_price_neg,
            fallback_cols=[colmap.pred_afrr_capacity_price_pos],
            default=0.0,
        ).to_numpy(dtype=float)
        # Fallback acceptance-rate priors (used when no quantile-derived p_acc bins are available).
        r_act_pos_base = self._clip_rate(
            self._finite_numeric_series(
                df,
                colmap.pred_afrr_activation_rate_pos,
                fallback_cols=[colmap.pred_afrr_activation_rate_neg],
                default=0.0,
            ).to_numpy(dtype=float)
        )
        r_act_neg_base = self._clip_rate(
            self._finite_numeric_series(
                df,
                colmap.pred_afrr_activation_rate_neg,
                fallback_cols=[colmap.pred_afrr_activation_rate_pos],
                default=0.0,
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
            ).to_numpy(dtype=float)
        )
        act_price_pos = self._finite_numeric_series(
            df,
            colmap.pred_afrr_activation_price_pos,
            fallback_cols=[],
            default=0.0,
        ).to_numpy(dtype=float)
        act_price_neg = self._finite_numeric_series(
            df,
            colmap.pred_afrr_activation_price_neg,
            fallback_cols=[],
            default=0.0,
        ).to_numpy(dtype=float)

        # Capacity acceptance probabilities by bid-price bin and hour.
        p_acc_cap_pos = np.zeros((n, n_bins), dtype=float)
        p_acc_cap_neg = np.zeros((n, n_bins), dtype=float)
        if deterministic_reserve_settlement:
            # Oracle mode keeps deterministic activation rates, but capacity
            # acceptance remains bin-specific where available.
            for b in range(n_bins):
                c_pos = f"pacc_pos_bin_{b}"
                c_neg = f"pacc_neg_bin_{b}"
                if c_pos in df.columns:
                    p_acc_cap_pos[:, b] = self._clip_rate(pd.to_numeric(df[c_pos], errors="coerce").to_numpy(dtype=float))
                else:
                    # Price-taker fallback: only most-competitive bin may clear.
                    p_acc_cap_pos[:, b] = r_act_pos_base if b == 0 else 0.0
                if c_neg in df.columns:
                    p_acc_cap_neg[:, b] = self._clip_rate(pd.to_numeric(df[c_neg], errors="coerce").to_numpy(dtype=float))
                else:
                    p_acc_cap_neg[:, b] = r_act_neg_base if b == 0 else 0.0
        else:
            for b in range(n_bins):
                c_pos = f"pacc_pos_bin_{b}"
                c_neg = f"pacc_neg_bin_{b}"
                if c_pos in df.columns:
                    p_acc_cap_pos[:, b] = self._clip_rate(pd.to_numeric(df[c_pos], errors="coerce").to_numpy(dtype=float))
                else:
                    # Fix "flat fallback" loophole:
                    # when quantile-derived p_acc is unavailable, enforce
                    # price-taker behavior by allowing acceptance only in the
                    # lowest bid-price bin.
                    p_acc_cap_pos[:, b] = r_act_pos_base if b == 0 else 0.0
                if c_neg in df.columns:
                    p_acc_cap_neg[:, b] = self._clip_rate(pd.to_numeric(df[c_neg], errors="coerce").to_numpy(dtype=float))
                else:
                    p_acc_cap_neg[:, b] = r_act_neg_base if b == 0 else 0.0

        c = np.zeros(n_vars, dtype=float)

        da_step = self.da_bid_granularity_mw
        afrr_step = self.afrr_bid_granularity_mw

        # Objective (maximize predicted margin, scipy.milp minimizes => negate coefficients).
        # Keep DA opportunity-cost structure intact.
        ch_coef = -(p_da / self.eta_in) - self.trans_eur_mwh / self.eta_in - self.deg_eur_mwh
        dis_coef = (p_da * self.eta_out) - self.trans_eur_mwh * self.eta_out - self.deg_eur_mwh
        c[sl["ch"]] = -(ch_coef * da_step)
        c[sl["dis"]] = -(dis_coef * da_step)
        for b, bid_price in enumerate(self.afrr_bid_prices_eur_mwh):
            # Expected activated-share (conditional on both capacity clearing and
            # activation-rate forecast).
            exp_act_pos = p_acc_cap_pos[:, b] * r_act_pos_base
            exp_act_neg = p_acc_cap_neg[:, b] * r_act_neg_base
            if deterministic_reserve_settlement:
                # Deterministic oracle EV equals settlement EV under perfect foresight.
                rpos_coef = (
                    p_acc_cap_pos[:, b] * p_cap_pos
                    + exp_act_pos * act_price_pos * self.eta_out
                    - self.trans_eur_mwh * exp_act_pos * self.eta_out
                    - self.deg_eur_mwh * exp_act_pos
                )
                rneg_coef = (
                    p_acc_cap_neg[:, b] * p_cap_neg
                    + exp_act_neg * act_price_neg / self.eta_in
                    - self.trans_eur_mwh * exp_act_neg / self.eta_in
                    - self.deg_eur_mwh * exp_act_neg
                )
            else:
                rpos_coef = (
                    p_acc_cap_pos[:, b] * p_cap_pos
                    + exp_act_pos * bid_price * self.eta_out
                    - self.trans_eur_mwh * exp_act_pos * self.eta_out
                    - self.deg_eur_mwh * exp_act_pos
                )
                rneg_coef = (
                    p_acc_cap_neg[:, b] * p_cap_neg
                    + exp_act_neg * bid_price / self.eta_in
                    - self.trans_eur_mwh * exp_act_neg / self.eta_in
                    - self.deg_eur_mwh * exp_act_neg
                )
            s_pos = sl["rpos_bin"].start + b * n
            s_neg = sl["rneg_bin"].start + b * n
            c[s_pos : s_pos + n] = -(rpos_coef * afrr_step)
            c[s_neg : s_neg + n] = -(rneg_coef * afrr_step)
        # Soft chance-constraint penalties (continuous non-delivery slack, €/MWh).
        c[sl["slack_pos"]] = self.imbalance_penalty_eur_mwh * self.dt_h
        c[sl["slack_neg"]] = self.imbalance_penalty_eur_mwh * self.dt_h
        # Terminal SoC opportunity value (anti end-of-horizon dumping):
        # Reward terminal inventory using a trailing DA reference with non-negative
        # floor and export efficiency scaling to approximate realizable grid value.
        # scipy.milp minimizes, so we add a negative coefficient on terminal SoC.
        tail_k = max(1, min(6, n))
        da_tail = pd.Series(p_da[-tail_k:], dtype="float64").dropna()
        da_ref = da_tail if not da_tail.empty else pd.Series(p_da, dtype="float64").dropna()
        ref_da_price = max(0.0, float(da_ref.mean()) if not da_ref.empty else 0.0)
        terminal_soc_value_eur_per_mwh = self.eta_out * ref_da_price
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
            for b in range(n_bins):
                # Base SoC drift on expected awarded-and-activated reserve.
                exp_pos = p_acc_cap_pos[t, b] * r_act_pos_base[t]
                exp_neg = p_acc_cap_neg[t, b] * r_act_neg_base[t]
                row[sl["rpos_bin"].start + b * n + t] = (exp_pos / self.eta_out) * afrr_step
                row[sl["rneg_bin"].start + b * n + t] = -self.eta_in * exp_neg * afrr_step
            a_eq.append(row)
            b_eq.append(-self.aux_mwh)

        # Initial and terminal SoC.
        row_init = np.zeros(n_vars, dtype=float)
        row_init[sl["soc"].start] = 1.0
        a_eq.append(row_init)
        b_eq.append(self.soc_init if soc_start is None else soc_start)

        if soc_end_target is not None:
            row_end = np.zeros(n_vars, dtype=float)
            row_end[sl["soc"].start + n] = 1.0
            a_eq.append(row_end)
            b_eq.append(soc_end_target)

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
            a_ub.append(row)
            b_ub.append(self.p_max_mw)

            # discharge + reserve_pos <= Pmax
            row = np.zeros(n_vars, dtype=float)
            row[sl["dis"].start + t] = da_step
            for b in range(n_bins):
                row[sl["rpos_bin"].start + b * n + t] = afrr_step
            a_ub.append(row)
            b_ub.append(self.p_max_mw)

            # reserve_pos + reserve_neg <= reserve_max
            row = np.zeros(n_vars, dtype=float)
            for b in range(n_bins):
                row[sl["rpos_bin"].start + b * n + t] = afrr_step
                row[sl["rneg_bin"].start + b * n + t] = afrr_step
            a_ub.append(row)
            b_ub.append(self.reserve_max_mw)

            # p90 chance-constraint safety bounds:
            # enforce sufficient SoC footroom/headroom for p90 activation-rate
            # on expected awarded reserve (capacity acceptance by bin).
            # pos reserve needs discharge energy from SoC:
            # soc_t - sum_b(p_acc_cap_pos[t,b] * r_act_pos_p90[t] * reserve_bin[b,t] * dt / eta_out) >= soc_min
            row = np.zeros(n_vars, dtype=float)
            row[sl["soc"].start + t] = -1.0
            for b in range(n_bins):
                chance_pos = p_acc_cap_pos[t, b] * r_act_pos_p90[t]
                row[sl["rpos_bin"].start + b * n + t] = (
                    chance_pos * self.dt_h / max(self.eta_out, 1e-12)
                ) * afrr_step
            row[sl["slack_pos"].start + t] = -1.0
            a_ub.append(row)
            b_ub.append(-self.soc_min)

            # neg reserve needs charging headroom in SoC:
            # soc_t + sum_b(p_acc_cap_neg[t,b] * r_act_neg_p90[t] * reserve_bin[b,t] * dt * eta_in) <= soc_max
            row = np.zeros(n_vars, dtype=float)
            row[sl["soc"].start + t] = 1.0
            for b in range(n_bins):
                chance_neg = p_acc_cap_neg[t, b] * r_act_neg_p90[t]
                row[sl["rneg_bin"].start + b * n + t] = (chance_neg * self.dt_h * self.eta_in) * afrr_step
            row[sl["slack_neg"].start + t] = -1.0
            a_ub.append(row)
            b_ub.append(self.soc_max)

            # Mixed-integer exclusivity (Big-M):
            # charge[t] <= is_charging[t] * Pmax
            row = np.zeros(n_vars, dtype=float)
            row[sl["ch"].start + t] = da_step
            row[sl["u"].start + t] = -self.p_max_mw
            a_ub.append(row)
            b_ub.append(0.0)

            # discharge[t] <= (1 - is_charging[t]) * Pmax
            # <=> discharge[t] + is_charging[t] * Pmax <= Pmax
            row = np.zeros(n_vars, dtype=float)
            row[sl["dis"].start + t] = da_step
            row[sl["u"].start + t] = self.p_max_mw
            a_ub.append(row)
            b_ub.append(self.p_max_mw)

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

        # Final SoC floor constraint: SoC_T >= soc_end_min_target
        if soc_end_min_target is not None:
            row = np.zeros(n_vars, dtype=float)
            row[sl["soc"].start + n] = -1.0
            a_ub.append(row)
            b_ub.append(-float(soc_end_min_target))

        lb = np.zeros(n_vars, dtype=float)
        ub = np.full(n_vars, np.inf, dtype=float)
        da_units_max = int(np.floor(self.p_max_mw / da_step + 1e-9)) if da_enabled else 0
        afrr_units_max = int(np.floor(self.reserve_max_mw / afrr_step + 1e-9)) if afrr_enabled else 0

        # Integer decision blocks.
        ub[sl["ch"]] = da_units_max
        ub[sl["dis"]] = da_units_max
        ub[sl["rpos_bin"]] = afrr_units_max
        ub[sl["rneg_bin"]] = afrr_units_max
        ub[sl["u"]] = 1.0

        # State and slack blocks.
        lb[sl["soc"]] = self.soc_min
        ub[sl["soc"]] = self.soc_max
        lb[sl["slack_pos"]] = 0.0
        lb[sl["slack_neg"]] = 0.0

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

        integrality = np.zeros(n_vars, dtype=int)
        integrality[sl["ch"]] = 1
        integrality[sl["dis"]] = 1
        integrality[sl["rpos_bin"]] = 1
        integrality[sl["rneg_bin"]] = 1
        integrality[sl["u"]] = 1

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
        out = df[[colmap.timestamp]].copy()
        out["charge_mw"] = x[sl["ch"]] * da_step
        out["discharge_mw"] = x[sl["dis"]] * da_step
        out["reserve_pos_mw"] = rpos_bin.sum(axis=1) * afrr_step
        out["reserve_neg_mw"] = rneg_bin.sum(axis=1) * afrr_step
        out["is_charging"] = x[sl["u"]]
        out["soc_lp_mwh"] = x[sl["soc"].start + 1 : sl["soc"].start + n + 1]
        out["slack_pos_mw"] = x[sl["slack_pos"]]
        out["slack_neg_mw"] = x[sl["slack_neg"]]
        out["predicted_objective_eur"] = -sol.fun
        return out

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
        id_charge = min(id_charge, max(0.0, self.p_max_mw - max(0.0, da_charge)))
        id_discharge = min(id_discharge, max(0.0, self.p_max_mw - max(0.0, da_discharge)))

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

        # Keep SoC feasible by scaling in/out streams if needed.
        delta = self.eta_in * in_internal - out_internal / self.eta_out - self.aux_mwh
        min_delta = self.soc_min - soc
        max_delta = self.soc_max - soc

        if delta < min_delta and out_internal > 0:
            excess = min_delta - delta
            scale = max(0.0, 1.0 - (excess * self.eta_out) / max(out_internal, 1e-12))
            out_internal *= scale
            da_dis_internal *= scale
            id_dis_internal *= scale
            act_pos_internal *= scale
            delta = self.eta_in * in_internal - out_internal / self.eta_out - self.aux_mwh
        if delta > max_delta and in_internal > 0:
            excess = delta - max_delta
            scale = max(0.0, 1.0 - excess / max(self.eta_in * in_internal, 1e-12))
            in_internal *= scale
            da_ch_internal *= scale
            id_ch_internal *= scale
            act_neg_internal *= scale
            delta = self.eta_in * in_internal - out_internal / self.eta_out - self.aux_mwh

        soc_next = float(np.clip(soc + delta, self.soc_min, self.soc_max))

        # Requested-vs-delivered activation energy for strict non-delivery accounting.
        act_pos_grid_req = act_pos_internal_req * self.eta_out
        act_neg_grid_req = act_neg_internal_req / self.eta_in

        # Settlement cashflows.
        da_buy_grid = da_ch_internal / self.eta_in
        da_sell_grid = da_dis_internal * self.eta_out
        id_buy_mwh = id_ch_internal / self.eta_in
        id_sell_mwh = id_dis_internal * self.eta_out
        act_pos_grid = act_pos_internal * self.eta_out
        act_neg_grid = act_neg_internal / self.eta_in

        rev_da = da_sell_grid * da_price
        cost_da = da_buy_grid * da_price
        # Synthetic ID rescue prices with EPEX technical caps.
        id_buy_price_eur_mwh = min(3000.0, float(da_price) + 30.0)
        id_sell_price_eur_mwh = max(-500.0, float(da_price) - 30.0)
        cost_id_eur = id_buy_mwh * id_buy_price_eur_mwh
        revenue_id_eur = id_sell_mwh * id_sell_price_eur_mwh
        id_trade_mwh = id_buy_mwh + id_sell_mwh
        id_slippage_cost_eur = id_trade_mwh * 30.0

        # Capacity non-delivery check on post-rescue SoC (ID already applied).
        soc_for_capacity = soc + self.eta_in * id_ch_internal - id_dis_internal / self.eta_out
        soc_for_capacity = float(np.clip(soc_for_capacity, self.soc_min, self.soc_max))
        # Inverter-aware capacity support: both SoC headroom/footroom and
        # residual converter power after DA+ID setpoints must allow reserve.
        soc_pos_limit_mw = max(
            0.0, (soc_for_capacity - self.soc_min) * self.eta_out / max(self.dt_h, 1e-12)
        )
        soc_neg_limit_mw = max(
            0.0, (self.soc_max - soc_for_capacity) / max(self.eta_in * self.dt_h, 1e-12)
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
        rev_cap = (
            delivered_capacity_pos_mw * cap_pos_settlement * self.dt_h
            + delivered_capacity_neg_mw * cap_neg_settlement * self.dt_h
        )

        # Activation revenue is paid on delivered activation energy. We still avoid
        # double counting by not adding synthetic internal energy replacement costs
        # here; replenishment economics are reflected through subsequent DA/ID trades.
        requested_activation_revenue_eur = act_pos_grid_req * act_pos_price + act_neg_grid_req * act_neg_price
        delivered_activation_revenue_eur = act_pos_grid * act_pos_price + act_neg_grid * act_neg_price
        missed_activation_revenue_eur = max(0.0, requested_activation_revenue_eur - delivered_activation_revenue_eur)
        rev_act = float(delivered_activation_revenue_eur)
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

        missed_activation_pos_mwh = max(0.0, act_pos_grid_req - act_pos_grid)
        missed_activation_neg_mwh = max(0.0, act_neg_grid_req - act_neg_grid)
        penalty_activation_pos_eur = missed_activation_pos_mwh * self.imbalance_penalty_eur_mwh
        penalty_activation_neg_eur = missed_activation_neg_mwh * self.imbalance_penalty_eur_mwh
        penalty_activation_eur = penalty_activation_pos_eur + penalty_activation_neg_eur
        # Stylized Proxy Penalty: Approximates the incentive-based settlement of
        # regelleistung.net. Uses a fixed multiplier of 2.0 and a 10 EUR/MW floor
        # as a conservative economic proxy for non-delivery exposure.
        cap_penalty_price_floor_eur_mw_h = 10.0
        penalty_capacity_pos_eur = (
            2.0 * missed_capacity_pos_mw * max(cap_penalty_price_floor_eur_mw_h, float(cap_pos)) * self.dt_h
        )
        penalty_capacity_neg_eur = (
            2.0 * missed_capacity_neg_mw * max(cap_penalty_price_floor_eur_mw_h, float(cap_neg)) * self.dt_h
        )
        penalty_capacity_eur = penalty_capacity_pos_eur + penalty_capacity_neg_eur
        penalty_eur = penalty_activation_eur + penalty_capacity_eur

        # Cashflow excludes non-cash degradation accounting.
        net_cashflow_eur = rev_da - cost_da + rev_cap + rev_act + revenue_id_eur - cost_id_eur - trans_cost - penalty_eur
        pnl = rev_da - cost_da + rev_cap + rev_act + revenue_id_eur - cost_id_eur - trans_cost - degr_cost - penalty_eur

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
            "revenue_id_eur": revenue_id_eur,
            "cost_id_eur": cost_id_eur,
            "id_slippage_cost_eur": id_slippage_cost_eur,
            "soc_for_capacity_mwh": soc_for_capacity,
            "penalty_activation_pos_eur": penalty_activation_pos_eur,
            "penalty_activation_neg_eur": penalty_activation_neg_eur,
            "penalty_activation_eur": penalty_activation_eur,
            "penalty_capacity_pos_eur": penalty_capacity_pos_eur,
            "penalty_capacity_neg_eur": penalty_capacity_neg_eur,
            "penalty_capacity_eur": penalty_capacity_eur,
            "penalty_eur": penalty_eur,
            "net_cashflow_eur": net_cashflow_eur,
            "pnl_eur": pnl,
        }
        return soc_next, metrics

    def _plan_id_rescue_for_next_hour(
        self,
        *,
        soc_next: float,
        reserve_pos_next_mw: float,
        reserve_neg_next_mw: float,
        da_charge_next_mw: float,
        da_discharge_next_mw: float,
    ) -> tuple[float, float]:
        """Compute ID rescue setpoints decided at t for execution in t+1."""
        id_charge = 0.0
        id_discharge = 0.0
        req_soc_min_for_pos = self.soc_min + (max(0.0, reserve_pos_next_mw) * self.dt_h) / max(self.eta_out, 1e-12)
        req_soc_max_for_neg = self.soc_max - (max(0.0, reserve_neg_next_mw) * self.dt_h) * self.eta_in
        if soc_next < req_soc_min_for_pos:
            soc_gap_up = req_soc_min_for_pos - soc_next
            needed_internal_charge_mwh = soc_gap_up / max(self.eta_in, 1e-12)
            needed_charge_mw = needed_internal_charge_mwh / max(self.dt_h, 1e-12)
            id_charge = max(0.0, min(needed_charge_mw, self.p_max_mw - max(0.0, da_charge_next_mw)))
        elif soc_next > req_soc_max_for_neg:
            soc_gap_down = soc_next - req_soc_max_for_neg
            needed_internal_discharge_mwh = soc_gap_down * self.eta_out
            needed_discharge_mw = needed_internal_discharge_mwh / max(self.dt_h, 1e-12)
            id_discharge = max(0.0, min(needed_discharge_mw, self.p_max_mw - max(0.0, da_discharge_next_mw)))
        return float(id_charge), float(id_discharge)

    def _apply_market_clearing(
        self,
        *,
        target_time_utc: pd.Timestamp | None = None,
        is_oracle: bool = False,
        planned_charge_mw: float,
        planned_discharge_mw: float,
        planned_reserve_pos_mw: float,
        planned_reserve_neg_mw: float,
        pred_da_price: float,
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
    ) -> dict[str, float]:
        """Sequential market clearing: aFRR capacity -> DA -> aFRR activation."""
        ch_plan, dis_plan = self._normalize_da_bid(planned_charge_mw, planned_discharge_mw)
        res_pos_plan = max(0.0, float(planned_reserve_pos_mw))
        res_neg_plan = max(0.0, float(planned_reserve_neg_mw))
        ts = pd.to_datetime(target_time_utc, utc=True, errors="coerce")
        if pd.isna(ts):
            ts = pd.Timestamp("1970-01-01T00:00:00Z")

        ob_pos = max(0.0, float(obligation_pos_mw))
        ob_neg = max(0.0, float(obligation_neg_mw))
        soc_ref = float(self.soc_init if soc_now is None else soc_now)
        if ob_pos > 0.0 or ob_neg > 0.0:
            # Capacity already auctioned at D-1 09:00 CET: use mandatory obligation.
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
                            is_oracle=is_oracle,
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
                            is_oracle=is_oracle,
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
            cap_bids = self.bid_builder.build_afrr_capacity_bids(
                ts=ts,
                reserve_pos_mw=res_pos_plan,
                reserve_neg_mw=res_neg_plan,
                pred_cap_pos=float(pred_cap_pos),
                pred_cap_neg=float(pred_cap_neg),
                pred_act_pos=float(pred_act_pos),
                pred_act_neg=float(pred_act_neg),
                is_oracle=is_oracle,
            )
            cap_res = self.market_clearing_engine.clear_afrr_capacity(
                cap_bids,
                true_cap_pos=float(true_cap_pos),
                true_cap_neg=float(true_cap_neg),
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
                        is_oracle=is_oracle,
                        true_act_price=float(true_act_pos if b.side == "pos" else true_act_neg),
                    ),
                )
                for b in cap_bids
            ]
        da_bids = self.bid_builder.build_da_bids_from_plan(
            ts=ts,
            planned_charge_mw=ch_plan,
            planned_discharge_mw=dis_plan,
            obligation_pos_mw=ob_pos if ob_pos > 0.0 else cap_res.awarded_pos_mw,
            obligation_neg_mw=ob_neg if ob_neg > 0.0 else cap_res.awarded_neg_mw,
            pred_da_price=float(pred_da_price),
            is_oracle=is_oracle,
        )
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
        act_res = self.market_clearing_engine.clear_afrr_activation(
            cap_bids,
            cap_res,
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
        res_pos_exec = float(cap_res.awarded_pos_mw)
        res_neg_exec = float(cap_res.awarded_neg_mw)
        cap_bid_pos_settlement = _weighted_awarded_bid_price(
            side="pos",
            awarded_mw=res_pos_exec,
            true_cap_price=float(true_cap_pos),
        )
        cap_bid_neg_settlement = _weighted_awarded_bid_price(
            side="neg",
            awarded_mw=res_neg_exec,
            true_cap_price=float(true_cap_neg),
        )
        rate_pos_exec = float(act_res.executed_rate_pos)
        rate_neg_exec = float(act_res.executed_rate_neg)
        da_buy_accepted = bool(da_res.buy_accepted)
        da_sell_accepted = bool(da_res.sell_accepted)
        cap_pos_awarded = bool(cap_res.pos_awarded)
        cap_neg_awarded = bool(cap_res.neg_awarded)
        act_pos_accepted = bool(act_res.pos_accepted)
        act_neg_accepted = bool(act_res.neg_accepted)
        energy_prices = [
            float(v)
            for v in (obligation_energy_pos, obligation_energy_neg)
            if v is not None and np.isfinite(v)
        ]
        mean_energy_price = float(np.mean(energy_prices)) if energy_prices else float("nan")

        return {
            "planned_charge_mw": ch_plan,
            "planned_discharge_mw": dis_plan,
            "planned_reserve_pos_mw": res_pos_plan,
            "planned_reserve_neg_mw": res_neg_plan,
            "submitted_da_buy_mw": float(da_res.submitted_buy_mw),
            "submitted_da_sell_mw": float(da_res.submitted_sell_mw),
            "submitted_afrr_pos_mw": float(cap_res.submitted_pos_mw),
            "submitted_afrr_neg_mw": float(cap_res.submitted_neg_mw),
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
            "da_buy_reason": 1.0 if da_res.reason_buy == "price_taker" else 2.0 if da_res.reason_buy == "limit_cleared" else 0.0,
            "da_sell_reason": 1.0 if da_res.reason_sell == "price_taker" else 2.0 if da_res.reason_sell == "limit_cleared" else 0.0,
            "aFRR_Capacity_Won_MW": float(max(ob_pos, ob_neg, res_pos_exec, res_neg_exec)),
            "DA_Energy_Sold_MW": float(dis_exec),
            "aFRR_Energy_Price_EUR_MWh": mean_energy_price,
            "Obligation_Fulfilled": float((ob_pos <= cap_res.submitted_pos_mw + 1e-9) and (ob_neg <= cap_res.submitted_neg_mw + 1e-9)),
            "aFRR_Energy_Gate_Closure_Min": 25.0,
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
        is_oracle: bool = False,
    ) -> dict[str, float]:
        # Gate closure for aFRR capacity auction: D-1 09:00 CET.
        if not self._is_gate_hour_cet(snapshot_ts, 9):
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
            if offered_pos <= 0.0 and offered_neg <= 0.0:
                for ts in blk["target_time_utc"]:
                    lock_pos[pd.to_datetime(ts, utc=True)] = 0.0
                    lock_neg[pd.to_datetime(ts, utc=True)] = 0.0
                continue

            # Stage 1 (forecast-only): formulate submitted capacity bids.
            cap_bids, ts_idx = self._formulate_afrr_capacity_block_bids(
                blk=blk,
                source=source,
                colmap=colmap,
                snapshot_ts=snapshot_ts,
                offered_pos=offered_pos,
                offered_neg=offered_neg,
                is_oracle=is_oracle,
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
        is_oracle: bool = False,
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
            is_oracle=is_oracle,
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
        da_gate_hour_cet: int = 12,
        soc_feedback_mode: str = "realized",
        enforce_final_soc_min: bool = True,
        deterministic_reserve_settlement: bool = False,
        is_oracle: bool = False,
        allowed_markets: list[str] | tuple[str, ...] | set[str] = ("DA", "aFRR"),
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Rolling-horizon LP (re-optimized repeatedly with SoC state carryover)."""
        if horizon_hours <= 0 or reopt_step_hours <= 0:
            raise ValueError("horizon_hours and reopt_step_hours must be > 0")
        if soc_feedback_mode not in {"realized", "predicted"}:
            raise ValueError("soc_feedback_mode must be one of {'realized', 'predicted'}")
        self._assert_valid_time_index(df, colmap.timestamp)
        allowed = {str(m).strip().lower() for m in allowed_markets}
        da_enabled = "da" in allowed
        afrr_enabled = "afrr" in allowed
        if df.empty:
            empty_dispatch = pd.DataFrame(
                columns=[
                    colmap.timestamp,
                    "charge_mw",
                    "discharge_mw",
                    "reserve_pos_mw",
                    "reserve_neg_mw",
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
        # ID rescue decided at t and executed at t+1 (one-hour execution lag).
        pending_id_charge_mw = 0.0
        pending_id_discharge_mw = 0.0
        reopt_restart_done: set[pd.Timestamp] = set()
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

                for pred_col in CANONICAL_PREDICTION_COLUMNS:
                    if pred_col not in window.columns:
                        window[pred_col] = np.nan
                    if pred_col not in forecast_warehouse:
                        continue
                    src_df = forecast_warehouse[pred_col]
                    sub = src_df[src_df["snapshot_time_utc"] == snapshot_ts]
                    if sub.empty:
                        continue
                    pred_map = (
                        sub.drop_duplicates(subset=["target_time_utc"])
                        .set_index("target_time_utc")["predicted_value"]
                    )
                    filled = target_times.map(pred_map)
                    window[pred_col] = pd.to_numeric(filled, errors="coerce").to_numpy(dtype=float)

                    # Fallback to wide per-target timestamp predictions when warehouse has gaps.
                    if pred_col in source.columns:
                        fallback = pd.to_numeric(window[pred_col], errors="coerce")
                        miss = fallback.isna()
                        if bool(miss.any()):
                            fb_vals = pd.to_numeric(window.loc[miss, colmap.timestamp].map(source[pred_col]), errors="coerce")
                            window.loc[miss, pred_col] = fb_vals.to_numpy()

                    # Build quantile->CDF acceptance bridge for aFRR CAPACITY prices.
                    # These probabilities are interpreted as capacity-market
                    # acceptance by bid-price bin.
                    if pred_col in {colmap.pred_afrr_capacity_price_pos, colmap.pred_afrr_capacity_price_neg}:
                        q_cols = [c for c in QUANTILE_COLUMNS if c in sub.columns]
                        if q_cols:
                            sub_u = (
                                sub.drop_duplicates(subset=["target_time_utc"])
                                .set_index("target_time_utc")
                                .sort_index()
                            )
                            for b, bid_price in enumerate(self.afrr_bid_prices_eur_mwh):
                                p_acc_vals: list[float] = []
                                for ts in target_times:
                                    if ts not in sub_u.index:
                                        p_acc_vals.append(np.nan)
                                        continue
                                    row = sub_u.loc[ts]
                                    if isinstance(row, pd.DataFrame):
                                        row = row.iloc[0]
                                    qmap = self._quantile_map_from_row(row)
                                    if len(qmap) < 2:
                                        p_acc_vals.append(np.nan)
                                        continue
                                    pacc_df = calculate_acceptance_probabilities(
                                        quantiles=qmap,
                                        price_bins=[float(bid_price)],
                                    )
                                    p_acc_vals.append(float(pacc_df["p_acc"].iloc[0]))
                                side = "pos" if pred_col == colmap.pred_afrr_capacity_price_pos else "neg"
                                col = f"pacc_{side}_bin_{b}"
                                window[col] = pd.to_numeric(pd.Series(p_acc_vals), errors="coerce").to_numpy(dtype=float)
                    # Propagate activation-rate quantiles into window columns
                    # for p90 chance constraints in optimize_dispatch().
                    if pred_col in {colmap.pred_afrr_activation_rate_pos, colmap.pred_afrr_activation_rate_neg}:
                        q_cols = [c for c in QUANTILE_COLUMNS if c in sub.columns]
                        if q_cols:
                            sub_u = (
                                sub.drop_duplicates(subset=["target_time_utc"])
                                .set_index("target_time_utc")
                                .sort_index()
                            )
                            base = colmap.pred_afrr_activation_rate_pos if pred_col == colmap.pred_afrr_activation_rate_pos else colmap.pred_afrr_activation_rate_neg
                            for qc in q_cols:
                                q_map = sub_u[qc]
                                filled_q = target_times.map(q_map)
                                window[f"{base}_{qc}"] = pd.to_numeric(filled_q, errors="coerce").to_numpy(dtype=float)

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
                for pred_col, fallbacks in pred_fallbacks.items():
                    if pred_col not in window.columns:
                        window[pred_col] = np.nan
                    window[pred_col] = self._finite_numeric_series(
                        window,
                        pred_col,
                        fallback_cols=[c for c in fallbacks if c in window.columns],
                        default=0.0,
                    ).to_numpy(dtype=float)

                # Ensure every p_acc bin has a robust fallback.
                # Price-taker fallback policy: when no quantile-derived p_acc is
                # available, only the most-competitive bin (b=0) carries the
                # scalar prior; higher bins get 0 probability.
                base_pos = self._clip_rate(
                    pd.to_numeric(window[colmap.pred_afrr_activation_rate_pos], errors="coerce").to_numpy(dtype=float)
                )
                base_neg = self._clip_rate(
                    pd.to_numeric(window[colmap.pred_afrr_activation_rate_neg], errors="coerce").to_numpy(dtype=float)
                )
                for b in range(len(self.afrr_bid_prices_eur_mwh)):
                    c_pos = f"pacc_pos_bin_{b}"
                    c_neg = f"pacc_neg_bin_{b}"
                    if c_pos not in window.columns:
                        window[c_pos] = base_pos if b == 0 else np.zeros_like(base_pos)
                    else:
                        window[c_pos] = (
                            pd.to_numeric(window[c_pos], errors="coerce")
                            .fillna(pd.Series(base_pos if b == 0 else np.zeros_like(base_pos), index=window.index))
                            .to_numpy(dtype=float)
                        )
                    if c_neg not in window.columns:
                        window[c_neg] = base_neg if b == 0 else np.zeros_like(base_neg)
                    else:
                        window[c_neg] = (
                            pd.to_numeric(window[c_neg], errors="coerce")
                            .fillna(pd.Series(base_neg if b == 0 else np.zeros_like(base_neg), index=window.index))
                            .to_numpy(dtype=float)
                        )
            else:
                w_end = min(n, i + horizon_hours)
                window = df.iloc[i:w_end].copy()

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
            try:
                plan = self.optimize_dispatch(
                    window,
                    colmap,
                    soc_start=soc,
                    soc_end_target=enforce_end,
                    soc_end_min_target=enforce_end_min,
                    fixed_da_dispatch=da_lockbook,
                    deterministic_reserve_settlement=deterministic_reserve_settlement,
                    allowed_markets=allowed_markets,
                )
            except RuntimeError as exc:
                msg = str(exc)
                if "infeasible" not in msg.lower():
                    raise
                optimization_error = msg
                if enforce_end_min is not None:
                    # Graceful fallback for the final rolling window: if the
                    # terminal SoC floor makes the MILP infeasible (e.g. due to
                    # fixed DA bids), re-optimize without terminal floor.
                    try:
                        plan = self.optimize_dispatch(
                            window,
                            colmap,
                            soc_start=soc,
                            soc_end_target=enforce_end,
                            soc_end_min_target=None,
                            fixed_da_dispatch=da_lockbook,
                            deterministic_reserve_settlement=deterministic_reserve_settlement,
                            allowed_markets=allowed_markets,
                        )
                        optimization_fallback = "relaxed_final_soc_min"
                    except RuntimeError as exc_relaxed:
                        if "infeasible" not in str(exc_relaxed).lower():
                            raise
                        # If even the relaxed-final-SoC solve is infeasible, keep
                        # simulation robust by falling back to hold plan.
                        ts_vals = pd.to_datetime(window[colmap.timestamp], utc=True, errors="coerce")
                        hold = pd.DataFrame(
                            {
                                colmap.timestamp: ts_vals,
                                "charge_mw": np.zeros(len(window), dtype=float),
                                "discharge_mw": np.zeros(len(window), dtype=float),
                                "reserve_pos_mw": np.zeros(len(window), dtype=float),
                                "reserve_neg_mw": np.zeros(len(window), dtype=float),
                                "soc_lp_mwh": np.full(len(window), float(soc), dtype=float),
                            },
                            index=window.index,
                        )
                        for b in range(len(self.afrr_bid_prices_eur_mwh)):
                            hold[f"reserve_pos_bin_{b}_mw"] = 0.0
                            hold[f"reserve_neg_bin_{b}_mw"] = 0.0
                        plan = hold
                        optimization_fallback = "safe_hold_plan_after_relaxed_failure"
                else:
                    # Last-resort fallback: if rolling MILP is infeasible even
                    # without terminal floor, degrade to a physically safe hold
                    # plan for this window so the simulation can proceed.
                    # This avoids full-run aborts from local pathological windows.
                    ts_vals = pd.to_datetime(window[colmap.timestamp], utc=True, errors="coerce")
                    hold = pd.DataFrame(
                        {
                            # Keep timezone-aware timestamps to avoid merge dtype
                            # mismatches (datetime64[ns, UTC] vs naive datetime64).
                            colmap.timestamp: ts_vals,
                            "charge_mw": np.zeros(len(window), dtype=float),
                            "discharge_mw": np.zeros(len(window), dtype=float),
                            "reserve_pos_mw": np.zeros(len(window), dtype=float),
                            "reserve_neg_mw": np.zeros(len(window), dtype=float),
                            "soc_lp_mwh": np.full(len(window), float(soc), dtype=float),
                        },
                        index=window.index,
                    )
                    for b in range(len(self.afrr_bid_prices_eur_mwh)):
                        hold[f"reserve_pos_bin_{b}_mw"] = 0.0
                        hold[f"reserve_neg_bin_{b}_mw"] = 0.0
                    plan = hold
                    optimization_fallback = "safe_hold_plan"

            snapshot_plan = plan.copy()
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
            snapshot_plan["optimization_error"] = optimization_error
            plan_history.append(snapshot_plan)

            # Phase 1 (D-1 09:00 CET): clear aFRR capacity in 4h blocks and
            # propagate awarded obligations to delivery intervals.
            if afrr_enabled:
                cap_gate_stats = self._update_afrr_capacity_lockbooks_from_snapshot(
                    snapshot_ts=pd.to_datetime(snapshot_plan["snapshot_time_utc"].iloc[0], utc=True, errors="coerce"),
                    snapshot_plan=snapshot_plan,
                    source=source,
                    colmap=colmap,
                    lock_pos=afrr_cap_pos_lockbook,
                    lock_neg=afrr_cap_neg_lockbook,
                    lock_energy_pos=afrr_energy_pos_lockbook,
                    lock_energy_neg=afrr_energy_neg_lockbook,
                    is_oracle=is_oracle,
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
            tsu_take = pd.to_datetime(take[colmap.timestamp], utc=True, errors="coerce")
            take["aFRR_Capacity_Won_Pos_MW"] = tsu_take.map(lambda ts: float(afrr_cap_pos_lockbook.get(ts, 0.0)))
            take["aFRR_Capacity_Won_Neg_MW"] = tsu_take.map(lambda ts: float(afrr_cap_neg_lockbook.get(ts, 0.0)))
            take["aFRR_Capacity_Won_MW"] = take[["aFRR_Capacity_Won_Pos_MW", "aFRR_Capacity_Won_Neg_MW"]].max(axis=1)
            take["aFRR_Energy_Price_EUR_MWh_Pos"] = tsu_take.map(lambda ts: float(afrr_energy_pos_lockbook.get(ts, np.nan)))
            take["aFRR_Energy_Price_EUR_MWh_Neg"] = tsu_take.map(lambda ts: float(afrr_energy_neg_lockbook.get(ts, np.nan)))
            take["event_reopt_triggered"] = float(cap_gate_triggered)
            take["event_reopt_rejected_mw_total"] = float(cap_gate_stats.get("rejected_mw_total", 0.0))
            decisions.append(take)

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
                def _sf(series: pd.Series, key: str, default: float = 0.0) -> float:
                    try:
                        v = pd.to_numeric(pd.Series([series.get(key, default)]), errors="coerce").iloc[0]
                        return float(default if pd.isna(v) else v)
                    except Exception:
                        return float(default)

                charge_plan = float(getattr(r, "charge_mw"))
                discharge_plan = float(getattr(r, "discharge_mw"))
                reserve_pos_plan = float(getattr(r, "reserve_pos_mw"))
                reserve_neg_plan = float(getattr(r, "reserve_neg_mw"))

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

                cleared = self._apply_market_clearing(
                    target_time_utc=pd.to_datetime(ts, utc=True, errors="coerce"),
                    is_oracle=is_oracle,
                    planned_charge_mw=charge_plan,
                    planned_discharge_mw=discharge_plan,
                    planned_reserve_pos_mw=reserve_pos_plan,
                    planned_reserve_neg_mw=reserve_neg_plan,
                    pred_da_price=_sf(src, colmap.pred_da_price, 0.0),
                    true_da_price=_sf(src, colmap.true_da_price, 0.0),
                    pred_cap_pos=_sf(src, colmap.pred_afrr_capacity_price_pos, 0.0),
                    true_cap_pos=_sf(src, colmap.true_afrr_capacity_price_pos, 0.0),
                    pred_cap_neg=_sf(src, colmap.pred_afrr_capacity_price_neg, 0.0),
                    true_cap_neg=_sf(src, colmap.true_afrr_capacity_price_neg, 0.0),
                    pred_act_pos=_sf(src, colmap.pred_afrr_activation_price_pos, 0.0),
                    true_act_pos=_sf(src, colmap.true_afrr_activation_price_pos, 0.0),
                    pred_act_neg=_sf(src, colmap.pred_afrr_activation_price_neg, 0.0),
                    true_act_neg=_sf(src, colmap.true_afrr_activation_price_neg, 0.0),
                    true_rate_pos=_sf(src, colmap.true_afrr_activation_rate_pos, 0.0),
                    true_rate_neg=_sf(src, colmap.true_afrr_activation_rate_neg, 0.0),
                    pred_rate_pos=_sf(src, colmap.pred_afrr_activation_rate_pos, 0.0),
                    pred_rate_neg=_sf(src, colmap.pred_afrr_activation_rate_neg, 0.0),
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
                )
                clearing_records[-1]["pending_id_charge_mw"] = float(pending_id_charge_mw)
                clearing_records[-1]["pending_id_discharge_mw"] = float(pending_id_discharge_mw)
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
                if next_idx < len(take):
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
                    pending_id_charge_mw, pending_id_discharge_mw = self._plan_id_rescue_for_next_hour(
                        soc_next=float(executed_s),
                        reserve_pos_next_mw=reserve_pos_next,
                        reserve_neg_next_mw=reserve_neg_next,
                        da_charge_next_mw=next_da_charge,
                        da_discharge_next_mw=next_da_discharge,
                    )
                else:
                    pending_id_charge_mw = 0.0
                    pending_id_discharge_mw = 0.0
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
                for c in clr_df.columns:
                    take[c] = clr_df[c]

        if not decisions:
            empty_dispatch = pd.DataFrame(
                columns=[
                    colmap.timestamp,
                    "charge_mw",
                    "discharge_mw",
                    "reserve_pos_mw",
                    "reserve_neg_mw",
                    "id_charge_mw",
                    "id_discharge_mw",
                    "pending_id_charge_mw",
                    "pending_id_discharge_mw",
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
                    "charge_mw",
                    "discharge_mw",
                    "reserve_pos_mw",
                    "reserve_neg_mw",
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
        oracle_mode: bool = False,
    ) -> pd.DataFrame:
        """Settle dispatch against either predicted or realized market values."""
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

        dispatch_cols = [
            colmap.timestamp,
            "charge_mw",
            "discharge_mw",
            "reserve_pos_mw",
            "reserve_neg_mw",
            "id_charge_mw",
            "id_discharge_mw",
            "pending_id_charge_mw",
            "pending_id_discharge_mw",
        ]
        dispatch_meta_cols = [
            "aFRR_Capacity_Won_Pos_MW",
            "aFRR_Capacity_Won_Neg_MW",
            "aFRR_Capacity_Won_MW",
            "aFRR_Energy_Price_EUR_MWh_Pos",
            "aFRR_Energy_Price_EUR_MWh_Neg",
            "shock_source",
            "soc_before_mwh",
            "soc_after_planned_mwh",
            "soc_after_executed_mwh",
        ]
        dispatch_clearing_cols = [
            "planned_charge_mw",
            "planned_discharge_mw",
            "planned_reserve_pos_mw",
            "planned_reserve_neg_mw",
            "submitted_da_buy_mw",
            "submitted_da_sell_mw",
            "submitted_afrr_pos_mw",
            "submitted_afrr_neg_mw",
            "executed_charge_mw",
            "executed_discharge_mw",
            "executed_reserve_pos_mw",
            "executed_reserve_neg_mw",
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
            "da_buy_reason",
            "da_sell_reason",
            "aFRR_Capacity_Won_MW",
            "DA_Energy_Sold_MW",
            "aFRR_Energy_Price_EUR_MWh",
            "Obligation_Fulfilled",
            "aFRR_Energy_Gate_Closure_Min",
            "id_charge_mw",
            "id_discharge_mw",
            "pending_id_charge_mw",
            "pending_id_discharge_mw",
        ]
        dispatch_cols = dispatch_cols + [c for c in dispatch_meta_cols if c in dispatch.columns]
        dispatch_cols = dispatch_cols + [c for c in dispatch_clearing_cols if c in dispatch.columns]

        # Backward compatibility: fallback/legacy dispatch paths may not include
        # all expected execution columns; synthesize safe defaults.
        required_dispatch_defaults: dict[str, float] = {
            colmap.timestamp: np.nan,
            "charge_mw": 0.0,
            "discharge_mw": 0.0,
            "reserve_pos_mw": 0.0,
            "reserve_neg_mw": 0.0,
            "id_charge_mw": 0.0,
            "id_discharge_mw": 0.0,
            "pending_id_charge_mw": 0.0,
            "pending_id_discharge_mw": 0.0,
        }
        for c, default_val in required_dispatch_defaults.items():
            if c not in dispatch.columns:
                dispatch[c] = default_val

        if predicted_settlement and all(c in dispatch.columns for c in [da_col, cap_pos_col, cap_neg_col, act_pos_col, act_neg_col, rate_pos_col, rate_neg_col]):
            merged = dispatch[
                dispatch_cols + [da_col, cap_pos_col, cap_neg_col, act_pos_col, act_neg_col, rate_pos_col, rate_neg_col]
            ].copy()
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
                        colmap.pred_afrr_capacity_price_pos,
                        colmap.pred_afrr_capacity_price_neg,
                        colmap.pred_afrr_activation_price_pos,
                        colmap.pred_afrr_activation_price_neg,
                    ]
                )
            base_cols = [c for c in dict.fromkeys(base_cols) if c in df.columns]
            merged = df[base_cols].copy()
            merged = merged.merge(dispatch[dispatch_cols], on=colmap.timestamp, how="inner")

        soc = self.soc_init
        rows: list[dict[str, float | pd.Timestamp]] = []
        for r in merged.itertuples(index=False):
            ts = getattr(r, colmap.timestamp)
            def _g(name: str, default: float = 0.0) -> float:
                try:
                    return float(getattr(r, name))
                except Exception:
                    return float(default)
            charge = float(getattr(r, "charge_mw"))
            discharge = float(getattr(r, "discharge_mw"))
            reserve_pos = float(getattr(r, "reserve_pos_mw"))
            reserve_neg = float(getattr(r, "reserve_neg_mw"))
            id_charge = _g("id_charge_mw", _g("pending_id_charge_mw", 0.0))
            id_discharge = _g("id_discharge_mw", _g("pending_id_discharge_mw", 0.0))
            rate_pos = float(getattr(r, rate_pos_col))
            rate_neg = float(getattr(r, rate_neg_col))

            clearing_rec: dict[str, float] = {}
            if apply_market_clearing and not predicted_settlement:
                has_precleared = all(c in merged.columns for c in [
                    "executed_charge_mw",
                    "executed_discharge_mw",
                    "executed_reserve_pos_mw",
                    "executed_reserve_neg_mw",
                    "executed_rate_pos",
                    "executed_rate_neg",
                ])
                if has_precleared:
                    charge = _g("executed_charge_mw", charge)
                    discharge = _g("executed_discharge_mw", discharge)
                    reserve_pos = _g("executed_reserve_pos_mw", reserve_pos)
                    reserve_neg = _g("executed_reserve_neg_mw", reserve_neg)
                    rate_pos = _g("executed_rate_pos", rate_pos)
                    rate_neg = _g("executed_rate_neg", rate_neg)
                    id_charge = _g("id_charge_mw", _g("pending_id_charge_mw", id_charge))
                    id_discharge = _g("id_discharge_mw", _g("pending_id_discharge_mw", id_discharge))
                    for c in dispatch_clearing_cols:
                        if c in merged.columns:
                            clearing_rec[c] = _g(c, 0.0)
                else:
                    cleared = self._apply_market_clearing(
                        target_time_utc=pd.to_datetime(ts, utc=True, errors="coerce"),
                        is_oracle=oracle_mode,
                        planned_charge_mw=charge,
                        planned_discharge_mw=discharge,
                        planned_reserve_pos_mw=reserve_pos,
                        planned_reserve_neg_mw=reserve_neg,
                        pred_da_price=_g(colmap.pred_da_price, 0.0),
                        true_da_price=_g(colmap.true_da_price, 0.0),
                        pred_cap_pos=_g(colmap.pred_afrr_capacity_price_pos, 0.0),
                        true_cap_pos=_g(colmap.true_afrr_capacity_price_pos, 0.0),
                        pred_cap_neg=_g(colmap.pred_afrr_capacity_price_neg, 0.0),
                        true_cap_neg=_g(colmap.true_afrr_capacity_price_neg, 0.0),
                        pred_act_pos=_g(colmap.pred_afrr_activation_price_pos, 0.0),
                        true_act_pos=_g(colmap.true_afrr_activation_price_pos, 0.0),
                        pred_act_neg=_g(colmap.pred_afrr_activation_price_neg, 0.0),
                        true_act_neg=_g(colmap.true_afrr_activation_price_neg, 0.0),
                        true_rate_pos=_g(colmap.true_afrr_activation_rate_pos, 0.0),
                        true_rate_neg=_g(colmap.true_afrr_activation_rate_neg, 0.0),
                        pred_rate_pos=_g(colmap.pred_afrr_activation_rate_pos, 0.0),
                        pred_rate_neg=_g(colmap.pred_afrr_activation_rate_neg, 0.0),
                        soc_now=float(soc),
                        pred_act_pos_q10=_g("pred_afrr_activation_price_pos_p10", np.nan),
                        pred_act_pos_q50=_g("pred_afrr_activation_price_pos_p50", np.nan),
                        pred_act_pos_q90=_g("pred_afrr_activation_price_pos_p90", np.nan),
                        pred_act_neg_q10=_g("pred_afrr_activation_price_neg_p10", np.nan),
                        pred_act_neg_q50=_g("pred_afrr_activation_price_neg_p50", np.nan),
                        pred_act_neg_q90=_g("pred_afrr_activation_price_neg_p90", np.nan),
                        obligation_pos_mw=_g("aFRR_Capacity_Won_Pos_MW", 0.0),
                        obligation_neg_mw=_g("aFRR_Capacity_Won_Neg_MW", 0.0),
                        obligation_energy_pos=_g("aFRR_Energy_Price_EUR_MWh_Pos", np.nan),
                        obligation_energy_neg=_g("aFRR_Energy_Price_EUR_MWh_Neg", np.nan),
                    )
                    charge = float(cleared["executed_charge_mw"])
                    discharge = float(cleared["executed_discharge_mw"])
                    reserve_pos = float(cleared["executed_reserve_pos_mw"])
                    reserve_neg = float(cleared["executed_reserve_neg_mw"])
                    rate_pos = float(cleared["executed_rate_pos"])
                    rate_neg = float(cleared["executed_rate_neg"])
                    clearing_rec = cleared

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
            rec = {colmap.timestamp: ts, **m, **clearing_rec}
            rows.append(rec)

        out = pd.DataFrame(rows)
        aux_clearing_cols = [
            "planned_charge_mw",
            "planned_discharge_mw",
            "planned_reserve_pos_mw",
            "planned_reserve_neg_mw",
            "submitted_da_buy_mw",
            "submitted_da_sell_mw",
            "submitted_afrr_pos_mw",
            "submitted_afrr_neg_mw",
            "executed_charge_mw",
            "executed_discharge_mw",
            "executed_reserve_pos_mw",
            "executed_reserve_neg_mw",
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
            "da_buy_reason",
            "da_sell_reason",
            "aFRR_Capacity_Won_MW",
            "DA_Energy_Sold_MW",
            "aFRR_Energy_Price_EUR_MWh",
            "Obligation_Fulfilled",
            "aFRR_Energy_Gate_Closure_Min",
        ]
        out.rename(
            columns={c: f"{kind}_{c}" for c in aux_clearing_cols if c in out.columns},
            inplace=True,
        )
        out.rename(columns={
            "pnl_eur": f"{kind}_pnl_eur",
            "da_buy_mwh": f"{kind}_da_buy_mwh",
            "da_sell_mwh": f"{kind}_da_sell_mwh",
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
            "transaction_cost_eur": f"{kind}_transaction_cost_eur",
            "degradation_cost_eur": f"{kind}_degradation_cost_eur",
            "missed_activation_mwh": f"{kind}_missed_activation_mwh",
            "missed_activation_pos_mwh": f"{kind}_missed_activation_pos_mwh",
            "missed_activation_neg_mwh": f"{kind}_missed_activation_neg_mwh",
            "missed_capacity_mw": f"{kind}_missed_capacity_mw",
            "missed_capacity_pos_mw": f"{kind}_missed_capacity_pos_mw",
            "missed_capacity_neg_mw": f"{kind}_missed_capacity_neg_mw",
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
        }, inplace=True)
        return out

    def run(
        self,
        df: pd.DataFrame,
        colmap: BacktestColumnMap,
        *,
        use_rolling_horizon: bool = True,
        horizon_hours: int = 48,
        reopt_step_hours: int = 1,
        forecast_warehouse: dict[str, pd.DataFrame] | None = None,
        da_gate_hour_cet: int = 12,
        soc_feedback_mode: str = "realized",
        enforce_final_soc_min: bool = True,
    ) -> BacktestOutputs:
        """Run optimization + predicted settlement + realized settlement."""
        if horizon_hours <= 0 or reopt_step_hours <= 0:
            raise ValueError("horizon_hours and reopt_step_hours must be > 0")
        self._assert_valid_time_index(df, colmap.timestamp)
        def _run_isolated_path(
            *,
            path_df: pd.DataFrame,
            allowed_markets_local: tuple[str, ...],
            deterministic_local: bool,
            is_oracle_local: bool,
        ) -> tuple[float, bool]:
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
                        is_oracle=is_oracle_local,
                        allowed_markets=allowed_markets_local,
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
                    return 0.0, False
                raise
            path_real = self.settle_dispatch(
                path_df,
                path_dispatch,
                colmap,
                predicted_settlement=False,
                apply_market_clearing=True,
                oracle_mode=is_oracle_local,
            )
            pnl_excl = float(path_real["real_pnl_eur"].sum()) if not path_real.empty else 0.0
            final_soc = float(path_real["real_soc_mwh"].iloc[-1]) if (not path_real.empty and "real_soc_mwh" in path_real.columns) else float(self.soc_init)
            da_true_last_local = self._finite_numeric_series(
                df,
                colmap.true_da_price,
                fallback_cols=[colmap.pred_da_price],
                default=0.0,
            )
            terminal_price_local = max(
                0.0,
                float(da_true_last_local.iloc[-1]) if len(da_true_last_local) else 0.0,
            )
            terminal_value_local = max(0.0, final_soc - self.soc_min) * self.eta_out * terminal_price_local
            return float(pnl_excl + terminal_value_local), True

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
                allowed_markets=("DA", "aFRR"),
            )
        else:
            dispatch = self.optimize_dispatch(
                df,
                colmap,
                soc_start=self.soc_init,
                soc_end_target=None,
                soc_end_min_target=self.soc_target_end if enforce_final_soc_min else None,
                deterministic_reserve_settlement=False,
                allowed_markets=("DA", "aFRR"),
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
                oracle_mode=False,
            )

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
                fallback = pd.to_numeric(naive_df[pred_col], errors="coerce") if pred_col in naive_df.columns else np.nan
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
                allowed_markets=("DA", "aFRR"),
            )
        else:
            naive_dispatch = self.optimize_dispatch(
                naive_df,
                colmap,
                soc_start=self.soc_init,
                soc_end_target=None,
                soc_end_min_target=self.soc_target_end if enforce_final_soc_min else None,
                deterministic_reserve_settlement=False,
                allowed_markets=("DA", "aFRR"),
            )
        with _phase_watchdog("settlement_naive_realized"):
            naive_real = self.settle_dispatch(
                df,
                naive_dispatch,
                colmap,
                predicted_settlement=False,
                apply_market_clearing=True,
                oracle_mode=False,
            )
        naive_real = naive_real.rename(
            columns={
                c: c.replace("real_", "naive_", 1)
                for c in naive_real.columns
                if c.startswith("real_")
            }
        )

        # Oracle run: optimize using realized market values as perfect foresight benchmark.
        # Oracle uses the same clearing/settlement pipeline as realized runs, but
        # with oracle-aware bid building to guarantee clearable bids.
        oracle_df = df.copy()
        oracle_df[colmap.pred_da_price] = oracle_df[colmap.true_da_price]
        oracle_df[colmap.pred_afrr_capacity_price_pos] = oracle_df[colmap.true_afrr_capacity_price_pos]
        oracle_df[colmap.pred_afrr_capacity_price_neg] = oracle_df[colmap.true_afrr_capacity_price_neg]
        oracle_df[colmap.pred_afrr_activation_price_pos] = oracle_df[colmap.true_afrr_activation_price_pos]
        oracle_df[colmap.pred_afrr_activation_price_neg] = oracle_df[colmap.true_afrr_activation_price_neg]
        oracle_df[colmap.pred_afrr_activation_rate_pos] = oracle_df[colmap.true_afrr_activation_rate_pos]
        oracle_df[colmap.pred_afrr_activation_rate_neg] = oracle_df[colmap.true_afrr_activation_rate_neg]
        if use_rolling_horizon:
            oracle_dispatch, _ = self.optimize_dispatch_rolling(
                oracle_df,
                colmap,
                horizon_hours=horizon_hours,
                reopt_step_hours=reopt_step_hours,
                da_gate_hour_cet=da_gate_hour_cet,
                soc_feedback_mode=soc_feedback_mode,
                enforce_final_soc_min=enforce_final_soc_min,
                deterministic_reserve_settlement=True,
                is_oracle=True,
                allowed_markets=("DA", "aFRR"),
            )
        else:
            oracle_dispatch = self.optimize_dispatch(
                oracle_df,
                colmap,
                soc_start=self.soc_init,
                soc_end_target=None,
                soc_end_min_target=self.soc_target_end if enforce_final_soc_min else None,
                deterministic_reserve_settlement=True,
                allowed_markets=("DA", "aFRR"),
            )
        with _phase_watchdog("settlement_oracle_realized"):
            oracle_real = self.settle_dispatch(
                oracle_df,
                oracle_dispatch,
                colmap,
                predicted_settlement=False,
                apply_market_clearing=True,
                oracle_mode=True,
            )
        oracle_real = oracle_real.rename(
            columns={
                c: c.replace("real_", "oracle_", 1)
                for c in oracle_real.columns
                if c.startswith("real_")
            }
        )

        # Isolated market ablation paths (value of stacking).
        realized_da_only_total, realized_da_only_feasible = _run_isolated_path(
            path_df=df,
            allowed_markets_local=("DA",),
            deterministic_local=False,
            is_oracle_local=False,
        )
        oracle_da_only_total, oracle_da_only_feasible = _run_isolated_path(
            path_df=oracle_df,
            allowed_markets_local=("DA",),
            deterministic_local=True,
            is_oracle_local=True,
        )
        realized_afrr_only_total, realized_afrr_only_feasible = _run_isolated_path(
            path_df=df,
            allowed_markets_local=("aFRR",),
            deterministic_local=False,
            is_oracle_local=False,
        )
        oracle_afrr_only_total, oracle_afrr_only_feasible = _run_isolated_path(
            path_df=oracle_df,
            allowed_markets_local=("aFRR",),
            deterministic_local=True,
            is_oracle_local=True,
        )

        def _merge_unique(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
            """Merge while dropping overlapping non-key columns from right."""
            right_cols = [colmap.timestamp] + [
                c for c in right.columns if c != colmap.timestamp and c not in left.columns
            ]
            return left.merge(right[right_cols], on=colmap.timestamp, how="left")

        hourly = dispatch.copy()
        hourly = _merge_unique(hourly, pred)
        hourly = _merge_unique(hourly, real)
        hourly = _merge_unique(hourly, naive_real)
        hourly = _merge_unique(hourly, oracle_real)
        hourly = hourly.sort_values(colmap.timestamp).reset_index(drop=True)

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
        if "oracle_net_cashflow_eur" in hourly.columns:
            hourly["oracle_cashflow_eur"] = hourly["oracle_net_cashflow_eur"]
        else:
            hourly["oracle_cashflow_eur"] = hourly["oracle_pnl_eur"] + hourly.get("oracle_degradation_cost_eur", 0.0)
        hourly["real_cum_cash_eur"] = self.initial_cash + hourly["real_cashflow_eur"].cumsum()
        hourly["pred_cum_cash_eur"] = self.initial_cash + hourly["pred_cashflow_eur"].cumsum()
        hourly["naive_cum_cash_eur"] = self.initial_cash + hourly["naive_cashflow_eur"].cumsum()
        hourly["oracle_cum_cash_eur"] = self.initial_cash + hourly["oracle_cashflow_eur"].cumsum()
        # Profit cumulatives (includes non-cash degradation by definition of *_pnl_eur).
        hourly["real_cum_pnl_eur"] = hourly["real_pnl_eur"].cumsum()
        hourly["pred_cum_pnl_eur"] = hourly["pred_pnl_eur"].cumsum()
        hourly["naive_cum_pnl_eur"] = hourly["naive_pnl_eur"].cumsum()
        hourly["oracle_cum_pnl_eur"] = hourly["oracle_pnl_eur"].cumsum()
        hourly["pnl_gap_eur"] = hourly["real_pnl_eur"] - hourly["pred_pnl_eur"]
        hourly["cost_of_forecast_error_eur"] = hourly["oracle_pnl_eur"] - hourly["real_pnl_eur"]

        min_cash = float(hourly["real_cum_cash_eur"].min()) if not hourly.empty else self.initial_cash
        capital_required = max(0.0, -min_cash)

        monthly = aggregate_periodic(hourly, colmap.timestamp, freq="ME")
        yearly = aggregate_periodic(hourly, colmap.timestamp, freq="YE")
        volatility = calculate_volatility(plan_history)
        naive_volatility = calculate_volatility(naive_plan_history)

        pred_pnl_raw = float(hourly["pred_pnl_eur"].sum())
        real_pnl_raw = float(hourly["real_pnl_eur"].sum())
        naive_pnl_raw = float(hourly["naive_pnl_eur"].sum())
        oracle_pnl_raw = float(hourly["oracle_pnl_eur"].sum())

        if not hourly.empty:
            final_pred_soc_mwh = float(hourly["pred_soc_mwh"].iloc[-1]) if "pred_soc_mwh" in hourly.columns else float(self.soc_init)
            final_real_soc_mwh = float(hourly["real_soc_mwh"].iloc[-1]) if "real_soc_mwh" in hourly.columns else float(self.soc_init)
            final_naive_soc_mwh = float(hourly["naive_soc_mwh"].iloc[-1]) if "naive_soc_mwh" in hourly.columns else float(self.soc_init)
            final_oracle_soc_mwh = float(hourly["oracle_soc_mwh"].iloc[-1]) if "oracle_soc_mwh" in hourly.columns else float(self.soc_init)
        else:
            final_pred_soc_mwh = float(self.soc_init)
            final_real_soc_mwh = float(self.soc_init)
            final_naive_soc_mwh = float(self.soc_init)
            final_oracle_soc_mwh = float(self.soc_init)

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

        liquidatable_pred_mwh = max(0.0, final_pred_soc_mwh - self.soc_min) * self.eta_out
        liquidatable_real_mwh = max(0.0, final_real_soc_mwh - self.soc_min) * self.eta_out
        liquidatable_naive_mwh = max(0.0, final_naive_soc_mwh - self.soc_min) * self.eta_out
        liquidatable_oracle_mwh = max(0.0, final_oracle_soc_mwh - self.soc_min) * self.eta_out

        terminal_value_pred_eur = liquidatable_pred_mwh * terminal_price_pred_eur_mwh
        terminal_value_real_eur = liquidatable_real_mwh * terminal_price_true_eur_mwh
        terminal_value_naive_eur = liquidatable_naive_mwh * terminal_price_true_eur_mwh
        terminal_value_oracle_eur = liquidatable_oracle_mwh * terminal_price_true_eur_mwh

        pred_pnl_total = pred_pnl_raw + terminal_value_pred_eur
        real_pnl_total = real_pnl_raw + terminal_value_real_eur
        naive_pnl_total = naive_pnl_raw + terminal_value_naive_eur
        oracle_pnl_total = oracle_pnl_raw + terminal_value_oracle_eur

        if np.isfinite(oracle_pnl_total) and abs(oracle_pnl_total) > 1e-9:
            opportunity_gap_ratio = float((oracle_pnl_total - real_pnl_total) / oracle_pnl_total)
        else:
            opportunity_gap_ratio = float("nan")
        summary = {
            "rows": float(len(hourly)),
            "planned_total_pnl_eur": float(pred_pnl_total),
            "predicted_total_pnl_eur": float(pred_pnl_total),
            "realized_total_pnl_eur": float(real_pnl_total),
            "naive_total_pnl_eur": float(naive_pnl_total),
            "oracle_total_pnl_eur": float(oracle_pnl_total),
            "predicted_pnl_excl_terminal_eur": float(pred_pnl_raw),
            "realized_pnl_excl_terminal_eur": float(real_pnl_raw),
            "naive_pnl_excl_terminal_eur": float(naive_pnl_raw),
            "oracle_pnl_excl_terminal_eur": float(oracle_pnl_raw),
            "terminal_value_predicted_eur": float(terminal_value_pred_eur),
            "terminal_value_realized_eur": float(terminal_value_real_eur),
            "terminal_value_naive_eur": float(terminal_value_naive_eur),
            "terminal_value_oracle_eur": float(terminal_value_oracle_eur),
            "terminal_liquidatable_predicted_mwh": float(liquidatable_pred_mwh),
            "terminal_liquidatable_realized_mwh": float(liquidatable_real_mwh),
            "terminal_liquidatable_naive_mwh": float(liquidatable_naive_mwh),
            "terminal_liquidatable_oracle_mwh": float(liquidatable_oracle_mwh),
            "terminal_price_predicted_eur_mwh": float(terminal_price_pred_eur_mwh),
            "terminal_price_realized_eur_mwh": float(terminal_price_true_eur_mwh),
            "realized_total_penalty_eur": float(hourly["real_penalty_eur"].sum()) if "real_penalty_eur" in hourly.columns else 0.0,
            "naive_total_penalty_eur": float(hourly["naive_penalty_eur"].sum()) if "naive_penalty_eur" in hourly.columns else 0.0,
            "oracle_total_penalty_eur": float(hourly["oracle_penalty_eur"].sum()) if "oracle_penalty_eur" in hourly.columns else 0.0,
            "pnl_gap_total_eur": float(real_pnl_total - pred_pnl_total),
            "economic_opportunity_gap_ratio": opportunity_gap_ratio,
            "cost_of_forecast_error_total_eur": float(oracle_pnl_total - real_pnl_total),
            "max_capital_required_eur": float(capital_required),
            "max_drawdown_eur": float(capital_required),
            "max_hourly_cash_outflow_eur": float(max(0.0, -hourly["real_cashflow_eur"].min())),
            "avg_reserve_pos_mw": float(hourly["reserve_pos_mw"].mean()),
            "avg_reserve_neg_mw": float(hourly["reserve_neg_mw"].mean()),
            "avg_charge_mw": float(hourly["charge_mw"].mean()),
            "avg_discharge_mw": float(hourly["discharge_mw"].mean()),
            "realized_da_only_total_pnl_eur": float(realized_da_only_total),
            "oracle_da_only_total_pnl_eur": float(oracle_da_only_total),
            "realized_afrr_only_total_pnl_eur": float(realized_afrr_only_total),
            "oracle_afrr_only_total_pnl_eur": float(oracle_afrr_only_total),
            "realized_da_only_feasible": float(realized_da_only_feasible),
            "oracle_da_only_feasible": float(oracle_da_only_feasible),
            "realized_afrr_only_feasible": float(realized_afrr_only_feasible),
            "oracle_afrr_only_feasible": float(oracle_afrr_only_feasible),
        }
        summary["stacking_value_realized_eur"] = float(
            summary["realized_total_pnl_eur"]
            - (
                summary["realized_da_only_total_pnl_eur"]
                + summary["realized_afrr_only_total_pnl_eur"]
            )
        )
        summary["stacking_value_oracle_eur"] = float(
            summary["oracle_total_pnl_eur"]
            - (
                summary["oracle_da_only_total_pnl_eur"]
                + summary["oracle_afrr_only_total_pnl_eur"]
            )
        )
        # Economic building blocks for thesis-grade PnL decomposition.
        summary["total_da_revenue_eur"] = float(hourly["real_revenue_da_eur"].sum())
        summary["total_da_cost_eur"] = float(hourly["real_cost_da_eur"].sum())
        summary["total_id_revenue_eur"] = float(hourly["real_revenue_id_eur"].sum()) if "real_revenue_id_eur" in hourly.columns else 0.0
        summary["total_id_cost_eur"] = float(hourly["real_cost_id_eur"].sum()) if "real_cost_id_eur" in hourly.columns else 0.0
        summary["total_afrr_capacity_revenue_eur"] = float(hourly["real_revenue_capacity_eur"].sum())
        summary["total_afrr_activation_revenue_eur"] = float(hourly["real_revenue_activation_eur"].sum())
        summary["total_degradation_cost_eur"] = float(hourly["real_degradation_cost_eur"].sum())
        summary["total_transaction_cost_eur"] = float(hourly["real_transaction_cost_eur"].sum())
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

        # Backward-compatible aliases used by older reports/scripts.
        summary["total_da_energy_revenue_eur"] = summary["total_da_revenue_eur"]
        summary["total_da_energy_cost_eur"] = summary["total_da_cost_eur"]

        pnl_components_rhs = (
            summary["total_da_revenue_eur"]
            + summary["total_afrr_capacity_revenue_eur"]
            + summary["total_afrr_activation_revenue_eur"]
            - summary["total_da_cost_eur"]
            - summary["total_degradation_cost_eur"]
            - summary["total_transaction_cost_eur"]
            - summary["total_penalty_cost_eur"]
        )
        summary["realized_pnl_from_components_eur"] = float(pnl_components_rhs)
        summary["realized_pnl_from_components_plus_terminal_eur"] = float(
            pnl_components_rhs + summary["terminal_value_realized_eur"]
        )
        summary["realized_pnl_balance_error_eur"] = float(
            summary["realized_total_pnl_eur"] - summary["realized_pnl_from_components_plus_terminal_eur"]
        )
        summary["realized_pnl_balance_ok"] = float(abs(summary["realized_pnl_balance_error_eur"]) <= 1e-6)

        # Oracle component decomposition / balance check (same settlement accounting identity).
        summary["oracle_total_da_revenue_eur"] = float(hourly["oracle_revenue_da_eur"].sum())
        summary["oracle_total_da_cost_eur"] = float(hourly["oracle_cost_da_eur"].sum())
        summary["oracle_total_afrr_capacity_revenue_eur"] = float(hourly["oracle_revenue_capacity_eur"].sum())
        summary["oracle_total_afrr_activation_revenue_eur"] = float(hourly["oracle_revenue_activation_eur"].sum())
        summary["oracle_total_degradation_cost_eur"] = float(hourly["oracle_degradation_cost_eur"].sum())
        summary["oracle_total_transaction_cost_eur"] = float(hourly["oracle_transaction_cost_eur"].sum())
        summary["oracle_total_penalty_cost_eur"] = float(hourly["oracle_penalty_eur"].sum()) if "oracle_penalty_eur" in hourly.columns else 0.0
        oracle_pnl_components_rhs = (
            summary["oracle_total_da_revenue_eur"]
            + summary["oracle_total_afrr_capacity_revenue_eur"]
            + summary["oracle_total_afrr_activation_revenue_eur"]
            - summary["oracle_total_da_cost_eur"]
            - summary["oracle_total_degradation_cost_eur"]
            - summary["oracle_total_transaction_cost_eur"]
            - summary["oracle_total_penalty_cost_eur"]
        )
        summary["oracle_pnl_from_components_eur"] = float(oracle_pnl_components_rhs)
        summary["oracle_pnl_from_components_plus_terminal_eur"] = float(
            oracle_pnl_components_rhs + summary["terminal_value_oracle_eur"]
        )
        summary["oracle_pnl_balance_error_eur"] = float(
            summary["oracle_total_pnl_eur"] - summary["oracle_pnl_from_components_plus_terminal_eur"]
        )
        summary["oracle_pnl_balance_ok"] = float(abs(summary["oracle_pnl_balance_error_eur"]) <= 1e-6)

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
        # Keep ROI numerically stable by flooring denominator at 1 EUR.
        roi_denom = max(1.0, float(summary["max_capital_required_eur"]))
        summary["roi_on_max_capital"] = float(summary["realized_total_pnl_eur"] / roi_denom)
        summary["capture_ratio_vs_naive"] = (
            float(summary["realized_total_pnl_eur"] / summary["naive_total_pnl_eur"])
            if abs(float(summary["naive_total_pnl_eur"])) > 1e-12 else float("nan")
        )
        summary["capture_ratio_vs_oracle"] = (
            float(summary["realized_total_pnl_eur"] / summary["oracle_total_pnl_eur"])
            if abs(float(summary["oracle_total_pnl_eur"])) > 1e-12 else float("nan")
        )
        summary["oracle_upper_bound_ok"] = float(summary["oracle_total_pnl_eur"] >= summary["realized_total_pnl_eur"] - 1e-9)
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
        summary["final_real_soc_mwh"] = final_real_soc_mwh
        summary["final_soc_min_target_mwh"] = float(self.soc_target_end)
        summary["final_soc_constraint_satisfied"] = float(final_real_soc_mwh >= float(self.soc_target_end) - 1e-9)
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

        return BacktestOutputs(
            hourly=hourly,
            monthly=monthly,
            yearly=yearly,
            plan_history=plan_history,
            volatility=volatility,
            summary=summary,
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
        "oracle_pnl_eur": ("oracle_pnl_eur", "sum"),
        "predicted_pnl_eur": ("pred_pnl_eur", "sum"),
        "realized_penalty_eur": ("real_penalty_eur", "sum"),
        "naive_penalty_eur": ("naive_penalty_eur", "sum"),
        "oracle_penalty_eur": ("oracle_penalty_eur", "sum"),
        "pnl_gap_eur": ("pnl_gap_eur", "sum"),
        "cost_of_forecast_error_eur": ("cost_of_forecast_error_eur", "sum"),
        "realized_revenue_da_eur": ("real_revenue_da_eur", "sum"),
        "realized_cost_da_eur": ("real_cost_da_eur", "sum"),
        "realized_revenue_capacity_eur": ("real_revenue_capacity_eur", "sum"),
        "realized_revenue_activation_eur": ("real_revenue_activation_eur", "sum"),
        "realized_transaction_cost_eur": ("real_transaction_cost_eur", "sum"),
        "realized_degradation_cost_eur": ("real_degradation_cost_eur", "sum"),
        "avg_charge_mw": ("charge_mw", "mean"),
        "avg_discharge_mw": ("discharge_mw", "mean"),
        "avg_reserve_pos_mw": ("reserve_pos_mw", "mean"),
        "avg_reserve_neg_mw": ("reserve_neg_mw", "mean"),
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
