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
        da_gate_hour_utc=11,
        soc_feedback_mode="realized",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from energy_trading.config import BATTERY_SPECS, FINANCIAL_PARAMS, MARKET_SPECS, MODEL_SPECS


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
    true_afrr_activation_rate_pos: str = "afrr_activation_rate_pos"
    true_afrr_activation_rate_neg: str = "afrr_activation_rate_neg"


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
        out = df[list(required)].copy()
        out["snapshot_time_utc"] = pd.to_datetime(out["snapshot_time_utc"], utc=True, errors="coerce")
        out["target_time_utc"] = pd.to_datetime(out["target_time_utc"], utc=True, errors="coerce")
        out["lead_time_h"] = pd.to_numeric(out["lead_time_h"], errors="coerce").astype("Int64")
        out["predicted_value"] = pd.to_numeric(out["predicted_value"], errors="coerce")
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

        self.bid_power_max_mw = float(MARKET_SPECS.get("bid_power_max_mw", self.p_max_mw))
        self.reserve_max_mw = min(self.p_max_mw, self.bid_power_max_mw)

    @staticmethod
    def _clip_rate(x: np.ndarray) -> np.ndarray:
        return np.clip(np.nan_to_num(x, nan=0.0), 0.0, 1.0)

    def _normalize_da_bid(self, charge_mw: float, discharge_mw: float) -> tuple[float, float]:
        """Project DA bid pair to a single net direction to avoid binary lock conflicts."""
        ch = max(0.0, float(charge_mw))
        dis = max(0.0, float(discharge_mw))
        net = dis - ch
        if net >= 0.0:
            return 0.0, min(self.p_max_mw, net)
        return min(self.p_max_mw, -net), 0.0

    def _variable_slices(self, n: int) -> dict[str, slice]:
        # Hourly decisions: charge, discharge, reserve_pos, reserve_neg,
        # binary is_charging flag, and soc state.
        return {
            "ch": slice(0, n),
            "dis": slice(n, 2 * n),
            "rpos": slice(2 * n, 3 * n),
            "rneg": slice(3 * n, 4 * n),
            "u": slice(4 * n, 5 * n),
            "soc": slice(5 * n, 6 * n + 1),
        }

    def optimize_dispatch(
        self,
        df: pd.DataFrame,
        colmap: BacktestColumnMap,
        *,
        soc_start: float | None = None,
        soc_end_target: float | None = None,
        soc_end_min_target: float | None = None,
        fixed_da_dispatch: dict[pd.Timestamp, tuple[float, float]] | None = None,
    ) -> pd.DataFrame:
        """Solve LP on predicted market signals to obtain hourly dispatch."""
        n = len(df)
        sl = self._variable_slices(n)
        n_vars = 6 * n + 1

        p_da = pd.to_numeric(df[colmap.pred_da_price], errors="coerce").to_numpy(dtype=float)
        p_cap_pos = pd.to_numeric(df[colmap.pred_afrr_capacity_price_pos], errors="coerce").to_numpy(dtype=float)
        p_cap_neg = pd.to_numeric(df[colmap.pred_afrr_capacity_price_neg], errors="coerce").to_numpy(dtype=float)
        p_act_pos = pd.to_numeric(df[colmap.pred_afrr_activation_price_pos], errors="coerce").to_numpy(dtype=float)
        p_act_neg = pd.to_numeric(df[colmap.pred_afrr_activation_price_neg], errors="coerce").to_numpy(dtype=float)
        r_act_pos = self._clip_rate(pd.to_numeric(df[colmap.pred_afrr_activation_rate_pos], errors="coerce").to_numpy(dtype=float))
        r_act_neg = self._clip_rate(pd.to_numeric(df[colmap.pred_afrr_activation_rate_neg], errors="coerce").to_numpy(dtype=float))

        c = np.zeros(n_vars, dtype=float)

        # Objective (maximize predicted margin, linprog minimizes => negate coefficients).
        ch_coef = -(p_da / self.eta_in) - self.trans_eur_mwh / self.eta_in - self.deg_eur_mwh
        dis_coef = (p_da * self.eta_out) - self.trans_eur_mwh * self.eta_out - self.deg_eur_mwh
        rpos_coef = (
            p_cap_pos
            + p_act_pos * r_act_pos * self.eta_out
            - self.trans_eur_mwh * r_act_pos * self.eta_out
            - self.deg_eur_mwh * r_act_pos
        )
        rneg_coef = (
            p_cap_neg
            + p_act_neg * r_act_neg / self.eta_in
            - self.trans_eur_mwh * r_act_neg / self.eta_in
            - self.deg_eur_mwh * r_act_neg
        )

        c[sl["ch"]] = -ch_coef
        c[sl["dis"]] = -dis_coef
        c[sl["rpos"]] = -rpos_coef
        c[sl["rneg"]] = -rneg_coef
        # Terminal SoC opportunity value:
        # V_terminal = SoC_T * mean(predicted_DA_price over horizon)
        # scipy.milp minimizes, so we add a negative coefficient on terminal SoC.
        da_ref = pd.Series(p_da).dropna()
        ref_da_price = float(da_ref.mean()) if not da_ref.empty else 0.0
        c[sl["soc"].start + n] = -ref_da_price

        a_eq = []
        b_eq = []

        # SoC dynamics.
        for t in range(n):
            row = np.zeros(n_vars, dtype=float)
            row[sl["soc"].start + t + 1] = 1.0
            row[sl["soc"].start + t] = -1.0
            row[sl["ch"].start + t] = -self.eta_in
            row[sl["dis"].start + t] = 1.0 / self.eta_out
            row[sl["rpos"].start + t] = r_act_pos[t] / self.eta_out
            row[sl["rneg"].start + t] = -self.eta_in * r_act_neg[t]
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
            row[sl["ch"].start + t] = 1.0
            row[sl["rneg"].start + t] = 1.0
            a_ub.append(row)
            b_ub.append(self.p_max_mw)

            # discharge + reserve_pos <= Pmax
            row = np.zeros(n_vars, dtype=float)
            row[sl["dis"].start + t] = 1.0
            row[sl["rpos"].start + t] = 1.0
            a_ub.append(row)
            b_ub.append(self.p_max_mw)

            # reserve_pos + reserve_neg <= reserve_max
            row = np.zeros(n_vars, dtype=float)
            row[sl["rpos"].start + t] = 1.0
            row[sl["rneg"].start + t] = 1.0
            a_ub.append(row)
            b_ub.append(self.reserve_max_mw)

            # Mixed-integer exclusivity (Big-M):
            # charge[t] <= is_charging[t] * Pmax
            row = np.zeros(n_vars, dtype=float)
            row[sl["ch"].start + t] = 1.0
            row[sl["u"].start + t] = -self.p_max_mw
            a_ub.append(row)
            b_ub.append(0.0)

            # discharge[t] <= (1 - is_charging[t]) * Pmax
            # <=> discharge[t] + is_charging[t] * Pmax <= Pmax
            row = np.zeros(n_vars, dtype=float)
            row[sl["dis"].start + t] = 1.0
            row[sl["u"].start + t] = self.p_max_mw
            a_ub.append(row)
            b_ub.append(self.p_max_mw)

            # DA gate-closure lock: fixed day-ahead charge/discharge bids.
            if fixed_da_dispatch:
                ts = ts_index.iloc[t]
                if pd.notna(ts) and ts in fixed_da_dispatch:
                    ch_fix, dis_fix = self._normalize_da_bid(*fixed_da_dispatch[ts])

                    row = np.zeros(n_vars, dtype=float)
                    row[sl["ch"].start + t] = 1.0
                    a_eq.append(row)
                    b_eq.append(float(ch_fix))

                    row = np.zeros(n_vars, dtype=float)
                    row[sl["dis"].start + t] = 1.0
                    a_eq.append(row)
                    b_eq.append(float(dis_fix))

        # Final SoC floor constraint: SoC_T >= soc_end_min_target
        if soc_end_min_target is not None:
            row = np.zeros(n_vars, dtype=float)
            row[sl["soc"].start + n] = -1.0
            a_ub.append(row)
            b_ub.append(-float(soc_end_min_target))

        lb = []
        ub = []
        lb.extend([0.0] * n)  # charge
        ub.extend([self.p_max_mw] * n)
        lb.extend([0.0] * n)  # discharge
        ub.extend([self.p_max_mw] * n)
        lb.extend([0.0] * n)  # reserve pos
        ub.extend([self.reserve_max_mw] * n)
        lb.extend([0.0] * n)  # reserve neg
        ub.extend([self.reserve_max_mw] * n)
        lb.extend([0.0] * n)  # is_charging
        ub.extend([1.0] * n)
        lb.extend([self.soc_min] * (n + 1))  # soc
        ub.extend([self.soc_max] * (n + 1))

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
        integrality[sl["u"]] = 1

        sol = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(np.array(lb), np.array(ub)),
        )
        if not sol.success:
            raise RuntimeError(f"MIP optimization failed: {sol.message}")

        x = sol.x
        out = df[[colmap.timestamp]].copy()
        out["charge_mw"] = x[sl["ch"]]
        out["discharge_mw"] = x[sl["dis"]]
        out["reserve_pos_mw"] = x[sl["rpos"]]
        out["reserve_neg_mw"] = x[sl["rneg"]]
        out["is_charging"] = x[sl["u"]]
        out["soc_lp_mwh"] = x[sl["soc"].start + 1 : sl["soc"].start + n + 1]
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
    ) -> tuple[float, dict[str, float]]:
        # Requested internal energies for this hour.
        act_pos_internal = max(0.0, act_pos_rate) * reserve_pos * self.dt_h
        act_neg_internal = max(0.0, act_neg_rate) * reserve_neg * self.dt_h
        ch_internal = charge * self.dt_h
        dis_internal = discharge * self.dt_h

        in_internal = ch_internal + act_neg_internal
        out_internal = dis_internal + act_pos_internal

        # Keep SoC feasible by scaling in/out streams if needed.
        delta = self.eta_in * in_internal - out_internal / self.eta_out - self.aux_mwh
        min_delta = self.soc_min - soc
        max_delta = self.soc_max - soc

        if delta < min_delta and out_internal > 0:
            excess = min_delta - delta
            scale = max(0.0, 1.0 - (excess * self.eta_out) / max(out_internal, 1e-12))
            out_internal *= scale
            dis_internal *= scale
            act_pos_internal *= scale
            delta = self.eta_in * in_internal - out_internal / self.eta_out - self.aux_mwh
        if delta > max_delta and in_internal > 0:
            excess = delta - max_delta
            scale = max(0.0, 1.0 - excess / max(self.eta_in * in_internal, 1e-12))
            in_internal *= scale
            ch_internal *= scale
            act_neg_internal *= scale
            delta = self.eta_in * in_internal - out_internal / self.eta_out - self.aux_mwh

        soc_next = float(np.clip(soc + delta, self.soc_min, self.soc_max))

        # Settlement cashflows.
        da_buy_grid = ch_internal / self.eta_in
        da_sell_grid = dis_internal * self.eta_out
        act_pos_grid = act_pos_internal * self.eta_out
        act_neg_grid = act_neg_internal / self.eta_in

        rev_da = da_sell_grid * da_price
        cost_da = da_buy_grid * da_price
        rev_cap = reserve_pos * cap_pos + reserve_neg * cap_neg
        rev_act = act_pos_grid * act_pos_price + act_neg_grid * act_neg_price

        trans_cost = self.trans_eur_mwh * (da_buy_grid + da_sell_grid + act_pos_grid + act_neg_grid)
        degr_cost = self.deg_eur_mwh * (ch_internal + dis_internal + act_pos_internal + act_neg_internal)

        pnl = rev_da - cost_da + rev_cap + rev_act - trans_cost - degr_cost

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
            "pnl_eur": pnl,
        }
        return soc_next, metrics

    def optimize_dispatch_rolling(
        self,
        df: pd.DataFrame,
        colmap: BacktestColumnMap,
        *,
        horizon_hours: int = 48,
        reopt_step_hours: int = 1,
        forecast_warehouse: dict[str, pd.DataFrame] | None = None,
        da_gate_hour_utc: int = 11,
        soc_feedback_mode: str = "realized",
        enforce_final_soc_min: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Rolling-horizon LP (re-optimized repeatedly with SoC state carryover)."""
        if horizon_hours <= 0 or reopt_step_hours <= 0:
            raise ValueError("horizon_hours and reopt_step_hours must be > 0")
        if soc_feedback_mode not in {"realized", "predicted"}:
            raise ValueError("soc_feedback_mode must be one of {'realized', 'predicted'}")
        if df.empty:
            empty_dispatch = pd.DataFrame(
                columns=[colmap.timestamp, "charge_mw", "discharge_mw", "reserve_pos_mw", "reserve_neg_mw", "soc_lp_mwh", "predicted_objective_eur"]
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
        i = 0
        while i < n:
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
            else:
                w_end = min(n, i + horizon_hours)
                window = df.iloc[i:w_end].copy()

            # Keep terminal equality soft in rolling mode, but optionally enforce
            # a minimum final SoC for the very last optimization window.
            enforce_end = None
            enforce_end_min = self.soc_target_end if (enforce_final_soc_min and w_end == n) else None
            plan = self.optimize_dispatch(
                window,
                colmap,
                soc_start=soc,
                soc_end_target=enforce_end,
                soc_end_min_target=enforce_end_min,
                fixed_da_dispatch=da_lockbook,
            )

            snapshot_plan = plan.copy()
            snapshot_plan["snapshot_time_utc"] = snapshot_ts if forecast_warehouse else pd.to_datetime(window.iloc[0][colmap.timestamp], utc=True, errors="coerce")
            snapshot_plan["target_time_utc"] = pd.to_datetime(snapshot_plan[colmap.timestamp], utc=True, errors="coerce")
            snapshot_plan["lead_time_h"] = (
                (snapshot_plan["target_time_utc"] - snapshot_plan["snapshot_time_utc"]).dt.total_seconds() // 3600
            ).astype("Int64")
            # Long-format plan warehouse fields for decision-volatility analysis.
            snapshot_plan["planned_charge_mw"] = snapshot_plan["charge_mw"]
            snapshot_plan["planned_discharge_mw"] = snapshot_plan["discharge_mw"]
            snapshot_plan["planned_reserve_mw"] = snapshot_plan["reserve_pos_mw"] + snapshot_plan["reserve_neg_mw"]
            window_price_map = pd.Series(
                pd.to_numeric(window[colmap.pred_da_price], errors="coerce").to_numpy(dtype=float),
                index=pd.to_datetime(window[colmap.timestamp], utc=True, errors="coerce"),
            )
            snapshot_plan["predicted_price"] = snapshot_plan["target_time_utc"].map(window_price_map)
            snapshot_plan["da_bid_locked"] = snapshot_plan["target_time_utc"].isin(set(da_lockbook.keys()))
            plan_history.append(snapshot_plan)

            # Lock-in DA bids at gate closure for next UTC day (24 hours).
            if pd.notna(snapshot_plan["snapshot_time_utc"].iloc[0]) and int(snapshot_plan["snapshot_time_utc"].iloc[0].hour) == int(da_gate_hour_utc):
                snapshot_ts_effective = pd.to_datetime(snapshot_plan["snapshot_time_utc"].iloc[0], utc=True)
                next_day = (snapshot_ts_effective + pd.Timedelta(days=1)).normalize()
                day_end = next_day + pd.Timedelta(hours=23)
                lock_rows = snapshot_plan[
                    (snapshot_plan["target_time_utc"] >= next_day) & (snapshot_plan["target_time_utc"] <= day_end)
                ]
                for r in lock_rows[[colmap.timestamp, "charge_mw", "discharge_mw"]].itertuples(index=False):
                    da_lockbook[pd.to_datetime(r[0], utc=True)] = self._normalize_da_bid(float(r[1]), float(r[2]))

            k = min(reopt_step_hours, len(plan))
            take = plan.iloc[:k].copy()
            pred_cols = [c for c in CANONICAL_PREDICTION_COLUMNS if c in window.columns]
            if pred_cols:
                pred_take = window[[colmap.timestamp, *pred_cols]].iloc[:k].copy()
                take = take.merge(pred_take, on=colmap.timestamp, how="left")
            decisions.append(take)

            # Propagate SoC using configured feedback mode (realized or predicted).
            window_src = window.set_index(colmap.timestamp)
            for r in take.itertuples(index=False):
                ts = getattr(r, colmap.timestamp)
                src = window_src.loc[ts] if ts in window_src.index else source.loc[ts]
                if soc_feedback_mode == "realized":
                    da_price = float(src[colmap.true_da_price])
                    cap_pos = float(src[colmap.true_afrr_capacity_price_pos])
                    cap_neg = float(src[colmap.true_afrr_capacity_price_neg])
                    act_pos_price = float(src[colmap.true_afrr_activation_price_pos])
                    act_neg_price = float(src[colmap.true_afrr_activation_price_neg])
                    act_pos_rate = float(src[colmap.true_afrr_activation_rate_pos])
                    act_neg_rate = float(src[colmap.true_afrr_activation_rate_neg])
                else:
                    da_price = float(src[colmap.pred_da_price])
                    cap_pos = float(src[colmap.pred_afrr_capacity_price_pos])
                    cap_neg = float(src[colmap.pred_afrr_capacity_price_neg])
                    act_pos_price = float(src[colmap.pred_afrr_activation_price_pos])
                    act_neg_price = float(src[colmap.pred_afrr_activation_price_neg])
                    act_pos_rate = float(src[colmap.pred_afrr_activation_rate_pos])
                    act_neg_rate = float(src[colmap.pred_afrr_activation_rate_neg])
                _, m = self._settle_one_hour(
                    soc=soc,
                    charge=float(getattr(r, "charge_mw")),
                    discharge=float(getattr(r, "discharge_mw")),
                    reserve_pos=float(getattr(r, "reserve_pos_mw")),
                    reserve_neg=float(getattr(r, "reserve_neg_mw")),
                    da_price=da_price,
                    cap_pos=cap_pos,
                    cap_neg=cap_neg,
                    act_pos_price=act_pos_price,
                    act_neg_price=act_neg_price,
                    act_pos_rate=act_pos_rate,
                    act_neg_rate=act_neg_rate,
                )
                soc = float(m["soc_mwh"])
                i += 1
                if i >= n:
                    break

        if not decisions:
            empty_dispatch = pd.DataFrame(
                columns=[
                    colmap.timestamp,
                    "charge_mw",
                    "discharge_mw",
                    "reserve_pos_mw",
                    "reserve_neg_mw",
                    "is_charging",
                    "soc_lp_mwh",
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

        dispatch_cols = [colmap.timestamp, "charge_mw", "discharge_mw", "reserve_pos_mw", "reserve_neg_mw"]
        if predicted_settlement and all(c in dispatch.columns for c in [da_col, cap_pos_col, cap_neg_col, act_pos_col, act_neg_col, rate_pos_col, rate_neg_col]):
            merged = dispatch[dispatch_cols + [da_col, cap_pos_col, cap_neg_col, act_pos_col, act_neg_col, rate_pos_col, rate_neg_col]].copy()
        else:
            merged = df[[
                colmap.timestamp,
                da_col,
                cap_pos_col,
                cap_neg_col,
                act_pos_col,
                act_neg_col,
                rate_pos_col,
                rate_neg_col,
            ]].copy()
            merged = merged.merge(dispatch[dispatch_cols], on=colmap.timestamp, how="inner")

        soc = self.soc_init
        rows: list[dict[str, float | pd.Timestamp]] = []
        for r in merged.itertuples(index=False):
            ts = getattr(r, colmap.timestamp)
            soc, m = self._settle_one_hour(
                soc=soc,
                charge=float(getattr(r, "charge_mw")),
                discharge=float(getattr(r, "discharge_mw")),
                reserve_pos=float(getattr(r, "reserve_pos_mw")),
                reserve_neg=float(getattr(r, "reserve_neg_mw")),
                da_price=float(getattr(r, da_col)),
                cap_pos=float(getattr(r, cap_pos_col)),
                cap_neg=float(getattr(r, cap_neg_col)),
                act_pos_price=float(getattr(r, act_pos_col)),
                act_neg_price=float(getattr(r, act_neg_col)),
                act_pos_rate=float(getattr(r, rate_pos_col)),
                act_neg_rate=float(getattr(r, rate_neg_col)),
            )
            rec = {colmap.timestamp: ts, **m}
            rows.append(rec)

        out = pd.DataFrame(rows)
        out.rename(columns={
            "pnl_eur": f"{kind}_pnl_eur",
            "revenue_da_eur": f"{kind}_revenue_da_eur",
            "cost_da_eur": f"{kind}_cost_da_eur",
            "revenue_capacity_eur": f"{kind}_revenue_capacity_eur",
            "revenue_activation_eur": f"{kind}_revenue_activation_eur",
            "transaction_cost_eur": f"{kind}_transaction_cost_eur",
            "degradation_cost_eur": f"{kind}_degradation_cost_eur",
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
        da_gate_hour_utc: int = 11,
        soc_feedback_mode: str = "realized",
        enforce_final_soc_min: bool = True,
    ) -> BacktestOutputs:
        """Run optimization + predicted settlement + realized settlement."""
        plan_history = pd.DataFrame()
        if use_rolling_horizon:
            dispatch, plan_history = self.optimize_dispatch_rolling(
                df,
                colmap,
                horizon_hours=horizon_hours,
                reopt_step_hours=reopt_step_hours,
                forecast_warehouse=forecast_warehouse,
                da_gate_hour_utc=da_gate_hour_utc,
                soc_feedback_mode=soc_feedback_mode,
                enforce_final_soc_min=enforce_final_soc_min,
            )
        else:
            dispatch = self.optimize_dispatch(
                df,
                colmap,
                soc_start=self.soc_init,
                soc_end_target=None,
                soc_end_min_target=self.soc_target_end if enforce_final_soc_min else None,
            )
        pred = self.settle_dispatch(df, dispatch, colmap, predicted_settlement=True)
        real = self.settle_dispatch(df, dispatch, colmap, predicted_settlement=False)

        # Oracle run: optimize using realized market values as perfect foresight benchmark.
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
                da_gate_hour_utc=da_gate_hour_utc,
                soc_feedback_mode=soc_feedback_mode,
                enforce_final_soc_min=enforce_final_soc_min,
            )
        else:
            oracle_dispatch = self.optimize_dispatch(
                oracle_df,
                colmap,
                soc_start=self.soc_init,
                soc_end_target=None,
                soc_end_min_target=self.soc_target_end if enforce_final_soc_min else None,
            )
        oracle_real = self.settle_dispatch(df, oracle_dispatch, colmap, predicted_settlement=False)
        oracle_real = oracle_real.rename(
            columns={
                "real_pnl_eur": "oracle_pnl_eur",
                "real_revenue_da_eur": "oracle_revenue_da_eur",
                "real_cost_da_eur": "oracle_cost_da_eur",
                "real_revenue_capacity_eur": "oracle_revenue_capacity_eur",
                "real_revenue_activation_eur": "oracle_revenue_activation_eur",
                "real_transaction_cost_eur": "oracle_transaction_cost_eur",
                "real_degradation_cost_eur": "oracle_degradation_cost_eur",
                "real_soc_mwh": "oracle_soc_mwh",
            }
        )

        hourly = (
            dispatch
            .merge(pred, on=colmap.timestamp, how="left")
            .merge(real, on=colmap.timestamp, how="left")
            .merge(oracle_real, on=colmap.timestamp, how="left")
        )
        hourly = hourly.sort_values(colmap.timestamp).reset_index(drop=True)

        hourly["real_cashflow_eur"] = hourly["real_pnl_eur"]
        hourly["pred_cashflow_eur"] = hourly["pred_pnl_eur"]
        hourly["real_cum_cash_eur"] = self.initial_cash + hourly["real_cashflow_eur"].cumsum()
        hourly["pred_cum_cash_eur"] = self.initial_cash + hourly["pred_cashflow_eur"].cumsum()
        hourly["pnl_gap_eur"] = hourly["real_pnl_eur"] - hourly["pred_pnl_eur"]
        hourly["cost_of_forecast_error_eur"] = hourly["oracle_pnl_eur"] - hourly["real_pnl_eur"]

        min_cash = float(hourly["real_cum_cash_eur"].min()) if not hourly.empty else self.initial_cash
        capital_required = max(0.0, -min_cash)

        monthly = aggregate_periodic(hourly, colmap.timestamp, freq="ME")
        yearly = aggregate_periodic(hourly, colmap.timestamp, freq="YE")
        volatility = calculate_volatility(plan_history)

        summary = {
            "rows": float(len(hourly)),
            "predicted_total_pnl_eur": float(hourly["pred_pnl_eur"].sum()),
            "realized_total_pnl_eur": float(hourly["real_pnl_eur"].sum()),
            "oracle_total_pnl_eur": float(hourly["oracle_pnl_eur"].sum()),
            "pnl_gap_total_eur": float(hourly["pnl_gap_eur"].sum()),
            "cost_of_forecast_error_total_eur": float(hourly["cost_of_forecast_error_eur"].sum()),
            "max_capital_required_eur": float(capital_required),
            "max_hourly_cash_outflow_eur": float(max(0.0, -hourly["real_cashflow_eur"].min())),
            "avg_reserve_pos_mw": float(hourly["reserve_pos_mw"].mean()),
            "avg_reserve_neg_mw": float(hourly["reserve_neg_mw"].mean()),
            "avg_charge_mw": float(hourly["charge_mw"].mean()),
            "avg_discharge_mw": float(hourly["discharge_mw"].mean()),
        }
        if not hourly.empty:
            final_real_soc_mwh = float(hourly["real_soc_mwh"].iloc[-1])
        else:
            final_real_soc_mwh = float(self.soc_init)
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

    out = grouped.agg(
        realized_pnl_eur=("real_pnl_eur", "sum"),
        oracle_pnl_eur=("oracle_pnl_eur", "sum"),
        predicted_pnl_eur=("pred_pnl_eur", "sum"),
        pnl_gap_eur=("pnl_gap_eur", "sum"),
        cost_of_forecast_error_eur=("cost_of_forecast_error_eur", "sum"),
        realized_revenue_da_eur=("real_revenue_da_eur", "sum"),
        realized_cost_da_eur=("real_cost_da_eur", "sum"),
        realized_revenue_capacity_eur=("real_revenue_capacity_eur", "sum"),
        realized_revenue_activation_eur=("real_revenue_activation_eur", "sum"),
        realized_transaction_cost_eur=("real_transaction_cost_eur", "sum"),
        realized_degradation_cost_eur=("real_degradation_cost_eur", "sum"),
        avg_charge_mw=("charge_mw", "mean"),
        avg_discharge_mw=("discharge_mw", "mean"),
        avg_reserve_pos_mw=("reserve_pos_mw", "mean"),
        avg_reserve_neg_mw=("reserve_neg_mw", "mean"),
    ).reset_index()

    out["realized_pnl_cum_eur"] = out["realized_pnl_eur"].cumsum()
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
