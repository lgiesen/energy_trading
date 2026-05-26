from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_trading.simulation.battery_backtest import BacktestColumnMap, BatteryBacktester  # noqa: E402
from energy_trading.simulation.bid_builder import AFRRCapacityBid  # noqa: E402
from energy_trading.simulation.market_clearing import MarketClearingEngine  # noqa: E402


def _mk_backtester() -> BatteryBacktester:
    return BatteryBacktester()


def _one_hour_pred_df(
    *,
    da: float,
    cap_pos: float,
    cap_neg: float,
    act_pos: float,
    act_neg: float,
    rate_pos: float,
    rate_neg: float,
) -> tuple[pd.DataFrame, BacktestColumnMap]:
    col = BacktestColumnMap()
    row = {
        col.timestamp: [pd.Timestamp("2026-01-01T00:00:00Z")],
        col.pred_da_price: [da],
        col.pred_afrr_capacity_price_pos: [cap_pos],
        col.pred_afrr_capacity_price_neg: [cap_neg],
        col.pred_afrr_activation_price_pos: [act_pos],
        col.pred_afrr_activation_price_neg: [act_neg],
        col.pred_afrr_activation_rate_pos: [rate_pos],
        col.pred_afrr_activation_rate_neg: [rate_neg],
    }
    for pref, val in [
        (col.pred_afrr_capacity_price_pos, cap_pos),
        (col.pred_afrr_capacity_price_neg, cap_neg),
        (col.pred_afrr_activation_price_pos, act_pos),
        (col.pred_afrr_activation_price_neg, act_neg),
        (col.pred_afrr_activation_rate_pos, rate_pos),
        (col.pred_afrr_activation_rate_neg, rate_neg),
    ]:
        for q in ["p01", "p05", "p10", "p30", "p50", "p70", "p90", "p95", "p99"]:
            row[f"{pref}_{q}"] = [val]
    return pd.DataFrame(row), col


def _tiny_backtest_df(hours: int = 6) -> tuple[pd.DataFrame, BacktestColumnMap]:
    col = BacktestColumnMap()
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=hours, freq="h")
    data = {
        col.timestamp: ts,
        col.pred_da_price: [0.0] * hours,
        "pred_da_price_p05": [0.0] * hours,
        "pred_da_price_p10": [0.0] * hours,
        "pred_da_price_p90": [0.0] * hours,
        "pred_da_price_p95": [0.0] * hours,
        col.pred_afrr_capacity_price_pos: [0.0] * hours,
        col.pred_afrr_capacity_price_neg: [0.0] * hours,
        col.pred_afrr_activation_price_pos: [0.0] * hours,
        col.pred_afrr_activation_price_neg: [0.0] * hours,
        col.pred_afrr_activation_rate_pos: [0.0] * hours,
        col.pred_afrr_activation_rate_neg: [0.0] * hours,
        col.true_da_price: [0.0] * hours,
        col.true_afrr_capacity_price_pos: [0.0] * hours,
        col.true_afrr_capacity_price_neg: [0.0] * hours,
        col.true_afrr_activation_price_pos: [0.0] * hours,
        col.true_afrr_activation_price_neg: [0.0] * hours,
        col.true_afrr_activation_rate_pos: [0.0] * hours,
        col.true_afrr_activation_rate_neg: [0.0] * hours,
    }
    for pref in [
        col.pred_afrr_capacity_price_pos,
        col.pred_afrr_capacity_price_neg,
        col.pred_afrr_activation_price_pos,
        col.pred_afrr_activation_price_neg,
        col.pred_afrr_activation_rate_pos,
        col.pred_afrr_activation_rate_neg,
    ]:
        for q in ["p01", "p05", "p10", "p30", "p50", "p70", "p90", "p95", "p99"]:
            data[f"{pref}_{q}"] = [0.0] * hours
    return pd.DataFrame(data), col


def test_bcm_pay_as_bid_capacity_settlement_price_and_revenue() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T08:00:00Z"),
        is_oracle=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=5.0,
        planned_reserve_neg_mw=0.0,
        pred_da_price=50.0,
        true_da_price=50.0,
        pred_cap_pos=20.0,
        true_cap_pos=100.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=80.0,
        true_act_pos=80.0,
        pred_act_neg=-100.0,
        true_act_neg=-100.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        obligation_pos_mw=5.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[5.0] + [0.0] * (n_bins - 1),
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[20.0] + [0.0] * (n_bins - 1),
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )
    assert np.isclose(float(out["executed_reserve_pos_mw"]), 5.0)
    assert np.isclose(float(out["settlement_cap_bid_price_pos_eur_mw"]), 20.0)

    _, m = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=0.0,
        discharge=0.0,
        reserve_pos=5.0,
        reserve_neg=0.0,
        da_price=50.0,
        cap_pos=100.0,
        cap_neg=0.0,
        act_pos_price=80.0,
        act_neg_price=-100.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        cap_bid_pos=20.0,
        cap_bid_neg=0.0,
    )
    assert np.isclose(float(m["revenue_capacity_eur"]), 100.0), m


def test_bcm_reject_no_capacity_award_and_no_capacity_revenue() -> None:
    mc = MarketClearingEngine()
    bids = [
        AFRRCapacityBid(
            ts=pd.Timestamp("2026-01-01T08:00:00Z"),
            side="pos",
            quantity_mw=5.0,
            capacity_price_eur_mw=200.0,
            energy_price_eur_mwh=80.0,
        )
    ]
    res = mc.clear_afrr_capacity(bids, true_cap_pos=100.0, true_cap_neg=0.0)
    assert np.isclose(float(res.awarded_pos_mw), 0.0)
    assert bool(res.pos_awarded) is False


def test_aux_cost_subtracted_once_in_pnl() -> None:
    bt = _mk_backtester()
    _, m = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=0.0,
        discharge=0.0,
        reserve_pos=1.0,
        reserve_neg=0.0,
        da_price=100.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        cap_bid_pos=0.0,
        cap_bid_neg=0.0,
    )
    assert float(m["aux_cost_eur"]) > 0.0
    expected_pnl = (
        float(m["revenue_da_eur"])
        - float(m["cost_da_eur"])
        + float(m["revenue_capacity_eur"])
        + float(m["revenue_activation_eur"])
        + float(m["revenue_id_eur"])
        - float(m["cost_id_eur"])
        - float(m["transaction_cost_eur"])
        - float(m["aux_cost_eur"])
        - float(m["degradation_cost_eur"])
        - float(m["penalty_eur"])
    )
    assert np.isclose(float(m["pnl_eur"]), expected_pnl), m


def test_da_revenue_formula() -> None:
    bt = _mk_backtester()
    _, m = bt._settle_one_hour(
        soc=bt.soc_max,
        charge=0.0,
        discharge=5.0,
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=100.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
    )
    expected = float(m["da_sell_mwh"]) * 100.0
    assert np.isclose(float(m["revenue_da_eur"]), expected, atol=1e-9)


def test_negative_activation_delivered_revenue_sign_convention() -> None:
    bt = _mk_backtester()
    _, m = bt._settle_one_hour(
        soc=bt.soc_min,
        charge=0.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=10.0,
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=-1000.0,
        act_pos_rate=0.0,
        act_neg_rate=1.0,
        cap_bid_pos=0.0,
        cap_bid_neg=0.0,
    )
    # Project convention: revenue_neg = -activated_neg_mwh * act_neg_price.
    delivered = float(m["delivered_activation_neg_mwh"])
    assert delivered > 0.0
    assert np.isclose(float(m["revenue_activation_eur"]), -delivered * (-1000.0), atol=1e-6), m


def test_bem_positive_activation_formula() -> None:
    bt = _mk_backtester()
    _, m = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=0.0,
        discharge=0.0,
        reserve_pos=10.0,
        reserve_neg=0.0,
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=140.0,
        act_neg_price=0.0,
        act_pos_rate=0.5,
        act_neg_rate=0.0,
    )
    expected_mwh = 10.0 * 0.5 * bt.dt_h * bt.eta_out
    assert np.isclose(float(m["delivered_activation_pos_mwh"]), expected_mwh, atol=1e-9)
    assert np.isclose(float(m["revenue_activation_eur"]), expected_mwh * 140.0, atol=1e-9)


def test_negative_missed_activation_penalty_is_positive() -> None:
    bt = _mk_backtester()
    _, m = bt._settle_one_hour(
        soc=bt.soc_max,
        charge=0.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=10.0,
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=-1000.0,
        act_pos_rate=0.0,
        act_neg_rate=1.0,
        cap_bid_pos=0.0,
        cap_bid_neg=0.0,
    )
    assert float(m["missed_activation_neg_mwh"]) > 0.0
    assert np.isclose(float(m["penalty_activation_basis_neg_eur_mwh"]), 1000.0, atol=1e-9)
    assert float(m["penalty_activation_neg_eur"]) > 0.0


def test_degradation_cost_subtracted_once() -> None:
    bt = _mk_backtester()
    _, m = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=5.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
    )
    expected = (
        float(m["revenue_da_eur"]) - float(m["cost_da_eur"]) + float(m["revenue_capacity_eur"])
        + float(m["revenue_activation_eur"]) + float(m["revenue_id_eur"]) - float(m["cost_id_eur"])
        - float(m["transaction_cost_eur"]) - float(m["aux_cost_eur"]) - float(m["degradation_cost_eur"])
        - float(m["penalty_eur"])
    )
    assert np.isclose(float(m["pnl_eur"]), expected, atol=1e-9)


def test_idle_case_optimizer_can_choose_zero_volume() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos=0.0,
        act_neg=0.0,
        rate_pos=0.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("DA", "aFRR"))
    assert np.isclose(float(out["charge_mw"].iloc[0]), 0.0)
    assert np.isclose(float(out["discharge_mw"].iloc[0]), 0.0)
    assert np.isclose(float(out["reserve_pos_mw"].iloc[0]), 0.0)
    assert np.isclose(float(out["reserve_neg_mw"].iloc[0]), 0.0)


def test_da_only_dominance_prefers_da_discharge() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=400.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos=0.0,
        act_neg=0.0,
        rate_pos=0.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("DA",))
    assert float(out["discharge_mw"].iloc[0]) > 0.0
    assert np.isclose(float(out["reserve_pos_mw"].iloc[0]), 0.0)
    assert np.isclose(float(out["reserve_neg_mw"].iloc[0]), 0.0)


def test_bcm_plus_bem_activation_revenue_no_double_counting_in_settlement() -> None:
    bt = _mk_backtester()
    _, m = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=0.0,
        discharge=0.0,
        reserve_pos=4.0,
        reserve_neg=0.0,
        da_price=0.0,
        cap_pos=25.0,
        cap_neg=0.0,
        act_pos_price=200.0,
        act_neg_price=-100.0,
        act_pos_rate=0.5,
        act_neg_rate=0.0,
        cap_bid_pos=25.0,
        cap_bid_neg=0.0,
    )
    assert np.isclose(float(m["revenue_capacity_eur"]), 100.0, atol=1e-9)
    delivered_pos = float(m["delivered_activation_pos_mwh"])
    assert delivered_pos > 0.0
    assert np.isclose(float(m["revenue_activation_eur"]), delivered_pos * 200.0, atol=1e-6)


def test_physical_exclusivity_limit_in_optimizer() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=500.0,
        cap_pos=300.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=0.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("DA", "aFRR"))
    dis = float(out["discharge_mw"].iloc[0])
    rpos = float(out["reserve_pos_mw"].iloc[0])
    assert dis + rpos <= bt.p_max_mw + 1e-6


def test_bem_only_mode_is_explicit_optimizer() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=2, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    assert out.summary.get("bem_only_mode") == "explicit_optimizer"
    assert float(out.summary.get("bem_only_explicit_optimizer", 0.0)) == 1.0


def test_comparable_benchmark_semantics_in_tiny_deterministic_case() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=2, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    assert out.summary.get("comparable_benchmark_type") == "rolling_perfect_foresight_same_rules"
    assert "comparable_perfect_foresight_market_pnl_eur" in out.summary


def test_bem_only_positive_activation_without_bcm_award_is_activation_revenue_only() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        is_oracle=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=3.0,
        planned_bem_only_neg_mw=0.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=999.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=50.0,
        true_act_pos=100.0,
        pred_act_neg=-10.0,
        true_act_neg=-50.0,
        true_rate_pos=1.0,
        true_rate_neg=0.0,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[0.0] * n_bins,
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[999.0] + [0.0] * (n_bins - 1),
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )
    assert np.isclose(float(out["afrr_bcm_auction_cleared"]), 0.0)
    assert np.isclose(float(out["settlement_cap_bid_price_pos_eur_mw"]), 0.0)
    assert float(out["bem_only_submitted_pos_mw"]) > 0.0
    assert float(out["executed_reserve_pos_mw"]) > 0.0
    assert float(out["executed_rate_pos"]) > 0.0
    _, m = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=float(out["executed_charge_mw"]),
        discharge=float(out["executed_discharge_mw"]),
        reserve_pos=float(out["executed_reserve_pos_mw"]),
        reserve_neg=float(out["executed_reserve_neg_mw"]),
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=100.0,
        act_neg_price=-50.0,
        act_pos_rate=float(out["executed_rate_pos"]),
        act_neg_rate=float(out["executed_rate_neg"]),
        cap_bid_pos=float(out["settlement_cap_bid_price_pos_eur_mw"]),
        cap_bid_neg=float(out["settlement_cap_bid_price_neg_eur_mw"]),
    )
    assert np.isclose(float(m["revenue_capacity_eur"]), 0.0, atol=1e-9)
    assert float(m["revenue_activation_eur"]) >= 0.0


def test_bem_only_negative_activation_sign_without_bcm_award() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        is_oracle=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=0.0,
        planned_bem_only_neg_mw=2.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=999.0,
        true_cap_neg=0.0,
        pred_act_pos=10.0,
        true_act_pos=10.0,
        pred_act_neg=-100.0,
        true_act_neg=-1000.0,
        true_rate_pos=0.0,
        true_rate_neg=1.0,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[0.0] * n_bins,
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[0.0] * n_bins,
        pred_cap_neg_bins_eur_mw=[999.0] + [0.0] * (n_bins - 1),
    )
    assert np.isclose(float(out["afrr_bcm_auction_cleared"]), 0.0)
    assert float(out["bem_only_submitted_neg_mw"]) > 0.0
    _, m = bt._settle_one_hour(
        soc=bt.soc_min,
        charge=float(out["executed_charge_mw"]),
        discharge=float(out["executed_discharge_mw"]),
        reserve_pos=float(out["executed_reserve_pos_mw"]),
        reserve_neg=float(out["executed_reserve_neg_mw"]),
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=10.0,
        act_neg_price=-1000.0,
        act_pos_rate=float(out["executed_rate_pos"]),
        act_neg_rate=float(out["executed_rate_neg"]),
        cap_bid_pos=0.0,
        cap_bid_neg=0.0,
    )
    assert np.isclose(float(m["revenue_capacity_eur"]), 0.0, atol=1e-9)
    if float(m["delivered_activation_neg_mwh"]) > 0.0:
        assert float(m["revenue_activation_eur"]) > 0.0


def test_bem_only_summary_fields_present() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=2, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    required = [
        "bem_only_mode",
        "bem_only_explicit_optimizer",
        "bem_only_hours",
        "bem_only_pos_bid_hours",
        "bem_only_neg_bid_hours",
        "bem_only_pos_bid_mw_sum",
        "bem_only_neg_bid_mw_sum",
        "bem_only_pos_activation_accept_count",
        "bem_only_neg_activation_accept_count",
        "bem_only_pos_activation_mwh_sum",
        "bem_only_neg_activation_mwh_sum",
        "bem_only_activation_revenue_eur",
    ]
    for k in required:
        assert k in out.summary


def test_bem_only_positive_dominance_optimizer_decision() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=-200.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=-10.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("DA", "aFRR"))
    assert "bem_only_pos_mw" in out.columns
    assert float(out["bem_only_pos_mw"].iloc[0]) > 0.0


def test_bem_only_negative_dominance_optimizer_decision() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos=0.0,
        act_neg=-300.0,
        rate_pos=0.0,
        rate_neg=1.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("DA", "aFRR"))
    assert "bem_only_neg_mw" in out.columns
    assert float(out["bem_only_neg_mw"].iloc[0]) > 0.0


def test_bem_only_vs_da_exclusivity() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=500.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos=500.0,
        act_neg=0.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("DA", "aFRR"))
    dis = float(out["discharge_mw"].iloc[0])
    bem = float(out["bem_only_pos_mw"].iloc[0])
    assert dis + bem <= bt.p_max_mw + 1e-6


def test_bem_only_positive_revenue_decomposition() -> None:
    bt = _mk_backtester()
    comp = bt._split_activation_revenue_components(
        delivered_pos_mwh=10.0,
        delivered_neg_mwh=0.0,
        bem_only_pos_mwh=10.0,
        bem_only_neg_mwh=0.0,
        act_pos_price_eur_mwh=140.0,
        act_neg_price_eur_mwh=0.0,
    )
    assert np.isclose(comp["bem_only_activation_revenue_eur"], 1400.0)
    assert np.isclose(comp["bcm_linked_activation_revenue_eur"], 0.0)
    assert np.isclose(comp["activation_revenue_reconciled_eur"], 1400.0)


def test_bem_only_negative_revenue_decomposition() -> None:
    bt = _mk_backtester()
    comp = bt._split_activation_revenue_components(
        delivered_pos_mwh=0.0,
        delivered_neg_mwh=10.0,
        bem_only_pos_mwh=0.0,
        bem_only_neg_mwh=10.0,
        act_pos_price_eur_mwh=0.0,
        act_neg_price_eur_mwh=-1000.0,
    )
    assert np.isclose(comp["bem_only_activation_revenue_eur"], 10000.0)
    assert np.isclose(comp["bcm_linked_activation_revenue_eur"], 0.0)
    assert np.isclose(comp["activation_revenue_reconciled_eur"], 10000.0)


def test_bem_only_not_double_counted_activation_revenue() -> None:
    bt = _mk_backtester()
    comp = bt._split_activation_revenue_components(
        delivered_pos_mwh=12.0,
        delivered_neg_mwh=6.0,
        bem_only_pos_mwh=5.0,
        bem_only_neg_mwh=2.0,
        act_pos_price_eur_mwh=100.0,
        act_neg_price_eur_mwh=-50.0,
    )
    lhs = comp["activation_revenue_reconciled_eur"]
    rhs = comp["bcm_linked_activation_revenue_eur"] + comp["bem_only_activation_revenue_eur"]
    assert np.isclose(lhs, rhs)


def test_bem_only_forced_scenario_has_nonzero_revenue_and_benchmark_diagnostics() -> None:
    bt = _mk_backtester()
    n = 48
    idx = pd.date_range("2025-05-01 00:00:00+00:00", periods=n, freq="h", tz="UTC")
    col = BacktestColumnMap()
    df = pd.DataFrame(
        {
            col.timestamp: idx,
            col.pred_da_price: np.full(n, -20.0),
            f"{col.pred_da_price}_p05": np.full(n, -25.0),
            f"{col.pred_da_price}_p10": np.full(n, -23.0),
            f"{col.pred_da_price}_p90": np.full(n, -17.0),
            f"{col.pred_da_price}_p95": np.full(n, -15.0),
            col.pred_afrr_capacity_price_pos: np.full(n, 1.0),
            col.pred_afrr_capacity_price_neg: np.full(n, 1.0),
            col.pred_afrr_activation_price_pos: np.full(n, 140.0),
            col.pred_afrr_activation_price_neg: np.full(n, 10.0),
            col.pred_afrr_activation_rate_pos: np.full(n, 0.95),
            col.pred_afrr_activation_rate_neg: np.full(n, 0.05),
            col.true_da_price: np.full(n, -18.0),
            col.true_afrr_capacity_price_pos: np.full(n, 0.5),
            col.true_afrr_capacity_price_neg: np.full(n, 0.5),
            col.true_afrr_activation_price_pos: np.full(n, 150.0),
            col.true_afrr_activation_price_neg: np.full(n, 20.0),
            col.true_afrr_activation_rate_pos: np.full(n, 0.95),
            col.true_afrr_activation_rate_neg: np.full(n, 0.05),
        }
    )
    prev_eps = os.environ.get("BACKTEST_ORACLE_UPPER_BOUND_EPS")
    os.environ["BACKTEST_ORACLE_UPPER_BOUND_EPS"] = "1e12"
    try:
        out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=24, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    finally:
        if prev_eps is None:
            os.environ.pop("BACKTEST_ORACLE_UPPER_BOUND_EPS", None)
        else:
            os.environ["BACKTEST_ORACLE_UPPER_BOUND_EPS"] = prev_eps
    h = out.hourly
    def _series_or_zero(col: str) -> pd.Series:
        if col in h.columns:
            return pd.to_numeric(h[col], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=h.index)
    pos_mwh = _series_or_zero("real_bem_only_executed_pos_mwh")
    bem_rev = _series_or_zero("real_bem_only_activation_revenue_eur")
    cap_rev = _series_or_zero("real_revenue_capacity_eur")
    da_mw = _series_or_zero("real_da_net_mw")
    cap_pos = _series_or_zero("real_afrr_bcm_executed_pos_mw")
    cap_neg = _series_or_zero("real_afrr_bcm_executed_neg_mw")
    bem_sub_pos = _series_or_zero("real_bem_only_submitted_pos_mw")
    bem_sub_neg = _series_or_zero("real_bem_only_submitted_neg_mw")
    pure_mask = (bem_sub_pos + bem_sub_neg > 1e-12) & (da_mw.abs() < 1e-12) & ((cap_pos + cap_neg).abs() < 1e-12)
    assert float(pos_mwh.sum()) > 0.0
    assert float(bem_rev.sum()) > 0.0
    assert np.isclose(float(cap_rev[pure_mask].sum()), 0.0, atol=1e-9)
    real_cmp = float(out.summary["comparable_realized_market_pnl_eur"])
    oracle_cmp = float(out.summary["comparable_rolling_perfect_foresight_same_rules_market_pnl_eur"])
    assert np.isfinite(real_cmp)
    assert np.isfinite(oracle_cmp)
    assert out.summary.get("comparable_benchmark_type") == "rolling_perfect_foresight_same_rules"


def test_no_headroom_violations_in_clean_forced_bem_only() -> None:
    bt = _mk_backtester()
    n = 24
    idx = pd.date_range("2025-05-01 00:00:00+00:00", periods=n, freq="h", tz="UTC")
    col = BacktestColumnMap()
    df = pd.DataFrame(
        {
            col.timestamp: idx,
            col.pred_da_price: np.full(n, -20.0),
            f"{col.pred_da_price}_p05": np.full(n, -25.0),
            f"{col.pred_da_price}_p10": np.full(n, -23.0),
            f"{col.pred_da_price}_p90": np.full(n, -17.0),
            f"{col.pred_da_price}_p95": np.full(n, -15.0),
            col.pred_afrr_capacity_price_pos: np.full(n, 0.5),
            col.pred_afrr_capacity_price_neg: np.full(n, 0.5),
            col.pred_afrr_activation_price_pos: np.full(n, 140.0),
            col.pred_afrr_activation_price_neg: np.full(n, 10.0),
            col.pred_afrr_activation_rate_pos: np.full(n, 0.95),
            col.pred_afrr_activation_rate_neg: np.full(n, 0.05),
            col.true_da_price: np.full(n, -18.0),
            col.true_afrr_capacity_price_pos: np.full(n, 0.5),
            col.true_afrr_capacity_price_neg: np.full(n, 0.5),
            col.true_afrr_activation_price_pos: np.full(n, 150.0),
            col.true_afrr_activation_price_neg: np.full(n, 20.0),
            col.true_afrr_activation_rate_pos: np.full(n, 0.95),
            col.true_afrr_activation_rate_neg: np.full(n, 0.05),
        }
    )
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=24, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    hv = float(out.summary.get("headroom_violation_count", 0.0))
    hmax = float(out.summary.get("headroom_violation_max_mwh", 0.0))
    assert hv <= 1e-9
    assert hmax <= 1e-9


def test_id_is_price_taker() -> None:
    bt = _mk_backtester()
    _, m = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=0.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=100.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        id_charge_mw=1.0,
        id_discharge_mw=0.0,
    )
    assert float(m["id_buy_mwh"]) > 0.0
    # ID cost follows modeled realized settlement price path directly.
    assert float(m["cost_id_eur"]) > 0.0


def test_bcm_positive_headroom_30min() -> None:
    bt = _mk_backtester()
    # Need >= reserve*0.5/eta_out above soc_min for 10 MW reserve.
    req = 10.0 * bt.reserve_activation_headroom_h / bt.eta_out
    soc_start = bt.soc_min + req - 0.2
    df, col = _one_hour_pred_df(
        da=0.0, cap_pos=300.0, cap_neg=0.0, act_pos=100.0, act_neg=0.0, rate_pos=1.0, rate_neg=0.0
    )
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=soc_start,
        allowed_markets=("aFRR",),
    )
    assert float(out["reserve_pos_mw"].iloc[0]) < 10.0


def test_bem_only_headroom_fields_present() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=-100.0, cap_pos=0.0, cap_neg=0.0, act_pos=400.0, act_neg=-10.0, rate_pos=1.0, rate_neg=0.0
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("DA", "aFRR"))
    for c in [
        "required_headroom_pos_mwh",
        "required_headroom_neg_mwh",
        "available_headroom_pos_mwh",
        "available_headroom_neg_mwh",
        "headroom_margin_pos_mwh",
        "headroom_margin_neg_mwh",
        "power_stack_pos_mw",
        "power_stack_neg_mw",
    ]:
        assert c in out.columns


def test_bcm_negative_headroom_30min() -> None:
    bt = _mk_backtester()
    req = 10.0 * bt.reserve_activation_headroom_h * bt.eta_in
    soc_start = bt.soc_max - req + 0.2
    df, col = _one_hour_pred_df(
        da=0.0, cap_pos=0.0, cap_neg=300.0, act_pos=0.0, act_neg=-100.0, rate_pos=0.0, rate_neg=1.0
    )
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=soc_start,
        allowed_markets=("aFRR",),
    )
    assert float(out["reserve_neg_mw"].iloc[0]) < 10.0


def test_headroom_with_efficiency() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.9
    bt.eta_out = 0.9
    df, col = _one_hour_pred_df(
        da=0.0, cap_pos=100.0, cap_neg=100.0, act_pos=100.0, act_neg=-100.0, rate_pos=1.0, rate_neg=1.0
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("aFRR",))
    row = out.iloc[0]
    pos_req_expected = (
        float(row["reserve_pos_mw"]) * bt.reserve_activation_headroom_h
        + float(row.get("bem_only_pos_mw", 0.0)) * bt.bem_activation_headroom_h
    ) / bt.eta_out
    neg_req_expected = (
        float(row["reserve_neg_mw"]) * bt.reserve_activation_headroom_h
        + float(row.get("bem_only_neg_mw", 0.0)) * bt.bem_activation_headroom_h
    ) * bt.eta_in
    assert np.isclose(float(row["required_headroom_pos_mwh"]), pos_req_expected, atol=1e-9)
    assert np.isclose(float(row["required_headroom_neg_mwh"]), neg_req_expected, atol=1e-9)


def test_bem_only_positive_headroom() -> None:
    bt = _mk_backtester()
    req = 10.0 * bt.bem_activation_headroom_h / bt.eta_out
    soc_start = bt.soc_min + req - 0.2
    df, col = _one_hour_pred_df(
        da=-100.0, cap_pos=0.0, cap_neg=0.0, act_pos=500.0, act_neg=-10.0, rate_pos=1.0, rate_neg=0.0
    )
    out = bt.optimize_dispatch(df, col, soc_start=soc_start, allowed_markets=("DA", "aFRR"))
    assert float(out["bem_only_pos_mw"].iloc[0]) < 10.0


def test_bem_only_negative_headroom() -> None:
    bt = _mk_backtester()
    req = 10.0 * bt.bem_activation_headroom_h * bt.eta_in
    soc_start = bt.soc_max - req + 0.2
    df, col = _one_hour_pred_df(
        da=100.0, cap_pos=0.0, cap_neg=0.0, act_pos=0.0, act_neg=-500.0, rate_pos=0.0, rate_neg=1.0
    )
    out = bt.optimize_dispatch(df, col, soc_start=soc_start, allowed_markets=("DA", "aFRR"))
    assert float(out["bem_only_neg_mw"].iloc[0]) < 10.0


def test_no_ex_post_missed_capacity_when_headroom_feasible() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=3)
    df[col.pred_afrr_capacity_price_pos] = 200.0
    df[col.pred_afrr_activation_price_pos] = 150.0
    df[col.pred_afrr_activation_rate_pos] = 1.0
    df[col.true_afrr_capacity_price_pos] = 200.0
    df[col.true_afrr_activation_price_pos] = 150.0
    df[col.true_afrr_activation_rate_pos] = 1.0
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=3, reopt_step_hours=1, allowed_markets=("aFRR",))
    h = out.hourly
    assert np.isclose(float(pd.to_numeric(h.get("real_missed_activation_pos_mwh", 0.0), errors="coerce").fillna(0.0).sum()), 0.0, atol=1e-9)
    assert np.isclose(float(pd.to_numeric(h.get("real_missed_activation_neg_mwh", 0.0), errors="coerce").fillna(0.0).sum()), 0.0, atol=1e-9)


def test_audit_fields_present() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=3)
    df[col.pred_da_price] = 10.0
    df[col.pred_afrr_capacity_price_pos] = 10.0
    df[col.pred_afrr_capacity_price_neg] = 10.0
    df[col.pred_afrr_activation_price_pos] = 20.0
    df[col.pred_afrr_activation_price_neg] = -20.0
    df[col.pred_afrr_activation_rate_pos] = 0.2
    df[col.pred_afrr_activation_rate_neg] = 0.2
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=3, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    h = out.hourly
    required = [
        "required_headroom_pos_mwh",
        "required_headroom_neg_mwh",
        "power_stack_pos_mw",
        "power_stack_neg_mw",
        "real_revenue_capacity_eur",
        "real_revenue_activation_eur",
        "real_pnl_eur",
        "afrr_cap_pos_awarded_mw",
        "afrr_cap_neg_awarded_mw",
        "delivered_activation_pos_mwh",
        "delivered_activation_neg_mwh",
        "real_afrr_cap_pos_awarded_mw",
        "real_afrr_cap_neg_awarded_mw",
        "real_delivered_activation_pos_mwh",
        "real_delivered_activation_neg_mwh",
        "real_required_headroom_pos_mwh",
        "real_required_headroom_neg_mwh",
    ]
    for c in required:
        assert c in h.columns
    s = out.summary
    for k in [
        "id_price_taker",
        "reserve_activation_headroom_h",
        "bem_activation_headroom_h",
        "headroom_violation_count",
        "headroom_violation_max_mwh",
    ]:
        assert k in s


def test_pnl_reconciliation() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=3)
    df[col.pred_da_price] = 50.0
    df[col.true_da_price] = 50.0
    df[col.pred_afrr_capacity_price_pos] = 20.0
    df[col.pred_afrr_capacity_price_neg] = 5.0
    df[col.pred_afrr_activation_price_pos] = 30.0
    df[col.pred_afrr_activation_price_neg] = -10.0
    df[col.pred_afrr_activation_rate_pos] = 0.3
    df[col.pred_afrr_activation_rate_neg] = 0.2
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=3, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    h = out.hourly
    pnl = pd.to_numeric(h["real_pnl_eur"], errors="coerce").fillna(0.0)
    rev_da = pd.to_numeric(h["real_revenue_da_eur"], errors="coerce").fillna(0.0)
    rev_id = pd.to_numeric(h["real_revenue_id_eur"], errors="coerce").fillna(0.0)
    rev_cap = pd.to_numeric(h["real_revenue_capacity_eur"], errors="coerce").fillna(0.0)
    rev_act = pd.to_numeric(h["real_revenue_activation_eur"], errors="coerce").fillna(0.0)
    cost_deg = pd.to_numeric(h["real_degradation_cost_eur"], errors="coerce").fillna(0.0)
    cost_aux = pd.to_numeric(h["real_aux_cost_eur"], errors="coerce").fillna(0.0)
    penalties = pd.to_numeric(h["real_penalty_eur"], errors="coerce").fillna(0.0)
    recon = rev_da + rev_id + rev_cap + rev_act - cost_deg - cost_aux - penalties
    assert np.isclose(float((pnl - recon).abs().max()), 0.0, atol=1e-6)
    assert "real_pnl_reconciliation_error_eur" in h.columns
    assert float(pd.to_numeric(h["real_pnl_reconciliation_error_eur"], errors="coerce").fillna(0.0).max()) <= 1e-6


def test_headroom_audit_matches_constraint_timing() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=0.0,
        cap_pos=250.0,
        cap_neg=0.0,
        act_pos=120.0,
        act_neg=0.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, soc_start=bt.soc_min + 6.0, allowed_markets=("aFRR",))
    r = out.iloc[0]
    expected_avail = float(r["soc_start_lp_mwh"]) - bt.soc_min
    expected_req = (
        float(r["reserve_pos_mw"]) * bt.reserve_activation_headroom_h
        + float(r.get("bem_only_pos_mw", 0.0)) * bt.bem_activation_headroom_h
    ) / bt.eta_out
    assert np.isclose(float(r["available_headroom_pos_mwh"]), expected_avail, atol=1e-9)
    assert np.isclose(float(r["required_headroom_pos_mwh"]), expected_req, atol=1e-9)
