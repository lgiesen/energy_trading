import math

import pandas as pd
import pytest

from energy_trading.simulation.battery_backtest import (
    BacktestColumnMap,
    BatteryBacktester,
    StrategyPermissions,
    clear_bcm_capacity_bid,
)


def test_clear_bcm_capacity_bid_reasons() -> None:
    assert clear_bcm_capacity_bid(0.0, 50.0, 40.0) == (0.0, 0.0, "no_bid")
    assert clear_bcm_capacity_bid(2.0, 40.0, 40.0) == (2.0, 0.0, "cleared")
    assert clear_bcm_capacity_bid(2.0, 41.0, 40.0) == (0.0, 2.0, "price_rejected")
    assert clear_bcm_capacity_bid(2.0, math.nan, 40.0) == (0.0, 2.0, "missing_bid_price")
    assert clear_bcm_capacity_bid(2.0, 40.0, math.nan) == (0.0, 2.0, "missing_clearing_price")


def _bcm_backtester() -> BatteryBacktester:
    bt = BatteryBacktester()
    bt.bcm_capacity_clearing_mode = "expost_price_threshold"
    bt._strategy_permissions = StrategyPermissions(
        allow_da=False,
        allow_bcm=True,
        allow_bcm_activation_obligations=True,
        allow_bem_only=False,
        id_mode="technical_repair",
    )
    return bt


def _market_clearing_kwargs(
    *,
    side: str,
    submitted_mw: float,
    bid_price: float,
    market_price: float,
) -> dict[str, object]:
    is_pos = side == "pos"
    return {
        "target_time_utc": pd.Timestamp("2025-06-12T17:00:00Z"),
        "is_perfect_foresight": False,
        "planned_charge_mw": 0.0,
        "planned_discharge_mw": 0.0,
        "planned_reserve_pos_mw": 0.0,
        "planned_reserve_neg_mw": 0.0,
        "pred_da_price": 0.0,
        "true_da_price": 0.0,
        "pred_cap_pos": bid_price if is_pos else 0.0,
        "true_cap_pos": market_price if is_pos else 0.0,
        "pred_cap_neg": bid_price if not is_pos else 0.0,
        "true_cap_neg": market_price if not is_pos else 0.0,
        "pred_act_pos": 100.0,
        "true_act_pos": 100.0,
        "pred_act_neg": 100.0,
        "true_act_neg": 100.0,
        "true_rate_pos": 0.5 if is_pos else 0.0,
        "true_rate_neg": 0.5 if not is_pos else 0.0,
        "pred_rate_pos": 0.5 if is_pos else 0.0,
        "pred_rate_neg": 0.5 if not is_pos else 0.0,
        "soc_now": 10.0,
        "settlement_soc_now": 10.0,
        "obligation_pos_mw": submitted_mw if is_pos else 0.0,
        "obligation_neg_mw": submitted_mw if not is_pos else 0.0,
        "obligation_capacity_price_pos": bid_price if is_pos else None,
        "obligation_capacity_price_neg": bid_price if not is_pos else None,
    }


def test_rejected_bcm_capacity_creates_no_capacity_or_activation_obligation() -> None:
    bt = _bcm_backtester()

    out = bt._apply_market_clearing(
        **_market_clearing_kwargs(side="pos", submitted_mw=2.0, bid_price=50.0, market_price=10.0)
    )

    assert float(out["submitted_bcm_capacity_pos_mw"]) == pytest.approx(2.0)
    assert float(out["bcm_capacity_awarded_pos_mw"]) == pytest.approx(0.0)
    assert float(out["bcm_capacity_rejected_pos_mw"]) == pytest.approx(2.0)
    assert str(out["bcm_capacity_rejection_reason_pos"]) == "price_rejected"
    assert float(out["locked_bcm_capacity_pos_mw"]) == pytest.approx(0.0)
    assert float(out["bcm_locked_pos_mw"]) == pytest.approx(0.0)
    assert float(out["bcm_available_capacity_pos_mw"]) == pytest.approx(0.0)
    assert float(out["bcm_capacity_accepted_mw"]) == pytest.approx(0.0)
    assert float(out["executed_bcm_capacity_pos_mw"]) == pytest.approx(0.0)
    assert float(out["fixed_reserve_obligation_pos_mw"]) == pytest.approx(0.0)
    assert float(out["bcm_to_bem_energy_obligation_pos_mw"]) == pytest.approx(0.0)
    assert float(out["bcm_linked_pos_activation_mwh"]) == pytest.approx(0.0)
    assert float(out["settlement_cap_bid_price_pos_eur_mw"]) == pytest.approx(0.0)


@pytest.mark.parametrize("side", ["pos", "neg"])
def test_rejected_bcm_capacity_is_not_written_to_settlement_lockbook_aliases(side: str) -> None:
    bt = _bcm_backtester()

    out = bt._apply_market_clearing(
        **_market_clearing_kwargs(side=side, submitted_mw=3.0, bid_price=55.0, market_price=20.0)
    )

    assert float(out[f"submitted_bcm_capacity_{side}_mw"]) == pytest.approx(3.0)
    assert float(out[f"bcm_capacity_rejected_{side}_mw"]) == pytest.approx(3.0)
    assert str(out[f"bcm_capacity_rejection_reason_{side}"]) == "price_rejected"

    # These are the accepted/settlement lockbook aliases. They must stay zero
    # when the submitted BCM capacity bid is above the realized capacity price.
    assert float(out[f"bcm_capacity_awarded_{side}_mw"]) == pytest.approx(0.0)
    assert float(out[f"locked_bcm_capacity_{side}_mw"]) == pytest.approx(0.0)
    assert float(out[f"bcm_locked_{side}_mw"]) == pytest.approx(0.0)
    assert float(out[f"bcm_awarded_capacity_{side}_mw"]) == pytest.approx(0.0)
    assert float(out[f"bcm_available_capacity_{side}_mw"]) == pytest.approx(0.0)
    assert float(out[f"bcm_to_bem_energy_obligation_{side}_mw"]) == pytest.approx(0.0)
    assert float(out[f"fixed_reserve_obligation_{side}_mw"]) == pytest.approx(0.0)
    assert float(out[f"executed_bcm_capacity_{side}_mw"]) == pytest.approx(0.0)
    assert float(out[f"settlement_cap_bid_price_{side}_eur_mw"]) == pytest.approx(0.0)


def test_equal_price_bcm_capacity_clears_and_creates_activation_obligation() -> None:
    bt = _bcm_backtester()

    out = bt._apply_market_clearing(
        **_market_clearing_kwargs(side="pos", submitted_mw=2.0, bid_price=10.0, market_price=10.0)
    )

    assert str(out["bcm_capacity_rejection_reason_pos"]) == "cleared"
    assert float(out["bcm_capacity_awarded_pos_mw"]) == pytest.approx(2.0)
    assert float(out["locked_bcm_capacity_pos_mw"]) == pytest.approx(2.0)
    assert float(out["bcm_locked_pos_mw"]) == pytest.approx(2.0)
    assert float(out["bcm_available_capacity_pos_mw"]) == pytest.approx(2.0)
    assert float(out["executed_bcm_capacity_pos_mw"]) == pytest.approx(2.0)
    assert float(out["fixed_reserve_obligation_pos_mw"]) == pytest.approx(2.0)
    assert float(out["bcm_to_bem_energy_obligation_pos_mw"]) == pytest.approx(2.0)
    assert float(out["bcm_linked_pos_activation_mwh"]) == pytest.approx(1.0)
    assert float(out["settlement_cap_bid_price_pos_eur_mw"]) == pytest.approx(10.0)


def _capacity_lockbook_frame(*, side: str, bid_price: float, market_price: float) -> pd.DataFrame:
    colmap = BacktestColumnMap()
    target_times = pd.date_range("2025-06-12T22:00:00Z", periods=4, freq="h")
    reserve_pos = 2.0 if side == "pos" else 0.0
    reserve_neg = 2.0 if side == "neg" else 0.0
    pred_pos = bid_price if side == "pos" else 0.0
    pred_neg = bid_price if side == "neg" else 0.0
    true_pos = market_price if side == "pos" else 0.0
    true_neg = market_price if side == "neg" else 0.0
    return pd.DataFrame(
        {
            "target_time_utc": target_times,
            "reserve_pos_mw": reserve_pos,
            "reserve_neg_mw": reserve_neg,
            colmap.pred_afrr_capacity_price_pos: pred_pos,
            colmap.pred_afrr_capacity_price_neg: pred_neg,
            colmap.true_afrr_capacity_price_pos: true_pos,
            colmap.true_afrr_capacity_price_neg: true_neg,
            colmap.pred_afrr_activation_price_pos: 100.0,
            colmap.pred_afrr_activation_price_neg: 100.0,
            colmap.true_afrr_activation_price_pos: 100.0,
            colmap.true_afrr_activation_price_neg: 100.0,
            colmap.pred_afrr_activation_rate_pos: 0.5,
            colmap.pred_afrr_activation_rate_neg: 0.5,
            colmap.true_afrr_activation_rate_pos: 0.5,
            colmap.true_afrr_activation_rate_neg: 0.5,
            "soc_mwh": 10.0,
        }
    )


@pytest.mark.parametrize(
    ("side", "bid_price", "market_price", "expected_locked_mw", "expected_price"),
    [
        ("neg", 9.0, 10.0, 2.0, 9.0),
        ("neg", 10.0, 10.0, 2.0, 10.0),
        ("neg", 11.0, 10.0, 0.0, 0.0),
    ],
)
def test_bcm_capacity_precommit_lockbook_uses_realized_capacity_price_cutoff(
    side: str,
    bid_price: float,
    market_price: float,
    expected_locked_mw: float,
    expected_price: float,
) -> None:
    bt = _bcm_backtester()
    colmap = BacktestColumnMap()
    snapshot_ts = pd.Timestamp("2025-06-12T06:00:00Z")  # 08:00 Europe/Berlin BCM gate.
    plan = _capacity_lockbook_frame(side=side, bid_price=bid_price, market_price=market_price)
    source = plan.set_index("target_time_utc", drop=False)
    source.index.name = None
    lock_pos: dict[pd.Timestamp, float] = {}
    lock_neg: dict[pd.Timestamp, float] = {}
    lock_energy_pos: dict[pd.Timestamp, float] = {}
    lock_energy_neg: dict[pd.Timestamp, float] = {}
    lock_capacity_price_pos: dict[pd.Timestamp, float] = {}
    lock_capacity_price_neg: dict[pd.Timestamp, float] = {}
    audit: dict[str, dict[pd.Timestamp, float | str]] = {}

    result = bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=snapshot_ts,
        snapshot_plan=plan,
        source=source,
        colmap=colmap,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos=lock_energy_pos,
        lock_energy_neg=lock_energy_neg,
        lock_capacity_price_pos=lock_capacity_price_pos,
        lock_capacity_price_neg=lock_capacity_price_neg,
        precommit_audit_by_ts=audit,
        strategy_permissions=bt._strategy_permissions,
        global_end_utc=plan["target_time_utc"].max(),
        current_soc_mwh=10.0,
    )

    expected_rejected_mw = 0.0 if expected_locked_mw > 0.0 else 2.0
    assert float(result["rejected_mw_total"]) == pytest.approx(expected_rejected_mw)
    target_times = [pd.Timestamp(ts) for ts in plan["target_time_utc"]]
    lockbook = lock_pos if side == "pos" else lock_neg
    lockbook_price = lock_capacity_price_pos if side == "pos" else lock_capacity_price_neg
    other_lockbook = lock_neg if side == "pos" else lock_pos

    for ts in target_times:
        assert float(lockbook[ts]) == pytest.approx(expected_locked_mw)
        assert float(lockbook_price[ts]) == pytest.approx(expected_price)
        assert float(other_lockbook[ts]) == pytest.approx(0.0)
        assert float(audit[f"bcm_precommit_locked_{side}_mw"][ts]) == pytest.approx(expected_locked_mw)
        assert float(audit[f"bcm_precommit_written_{side}_mw"][ts]) == pytest.approx(expected_locked_mw)

    if bid_price > market_price:
        assert expected_locked_mw == 0.0
