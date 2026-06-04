from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_trading.simulation.battery_backtest import BacktestColumnMap, BatteryBacktester  # noqa: E402
from energy_trading.config import MODEL_SPECS  # noqa: E402
from scripts.run_battery_backtest import _build_afrr_bin_ev_audit, _validate_afrr_bin_ev_audit  # noqa: E402
from energy_trading.simulation.bid_builder import AFRRCapacityBid  # noqa: E402
from energy_trading.simulation.market_clearing import AFRRCapacityClearingResult  # noqa: E402


def _mk_backtester(forecast_value_mode: str = "canonical_economic") -> BatteryBacktester:
    MODEL_SPECS["forecast_value_mode"] = forecast_value_mode
    return BatteryBacktester()


def _zero_activation_costs(bt: BatteryBacktester) -> None:
    bt.trans_eur_mwh = 0.0
    bt.deg_eur_mwh = 0.0
    bt.aux_afrr_active_mw = 0.0
    bt.aux_standby_mw = 0.0
    bt.afrr_offer_cost_eur_mw_h = 0.0


def _tiny_df(hours: int = 4) -> tuple[pd.DataFrame, BacktestColumnMap]:
    col = BacktestColumnMap()
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=hours, freq="h")
    df = pd.DataFrame(
        {
            col.timestamp: ts,
            col.pred_da_price: np.full(hours, 50.0),
            f"{col.pred_da_price}_p05": np.full(hours, 45.0),
            f"{col.pred_da_price}_p10": np.full(hours, 46.0),
            f"{col.pred_da_price}_p90": np.full(hours, 54.0),
            f"{col.pred_da_price}_p95": np.full(hours, 55.0),
            col.pred_afrr_capacity_price_pos: np.full(hours, 10.0),
            col.pred_afrr_capacity_price_neg: np.full(hours, 10.0),
            col.pred_afrr_activation_price_pos: np.full(hours, 20.0),
            col.pred_afrr_activation_price_neg: np.full(hours, 20.0),
            col.pred_afrr_activation_rate_pos: np.full(hours, 0.5),
            col.pred_afrr_activation_rate_neg: np.full(hours, 0.5),
        }
    )
    bins = ["p01", "p05", "p10", "p30", "p50", "p70", "p90", "p95", "p99"]
    for pref in [
        col.pred_afrr_capacity_price_pos,
        col.pred_afrr_capacity_price_neg,
        col.pred_afrr_activation_price_pos,
        col.pred_afrr_activation_price_neg,
        col.pred_afrr_activation_rate_pos,
        col.pred_afrr_activation_rate_neg,
    ]:
        for q in bins:
            df[f"{pref}_{q}"] = df[pref].to_numpy(dtype=float)
    return df, col


def test_bem_uses_bin_variable_slices() -> None:
    bt = _mk_backtester()
    n = 5
    nb = len(bt.afrr_quantile_bins)
    sl = bt._variable_slices(n=n, n_bins=nb)
    assert "bem_pos_bin" in sl and "bem_neg_bin" in sl
    assert (sl["bem_pos_bin"].stop - sl["bem_pos_bin"].start) == n * nb
    assert (sl["bem_neg_bin"].stop - sl["bem_neg_bin"].start) == n * nb


def test_strict_mode_fails_on_missing_required_bem_quantile_inputs() -> None:
    bt = _mk_backtester()
    df, col = _tiny_df(hours=3)
    df = df.drop(columns=[f"{col.pred_afrr_activation_price_pos}_p30"])
    with pytest.raises(ValueError, match="Missing required aFRR quantile-bin inputs"):
        bt.optimize_dispatch(df, col, strict_input_validation=True)


def test_optimize_dispatch_emits_bem_per_bin_outputs() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30", "p50", "p70"]
    bt.afrr_quantile_prob = {q: 1.0 - float(q[1:]) / 100.0 for q in bt.afrr_quantile_bins}
    df, col = _tiny_df(hours=3)
    out = bt.optimize_dispatch(df, col, strict_input_validation=True)
    for b in range(len(bt.afrr_quantile_bins)):
        assert f"bem_pos_bin_{b}_mw" in out.columns
        assert f"bem_neg_bin_{b}_mw" in out.columns
        assert f"ev_bem_pos_coef_bin_{b}_eur_per_mw" in out.columns
        assert f"ev_bem_neg_coef_bin_{b}_eur_per_mw" in out.columns
        assert f"afrr_bin_{b}_quantile" in out.columns
        assert f"afrr_bin_{b}_cap_price_pos" in out.columns
        assert f"afrr_bin_{b}_cap_price_neg" in out.columns


def test_optimize_dispatch_emits_semantic_quantile_bin_aliases() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30", "p50", "p70"]
    bt.afrr_quantile_prob = {q: 1.0 - float(q[1:]) / 100.0 for q in bt.afrr_quantile_bins}
    df, col = _tiny_df(hours=2)
    out = bt.optimize_dispatch(df, col, strict_input_validation=True)

    expected_aliases = [
        ("reserve_pos_bin_0_mw", "bcm_p30_reserve_pos_mw"),
        ("reserve_neg_bin_0_mw", "bcm_p30_reserve_neg_mw"),
        ("bem_pos_bin_0_mw", "bem_p30_pos_mw"),
        ("bem_neg_bin_0_mw", "bem_p30_neg_mw"),
        ("afrr_bin_0_cap_price_pos", "afrr_p30_cap_price_pos"),
        ("ev_rpos_coef_bin_0_eur_per_mw", "bcm_p30_reserve_pos_coef_eur_per_mw"),
        ("ev_bem_pos_coef_bin_0_eur_per_mw", "bem_p30_pos_coef_eur_per_mw"),
        ("ev_bcm_expected_capacity_revenue_pos_bin_0", "bcm_p30_expected_capacity_revenue_pos"),
        ("ev_bem_expected_activation_revenue_pos_bin_0", "bem_p30_expected_activation_revenue_pos"),
        ("ev_bem_bin_0_p_exec_pos", "bem_p30_p_exec_pos"),
    ]
    for old_col, alias_col in expected_aliases:
        assert old_col in out.columns
        assert alias_col in out.columns
        assert np.allclose(
            pd.to_numeric(out[old_col], errors="coerce").fillna(0.0),
            pd.to_numeric(out[alias_col], errors="coerce").fillna(0.0),
        )


def test_quantile_bin_aliases_support_namespaced_settlement_columns() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30"]
    frame = pd.DataFrame(
        {
            "real_submitted_afrr_pos_bin_0_mw": [1.25],
            "real_executed_afrr_act_neg_bin_0_mw": [0.5],
            "pred_submitted_afrr_neg_bin_0_price_eur_mw": [42.0],
        }
    )
    out = bt._add_quantile_bin_alias_columns(frame)
    assert float(out["real_bcm_p30_submitted_pos_mw"].iloc[0]) == pytest.approx(1.25)
    assert float(out["real_bcm_p30_executed_act_neg_mw"].iloc[0]) == pytest.approx(0.5)
    assert float(out["pred_bcm_p30_submitted_neg_price_eur_mw"].iloc[0]) == pytest.approx(42.0)


def test_canonical_neg_activation_value_is_not_subtracted_in_optimizer_ev() -> None:
    bt = _mk_backtester("canonical_economic")
    _zero_activation_costs(bt)
    bt.afrr_quantile_bins = ["p50"]
    bt.afrr_quantile_prob = {"p50": 0.5}
    df, col = _tiny_df(hours=2)
    df[col.pred_afrr_capacity_price_neg] = 0.0
    df[f"{col.pred_afrr_capacity_price_neg}_p50"] = 0.0
    df[col.pred_afrr_activation_price_neg] = 1000.0
    df[f"{col.pred_afrr_activation_price_neg}_p50"] = 1000.0
    df[col.pred_afrr_activation_rate_neg] = 1.0
    df[f"{col.pred_afrr_activation_rate_neg}_p50"] = 1.0

    out = bt.optimize_dispatch(df, col, strict_input_validation=True)
    row = out.iloc[0]

    assert float(row["ev_bcm_activation_margin_neg_bin_0"]) == pytest.approx(1000.0)
    assert float(row["ev_bem_activation_margin_neg_bin_0"]) == pytest.approx(1000.0)
    assert float(row["ev_bcm_expected_activation_revenue_neg_bin_0"]) == pytest.approx(500.0)
    assert float(row["ev_bem_neg_coef_bin_0_eur_per_mw"]) == pytest.approx(500.0)


def test_raw_signed_neg_activation_price_is_negated_once_in_optimizer_ev() -> None:
    bt = _mk_backtester("raw_signed")
    _zero_activation_costs(bt)
    bt.afrr_quantile_bins = ["p50"]
    bt.afrr_quantile_prob = {"p50": 0.5}
    df, col = _tiny_df(hours=2)
    df[col.pred_afrr_capacity_price_neg] = 0.0
    df[f"{col.pred_afrr_capacity_price_neg}_p50"] = 0.0
    df[col.pred_afrr_activation_price_neg] = -1000.0
    df[f"{col.pred_afrr_activation_price_neg}_p50"] = -1000.0
    df[col.pred_afrr_activation_rate_neg] = 1.0
    df[f"{col.pred_afrr_activation_rate_neg}_p50"] = 1.0

    out = bt.optimize_dispatch(df, col, strict_input_validation=True)
    row = out.iloc[0]

    assert float(row["ev_bcm_activation_margin_neg_bin_0"]) == pytest.approx(1000.0)
    assert float(row["ev_bem_activation_margin_neg_bin_0"]) == pytest.approx(1000.0)
    assert float(row["ev_bcm_expected_activation_revenue_neg_bin_0"]) == pytest.approx(500.0)
    assert float(row["ev_bem_neg_coef_bin_0_eur_per_mw"]) == pytest.approx(500.0)


def test_optimizer_and_settlement_agree_on_canonical_neg_activation_value() -> None:
    bt = _mk_backtester("canonical_economic")
    _zero_activation_costs(bt)
    bt.afrr_quantile_bins = ["p50"]
    bt.afrr_quantile_prob = {"p50": 0.5}
    df, col = _tiny_df(hours=2)
    df[col.pred_afrr_capacity_price_neg] = 0.0
    df[f"{col.pred_afrr_capacity_price_neg}_p50"] = 0.0
    df[col.pred_afrr_activation_price_neg] = 100.0
    df[f"{col.pred_afrr_activation_price_neg}_p50"] = 100.0
    df[col.pred_afrr_activation_rate_neg] = 1.0
    df[f"{col.pred_afrr_activation_rate_neg}_p50"] = 1.0

    out = bt.optimize_dispatch(df, col, strict_input_validation=True)
    row = out.iloc[0]
    optimizer_expected_revenue_per_mw = float(row["ev_bcm_expected_activation_revenue_neg_bin_0"])

    _, settlement = bt._settle_one_hour(
        soc=bt.soc_min,
        charge=0.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=1.0,
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=100.0,
        act_pos_rate=0.0,
        act_neg_rate=0.5,
        cap_bid_pos=0.0,
        cap_bid_neg=0.0,
    )

    assert optimizer_expected_revenue_per_mw == pytest.approx(50.0)
    delivered_neg_mwh = float(settlement["delivered_activation_neg_mwh"])
    assert delivered_neg_mwh > 0.0
    assert float(settlement["revenue_activation_eur"]) == pytest.approx(delivered_neg_mwh * 100.0)
    assert float(settlement["revenue_activation_eur"]) > 0.0


def test_ev_audit_reconciles_canonical_negative_activation_terms() -> None:
    bt = _mk_backtester("canonical_economic")
    _zero_activation_costs(bt)
    bt.afrr_quantile_bins = ["p50"]
    bt.afrr_quantile_prob = {"p50": 0.5}
    df, col = _tiny_df(hours=2)
    df[col.pred_afrr_capacity_price_neg] = 0.0
    df[f"{col.pred_afrr_capacity_price_neg}_p50"] = 0.0
    df[col.pred_afrr_activation_price_neg] = 100.0
    df[f"{col.pred_afrr_activation_price_neg}_p50"] = 100.0
    df[col.pred_afrr_activation_rate_neg] = 1.0
    df[f"{col.pred_afrr_activation_rate_neg}_p50"] = 1.0

    out = bt.optimize_dispatch(df, col, strict_input_validation=True)
    out = out.rename(columns={col.timestamp: "timestamp_utc"})
    audit = _build_afrr_bin_ev_audit(
        hourly=out,
        scenario_name="x",
        trading_strategy="multi",
        active_bins=["p50"],
        backtester=bt,
        timestamp_col="timestamp_utc",
    )
    neg = audit.loc[audit["direction"].astype(str).eq("neg")].copy()
    assert not neg.empty
    assert float(neg["activation_margin"].min()) == pytest.approx(100.0)
    assert float(neg["expected_activation_revenue"].min()) == pytest.approx(50.0)

    stats = _validate_afrr_bin_ev_audit(audit, tol=1e-6)
    assert float(stats["ev_audit_max_bcm_formula_error"]) <= 1e-6
    assert float(stats["ev_audit_max_bem_formula_error"]) <= 1e-6


def test_canonical_neg_activation_bid_price_and_clearing_use_positive_provider_value() -> None:
    bt = _mk_backtester("canonical_economic")
    bt.bid_builder.afrr_energy_bid_strategy = "forecast"
    bid_price = bt.bid_builder.dynamic_afrr_energy_price(
        side="neg",
        pred_act_price=1000.0,
        soc_now_mwh=(bt.soc_min + bt.soc_max) / 2.0,
        soc_min_mwh=bt.soc_min,
        soc_max_mwh=bt.soc_max,
        obligation_mw=0.0,
    )
    assert bid_price == pytest.approx(1000.0)

    clearing = bt.market_clearing_engine.clear_afrr_activation(
        [
            AFRRCapacityBid(
                ts=pd.Timestamp("2026-01-01T00:00:00Z"),
                side="neg",
                quantity_mw=1.0,
                capacity_price_eur_mw=0.0,
                energy_price_eur_mwh=bid_price,
            )
        ],
        AFRRCapacityClearingResult(
            submitted_pos_mw=0.0,
            submitted_neg_mw=1.0,
            awarded_pos_mw=0.0,
            awarded_neg_mw=1.0,
            pos_awarded=False,
            neg_awarded=True,
        ),
        true_act_pos=0.0,
        true_act_neg=1000.0,
        true_rate_pos=0.0,
        true_rate_neg=1.0,
    )
    assert clearing.neg_accepted is True
    assert clearing.executed_rate_neg == pytest.approx(1.0)


def test_bcm_preselection_clears_on_activation_price_not_capacity_price() -> None:
    bt = _mk_backtester("canonical_economic")
    bid = AFRRCapacityBid(
        ts=pd.Timestamp("2026-01-01T00:00:00Z"),
        side="pos",
        quantity_mw=1.0,
        capacity_price_eur_mw=9999.0,
        energy_price_eur_mwh=50.0,
    )

    res = bt.market_clearing_engine.clear_afrr_capacity(
        [bid],
        true_cap_pos=0.0,
        true_cap_neg=0.0,
        true_act_pos=60.0,
        true_act_neg=0.0,
        clearing_price_basis="activation_price",
    )

    assert res.pos_awarded is True
    assert res.awarded_pos_mw == pytest.approx(1.0)


def test_bcm_preselection_rejects_expensive_activation_bid_even_if_capacity_price_low() -> None:
    bt = _mk_backtester("canonical_economic")
    bid = AFRRCapacityBid(
        ts=pd.Timestamp("2026-01-01T00:00:00Z"),
        side="pos",
        quantity_mw=1.0,
        capacity_price_eur_mw=0.0,
        energy_price_eur_mwh=70.0,
    )

    res = bt.market_clearing_engine.clear_afrr_capacity(
        [bid],
        true_cap_pos=9999.0,
        true_cap_neg=0.0,
        true_act_pos=60.0,
        true_act_neg=0.0,
        clearing_price_basis="activation_price",
    )

    assert res.pos_awarded is False
    assert res.awarded_pos_mw == pytest.approx(0.0)


def test_raw_signed_neg_activation_bid_price_and_clearing_keep_legacy_sign() -> None:
    bt = _mk_backtester("raw_signed")
    bt.bid_builder.afrr_energy_bid_strategy = "forecast"
    bid_price = bt.bid_builder.dynamic_afrr_energy_price(
        side="neg",
        pred_act_price=-1000.0,
        soc_now_mwh=(bt.soc_min + bt.soc_max) / 2.0,
        soc_min_mwh=bt.soc_min,
        soc_max_mwh=bt.soc_max,
        obligation_mw=0.0,
    )
    assert bid_price == pytest.approx(-1000.0)

    clearing = bt.market_clearing_engine.clear_afrr_activation(
        [
            AFRRCapacityBid(
                ts=pd.Timestamp("2026-01-01T00:00:00Z"),
                side="neg",
                quantity_mw=1.0,
                capacity_price_eur_mw=0.0,
                energy_price_eur_mwh=bid_price,
            )
        ],
        AFRRCapacityClearingResult(
            submitted_pos_mw=0.0,
            submitted_neg_mw=1.0,
            awarded_pos_mw=0.0,
            awarded_neg_mw=1.0,
            pos_awarded=False,
            neg_awarded=True,
        ),
        true_act_pos=0.0,
        true_act_neg=-1000.0,
        true_rate_pos=0.0,
        true_rate_neg=1.0,
    )
    assert clearing.neg_accepted is True
    assert clearing.executed_rate_neg == pytest.approx(1.0)


def test_naive_24h_replaces_quantile_columns_not_only_base_predictions() -> None:
    col = BacktestColumnMap()
    n = 26
    df = pd.DataFrame(
        {
            col.timestamp: pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="h"),
            col.true_afrr_activation_price_neg: np.arange(n, dtype=float) + 100.0,
            col.pred_afrr_activation_price_neg: np.full(n, 1000.0),
            f"{col.pred_afrr_activation_price_neg}_p50": np.full(n, 2000.0),
            f"{col.pred_afrr_activation_price_neg}_p90": np.full(n, 3000.0),
            col.true_afrr_activation_rate_neg: np.linspace(0.0, 1.0, n),
            col.pred_afrr_activation_rate_neg: np.full(n, 0.9),
            f"{col.pred_afrr_activation_rate_neg}_p50": np.full(n, 0.8),
        }
    )

    out = BatteryBacktester._apply_naive_24h_predictions(df, col)

    assert float(out.loc[24, col.pred_afrr_activation_price_neg]) == pytest.approx(100.0)
    assert float(out.loc[24, f"{col.pred_afrr_activation_price_neg}_p50"]) == pytest.approx(100.0)
    assert float(out.loc[24, f"{col.pred_afrr_activation_price_neg}_p90"]) == pytest.approx(100.0)
    assert float(out.loc[25, f"{col.pred_afrr_activation_price_neg}_p50"]) == pytest.approx(101.0)
    assert float(out.loc[24, col.pred_afrr_activation_rate_neg]) == pytest.approx(0.0)
    assert float(out.loc[24, f"{col.pred_afrr_activation_rate_neg}_p50"]) == pytest.approx(0.0)
    # No 24h history is available for the first rows, so the existing forecast fallback remains.
    assert float(out.loc[0, f"{col.pred_afrr_activation_price_neg}_p50"]) == pytest.approx(2000.0)


def test_bcm_same_q_uses_bin_specific_activation_inputs() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30", "p50", "p70"]
    bt.afrr_quantile_prob = {q: 1.0 - float(q[1:]) / 100.0 for q in bt.afrr_quantile_bins}
    df, col = _tiny_df(hours=2)
    for q, price, rate in [("p30", 10.0, 0.1), ("p50", 20.0, 0.2), ("p70", 30.0, 0.3)]:
        df[f"{col.pred_afrr_activation_price_pos}_{q}"] = price
        df[f"{col.pred_afrr_activation_rate_pos}_{q}"] = rate
    out_a = bt.optimize_dispatch(df, col, strict_input_validation=True)
    coef30_a = float(out_a.iloc[0]["ev_rpos_coef_bin_0_eur_per_mw"])
    coef50_a = float(out_a.iloc[0]["ev_rpos_coef_bin_1_eur_per_mw"])
    coef70_a = float(out_a.iloc[0]["ev_rpos_coef_bin_2_eur_per_mw"])
    assert coef30_a != coef50_a != coef70_a

    # Change only p70 inputs; only p70 coefficient should move materially.
    df2 = df.copy()
    df2[f"{col.pred_afrr_activation_price_pos}_p70"] = 60.0
    df2[f"{col.pred_afrr_activation_rate_pos}_p70"] = 0.6
    out_b = bt.optimize_dispatch(df2, col, strict_input_validation=True)
    coef30_b = float(out_b.iloc[0]["ev_rpos_coef_bin_0_eur_per_mw"])
    coef50_b = float(out_b.iloc[0]["ev_rpos_coef_bin_1_eur_per_mw"])
    coef70_b = float(out_b.iloc[0]["ev_rpos_coef_bin_2_eur_per_mw"])
    assert np.isclose(coef30_a, coef30_b)
    assert np.isclose(coef50_a, coef50_b)
    assert not np.isclose(coef70_a, coef70_b)


def test_bem_neg_uses_neg_side_execution_probability() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30", "p50", "p70"]
    bt.afrr_quantile_prob = {q: 1.0 - float(q[1:]) / 100.0 for q in bt.afrr_quantile_bins}
    df, col = _tiny_df(hours=2)
    # Force asymmetric p_acc fallback: remove negative activation-price quantile columns only.
    for q in bt.afrr_quantile_bins:
        df = df.drop(columns=[f"{col.pred_afrr_activation_price_neg}_{q}"])
    out = bt.optimize_dispatch(df, col, strict_input_validation=False)
    p_pos = float(out.iloc[0]["ev_bem_bin_0_p_exec_pos"])
    p_neg = float(out.iloc[0]["ev_bem_bin_0_p_exec_neg"])
    assert p_pos != p_neg
    assert float(pd.to_numeric(out["ev_pacc_neg_fallback_used"], errors="coerce").fillna(0.0).max()) > 0.0


def test_strict_missing_active_activation_price_quantile_fails() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30", "p50", "p70"]
    bt.afrr_quantile_prob = {q: 1.0 - float(q[1:]) / 100.0 for q in bt.afrr_quantile_bins}
    df, col = _tiny_df(hours=2)
    df = df.drop(columns=[f"{col.pred_afrr_activation_price_pos}_p70"])
    with pytest.raises(ValueError, match="Missing required aFRR activation-price/rate quantile-bin inputs"):
        bt.optimize_dispatch(df, col, strict_input_validation=True)


def test_strict_nonfinite_active_bin_fails() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30", "p50", "p70"]
    bt.afrr_quantile_prob = {q: 1.0 - float(q[1:]) / 100.0 for q in bt.afrr_quantile_bins}
    df, col = _tiny_df(hours=2)
    df.loc[0, f"{col.pred_afrr_activation_price_pos}_p70"] = np.nan
    with pytest.raises(ValueError):
        bt.optimize_dispatch(df, col, strict_input_validation=True)


def test_ev_coefficient_matches_component_formula() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30", "p50", "p70"]
    bt.afrr_quantile_prob = {q: 1.0 - float(q[1:]) / 100.0 for q in bt.afrr_quantile_bins}
    df, col = _tiny_df(hours=2)
    out = bt.optimize_dispatch(df, col, strict_input_validation=True)
    row = out.iloc[0]
    b = 1  # p50 bin
    assert float(row[f"ev_bcm_expected_capacity_revenue_pos_bin_{b}"]) == pytest.approx(0.0)
    bcm_ev = (
        float(row[f"ev_bcm_expected_capacity_revenue_pos_bin_{b}"])
        + float(row[f"ev_bcm_expected_activation_revenue_pos_bin_{b}"])
        - float(row[f"ev_bcm_expected_aux_cost_pos_bin_{b}"])
        - float(row[f"ev_bcm_offer_cost_bin_{b}"])
    )
    bem_ev = (
        float(row[f"ev_bem_expected_activation_revenue_pos_bin_{b}"])
        - float(row[f"ev_bem_expected_aux_cost_pos_bin_{b}"])
    )
    assert np.isclose(bcm_ev, float(row[f"ev_rpos_coef_bin_{b}_eur_per_mw"]), atol=1e-9)
    assert np.isclose(bem_ev, float(row[f"ev_bem_pos_coef_bin_{b}_eur_per_mw"]), atol=1e-9)


def test_ev_audit_builder_no_quantile_col_uses_active_bins() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p50"]
    bt.afrr_quantile_prob = {"p50": 0.5}
    h = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="h"),
            "reserve_pos_bin_0_mw": [1.0, 0.0],
            "reserve_neg_bin_0_mw": [0.0, 1.0],
            "bem_pos_bin_0_mw": [0.0, 0.0],
            "bem_neg_bin_0_mw": [0.0, 0.0],
            "ev_rpos_coef_bin_0_eur_per_mw": [1.0, 1.0],
            "ev_rneg_coef_bin_0_eur_per_mw": [1.0, 1.0],
            "ev_bem_pos_coef_bin_0_eur_per_mw": [0.0, 0.0],
            "ev_bem_neg_coef_bin_0_eur_per_mw": [0.0, 0.0],
        }
    )
    audit = _build_afrr_bin_ev_audit(
        hourly=h,
        scenario_name="x",
        trading_strategy="da",
        active_bins=["p50"],
        backtester=bt,
        timestamp_col="timestamp_utc",
    )
    assert audit.empty


def test_ev_audit_builder_labels_p30_p50_p70_and_four_components() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30", "p50", "p70"]
    bt.afrr_quantile_prob = {q: 1.0 - float(q[1:]) / 100.0 for q in bt.afrr_quantile_bins}
    df, col = _tiny_df(hours=2)
    out = bt.optimize_dispatch(df, col, strict_input_validation=True)
    out = out.rename(columns={col.timestamp: "timestamp_utc"})
    audit = _build_afrr_bin_ev_audit(
        hourly=out,
        scenario_name="x",
        trading_strategy="multi",
        active_bins=["p30", "p50", "p70"],
        backtester=bt,
        timestamp_col="timestamp_utc",
    )
    assert set(audit["quantile_bin"].astype(str).unique()) == {"p30", "p50", "p70"}
    assert set(audit["market_component"].astype(str).unique()) == {"BCM", "BEM"}
    assert set(audit["direction"].astype(str).unique()) == {"pos", "neg"}
    assert len(audit) == 2 * 3 * 4
    stats = _validate_afrr_bin_ev_audit(audit, tol=1e-6)
    assert float(stats["ev_audit_max_bcm_formula_error"]) <= 1e-6
    assert float(stats["ev_audit_max_bem_formula_error"]) <= 1e-6


def test_ev_audit_da_only_skips_afrr_components() -> None:
    bt = _mk_backtester()
    h = pd.DataFrame({"timestamp_utc": pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="h")})
    status: dict[str, object] = {}
    audit = _build_afrr_bin_ev_audit(
        hourly=h,
        scenario_name="x",
        trading_strategy="da",
        active_bins=["p50"],
        backtester=bt,
        timestamp_col="timestamp_utc",
        status_out=status,
    )
    assert audit.empty
    assert status.get("components_skipped", {}).get("BCM") == "inactive_for_strategy"
    assert status.get("components_skipped", {}).get("BEM") == "inactive_for_strategy"
    stats = _validate_afrr_bin_ev_audit(audit, tol=1e-6, scenario_name="x", trading_strategy="da")
    assert float(stats["ev_audit_row_count"]) == 0.0


def test_ev_audit_bcm_only_requires_bcm_columns() -> None:
    bt = _mk_backtester()
    h = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-01-01T00:00:00Z", periods=1, freq="h"),
            "reserve_pos_bin_0_mw": [1.0],
            "reserve_neg_bin_0_mw": [1.0],
        }
    )
    with pytest.raises(ValueError, match="missing/nonfinite required EV fields"):
        _build_afrr_bin_ev_audit(
            hourly=h,
            scenario_name="x",
            trading_strategy="bcm",
            active_bins=["p50"],
            backtester=bt,
            timestamp_col="timestamp_utc",
            strict=True,
        )


def test_ev_audit_bem_only_requires_bem_columns() -> None:
    bt = _mk_backtester()
    h = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-01-01T00:00:00Z", periods=1, freq="h"),
            "bem_pos_bin_0_mw": [1.0],
            "bem_neg_bin_0_mw": [1.0],
        }
    )
    with pytest.raises(ValueError, match="missing/nonfinite required EV fields"):
        _build_afrr_bin_ev_audit(
            hourly=h,
            scenario_name="x",
            trading_strategy="bem",
            active_bins=["p50"],
            backtester=bt,
            timestamp_col="timestamp_utc",
            strict=True,
        )


def test_ev_audit_bcm_only_emits_only_bcm() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30", "p50", "p70"]
    bt.afrr_quantile_prob = {q: 1.0 - float(q[1:]) / 100.0 for q in bt.afrr_quantile_bins}
    df, col = _tiny_df(hours=2)
    out = bt.optimize_dispatch(df, col, strict_input_validation=True)
    out = out.rename(columns={col.timestamp: "timestamp_utc"})
    audit = _build_afrr_bin_ev_audit(
        hourly=out,
        scenario_name="x",
        trading_strategy="bcm",
        active_bins=["p30", "p50", "p70"],
        backtester=bt,
        timestamp_col="timestamp_utc",
    )
    assert set(audit["market_component"].astype(str).unique()) == {"BCM"}
    _validate_afrr_bin_ev_audit(audit, tol=1e-6, scenario_name="x", trading_strategy="bcm")


def test_ev_audit_bem_only_emits_only_bem() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30", "p50", "p70"]
    bt.afrr_quantile_prob = {q: 1.0 - float(q[1:]) / 100.0 for q in bt.afrr_quantile_bins}
    df, col = _tiny_df(hours=2)
    out = bt.optimize_dispatch(df, col, strict_input_validation=True)
    out = out.rename(columns={col.timestamp: "timestamp_utc"})
    audit = _build_afrr_bin_ev_audit(
        hourly=out,
        scenario_name="x",
        trading_strategy="bem",
        active_bins=["p30", "p50", "p70"],
        backtester=bt,
        timestamp_col="timestamp_utc",
    )
    assert set(audit["market_component"].astype(str).unique()) == {"BEM"}
    _validate_afrr_bin_ev_audit(audit, tol=1e-6, scenario_name="x", trading_strategy="bem")


def test_ev_audit_validator_message_identifies_bad_component() -> None:
    audit = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "market_component": ["BCM"],
            "direction": ["pos"],
            "quantile_bin": ["p50"],
            "decision_variable_name": ["reserve_pos_bin_0_mw"],
            "expected_capacity_revenue": [1.0],
            "expected_activation_revenue": [2.0],
            "expected_aux_cost": [np.nan],
            "offer_cost": [0.1],
            "ev_coefficient": [2.9],
        }
    )
    with pytest.raises(ValueError, match="bad_columns"):
        _validate_afrr_bin_ev_audit(
            audit,
            tol=1e-6,
            scenario_name="x",
            trading_strategy="multi",
            audit_path="mem.csv",
        )


def test_afrr_only_zero_decision_missing_fields_are_skipped() -> None:
    bt = _mk_backtester()
    h = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2025-05-02T21:00:00Z")],
            "optimization_error_code": ["hard_final_soc_infeasible"],
            "reserve_pos_bin_0_mw": [0.0],
            "reserve_neg_bin_0_mw": [0.0],
            "bem_pos_bin_0_mw": [0.0],
            "bem_neg_bin_0_mw": [0.0],
            "ev_rpos_coef_bin_0_eur_per_mw": [0.0],
            "ev_rneg_coef_bin_0_eur_per_mw": [0.0],
            "ev_bem_pos_coef_bin_0_eur_per_mw": [0.0],
            "ev_bem_neg_coef_bin_0_eur_per_mw": [0.0],
        }
    )
    status: dict[str, object] = {}
    audit = _build_afrr_bin_ev_audit(
        hourly=h,
        scenario_name="p30_p70",
        trading_strategy="afrr",
        active_bins=["p30"],
        backtester=bt,
        timestamp_col="timestamp_utc",
        strict=True,
        status_out=status,
    )
    assert not audit.empty
    bcm = audit.loc[audit["market_component"] == "BCM"].copy()
    assert (bcm["audit_row_status"] == "benchmark_or_nonaccepted_path_skipped").all()
    stats = _validate_afrr_bin_ev_audit(
        audit,
        tol=1e-6,
        scenario_name="p30_p70",
        trading_strategy="afrr",
        audit_path="mem.csv",
    )
    assert float(stats["ev_audit_max_bcm_formula_error"]) == 0.0


def test_bem_only_nan_selected_mw_is_treated_as_inactive_skipped() -> None:
    bt = _mk_backtester()
    h = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2025-05-02T21:00:00Z")],
            "optimization_error_code": ["hard_final_soc_infeasible"],
            "bem_pos_bin_0_mw": [np.nan],
            "bem_neg_bin_0_mw": [np.nan],
            "ev_bem_pos_coef_bin_0_eur_per_mw": [np.nan],
            "ev_bem_neg_coef_bin_0_eur_per_mw": [np.nan],
        }
    )
    status: dict[str, object] = {}
    audit = _build_afrr_bin_ev_audit(
        hourly=h,
        scenario_name="p50_p50",
        trading_strategy="bem",
        active_bins=["p50"],
        backtester=bt,
        timestamp_col="timestamp_utc",
        strict=True,
        status_out=status,
    )
    bem = audit.loc[audit["market_component"] == "BEM"].copy()
    assert not bem.empty
    assert (bem["audit_row_status"] == "benchmark_or_nonaccepted_path_skipped").all()
    assert int(status.get("active_missing_ev_field_count", 0)) == 0
    _validate_afrr_bin_ev_audit(
        audit,
        tol=1e-6,
        scenario_name="p50_p50",
        trading_strategy="bem",
        audit_path="mem.csv",
    )


def test_afrr_only_active_selected_missing_fields_fail() -> None:
    bt = _mk_backtester()
    h = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2025-05-02T21:00:00Z")],
            "optimization_error_code": ["ok"],
            "reserve_pos_bin_0_mw": [1.0],
            "reserve_neg_bin_0_mw": [0.0],
            "bem_pos_bin_0_mw": [0.0],
            "bem_neg_bin_0_mw": [0.0],
            "ev_rpos_coef_bin_0_eur_per_mw": [1.0],
            "ev_rneg_coef_bin_0_eur_per_mw": [0.0],
            "ev_bem_pos_coef_bin_0_eur_per_mw": [0.0],
            "ev_bem_neg_coef_bin_0_eur_per_mw": [0.0],
        }
    )
    with pytest.raises(ValueError, match="missing/nonfinite required EV fields"):
        _build_afrr_bin_ev_audit(
            hourly=h,
            scenario_name="p30_p70",
            trading_strategy="afrr",
            active_bins=["p30"],
            backtester=bt,
            timestamp_col="timestamp_utc",
            strict=True,
        )


def test_ev_audit_selected_row_with_nonfinite_required_fields_fails_strict() -> None:
    bt = _mk_backtester()
    h = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2025-05-02T21:00:00Z")],
            "optimization_error_code": ["ok"],
            "bem_pos_bin_0_mw": [1.0],
            "bem_neg_bin_0_mw": [0.0],
            "ev_bem_expected_activation_revenue_pos_bin_0": [np.nan],
            "ev_bem_expected_aux_cost_pos_bin_0": [0.1],
            "ev_bem_pos_coef_bin_0_eur_per_mw": [0.2],
            "ev_bem_expected_activation_revenue_neg_bin_0": [0.0],
            "ev_bem_expected_aux_cost_neg_bin_0": [0.0],
            "ev_bem_neg_coef_bin_0_eur_per_mw": [0.0],
        }
    )
    with pytest.raises(ValueError, match="missing/nonfinite required EV fields"):
        _build_afrr_bin_ev_audit(
            hourly=h,
            scenario_name="p50_p50",
            trading_strategy="bem",
            active_bins=["p50"],
            backtester=bt,
            timestamp_col="timestamp_utc",
            strict=True,
        )


def test_ev_audit_nonaccepted_row_with_nonfinite_required_fields_is_skipped() -> None:
    bt = _mk_backtester()
    h = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2025-05-02T21:00:00Z")],
            "optimization_error_code": ["hard_final_soc_infeasible"],
            "reserve_pos_bin_0_mw": [1.0],
            "reserve_neg_bin_0_mw": [0.0],
            "ev_bcm_expected_capacity_revenue_pos_bin_0": [np.nan],
            "ev_bcm_expected_activation_revenue_pos_bin_0": [0.2],
            "ev_bcm_expected_aux_cost_pos_bin_0": [0.1],
            "ev_bcm_offer_cost_bin_0": [0.0],
            "ev_rpos_coef_bin_0_eur_per_mw": [0.1],
            "ev_bcm_expected_capacity_revenue_neg_bin_0": [0.0],
            "ev_bcm_expected_activation_revenue_neg_bin_0": [0.0],
            "ev_bcm_expected_aux_cost_neg_bin_0": [0.0],
            "ev_rneg_coef_bin_0_eur_per_mw": [0.0],
        }
    )
    status: dict[str, object] = {}
    audit = _build_afrr_bin_ev_audit(
        hourly=h,
        scenario_name="p30_p30",
        trading_strategy="bcm",
        active_bins=["p30"],
        backtester=bt,
        timestamp_col="timestamp_utc",
        strict=True,
        status_out=status,
    )
    assert (audit["audit_row_status"] == "benchmark_or_nonaccepted_path_skipped").any()
    assert int(status.get("active_missing_ev_field_count", 0)) == 0
    assert int(status.get("benchmark_or_nonaccepted_path_skipped_count", 0)) > 0
