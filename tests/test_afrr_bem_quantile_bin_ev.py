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


def _mk_backtester() -> BatteryBacktester:
    MODEL_SPECS["forecast_value_mode"] = "canonical_economic"
    return BatteryBacktester()


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
    # Force asymmetric p_acc fallback: remove negative capacity quantile columns only.
    for q in bt.afrr_quantile_bins:
        df = df.drop(columns=[f"{col.pred_afrr_capacity_price_neg}_{q}"])
    out = bt.optimize_dispatch(df, col, strict_input_validation=False)
    p_pos = float(out.iloc[0]["ev_bem_bin_0_p_exec_pos"])
    p_neg = float(out.iloc[0]["ev_bem_bin_0_p_exec_neg"])
    assert p_pos != p_neg
    assert float(pd.to_numeric(out["ev_pacc_neg_fallback_used"], errors="coerce").fillna(0.0).max()) > 0.0


def test_strict_missing_active_capacity_quantile_fails() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30", "p50", "p70"]
    bt.afrr_quantile_prob = {q: 1.0 - float(q[1:]) / 100.0 for q in bt.afrr_quantile_bins}
    df, col = _tiny_df(hours=2)
    df = df.drop(columns=[f"{col.pred_afrr_capacity_price_pos}_p70"])
    with pytest.raises(ValueError, match="Missing required aFRR capacity quantile-bin inputs in strict mode"):
        bt.optimize_dispatch(df, col, strict_input_validation=True)


def test_strict_nonfinite_active_bin_fails() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30", "p50", "p70"]
    bt.afrr_quantile_prob = {q: 1.0 - float(q[1:]) / 100.0 for q in bt.afrr_quantile_bins}
    df, col = _tiny_df(hours=2)
    df.loc[0, f"{col.pred_afrr_capacity_price_pos}_p70"] = np.nan
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
        trading_strategy="da_only",
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
        trading_strategy="da_only",
        active_bins=["p50"],
        backtester=bt,
        timestamp_col="timestamp_utc",
        status_out=status,
    )
    assert audit.empty
    assert status.get("components_skipped", {}).get("BCM") == "inactive_for_strategy"
    assert status.get("components_skipped", {}).get("BEM") == "inactive_for_strategy"
    stats = _validate_afrr_bin_ev_audit(audit, tol=1e-6, scenario_name="x", trading_strategy="da_only")
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
            trading_strategy="bcm_only",
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
            trading_strategy="bem_only",
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
        trading_strategy="bcm_only",
        active_bins=["p30", "p50", "p70"],
        backtester=bt,
        timestamp_col="timestamp_utc",
    )
    assert set(audit["market_component"].astype(str).unique()) == {"BCM"}
    _validate_afrr_bin_ev_audit(audit, tol=1e-6, scenario_name="x", trading_strategy="bcm_only")


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
        trading_strategy="bem_only",
        active_bins=["p30", "p50", "p70"],
        backtester=bt,
        timestamp_col="timestamp_utc",
    )
    assert set(audit["market_component"].astype(str).unique()) == {"BEM"}
    _validate_afrr_bin_ev_audit(audit, tol=1e-6, scenario_name="x", trading_strategy="bem_only")


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
        trading_strategy="afrr_only",
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
        trading_strategy="afrr_only",
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
        trading_strategy="bem_only",
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
        trading_strategy="bem_only",
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
            trading_strategy="afrr_only",
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
            trading_strategy="bem_only",
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
        trading_strategy="bcm_only",
        active_bins=["p30"],
        backtester=bt,
        timestamp_col="timestamp_utc",
        strict=True,
        status_out=status,
    )
    assert (audit["audit_row_status"] == "benchmark_or_nonaccepted_path_skipped").any()
    assert int(status.get("active_missing_ev_field_count", 0)) == 0
    assert int(status.get("benchmark_or_nonaccepted_path_skipped_count", 0)) > 0
