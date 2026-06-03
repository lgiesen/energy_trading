from __future__ import annotations

import argparse
from dataclasses import FrozenInstanceError, replace

import pytest
import pandas as pd

from energy_trading.simulation.battery_backtest import BacktestOutputs
from scripts.run_battery_backtest import (
    _build_daily_performance_metrics,
    _build_performance_reconciliation_debug,
    _build_performance_metrics,
    _compute_hourly_throughput_mwh,
    _ensure_hourly_throughput,
    _performance_reconciliation_failure_detail,
    _validate_performance_metrics,
)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        model_key="xgb",
        run_manifest="artifacts/model_runs/latest_xgb.json",
        split="test",
        trading_strategy="multi",
        id_recourse_mode="rule_based",
        da_quantile_role="mid",
        start="2025-05-01T00:00:00Z",
        end="2025-05-01T03:00:00Z",
    )


def _hourly() -> pd.DataFrame:
    ts = pd.date_range("2025-05-01T00:00:00Z", periods=4, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "real_pnl_eur": [57.0, 60.0, 40.0, 50.0],
            "real_submitted_da_buy_mw": [1.0, 0.0, 0.0, 1.0],
            "real_submitted_da_sell_mw": [0.0, 1.0, 1.0, 0.0],
            "real_da_buy_mwh": [1.0, 0.0, 0.0, 1.0],
            "real_da_sell_mwh": [0.0, 1.0, 1.0, 0.0],
            "real_bem_only_submitted_pos_mw": [0.5, 0.0, 0.0, 0.5],
            "real_bem_only_submitted_neg_mw": [0.0, 0.5, 0.5, 0.0],
            "real_bem_only_executed_pos_mwh": [0.25, 0.0, 0.0, 0.25],
            "real_bem_only_executed_neg_mwh": [0.0, 0.25, 0.25, 0.0],
            "real_act_pos_mwh": [0.8, 0.0, 0.0, 0.8],
            "real_act_neg_mwh": [0.0, 0.9, 0.9, 0.0],
            "real_submitted_afrr_pos_mw": [1.0, 1.0, 1.0, 1.0],
            "real_submitted_afrr_neg_mw": [1.0, 1.0, 1.0, 1.0],
            "real_executed_reserve_pos_mw": [1.0, 1.0, 1.0, 1.0],
            "real_executed_reserve_neg_mw": [1.0, 1.0, 1.0, 1.0],
            "real_id_buy_mwh": [0.0, 0.1, 0.1, 0.0],
            "real_id_sell_mwh": [0.2, 0.0, 0.0, 0.2],
            "real_throughput_mwh": [2.0, 2.0, 2.0, 2.0],
            "real_soc_mwh": [10.0, 11.0, 12.0, 11.0],
            "real_revenue_da_eur": [40.0, 50.0, 40.0, 50.0],
            "real_cost_da_eur": [20.0, 20.0, 15.0, 15.0],
            "real_revenue_id_eur": [10.0, 10.0, 10.0, 10.0],
            "real_cost_id_eur": [5.0, 5.0, 5.0, 5.0],
            "real_revenue_capacity_eur": [15.0, 15.0, 15.0, 15.0],
            "real_revenue_activation_eur": [6.0, 6.0, 6.0, 6.0],
            "real_bcm_linked_activation_revenue_eur": [3.0, 3.0, 3.0, 3.0],
            "real_bem_only_activation_revenue_eur": [3.0, 3.0, 3.0, 3.0],
            "real_degradation_cost_eur": [1.0, 1.0, 1.0, 1.0],
            "real_aux_cost_eur": [0.5, 0.5, 0.5, 0.5],
            "real_transaction_cost_eur": [0.25, 0.25, 0.25, 0.25],
            "real_penalty_eur": [0.0, 0.0, 0.0, 0.0],
            "is_fallback_hour": [0.0, 0.0, 0.0, 0.0],
            "optimization_error_code": ["ok", "ok", "ok", "ok"],
        }
    )


def _summary() -> dict[str, object]:
    return {
        "realized_total_pnl_eur": 207.0,
        "predicted_total_pnl_eur": 180.0,
        "p_max_mw": 4.0,
        "capacity_mwh": 20.0,
        "total_da_revenue_eur": 180.0,
        "total_da_cost_eur": 70.0,
        "total_id_revenue_eur": 40.0,
        "total_id_cost_eur": 20.0,
        "total_afrr_capacity_revenue_eur": 60.0,
        "total_afrr_activation_revenue_eur": 24.0,
        "total_bcm_linked_activation_revenue_eur": 12.0,
        "total_bem_only_activation_revenue_eur": 12.0,
        "total_degradation_cost_eur": 4.0,
        "total_auxiliary_cost_eur": 2.0,
        "total_transaction_cost_eur": 1.0,
        "total_offer_cost_eur": 0.0,
        "total_penalty_cost_eur": 0.0,
        "terminal_soc_repair_cost_eur": 0.0,
        "simulation_valid": 1.0,
        "thesis_reportable": 1.0,
        "invalid_reason": "",
        "final_soc_actual_mwh": 10.0,
        "final_soc_target_mwh": 10.0,
    }


def test_hourly_throughput_helper_reproduces_existing_formula_without_existing_column():
    hourly = _hourly().drop(columns=["real_throughput_mwh"])
    throughput = _compute_hourly_throughput_mwh(hourly)
    expected = (
        hourly["real_da_buy_mwh"].abs()
        + hourly["real_da_sell_mwh"].abs()
        + hourly["real_id_buy_mwh"].abs()
        + hourly["real_id_sell_mwh"].abs()
        + hourly["real_act_pos_mwh"].abs()
        + hourly["real_act_neg_mwh"].abs()
    )
    assert throughput.tolist() == expected.tolist()
    assert float(throughput.sum()) == pytest.approx(float(expected.sum()))


def test_frozen_backtest_outputs_hourly_throughput_uses_replace():
    hourly = _hourly().drop(columns=["real_throughput_mwh"])
    outputs = BacktestOutputs(
        hourly=hourly,
        monthly=pd.DataFrame(),
        yearly=pd.DataFrame(),
        plan_history=pd.DataFrame(),
        volatility=pd.DataFrame(),
        summary={},
    )
    with pytest.raises(FrozenInstanceError):
        outputs.hourly = _ensure_hourly_throughput(outputs.hourly)

    replaced = replace(outputs, hourly=_ensure_hourly_throughput(outputs.hourly))
    assert "real_throughput_mwh" in replaced.hourly.columns
    assert "real_throughput_mwh" not in outputs.hourly.columns
    assert float(replaced.hourly["real_throughput_mwh"].sum()) > 0.0


def test_performance_metrics_core_formulas():
    hourly = _hourly()
    summary = _summary()
    perf_df, _ = _build_performance_metrics(
        hourly=hourly,
        summary=summary,
        args=_args(),
        scenario_name="p50_p50",
        scenario_bins=["p50"],
        scenario_start_utc=None,
        scenario_end_utc=None,
    )
    row = perf_df.iloc[0]
    assert row["realized_net_revenue_eur"] == 207.0
    assert row["realized_net_revenue_eur_per_mw"] == 51.75
    assert row["total_costs_eur"] == 7.0
    assert row["throughput_mwh_total"] == 8.0
    assert row["equivalent_full_cycles_total"] == 0.2
    assert row["da_bid_buy_mwh_total"] == 2.0
    assert row["bem_bid_pos_mwh_total"] == 1.0
    assert row["bcm_activation_revenue_eur"] == 12.0
    assert row["bcm_total_revenue_eur"] == 72.0
    assert row["bem_total_revenue_eur"] == 12.0
    assert row["bcm_revenue_before_shared_costs_eur"] == 72.0
    assert row["bem_revenue_before_shared_costs_eur"] == 12.0
    assert row["bcm_bem_activation_split_reconciliation_error_eur"] == 0.0
    assert row["market_revenue_split_method"] == "source_mwh"
    assert abs(float(row["net_revenue_reconciliation_error_eur"])) < 1e-9


def test_daily_metrics_reconcile_to_scenario():
    hourly = _hourly()
    summary = _summary()
    perf_df, _ = _build_performance_metrics(
        hourly=hourly,
        summary=summary,
        args=_args(),
        scenario_name="p50_p50",
        scenario_bins=["p50"],
        scenario_start_utc=None,
        scenario_end_utc=None,
    )
    daily_df = _build_daily_performance_metrics(hourly=hourly, perf_row=perf_df.iloc[0])
    checks = _validate_performance_metrics(perf_row=perf_df.iloc[0], daily_df=daily_df)
    assert checks["net_revenue_reconciliation_ok"] is True
    assert checks["cost_reconciliation_ok"] is True
    assert checks["daily_to_scenario_reconciliation_ok"] is True
    daily_row = daily_df.iloc[0]
    assert daily_row["bcm_activation_revenue_eur"] == 12.0
    assert daily_row["bcm_total_revenue_eur"] == 72.0
    assert daily_row["bem_total_revenue_eur"] == 12.0
    assert daily_row["bcm_bem_activation_split_reconciliation_error_eur"] == 0.0


def test_offer_cost_is_reported_but_not_subtracted_from_settlement_pnl_reconciliation():
    hourly = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-09-04T00:00:00Z", periods=1, freq="h", tz="UTC"),
            "real_pnl_eur": [-153.72621966902076],
            "real_revenue_da_eur": [0.0],
            "real_cost_da_eur": [153.72621966902076],
            "real_revenue_id_eur": [0.0],
            "real_cost_id_eur": [0.0],
            "real_revenue_capacity_eur": [0.0],
            "real_revenue_activation_eur": [0.0],
            "real_degradation_cost_eur": [0.0],
            "real_aux_cost_eur": [0.0],
            "real_transaction_cost_eur": [0.0],
            "real_offer_cost_eur": [17.990632302902526],
            "real_penalty_eur": [0.0],
            "real_throughput_mwh": [0.0],
        }
    )
    perf_df, _ = _build_performance_metrics(
        hourly=hourly,
        summary={
            "realized_total_pnl_eur": 0.0,
            "predicted_total_pnl_eur": 0.0,
            "p_max_mw": 4.0,
            "capacity_mwh": 20.0,
            "terminal_soc_repair_cost_eur": 0.0,
        },
        args=_args(),
        scenario_name="p30_p30",
        scenario_bins=["p30"],
        scenario_start_utc=None,
        scenario_end_utc=None,
    )
    daily_df = _build_daily_performance_metrics(hourly=hourly, perf_row=perf_df.iloc[0])
    checks = _validate_performance_metrics(perf_row=perf_df.iloc[0], daily_df=daily_df)
    row = perf_df.iloc[0]
    assert float(row["offer_cost_eur"]) == pytest.approx(17.990632302902526)
    assert float(row["total_costs_eur"]) == pytest.approx(0.0)
    assert float(row["net_revenue_reconciliation_error_eur"]) == pytest.approx(0.0, abs=1e-9)
    assert checks["net_revenue_reconciliation_ok"] is True
    assert checks["cost_reconciliation_ok"] is True
    assert checks["daily_to_scenario_reconciliation_ok"] is True


def test_terminal_soc_repair_cost_is_reported_but_not_subtracted_from_operational_pnl_reconciliation():
    hourly = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-09-04T23:00:00Z", periods=1, freq="h", tz="UTC"),
            "real_pnl_eur": [-5.612890088941258],
            "real_revenue_da_eur": [0.0],
            "real_cost_da_eur": [0.0],
            "real_revenue_id_eur": [0.0],
            "real_cost_id_eur": [0.0],
            "real_revenue_capacity_eur": [0.0],
            "real_revenue_activation_eur": [10.74385188310444],
            "real_bcm_linked_activation_revenue_eur": [0.551159601603],
            "real_bem_only_activation_revenue_eur": [10.192692281501],
            "real_degradation_cost_eur": [0.150976361768],
            "real_aux_cost_eur": [16.1994],
            "real_transaction_cost_eur": [0.006365610278],
            "real_offer_cost_eur": [0.0],
            "real_penalty_eur": [0.0],
            "real_throughput_mwh": [0.006365610278],
        }
    )
    terminal_repair = 16.548822406254786
    perf_df, _ = _build_performance_metrics(
        hourly=hourly,
        summary={
            "realized_total_pnl_eur": 0.0,
            "predicted_total_pnl_eur": 0.0,
            "p_max_mw": 4.0,
            "capacity_mwh": 20.0,
            "terminal_soc_repair_cost_eur": terminal_repair,
        },
        args=_args(),
        scenario_name="p50_p50",
        scenario_bins=["p50"],
        scenario_start_utc=None,
        scenario_end_utc=None,
    )
    daily_df = _build_daily_performance_metrics(hourly=hourly, perf_row=perf_df.iloc[0])
    checks = _validate_performance_metrics(perf_row=perf_df.iloc[0], daily_df=daily_df)
    row = perf_df.iloc[0]

    assert float(row["terminal_soc_repair_cost_eur"]) == pytest.approx(terminal_repair)
    assert float(row["total_costs_eur"]) == pytest.approx(16.356741971768 + 0.0, abs=1e-9)
    assert float(row["net_revenue_reconciliation_error_eur"]) == pytest.approx(0.0, abs=1e-9)
    assert checks["net_revenue_reconciliation_ok"] is True
    assert checks["cost_reconciliation_ok"] is True


def test_performance_validation_allows_exact_negative_revenue_reconciliation():
    value = -153.72621966902076
    perf_row = pd.Series(
        {
            "realized_net_revenue_eur": value,
            "net_revenue_reconciliation_error_eur": 0.0,
            "total_costs_eur": 0.0,
            "realized_degradation_cost_eur": 0.0,
            "realized_aux_cost_eur": 0.0,
            "transaction_cost_eur": 0.0,
            "offer_cost_eur": 0.0,
            "penalty_cost_eur": 0.0,
            "terminal_soc_repair_cost_eur": 0.0,
        }
    )
    daily_df = pd.DataFrame({"date_utc": ["2025-09-04"], "net_revenue_eur": [value]})
    checks = _validate_performance_metrics(perf_row=perf_row, daily_df=daily_df)
    assert checks["net_revenue_reconciliation_ok"] is True
    assert checks["cost_reconciliation_ok"] is True
    assert checks["daily_to_scenario_reconciliation_ok"] is True
    assert float(checks["daily_to_scenario_error_max_abs"]) == pytest.approx(0.0)


def test_performance_failure_detail_reports_net_decomposition_mismatch():
    perf_row = pd.Series(
        {
            "realized_net_revenue_eur": -153.72621966902076,
            "net_revenue_reconciliation_error_eur": 0.01,
            "total_costs_eur": 0.0,
            "realized_degradation_cost_eur": 0.0,
            "realized_aux_cost_eur": 0.0,
            "transaction_cost_eur": 0.0,
            "offer_cost_eur": 0.0,
            "penalty_cost_eur": 0.0,
            "terminal_soc_repair_cost_eur": 0.0,
        }
    )
    daily_df = pd.DataFrame({"date_utc": ["2025-09-04"], "net_revenue_eur": [-153.72621966902076]})
    checks = _validate_performance_metrics(perf_row=perf_row, daily_df=daily_df)
    debug_df = _build_performance_reconciliation_debug(
        scenario="p30_p30",
        perf_row=perf_row,
        daily_df=daily_df,
        hourly=pd.DataFrame(
            {
                "timestamp_utc": [pd.Timestamp("2025-09-04T00:00:00Z")],
                "real_pnl_eur": [-153.72621966902076],
            }
        ),
    )
    detail = _performance_reconciliation_failure_detail(
        checks=checks,
        recon_debug_df=debug_df,
        perf_row=perf_row,
    )
    assert checks["net_revenue_reconciliation_ok"] is False
    assert checks["daily_to_scenario_reconciliation_ok"] is True
    assert detail["failure_check"] == "net_revenue_reconciliation_ok"
    assert detail["metric"] == "net_revenue_decomposition"
    assert "net_revenue_reconciliation_error_eur" in str(detail["scenario_col"])


def test_scenario_daily_hourly_throughput_reconciliation_without_existing_hourly_column():
    hourly = _hourly().drop(columns=["real_throughput_mwh"])
    perf_df, _ = _build_performance_metrics(
        hourly=hourly,
        summary=_summary(),
        args=_args(),
        scenario_name="p50_p50",
        scenario_bins=["p50"],
        scenario_start_utc=None,
        scenario_end_utc=None,
    )
    hourly_with_throughput = hourly.copy()
    hourly_with_throughput["real_throughput_mwh"] = _compute_hourly_throughput_mwh(hourly)
    daily_df = _build_daily_performance_metrics(hourly=hourly, perf_row=perf_df.iloc[0])
    debug = _build_performance_reconciliation_debug(
        scenario="p50_p50",
        perf_row=perf_df.iloc[0],
        daily_df=daily_df,
        hourly=hourly_with_throughput,
    )
    row = debug.loc[debug["metric"].eq("throughput_mwh_total")].iloc[0]
    assert float(perf_df.iloc[0]["throughput_mwh_total"]) > 0.0
    assert float(row["scenario_value"]) == pytest.approx(float(row["daily_sum_value"]))
    assert float(row["scenario_value"]) == pytest.approx(float(row["hourly_sum_value"]))
    assert float(row["daily_abs_error"]) <= 1e-9
    assert float(row["hourly_abs_error"]) <= 1e-9


def test_regression_throughput_51_not_zero_or_nan():
    target = 51.02155165071409
    hourly = _hourly().drop(columns=["real_throughput_mwh"])
    hourly[["real_da_buy_mwh", "real_da_sell_mwh", "real_id_buy_mwh", "real_id_sell_mwh", "real_act_pos_mwh", "real_act_neg_mwh"]] = 0.0
    hourly.loc[0, "real_da_buy_mwh"] = target
    perf_df, _ = _build_performance_metrics(
        hourly=hourly,
        summary=_summary(),
        args=_args(),
        scenario_name="p50_p50",
        scenario_bins=["p50"],
        scenario_start_utc=None,
        scenario_end_utc=None,
    )
    daily_df = _build_daily_performance_metrics(hourly=hourly, perf_row=perf_df.iloc[0])
    hourly_with_throughput = hourly.copy()
    hourly_with_throughput["real_throughput_mwh"] = _compute_hourly_throughput_mwh(hourly)
    debug = _build_performance_reconciliation_debug(
        scenario="p50_p50",
        perf_row=perf_df.iloc[0],
        daily_df=daily_df,
        hourly=hourly_with_throughput,
    )
    row = debug.loc[debug["metric"].eq("throughput_mwh_total")].iloc[0]
    assert float(perf_df.iloc[0]["throughput_mwh_total"]) == pytest.approx(target)
    assert float(daily_df["throughput_mwh"].sum()) == pytest.approx(target)
    assert float(row["hourly_sum_value"]) == pytest.approx(target)


def test_missing_throughput_sources_do_not_silently_default_to_zero():
    hourly = _hourly().drop(columns=["real_throughput_mwh", "real_act_pos_mwh"])
    with pytest.raises(ValueError, match="Cannot compute real_throughput_mwh"):
        _compute_hourly_throughput_mwh(hourly)
    with pytest.raises(ValueError, match="Cannot compute real_throughput_mwh"):
        _build_daily_performance_metrics(hourly=hourly, perf_row=pd.Series(_summary()))


def test_equivalent_full_cycles_reconcile_from_canonical_throughput():
    hourly = _hourly().drop(columns=["real_throughput_mwh"])
    perf_df, _ = _build_performance_metrics(
        hourly=hourly,
        summary=_summary(),
        args=_args(),
        scenario_name="p50_p50",
        scenario_bins=["p50"],
        scenario_start_utc=None,
        scenario_end_utc=None,
    )
    daily_df = _build_daily_performance_metrics(hourly=hourly, perf_row=perf_df.iloc[0])
    assert float(perf_df.iloc[0]["equivalent_full_cycles_total"]) == pytest.approx(
        float(daily_df["equivalent_full_cycles"].sum())
    )
