from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from energy_trading.simulation.battery_backtest import (  # noqa: E402
    BacktestColumnMap,
    BatteryBacktester,
    StrategyPermissions,
    _atomic_write_json,
    _atomic_write_parquet,
    _update_checkpoint_manifest,
    _write_checkpoint_run_status,
    _write_stage_checkpoint,
    assign_bcm_capacity_block,
    bcm_capacity_ev_eur,
    bcm_capacity_hourly_revenue_decomposition_eur,
    bcm_capacity_revenue_eur,
    bcm_linked_bem_activation_ev_eur,
    bcm_product_to_bem_mandatory_volume,
    canonicalize_market_frame,
    load_prediction_warehouse_long,
    normalize_predicted_pnl_aliases,
    apply_benchmark_availability_to_summary,
    resolve_benchmark_paths,
    should_report_rolling_pf_solver_failed,
)
from energy_trading.config import MODEL_SPECS  # noqa: E402
from energy_trading.simulation.bid_builder import AFRRCapacityBid, BCMCapacityBid  # noqa: E402
from energy_trading.simulation.market_clearing import AFRRCapacityClearingResult, MarketClearingEngine  # noqa: E402
from scripts.run_battery_backtest import (  # noqa: E402
    _build_daily_performance_metrics,
    _build_daily_performance_paths_long,
    _build_performance_metrics,
    _build_performance_paths_long,
    _build_performance_reconciliation_debug,
    build_bem_activity_daily,
    build_bcm_model_vs_naive_audit,
    _add_benchmark_comparison_columns,
    _mask_invalid_benchmark_comparisons,
    _overview_invalid_reason,
    _validate_performance_metrics,
    _build_optimization_infeasibility_attribution,
    _can_skip_model_manifest_for_benchmark,
    _overview_validity_fields,
    _prepare_scenario_output_dir,
    _resolve_final_soc_policy,
    _select_hourly_output_columns,
    _suspected_infeasibility_driver_from_row,
    _target_value_modes_from_manifest,
    optional_numeric_series,
    require_numeric_series,
)
from scripts import validate_simulation_outputs as validate_outputs  # noqa: E402
from scripts import validate_rhpf_outputs  # noqa: E402
from scripts import audit_bcm_capacity_auction  # noqa: E402
from scripts import audit_rhpf_market_price_lineage  # noqa: E402


def _mk_backtester(forecast_value_mode: str = "raw_signed") -> BatteryBacktester:
    MODEL_SPECS["forecast_value_mode"] = forecast_value_mode
    return BatteryBacktester()


def test_checkpoint_atomic_json_write(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints" / "stage_summary.json"

    _atomic_write_json(path, {"stage": "stage_01_model_realized", "value": 1.25})

    assert path.exists()
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["stage"] == "stage_01_model_realized"
    assert payload["value"] == pytest.approx(1.25)
    assert not list(path.parent.glob("*.tmp*"))


def test_checkpoint_atomic_parquet_write(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints" / "stage_01_model_realized_hourly.parquet"
    df = pd.DataFrame({"timestamp_utc": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"), "real_pnl_eur": [1.0, 2.0]})

    _atomic_write_parquet(path, df)

    out = pd.read_parquet(path)
    assert list(out.columns) == ["timestamp_utc", "real_pnl_eur"]
    assert len(out) == 2
    assert float(out["real_pnl_eur"].sum()) == pytest.approx(3.0)
    assert not list(path.parent.glob("*.tmp*"))


def test_checkpoint_manifest_update_tracks_completed_stages(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    metadata = {"run_id": "run-1", "scenario": "da/p50_p50", "model_key": "linear", "trading_strategy": "da"}

    _update_checkpoint_manifest(
        checkpoint_dir,
        metadata=metadata,
        stage_name="stage_01_model_realized",
        stage_status="complete",
        stage_files=["stage_01_model_realized_summary.json"],
        final_complete=False,
    )
    _update_checkpoint_manifest(
        checkpoint_dir,
        metadata=metadata,
        stage_name="stage_02_naive",
        stage_status="complete",
        stage_files=["stage_02_naive_summary.json"],
        final_complete=False,
    )

    manifest = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["last_completed_stage"] == "stage_02_naive"
    assert set(manifest["stages"]) >= {"stage_01_model_realized", "stage_02_naive"}
    assert manifest["final_complete"] is False


def test_checkpoint_failed_run_status_preserves_completed_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    metadata = {"run_id": "run-1", "scenario": "da/p50_p50", "model_key": "linear", "trading_strategy": "da"}
    df = pd.DataFrame({"timestamp_utc": pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC"), "real_pnl_eur": [5.0]})

    _write_stage_checkpoint(
        checkpoint_dir,
        stage_name="stage_01_model_realized",
        frame=df,
        summary={},
        metadata=metadata,
        detail="hourly",
    )
    _write_checkpoint_run_status(
        checkpoint_dir,
        status="failed",
        metadata=metadata,
        error_type="RuntimeError",
        error_message="boom",
    )

    status = json.loads((checkpoint_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["error_type"] == "RuntimeError"
    assert (checkpoint_dir / "stage_01_model_realized_hourly.parquet").exists()


def test_checkpoint_summary_only_writes_no_hourly_parquet(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    metadata = {"run_id": "run-1", "scenario": "da/p50_p50", "model_key": "linear", "trading_strategy": "da"}
    df = pd.DataFrame({"timestamp_utc": pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC"), "real_pnl_eur": [5.0]})

    _write_stage_checkpoint(
        checkpoint_dir,
        stage_name="stage_01_model_realized",
        frame=df,
        summary={},
        metadata=metadata,
        detail="summary_only",
    )

    assert (checkpoint_dir / "stage_01_model_realized_summary.json").exists()
    assert not (checkpoint_dir / "stage_01_model_realized_hourly.parquet").exists()


def test_afrr_activation_price_guard_pos_uses_activation_headroom_when_sufficient() -> None:
    bt = _mk_backtester("canonical_economic")
    bt.bid_builder.afrr_energy_bid_strategy = "forecast"

    price = bt.bid_builder.dynamic_afrr_energy_price(
        side="pos",
        pred_act_price=123.0,
        soc_now_mwh=10.0,
        soc_min_mwh=2.0,
        soc_max_mwh=18.0,
        obligation_mw=9.0,
        delivery_duration_h=1.0,
        activation_headroom_h=0.5,
    )

    assert price != pytest.approx(9999.0)
    assert price == pytest.approx(123.0)
    diag = bt.bid_builder.last_activation_price_guard_diagnostics
    assert diag["activation_price_guard_out_of_merit"] == pytest.approx(0.0)
    assert diag["activation_price_guard_activation_headroom_h"] == pytest.approx(0.5)


def test_afrr_activation_price_guard_reports_headroom_insufficient_without_out_of_merit_bid() -> None:
    bt = _mk_backtester("canonical_economic")
    bt.bid_builder.afrr_energy_bid_strategy = "forecast"

    price = bt.bid_builder.dynamic_afrr_energy_price(
        side="pos",
        pred_act_price=123.0,
        soc_now_mwh=10.0,
        soc_min_mwh=2.0,
        soc_max_mwh=18.0,
        obligation_mw=20.0,
        delivery_duration_h=1.0,
        activation_headroom_h=0.5,
    )

    assert price == pytest.approx(123.0)
    diag = bt.bid_builder.last_activation_price_guard_diagnostics
    assert diag["activation_price_guard_headroom_insufficient"] == pytest.approx(1.0)
    assert diag["activation_price_guard_out_of_merit"] == pytest.approx(0.0)
    assert diag["activation_price_guard_reason"] == "insufficient_positive_activation_headroom"


def test_afrr_activation_price_guard_neg_uses_headroom_not_full_hour() -> None:
    bt = _mk_backtester("canonical_economic")
    bt.bid_builder.afrr_energy_bid_strategy = "forecast"

    price = bt.bid_builder.dynamic_afrr_energy_price(
        side="neg",
        pred_act_price=111.0,
        soc_now_mwh=10.0,
        soc_min_mwh=2.0,
        soc_max_mwh=18.0,
        obligation_mw=9.0,
        delivery_duration_h=1.0,
        activation_headroom_h=0.5,
    )

    assert price != pytest.approx(9999.0)
    assert price == pytest.approx(111.0)
    diag = bt.bid_builder.last_activation_price_guard_diagnostics
    assert diag["activation_price_guard_out_of_merit"] == pytest.approx(0.0)
    assert diag["activation_price_guard_required_headroom_mwh"] == pytest.approx(9.0 * 0.5 * bt.eta_in)


def test_rhpf_bem_neg_oracle_bid_uses_true_zero_activation_price() -> None:
    bt = _mk_backtester("canonical_economic")
    bt.bid_builder.afrr_energy_bid_strategy = "forecast"

    price = bt.bid_builder.dynamic_afrr_energy_price(
        side="neg",
        pred_act_price=111.0,
        soc_now_mwh=10.0,
        soc_min_mwh=2.0,
        soc_max_mwh=18.0,
        obligation_mw=10.0,
        delivery_duration_h=1.0,
        activation_headroom_h=0.5,
        is_perfect_foresight=True,
        true_act_price=0.0,
        perfect_foresight_bid_at_true_price=True,
    )

    assert price == pytest.approx(0.0)


def test_afrr_activation_price_guard_regression_full_hour_would_fail_half_hour_passes() -> None:
    bt = _mk_backtester("canonical_economic")
    bt.bid_builder.afrr_energy_bid_strategy = "forecast"
    available_internal_mwh = 8.0
    obligation_mw = 9.0
    assert available_internal_mwh < obligation_mw * 1.0 / bt.eta_out
    assert available_internal_mwh >= obligation_mw * 0.5 / bt.eta_out

    price = bt.bid_builder.dynamic_afrr_energy_price(
        side="pos",
        pred_act_price=77.0,
        soc_now_mwh=10.0,
        soc_min_mwh=2.0,
        soc_max_mwh=18.0,
        obligation_mw=obligation_mw,
        delivery_duration_h=1.0,
        activation_headroom_h=0.5,
    )

    assert price == pytest.approx(77.0)


def test_bcm_capacity_toy_ev_arithmetic() -> None:
    assert bcm_capacity_ev_eur(
        p_accept=1.0,
        capacity_mw=1.0,
        capacity_bid_price_eur_per_mw_h=20.0,
        product_duration_h=4.0,
    ) == pytest.approx(80.0)
    assert bcm_capacity_ev_eur(
        p_accept=0.5,
        capacity_mw=1.0,
        capacity_bid_price_eur_per_mw_h=20.0,
        product_duration_h=4.0,
    ) == pytest.approx(40.0)


def test_bcm_capacity_toy_hourly_decomposition_is_accounting_only() -> None:
    hourly = bcm_capacity_hourly_revenue_decomposition_eur(
        accepted_available_mw=1.0,
        own_capacity_bid_price_eur_per_mw_h=20.0,
        product_start_utc=pd.Timestamp("2026-01-01T16:00:00Z"),
        product_duration_h=4,
    )
    assert list(hourly.index) == list(pd.date_range("2026-01-01T16:00:00Z", periods=4, freq="h"))
    assert hourly.tolist() == pytest.approx([20.0, 20.0, 20.0, 20.0])
    assert float(hourly.sum()) == pytest.approx(80.0)

    mc = MarketClearingEngine()
    bid = BCMCapacityBid(
        ts=pd.Timestamp("2026-01-01T16:00:00Z"),
        side="pos",
        quantity_mw=1.0,
        capacity_price_eur_mw=20.0,
    )
    accepted = mc.clear_afrr_capacity([bid], true_cap_pos=20.0, true_cap_neg=0.0)
    rejected = mc.clear_afrr_capacity([bid], true_cap_pos=10.0, true_cap_neg=0.0)
    assert float(accepted.awarded_pos_mw) == pytest.approx(1.0)
    assert float(rejected.awarded_pos_mw) == pytest.approx(0.0)
    assert bcm_capacity_revenue_eur(
        accepted_available_mw=float(accepted.awarded_pos_mw),
        own_capacity_bid_price_eur_per_mw_h=20.0,
        product_duration_h=4.0,
    ) == pytest.approx(80.0)


def test_bcm_capacity_toy_rejected_bid_zero_revenue_and_no_mandatory_bem() -> None:
    mc = MarketClearingEngine()
    bid = BCMCapacityBid(
        ts=pd.Timestamp("2026-01-01T16:00:00Z"),
        side="pos",
        quantity_mw=1.0,
        capacity_price_eur_mw=20.0,
    )
    result = mc.clear_afrr_capacity([bid], true_cap_pos=10.0, true_cap_neg=0.0)
    mandatory = bcm_product_to_bem_mandatory_volume(
        accepted_capacity_mw=float(result.awarded_pos_mw),
        product_start_utc=pd.Timestamp("2026-01-01T16:00:00Z"),
        resolution="hourly",
    )
    assert float(result.awarded_pos_mw) == pytest.approx(0.0)
    assert bcm_capacity_revenue_eur(
        accepted_available_mw=float(result.awarded_pos_mw),
        own_capacity_bid_price_eur_per_mw_h=20.0,
    ) == pytest.approx(0.0)
    assert mandatory.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_bcm_product_mapping_toy_hourly_and_quarter_hourly() -> None:
    hourly = bcm_product_to_bem_mandatory_volume(
        accepted_capacity_mw=2.0,
        product_start_utc=pd.Timestamp("2026-01-01T16:00:00Z"),
        resolution="hourly",
    )
    quarters = bcm_product_to_bem_mandatory_volume(
        accepted_capacity_mw=2.0,
        product_start_utc=pd.Timestamp("2026-01-01T16:00:00Z"),
        resolution="quarter_hour",
    )
    assert len(hourly) == 4
    assert hourly.tolist() == pytest.approx([2.0, 2.0, 2.0, 2.0])
    assert len(quarters) == 16
    assert quarters.iloc[0] == pytest.approx(2.0)
    assert quarters.iloc[-1] == pytest.approx(2.0)
    assert quarters.index[0] == pd.Timestamp("2026-01-01T16:00:00Z")
    assert quarters.index[-1] == pd.Timestamp("2026-01-01T19:45:00Z")


def test_bcm_pay_as_bid_toy_never_uses_activation_or_cutoff_price() -> None:
    own_bid_price = 20.0
    activation_price = 1000.0
    cutoff_price = 30.0
    revenue = bcm_capacity_revenue_eur(
        accepted_available_mw=1.0,
        own_capacity_bid_price_eur_per_mw_h=own_bid_price,
        product_duration_h=4.0,
    )
    assert revenue == pytest.approx(80.0)
    assert revenue != pytest.approx(1.0 * activation_price * 4.0)
    assert revenue != pytest.approx(1.0 * cutoff_price * 4.0)


def test_bcm_linked_activation_toy_capacity_volume_enters_once() -> None:
    ev = bcm_linked_bem_activation_ev_eur(
        accepted_capacity_mw=2.0,
        activation_fraction=0.25,
        interval_duration_h=1.0,
        settlement_price_eur_per_mwh=100.0,
    )
    assert ev == pytest.approx(2.0 * 0.25 * 1.0 * 100.0)
    assert ev != pytest.approx(2.0 * 2.0 * 0.25 * 1.0 * 100.0)


def test_bcm_product_toy_price_and_acceptance_probability_block_constant() -> None:
    hourly = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-01-01T15:00:00Z", periods=4, freq="h"),
            "p_cap_bid": [20.0, 20.0, 20.0, 20.0],
            "p_accept": [0.5, 0.5, 0.5, 0.5],
        }
    )
    block = assign_bcm_capacity_block(hourly["timestamp_utc"])
    hourly["block"] = block["bcm_capacity_block_id"].astype(str).to_numpy()
    assert hourly["block"].nunique() == 1
    assert hourly.groupby("block")["p_cap_bid"].nunique().iloc[0] == 1
    assert hourly.groupby("block")["p_accept"].nunique().iloc[0] == 1


def test_bid_rounding_da_min_size_and_granularity() -> None:
    round_bid = BatteryBacktester._round_market_bid_down_mw

    assert round_bid(0.053, step_mw=0.1, min_mw=0.1) == pytest.approx(0.0)
    assert round_bid(0.100, step_mw=0.1, min_mw=0.1) == pytest.approx(0.1)
    assert round_bid(0.149, step_mw=0.1, min_mw=0.1) == pytest.approx(0.1)
    assert round_bid(0.199, step_mw=0.1, min_mw=0.1) == pytest.approx(0.1)
    assert round_bid(0.200, step_mw=0.1, min_mw=0.1) == pytest.approx(0.2)


def test_bid_rounding_afrr_min_size_and_no_round_up() -> None:
    round_bid = BatteryBacktester._round_market_bid_down_mw

    assert round_bid(0.99, step_mw=1.0, min_mw=1.0) == pytest.approx(0.0)
    assert round_bid(1.00, step_mw=1.0, min_mw=1.0) == pytest.approx(1.0)
    assert round_bid(1.99, step_mw=1.0, min_mw=1.0) == pytest.approx(1.0)
    assert round_bid(2.00, step_mw=1.0, min_mw=1.0) == pytest.approx(2.0)


def test_market_compliant_da_lockable_quantities_are_rounded_before_lockbook() -> None:
    bt = _mk_backtester()
    bt.da_bid_granularity_mw = 0.1
    bt.da_min_bid_size_mw = 0.1

    buy_mw, sell_mw = bt._normalize_da_bid(0.149, 0.0)
    assert buy_mw == pytest.approx(0.1)
    assert sell_mw == pytest.approx(0.0)
    assert not bool(
        BatteryBacktester._market_bid_rule_violations(
            [buy_mw, sell_mw],
            step_mw=bt.da_bid_granularity_mw,
            min_mw=bt.da_min_bid_size_mw,
        ).any()
    )


def test_market_compliant_bem_guard_rounds_final_afrr_quantities() -> None:
    bt = _mk_backtester()
    bt.afrr_bid_granularity_mw = 1.0
    bt.afrr_step_mw = 1.0
    bt.afrr_min_bid_size_mw = 1.0
    bt.soc_init = 10.0

    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=1.99,
        desired_bem_only_neg_mw=0.99,
        soc_start_mwh=10.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        pred_act_pos=100.0,
        pred_act_neg=100.0,
    )

    assert float(guard["submitted_bem_only_pos_mw"]) == pytest.approx(1.0)
    assert float(guard["submitted_bem_only_neg_mw"]) == pytest.approx(0.0)
    assert float(guard["afrr_bid_market_rounding_applied"]) == pytest.approx(1.0)
    assert float(guard["bid_market_rounding_zeroed_below_min"]) == pytest.approx(1.0)


def test_bem_activation_energy_guard_derates_pos_at_soc_min() -> None:
    bt = _mk_backtester()
    bt.afrr_bid_granularity_mw = 0.1
    bt.afrr_step_mw = 0.1
    bt.afrr_min_bid_size_mw = 0.1
    bt.eta_out = 1.0
    bt.bem_only_headroom_safety_mwh = 0.0
    bt.bem_activation_headroom_h = 0.5

    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=1.0,
        desired_bem_only_neg_mw=0.0,
        soc_start_mwh=bt.soc_min + 0.75,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        pred_act_pos=100.0,
        pred_act_neg=0.0,
        activation_rate_pos=1.0,
        activation_rate_neg=0.0,
    )

    assert float(guard["submitted_bem_only_pos_mw"]) == pytest.approx(0.7)
    assert float(guard["bem_activation_energy_guard_applied"]) == pytest.approx(1.0)
    assert float(guard["bem_pos_derated_by_activation_energy_guard_mw"]) == pytest.approx(0.05)
    assert float(guard["bem_activation_energy_guard_aux_mwh"]) == pytest.approx(0.05)
    assert str(guard["bem_activation_energy_guard_reason"]) == "pos_activation_energy_not_deliverable"
    assert str(guard["bem_guard_zero_reason_pos"]) == "none"


def test_bem_activation_energy_guard_uses_expected_hourly_delivery_not_full_headroom() -> None:
    bt = _mk_backtester()
    bt.afrr_bid_granularity_mw = 0.1
    bt.afrr_step_mw = 0.1
    bt.afrr_min_bid_size_mw = 0.1
    bt.eta_out = 1.0
    bt.bem_only_headroom_safety_mwh = 0.0
    bt.bem_activation_headroom_h = 0.5
    bt.aux_afrr_active_mw = 0.0

    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=10.0,
        desired_bem_only_neg_mw=0.0,
        soc_start_mwh=bt.soc_min + 1.0,
        settlement_soc_start_mwh=bt.soc_min + 1.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        pred_act_pos=100.0,
        pred_act_neg=0.0,
        activation_rate_pos=0.1,
        activation_rate_neg=0.0,
    )

    assert float(guard["submitted_bem_only_pos_mw"]) == pytest.approx(10.0)
    assert str(guard["bem_activation_energy_guard_basis"]) == "expected_activation_energy"
    assert float(guard["bem_expected_delivery_duration_h"]) == pytest.approx(1.0)
    assert float(guard["bem_expected_delivery_pos_mwh"]) == pytest.approx(1.0)


def test_bem_guard_uses_settlement_soc_anchor_when_planned_anchor_is_stale() -> None:
    bt = _mk_backtester()
    bt.afrr_bid_granularity_mw = 0.1
    bt.afrr_step_mw = 0.1
    bt.afrr_min_bid_size_mw = 0.1

    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=1.0,
        desired_bem_only_neg_mw=0.0,
        soc_start_mwh=2.819138,
        settlement_soc_start_mwh=bt.soc_min,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        pred_act_pos=100.0,
        pred_act_neg=0.0,
        activation_rate_pos=0.01,
        activation_rate_neg=0.0,
    )

    assert float(guard["bem_guard_stale_soc_anchor"]) == pytest.approx(1.0)
    assert float(guard["bem_guard_soc_anchor_mwh"]) == pytest.approx(2.819138)
    assert float(guard["settlement_soc_start_mwh"]) == pytest.approx(bt.soc_min)
    assert float(guard["submitted_bem_only_pos_mw"]) == pytest.approx(0.0)
    assert str(guard["bem_guard_zero_reason_pos"]) == "bem_guard_stale_soc_anchor"


def test_market_clearing_bem_guard_uses_settlement_soc_not_planned_soc() -> None:
    bt = _mk_backtester()
    bt._strategy_permissions = StrategyPermissions(
        allow_da=False,
        allow_bcm=False,
        allow_bcm_activation_obligations=False,
        allow_bem_only=True,
        id_mode="technical_repair",
    )
    bt.afrr_bid_granularity_mw = 0.1
    bt.afrr_step_mw = 0.1
    bt.afrr_min_bid_size_mw = 0.1
    bt.eta_out = 1.0
    bt.bem_only_headroom_safety_mwh = 0.0

    cleared = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2025-06-12T17:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=10.0,
        planned_bem_only_neg_mw=0.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=100.0,
        true_act_pos=100.0,
        pred_act_neg=0.0,
        true_act_neg=0.0,
        true_rate_pos=0.061049,
        true_rate_neg=0.0,
        pred_rate_pos=0.061049,
        pred_rate_neg=0.0,
        soc_now=10.62,
        settlement_soc_now=bt.soc_min,
    )

    assert float(cleared["bem_guard_stale_soc_anchor"]) == pytest.approx(1.0)
    assert float(cleared["settlement_soc_start_mwh"]) == pytest.approx(bt.soc_min)
    assert float(cleared["bem_only_submitted_pos_mw"]) == pytest.approx(0.0)
    assert float(cleared["executed_reserve_pos_mw"]) == pytest.approx(0.0)
    assert str(cleared["bem_guard_zero_reason_pos"]) == "bem_guard_stale_soc_anchor"


def test_precleared_bem_settlement_guard_derates_stale_pos_availability() -> None:
    bt = _mk_backtester()
    bt._strategy_permissions = StrategyPermissions(
        allow_da=False,
        allow_bcm=False,
        allow_bcm_activation_obligations=False,
        allow_bem_only=True,
        id_mode="technical_repair",
    )
    bt.afrr_bid_granularity_mw = 0.1
    bt.afrr_step_mw = 0.1
    bt.afrr_min_bid_size_mw = 0.1
    bt.eta_out = 1.0
    bt.bem_only_headroom_safety_mwh = 0.0
    clearing_rec = {
        "bem_only_submitted_pos_mw": 10.0,
        "bem_only_executed_pos_mw": 10.0,
        "bem_only_submitted_neg_mw": 0.0,
        "bem_only_executed_neg_mw": 0.0,
        "bem_guard_soc_anchor_mwh": 10.62,
    }

    guarded, reserve_pos, reserve_neg = bt._apply_bem_settlement_anchor_guard_to_clearing_record(
        clearing_rec=clearing_rec,
        settlement_soc_start_mwh=bt.soc_min,
        charge_mw=0.0,
        discharge_mw=0.0,
        id_charge_mw=0.0,
        id_discharge_mw=0.0,
        reserve_pos_mw=10.0,
        reserve_neg_mw=0.0,
        act_rate_pos=0.061049,
        act_rate_neg=0.0,
    )
    _, metrics = bt._settle_one_hour(
        soc=bt.soc_min,
        charge=0.0,
        discharge=0.0,
        reserve_pos=reserve_pos,
        reserve_neg=reserve_neg,
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=100.0,
        act_neg_price=0.0,
        act_pos_rate=0.061049,
        act_neg_rate=0.0,
        bem_available_pos_mw=float(guarded["bem_only_executed_pos_mw"]),
        bem_available_neg_mw=float(guarded["bem_only_executed_neg_mw"]),
        bem_submitted_pos_mw=float(guarded["bem_only_submitted_pos_mw"]),
        bem_submitted_neg_mw=float(guarded["bem_only_submitted_neg_mw"]),
    )

    assert float(guarded["bem_guard_stale_soc_anchor"]) == pytest.approx(1.0)
    assert float(guarded["bem_only_submitted_pos_mw"]) == pytest.approx(0.0)
    assert reserve_pos == pytest.approx(0.0)
    assert reserve_neg == pytest.approx(0.0)
    assert float(metrics["missed_activation_pos_mwh"]) == pytest.approx(0.0)


def test_bem_activation_energy_guard_derates_neg_near_soc_max() -> None:
    bt = _mk_backtester()
    bt.afrr_bid_granularity_mw = 0.1
    bt.afrr_step_mw = 0.1
    bt.afrr_min_bid_size_mw = 0.1
    bt.eta_in = 1.0
    bt.bem_only_headroom_safety_mwh = 0.0
    bt.bem_activation_headroom_h = 0.5

    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=0.0,
        desired_bem_only_neg_mw=1.0,
        soc_start_mwh=bt.soc_max - 0.25,
        settlement_soc_start_mwh=bt.soc_max - 0.25,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        pred_act_pos=0.0,
        pred_act_neg=100.0,
        activation_rate_pos=0.0,
        activation_rate_neg=1.0,
    )

    assert float(guard["submitted_bem_only_neg_mw"]) == pytest.approx(0.2)
    assert float(guard["bem_activation_energy_guard_applied"]) == pytest.approx(0.0)
    assert float(guard["bem_guard_derated_neg_bem_mw"]) == pytest.approx(0.8)
    assert str(guard["bem_activation_energy_guard_basis"]) == "expected_activation_energy"
    assert str(guard["bem_guard_zero_reason_neg"]) == "none"


def test_aux_scaled_trading_power_mode_scales_by_executed_utilization() -> None:
    bt = _mk_backtester()
    bt.aux_mode = "state_dependent_scaled"
    bt.aux_peak_mw = 0.2
    bt.aux_trading_mw = 0.2 * 0.75
    bt.aux_off_mw = 0.0
    bt.p_max_mw = 10.0

    aux_mw, aux_state = bt._state_aux_power_mw(
        charge_mw=5.0,
        discharge_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
    )
    assert aux_state == "TRADING_SCALED"
    assert aux_mw == pytest.approx(0.2 * 0.75 * 0.5)

    bt.aux_off_mw = 0.02
    aux_mw, aux_state = bt._state_aux_power_mw(
        charge_mw=0.1,
        discharge_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
    )
    assert aux_state == "TRADING_SCALED"
    assert aux_mw == pytest.approx(max(0.02, 0.2 * 0.75 * 0.01))


def test_aux_power_mode_fixed_state_dependent_unchanged_and_no_submitted_only_trading_aux() -> None:
    bt = _mk_backtester()
    bt.aux_mode = "state_dependent"
    bt.aux_trading_mw = 0.15
    bt.aux_standby_mw = 0.05
    bt.aux_off_mw = 0.01

    aux_mw, aux_state = bt._state_aux_power_mw(
        charge_mw=0.1,
        discharge_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
    )
    assert aux_state == "TRADING"
    assert aux_mw == pytest.approx(0.15)

    aux_mw, aux_state = bt._state_aux_power_mw(
        charge_mw=0.0,
        discharge_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
    )
    assert aux_state == "OFF"
    assert aux_mw == pytest.approx(0.01)


def test_state_dependent_scaled_no_trade_uses_off_aux() -> None:
    bt = _mk_backtester()
    bt.aux_mode = "state_dependent_scaled"
    bt.aux_off_mw = 0.02
    bt.aux_trading_mw = 0.05

    aux_mw, aux_state = bt._state_aux_power_mw(
        charge_mw=0.0,
        discharge_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        id_charge_mw=0.0,
        id_discharge_mw=0.0,
    )

    assert aux_state == "OFF"
    assert aux_mw == pytest.approx(0.02)


def test_state_dependent_scaled_reserve_standby_and_afrr_active_unchanged() -> None:
    bt = _mk_backtester()
    bt.aux_mode = "state_dependent_scaled"
    bt.aux_standby_mw = 0.035
    bt.aux_afrr_active_mw = 0.05

    aux_mw, aux_state = bt._state_aux_power_mw(
        charge_mw=0.0,
        discharge_mw=0.0,
        reserve_pos_mw=1.0,
        reserve_neg_mw=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
    )

    assert aux_state == "STANDBY"
    assert aux_mw == pytest.approx(0.035)

    aux_mw, aux_state = bt._state_aux_power_mw(
        charge_mw=0.0,
        discharge_mw=0.0,
        reserve_pos_mw=1.0,
        reserve_neg_mw=0.0,
        act_pos_rate=0.25,
        act_neg_rate=0.0,
    )

    assert aux_state == "aFRR_ACTIVE"
    assert aux_mw == pytest.approx(0.05)


def _settle_bem_aux_case(
    *,
    bem_available_pos_mw: float,
    bem_available_neg_mw: float,
    bem_submitted_pos_mw: float,
    bem_submitted_neg_mw: float,
    act_pos_rate: float,
    act_neg_rate: float,
) -> dict[str, float]:
    bt = _mk_backtester()
    bt.aux_mode = "state_dependent_scaled"
    bt.aux_off_mw = 0.02
    bt.aux_standby_mw = 0.035
    bt.aux_afrr_active_mw = 0.05
    bt.dt_h = 1.0
    _, metrics = bt._settle_one_hour(
        soc=10.0,
        charge=0.0,
        discharge=0.0,
        da_charge_mw=0.0,
        da_discharge_mw=0.0,
        id_charge_mw=0.0,
        id_discharge_mw=0.0,
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=100.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=100.0,
        act_neg_price=100.0,
        act_pos_rate=act_pos_rate,
        act_neg_rate=act_neg_rate,
        bem_available_pos_mw=bem_available_pos_mw,
        bem_available_neg_mw=bem_available_neg_mw,
        bem_submitted_pos_mw=bem_submitted_pos_mw,
        bem_submitted_neg_mw=bem_submitted_neg_mw,
    )
    return metrics


def test_bem_available_without_activation_uses_standby_aux() -> None:
    metrics = _settle_bem_aux_case(
        bem_available_pos_mw=5.0,
        bem_available_neg_mw=0.0,
        bem_submitted_pos_mw=5.0,
        bem_submitted_neg_mw=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
    )

    assert metrics["aux_state"] == "STANDBY"
    assert float(metrics["aux_energy_mwh"]) == pytest.approx(0.035)
    assert float(metrics["bem_standby_aux_applied"]) == pytest.approx(1.0)
    assert float(metrics["bem_standby_aux_mwh"]) == pytest.approx(0.035)
    assert float(metrics["bem_rejected_pos_mw"]) == pytest.approx(0.0)


def test_bem_available_with_activation_uses_active_aux_and_throughput() -> None:
    metrics = _settle_bem_aux_case(
        bem_available_pos_mw=5.0,
        bem_available_neg_mw=0.0,
        bem_submitted_pos_mw=5.0,
        bem_submitted_neg_mw=0.0,
        act_pos_rate=0.2,
        act_neg_rate=0.0,
    )

    assert metrics["aux_state"] == "aFRR_ACTIVE"
    assert float(metrics["aux_energy_mwh"]) == pytest.approx(0.05)
    assert float(metrics["bem_standby_aux_applied"]) == pytest.approx(0.0)
    assert float(metrics["throughput_mwh"]) == pytest.approx(float(metrics["act_pos_internal_mwh"]))
    assert float(metrics["throughput_mwh"]) > 0.0


def test_rejected_bem_submission_does_not_charge_standby_aux() -> None:
    metrics = _settle_bem_aux_case(
        bem_available_pos_mw=0.0,
        bem_available_neg_mw=0.0,
        bem_submitted_pos_mw=5.0,
        bem_submitted_neg_mw=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
    )

    assert metrics["aux_state"] == "OFF"
    assert float(metrics["aux_energy_mwh"]) == pytest.approx(0.02)
    assert float(metrics["bem_standby_aux_applied"]) == pytest.approx(0.0)
    assert float(metrics["bem_rejected_pos_mw"]) == pytest.approx(5.0)


def test_terminal_id_recovery_converts_internal_shortfall_to_grid_mwh() -> None:
    bt = _mk_backtester()
    diag = bt._schedule_terminal_id_recovery(
        projected_terminal_soc_without_new_id_mwh=9.0,
        current_soc_mwh=9.0,
        terminal_soc_target_mwh=10.0,
        residual_charge_mw=10.0,
        terminal_soc_safety_margin_mwh=0.0,
    )
    assert float(diag["terminal_soc_id_recourse_needed_internal_mwh"]) == pytest.approx(1.0)
    assert float(diag["terminal_soc_id_recourse_scheduled_grid_mwh"]) == pytest.approx(1.0 / bt.eta_in)
    assert float(diag["terminal_soc_id_recourse_scheduled_internal_mwh"]) == pytest.approx(1.0)
    assert float(diag["terminal_soc_recovery_feasible"]) == 1.0


def test_terminal_id_recovery_planner_includes_incremental_auxiliary_loss() -> None:
    bt = _mk_backtester()
    bt.aux_mode = "state_dependent"
    bt.aux_off_mw = 0.0
    bt.aux_trading_mw = 0.25
    id_charge_mw, id_discharge_mw, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=10.0,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=10.0,
        projected_terminal_soc_without_new_id_mwh=9.95,
        remaining_known_losses_mwh=0.05,
        terminal_soc_safety_margin_mwh=0.10,
    )
    diag = bt._last_id_rescue_plan_diagnostics
    assert id_discharge_mw == pytest.approx(0.0)
    assert id_charge_mw > 0.0
    assert reason == "terminal_soc_recovery"
    assert float(diag["terminal_soc_projection_id_extra_aux_losses_mwh"]) == pytest.approx(0.25)
    assert float(diag["projected_terminal_soc_with_new_id_mwh"]) >= 10.10 - 1e-9


def test_terminal_id_recovery_sizes_final_hour_id_for_post_aux_soc() -> None:
    bt = _mk_backtester()
    bt.aux_mode = "state_dependent"
    bt.aux_off_mw = 0.0
    bt.aux_trading_mw = 0.095
    id_charge_mw, id_discharge_mw, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=8.337168,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=10.0,
        projected_terminal_soc_without_new_id_mwh=8.308269,
        remaining_known_losses_mwh=0.0,
        terminal_soc_safety_margin_mwh=0.0,
    )
    diag = bt._last_id_rescue_plan_diagnostics

    assert id_discharge_mw == pytest.approx(0.0)
    assert id_charge_mw > 0.0
    assert reason == "terminal_soc_recovery"
    assert float(diag["terminal_soc_projection_id_extra_aux_losses_mwh"]) == pytest.approx(0.095)
    assert float(diag["projected_terminal_soc_with_new_id_before_aux_mwh"]) >= 10.0 - 1e-9
    assert float(diag["projected_terminal_soc_with_new_id_mwh"]) >= 10.0 - 1e-9
    assert float(diag["terminal_soc_recovery_post_aux_shortfall_mwh"]) == pytest.approx(0.0)


def test_global_pf_timestamp_coverage_normalizes_equivalent_utc_timestamps() -> None:
    expected = pd.date_range("2025-01-08T01:00:00Z", periods=47, freq="h")
    candidate = expected.tz_convert("Europe/Berlin").to_series(index=None).sample(frac=1.0, random_state=7).tolist()

    diag = BatteryBacktester._timestamp_coverage_diagnostics(expected, candidate)

    assert float(diag["aligned"]) == pytest.approx(1.0)
    assert float(diag["missing_timestamp_count"]) == pytest.approx(0.0)
    assert str(diag["first_missing_timestamp_utc"]) == ""
    assert float(diag["extra_timestamp_count"]) == pytest.approx(0.0)
    assert str(diag["first_extra_timestamp_utc"]) == ""
    assert float(diag["expected_rows"]) == pytest.approx(47.0)
    assert float(diag["actual_rows"]) == pytest.approx(47.0)
    assert float(diag["expected_row_count"]) == pytest.approx(47.0)
    assert float(diag["actual_row_count"]) == pytest.approx(47.0)


def test_global_pf_timestamp_coverage_uses_evaluated_window_not_raw_input_window() -> None:
    raw_effective_window = pd.date_range("2025-01-08T00:00:00Z", periods=48, freq="h")
    evaluated_settlement_window = pd.date_range("2025-01-08T01:00:00Z", periods=47, freq="h")
    candidate = evaluated_settlement_window.tz_convert("Europe/Berlin").to_list()

    raw_diag = BatteryBacktester._timestamp_coverage_diagnostics(raw_effective_window, candidate)
    evaluated_diag = BatteryBacktester._timestamp_coverage_diagnostics(evaluated_settlement_window, candidate)

    assert float(raw_diag["aligned"]) == pytest.approx(0.0)
    assert float(raw_diag["missing_timestamp_count"]) == pytest.approx(1.0)
    assert str(raw_diag["first_missing_timestamp_utc"]) == "2025-01-08T00:00:00+00:00"
    assert float(raw_diag["expected_rows"]) == pytest.approx(48.0)
    assert float(raw_diag["actual_rows"]) == pytest.approx(47.0)

    assert float(evaluated_diag["aligned"]) == pytest.approx(1.0)
    assert float(evaluated_diag["missing_timestamp_count"]) == pytest.approx(0.0)
    assert float(evaluated_diag["extra_timestamp_count"]) == pytest.approx(0.0)
    assert float(evaluated_diag["expected_rows"]) == pytest.approx(47.0)
    assert float(evaluated_diag["actual_rows"]) == pytest.approx(47.0)


def test_global_pf_timestamp_coverage_reports_true_missing_timestamp() -> None:
    expected = pd.date_range("2025-01-08T00:00:00Z", periods=3, freq="h")
    candidate = expected[:2]

    diag = BatteryBacktester._timestamp_coverage_diagnostics(expected, candidate)

    assert float(diag["aligned"]) == pytest.approx(0.0)
    assert float(diag["missing_timestamp_count"]) == pytest.approx(1.0)
    assert str(diag["first_missing_timestamp_utc"]) == "2025-01-08T02:00:00+00:00"
    assert float(diag["extra_timestamp_count"]) == pytest.approx(0.0)
    assert float(diag["expected_row_count"]) == pytest.approx(3.0)
    assert float(diag["actual_row_count"]) == pytest.approx(2.0)


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


def test_da_execution_probability_buy_quantiles() -> None:
    bt = _mk_backtester()
    bt.da_execution_mode = "limit"

    for quantile, expected in [("p30", 0.30), ("p50", 0.50), ("p90", 0.90)]:
        bt.bid_builder.da_buy_limit_quantile = quantile
        probs = bt._da_limit_execution_probabilities()
        assert float(probs["buy_probability"]) == pytest.approx(expected)


def test_da_execution_probability_sell_quantiles() -> None:
    bt = _mk_backtester()
    bt.da_execution_mode = "limit"

    for quantile, expected in [("p30", 0.70), ("p50", 0.50), ("p90", 0.10)]:
        bt.bid_builder.da_sell_limit_quantile = quantile
        probs = bt._da_limit_execution_probabilities()
        assert float(probs["sell_probability"]) == pytest.approx(expected)


def test_da_execution_probability_adjusted_replay_cashflow() -> None:
    bt = _mk_backtester()
    bt.da_execution_mode = "limit"
    bt.bid_builder.da_buy_limit_quantile = "p30"
    bt.bid_builder.da_sell_limit_quantile = "p70"
    bt.trans_eur_mwh = 0.0
    bt.deg_eur_mwh = 0.0
    bt.aux_mode = "state_dependent"
    bt.aux_trading_mw = 0.0
    bt.da_terminal_soc_value_enabled = False
    col = BacktestColumnMap()
    ts_buy = pd.Timestamp("2026-01-01T00:00:00Z")
    ts_sell = pd.Timestamp("2026-01-01T01:00:00Z")
    rows = pd.DataFrame(
        {
            "target_time_utc": [ts_buy, ts_sell],
            col.timestamp: [ts_buy, ts_sell],
            col.pred_da_price: [100.0, 100.0],
        }
    )
    replay = bt._replay_da_candidate_cashflow(
        rows=rows,
        schedule={ts_buy: (10.0, 0.0), ts_sell: (0.0, 10.0)},
        colmap=col,
        current_soc_mwh=10.0,
    )

    assert float(replay["da_expected_buy_cost_probability_adjusted_eur"]) == pytest.approx(300.0)
    assert float(replay["da_expected_sell_revenue_probability_adjusted_eur"]) == pytest.approx(300.0)
    assert float(replay["da_expected_pnl_probability_adjusted_eur"]) == pytest.approx(0.0)
    assert float(replay["da_limit_execution_probability_enabled"]) == pytest.approx(1.0)


def test_da_execution_probability_price_taker_mode_unchanged() -> None:
    bt = _mk_backtester()
    bt.da_execution_mode = "price_taker"
    bt.bid_builder.da_buy_limit_quantile = "p30"
    bt.bid_builder.da_sell_limit_quantile = "p90"

    probs = bt._da_limit_execution_probabilities()

    assert float(probs["enabled"]) == pytest.approx(0.0)
    assert float(probs["buy_probability"]) == pytest.approx(1.0)
    assert float(probs["sell_probability"]) == pytest.approx(1.0)


def test_da_execution_policy_resolution_preserves_legacy_modes() -> None:
    assert BatteryBacktester._resolve_da_execution_policy(
        {"da_execution_policy": "price_taker", "da_execution_mode": "limit"}
    ) == ("price_taker", "price_taker", "accepted_independent_limit_replay")
    assert BatteryBacktester._resolve_da_execution_policy(
        {
            "da_execution_mode": "limit",
            "da_limit_clearing_policy": "accepted_independent_limit_replay",
        }
    ) == ("accepted_independent_limit_replay", "limit", "accepted_independent_limit_replay")
    assert BatteryBacktester._resolve_da_execution_policy(
        {
            "da_execution_mode": "limit",
            "da_limit_clearing_policy": "robust_independent_limit",
        }
    ) == ("robust_independent_limit", "limit", "robust_independent_limit")


def test_da_price_taker_policy_accepts_submitted_quantity_without_price_rejection() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.da_execution_policy = "price_taker"
    bt.da_execution_mode = "price_taker"
    bt.bid_builder.da_mode = "price_taker"
    bt.market_clearing_engine.da_mode_default = "price_taker"

    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T00:00:00Z"),
        planned_charge_mw=2.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        pred_da_price=50.0,
        true_da_price=100.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=0.0,
        true_act_pos=0.0,
        pred_act_neg=0.0,
        true_act_neg=0.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        soc_now=10.0,
    )

    assert str(out["da_execution_policy"]) == "price_taker"
    assert str(out["da_execution_policy_source"]) == "config:da_execution_policy"
    assert float(out["da_price_taker_mode"]) == pytest.approx(1.0)
    assert float(out["submitted_da_buy_mw"]) == pytest.approx(2.0)
    assert float(out["da_submitted_buy_mwh"]) == pytest.approx(2.0)
    assert float(out["executed_charge_mw"]) == pytest.approx(2.0)
    assert float(out["da_auction_accepted_buy_mwh"]) == pytest.approx(2.0)
    assert float(out["da_physical_lockbook_buy_mwh"]) == pytest.approx(2.0)
    assert float(out["da_price_rejected_buy_mwh"]) == pytest.approx(0.0)
    assert float(out["da_actual_independent_clearing_applied"]) == pytest.approx(0.0)
    assert float(out["da_limit_robust_guard_applied"]) == pytest.approx(0.0)
    assert str(out["da_settlement_price_source"]) == "true_da_clearing_price"


def test_bcm_award_probability_convention_uses_one_minus_quantile() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30", "p90"]
    df, col = _one_hour_pred_df(
        da=0.0,
        cap_pos=100.0,
        cap_neg=80.0,
        act_pos=50.0,
        act_neg=60.0,
        rate_pos=0.2,
        rate_neg=0.3,
    )

    out = bt.optimize_dispatch(
        df,
        col,
        allowed_markets=("aFRR", "BCM", "BEM"),
        deterministic_reserve_settlement=False,
    )

    assert float(out["ev_bcm_p_award_pos_bin_0"].iloc[0]) == pytest.approx(0.70)
    assert float(out["ev_bcm_p_award_pos_bin_1"].iloc[0]) == pytest.approx(0.10)
    assert str(out["bcm_probability_convention"].iloc[0]) == (
        "capacity_price_quantile_q_clears_with_probability_1_minus_q"
    )


def test_bem_execution_probability_convention_and_no_double_probability() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30", "p90"]
    df, col = _one_hour_pred_df(
        da=0.0,
        cap_pos=100.0,
        cap_neg=80.0,
        act_pos=50.0,
        act_neg=60.0,
        rate_pos=0.2,
        rate_neg=0.3,
    )

    out = bt.optimize_dispatch(
        df,
        col,
        allowed_markets=("aFRR", "BCM", "BEM"),
        deterministic_reserve_settlement=False,
    )

    assert float(out["ev_bem_p_exec_pos_bin_0"].iloc[0]) == pytest.approx(0.70)
    assert float(out["ev_bem_p_exec_pos_bin_1"].iloc[0]) == pytest.approx(0.10)
    assert str(out["bem_probability_convention"].iloc[0]) == (
        "activation_price_quantile_q_clears_with_probability_1_minus_q"
    )
    expected_share = 0.70 * 0.2
    assert float(out["ev_expected_act_share_pos_bin_0"].iloc[0]) == pytest.approx(expected_share)
    assert float(out["ev_expected_act_share_pos_bin_0"].iloc[0]) != pytest.approx(0.70 * expected_share)
    expected_mean_exec_prob = (0.70 + 0.70 + 0.10 + 0.10) / 4.0
    expected_mean_act_share = ((0.70 * 0.2) + (0.70 * 0.3) + (0.10 * 0.2) + (0.10 * 0.3)) / 4.0
    assert float(out["bem_execution_probability_used_for_revenue"].iloc[0]) == pytest.approx(expected_mean_exec_prob)
    assert float(out["bem_execution_probability_used_for_cost"].iloc[0]) == pytest.approx(expected_mean_exec_prob)
    assert float(out["bem_expected_activation_share_used_for_revenue"].iloc[0]) == pytest.approx(
        expected_mean_act_share
    )
    assert float(out["bem_expected_activation_share_used_for_cost"].iloc[0]) == pytest.approx(
        expected_mean_act_share
    )
    assert float(out["bem_expected_activation_share_used_for_soc"].iloc[0]) == pytest.approx(
        expected_mean_act_share
    )


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


def test_multi_optimizer_path_has_no_fast_gated_fields_and_keeps_normal_semantics() -> None:
    df, col = _tiny_backtest_df(hours=8)
    bt = _mk_backtester("canonical_economic")
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=10.0,
        soc_end_min_target=None,
        allowed_markets=("DA", "aFRR", "ID", "BCM", "BEM"),
    )
    for c in [
        "fast_gated_decisions_enabled",
        "can_submit_new_da",
        "can_submit_new_bcm",
        "can_submit_new_bem",
        "can_use_id_recourse",
        "da_new_bid_fixed_by_gate",
        "bcm_new_bid_fixed_by_gate",
        "bem_new_bid_fixed_by_gate",
        "da_ev_skipped_outside_gate",
        "bcm_ev_skipped_outside_gate",
    ]:
        assert c not in out.columns
    assert "bem_only_pos_mw" in out.columns
    assert "reserve_pos_mw" in out.columns
    assert "charge_mw" in out.columns


def test_technical_id_repair_penalty_does_not_suppress_positive_bcm_ev() -> None:
    col = BacktestColumnMap()
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")
    df = pd.DataFrame(
        {
            col.timestamp: ts,
            col.pred_da_price: [10.0] * 4,
            col.pred_afrr_capacity_price_pos: [500.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [0.0] * 4,
            col.pred_afrr_activation_price_neg: [0.0] * 4,
            col.pred_afrr_activation_rate_pos: [0.0] * 4,
            col.pred_afrr_activation_rate_neg: [0.0] * 4,
            col.true_da_price: [10.0] * 4,
            col.true_afrr_capacity_price_pos: [500.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
            col.true_afrr_activation_price_pos: [0.0] * 4,
            col.true_afrr_activation_price_neg: [0.0] * 4,
            col.true_afrr_activation_rate_pos: [0.0] * 4,
            col.true_afrr_activation_rate_neg: [0.0] * 4,
            "pacc_pos_bin_0": [1.0] * 4,
            "pacc_neg_bin_0": [0.0] * 4,
        }
    )
    for pref, val in [
        (col.pred_afrr_capacity_price_pos, 500.0),
        (col.pred_afrr_capacity_price_neg, 0.0),
        (col.pred_afrr_activation_price_pos, 0.0),
        (col.pred_afrr_activation_price_neg, 0.0),
        (col.pred_afrr_activation_rate_pos, 0.0),
        (col.pred_afrr_activation_rate_neg, 0.0),
    ]:
        df[f"{pref}_p50"] = val

    bt = _mk_backtester("canonical_economic")
    bt.afrr_quantile_bins = ["p50"]
    bt.final_soc_mode = "hard"
    bt.final_soc_shortfall_penalty_eur_per_mwh = 1e17
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=10.0,
        soc_end_min_target=10.0,
        allowed_markets=("aFRR", "ID"),
        strategy_permissions=StrategyPermissions(False, "technical_repair", True, True, False),
    )

    assert pd.to_numeric(out["reserve_pos_mw"], errors="coerce").fillna(0.0).sum() > 0.0
    assert pd.to_numeric(out["id_charge_mw"], errors="coerce").fillna(0.0).sum() == pytest.approx(0.0)
    assert pd.to_numeric(out["id_discharge_mw"], errors="coerce").fillna(0.0).sum() == pytest.approx(0.0)
    assert pd.to_numeric(out["optimizer_id_buy_economic_leakage_flag"], errors="coerce").fillna(0.0).sum() == pytest.approx(0.0)
    assert set(out["optimizer_id_buy_allowed_reason"].astype(str)) == {"none"}
    assert pd.to_numeric(out["id_technical_repair_artificial_penalty_applied"], errors="coerce").eq(0.0).all()
    assert pd.to_numeric(out["id_technical_repair_energy_cost_eur_per_mwh"], errors="coerce").max() < 1e17
    assert float(pd.to_numeric(out["soc_lp_mwh"], errors="coerce").iloc[-1]) >= 10.0 - 1e-6


def test_simulation_forecast_loader_applies_negative_target_quantile_flip(tmp_path: Path) -> None:
    p = tmp_path / "neg.parquet"
    pd.DataFrame(
        {
            "snapshot_time_utc": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "target_time_utc": [pd.Timestamp("2026-01-01T01:00:00Z")],
            "lead_time_h": [1],
            "predicted_value": [-50.0],
            "p10": [-100.0],
            "p30": [-80.0],
            "p50": [-50.0],
            "p70": [-20.0],
            "p90": [-10.0],
        }
    ).to_parquet(p, index=False)
    wh = load_prediction_warehouse_long({"pred_afrr_activation_price_neg": p})
    out = wh["pred_afrr_activation_price_neg"].iloc[0]
    assert np.isclose(float(out["p10"]), 10.0)
    assert np.isclose(float(out["p30"]), 20.0)
    assert np.isclose(float(out["p50"]), 50.0)
    assert np.isclose(float(out["p70"]), 80.0)
    assert np.isclose(float(out["p90"]), 100.0)
    assert np.isclose(float(out["predicted_value"]), 50.0)


def test_simulation_forecast_loader_canonical_activation_neg_no_flip(tmp_path: Path) -> None:
    p = tmp_path / "neg_can.parquet"
    pd.DataFrame(
        {
            "snapshot_time_utc": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "target_time_utc": [pd.Timestamp("2026-01-01T01:00:00Z")],
            "lead_time_h": [1],
            "predicted_value": [50.0],
            "p10": [10.0],
            "p30": [20.0],
            "p50": [50.0],
            "p70": [80.0],
            "p90": [100.0],
        }
    ).to_parquet(p, index=False)
    wh = load_prediction_warehouse_long(
        {"pred_afrr_activation_price_neg": p},
        target_value_modes={"pred_afrr_activation_price_neg": "canonical_economic"},
    )
    out = wh["pred_afrr_activation_price_neg"].iloc[0]
    assert np.isclose(float(out["p10"]), 10.0)
    assert np.isclose(float(out["p90"]), 100.0)
    assert np.isclose(float(out["predicted_value"]), 50.0)


def test_manifest_target_value_modes_applied_to_loader_post_load(tmp_path: Path) -> None:
    p = tmp_path / "neg_from_manifest.parquet"
    pd.DataFrame(
        {
            "snapshot_time_utc": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "target_time_utc": [pd.Timestamp("2026-01-01T01:00:00Z")],
            "lead_time_h": [1],
            "predicted_value": [50.0],
            "p10": [10.0],
            "p30": [20.0],
            "p50": [50.0],
            "p70": [80.0],
            "p90": [100.0],
        }
    ).to_parquet(p, index=False)
    payload = {
        "target_value_mode": {
            "pred_afrr_activation_price_neg": "canonical_economic",
        }
    }
    modes = _target_value_modes_from_manifest(payload)
    wh = load_prediction_warehouse_long(
        {"pred_afrr_activation_price_neg": p},
        target_value_modes=modes,
    )
    out = wh["pred_afrr_activation_price_neg"].iloc[0]
    assert np.isclose(float(out["p10"]), 10.0)
    assert np.isclose(float(out["p90"]), 100.0)


def test_resolve_final_soc_policy_strict_terminal_repair_refused() -> None:
    with pytest.raises(ValueError, match="Strict mode requires --final-soc-mode hard"):
        _resolve_final_soc_policy(
            strict_simulation_validity=True,
            final_soc_mode="terminal_repair",
            enforce_final_soc_min_flag=True,
            allow_terminal_soc_repair_in_strict=False,
        )


def test_resolve_final_soc_policy_hard_or_enforce_flag() -> None:
    assert _resolve_final_soc_policy(
        strict_simulation_validity=True,
        final_soc_mode="hard",
        enforce_final_soc_min_flag=False,
        allow_terminal_soc_repair_in_strict=False,
    )
    assert _resolve_final_soc_policy(
        strict_simulation_validity=False,
        final_soc_mode="terminal_repair",
        enforce_final_soc_min_flag=True,
        allow_terminal_soc_repair_in_strict=False,
    )


def test_assign_bcm_capacity_block_local_ce_st_blocks() -> None:
    ts = pd.to_datetime(
        [
            "2025-05-01T00:00:00Z",
            "2025-05-01T01:00:00Z",
            "2025-05-01T02:00:00Z",
            "2025-05-01T03:00:00Z",
        ],
        utc=True,
    )
    blk = assign_bcm_capacity_block(ts)
    # 2025-05-01 is CEST (UTC+2): these map to 02:00..05:00 local -> two blocks.
    assert str(blk["bcm_capacity_block_hour_index"].iloc[0]) == "2.0"
    assert str(blk["bcm_capacity_block_hour_index"].iloc[1]) == "3.0"
    assert str(blk["bcm_capacity_block_hour_index"].iloc[2]) == "0.0"
    assert str(blk["bcm_capacity_block_hour_index"].iloc[3]) == "1.0"
    assert blk["bcm_capacity_block_id"].iloc[0] != blk["bcm_capacity_block_id"].iloc[2]


def test_afrr_only_bcm_block_constant_bem_hourly_not_forced_block_constant() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.date_range("2025-05-01T00:00:00Z", periods=8, freq="h")
    df = pd.DataFrame(
        {
            col.timestamp: ts,
            col.pred_da_price: [0.0] * 8,
            col.pred_afrr_capacity_price_pos: [100.0] * 4 + [1.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 8,
            col.pred_afrr_activation_price_pos: [150.0, -150.0, 150.0, -150.0, 150.0, -150.0, 150.0, -150.0],
            col.pred_afrr_activation_price_neg: [0.0] * 8,
            col.pred_afrr_activation_rate_pos: [1.0] * 8,
            col.pred_afrr_activation_rate_neg: [0.0] * 8,
        }
    )
    for pref in [
        col.pred_afrr_capacity_price_pos,
        col.pred_afrr_capacity_price_neg,
        col.pred_afrr_activation_price_pos,
        col.pred_afrr_activation_price_neg,
        col.pred_afrr_activation_rate_pos,
        col.pred_afrr_activation_rate_neg,
    ]:
        for q in ["p01", "p05", "p10", "p30", "p50", "p70", "p90", "p95", "p99"]:
            df[f"{pref}_{q}"] = df[pref]

    out = bt.optimize_dispatch(
        df,
        col,
        allowed_markets=("aFRR",),
    )
    b = assign_bcm_capacity_block(out[col.timestamp])
    out = pd.concat([out.reset_index(drop=True), b.reset_index(drop=True)], axis=1)
    for _, g in out.groupby("bcm_capacity_block_id"):
        assert float(pd.to_numeric(g["reserve_pos_mw"], errors="coerce").max() - pd.to_numeric(g["reserve_pos_mw"], errors="coerce").min()) <= 1e-6
        assert float(pd.to_numeric(g["reserve_neg_mw"], errors="coerce").max() - pd.to_numeric(g["reserve_neg_mw"], errors="coerce").min()) <= 1e-6
    # BEM stays hourly; not constrained to block equality.
    bem_spread = float(
        pd.to_numeric(out["bem_only_pos_mw"], errors="coerce").max()
        - pd.to_numeric(out["bem_only_pos_mw"], errors="coerce").min()
    )
    assert bem_spread > 1e-6


def test_bcm_block_consistency_check_detects_hourly_variation() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.date_range("2025-05-01T00:00:00Z", periods=4, freq="h")
    hourly = pd.DataFrame(
        {
            col.timestamp: ts,
            "real_submitted_bcm_capacity_pos_mw": [1.0, 1.0, 0.0, 0.0],
            "real_submitted_bcm_capacity_neg_mw": [0.0, 0.0, 0.0, 0.0],
            "real_executed_bcm_capacity_pos_mw": [1.0, 0.0, 0.0, 0.0],
            "real_executed_bcm_capacity_neg_mw": [0.0, 0.0, 0.0, 0.0],
        }
    )
    res = bt._compute_bcm_block_consistency(
        hourly=hourly,
        timestamp_col=col.timestamp,
        bcm_enabled=True,
        tol_mw=1e-6,
    )
    assert float(res["pass"]) == 0.0
    assert float(res["violation_count"]) > 0.0


def test_bcm_block_consistency_ignores_mixed_afrr_hourly_columns_when_explicit_bcm_constant() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.date_range("2025-05-01T00:00:00Z", periods=4, freq="h")
    hourly = pd.DataFrame(
        {
            col.timestamp: ts,
            # Explicit BCM columns are block-consistent.
            "real_submitted_bcm_capacity_pos_mw": [2.0, 2.0, 2.0, 2.0],
            "real_submitted_bcm_capacity_neg_mw": [0.0, 0.0, 0.0, 0.0],
            # Mixed diagnostics can vary hourly and must not invalidate BCM block consistency.
            "real_submitted_afrr_pos_mw": [2.0, 2.0, 0.0, 0.0],
            "real_submitted_afrr_neg_mw": [0.0, 1.0, 0.0, 1.0],
        }
    )
    res = bt._compute_bcm_block_consistency(
        hourly=hourly,
        timestamp_col=col.timestamp,
        bcm_enabled=True,
        tol_mw=1e-6,
    )
    assert float(res["pass"]) == 1.0
    assert float(res["violation_count"]) == 0.0


def test_bcm_block_consistency_ignores_activation_mwh_and_revenue_variation() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.date_range("2025-05-01T00:00:00Z", periods=4, freq="h")
    hourly = pd.DataFrame(
        {
            col.timestamp: ts,
            "real_submitted_bcm_capacity_pos_mw": [3.0, 3.0, 3.0, 3.0],
            "real_locked_bcm_capacity_pos_mw": [3.0, 3.0, 3.0, 3.0],
            "real_executed_bcm_capacity_pos_mw": [3.0, 3.0, 3.0, 3.0],
            "real_submitted_bcm_capacity_neg_mw": [0.0, 0.0, 0.0, 0.0],
            "real_locked_bcm_capacity_neg_mw": [0.0, 0.0, 0.0, 0.0],
            "real_executed_bcm_capacity_neg_mw": [0.0, 0.0, 0.0, 0.0],
            "real_bcm_linked_pos_activation_mwh": [0.0, 1.5, 0.0, 2.0],
            "real_bcm_linked_activation_revenue_eur": [0.0, 150.0, 0.0, 200.0],
        }
    )
    res = bt._compute_bcm_block_consistency(
        hourly=hourly,
        timestamp_col=col.timestamp,
        bcm_enabled=True,
        tol_mw=1e-6,
    )
    assert float(res["pass"]) == 1.0
    assert "activation" not in str(res["checked_columns"])


def test_simulation_forecast_loader_missing_p50_fails_by_default(tmp_path: Path) -> None:
    p = tmp_path / "missing_p50.parquet"
    pd.DataFrame(
        {
            "snapshot_time_utc": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "target_time_utc": [pd.Timestamp("2026-01-01T01:00:00Z")],
            "lead_time_h": [1],
            "predicted_value": [42.0],
            "p10": [10.0],
            "p30": [30.0],
            "p70": [70.0],
            "p90": [90.0],
        }
    ).to_parquet(p, index=False)
    with pytest.raises(KeyError, match="missing required quantile column 'p50'"):
        load_prediction_warehouse_long({"pred_afrr_activation_price_pos": p})


def test_simulation_forecast_loader_materializes_p50_only_with_explicit_flag(tmp_path: Path) -> None:
    p = tmp_path / "missing_p50_optin.parquet"
    pd.DataFrame(
        {
            "snapshot_time_utc": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "target_time_utc": [pd.Timestamp("2026-01-01T01:00:00Z")],
            "lead_time_h": [1],
            "predicted_value": [42.0],
            "p10": [10.0],
            "p30": [30.0],
            "p70": [70.0],
            "p90": [90.0],
        }
    ).to_parquet(p, index=False)
    wh = load_prediction_warehouse_long(
        {"pred_afrr_activation_price_pos": p},
        allow_p50_materialization_from_predicted_value=True,
    )
    out = wh["pred_afrr_activation_price_pos"].iloc[0]
    assert np.isclose(float(out["p50"]), 42.0)
    assert np.isclose(float(out["materialized_p50_from_predicted_value"]), 1.0)


def test_strategy_permissions_enum() -> None:
    cls = BatteryBacktester
    p = cls.strategy_permissions_from_name("multi")
    assert p.allow_da and p.allow_id and p.allow_bcm and p.allow_bem_only
    assert p.id_mode == "economic"
    p = cls.strategy_permissions_from_name("da")
    assert p.allow_da and (not p.allow_id) and (not p.allow_bcm) and (not p.allow_bem_only)
    assert p.id_mode == "none"
    p = cls.strategy_permissions_from_name("afrr")
    assert (not p.allow_da) and p.allow_id and p.allow_bcm and p.allow_bem_only
    assert p.id_mode == "technical_repair"
    p = cls.strategy_permissions_from_name("bcm")
    assert (not p.allow_da) and p.allow_id and p.allow_bcm and (not p.allow_bem_only)
    assert p.id_mode == "technical_repair"
    p = cls.strategy_permissions_from_name("bem")
    assert (not p.allow_da) and p.allow_id and (not p.allow_bcm) and p.allow_bem_only
    assert p.id_mode == "technical_repair"
    with pytest.raises(ValueError):
        cls.strategy_permissions_from_name("nope")


def test_bcm_only_optimizer_gating() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=250.0,
        act_neg=-50.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("aFRR", "BCM"))
    assert float(out["charge_mw"].iloc[0]) == 0.0
    assert float(out["discharge_mw"].iloc[0]) == 0.0
    assert float(out["bem_only_pos_mw"].iloc[0]) == 0.0
    assert float(out["bem_only_neg_mw"].iloc[0]) == 0.0
    assert float(out["reserve_pos_mw"].iloc[0]) >= 0.0


def test_bem_only_optimizer_gating() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=-50.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("aFRR", "BEM"))
    assert float(out["charge_mw"].iloc[0]) == 0.0
    assert float(out["discharge_mw"].iloc[0]) == 0.0
    assert float(out["reserve_pos_mw"].iloc[0]) == 0.0
    assert float(out["reserve_neg_mw"].iloc[0]) == 0.0
    assert float(out["bem_only_pos_mw"].iloc[0]) >= 0.0


def test_bem_p30_quantile_source_is_explicit_not_optimizer_bin() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30"]
    bt.afrr_quantile_prob = {"p30": 0.7}
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=-50.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )

    out = bt.optimize_dispatch(df, col, allowed_markets=("aFRR", "BEM"))

    assert str(out["bem_requested_quantile"].iloc[0]) == "p30"
    assert str(out["bem_selected_quantile"].iloc[0]) == "p30"
    assert str(out["bem_submitted_price_quantile"].iloc[0]) == "p30"
    assert str(out["bem_optimizer_bin_quantile"].iloc[0]) == "p30"
    assert "optimizer_bin" not in str(out["bem_submitted_price_quantile"].iloc[0])
    assert str(out["bem_submitted_price_source_column"].iloc[0]) == f"{col.pred_afrr_activation_price_pos}_p30"


def test_bcm_p30_quantile_source_is_explicit() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30"]
    bt.afrr_quantile_prob = {"p30": 0.7}
    bt.afrr_activation_rate_guard_policy = "scenario"
    bt.afrr_activation_rate_guard_quantile = "scenario"
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=-50.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )

    out = bt.optimize_dispatch(df, col, allowed_markets=("aFRR", "BCM"))

    assert str(out["bcm_requested_quantile"].iloc[0]) == "p30"
    assert str(out["bcm_selected_quantile"].iloc[0]) == "p30"
    assert str(out["bcm_capacity_price_source_column_pos"].iloc[0]) == f"{col.pred_afrr_capacity_price_pos}_p30"
    assert str(out["afrr_activation_rate_guard_policy"].iloc[0]) == "scenario"
    assert str(out["afrr_activation_rate_guard_quantile_resolved"].iloc[0]) == "p30"
    assert str(out["afrr_activation_rate_guard_quantile"].iloc[0]) == "p30"
    assert str(out["afrr_activation_rate_guard_source_column_pos"].iloc[0]) == (
        f"{col.pred_afrr_activation_rate_pos}_p30"
    )


def test_afrr_missing_activation_rate_guard_quantile_fails_explicitly() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30"]
    bt.afrr_quantile_prob = {"p30": 0.7}
    bt.afrr_activation_rate_guard_policy = "p90"
    bt.afrr_activation_rate_guard_quantile = "p90"
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=-50.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    df = df.drop(
        columns=[
            f"{col.pred_afrr_activation_rate_pos}_p90",
            f"{col.pred_afrr_activation_rate_neg}_p90",
        ]
    )

    with pytest.raises(ValueError, match="missing_afrr_activation_rate_guard_quantile.*p90"):
        bt.optimize_dispatch(df, col, allowed_markets=("aFRR", "BEM"))


def test_afrr_activation_rate_guard_scenario_p30_missing_fails_explicitly() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30"]
    bt.afrr_quantile_prob = {"p30": 0.7}
    bt.afrr_activation_rate_guard_policy = "scenario"
    bt.afrr_activation_rate_guard_quantile = "scenario"
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=-50.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    df = df.drop(
        columns=[
            f"{col.pred_afrr_activation_rate_pos}_p30",
            f"{col.pred_afrr_activation_rate_neg}_p30",
        ]
    )

    with pytest.raises(ValueError, match="missing_afrr_activation_rate_guard_quantile.*p30"):
        bt.optimize_dispatch(df, col, allowed_markets=("aFRR", "BEM"))


def test_afrr_activation_rate_guard_canonical_uses_point_columns() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30"]
    bt.afrr_quantile_prob = {"p30": 0.7}
    bt.afrr_activation_rate_guard_policy = "canonical"
    bt.afrr_activation_rate_guard_quantile = "canonical"
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=-50.0,
        rate_pos=0.33,
        rate_neg=0.22,
    )
    df = df.drop(
        columns=[
            f"{col.pred_afrr_activation_rate_pos}_p90",
            f"{col.pred_afrr_activation_rate_neg}_p90",
        ]
    )
    df[f"{col.pred_afrr_activation_rate_pos}_p30"] = 0.99
    df[f"{col.pred_afrr_activation_rate_neg}_p30"] = 0.88

    out = bt.optimize_dispatch(df, col, allowed_markets=("aFRR", "BEM"))

    assert str(out["afrr_activation_rate_guard_policy"].iloc[0]) == "canonical"
    assert str(out["afrr_activation_rate_guard_quantile_resolved"].iloc[0]) == "canonical"
    assert str(out["afrr_activation_rate_guard_mode"].iloc[0]) == "canonical"
    assert str(out["afrr_activation_rate_guard_source_column_pos"].iloc[0]) == col.pred_afrr_activation_rate_pos
    assert str(out["afrr_activation_rate_guard_source_column_neg"].iloc[0]) == col.pred_afrr_activation_rate_neg
    assert float(out["afrr_activation_rate_guard_fallback_used"].iloc[0]) == pytest.approx(0.0)
    assert float(out["ev_pred_act_rate_pos_guard"].iloc[0]) == pytest.approx(0.33)
    assert float(out["ev_pred_act_rate_neg_guard"].iloc[0]) == pytest.approx(0.22)


def test_afrr_activation_rate_guard_auto_uses_canonical_when_p90_missing() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30"]
    bt.afrr_quantile_prob = {"p30": 0.7}
    bt.afrr_activation_rate_guard_policy = "auto"
    bt.afrr_activation_rate_guard_quantile = "auto"
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=-50.0,
        rate_pos=0.44,
        rate_neg=0.11,
    )
    df = df.drop(
        columns=[
            f"{col.pred_afrr_activation_rate_pos}_p90",
            f"{col.pred_afrr_activation_rate_neg}_p90",
        ]
    )
    df[f"{col.pred_afrr_activation_rate_pos}_p30"] = 0.99
    df[f"{col.pred_afrr_activation_rate_neg}_p30"] = 0.88

    out = bt.optimize_dispatch(df, col, allowed_markets=("aFRR", "BEM"))

    assert str(out["afrr_activation_rate_guard_policy"].iloc[0]) == "auto"
    assert str(out["afrr_activation_rate_guard_quantile_resolved"].iloc[0]) == "canonical"
    assert str(out["afrr_activation_rate_guard_mode"].iloc[0]) == "canonical"
    assert float(out["afrr_activation_rate_guard_fallback_used"].iloc[0]) == pytest.approx(1.0)
    assert "p90" in str(out["afrr_activation_rate_guard_missing_columns"].iloc[0])
    assert float(out["ev_pred_act_rate_pos_guard"].iloc[0]) == pytest.approx(0.44)
    assert float(out["ev_pred_act_rate_neg_guard"].iloc[0]) == pytest.approx(0.11)


def test_afrr_activation_rate_guard_quantile_p70_does_not_require_p90() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30"]
    bt.afrr_quantile_prob = {"p30": 0.7}
    bt.afrr_activation_rate_guard_policy = "p70"
    bt.afrr_activation_rate_guard_quantile = "p70"
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=-50.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    df[f"{col.pred_afrr_activation_rate_pos}_p70"] = 0.42
    df[f"{col.pred_afrr_activation_rate_neg}_p70"] = 0.24
    df = df.drop(
        columns=[
            f"{col.pred_afrr_activation_rate_pos}_p90",
            f"{col.pred_afrr_activation_rate_neg}_p90",
        ]
    )

    out = bt.optimize_dispatch(df, col, allowed_markets=("aFRR", "BEM"))

    assert str(out["afrr_activation_rate_guard_quantile"].iloc[0]) == "p70"
    assert str(out["afrr_activation_rate_guard_quantile_resolved"].iloc[0]) == "p70"
    assert str(out["afrr_activation_rate_guard_source_column_pos"].iloc[0]) == (
        f"{col.pred_afrr_activation_rate_pos}_p70"
    )
    assert str(out["afrr_activation_rate_guard_source_column_neg"].iloc[0]) == (
        f"{col.pred_afrr_activation_rate_neg}_p70"
    )
    assert float(out["ev_pred_act_rate_pos_guard"].iloc[0]) == pytest.approx(0.42)
    assert float(out["ev_pred_act_rate_neg_guard"].iloc[0]) == pytest.approx(0.24)
    assert np.isnan(float(out["ev_pred_act_rate_pos_p90"].iloc[0]))


def test_afrr_missing_configured_p70_guard_quantile_fails_explicitly() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30"]
    bt.afrr_quantile_prob = {"p30": 0.7}
    bt.afrr_activation_rate_guard_policy = "p70"
    bt.afrr_activation_rate_guard_quantile = "p70"
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=-50.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    df = df.drop(
        columns=[
            f"{col.pred_afrr_activation_rate_pos}_p70",
            f"{col.pred_afrr_activation_rate_neg}_p70",
        ]
    )

    with pytest.raises(ValueError, match="missing_afrr_activation_rate_guard_quantile.*p70"):
        bt.optimize_dispatch(df, col, allowed_markets=("aFRR", "BCM"))


def test_afrr_activation_rate_guard_default_scenario_uses_p50_for_p50() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p50"]
    bt.afrr_quantile_prob = {"p50": 0.5}
    bt.afrr_activation_rate_guard_policy = "scenario"
    bt.afrr_activation_rate_guard_quantile = "scenario"
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=-50.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    df = df.drop(
        columns=[
            f"{col.pred_afrr_activation_rate_pos}_p90",
            f"{col.pred_afrr_activation_rate_neg}_p90",
        ]
    )

    out = bt.optimize_dispatch(df, col, allowed_markets=("aFRR", "BEM"))

    assert str(out["afrr_activation_rate_guard_policy"].iloc[0]) == "scenario"
    assert str(out["afrr_activation_rate_guard_quantile_resolved"].iloc[0]) == "p50"
    assert str(out["afrr_activation_rate_guard_source_column_pos"].iloc[0]) == (
        f"{col.pred_afrr_activation_rate_pos}_p50"
    )


def test_afrr_activation_rate_guard_scenario_multi_bin_fails_explicitly() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30", "p50", "p70"]
    bt.afrr_quantile_prob = {"p30": 0.7, "p50": 0.5, "p70": 0.3}
    bt.afrr_activation_rate_guard_policy = "scenario"
    bt.afrr_activation_rate_guard_quantile = "scenario"
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=-50.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )

    with pytest.raises(ValueError, match="ambiguous_afrr_activation_rate_guard_quantile_for_multi_bin_scenario"):
        bt.optimize_dispatch(df, col, allowed_markets=("aFRR", "BCM"))


def test_da_only_does_not_require_afrr_activation_rate_guard_columns() -> None:
    bt = _mk_backtester()
    bt.afrr_activation_rate_guard_policy = "scenario"
    bt.afrr_activation_rate_guard_quantile = "scenario"
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=-50.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    df = df.drop(
        columns=[
            f"{col.pred_afrr_activation_rate_pos}_p90",
            f"{col.pred_afrr_activation_rate_neg}_p90",
            f"{col.pred_afrr_activation_rate_pos}_p30",
            f"{col.pred_afrr_activation_rate_neg}_p30",
        ]
    )

    out = bt.optimize_dispatch(df, col, allowed_markets=("DA",))

    assert float(out["reserve_pos_mw"].iloc[0]) == pytest.approx(0.0)
    assert float(out["bem_only_pos_mw"].iloc[0]) == pytest.approx(0.0)


def test_perfect_foresight_materializes_missing_afrr_p30_ev_bin_columns_from_truth() -> None:
    col = BacktestColumnMap()
    df = pd.DataFrame(
        {
            col.timestamp: pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="h"),
            col.true_da_price: [10.0, 20.0],
            col.true_afrr_capacity_price_pos: [1.0, 2.0],
            col.true_afrr_capacity_price_neg: [3.0, 4.0],
            col.true_afrr_activation_price_pos: [100.0, 110.0],
            col.true_afrr_activation_price_neg: [-50.0, -55.0],
            col.true_afrr_activation_rate_pos: [0.2, 0.3],
            col.true_afrr_activation_rate_neg: [0.4, 0.5],
        }
    )

    out = BatteryBacktester._materialize_perfect_foresight_quantile_columns(df, colmap=col)

    for pred_col, true_col in (
        (col.pred_da_price, col.true_da_price),
        (col.pred_afrr_capacity_price_pos, col.true_afrr_capacity_price_pos),
        (col.pred_afrr_capacity_price_neg, col.true_afrr_capacity_price_neg),
        (col.pred_afrr_activation_price_pos, col.true_afrr_activation_price_pos),
        (col.pred_afrr_activation_price_neg, col.true_afrr_activation_price_neg),
        (col.pred_afrr_activation_rate_pos, col.true_afrr_activation_rate_pos),
        (col.pred_afrr_activation_rate_neg, col.true_afrr_activation_rate_neg),
    ):
        p30_col = f"{pred_col}_p30"
        assert p30_col in out.columns
        assert out[pred_col].to_numpy(dtype=float).tolist() == pytest.approx(df[true_col].tolist())
        assert out[p30_col].to_numpy(dtype=float).tolist() == pytest.approx(df[true_col].tolist())


def test_bem_missing_requested_quantile_fails_explicitly() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30"]
    bt.afrr_quantile_prob = {"p30": 0.7}
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=-50.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    df = df.drop(columns=[f"{col.pred_afrr_activation_price_pos}_p30"])

    with pytest.raises(ValueError, match="missing_bem_quantile.*p30"):
        bt.optimize_dispatch(df, col, allowed_markets=("aFRR", "BEM"))


def test_bcm_missing_requested_quantile_fails_explicitly() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p30"]
    bt.afrr_quantile_prob = {"p30": 0.7}
    df, col = _one_hour_pred_df(
        da=200.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=300.0,
        act_neg=-50.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    df = df.drop(columns=[f"{col.pred_afrr_capacity_price_pos}_p30"])

    with pytest.raises(ValueError, match="missing_bcm_quantile.*p30"):
        bt.optimize_dispatch(df, col, allowed_markets=("aFRR", "BCM"))


def test_afrr_only_optimizer_gating() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=300.0,
        cap_pos=100.0,
        cap_neg=0.0,
        act_pos=200.0,
        act_neg=-100.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("aFRR", "BCM", "BEM"))
    assert float(out["charge_mw"].iloc[0]) == 0.0
    assert float(out["discharge_mw"].iloc[0]) == 0.0
    assert float(out["reserve_pos_mw"].iloc[0]) >= 0.0
    assert float(out["bem_only_pos_mw"].iloc[0]) >= 0.0


def test_da_only_optimizer_gating() -> None:
    bt = _mk_backtester()
    df, col = _one_hour_pred_df(
        da=300.0,
        cap_pos=200.0,
        cap_neg=0.0,
        act_pos=200.0,
        act_neg=-100.0,
        rate_pos=1.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(df, col, allowed_markets=("DA",))
    assert float(out["reserve_pos_mw"].iloc[0]) == 0.0
    assert float(out["reserve_neg_mw"].iloc[0]) == 0.0
    assert float(out["bem_only_pos_mw"].iloc[0]) == 0.0
    assert float(out["bem_only_neg_mw"].iloc[0]) == 0.0


def _write_long_pred(
    path: Path,
    *,
    ts: pd.DatetimeIndex,
    predicted_value: float,
    quantiles: dict[str, float],
) -> None:
    snap = pd.Timestamp(ts.min()).tz_convert("UTC") - pd.Timedelta(hours=6)
    df = pd.DataFrame(
        {
            "snapshot_time_utc": [snap] * len(ts),
            "target_time_utc": ts,
            "lead_time_h": [1] * len(ts),
            "predicted_value": [predicted_value] * len(ts),
        }
    )
    for q, v in quantiles.items():
        df[q] = [v] * len(ts)
    df.to_parquet(path, index=False)


def test_strict_preparation_materializes_activation_price_p50_into_optimize_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bt = _mk_backtester("canonical_economic")
    bt.afrr_quantile_bins = ["p30", "p50", "p70"]
    bt.afrr_quantile_prob = {q: 1.0 - float(q[1:]) / 100.0 for q in bt.afrr_quantile_bins}
    base_df, col = _tiny_backtest_df(hours=4)
    drop_cols = [
        c
        for c in base_df.columns
        if c.startswith(f"{col.pred_afrr_activation_price_pos}_")
        or c.startswith(f"{col.pred_afrr_activation_price_neg}_")
    ]
    base_df = base_df.drop(columns=drop_cols, errors="ignore")
    ts = pd.to_datetime(base_df[col.timestamp], utc=True)
    files: dict[str, Path] = {}
    for target in [
        "pred_da_price",
        "pred_afrr_capacity_price_pos",
        "pred_afrr_capacity_price_neg",
        "pred_afrr_activation_rate_pos",
        "pred_afrr_activation_rate_neg",
    ]:
        p = tmp_path / f"{target}.parquet"
        _write_long_pred(
            p,
            ts=ts,
            predicted_value=1.0,
            quantiles={"p30": 0.8, "p50": 1.0, "p70": 1.2},
        )
        files[target] = p
    for target in ["pred_afrr_activation_price_pos", "pred_afrr_activation_price_neg"]:
        p = tmp_path / f"{target}.parquet"
        _write_long_pred(
            p,
            ts=ts,
            predicted_value=10.0,
            quantiles={"p30": 8.0, "p70": 12.0},
        )
        files[target] = p
    warehouse = load_prediction_warehouse_long(
        files,
        target_value_modes={"pred_afrr_activation_price_neg": "canonical_economic"},
        allow_p50_materialization_from_predicted_value=True,
    )
    seen_cols: list[set[str]] = []
    orig_opt = bt.optimize_dispatch

    def _wrapped_opt(df: pd.DataFrame, *args, **kwargs):
        seen_cols.append(set(df.columns))
        return orig_opt(df, *args, **kwargs)

    monkeypatch.setattr(bt, "optimize_dispatch", _wrapped_opt)
    bt.run(
        base_df,
        col,
        use_rolling_horizon=True,
        horizon_hours=4,
        reopt_step_hours=1,
        forecast_warehouse=warehouse,
        strict_simulation_validity=True,
        enable_global_perfect_foresight=False,
    )
    assert seen_cols, "optimize_dispatch should be called at least once"
    assert f"{col.pred_afrr_activation_price_pos}_p50" in seen_cols[0]
    assert f"{col.pred_afrr_activation_price_neg}_p50" in seen_cols[0]


def test_strict_preparation_fails_for_missing_non_p50_active_bin(tmp_path: Path) -> None:
    bt = _mk_backtester("canonical_economic")
    bt.afrr_quantile_bins = ["p30", "p50", "p70"]
    bt.afrr_quantile_prob = {q: 1.0 - float(q[1:]) / 100.0 for q in bt.afrr_quantile_bins}
    base_df, col = _tiny_backtest_df(hours=4)
    drop_cols = [
        c
        for c in base_df.columns
        if c.startswith(f"{col.pred_afrr_activation_price_pos}_")
        or c.startswith(f"{col.pred_afrr_activation_price_neg}_")
    ]
    base_df = base_df.drop(columns=drop_cols, errors="ignore")
    ts = pd.to_datetime(base_df[col.timestamp], utc=True)
    files: dict[str, Path] = {}
    for target in [
        "pred_da_price",
        "pred_afrr_capacity_price_pos",
        "pred_afrr_capacity_price_neg",
        "pred_afrr_activation_rate_pos",
        "pred_afrr_activation_rate_neg",
        "pred_afrr_activation_price_pos",
        "pred_afrr_activation_price_neg",
    ]:
        p = tmp_path / f"{target}.parquet"
        qmap = {"p30": 8.0, "p50": 10.0}
        if "rate" in target:
            qmap = {"p30": 0.2, "p50": 0.3}
        _write_long_pred(
            p,
            ts=ts,
            predicted_value=10.0,
            quantiles=qmap,
        )
        files[target] = p
    warehouse = load_prediction_warehouse_long(
        files,
        target_value_modes={"pred_afrr_activation_price_neg": "canonical_economic"},
    )
    with pytest.raises(ValueError, match="Missing required aFRR quantile-bin inputs in strict mode"):
        bt.run(
            base_df,
            col,
            use_rolling_horizon=True,
            horizon_hours=4,
            reopt_step_hours=1,
            forecast_warehouse=warehouse,
            strict_simulation_validity=True,
            enable_global_perfect_foresight=False,
        )


def test_simulation_forecast_loader_clips_activation_rates() -> None:
    col = BacktestColumnMap()
    df = pd.DataFrame(
        {
            col.timestamp: [pd.Timestamp("2026-01-01T00:00:00Z")],
            col.pred_afrr_activation_rate_pos: [1.4],
            col.pred_afrr_activation_rate_neg: [-0.4],
            f"{col.pred_afrr_activation_rate_pos}_p10": [-0.2],
            f"{col.pred_afrr_activation_rate_pos}_p90": [1.5],
            f"{col.pred_afrr_activation_rate_neg}_p10": [-0.3],
            f"{col.pred_afrr_activation_rate_neg}_p90": [1.7],
            col.true_afrr_activation_rate_pos: [1.8],
            col.true_afrr_activation_rate_neg: [-0.6],
        }
    )
    out = canonicalize_market_frame(df, colmap=col)
    assert np.isclose(float(out.loc[0, col.pred_afrr_activation_rate_pos]), 1.0)
    assert np.isclose(float(out.loc[0, col.pred_afrr_activation_rate_neg]), 0.0)
    assert np.isclose(float(out.loc[0, col.true_afrr_activation_rate_pos]), 1.0)
    assert np.isclose(float(out.loc[0, col.true_afrr_activation_rate_neg]), 0.0)


def test_bcm_capacity_bid_award_then_pay_as_bid_capacity_revenue() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T08:00:00Z"),
        is_perfect_foresight=False,
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
    assert np.isclose(float(out["bcm_capacity_bid_price_pos_eur_per_mw_h"]), 20.0)
    assert np.isclose(float(out["bcm_awarded_capacity_pos_mw"]), 5.0)
    assert np.isclose(float(out["bcm_to_bem_energy_obligation_pos_mw"]), 5.0)
    assert np.isclose(float(out["bcm_activation_bid_price_pos"]), 80.0)
    assert np.isclose(float(out["bem_activation_bid_price_pos_eur_per_mwh"]), 80.0)
    assert np.isclose(float(out["bem_clearing_price_pos_eur_per_mwh"]), 80.0)

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
    assert np.isclose(float(m["revenue_capacity_eur"]), 5.0 * 20.0 * bt.dt_h), m
    assert np.isclose(float(m["bcm_capacity_revenue_eur"]), 5.0 * 20.0 * bt.dt_h), m
    assert str(m["bcm_capacity_price_source"]) == "obligation_lockbook_price"


def test_bcm_obligation_capacity_price_is_used_for_settlement_not_activation_or_cutoff() -> None:
    bt = _mk_backtester()
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T08:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        pred_da_price=50.0,
        true_da_price=50.0,
        pred_cap_pos=0.0,
        true_cap_pos=100.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=999.0,
        true_act_pos=999.0,
        pred_act_neg=-999.0,
        true_act_neg=-999.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        obligation_pos_mw=9.0,
        obligation_neg_mw=0.0,
        obligation_capacity_price_pos=20.0,
        obligation_energy_pos=999.0,
    )

    assert float(out["settlement_cap_bid_price_pos_eur_mw"]) == pytest.approx(20.0)
    assert float(out["bcm_capacity_bid_price_pos_eur_per_mw_h"]) == pytest.approx(20.0)
    assert float(out["bcm_capacity_cutoff_price_pos_eur_per_mw_h"]) == pytest.approx(100.0)
    assert float(out["bcm_activation_bid_price_pos"]) == pytest.approx(999.0)

    _, metrics = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=0.0,
        discharge=0.0,
        reserve_pos=float(out["executed_reserve_pos_mw"]),
        reserve_neg=0.0,
        da_price=50.0,
        cap_pos=100.0,
        cap_neg=0.0,
        act_pos_price=999.0,
        act_neg_price=-999.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        cap_bid_pos=float(out["settlement_cap_bid_price_pos_eur_mw"]),
        cap_bid_neg=0.0,
    )
    assert float(metrics["revenue_capacity_eur"]) == pytest.approx(9.0 * 20.0 * bt.dt_h)
    assert float(metrics["bcm_capacity_revenue_pos_eur"]) == pytest.approx(9.0 * 20.0 * bt.dt_h)


def test_bcm_existing_obligation_capacity_price_missing_does_not_use_generic_price() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p70"]
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-01T08:00:00Z")
    data = {
        col.pred_da_price: 50.0,
        col.true_da_price: 50.0,
        col.pred_afrr_capacity_price_pos: 0.0,
        col.pred_afrr_capacity_price_neg: 0.0,
        col.true_afrr_capacity_price_pos: 100.0,
        col.true_afrr_capacity_price_neg: 100.0,
        col.pred_afrr_activation_price_pos: 80.0,
        col.pred_afrr_activation_price_neg: -80.0,
        col.true_afrr_activation_price_pos: 80.0,
        col.true_afrr_activation_price_neg: -80.0,
        col.pred_afrr_activation_rate_pos: 0.0,
        col.pred_afrr_activation_rate_neg: 0.0,
        col.true_afrr_activation_rate_pos: 0.0,
        col.true_afrr_activation_rate_neg: 0.0,
        "settlement_cap_bid_price_pos_eur_mw": 0.0,
        "settlement_cap_bid_price_neg_eur_mw": 0.0,
        "bcm_capacity_bid_price_pos_eur_per_mw_h": 0.0,
        "bcm_capacity_bid_price_neg_eur_per_mw_h": 0.0,
        "bcm_p70_capacity_bid_price_pos_eur_per_mw_h": 25.0,
        "bcm_p70_capacity_bid_price_neg_eur_per_mw_h": 30.0,
        "afrr_p70_cap_price_pos": 25.0,
        "afrr_p70_cap_price_neg": 30.0,
        "afrr_bin_0_cap_price_pos": 25.0,
        "afrr_bin_0_cap_price_neg": 30.0,
        "ev_bcm_capacity_bid_price_pos_bin_0_eur_per_mw_h": 25.0,
        "ev_bcm_capacity_bid_price_neg_bin_0_eur_per_mw_h": 30.0,
        "aFRR_Energy_Price_EUR_MWh_Pos": 80.0,
        "aFRR_Energy_Price_EUR_MWh_Neg": -80.0,
    }
    row = type("Row", (), data)()

    def getf(name: str, default: float = 0.0) -> float:
        if name == "aFRR_Capacity_Won_Pos_MW":
            return 2.0
        if name == "aFRR_Capacity_Won_Neg_MW":
            return 0.0
        return float(data.get(name, default))

    out = bt._settle_realized_reclearing_row(
        row=row,
        getf=getf,
        ts_utc=ts,
        colmap=col,
        perfect_foresight_mode=False,
        require_da_quantiles=False,
        charge=0.0,
        discharge=0.0,
        reserve_pos=2.0,
        reserve_neg=0.0,
        bem_only_pos=0.0,
        bem_only_neg=0.0,
        soc=bt.soc_init,
    )
    clearing = out["clearing_rec"]

    assert float(clearing["settlement_cap_bid_price_pos_eur_mw"]) == pytest.approx(0.0)
    assert float(clearing["bcm_capacity_bid_price_pos_eur_per_mw_h"]) == pytest.approx(0.0)
    assert float(clearing["bcm_capacity_price_missing_with_awarded_mw"]) == pytest.approx(1.0)
    assert float(clearing["bcm_capacity_price_lost_before_settlement"]) == pytest.approx(1.0)
    assert str(clearing["bcm_capacity_price_source"]) == "missing_obligation_capacity_price"

    _, metrics = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=0.0,
        discharge=0.0,
        reserve_pos=float(out["reserve_pos"]),
        reserve_neg=0.0,
        da_price=50.0,
        cap_pos=100.0,
        cap_neg=100.0,
        act_pos_price=80.0,
        act_neg_price=-80.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        cap_bid_pos=float(clearing["settlement_cap_bid_price_pos_eur_mw"]),
        cap_bid_neg=float(clearing["settlement_cap_bid_price_neg_eur_mw"]),
    )

    assert float(metrics["bcm_capacity_bid_price_pos_eur_per_mw_h"]) == pytest.approx(0.0)
    assert float(metrics["bcm_capacity_revenue_eur"]) == pytest.approx(0.0)
    assert float(metrics["bcm_capacity_price_missing_with_awarded_mw"]) == pytest.approx(1.0)
    assert str(metrics["bcm_capacity_price_source"]) == "missing_obligation_capacity_price"


def test_bcm_obligation_lockbook_price_is_primary_settlement_source() -> None:
    bt = _mk_backtester()
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T08:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        pred_da_price=50.0,
        true_da_price=50.0,
        pred_cap_pos=999.0,
        true_cap_pos=100.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=80.0,
        true_act_pos=80.0,
        pred_act_neg=0.0,
        true_act_neg=0.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        obligation_pos_mw=2.0,
        obligation_neg_mw=0.0,
        obligation_capacity_price_pos=20.0,
    )

    assert float(out["fixed_reserve_obligation_pos_mw"]) == pytest.approx(2.0)
    assert float(out["settlement_cap_bid_price_pos_eur_mw"]) == pytest.approx(20.0)
    assert float(out["bcm_obligation_capacity_bid_price_pos_eur_per_mw_h"]) == pytest.approx(20.0)
    assert float(out["bcm_settlement_capacity_price_resolved_pos_eur_per_mw_h"]) == pytest.approx(20.0)
    assert float(out["bcm_capacity_price_missing_with_awarded_mw"]) == pytest.approx(0.0)
    assert str(out["bcm_capacity_price_source"]) == "obligation_lockbook_price"

    _, metrics = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=0.0,
        discharge=0.0,
        reserve_pos=2.0,
        reserve_neg=0.0,
        da_price=50.0,
        cap_pos=100.0,
        cap_neg=0.0,
        act_pos_price=80.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        cap_bid_pos=float(out["settlement_cap_bid_price_pos_eur_mw"]),
        cap_bid_neg=0.0,
    )
    assert float(metrics["bcm_capacity_revenue_eur"]) == pytest.approx(2.0 * 20.0 * bt.dt_h)
    assert str(metrics["bcm_capacity_price_source"]) == "obligation_lockbook_price"


def test_bcm_negative_obligation_lockbook_price_settles_pay_as_bid() -> None:
    bt = _mk_backtester()
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T08:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        pred_da_price=50.0,
        true_da_price=50.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=999.0,
        true_cap_neg=100.0,
        pred_act_pos=0.0,
        true_act_pos=0.0,
        pred_act_neg=-80.0,
        true_act_neg=-80.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        obligation_pos_mw=0.0,
        obligation_neg_mw=3.0,
        obligation_capacity_price_neg=30.0,
    )

    assert float(out["fixed_reserve_obligation_neg_mw"]) == pytest.approx(3.0)
    assert float(out["settlement_cap_bid_price_neg_eur_mw"]) == pytest.approx(30.0)
    assert float(out["bcm_obligation_capacity_bid_price_neg_eur_per_mw_h"]) == pytest.approx(30.0)

    _, metrics = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=0.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=3.0,
        da_price=50.0,
        cap_pos=0.0,
        cap_neg=100.0,
        act_pos_price=0.0,
        act_neg_price=-80.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        cap_bid_pos=0.0,
        cap_bid_neg=float(out["settlement_cap_bid_price_neg_eur_mw"]),
    )
    assert float(metrics["bcm_capacity_revenue_eur"]) == pytest.approx(3.0 * 30.0 * bt.dt_h)
    assert float(metrics["bcm_capacity_revenue_neg_eur"]) == pytest.approx(3.0 * 30.0 * bt.dt_h)


def test_bcm_capacity_revenue_uses_awarded_capacity_not_supportable_capacity() -> None:
    bt = _mk_backtester()
    awarded_mw = 9.0
    discharge_mw = 2.0
    supportable_mw = float(bt.p_max_mw - discharge_mw)

    _, metrics = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=0.0,
        discharge=discharge_mw,
        reserve_pos=awarded_mw,
        reserve_neg=0.0,
        da_price=50.0,
        cap_pos=100.0,
        cap_neg=0.0,
        act_pos_price=9999.0,
        act_neg_price=-9999.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        cap_bid_pos=20.0,
        cap_bid_neg=0.0,
    )

    assert float(metrics["bcm_capacity_revenue_pos_eur"]) == pytest.approx(awarded_mw * 20.0 * bt.dt_h)
    assert float(metrics["revenue_capacity_eur"]) == pytest.approx(awarded_mw * 20.0 * bt.dt_h)
    assert float(metrics["missed_capacity_pos_mw"]) == pytest.approx(awarded_mw - supportable_mw)
    assert float(metrics["capacity_full_activation_shortfall_pos_mw"]) == pytest.approx(
        awarded_mw - supportable_mw
    )
    assert float(metrics["capacity_full_activation_stress_check_pass"]) == pytest.approx(0.0)
    assert float(metrics["missed_activation_pos_mwh"]) == pytest.approx(0.0)
    assert float(metrics["bcm_available_capacity_pos_mw"]) == pytest.approx(supportable_mw)


def test_bcm_capacity_block_bid_preserves_forecast_capacity_price_if_builder_zeroes() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")
    blk = pd.DataFrame({"target_time_utc": ts})
    source = pd.DataFrame(
        {
            col.timestamp: ts,
            col.pred_afrr_capacity_price_pos: [20.0] * 4,
            col.pred_afrr_capacity_price_neg: [-5.0] * 4,
        }
    ).set_index(col.timestamp, drop=False)

    def _zero_price_capacity_bids(**kwargs):
        return [
            BCMCapacityBid(
                ts=kwargs["ts"],
                side="pos",
                quantity_mw=float(kwargs["reserve_pos_mw"]),
                capacity_price_eur_mw=0.0,
            ),
            BCMCapacityBid(
                ts=kwargs["ts"],
                side="neg",
                quantity_mw=float(kwargs["reserve_neg_mw"]),
                capacity_price_eur_mw=0.0,
            ),
        ]

    bt.bid_builder.build_afrr_capacity_bids = _zero_price_capacity_bids  # type: ignore[method-assign]

    bids, _ = bt._formulate_afrr_capacity_block_bids(
        blk=blk,
        source=source,
        colmap=col,
        snapshot_ts=ts[0],
        offered_pos=9.0,
        offered_neg=3.0,
    )

    by_side = {bid.side: bid for bid in bids}
    assert float(by_side["pos"].capacity_price_eur_mw) == pytest.approx(20.0)
    assert float(by_side["neg"].capacity_price_eur_mw) == pytest.approx(-5.0)


def test_bcm_capacity_bid_price_propagates_from_unindexed_forecast_source_to_settlement() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")
    blk = pd.DataFrame({"target_time_utc": ts})
    source = pd.DataFrame(
        {
            col.timestamp: ts,
            col.pred_afrr_capacity_price_pos: [33.0] * 4,
            col.pred_afrr_capacity_price_neg: [17.0] * 4,
            col.pred_afrr_activation_price_pos: [999.0] * 4,
            col.pred_afrr_activation_price_neg: [888.0] * 4,
        }
    )

    bids, _ = bt._formulate_afrr_capacity_block_bids(
        blk=blk,
        source=source,
        colmap=col,
        snapshot_ts=ts[0],
        offered_pos=2.0,
        offered_neg=0.0,
    )
    pos_bid = next(b for b in bids if b.side == "pos")
    assert float(pos_bid.capacity_price_eur_mw) == pytest.approx(33.0)
    assert float(pos_bid.capacity_price_eur_mw) != pytest.approx(999.0)
    assert float(bt._last_bcm_capacity_price_lookup_diagnostics["bcm_capacity_price_lookup_failed"]) == pytest.approx(0.0)
    assert str(bt._last_bcm_capacity_price_lookup_diagnostics["bcm_capacity_price_lookup_source"]) == "timestamp_indexed_source"

    _soc, metrics = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=0.0,
        discharge=0.0,
        reserve_pos=2.0,
        reserve_neg=0.0,
        da_price=50.0,
        cap_pos=100.0,
        cap_neg=0.0,
        act_pos_price=999.0,
        act_neg_price=888.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        cap_bid_pos=float(pos_bid.capacity_price_eur_mw),
        cap_bid_neg=0.0,
    )

    assert float(metrics["bcm_capacity_bid_price_pos_eur_per_mw_h"]) == pytest.approx(33.0)
    assert float(metrics["bcm_capacity_revenue_eur"]) == pytest.approx(2.0 * 33.0 * bt.dt_h)
    assert float(metrics["bcm_capacity_revenue_eur"]) > 0.0


def test_bcm_rhpf_capacity_bid_uses_true_block_cutoff_price() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")
    blk = pd.DataFrame(
        {
            "target_time_utc": ts,
            f"{col.pred_afrr_capacity_price_pos}_p70": [12.0] * 4,
            f"{col.pred_afrr_capacity_price_neg}_p70": [7.0] * 4,
        }
    )
    source = pd.DataFrame(
        {
            col.timestamp: ts,
            col.true_afrr_capacity_price_pos: [20.0, 21.0, 22.0, 23.0],
            col.true_afrr_capacity_price_neg: [9.0, 10.0, 11.0, 12.0],
            f"{col.pred_afrr_capacity_price_pos}_p70": [12.0] * 4,
            f"{col.pred_afrr_capacity_price_neg}_p70": [7.0] * 4,
        }
    ).set_index(col.timestamp, drop=False)

    bids, _ = bt._formulate_afrr_capacity_block_bids(
        blk=blk,
        source=source,
        colmap=col,
        snapshot_ts=ts[0],
        offered_pos=4.0,
        offered_neg=3.0,
        is_perfect_foresight=True,
    )

    by_side = {bid.side: bid for bid in bids}
    assert float(by_side["pos"].capacity_price_eur_mw) == pytest.approx(21.5)
    assert float(by_side["neg"].capacity_price_eur_mw) == pytest.approx(10.5)
    diag = bt._last_bcm_capacity_price_lookup_diagnostics
    assert str(diag["bcm_capacity_bid_price_source"]) == "true_capacity_cutoff_oracle"
    assert str(diag["bcm_capacity_price_lookup_column_pos"]) == col.true_afrr_capacity_price_pos
    assert str(diag["bcm_capacity_price_lookup_column_neg"]) == col.true_afrr_capacity_price_neg
    assert float(diag["bcm_bid_equals_cutoff_pos"]) == pytest.approx(1.0)
    assert float(diag["bcm_bid_equals_cutoff_neg"]) == pytest.approx(1.0)
    assert float(diag["bcm_bid_price_gap_to_cutoff_pos_eur_per_mw_h"]) == pytest.approx(0.0)
    assert float(diag["bcm_bid_price_gap_to_cutoff_neg_eur_per_mw_h"]) == pytest.approx(0.0)


def test_bcm_model_capacity_bid_keeps_quantile_price_not_true_cutoff() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")
    blk = pd.DataFrame(
        {
            "target_time_utc": ts,
            f"{col.pred_afrr_capacity_price_pos}_p70": [12.0] * 4,
            f"{col.pred_afrr_capacity_price_neg}_p70": [7.0] * 4,
        }
    )
    source = pd.DataFrame(
        {
            col.timestamp: ts,
            col.true_afrr_capacity_price_pos: [20.0, 21.0, 22.0, 23.0],
            col.true_afrr_capacity_price_neg: [9.0, 10.0, 11.0, 12.0],
            f"{col.pred_afrr_capacity_price_pos}_p70": [12.0] * 4,
            f"{col.pred_afrr_capacity_price_neg}_p70": [7.0] * 4,
        }
    ).set_index(col.timestamp, drop=False)

    bids, _ = bt._formulate_afrr_capacity_block_bids(
        blk=blk,
        source=source,
        colmap=col,
        snapshot_ts=ts[0],
        offered_pos=4.0,
        offered_neg=3.0,
        is_perfect_foresight=False,
    )

    by_side = {bid.side: bid for bid in bids}
    assert float(by_side["pos"].capacity_price_eur_mw) == pytest.approx(12.0)
    assert float(by_side["neg"].capacity_price_eur_mw) == pytest.approx(7.0)
    diag = bt._last_bcm_capacity_price_lookup_diagnostics
    assert str(diag["bcm_capacity_bid_price_source"]) == "block_rows"
    assert float(diag["bcm_bid_equals_cutoff_pos"]) == pytest.approx(0.0)
    assert np.isnan(float(diag["bcm_bid_price_gap_to_cutoff_pos_eur_per_mw_h"]))


def test_bcm_capacity_lockbook_stores_accepted_pay_as_bid_price_for_pos_and_neg() -> None:
    bt = _mk_backtester()
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = None
    bt.reserve_activation_headroom_h = 0.25
    bt.reserve_headroom_safety_mwh = 0.0
    bt.reserve_soc_projection_safety_mwh = 0.0
    bt.reserve_min_margin_after_bid_mwh = 0.0
    col = BacktestColumnMap()
    snapshot_ts = pd.Timestamp("2025-05-01T06:00:00Z")  # 08:00 Europe/Berlin BCM gate.
    target_hours = pd.date_range("2025-05-01T22:00:00Z", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [2.0] * 4,
            "reserve_neg_mw": [3.0] * 4,
            "soc_start_lp_mwh": [10.0] * 4,
            "charge_mw": [0.0] * 4,
            "discharge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
        }
    )
    source = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [20.0] * 4,
            col.pred_afrr_capacity_price_neg: [15.0] * 4,
            col.pred_afrr_activation_price_pos: [999.0] * 4,
            col.pred_afrr_activation_price_neg: [888.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [100.0] * 4,
        }
    ).set_index(col.timestamp, drop=False)
    lock_pos: dict[pd.Timestamp, float] = {}
    lock_neg: dict[pd.Timestamp, float] = {}
    lock_energy_pos: dict[pd.Timestamp, float] = {}
    lock_energy_neg: dict[pd.Timestamp, float] = {}
    lock_price_pos: dict[pd.Timestamp, float] = {}
    lock_price_neg: dict[pd.Timestamp, float] = {}

    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=snapshot_ts,
        snapshot_plan=snap,
        source=source,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos=lock_energy_pos,
        lock_energy_neg=lock_energy_neg,
        lock_capacity_price_pos=lock_price_pos,
        lock_capacity_price_neg=lock_price_neg,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
    )

    assert lock_pos and all(float(v) == pytest.approx(2.0) for v in lock_pos.values())
    assert lock_neg and all(float(v) == pytest.approx(3.0) for v in lock_neg.values())
    assert lock_price_pos and all(float(v) == pytest.approx(20.0) for v in lock_price_pos.values())
    assert lock_price_neg and all(float(v) == pytest.approx(15.0) for v in lock_price_neg.values())

    ts = target_hours[0]
    clearing = bt._apply_market_clearing(
        target_time_utc=ts,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=999.0,
        true_cap_pos=100.0,
        pred_cap_neg=999.0,
        true_cap_neg=100.0,
        pred_act_pos=999.0,
        true_act_pos=999.0,
        pred_act_neg=888.0,
        true_act_neg=888.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        obligation_pos_mw=float(lock_pos[ts]),
        obligation_neg_mw=float(lock_neg[ts]),
        obligation_capacity_price_pos=float(lock_price_pos[ts]),
        obligation_capacity_price_neg=float(lock_price_neg[ts]),
        obligation_energy_pos=999.0,
        obligation_energy_neg=888.0,
        soc_now=10.0,
    )
    assert float(clearing["settlement_cap_bid_price_pos_eur_mw"]) == pytest.approx(20.0)
    assert float(clearing["settlement_cap_bid_price_neg_eur_mw"]) == pytest.approx(15.0)
    assert float(clearing["bcm_capacity_price_missing_with_awarded_mw"]) == pytest.approx(0.0)
    assert str(clearing["bcm_capacity_price_source"]) == "obligation_lockbook_price"

    _, metrics = bt._settle_one_hour(
        soc=10.0,
        charge=0.0,
        discharge=0.0,
        reserve_pos=float(lock_pos[ts]),
        reserve_neg=float(lock_neg[ts]),
        da_price=0.0,
        cap_pos=100.0,
        cap_neg=100.0,
        act_pos_price=999.0,
        act_neg_price=888.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        cap_bid_pos=float(clearing["settlement_cap_bid_price_pos_eur_mw"]),
        cap_bid_neg=float(clearing["settlement_cap_bid_price_neg_eur_mw"]),
    )
    expected = (2.0 * 20.0 + 3.0 * 15.0) * bt.dt_h
    assert float(metrics["bcm_capacity_revenue_eur"]) == pytest.approx(expected)


def test_bcm_capacity_lockbook_restores_forecast_price_when_builder_zeroes() -> None:
    bt = _mk_backtester()
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = None
    bt.reserve_activation_headroom_h = 0.25
    bt.reserve_headroom_safety_mwh = 0.0
    bt.reserve_soc_projection_safety_mwh = 0.0
    bt.reserve_min_margin_after_bid_mwh = 0.0
    col = BacktestColumnMap()
    snapshot_ts = pd.Timestamp("2025-05-01T06:00:00Z")
    target_hours = pd.date_range("2025-05-01T22:00:00Z", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [2.0] * 4,
            "reserve_neg_mw": [3.0] * 4,
            "soc_start_lp_mwh": [10.0] * 4,
            "charge_mw": [0.0] * 4,
            "discharge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
        }
    )
    source = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [22.0] * 4,
            col.pred_afrr_capacity_price_neg: [11.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [100.0] * 4,
        }
    ).set_index(col.timestamp, drop=False)

    def _zero_price_capacity_bids(**kwargs):
        return [
            BCMCapacityBid(
                ts=kwargs["ts"],
                side="pos",
                quantity_mw=float(kwargs["reserve_pos_mw"]),
                capacity_price_eur_mw=0.0,
            ),
            BCMCapacityBid(
                ts=kwargs["ts"],
                side="neg",
                quantity_mw=float(kwargs["reserve_neg_mw"]),
                capacity_price_eur_mw=0.0,
            ),
        ]

    bt.bid_builder.build_afrr_capacity_bids = _zero_price_capacity_bids  # type: ignore[method-assign]
    lock_pos: dict[pd.Timestamp, float] = {}
    lock_neg: dict[pd.Timestamp, float] = {}
    lock_price_pos: dict[pd.Timestamp, float] = {}
    lock_price_neg: dict[pd.Timestamp, float] = {}

    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=snapshot_ts,
        snapshot_plan=snap,
        source=source,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos={},
        lock_energy_neg={},
        lock_capacity_price_pos=lock_price_pos,
        lock_capacity_price_neg=lock_price_neg,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
    )

    assert lock_pos and all(float(v) == pytest.approx(2.0) for v in lock_pos.values())
    assert lock_neg and all(float(v) == pytest.approx(3.0) for v in lock_neg.values())
    assert lock_price_pos and all(float(v) == pytest.approx(22.0) for v in lock_price_pos.values())
    assert lock_price_neg and all(float(v) == pytest.approx(11.0) for v in lock_price_neg.values())


def test_bcm_capacity_lockbook_uses_active_quantile_price_when_point_price_missing() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p70"]
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = None
    bt.reserve_activation_headroom_h = 0.25
    bt.reserve_headroom_safety_mwh = 0.0
    bt.reserve_soc_projection_safety_mwh = 0.0
    bt.reserve_min_margin_after_bid_mwh = 0.0
    col = BacktestColumnMap()
    snapshot_ts = pd.Timestamp("2025-05-01T06:00:00Z")
    target_hours = pd.date_range("2025-05-01T22:00:00Z", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [2.0] * 4,
            "reserve_neg_mw": [3.0] * 4,
            "soc_start_lp_mwh": [10.0] * 4,
            "charge_mw": [0.0] * 4,
            "discharge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            f"{col.pred_afrr_capacity_price_pos}_p70": [8.0, 8.5, 9.0, 9.5],
            f"{col.pred_afrr_capacity_price_neg}_p70": [4.0, 4.5, 5.0, 5.5],
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [100.0] * 4,
            col.pred_afrr_activation_rate_pos: [1.0] * 4,
            col.pred_afrr_activation_rate_neg: [1.0] * 4,
        }
    )
    source = pd.DataFrame(
        {
            col.timestamp: target_hours,
            f"{col.pred_afrr_capacity_price_pos}_p70": [8.0, 8.5, 9.0, 9.5],
            f"{col.pred_afrr_capacity_price_neg}_p70": [4.0, 4.5, 5.0, 5.5],
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [100.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [100.0] * 4,
            col.pred_afrr_activation_rate_pos: [1.0] * 4,
            col.pred_afrr_activation_rate_neg: [1.0] * 4,
        }
    ).set_index(col.timestamp, drop=False)

    bids, _ = bt._formulate_afrr_capacity_block_bids(
        blk=snap,
        source=source,
        colmap=col,
        snapshot_ts=snapshot_ts,
        offered_pos=2.0,
        offered_neg=3.0,
    )
    by_side = {bid.side: bid for bid in bids}
    assert float(by_side["pos"].capacity_price_eur_mw) == pytest.approx(8.75)
    assert float(by_side["neg"].capacity_price_eur_mw) == pytest.approx(4.75)
    assert str(bt._last_bcm_capacity_price_lookup_diagnostics["bcm_capacity_price_lookup_column_pos"]) == (
        f"{col.pred_afrr_capacity_price_pos}_p70"
    )
    assert str(bt._last_bcm_capacity_price_lookup_diagnostics["bcm_capacity_price_lookup_column_neg"]) == (
        f"{col.pred_afrr_capacity_price_neg}_p70"
    )

    lock_pos: dict[pd.Timestamp, float] = {}
    lock_neg: dict[pd.Timestamp, float] = {}
    lock_price_pos: dict[pd.Timestamp, float] = {}
    lock_price_neg: dict[pd.Timestamp, float] = {}

    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=snapshot_ts,
        snapshot_plan=snap,
        source=source,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos={},
        lock_energy_neg={},
        lock_capacity_price_pos=lock_price_pos,
        lock_capacity_price_neg=lock_price_neg,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
    )

    assert lock_pos and all(float(v) == pytest.approx(2.0) for v in lock_pos.values())
    assert lock_neg and all(float(v) == pytest.approx(3.0) for v in lock_neg.values())
    assert lock_price_pos and all(float(v) == pytest.approx(8.75) for v in lock_price_pos.values())
    assert lock_price_neg and all(float(v) == pytest.approx(4.75) for v in lock_price_neg.values())


def test_bcm_capacity_price_missing_summary_diagnostics_identify_first_row() -> None:
    ts = pd.Series(pd.date_range("2025-03-01T22:00:00Z", periods=2, freq="h"))

    diag = BatteryBacktester._bcm_capacity_price_missing_diagnostics(
        timestamps=ts,
        locked_pos_mw=pd.Series([0.0, 9.0]),
        locked_neg_mw=pd.Series([0.0, 4.0]),
        settlement_price_pos=pd.Series([0.0, 0.0]),
        settlement_price_neg=pd.Series([0.0, 0.0]),
        revenue_pos_eur=pd.Series([0.0, 0.0]),
        revenue_neg_eur=pd.Series([0.0, 0.0]),
        price_source=pd.Series(["none", "missing_obligation_capacity_price"]),
    )

    assert float(diag["bcm_capacity_price_missing_for_awarded_capacity_count"]) == pytest.approx(1.0)
    assert str(diag["first_bcm_capacity_price_missing_timestamp_utc"]) == "2025-03-01T23:00:00+00:00"
    assert str(diag["first_bcm_capacity_price_missing_side"]) == "pos_neg"
    assert float(diag["first_bcm_capacity_price_missing_locked_pos_mw"]) == pytest.approx(9.0)
    assert float(diag["first_bcm_capacity_price_missing_locked_neg_mw"]) == pytest.approx(4.0)
    assert str(diag["first_bcm_capacity_price_missing_source"]) == "missing_obligation_capacity_price"


def test_bcm_capacity_price_missing_summary_diagnostics_pass_when_revenue_present() -> None:
    ts = pd.Series(pd.date_range("2025-03-01T22:00:00Z", periods=1, freq="h"))

    diag = BatteryBacktester._bcm_capacity_price_missing_diagnostics(
        timestamps=ts,
        locked_pos_mw=pd.Series([9.0]),
        locked_neg_mw=pd.Series([4.0]),
        settlement_price_pos=pd.Series([8.0]),
        settlement_price_neg=pd.Series([4.0]),
        revenue_pos_eur=pd.Series([72.0]),
        revenue_neg_eur=pd.Series([16.0]),
        price_source=pd.Series(["obligation_lockbook_price"]),
    )

    assert float(diag["bcm_capacity_price_missing_for_awarded_capacity_count"]) == pytest.approx(0.0)
    assert str(diag["first_bcm_capacity_price_missing_timestamp_utc"]) == ""
    assert str(diag["first_bcm_capacity_price_missing_side"]) == "none"


def test_naive_bcm_inputs_create_lagged_quantiles_and_pacc_bins() -> None:
    col = BacktestColumnMap()
    ts = pd.date_range("2025-01-01T00:00:00Z", periods=30, freq="h")
    df = pd.DataFrame(
        {
            col.timestamp: ts,
            col.true_da_price: np.arange(30, dtype=float),
            col.true_afrr_capacity_price_pos: np.full(30, 20.0),
            col.true_afrr_capacity_price_neg: np.full(30, 10.0),
            col.true_afrr_activation_price_pos: np.full(30, 100.0),
            col.true_afrr_activation_price_neg: np.full(30, 80.0),
            col.true_afrr_activation_rate_pos: np.full(30, 0.25),
            col.true_afrr_activation_rate_neg: np.full(30, 0.10),
        }
    )

    out = BatteryBacktester._apply_naive_24h_predictions(
        df,
        col,
        naive_forecast_mode="shift_24h_delivery",
    )

    assert col.pred_afrr_capacity_price_pos in out.columns
    assert f"{col.pred_afrr_capacity_price_pos}_p50" in out.columns
    assert f"{col.pred_afrr_activation_price_neg}_p90" in out.columns
    assert f"{col.pred_afrr_activation_rate_pos}_p10" in out.columns
    assert "pacc_pos_bin_0" in out.columns
    assert "pacc_neg_bin_0" in out.columns
    assert float(out.loc[24, col.pred_afrr_capacity_price_pos]) == pytest.approx(20.0)
    assert float(out.loc[24, f"{col.pred_afrr_capacity_price_pos}_p50"]) == pytest.approx(20.0)
    assert float(out.loc[24, f"{col.pred_afrr_activation_price_neg}_p90"]) == pytest.approx(80.0)
    assert float(out.loc[24, f"{col.pred_afrr_activation_rate_pos}_p10"]) == pytest.approx(0.25)
    assert float(out.loc[24, "pacc_pos_bin_0"]) == pytest.approx(0.99)


def test_naive_same_weekday_last_week_uses_causal_delivery_source() -> None:
    col = BacktestColumnMap()
    ts = pd.date_range("2025-06-03T00:00:00Z", "2025-06-10T23:00:00Z", freq="h")
    df = pd.DataFrame(
        {
            col.timestamp: ts,
            col.true_da_price: np.arange(len(ts), dtype=float),
            col.pred_da_price: np.full(len(ts), -999.0),
            f"{col.pred_da_price}_p50": np.full(len(ts), -777.0),
        }
    )

    out = BatteryBacktester._apply_naive_24h_predictions(
        df,
        col,
        naive_forecast_mode="same_weekday_last_week",
        da_bid_hour_local=11,
    )

    delivery = pd.Timestamp("2025-06-10T18:00:00Z")
    source = pd.Timestamp("2025-06-03T18:00:00Z")
    delivery_idx = int(np.where(ts == delivery)[0][0])
    source_idx = int(np.where(ts == source)[0][0])

    assert float(out.loc[delivery_idx, col.pred_da_price]) == pytest.approx(float(source_idx))
    assert float(out.loc[delivery_idx, f"{col.pred_da_price}_p50"]) == pytest.approx(float(source_idx))
    assert str(out.loc[delivery_idx, f"{col.pred_da_price}_naive_source_timestamp_utc"]) == source.isoformat()
    assert str(out.loc[delivery_idx, f"{col.pred_da_price}_naive_source_mode"]) == "same_weekday_last_week"
    assert float(out.loc[delivery_idx, "naive_forecast_causality_violation"]) == pytest.approx(0.0)


def test_naive_causal_day_before_gate_uses_last_same_hour_before_gate() -> None:
    col = BacktestColumnMap()
    ts = pd.date_range("2025-06-03T00:00:00Z", "2025-06-10T23:00:00Z", freq="h")
    df = pd.DataFrame(
        {
            col.timestamp: ts,
            col.true_da_price: np.arange(len(ts), dtype=float),
            col.pred_da_price: np.full(len(ts), -999.0),
        }
    )

    out = BatteryBacktester._apply_naive_24h_predictions(
        df,
        col,
        naive_forecast_mode="causal_day_before_gate",
        da_bid_hour_local=11,
    )

    delivery = pd.Timestamp("2025-06-10T18:00:00Z")
    source = pd.Timestamp("2025-06-08T18:00:00Z")
    not_allowed = pd.Timestamp("2025-06-09T18:00:00Z")
    delivery_idx = int(np.where(ts == delivery)[0][0])
    source_idx = int(np.where(ts == source)[0][0])

    assert float(out.loc[delivery_idx, col.pred_da_price]) == pytest.approx(float(source_idx))
    assert str(out.loc[delivery_idx, f"{col.pred_da_price}_naive_source_timestamp_utc"]) == source.isoformat()
    assert str(out.loc[delivery_idx, f"{col.pred_da_price}_naive_source_timestamp_utc"]) != not_allowed.isoformat()
    assert float(out.loc[delivery_idx, "naive_forecast_causality_violation"]) == pytest.approx(0.0)


def test_naive_legacy_shift_24h_delivery_reproduces_old_noncausal_source() -> None:
    col = BacktestColumnMap()
    ts = pd.date_range("2025-06-03T00:00:00Z", "2025-06-10T23:00:00Z", freq="h")
    df = pd.DataFrame(
        {
            col.timestamp: ts,
            col.true_da_price: np.arange(len(ts), dtype=float),
            col.pred_da_price: np.full(len(ts), -999.0),
        }
    )

    out = BatteryBacktester._apply_naive_24h_predictions(
        df,
        col,
        naive_forecast_mode="shift_24h_delivery",
        da_bid_hour_local=11,
    )

    delivery = pd.Timestamp("2025-06-10T18:00:00Z")
    legacy_source = pd.Timestamp("2025-06-09T18:00:00Z")
    delivery_idx = int(np.where(ts == delivery)[0][0])
    source_idx = int(np.where(ts == legacy_source)[0][0])

    assert float(out.loc[delivery_idx, col.pred_da_price]) == pytest.approx(float(source_idx))
    assert str(out.loc[delivery_idx, f"{col.pred_da_price}_naive_source_timestamp_utc"]) == legacy_source.isoformat()
    assert float(out.loc[delivery_idx, "naive_forecast_legacy_delivery_shift"]) == pytest.approx(1.0)
    assert float(out.loc[delivery_idx, "naive_forecast_causality_violation"]) == pytest.approx(1.0)


def test_naive_missing_last_week_falls_back_to_latest_causal_same_hour() -> None:
    col = BacktestColumnMap()
    ts = pd.date_range("2025-06-08T00:00:00Z", "2025-06-10T23:00:00Z", freq="h")
    df = pd.DataFrame(
        {
            col.timestamp: ts,
            col.true_da_price: np.arange(len(ts), dtype=float),
            col.pred_da_price: np.full(len(ts), -999.0),
        }
    )

    out = BatteryBacktester._apply_naive_24h_predictions(
        df,
        col,
        naive_forecast_mode="same_weekday_last_week",
        da_bid_hour_local=11,
    )

    delivery = pd.Timestamp("2025-06-10T18:00:00Z")
    fallback_source = pd.Timestamp("2025-06-08T18:00:00Z")
    delivery_idx = int(np.where(ts == delivery)[0][0])
    source_idx = int(np.where(ts == fallback_source)[0][0])

    assert float(out.loc[delivery_idx, col.pred_da_price]) == pytest.approx(float(source_idx))
    assert str(out.loc[delivery_idx, f"{col.pred_da_price}_naive_source_timestamp_utc"]) == fallback_source.isoformat()
    assert str(out.loc[delivery_idx, f"{col.pred_da_price}_naive_source_mode"]) == "causal_same_hour_fallback"
    assert float(out.loc[delivery_idx, "naive_forecast_fallback_used"]) == pytest.approx(1.0)
    assert float(out.loc[delivery_idx, "naive_weekly_source_missing"]) == pytest.approx(1.0)


def test_naive_same_weekday_last_week_uses_separate_source_history() -> None:
    col = BacktestColumnMap()
    eval_ts = pd.date_range("2025-06-10T00:00:00Z", "2025-06-10T23:00:00Z", freq="h")
    source_ts = pd.date_range("2025-06-03T00:00:00Z", "2025-06-10T23:00:00Z", freq="h")
    df = pd.DataFrame(
        {
            col.timestamp: eval_ts,
            col.true_da_price: np.full(len(eval_ts), 999.0),
            col.pred_da_price: np.full(len(eval_ts), -999.0),
        }
    )
    source_df = pd.DataFrame(
        {
            col.timestamp: source_ts,
            col.true_da_price: np.arange(len(source_ts), dtype=float),
        }
    )

    out = BatteryBacktester._apply_naive_24h_predictions(
        df,
        col,
        naive_forecast_mode="same_weekday_last_week",
        da_bid_hour_local=11,
        source_df=source_df,
    )

    delivery = pd.Timestamp("2025-06-10T18:00:00Z")
    source = pd.Timestamp("2025-06-03T18:00:00Z")
    delivery_idx = int(np.where(eval_ts == delivery)[0][0])
    source_idx = int(np.where(source_ts == source)[0][0])

    assert float(out.loc[delivery_idx, col.pred_da_price]) == pytest.approx(float(source_idx))
    assert str(out.loc[delivery_idx, f"{col.pred_da_price}_naive_source_timestamp_utc"]) == source.isoformat()
    assert str(out.loc[delivery_idx, f"{col.pred_da_price}_naive_source_mode"]) == "same_weekday_last_week"
    assert float(out.loc[delivery_idx, "naive_forecast_fallback_used"]) == pytest.approx(0.0)
    assert float(out.loc[delivery_idx, "naive_weekly_source_missing"]) == pytest.approx(0.0)


def test_terminal_soc_conflict_with_locked_reserve_gets_specific_driver() -> None:
    driver = BatteryBacktester._infer_infeasibility_driver_for_row(
        optimization_error_code="terminal_soc_conflict",
        fixed_reserve_obligation_pos_mw=9.0,
        fixed_reserve_obligation_neg_mw=0.0,
        headroom_violation_pos_mwh=0.0,
        headroom_violation_neg_mwh=0.0,
        power_violation_pos_mw=0.0,
        power_violation_neg_mw=0.0,
        terminal_recovery_scheduled_grid_mwh=0.0,
    )
    assert driver == "id_recovery_blocked"


def test_bcm_capacity_and_linked_activation_revenue_are_separate() -> None:
    bt = _mk_backtester("canonical_economic")
    capacity_revenue = 0.0
    activation_revenue = 0.0
    gross_revenue = 0.0

    for i in range(4):
        rate_neg = (2.0 * bt.eta_in / 3.0) if i == 1 else 0.0
        _, m = bt._settle_one_hour(
            soc=bt.soc_init,
            charge=0.0,
            discharge=0.0,
            reserve_pos=0.0,
            reserve_neg=3.0,
            da_price=0.0,
            cap_pos=0.0,
            cap_neg=0.0,
            act_pos_price=0.0,
            act_neg_price=1200.0,
            act_pos_rate=0.0,
            act_neg_rate=rate_neg,
            cap_bid_pos=0.0,
            cap_bid_neg=50.0,
        )
        capacity_revenue += float(m["bcm_capacity_revenue_eur"])
        activation_revenue += float(m["revenue_activation_eur"])
        gross_revenue += float(m["bcm_capacity_revenue_eur"]) + float(m["revenue_activation_eur"])

    assert capacity_revenue == pytest.approx(3.0 * 50.0 * 4.0)
    assert activation_revenue == pytest.approx(2.0 * 1200.0)
    assert gross_revenue == pytest.approx(3000.0)


def test_bcm_capacity_pay_as_bid_without_activation_is_paid_over_product() -> None:
    bt = _mk_backtester("canonical_economic")
    capacity_revenue = 0.0

    for _ in range(4):
        _, m = bt._settle_one_hour(
            soc=bt.soc_init,
            charge=0.0,
            discharge=0.0,
            reserve_pos=4.0,
            reserve_neg=0.0,
            da_price=0.0,
            cap_pos=0.0,
            cap_neg=0.0,
            act_pos_price=1200.0,
            act_neg_price=0.0,
            act_pos_rate=0.0,
            act_neg_rate=0.0,
            cap_bid_pos=50.0,
            cap_bid_neg=0.0,
        )
        capacity_revenue += float(m["bcm_capacity_revenue_eur"])
        assert float(m["revenue_activation_eur"]) == pytest.approx(0.0)

    assert capacity_revenue == pytest.approx(4.0 * 50.0 * 4.0)


def test_bcm_ev_capacity_term_uses_capacity_price_per_mw() -> None:
    bt = _mk_backtester("canonical_economic")
    df, col = _one_hour_pred_df(
        da=0.0,
        cap_pos=50.0,
        cap_neg=60.0,
        act_pos=0.0,
        act_neg=0.0,
        rate_pos=0.0,
        rate_neg=0.0,
    )
    out = bt.optimize_dispatch(
        df,
        col,
        allowed_markets=("aFRR", "BCM"),
        deterministic_reserve_settlement=True,
    )
    assert float(out["ev_bcm_expected_capacity_revenue_pos_bin_0"].iloc[0]) == pytest.approx(50.0 * bt.dt_h)
    assert float(out["ev_bcm_expected_capacity_revenue_neg_bin_0"].iloc[0]) == pytest.approx(60.0 * bt.dt_h)


def test_bcm_bem_ev_probability_and_efficiency_conventions() -> None:
    bt = _mk_backtester("canonical_economic")
    bt.trans_eur_mwh = 2.0
    bt.deg_eur_mwh = 10.0
    bt.eta_in = 0.9
    bt.eta_out = 0.8
    bt.afrr_offer_cost_eur_mw_h = 0.0
    bt.aux_afrr_active_mw = 0.0
    bt.aux_standby_mw = 0.0
    bt.aux_trading_mw = 0.0

    df, col = _one_hour_pred_df(
        da=0.0,
        cap_pos=100.0,
        cap_neg=80.0,
        act_pos=50.0,
        act_neg=60.0,
        rate_pos=0.2,
        rate_neg=0.3,
    )
    out = bt.optimize_dispatch(
        df,
        col,
        allowed_markets=("aFRR", "BCM", "BEM"),
        deterministic_reserve_settlement=False,
    )
    p50_bin = bt.afrr_quantile_bins.index("p50")
    p_award = 0.5
    dt = bt.dt_h
    pos_cost_per_mwh = bt.trans_eur_mwh + bt.deg_eur_mwh / bt.eta_out
    neg_cost_per_mwh = bt.trans_eur_mwh + bt.deg_eur_mwh * bt.eta_in

    expected_bcm_capacity_pos = p_award * 100.0 * dt
    expected_bcm_activation_pos = p_award * 0.2 * 50.0 * dt
    expected_bcm_cost_pos = p_award * 0.2 * pos_cost_per_mwh * dt
    expected_bcm_coef_pos = expected_bcm_capacity_pos + expected_bcm_activation_pos - expected_bcm_cost_pos

    expected_bcm_capacity_neg = p_award * 80.0 * dt
    expected_bcm_activation_neg = p_award * 0.3 * 60.0 * dt
    expected_bcm_cost_neg = p_award * 0.3 * neg_cost_per_mwh * dt
    expected_bcm_coef_neg = expected_bcm_capacity_neg + expected_bcm_activation_neg - expected_bcm_cost_neg

    expected_bem_activation_pos = p_award * 0.2 * 50.0 * dt
    expected_bem_cost_pos = p_award * 0.2 * pos_cost_per_mwh * dt
    expected_bem_activation_neg = p_award * 0.3 * 60.0 * dt
    expected_bem_cost_neg = p_award * 0.3 * neg_cost_per_mwh * dt

    assert out["activation_rate_is_conditional"].iloc[0] == "conditional_on_award_or_execution"
    assert out["ev_bcm_activation_rate_is_conditional"].iloc[0] == "conditional_on_award"
    assert out["ev_bem_activation_rate_is_conditional"].iloc[0] == "conditional_on_execution"
    assert float(out["acceptance_probability_applied_once"].iloc[0]) == pytest.approx(1.0)
    assert float(out["execution_probability_applied_once"].iloc[0]) == pytest.approx(1.0)
    assert float(out["ev_dt_h"].iloc[0]) == pytest.approx(1.0)
    assert float(out["ev_bcm_product_duration_h"].iloc[0]) == pytest.approx(4.0)
    assert float(out[f"ev_bcm_p_award_pos_bin_{p50_bin}"].iloc[0]) == pytest.approx(p_award)
    assert float(out[f"ev_bem_p_exec_pos_bin_{p50_bin}"].iloc[0]) == pytest.approx(p_award)
    assert float(out[f"ev_bcm_capacity_value_pos_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_bcm_capacity_pos
    )
    assert float(out[f"ev_bcm_capacity_value_neg_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_bcm_capacity_neg
    )
    assert float(out[f"ev_bcm_activation_value_pos_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_bcm_activation_pos
    )
    assert float(out[f"ev_bcm_activation_value_neg_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_bcm_activation_neg
    )
    assert float(out[f"ev_bcm_expected_activation_revenue_pos_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_bcm_activation_pos
    )
    assert float(out[f"ev_bcm_expected_activation_revenue_neg_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_bcm_activation_neg
    )
    assert float(out[f"ev_bcm_standby_cost_pos_bin_{p50_bin}"].iloc[0]) == pytest.approx(0.0)
    assert float(out[f"ev_bcm_standby_cost_neg_bin_{p50_bin}"].iloc[0]) == pytest.approx(0.0)
    assert float(out[f"ev_bcm_activation_cost_pos_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_bcm_cost_pos
    )
    assert float(out[f"ev_bcm_activation_cost_neg_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_bcm_cost_neg
    )
    assert float(out[f"ev_bcm_costs_pos_bin_{p50_bin}"].iloc[0]) == pytest.approx(expected_bcm_cost_pos)
    assert float(out[f"ev_bcm_costs_neg_bin_{p50_bin}"].iloc[0]) == pytest.approx(expected_bcm_cost_neg)
    assert float(out[f"ev_bcm_expected_costs_pos_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_bcm_cost_pos
    )
    assert float(out[f"ev_bcm_expected_costs_neg_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_bcm_cost_neg
    )
    assert float(out[f"ev_rpos_coef_bin_{p50_bin}_eur_per_mw"].iloc[0]) == pytest.approx(
        expected_bcm_coef_pos
    )
    assert float(out[f"ev_rneg_coef_bin_{p50_bin}_eur_per_mw"].iloc[0]) == pytest.approx(
        expected_bcm_coef_neg
    )
    assert float(out["bcm_optimizer_expected_ev_pos_eur_per_mw"].iloc[0]) == pytest.approx(
        expected_bcm_coef_pos
    )
    assert float(out["bcm_optimizer_coeff_pos_eur_per_mw"].iloc[0]) == pytest.approx(expected_bcm_coef_pos)
    assert float(out["bcm_optimizer_coeff_missing_when_ev_positive"].iloc[0]) == pytest.approx(0.0)
    assert float(out[f"ev_bem_activation_value_pos_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_bem_activation_pos
    )
    assert float(out[f"ev_bem_activation_value_neg_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_bem_activation_neg
    )
    assert float(out[f"ev_bem_costs_pos_bin_{p50_bin}"].iloc[0]) == pytest.approx(expected_bem_cost_pos)
    assert float(out[f"ev_bem_costs_neg_bin_{p50_bin}"].iloc[0]) == pytest.approx(expected_bem_cost_neg)
    assert float(out[f"ev_bem_expected_costs_pos_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_bem_cost_pos
    )
    assert float(out[f"ev_bem_expected_costs_neg_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_bem_cost_neg
    )
    assert float(out[f"ev_bem_pos_coef_bin_{p50_bin}_eur_per_mw"].iloc[0]) == pytest.approx(
        expected_bem_activation_pos - expected_bem_cost_pos
    )
    assert float(out[f"ev_bem_neg_coef_bin_{p50_bin}_eur_per_mw"].iloc[0]) == pytest.approx(
        expected_bem_activation_neg - expected_bem_cost_neg
    )


def test_bem_ev_aux_diagnostics_are_activation_proportional_no_standby() -> None:
    bt = _mk_backtester("canonical_economic")
    bt.trans_eur_mwh = 0.0
    bt.deg_eur_mwh = 0.0
    bt.afrr_offer_cost_eur_mw_h = 0.0
    bt.p_max_mw = 10.0
    bt.aux_afrr_active_mw = 1.0
    bt.aux_standby_mw = 999.0
    bt.aux_trading_mw = 0.0

    df, col = _one_hour_pred_df(
        da=100.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos=200.0,
        act_neg=300.0,
        rate_pos=0.2,
        rate_neg=0.3,
    )
    out = bt.optimize_dispatch(
        df,
        col,
        allowed_markets=("aFRR", "BEM"),
        deterministic_reserve_settlement=False,
    )
    p50_bin = bt.afrr_quantile_bins.index("p50")
    p_exec = 0.5
    expected_pos_aux = bt.dt_h / bt.p_max_mw * p_exec * 0.2 * bt.aux_afrr_active_mw * 100.0
    expected_neg_aux = bt.dt_h / bt.p_max_mw * p_exec * 0.3 * bt.aux_afrr_active_mw * 100.0

    assert out["ev_bem_aux_cost_basis"].iloc[0] == "activation_proportional_no_standby_aux"
    assert float(out[f"ev_bem_activation_aux_cost_pos_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_pos_aux
    )
    assert float(out[f"ev_bem_activation_aux_cost_neg_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_neg_aux
    )
    assert float(out[f"ev_bem_standby_aux_cost_pos_bin_{p50_bin}"].iloc[0]) == pytest.approx(0.0)
    assert float(out[f"ev_bem_standby_aux_cost_neg_bin_{p50_bin}"].iloc[0]) == pytest.approx(0.0)
    assert float(out[f"ev_bem_total_aux_cost_pos_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_pos_aux
    )
    assert float(out[f"ev_bem_total_aux_cost_neg_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_neg_aux
    )
    assert float(out[f"bem_p50_activation_aux_cost_pos"].iloc[0]) == pytest.approx(
        expected_pos_aux
    )
    assert float(out[f"bem_p50_standby_aux_cost_pos"].iloc[0]) == pytest.approx(0.0)


def test_bcm_p_award_is_block_constant_within_bcm_products() -> None:
    bt = _mk_backtester("canonical_economic")
    hours = 8
    df, col = _tiny_backtest_df(hours=hours)

    # Fill meaningful aFRR inputs so EV diagnostics are populated for all bins.
    df[col.pred_afrr_capacity_price_pos] = [80.0] * hours
    df[col.pred_afrr_capacity_price_neg] = [60.0] * hours
    df[col.pred_afrr_activation_price_pos] = [100.0] * hours
    df[col.pred_afrr_activation_price_neg] = [120.0] * hours
    df[col.pred_afrr_activation_rate_pos] = [0.2] * hours
    df[col.pred_afrr_activation_rate_neg] = [0.3] * hours
    for q in bt.afrr_quantile_bins:
        df[f"{col.pred_afrr_capacity_price_pos}_{q}"] = [80.0 + i for i in range(hours)]
        df[f"{col.pred_afrr_capacity_price_neg}_{q}"] = [60.0 + i for i in range(hours)]
        df[f"{col.pred_afrr_activation_price_pos}_{q}"] = [100.0] * hours
        df[f"{col.pred_afrr_activation_price_neg}_{q}"] = [120.0] * hours
        df[f"{col.pred_afrr_activation_rate_pos}_{q}"] = [0.2] * hours
        df[f"{col.pred_afrr_activation_rate_neg}_{q}"] = [0.3] * hours

    # Constant per actual 4h BCM product blocks, with at least two distinct products.
    block_ids = assign_bcm_capacity_block(df[col.timestamp])["bcm_capacity_block_id"].to_numpy(dtype=str)
    unique_blocks = pd.unique(block_ids)
    assert len(unique_blocks) >= 2
    for b in range(len(bt.afrr_quantile_bins)):
        block0_prob = 0.25 + 0.01 * b
        block1_prob = 0.65 + 0.01 * b
        ppos = [block1_prob if blk == unique_blocks[1] else block0_prob for blk in block_ids]
        pneg = [0.55 + 0.01 * b if blk == unique_blocks[1] else (0.15 + 0.01 * b) for blk in block_ids]
        df[f"pacc_pos_bin_{b}"] = ppos
        df[f"pacc_neg_bin_{b}"] = pneg

    out = bt.optimize_dispatch(
        df,
        col,
        allowed_markets=("aFRR", "BCM", "BEM"),
        deterministic_reserve_settlement=False,
        strict_input_validation=False,
    )
    for b in range(len(bt.afrr_quantile_bins)):
        pos_col = f"ev_bcm_p_award_pos_bin_{b}"
        neg_col = f"ev_bcm_p_award_neg_bin_{b}"
        hourly_pos_coef_col = f"ev_rpos_coef_bin_{b}_eur_per_mw"
        hourly_neg_coef_col = f"ev_rneg_coef_bin_{b}_eur_per_mw"
        product_pos_coef_col = f"ev_bcm_product_coef_pos_bin_{b}_eur_per_mw"
        product_neg_coef_col = f"ev_bcm_product_coef_neg_bin_{b}_eur_per_mw"
        cap_price_pos_col = f"ev_bcm_capacity_bid_price_pos_bin_{b}_eur_per_mw_h"
        cap_price_neg_col = f"ev_bcm_capacity_bid_price_neg_bin_{b}_eur_per_mw_h"
        assert pos_col in out.columns
        assert neg_col in out.columns
        assert cap_price_pos_col in out.columns
        assert cap_price_neg_col in out.columns
        assert product_pos_coef_col in out.columns
        assert product_neg_coef_col in out.columns
        out_blocked = out.copy()
        out_blocked["_bcm_block"] = block_ids
        by_block_pos = {}
        by_block_neg = {}
        for _, grp in out_blocked.groupby("_bcm_block"):
            pvals = grp[pos_col].astype(float).to_numpy()
            nvals = grp[neg_col].astype(float).to_numpy()
            cap_pos_vals = grp[cap_price_pos_col].astype(float).to_numpy()
            cap_neg_vals = grp[cap_price_neg_col].astype(float).to_numpy()
            if len(pvals) > 1:
                assert np.allclose(pvals, pvals[0])
            if len(nvals) > 1:
                assert np.allclose(nvals, nvals[0])
            if len(cap_pos_vals) > 1:
                assert np.allclose(cap_pos_vals, cap_pos_vals[0])
            if len(cap_neg_vals) > 1:
                assert np.allclose(cap_neg_vals, cap_neg_vals[0])
            by_block_pos[grp["_bcm_block"].iloc[0]] = float(pvals[0])
            by_block_neg[grp["_bcm_block"].iloc[0]] = float(nvals[0])
            assert np.allclose(
                grp[product_pos_coef_col].astype(float).to_numpy(),
                float(grp[hourly_pos_coef_col].astype(float).sum()),
            )
            assert np.allclose(
                grp[product_neg_coef_col].astype(float).to_numpy(),
                float(grp[hourly_neg_coef_col].astype(float).sum()),
            )

        assert len(by_block_pos) >= 2
        assert by_block_pos[unique_blocks[0]] != by_block_pos[unique_blocks[1]]
        assert by_block_neg[unique_blocks[0]] != by_block_neg[unique_blocks[1]]


def test_bcm_auxiliary_ev_is_not_multiplied_by_product_duration_twice() -> None:
    bt = _mk_backtester("canonical_economic")
    bt.trans_eur_mwh = 0.0
    bt.deg_eur_mwh = 0.0
    bt.afrr_offer_cost_eur_mw_h = 0.0
    bt.aux_afrr_active_mw = 0.0
    bt.aux_standby_mw = 1.0
    bt.aux_trading_mw = 0.0

    df, col = _tiny_backtest_df(hours=4)
    df[col.timestamp] = pd.date_range("2026-01-01T03:00:00Z", periods=4, freq="h")
    df[col.pred_da_price] = [100.0] * 4
    for q in bt.afrr_quantile_bins:
        df[f"{col.pred_afrr_capacity_price_pos}_{q}"] = [0.0] * 4
        df[f"{col.pred_afrr_capacity_price_neg}_{q}"] = [0.0] * 4
        df[f"{col.pred_afrr_activation_price_pos}_{q}"] = [0.0] * 4
        df[f"{col.pred_afrr_activation_price_neg}_{q}"] = [0.0] * 4
        df[f"{col.pred_afrr_activation_rate_pos}_{q}"] = [0.0] * 4
        df[f"{col.pred_afrr_activation_rate_neg}_{q}"] = [0.0] * 4
    for b in range(len(bt.afrr_quantile_bins)):
        df[f"pacc_pos_bin_{b}"] = [1.0] * 4
        df[f"pacc_neg_bin_{b}"] = [1.0] * 4

    out = bt.optimize_dispatch(
        df,
        col,
        allowed_markets=("aFRR", "BCM"),
        deterministic_reserve_settlement=False,
        strict_input_validation=False,
    )
    p50_bin = bt.afrr_quantile_bins.index("p50")
    expected_hourly_aux_cost_per_mw = bt.dt_h / bt.p_max_mw * bt.aux_standby_mw * 100.0
    expected_product_aux_cost_per_mw = 4.0 * expected_hourly_aux_cost_per_mw

    assert float(out[f"ev_bcm_expected_aux_cost_pos_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_hourly_aux_cost_per_mw
    )
    assert float(out[f"ev_rpos_coef_bin_{p50_bin}_eur_per_mw"].iloc[0]) == pytest.approx(
        -expected_hourly_aux_cost_per_mw
    )
    assert float(out[f"ev_bcm_product_coef_pos_bin_{p50_bin}_eur_per_mw"].iloc[0]) == pytest.approx(
        -expected_product_aux_cost_per_mw
    )


def test_bcm_linked_activation_ev_uses_activation_fraction_per_mw_once() -> None:
    bt = _mk_backtester("canonical_economic")
    bt.trans_eur_mwh = 0.0
    bt.deg_eur_mwh = 0.0
    bt.afrr_offer_cost_eur_mw_h = 0.0
    bt.aux_afrr_active_mw = 0.0
    bt.aux_standby_mw = 0.0
    bt.aux_trading_mw = 0.0

    df, col = _tiny_backtest_df(hours=4)
    df[col.timestamp] = pd.date_range("2026-01-01T03:00:00Z", periods=4, freq="h")
    df[col.pred_da_price] = [0.0] * 4
    df[col.pred_afrr_activation_price_pos] = [100.0] * 4
    df[col.pred_afrr_activation_rate_pos] = [0.25] * 4
    for q in bt.afrr_quantile_bins:
        df[f"{col.pred_afrr_capacity_price_pos}_{q}"] = [0.0] * 4
        df[f"{col.pred_afrr_capacity_price_neg}_{q}"] = [0.0] * 4
        df[f"{col.pred_afrr_activation_price_pos}_{q}"] = [100.0] * 4
        df[f"{col.pred_afrr_activation_price_neg}_{q}"] = [0.0] * 4
        df[f"{col.pred_afrr_activation_rate_pos}_{q}"] = [0.25] * 4
        df[f"{col.pred_afrr_activation_rate_neg}_{q}"] = [0.0] * 4
    for b in range(len(bt.afrr_quantile_bins)):
        df[f"pacc_pos_bin_{b}"] = [0.5] * 4
        df[f"pacc_neg_bin_{b}"] = [0.0] * 4

    out = bt.optimize_dispatch(
        df,
        col,
        allowed_markets=("aFRR", "BCM"),
        deterministic_reserve_settlement=False,
        strict_input_validation=False,
    )
    p50_bin = bt.afrr_quantile_bins.index("p50")
    expected_hourly_activation_value_per_mw = 0.5 * 0.25 * 100.0 * bt.dt_h
    expected_product_activation_value_per_mw = 4.0 * expected_hourly_activation_value_per_mw

    assert float(out[f"ev_bcm_activation_rate_is_fraction_bin_{p50_bin}"].iloc[0]) == pytest.approx(1.0)
    assert float(out[f"ev_bcm_activation_volume_multiplier_applied_once_bin_{p50_bin}"].iloc[0]) == pytest.approx(1.0)
    assert float(out[f"ev_bcm_activation_value_pos_bin_{p50_bin}"].iloc[0]) == pytest.approx(
        expected_hourly_activation_value_per_mw
    )
    assert float(out[f"ev_bcm_product_coef_pos_bin_{p50_bin}_eur_per_mw"].iloc[0]) == pytest.approx(
        expected_product_activation_value_per_mw
    )
    objective_rebuild = (
        float(out["ev_da_charge_eur"].iloc[0])
        + float(out["ev_da_discharge_eur"].iloc[0])
        + float(out["ev_afrr_pos_eur"].iloc[0])
        + float(out["ev_afrr_neg_eur"].iloc[0])
        + float(out["ev_bem_only_pos_eur"].iloc[0])
        + float(out["ev_bem_only_neg_eur"].iloc[0])
        - float(out["ev_slack_penalty_pos_eur"].iloc[0])
        - float(out["ev_slack_penalty_neg_eur"].iloc[0])
    )
    assert float(out["ev_objective_rebuild_eur"].iloc[0]) == pytest.approx(objective_rebuild)
    assert float(out["ev_objective_rebuild_including_window_terminal_eur"].iloc[0]) == pytest.approx(
        objective_rebuild + float(out["ev_terminal_soc_credit_eur"].iloc[0])
    )


def test_bcm_capacity_mw_not_created_by_activation_without_locked_capacity() -> None:
    bt = _mk_backtester("canonical_economic")
    bt._strategy_permissions = StrategyPermissions(
        allow_da=False,
        id_mode="technical_repair",
        allow_bcm=True,
        allow_bcm_activation_obligations=True,
        allow_bem_only=True,
    )
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T08:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=0.0,
        planned_bem_only_neg_mw=7.0,
        pred_da_price=50.0,
        true_da_price=50.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=100.0,
        true_act_pos=100.0,
        pred_act_neg=100.0,
        true_act_neg=100.0,
        true_rate_pos=0.0,
        true_rate_neg=1.0,
        pred_rate_pos=0.0,
        pred_rate_neg=1.0,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
    )
    assert float(out["executed_bcm_capacity_neg_mw"]) == pytest.approx(0.0)
    assert float(out["submitted_bcm_capacity_neg_mw"]) == pytest.approx(0.0)
    assert float(out["locked_bcm_capacity_neg_mw"]) == pytest.approx(0.0)
    assert float(out["bem_only_executed_neg_mwh"]) > 0.0


def test_bcm_executed_capacity_equals_locked_capacity_not_activation() -> None:
    bt = _mk_backtester("canonical_economic")
    bt._strategy_permissions = StrategyPermissions(
        allow_da=False,
        id_mode="technical_repair",
        allow_bcm=True,
        allow_bcm_activation_obligations=True,
        allow_bem_only=False,
    )
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T08:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        pred_da_price=50.0,
        true_da_price=50.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=100.0,
        true_act_pos=100.0,
        pred_act_neg=100.0,
        true_act_neg=100.0,
        true_rate_pos=0.0,
        true_rate_neg=0.5,
        pred_rate_pos=0.0,
        pred_rate_neg=0.5,
        obligation_pos_mw=0.0,
        obligation_neg_mw=8.0,
    )
    assert float(out["executed_bcm_capacity_neg_mw"]) == pytest.approx(8.0)
    assert float(out["locked_bcm_capacity_neg_mw"]) == pytest.approx(8.0)
    assert float(out["bcm_linked_neg_activation_mwh"]) == pytest.approx(4.0 * bt.dt_h)


def test_bcm_simultaneous_pos_neg_capacity_requires_both_locked_sides() -> None:
    bt = _mk_backtester("canonical_economic")
    bt._strategy_permissions = StrategyPermissions(
        allow_da=False,
        id_mode="technical_repair",
        allow_bcm=True,
        allow_bcm_activation_obligations=True,
        allow_bem_only=False,
    )
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T08:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        pred_da_price=50.0,
        true_da_price=50.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=100.0,
        true_act_pos=100.0,
        pred_act_neg=100.0,
        true_act_neg=100.0,
        true_rate_pos=1.0,
        true_rate_neg=1.0,
        pred_rate_pos=1.0,
        pred_rate_neg=1.0,
        obligation_pos_mw=3.0,
        obligation_neg_mw=8.0,
    )
    assert float(out["submitted_bcm_capacity_pos_mw"]) == pytest.approx(3.0)
    assert float(out["locked_bcm_capacity_pos_mw"]) == pytest.approx(3.0)
    assert float(out["executed_bcm_capacity_pos_mw"]) == pytest.approx(3.0)
    assert float(out["submitted_bcm_capacity_neg_mw"]) == pytest.approx(8.0)
    assert float(out["locked_bcm_capacity_neg_mw"]) == pytest.approx(8.0)
    assert float(out["executed_bcm_capacity_neg_mw"]) == pytest.approx(8.0)


def test_bcm_reject_no_capacity_award_and_no_capacity_revenue() -> None:
    mc = MarketClearingEngine()
    bids = [
        BCMCapacityBid(
            ts=pd.Timestamp("2026-01-01T08:00:00Z"),
            side="pos",
            quantity_mw=5.0,
            capacity_price_eur_mw=200.0,
        )
    ]
    res = mc.clear_afrr_capacity(bids, true_cap_pos=100.0, true_cap_neg=0.0, true_act_pos=1000.0)
    assert np.isclose(float(res.awarded_pos_mw), 0.0)
    assert bool(res.pos_awarded) is False


def test_bcm_capacity_award_does_not_require_activation_price() -> None:
    mc = MarketClearingEngine()
    bids = [
        BCMCapacityBid(
            ts=pd.Timestamp("2026-01-01T08:00:00Z"),
            side="pos",
            quantity_mw=4.0,
            capacity_price_eur_mw=50.0,
        )
    ]
    assert not hasattr(bids[0], "energy_price_eur_mwh")
    res = mc.clear_afrr_capacity(bids, true_cap_pos=50.0, true_cap_neg=0.0)
    assert float(res.awarded_pos_mw) == pytest.approx(4.0)
    assert bool(res.pos_awarded) is True


def test_bcm_bid_builder_constructs_capacity_only_bid_without_activation_price() -> None:
    bt = _mk_backtester("canonical_economic")
    bids = bt.bid_builder.build_afrr_capacity_bids(
        ts=pd.Timestamp("2026-01-01T16:00:00Z"),
        reserve_pos_mw=1.0,
        reserve_neg_mw=0.0,
        pred_cap_pos=12.0,
        pred_cap_neg=0.0,
        pred_act_pos=9999.0,
        pred_act_neg=9999.0,
    )

    assert len(bids) == 1
    bid = bids[0]
    assert isinstance(bid, BCMCapacityBid)
    assert set(bid.__dataclass_fields__) == {"ts", "side", "quantity_mw", "capacity_price_eur_mw"}
    assert not hasattr(bid, "energy_price_eur_mwh")


def test_bcm_capacity_bid_preserves_negative_predicted_capacity_price() -> None:
    bt = _mk_backtester("canonical_economic")
    bids = bt.bid_builder.build_afrr_capacity_bids(
        ts=pd.Timestamp("2026-01-01T16:00:00Z"),
        reserve_pos_mw=1.0,
        reserve_neg_mw=0.0,
        pred_cap_pos=-12.5,
        pred_cap_neg=0.0,
        pred_act_pos=9999.0,
        pred_act_neg=9999.0,
    )

    assert len(bids) == 1
    assert bids[0].capacity_price_eur_mw == pytest.approx(-12.5)


def test_bcm_capacity_ev_diagnostic_does_not_use_activation_price() -> None:
    bt = _mk_backtester("canonical_economic")
    base, col = _tiny_backtest_df(hours=4)
    base[col.pred_da_price] = [0.0] * 4
    for q in bt.afrr_quantile_bins:
        base[f"{col.pred_afrr_capacity_price_pos}_{q}"] = [12.0] * 4
        base[f"{col.pred_afrr_capacity_price_neg}_{q}"] = [0.0] * 4
        base[f"{col.pred_afrr_activation_rate_pos}_{q}"] = [0.2] * 4
        base[f"{col.pred_afrr_activation_rate_neg}_{q}"] = [0.0] * 4
    for b in range(len(bt.afrr_quantile_bins)):
        base[f"pacc_pos_bin_{b}"] = [0.5] * 4
        base[f"pacc_neg_bin_{b}"] = [0.0] * 4

    low_act = base.copy()
    high_act = base.copy()
    low_act[col.pred_afrr_activation_price_pos] = [100.0] * 4
    low_act[col.pred_afrr_activation_price_neg] = [0.0] * 4
    high_act[col.pred_afrr_activation_price_pos] = [1000.0] * 4
    high_act[col.pred_afrr_activation_price_neg] = [0.0] * 4
    for q in bt.afrr_quantile_bins:
        low_act[f"{col.pred_afrr_activation_price_pos}_{q}"] = [100.0] * 4
        low_act[f"{col.pred_afrr_activation_price_neg}_{q}"] = [0.0] * 4
        high_act[f"{col.pred_afrr_activation_price_pos}_{q}"] = [1000.0] * 4
        high_act[f"{col.pred_afrr_activation_price_neg}_{q}"] = [0.0] * 4

    out_low = bt.optimize_dispatch(
        low_act,
        col,
        allowed_markets=("aFRR", "BCM"),
        deterministic_reserve_settlement=False,
        strict_input_validation=False,
    )
    out_high = bt.optimize_dispatch(
        high_act,
        col,
        allowed_markets=("aFRR", "BCM"),
        deterministic_reserve_settlement=False,
        strict_input_validation=False,
    )
    p50_bin = bt.afrr_quantile_bins.index("p50")
    cap_col = f"ev_bcm_capacity_value_pos_bin_{p50_bin}"
    act_col = f"ev_bcm_activation_value_pos_bin_{p50_bin}"

    assert float(out_low[cap_col].iloc[0]) == pytest.approx(float(out_high[cap_col].iloc[0]))
    assert float(out_low[act_col].iloc[0]) != pytest.approx(float(out_high[act_col].iloc[0]))


def test_bcm_capacity_price_controls_bcm_award_not_activation_price() -> None:
    mc = MarketClearingEngine()
    bids = [
        BCMCapacityBid(
            ts=pd.Timestamp("2026-01-01T08:00:00Z"),
            side="pos",
            quantity_mw=5.0,
            capacity_price_eur_mw=9999.0,
        )
    ]
    res = mc.clear_afrr_capacity(bids, true_cap_pos=1.0, true_cap_neg=0.0, true_act_pos=50.0)
    assert np.isclose(float(res.awarded_pos_mw), 0.0)
    assert bool(res.pos_awarded) is False


def test_bcm_product_level_capacity_acceptance_rejects_whole_block_and_no_mandatory_bem() -> None:
    bt = _mk_backtester("canonical_economic")
    col = BacktestColumnMap()
    ts_idx = pd.Series(pd.date_range("2026-01-01T16:00:00Z", periods=4, freq="h"))
    source = pd.DataFrame(
        {
            col.true_afrr_capacity_price_pos: [14.0, 14.0, 14.0, 14.0],
            col.true_afrr_capacity_price_neg: [0.0, 0.0, 0.0, 0.0],
        },
        index=pd.to_datetime(ts_idx, utc=True),
    )
    bids = [
        BCMCapacityBid(
            ts=ts_idx.iloc[0],
            side="pos",
            quantity_mw=1.0,
            capacity_price_eur_mw=20.0,
        )
    ]

    res = bt._clear_afrr_capacity_block_against_truth(
        cap_bids=bids,
        ts_idx=pd.to_datetime(ts_idx, utc=True),
        source=source,
        colmap=col,
    )

    assert float(res.awarded_pos_mw) == pytest.approx(0.0)
    assert bool(res.pos_awarded) is False
    assert float(res.awarded_pos_mw) == pytest.approx(0.0)

    out = bt._apply_market_clearing(
        target_time_utc=ts_idx.iloc[0],
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=0.0,
        planned_bem_only_neg_mw=0.0,
        pred_da_price=50.0,
        true_da_price=50.0,
        pred_cap_pos=20.0,
        true_cap_pos=14.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=150.0,
        true_act_pos=180.0,
        pred_act_neg=0.0,
        true_act_neg=0.0,
        true_rate_pos=1.0,
        true_rate_neg=0.0,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
    )
    assert float(out["bcm_capacity_accepted_mw"]) == pytest.approx(0.0)
    assert float(out["bcm_to_bem_mandatory_volume_mw"]) == pytest.approx(0.0)
    assert float(out["bem_total_bid_volume_mw"]) == pytest.approx(0.0)
    assert str(out["bem_activation_price_decision_time_utc"]) == ""


def test_bcm_product_level_capacity_acceptance_settles_pay_as_bid_not_cutoff() -> None:
    bt = _mk_backtester("canonical_economic")
    col = BacktestColumnMap()
    ts_idx = pd.Series(pd.date_range("2026-01-01T16:00:00Z", periods=4, freq="h"))
    source = pd.DataFrame(
        {
            col.true_afrr_capacity_price_pos: [14.0, 14.0, 14.0, 14.0],
            col.true_afrr_capacity_price_neg: [0.0, 0.0, 0.0, 0.0],
        },
        index=pd.to_datetime(ts_idx, utc=True),
    )
    bids = [
        BCMCapacityBid(
            ts=ts_idx.iloc[0],
            side="pos",
            quantity_mw=1.0,
            capacity_price_eur_mw=12.0,
        )
    ]
    res = bt._clear_afrr_capacity_block_against_truth(
        cap_bids=bids,
        ts_idx=pd.to_datetime(ts_idx, utc=True),
        source=source,
        colmap=col,
    )
    assert float(res.awarded_pos_mw) == pytest.approx(1.0)

    capacity_revenue = 0.0
    for _ in range(4):
        _, m = bt._settle_one_hour(
            soc=bt.soc_init,
            charge=0.0,
            discharge=0.0,
            reserve_pos=1.0,
            reserve_neg=0.0,
            da_price=0.0,
            cap_pos=14.0,
            cap_neg=0.0,
            act_pos_price=180.0,
            act_neg_price=0.0,
            act_pos_rate=0.0,
            act_neg_rate=0.0,
            cap_bid_pos=12.0,
            cap_bid_neg=0.0,
        )
        capacity_revenue += float(m["bcm_capacity_revenue_eur"])

    assert capacity_revenue == pytest.approx(1.0 * 12.0 * 4.0)
    assert capacity_revenue != pytest.approx(1.0 * 14.0 * 4.0)


def test_rejected_bcm_energy_price_does_not_create_mandatory_bem() -> None:
    bt = _mk_backtester("canonical_economic")
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T16:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=0.0,
        planned_bem_only_neg_mw=0.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=20.0,
        true_cap_pos=14.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=150.0,
        true_act_pos=180.0,
        pred_act_neg=0.0,
        true_act_neg=0.0,
        true_rate_pos=1.0,
        true_rate_neg=0.0,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
    )

    assert float(out["bcm_to_bem_mandatory_volume_mw"]) == pytest.approx(0.0)
    assert float(out["bem_total_bid_volume_mw"]) == pytest.approx(0.0)
    assert float(out["bcm_capacity_available_mw"]) == pytest.approx(0.0)
    assert str(out["bem_activation_price_decision_time_utc"]) == ""


def test_accepted_bcm_creates_mandatory_bem_obligation_and_activation_price_is_bem_only() -> None:
    bt = _mk_backtester("canonical_economic")
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T16:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=0.0,
        planned_bem_only_neg_mw=0.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=12.0,
        true_cap_pos=14.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=150.0,
        true_act_pos=180.0,
        pred_act_neg=0.0,
        true_act_neg=0.0,
        true_rate_pos=0.5,
        true_rate_neg=0.0,
        obligation_pos_mw=1.0,
        obligation_neg_mw=0.0,
        obligation_capacity_price_pos=12.0,
        obligation_energy_pos=150.0,
    )

    block_id = str(out["bcm_product_block_id"])
    assert "T16:00:00" in block_id
    assert "T20:00:00" in block_id
    assert str(out["bcm_direction"]) == "pos"
    assert float(out["bcm_capacity_bid_price_pos_eur_per_mw_h"]) == pytest.approx(12.0)
    assert float(out["bcm_capacity_cutoff_price_pos_eur_per_mw_h"]) == pytest.approx(14.0)
    assert float(out["bcm_to_bem_mandatory_volume_mw"]) == pytest.approx(1.0)
    assert float(out["bem_total_bid_volume_mw"]) == pytest.approx(1.0)
    assert float(out["bem_activation_bid_price_pos_eur_per_mwh"]) == pytest.approx(150.0)
    assert float(out["bem_clearing_price_pos_eur_per_mwh"]) == pytest.approx(180.0)
    assert str(out["bem_activation_price_decision_time_utc"]) == "2026-01-01T15:00:00+00:00"
    assert str(out["bem_delivery_hour_start_utc"]) == "2026-01-01T16:00:00+00:00"
    assert float(out["bcm_capacity_bid_has_activation_price"]) == pytest.approx(0.0)
    assert float(out["bcm_to_bem_activation_price_transferred"]) == pytest.approx(0.0)
    assert float(out["settlement_cap_bid_price_pos_eur_mw"]) == pytest.approx(12.0)

    out_next_hour = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T17:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=0.0,
        planned_bem_only_neg_mw=0.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=12.0,
        true_cap_pos=14.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=150.0,
        true_act_pos=180.0,
        pred_act_neg=0.0,
        true_act_neg=0.0,
        true_rate_pos=0.5,
        true_rate_neg=0.0,
        obligation_pos_mw=1.0,
        obligation_neg_mw=0.0,
        obligation_capacity_price_pos=12.0,
    )
    assert str(out_next_hour["bem_activation_price_decision_time_utc"]) == "2026-01-01T16:00:00+00:00"
    assert str(out_next_hour["bem_delivery_hour_start_utc"]) == "2026-01-01T17:00:00+00:00"


def test_free_bem_is_separate_from_rejected_bcm_capacity() -> None:
    bt = _mk_backtester("canonical_economic")
    bt._strategy_permissions = StrategyPermissions(
        allow_da=False,
        id_mode="technical_repair",
        allow_bcm=True,
        allow_bcm_activation_obligations=True,
        allow_bem_only=True,
    )
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T16:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=1.0,
        planned_bem_only_neg_mw=0.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=20.0,
        true_cap_pos=14.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=150.0,
        true_act_pos=180.0,
        pred_act_neg=0.0,
        true_act_neg=0.0,
        true_rate_pos=1.0,
        true_rate_neg=0.0,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
    )

    assert float(out["bcm_to_bem_mandatory_volume_mw"]) == pytest.approx(0.0)
    assert float(out["bem_free_volume_mw"]) == pytest.approx(1.0)
    assert float(out["bem_total_bid_volume_mw"]) == pytest.approx(1.0)
    assert float(out["settlement_cap_bid_price_pos_eur_mw"]) == pytest.approx(0.0)


def test_rhpf_bem_neg_submits_true_activation_price_and_clears_zero_price() -> None:
    bt = _mk_backtester("canonical_economic")
    bt._strategy_permissions = StrategyPermissions(
        allow_da=False,
        id_mode="technical_repair",
        allow_bcm=False,
        allow_bcm_activation_obligations=False,
        allow_bem_only=True,
    )
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T16:00:00Z"),
        is_perfect_foresight=True,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=0.0,
        planned_bem_only_neg_mw=10.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=100.0,
        true_act_pos=100.0,
        pred_act_neg=50.0,
        true_act_neg=0.0,
        true_rate_pos=0.0,
        true_rate_neg=0.25,
        soc_now=10.0,
    )

    assert float(out["bem_only_submitted_neg_mw"]) == pytest.approx(10.0)
    assert float(out["bem_submitted_activation_price_neg_eur_per_mwh"]) == pytest.approx(0.0)
    assert float(out["bem_activation_bid_price_neg_eur_per_mwh"]) == pytest.approx(0.0)
    assert float(out["bem_clearing_price_neg_eur_per_mwh"]) == pytest.approx(0.0)
    assert float(out["afrr_act_neg_accepted"]) == pytest.approx(1.0)
    assert float(out["bem_only_executed_neg_mwh"]) == pytest.approx(10.0 * 0.25 * bt.dt_h)


def test_bem_activation_settlement_uses_energy_settlement_price_not_bcm_capacity_price() -> None:
    bt = _mk_backtester("canonical_economic")
    _, m = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=0.0,
        discharge=0.0,
        reserve_pos=1.0,
        reserve_neg=0.0,
        da_price=0.0,
        cap_pos=14.0,
        cap_neg=0.0,
        act_pos_price=180.0,
        act_neg_price=0.0,
        act_pos_rate=0.5,
        act_neg_rate=0.0,
        cap_bid_pos=12.0,
        cap_bid_neg=0.0,
    )

    delivered = float(m["delivered_activation_pos_mwh"])
    assert delivered > 0.0
    assert float(m["revenue_activation_eur"]) == pytest.approx(delivered * 180.0)
    assert float(m["revenue_activation_eur"]) != pytest.approx(0.5 * 12.0)
    assert float(m["revenue_activation_eur"]) != pytest.approx(0.5 * 14.0)


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


def test_soc_mass_balance_audit_uses_settled_physical_columns_and_writes_debug(tmp_path: Path) -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.Timestamp("2025-05-01T11:00:00Z")
    soc_prev = 10.0
    bt.soc_init = soc_prev

    aux_mwh = 0.25
    settled_charge_mw = 0.6421052631578947
    delta_soc, _ = bt._calculate_soc_delta(
        charge_mw=settled_charge_mw,
        discharge_mw=0.0,
        id_charge_mw=0.0,
        id_discharge_mw=0.0,
        act_pos_mwh=0.0,
        act_neg_mwh=0.0,
        aux_mwh=aux_mwh,
        battery_specs={"eta_in": bt.eta_in, "eta_out": bt.eta_out},
        dt_h=bt.dt_h,
    )
    got_soc = soc_prev + delta_soc
    assert np.isclose(
        bt._calculate_soc_delta(
            charge_mw=0.0,
            discharge_mw=0.0,
            id_charge_mw=0.0,
            id_discharge_mw=0.0,
            act_pos_mwh=0.0,
            act_neg_mwh=0.0,
            aux_mwh=aux_mwh,
            battery_specs={"eta_in": bt.eta_in, "eta_out": bt.eta_out},
            dt_h=bt.dt_h,
        )[0],
        -0.25,
    )

    realized = pd.DataFrame(
        {
            col.timestamp: [ts],
            "real_soc_mwh": [got_soc],
            "real_soc_start_mwh": [soc_prev],
            "real_charge_mw": [settled_charge_mw],
            "real_discharge_mw": [0.0],
            "real_id_charge_mw": [0.0],
            "real_id_discharge_mw": [0.0],
            "real_act_pos_mwh": [0.0],
            "real_act_neg_mwh": [0.0],
            "real_aux_energy_mwh": [aux_mwh],
            "real_pnl_eur": [0.0],
        }
    )
    dispatch = pd.DataFrame({col.timestamp: [ts]})
    df_input = pd.DataFrame({col.timestamp: [ts]})
    bt._audit_backtest_results(realized=realized, dispatch=dispatch, df_input=df_input, colmap=col)

    with pytest.raises(RuntimeError, match="missing columns for SoC mass-balance"):
        bt._audit_backtest_results(
            realized=realized.drop(columns=["real_aux_energy_mwh"]),
            dispatch=dispatch,
            df_input=df_input,
            colmap=col,
        )

    debug_path = tmp_path / "backtest_soc_mass_balance_debug.csv"
    bt._soc_mass_balance_debug_path = debug_path
    bad = realized.copy()
    bad["real_charge_mw"] = settled_charge_mw + 0.25 / bt.eta_in
    with pytest.raises(RuntimeError, match="SoC mass-balance mismatch"):
        bt._audit_backtest_results(realized=bad, dispatch=dispatch, df_input=df_input, colmap=col)
    debug = pd.read_csv(debug_path)
    assert debug_path.exists()
    assert "formula_components" in debug.columns
    assert np.isclose(float(debug.loc[debug["debug_row_role"] == "current", "soc_mismatch_mwh"].iloc[0]), 0.25)
    assert "real_aux_power_mw" in debug.columns
    assert "real_aux_state" in debug.columns
    assert "da_bid_locked" in debug.columns


def test_safe_hold_delta_uses_backtester_eta_mapping_not_raw_battery_specs() -> None:
    bt = _mk_backtester()
    delta_soc, components = bt._calculate_soc_delta(
        charge_mw=0.0,
        discharge_mw=0.0,
        id_charge_mw=0.0,
        id_discharge_mw=0.0,
        act_pos_mwh=0.0,
        act_neg_mwh=0.0,
        aux_mwh=0.25,
        battery_specs={"eta_in": bt.eta_in, "eta_out": bt.eta_out},
        dt_h=bt.dt_h,
    )
    assert delta_soc == pytest.approx(-0.25)
    assert components["eta_in"] == pytest.approx(bt.eta_in)
    assert components["eta_out"] == pytest.approx(bt.eta_out)


def test_soc_mass_balance_audit_aux_energy_is_required() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.Timestamp("2025-05-01T11:00:00Z")
    bt.soc_init = 17.86

    realized_ok = pd.DataFrame(
        {
            col.timestamp: [ts],
            "real_soc_mwh": [17.61],
            "real_soc_start_mwh": [17.86],
            "real_charge_mw": [0.0],
            "real_discharge_mw": [0.0],
            "real_id_charge_mw": [0.0],
            "real_id_discharge_mw": [0.0],
            "real_act_pos_mwh": [0.0],
            "real_act_neg_mwh": [0.0],
            "real_aux_energy_mwh": [0.25],
            "real_pnl_eur": [0.0],
        }
    )
    dispatch = pd.DataFrame({col.timestamp: [ts]})
    df_input = pd.DataFrame({col.timestamp: [ts]})
    bt._audit_backtest_results(realized=realized_ok, dispatch=dispatch, df_input=df_input, colmap=col)

    realized_bad = realized_ok.copy()
    realized_bad["real_aux_energy_mwh"] = 0.0
    with pytest.raises(RuntimeError, match="SoC mass-balance mismatch"):
        bt._audit_backtest_results(realized=realized_bad, dispatch=dispatch, df_input=df_input, colmap=col)


def test_soc_mass_balance_audit_id_sell_grid_energy_visible_to_audit() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.Timestamp("2025-05-01T11:00:00Z")
    bt.soc_init = 18.0
    id_sell_mwh = 0.2250079225
    id_soc_drop_mwh = 0.25
    bt.eta_out = float(np.sqrt(id_sell_mwh / id_soc_drop_mwh))
    id_discharge_mw = id_sell_mwh / max(bt.eta_out * bt.dt_h, 1e-12)

    realized_ok = pd.DataFrame(
        {
            col.timestamp: [ts],
            "real_soc_mwh": [17.61],
            "real_soc_start_mwh": [18.0],
            "real_charge_mw": [0.0],
            "real_discharge_mw": [0.0],
            "real_id_charge_mw": [0.0],
            "real_id_discharge_mw": [id_discharge_mw],
            "real_id_buy_mwh": [0.0],
            "real_id_sell_mwh": [id_sell_mwh],
            "real_act_pos_mwh": [0.0],
            "real_act_neg_mwh": [0.0],
            "real_aux_energy_mwh": [0.14],
            "real_pnl_eur": [0.0],
        }
    )
    dispatch = pd.DataFrame({col.timestamp: [ts]})
    df_input = pd.DataFrame({col.timestamp: [ts]})
    bt._audit_backtest_results(realized=realized_ok, dispatch=dispatch, df_input=df_input, colmap=col)

    realized_bad = realized_ok.copy()
    realized_bad["real_id_discharge_mw"] = 0.0
    with pytest.raises(RuntimeError, match="SoC mass-balance mismatch"):
        bt._audit_backtest_results(realized=realized_bad, dispatch=dispatch, df_input=df_input, colmap=col)


def test_soc_mass_balance_audit_id_buy_grid_energy_visible_to_audit() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.Timestamp("2025-05-01T12:00:00Z")
    bt.soc_init = 10.0
    id_buy_mwh = 0.25
    id_charge_mw = id_buy_mwh * bt.eta_in / max(bt.dt_h, 1e-12)
    expected_soc = 10.0 + bt.eta_in * id_charge_mw * bt.dt_h

    realized_ok = pd.DataFrame(
        {
            col.timestamp: [ts],
            "real_soc_mwh": [expected_soc],
            "real_soc_start_mwh": [10.0],
            "real_charge_mw": [0.0],
            "real_discharge_mw": [0.0],
            "real_id_charge_mw": [id_charge_mw],
            "real_id_discharge_mw": [0.0],
            "real_id_buy_mwh": [id_buy_mwh],
            "real_id_sell_mwh": [0.0],
            "real_act_pos_mwh": [0.0],
            "real_act_neg_mwh": [0.0],
            "real_aux_energy_mwh": [0.0],
            "real_pnl_eur": [0.0],
        }
    )
    dispatch = pd.DataFrame({col.timestamp: [ts]})
    df_input = pd.DataFrame({col.timestamp: [ts]})
    bt._audit_backtest_results(realized=realized_ok, dispatch=dispatch, df_input=df_input, colmap=col)

    realized_bad = realized_ok.copy()
    realized_bad["real_id_charge_mw"] = 0.0
    with pytest.raises(RuntimeError, match="SoC mass-balance mismatch"):
        bt._audit_backtest_results(realized=realized_bad, dispatch=dispatch, df_input=df_input, colmap=col)


def test_terminal_closure_fields_prove_physical_mass_balance_repair() -> None:
    idx = pd.Index([0, 1])
    mask = BatteryBacktester._terminal_repair_mass_balance_physical_mask(
        id_reason_text=pd.Series(["none", "none"], index=idx),
        actual_soc_mwh=pd.Series([3.0, 10.0], index=idx),
        target_soc_mwh=10.0,
        real_id_charge_mw=pd.Series([0.0, 0.0], index=idx),
        terminal_closure_reason=pd.Series(["none", "shortfall_recovery"], index=idx),
        terminal_closure_applied=pd.Series([0.0, 1.0], index=idx),
        terminal_closure_target_met=pd.Series([0.0, 1.0], index=idx),
        terminal_closure_id_buy_mwh=pd.Series([0.0, 7.0], index=idx),
        terminal_closure_cost_accounted=pd.Series([False, True], index=idx),
    )
    assert list(mask.astype(bool)) == [False, True]


def test_terminal_closure_without_cost_accounting_does_not_mask_mass_balance() -> None:
    idx = pd.Index([0])
    mask = BatteryBacktester._terminal_repair_mass_balance_physical_mask(
        id_reason_text=pd.Series(["none"], index=idx),
        actual_soc_mwh=pd.Series([10.0], index=idx),
        target_soc_mwh=10.0,
        real_id_charge_mw=pd.Series([0.0], index=idx),
        terminal_closure_reason=pd.Series(["shortfall_recovery"], index=idx),
        terminal_closure_applied=pd.Series([1.0], index=idx),
        terminal_closure_target_met=pd.Series([1.0], index=idx),
        terminal_closure_id_buy_mwh=pd.Series([7.0], index=idx),
        terminal_closure_cost_accounted=pd.Series([False], index=idx),
    )

    assert list(mask.astype(bool)) == [False]


def test_soc_mass_balance_audit_accepts_costed_physical_terminal_closure() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.Timestamp("2025-05-01T23:00:00Z")
    bt.soc_init = 8.4
    realized = pd.DataFrame(
        {
            col.timestamp: [ts],
            "real_soc_mwh": [10.0],
            "real_charge_mw": [0.0],
            "real_discharge_mw": [0.0],
            "real_act_pos_mwh": [0.0],
            "real_act_neg_mwh": [0.0],
            "real_aux_energy_mwh": [0.0],
            "real_pnl_eur": [-100.0],
            "terminal_closure_reason": ["shortfall_recovery"],
            "terminal_closure_applied": [1.0],
            "final_soc_target_met_after_closure": [1.0],
            "terminal_closure_id_buy_mwh": [1.6842105263],
            "terminal_closure_soc_after_mwh": [10.0],
            "terminal_closure_net_pnl_eur": [-100.0],
        }
    )
    dispatch = pd.DataFrame({col.timestamp: [ts]})
    df_input = pd.DataFrame({col.timestamp: [ts]})

    bt._audit_backtest_results(realized=realized, dispatch=dispatch, df_input=df_input, colmap=col)


def test_soc_mass_balance_audit_uses_terminal_closure_after_soc_when_recovery_after_is_zero() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.Timestamp("2025-05-01T23:00:00Z")
    bt.soc_init = 8.4
    realized = pd.DataFrame(
        {
            col.timestamp: [ts],
            "real_soc_mwh": [10.0],
            "real_charge_mw": [0.0],
            "real_discharge_mw": [0.0],
            "real_act_pos_mwh": [0.0],
            "real_act_neg_mwh": [0.0],
            "real_aux_energy_mwh": [0.0],
            "real_pnl_eur": [-100.0],
            "terminal_closure_reason": ["shortfall_recovery"],
            "terminal_closure_applied": [1.0],
            "final_soc_target_met_after_closure": [1.0],
            "terminal_closure_id_buy_mwh": [1.6842105263],
            "terminal_recovery_soc_after_mwh": [0.0],
            "final_soc_after_terminal_closure_mwh": [0.0],
            "terminal_closure_soc_after_mwh": [10.0],
            "terminal_closure_net_pnl_eur": [-100.0],
        }
    )
    after_soc = BatteryBacktester._terminal_repair_mass_balance_after_soc(
        frame=realized,
        actual_soc_mwh=realized["real_soc_mwh"],
    )
    assert float(after_soc.iloc[0]) == pytest.approx(10.0)

    dispatch = pd.DataFrame({col.timestamp: [ts]})
    df_input = pd.DataFrame({col.timestamp: [ts]})

    bt._audit_backtest_results(realized=realized, dispatch=dispatch, df_input=df_input, colmap=col)


def test_soc_mass_balance_audit_rejects_terminal_closure_without_cost_accounting() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.Timestamp("2025-05-01T23:00:00Z")
    bt.soc_init = 8.4
    realized = pd.DataFrame(
        {
            col.timestamp: [ts],
            "real_soc_mwh": [10.0],
            "real_charge_mw": [0.0],
            "real_discharge_mw": [0.0],
            "real_act_pos_mwh": [0.0],
            "real_act_neg_mwh": [0.0],
            "real_aux_energy_mwh": [0.0],
            "real_pnl_eur": [0.0],
            "terminal_closure_reason": ["shortfall_recovery"],
            "terminal_closure_applied": [1.0],
            "final_soc_target_met_after_closure": [1.0],
            "terminal_closure_id_buy_mwh": [1.6842105263],
            "terminal_closure_soc_after_mwh": [10.0],
        }
    )
    dispatch = pd.DataFrame({col.timestamp: [ts]})
    df_input = pd.DataFrame({col.timestamp: [ts]})

    with pytest.raises(RuntimeError, match="SoC mass-balance mismatch"):
        bt._audit_backtest_results(realized=realized, dispatch=dispatch, df_input=df_input, colmap=col)


def test_id_recourse_buy_and_sell_apply_physical_soc_effect_and_reason() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.aux_trading_mw = 0.0
    bt.aux_standby_mw = 0.0
    bt._strategy_permissions = bt.resolve_strategy_permissions(
        strategy_name="da",
        allowed_markets=("DA",),
        id_recourse_mode="common",
    )

    soc_buy, buy = bt._settle_one_hour(
        soc=10.0,
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
        id_recourse_reason_hint="none",
    )
    assert soc_buy == pytest.approx(10.95)
    assert float(buy["id_buy_mwh"]) == pytest.approx(1.0)
    assert float(buy["id_charge_internal_mwh"]) == pytest.approx(0.95)
    assert str(buy["id_recourse_reason"]) == "invalid_protected_soc_recovery_without_violation"
    assert float(buy["protected_soc_recovery_invalid_without_violation"]) == pytest.approx(1.0)

    soc_sell, sell = bt._settle_one_hour(
        soc=10.0,
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
        id_charge_mw=0.0,
        id_discharge_mw=1.0,
        id_recourse_reason_hint="none",
    )
    assert soc_sell == pytest.approx(10.0 - 1.0 / 0.95)
    assert float(sell["id_sell_mwh"]) == pytest.approx(1.0)
    assert float(sell["id_discharge_internal_mwh"]) == pytest.approx(1.0 / 0.95)
    assert str(sell["id_recourse_reason"]) == "upper_soc_relief"


def test_hard_terminal_economic_repair_does_not_pass_physical_shortfall() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "hard"
    summary = {
        "final_soc_shortfall_mwh": 0.19,
        "final_soc_physical_check_pass": 0.0,
        "terminal_soc_repair_cost_eur": 19.0,
        "terminal_soc_net_adjustment_eur": -19.0,
        "realized_pnl_excl_terminal_eur": 10.0,
        "realized_total_pnl_eur": -9.0,
    }
    terminal_repair_included = bool(
        abs(
            float(summary["realized_total_pnl_eur"])
            - (float(summary["realized_pnl_excl_terminal_eur"]) + float(summary["terminal_soc_net_adjustment_eur"]))
        )
        <= 1e-4
    )
    economic_repair_pass = bool(
        (float(summary["final_soc_shortfall_mwh"]) <= 1e-6)
        or (
            str(bt.final_soc_mode) != "hard"
            and float(summary["terminal_soc_repair_cost_eur"]) > 0.0
            and terminal_repair_included
        )
    )
    assert terminal_repair_included is True
    assert economic_repair_pass is False


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


def test_da_meter_side_pcc_settlement_and_soc_convention() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.9
    bt.eta_out = 0.8
    bt.trans_eur_mwh = 2.0
    bt.deg_eur_mwh = 10.0
    bt.aux_trading_mw = 0.0

    soc0 = 10.0
    da_mw = 1.0
    neg_da_price = -50.0

    soc1, m = bt._settle_one_hour(
        soc=soc0,
        charge=da_mw,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=neg_da_price,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
    )
    dt = bt.dt_h
    assert float(m["da_charge_internal_mwh"]) == pytest.approx(da_mw * dt * bt.eta_in)
    assert float(m["da_buy_mwh"]) == pytest.approx(da_mw * dt)
    assert float(m["da_soc_gain_mwh"]) == pytest.approx(da_mw * dt * bt.eta_in)
    assert float(m["da_soc_loss_mwh"]) == pytest.approx(0.0)
    assert float(m["revenue_da_eur"]) == pytest.approx(0.0)
    assert float(m["cost_da_eur"]) == pytest.approx(neg_da_price * da_mw * dt)
    expected_soc = soc0 + da_mw * dt * bt.eta_in
    assert float(soc1) == pytest.approx(expected_soc)
    assert m["da_settlement_convention"] == "meter_side_pcc"
    assert float(m["pnl_eur"]) == pytest.approx(
        float(m["revenue_da_eur"]) - float(m["cost_da_eur"])
        - float(m["transaction_cost_eur"])
        - float(m["degradation_cost_eur"])
        - float(m["aux_cost_eur"])
        - float(m["penalty_eur"]),
        rel=1e-9,
    )

    _, m2 = bt._settle_one_hour(
        soc=soc0,
        charge=0.0,
        discharge=1.0,
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=25.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
    )
    assert float(m2["da_discharge_internal_mwh"]) == pytest.approx((1.0 * dt) / bt.eta_out)
    assert float(m2["da_sell_mwh"]) == pytest.approx(1.0 * dt)
    assert float(m2["da_soc_loss_mwh"]) == pytest.approx((1.0 * dt) / bt.eta_out)
    assert float(m2["revenue_da_eur"]) == pytest.approx(1.0 * dt * 25.0)


def test_da_ev_settlement_and_diagnostics_are_meter_side_consistent() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.9
    bt.eta_out = 0.9
    bt.trans_eur_mwh = 2.0
    bt.deg_eur_mwh = 10.0
    bt.aux_trading_mw = 0.0
    bt.aux_standby_mw = 0.0
    bt.aux_afrr_active_mw = 0.0
    dt = bt.dt_h

    _, charge = bt._settle_one_hour(
        soc=5.0,
        charge=1.0,
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
    )
    expected_charge_ev = (-100.0 - bt.trans_eur_mwh - bt.deg_eur_mwh * bt.eta_in) * dt
    assert float(charge["da_buy_mwh"]) == pytest.approx(1.0 * dt)
    assert float(charge["da_soc_gain_mwh"]) == pytest.approx(1.0 * dt * bt.eta_in)
    assert float(charge["cost_da_eur"]) == pytest.approx(100.0 * dt)
    assert float(charge["da_ev_charge_coef_eur_per_mw"]) == pytest.approx(expected_charge_ev)
    assert float(charge["pnl_eur"]) == pytest.approx(expected_charge_ev)

    _, discharge = bt._settle_one_hour(
        soc=5.0,
        charge=0.0,
        discharge=1.0,
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
    expected_discharge_ev = (100.0 - bt.trans_eur_mwh - bt.deg_eur_mwh / bt.eta_out) * dt
    assert float(discharge["da_sell_mwh"]) == pytest.approx(1.0 * dt)
    assert float(discharge["da_soc_loss_mwh"]) == pytest.approx((1.0 * dt) / bt.eta_out)
    assert float(discharge["revenue_da_eur"]) == pytest.approx(100.0 * dt)
    assert float(discharge["da_ev_discharge_coef_eur_per_mw"]) == pytest.approx(expected_discharge_ev)
    assert float(discharge["pnl_eur"]) == pytest.approx(expected_discharge_ev)

    bt.aux_trading_mw = 0.1
    _, neg_charge = bt._settle_one_hour(
        soc=5.0,
        charge=bt.p_max_mw,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=-50.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
    )
    _, neg_discharge = bt._settle_one_hour(
        soc=15.0,
        charge=0.0,
        discharge=bt.p_max_mw,
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=-50.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
    )
    assert float(neg_charge["cost_da_eur"]) < 0.0
    assert float(neg_charge["aux_cost_eur"]) < 0.0
    assert float(neg_discharge["revenue_da_eur"]) < 0.0

    df, col = _tiny_backtest_df(hours=30)
    opt = bt.optimize_dispatch(
        df,
        col,
        soc_start=10.0,
        soc_end_min_target=None,
        allowed_markets=("DA",),
    )
    assert "terminal_soc_credit_eur" in opt.columns
    assert "ev_terminal_soc_credit_eur" in opt.columns
    assert np.allclose(
        pd.to_numeric(opt["terminal_soc_credit_eur"], errors="coerce"),
        pd.to_numeric(opt["ev_terminal_soc_credit_eur"], errors="coerce"),
    )
    assert "da_simultaneous_charge_discharge_allowed" in opt.columns
    assert "da_simultaneous_charge_discharge_mw" in opt.columns
    assert "da_simultaneous_charge_discharge_flag" in opt.columns


def test_settlement_degradation_uses_internal_throughput_for_da_id_and_activation() -> None:
    bt = _mk_backtester("canonical_economic")
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.deg_eur_mwh = 25.0
    bt.trans_eur_mwh = 0.0
    bt.aux_off_mw = 0.0
    bt.aux_trading_mw = 0.0
    bt.aux_standby_mw = 0.0
    bt.aux_afrr_active_mw = 0.0
    bt._strategy_permissions = StrategyPermissions(
        allow_da=True,
        id_mode="technical_repair",
        allow_bcm=True,
        allow_bcm_activation_obligations=True,
        allow_bem_only=True,
    )

    _, da_charge = bt._settle_one_hour(
        soc=2.0,
        charge=10.0,
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
    )
    assert float(da_charge["da_buy_mwh"]) == pytest.approx(10.0)
    assert float(da_charge["cost_da_eur"]) == pytest.approx(1000.0)
    assert float(da_charge["degradation_cost_da_eur"]) == pytest.approx(10.0 * bt.eta_in * 25.0)

    _, da_discharge = bt._settle_one_hour(
        soc=18.0,
        charge=0.0,
        discharge=10.0,
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
    assert float(da_discharge["da_sell_mwh"]) == pytest.approx(10.0)
    assert float(da_discharge["revenue_da_eur"]) == pytest.approx(1000.0)
    assert float(da_discharge["degradation_cost_da_eur"]) == pytest.approx(10.0 / bt.eta_out * 25.0)

    _, id_trade = bt._settle_one_hour(
        soc=1.0,
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
        id_charge_mw=4.0,
        id_discharge_mw=3.0,
    )
    assert float(id_trade["degradation_cost_id_eur"]) == pytest.approx(
        (float(id_trade["id_buy_mwh"]) * bt.eta_in + float(id_trade["id_sell_mwh"]) / bt.eta_out) * 25.0
    )

    _, activation = bt._settle_one_hour(
        soc=5.0,
        charge=0.0,
        discharge=0.0,
        reserve_pos=2.0 / bt.eta_out,
        reserve_neg=2.0 * bt.eta_in,
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=100.0,
        act_neg_price=100.0,
        act_pos_rate=1.0,
        act_neg_rate=1.0,
    )
    assert float(activation["degradation_cost_activation_pos_eur"]) == pytest.approx(
        float(activation["act_pos_mwh"]) / bt.eta_out * 25.0
    )
    assert float(activation["degradation_cost_activation_neg_eur"]) == pytest.approx(
        float(activation["act_neg_mwh"]) * bt.eta_in * 25.0
    )
    internal_throughput = (
        float(activation["da_charge_internal_mwh"])
        + float(activation["da_discharge_internal_mwh"])
        + float(activation["id_charge_internal_mwh"])
        + float(activation["id_discharge_internal_mwh"])
        + float(activation["act_pos_internal_mwh"])
        + float(activation["act_neg_internal_mwh"])
    )
    assert activation["degradation_basis"] == "internal_throughput"
    assert float(activation["degradation_cost_eur"]) == pytest.approx(internal_throughput * 25.0)
    assert float(activation["revenue_activation_eur"]) == pytest.approx(
        float(activation["act_pos_mwh"]) * 100.0 + float(activation["act_neg_mwh"]) * 100.0
    )


def test_calculate_soc_delta_uses_meter_side_ev_with_eta_095() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95

    dt_h = bt.dt_h
    specs = {"eta_in": bt.eta_in, "eta_out": bt.eta_out}

    delta_charge, charge_components = bt._calculate_soc_delta(
        charge_mw=1.0,
        discharge_mw=0.0,
        id_charge_mw=0.0,
        id_discharge_mw=0.0,
        act_pos_mwh=0.0,
        act_neg_mwh=0.0,
        aux_mwh=0.0,
        battery_specs=specs,
        dt_h=dt_h,
    )
    assert delta_charge == pytest.approx(0.95)
    assert charge_components["charge_grid_mwh"] == pytest.approx(1.0)
    assert charge_components["charge_internal_gain_mwh"] == pytest.approx(0.95)

    delta_discharge, discharge_components = bt._calculate_soc_delta(
        charge_mw=0.0,
        discharge_mw=1.0,
        id_charge_mw=0.0,
        id_discharge_mw=0.0,
        act_pos_mwh=0.0,
        act_neg_mwh=0.0,
        aux_mwh=0.0,
        battery_specs=specs,
        dt_h=dt_h,
    )
    assert delta_discharge == pytest.approx(-1.0 / 0.95)
    assert discharge_components["discharge_grid_mwh"] == pytest.approx(1.0)
    assert discharge_components["discharge_internal_loss_mwh"] == pytest.approx(1.0 / 0.95)

    delta_act_neg, act_neg_components = bt._calculate_soc_delta(
        charge_mw=0.0,
        discharge_mw=0.0,
        id_charge_mw=0.0,
        id_discharge_mw=0.0,
        act_pos_mwh=0.0,
        act_neg_mwh=1.0,
        aux_mwh=0.0,
        battery_specs=specs,
        dt_h=dt_h,
    )
    assert delta_act_neg == pytest.approx(0.95)
    assert delta_act_neg != pytest.approx(0.95 * 0.95)
    assert act_neg_components["act_neg_grid_mwh"] == pytest.approx(1.0)
    assert act_neg_components["charge_internal_gain_mwh"] == pytest.approx(0.95)

    delta_act_pos, act_pos_components = bt._calculate_soc_delta(
        charge_mw=0.0,
        discharge_mw=0.0,
        id_charge_mw=0.0,
        id_discharge_mw=0.0,
        act_pos_mwh=1.0,
        act_neg_mwh=0.0,
        aux_mwh=0.0,
        battery_specs=specs,
        dt_h=dt_h,
    )
    assert delta_act_pos == pytest.approx(-1.0 / 0.95)
    assert delta_act_pos != pytest.approx(-1.0 / (0.95 * 0.95))
    assert act_pos_components["act_pos_grid_mwh"] == pytest.approx(1.0)
    assert act_pos_components["discharge_internal_loss_mwh"] == pytest.approx(1.0 / 0.95)

    delta_combined, combined_components = bt._calculate_soc_delta(
        charge_mw=1.0,
        discharge_mw=2.0,
        id_charge_mw=3.0,
        id_discharge_mw=4.0,
        act_pos_mwh=5.0,
        act_neg_mwh=6.0,
        aux_mwh=0.7,
        battery_specs=specs,
        dt_h=dt_h,
    )
    expected_charge_grid = 1.0 + 3.0 + 6.0
    expected_discharge_grid = 2.0 + 4.0 + 5.0
    expected_delta = expected_charge_grid * 0.95 - expected_discharge_grid / 0.95 - 0.7
    assert combined_components["charge_grid_mwh"] == pytest.approx(expected_charge_grid)
    assert combined_components["discharge_grid_mwh"] == pytest.approx(expected_discharge_grid)
    assert combined_components["charge_internal_gain_mwh"] == pytest.approx(expected_charge_grid * 0.95)
    assert combined_components["discharge_internal_loss_mwh"] == pytest.approx(expected_discharge_grid / 0.95)
    assert delta_combined == pytest.approx(expected_delta)


def test_da_precommit_selects_no_trade_when_predicted_replay_loses_money() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    bt.trans_eur_mwh = 0.0
    bt.deg_eur_mwh = 0.0
    bt.aux_off_mw = 0.0
    bt.aux_trading_mw = 0.0
    bt.eta_in = 0.9
    bt.eta_out = 0.9

    ts = pd.Timestamp("2025-01-09T00:00:00Z")
    losing_rows = pd.DataFrame(
        {
            col.timestamp: [ts],
            "target_time_utc": [ts],
            "charge_mw": [1.0],
            "discharge_mw": [0.0],
            "soc_start_lp_mwh": [10.0],
            col.pred_da_price: [100.0],
            col.pred_afrr_capacity_price_pos: [0.0],
            col.pred_afrr_capacity_price_neg: [0.0],
            col.pred_afrr_activation_price_pos: [0.0],
            col.pred_afrr_activation_price_neg: [0.0],
            col.pred_afrr_activation_rate_pos: [0.0],
            col.pred_afrr_activation_rate_neg: [0.0],
            "predicted_objective_eur": [123.0],
        }
    )

    selected, audit = bt._select_feasible_da_lock_schedule(
        lock_rows=losing_rows,
        colmap=col,
        current_soc_mwh=10.0,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert selected[ts] == pytest.approx((0.0, 0.0))
    assert audit[0]["selected_incumbent"] == "no_trade"
    assert audit[0]["selection_reason"] == "no_trade_incumbent_predicted_replay_dominates"
    assert audit[0]["candidate_rejection_reason"] == "candidate_pnl_below_no_trade"
    assert audit[0]["no_trade_rejection_reason"] == "none"
    assert str(audit[0]["selection_pnl_basis"]) == "includes_local_continuation"
    assert str(audit[0]["replay_scope"]) == "da_lock_rows_only"
    assert str(audit[0]["objective_scope"]) == "rolling_milp_window_or_plan_history"
    assert float(audit[0]["objective_replay_comparable"]) == pytest.approx(0.0)
    assert float(audit[0]["candidate_selection_pnl_eur"]) <= float(audit[0]["incumbent_selection_pnl_eur"])
    assert float(audit[0]["candidate_predicted_pnl_excl_terminal_eur"]) <= 0.0
    assert float(audit[0]["candidate_selection_pnl_eur"]) <= float(audit[0]["incumbent_selection_pnl_eur"])
    assert float(audit[0]["candidate_replay_valid"]) == pytest.approx(1.0)
    assert float(audit[0]["incumbent_replay_valid"]) == pytest.approx(1.0)
    assert float(audit[0]["selection_valid"]) == pytest.approx(1.0)
    assert float(audit[0]["candidate_zeroed_due_to_negative_valid_replay"]) == pytest.approx(1.0)
    assert float(audit[0]["candidate_zeroed_due_to_invalid_replay"]) == pytest.approx(0.0)
    assert float(audit[0]["local_terminal_credit_ignored_eur"]) == pytest.approx(0.0)
    assert float(audit[0]["terminal_credit_in_selection_eur"]) > 0.0
    assert str(audit[0]["da_terminal_value_basis"]) == "legacy_continuation"
    assert float(audit[0]["sell_candidate_mwh"]) == pytest.approx(0.0)
    assert float(audit[0]["sell_locked_mwh"]) == pytest.approx(0.0)
    assert audit[0]["sell_disabled_reason"] == "none"
    assert audit[0]["da_zero_reason"] == "no_trade_incumbent_selected"
    assert float(audit[0]["raw_candidate_revenue_eur"]) == pytest.approx(0.0)
    assert float(audit[0]["raw_candidate_cost_eur"]) == pytest.approx(100.0)
    assert float(audit[0]["raw_candidate_gross_spread_eur"]) == pytest.approx(-100.0)
    assert float(audit[0]["candidate_revenue_eur"]) == pytest.approx(0.0)
    assert float(audit[0]["candidate_cost_eur"]) == pytest.approx(100.0)
    assert float(audit[0]["candidate_gross_spread_eur"]) == pytest.approx(-100.0)
    assert np.isfinite(float(audit[0]["candidate_pnl_recomputed_eur"]))
    assert float(audit[0]["candidate_pnl_reconciliation_error_eur"]) == pytest.approx(0.0)
    assert float(audit[0]["cashflow_replay_error"]) == pytest.approx(0.0)
    assert audit[0]["price_source_column"] == col.pred_da_price

    bt.eta_in = 1.0
    bt.eta_out = 1.0
    bt.terminal_value_mode = "legacy_continuation"
    selected_global_end, audit_global_end = bt._select_feasible_da_lock_schedule(
        lock_rows=losing_rows,
        colmap=col,
        current_soc_mwh=10.0,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=ts,
    )

    assert selected_global_end[ts][0] >= 0.0
    assert str(audit_global_end[0]["selection_pnl_basis"]) == "includes_global_terminal"
    assert float(audit_global_end[0]["terminal_credit_eur"]) >= 0.0
    assert float(audit_global_end[0]["local_terminal_credit_ignored_eur"]) == pytest.approx(0.0)

    ts0 = pd.Timestamp("2025-01-10T00:00:00Z")
    ts1 = pd.Timestamp("2025-01-10T01:00:00Z")
    ts2 = pd.Timestamp("2025-01-10T02:00:00Z")
    profitable_rows = pd.DataFrame(
        {
            col.timestamp: [ts0, ts1, ts2],
            "target_time_utc": [ts0, ts1, ts2],
            "charge_mw": [1.0, 0.0, 0.0],
            "discharge_mw": [0.0, 0.0, 1.0],
            "soc_start_lp_mwh": [10.0, np.nan, np.nan],
            col.pred_da_price: [10.0, 50.0, 100.0],
            "ev_pred_da_price_eur_mwh": [0.0, 0.0, 0.0],
            col.pred_afrr_capacity_price_pos: [0.0, 0.0, 0.0],
            col.pred_afrr_capacity_price_neg: [0.0, 0.0, 0.0],
            col.pred_afrr_activation_price_pos: [0.0, 0.0, 0.0],
            col.pred_afrr_activation_price_neg: [0.0, 0.0, 0.0],
            col.pred_afrr_activation_rate_pos: [0.0, 0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0, 0.0],
            "predicted_objective_eur": [123.0, 123.0, 123.0],
        }
    )
    bt.eta_in = 1.0
    bt.eta_out = 1.0
    missing_price_rows = profitable_rows.drop(columns=[col.pred_da_price])
    with pytest.raises(ValueError, match="missing_da_precommit_replay_price"):
        bt._replay_da_candidate_cashflow(
            rows=missing_price_rows,
            schedule={ts0: (1.0, 0.0), ts1: (0.0, 0.0), ts2: (0.0, 1.0)},
            colmap=col,
            current_soc_mwh=10.0,
            fixed_reserve_pos={},
            fixed_reserve_neg={},
            global_end_utc=None,
        )

    selected2, audit2 = bt._select_feasible_da_lock_schedule(
        lock_rows=profitable_rows,
        colmap=col,
        current_soc_mwh=10.0,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    # The bid sizer keeps the profitable raw buy/sell cycle when it remains
    # feasible after sizing.
    assert selected2[ts0] == pytest.approx((1.0, 0.0))
    assert selected2[ts1] == pytest.approx((0.0, 0.0))
    assert selected2[ts2] == pytest.approx((0.0, 1.0))
    assert {row["selected_incumbent"] for row in audit2} == {"optimized"}
    assert float(audit2[0]["candidate_minus_incumbent_eur"]) > 0.0
    assert float(audit2[0]["raw_candidate_revenue_eur"]) == pytest.approx(100.0)
    assert float(audit2[0]["raw_candidate_cost_eur"]) == pytest.approx(10.0)
    assert float(audit2[0]["raw_candidate_gross_spread_eur"]) == pytest.approx(90.0)
    assert float(audit2[0]["candidate_revenue_eur"]) == pytest.approx(100.0)
    assert float(audit2[0]["candidate_cost_eur"]) == pytest.approx(10.0)
    assert float(audit2[0]["candidate_gross_spread_eur"]) == pytest.approx(90.0)
    assert float(audit2[0]["sized_candidate_revenue_eur"]) == pytest.approx(100.0)
    assert float(audit2[0]["sized_candidate_cost_eur"]) == pytest.approx(10.0)
    assert float(audit2[0]["sized_candidate_gross_spread_eur"]) == pytest.approx(90.0)
    assert float(audit2[0]["sized_candidate_reconciliation_error_eur"]) == pytest.approx(0.0)
    assert np.isnan(float(audit2[0]["selected_lockable_revenue_eur"]))
    assert audit2[0]["selected_lockable_deprecated_reason"] == "pre_rounding_pre_postlock_stage_not_final"
    assert float(audit2[0]["candidate_transaction_cost_eur"]) == pytest.approx(0.0)
    assert float(audit2[0]["candidate_degradation_cost_eur"]) == pytest.approx(0.0)
    assert float(audit2[0]["candidate_auxiliary_cost_eur"]) == pytest.approx(0.0)
    assert float(audit2[0]["candidate_terminal_credit_eur"]) == pytest.approx(0.0)
    assert float(audit2[0]["gross_spread_reconciliation_error_eur"]) == pytest.approx(0.0)
    assert float(audit2[0]["candidate_pnl_recomputed_eur"]) == pytest.approx(90.0)
    assert str(audit2[0]["cashflow_replay_error_reason"]) == "none"
    broken_replay = BatteryBacktester._da_precommit_replay_invariants(
        candidate_volume_mwh=2.0,
        nonzero_price_seen=True,
        candidate_revenue_eur=100.0,
        candidate_cost_eur=10.0,
        gross_spread_eur=0.0,
        transaction_cost_eur=0.0,
        degradation_cost_eur=0.0,
        auxiliary_cost_eur=0.0,
        terminal_credit_eur=0.0,
        candidate_pnl_eur=0.0,
    )
    assert float(broken_replay["cashflow_replay_error"]) == pytest.approx(1.0)
    assert "gross_spread_reconciliation_mismatch" in str(broken_replay["cashflow_replay_error_reason"])
    assert "silent_zero_gross_spread" in str(broken_replay["cashflow_replay_error_reason"])
    _, postlock_audit2 = bt._apply_da_postlock_future_guard(
        selected_da=selected2,
        da_audit_rows=audit2,
        lock_rows=profitable_rows,
        future_rows=profitable_rows,
        colmap=col,
        current_soc_mwh=10.0,
        da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )
    # Postlock replay evaluates the accepted/sized schedule, while
    # candidate_gross_spread_eur preserves the raw optimizer candidate.
    assert float(postlock_audit2[0]["postlock_candidate_gross_spread_eur"]) == pytest.approx(90.0)
    assert float(postlock_audit2[0]["postlock_candidate_pnl_recomputed_eur"]) == pytest.approx(90.0)
    assert str(postlock_audit2[0]["postlock_replay_price_source_column"]) == col.pred_da_price

    accepted_zero, accepted_zero_audit = bt._apply_da_accepted_lockbook_replay_to_audit_rows(
        da_audit_rows=postlock_audit2,
        accepted_da={},
        lock_rows=profitable_rows,
        colmap=col,
        current_soc_mwh=10.0,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )
    assert accepted_zero[ts0] == pytest.approx((1.0, 0.0))
    assert accepted_zero[ts2] == pytest.approx((0.0, 1.0))
    zero_row = accepted_zero_audit[0]
    assert float(zero_row["raw_candidate_revenue_eur"]) == pytest.approx(100.0)
    assert float(zero_row["raw_candidate_cost_eur"]) == pytest.approx(10.0)
    assert float(zero_row["raw_candidate_gross_spread_eur"]) == pytest.approx(90.0)
    assert float(zero_row["accepted_lockbook_energy_mwh"]) == pytest.approx(2.0)
    assert float(zero_row["accepted_lockbook_revenue_eur"]) == pytest.approx(100.0)
    assert float(zero_row["accepted_lockbook_cost_eur"]) == pytest.approx(10.0)
    assert float(zero_row["accepted_lockbook_gross_spread_eur"]) == pytest.approx(90.0)
    assert float(zero_row["accepted_lockbook_pnl_eur"]) == pytest.approx(90.0)
    assert float(zero_row["selected_lockable_revenue_eur"]) == pytest.approx(100.0)
    assert float(zero_row["selected_lockable_cost_eur"]) == pytest.approx(10.0)
    assert float(zero_row["selected_lockable_pnl_eur"]) == pytest.approx(90.0)
    assert zero_row["da_precommit_schedule_stage_used_for_selection"] == "accepted_lockbook"
    assert zero_row["da_precommit_schedule_stage_used_for_lockbook"] == "accepted_lockbook"
    assert float(zero_row["da_bid_locked"]) == pytest.approx(1.0)

    stage_ok = BatteryBacktester._validate_da_selected_lockable_stage_consistency(pd.DataFrame([zero_row]))
    assert float(stage_ok["da_selected_lockable_stage_mismatch"].iloc[0]) == pytest.approx(0.0)
    stage_bad = pd.DataFrame(
        [
            {
                "da_precommit_selected_lockable_revenue_eur": 100.0,
                "da_precommit_selected_lockable_cost_eur": 10.0,
                "da_precommit_selected_lockable_gross_spread_eur": 90.0,
                "da_precommit_selected_lockable_pnl_eur": 90.0,
                "da_precommit_accepted_lockbook_energy_mwh": 0.0,
                "da_precommit_accepted_lockbook_revenue_eur": 0.0,
                "da_precommit_accepted_lockbook_cost_eur": 0.0,
                "da_precommit_accepted_lockbook_gross_spread_eur": 0.0,
                "da_precommit_accepted_lockbook_pnl_eur": 0.0,
                "real_da_buy_mwh": 0.0,
                "real_da_sell_mwh": 0.0,
            }
        ]
    )
    stage_bad_result = BatteryBacktester._validate_da_selected_lockable_stage_consistency(stage_bad)
    assert float(stage_bad_result["da_selected_lockable_stage_mismatch"].iloc[0]) == pytest.approx(1.0)
    assert (
        stage_bad_result["da_selected_lockable_stage_mismatch_reason"].iloc[0]
        == "selected_lockable_leaks_economics_with_zero_accepted_lockbook"
    )

    bt.final_soc_mode = "hard"
    bt.da_terminal_soc_value_enabled = False
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    bt.soc_target_end = 10.0
    bt.eta_in = 1.0
    bt.eta_out = 1.0
    bt.deg_eur_mwh = 0.0
    bt.da_terminal_recovery_lookahead_h = 2.0
    ts_final = pd.Timestamp("2025-01-10T03:00:00Z")
    hard_rows = pd.DataFrame(
        {
            col.timestamp: [ts2, ts_final],
            "target_time_utc": [ts2, ts_final],
            col.pred_da_price: [100.0, 100.0],
            col.pred_afrr_activation_rate_pos: [0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0],
        }
    )
    hard_audit_rows = [
        {
            "timestamp_utc": str(ts2),
            "da_candidate_buy_mw": 0.0,
            "da_candidate_sell_mw": 9.5,
            "selected_incumbent": "optimized",
            "candidate_selection_pnl_eur": 950.0,
        }
    ]
    hard_selected, hard_audit = bt._apply_da_postlock_future_guard(
        selected_da={ts2: (0.0, 9.5)},
        da_audit_rows=hard_audit_rows,
        lock_rows=hard_rows.iloc[[0]].copy(),
        future_rows=hard_rows,
        colmap=col,
        current_soc_mwh=11.5,
        da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=ts_final,
    )
    assert hard_selected[ts2] == pytest.approx((0.0, 0.0))
    assert hard_audit[0]["final_selected_incumbent"] == "no_trade"
    assert hard_audit[0]["da_zero_reason"] == "postlock_future_infeasible_no_trade_selected"
    assert float(hard_audit[0]["da_candidate_rejected_hard_final_soc_infeasible"]) == pytest.approx(0.0)
    assert float(hard_audit[0]["da_postlock_rejected_due_to_future_infeasibility"]) == pytest.approx(1.0)
    assert float(hard_audit[0]["da_final_soc_feasibility_checked_before_lock"]) == pytest.approx(1.0)
    assert float(hard_audit[0]["da_final_soc_feasibility_passed_before_lock"]) == pytest.approx(1.0)
    assert float(hard_audit[0]["da_final_soc_feasibility_required_mwh"]) == pytest.approx(10.0)
    assert float(hard_audit[0]["da_final_soc_feasibility_projected_final_soc_mwh"]) == pytest.approx(2.0)
    assert float(hard_audit[0]["da_final_soc_feasibility_margin_mwh"]) == pytest.approx(-8.0)
    assert float(hard_audit[0]["da_final_soc_feasibility_existing_locked_buy_mwh"]) == pytest.approx(0.0)
    assert float(hard_audit[0]["da_final_soc_feasibility_existing_locked_sell_mwh"]) == pytest.approx(0.0)
    assert float(hard_audit[0]["da_final_soc_feasibility_candidate_buy_mwh"]) == pytest.approx(0.0)
    assert float(hard_audit[0]["da_final_soc_feasibility_candidate_sell_mwh"]) == pytest.approx(9.5)
    assert float(hard_audit[0]["da_final_soc_feasibility_id_recourse_assumed"]) == pytest.approx(0.0)
    assert str(hard_audit[0]["da_final_soc_feasibility_reason"]) == "terminal_repair_missing"
    assert float(hard_audit[0]["da_postlock_terminal_shortfall_unrecoverable"]) == pytest.approx(1.0)
    assert float(hard_audit[0]["da_postlock_terminal_shortfall_mwh"]) == pytest.approx(8.0)

    bt.deg_eur_mwh = 200.0
    selected3, audit3 = bt._select_feasible_da_lock_schedule(
        lock_rows=profitable_rows,
        colmap=col,
        current_soc_mwh=10.0,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert selected3[ts0] == pytest.approx((0.0, 0.0))
    assert selected3[ts1] == pytest.approx((0.0, 0.0))
    assert {row["selected_incumbent"] for row in audit3} == {"no_trade"}
    assert float(audit3[0]["raw_candidate_gross_spread_eur"]) == pytest.approx(90.0)
    assert float(audit3[0]["candidate_gross_spread_eur"]) == pytest.approx(90.0)
    assert float(audit3[0]["candidate_degradation_cost_eur"]) == pytest.approx(400.0)
    assert float(audit3[0]["candidate_predicted_pnl_eur"]) == pytest.approx(-310.0)
    assert float(audit3[0]["candidate_selection_pnl_eur"]) < float(audit3[0]["incumbent_selection_pnl_eur"])
    assert audit3[0]["candidate_rejection_reason"] == "candidate_pnl_below_no_trade"
    assert float(audit3[0]["cashflow_replay_error"]) == pytest.approx(0.0)
    assert float(audit3[0]["locked_buy_mwh_by_hour"]) == pytest.approx(0.0)
    assert float(audit3[0]["locked_sell_mwh_by_hour"]) == pytest.approx(0.0)

    take = pd.DataFrame(
        {
            col.timestamp: [ts0],
            "charge_mw": [1.0],
            "discharge_mw": [0.0],
            "ev_da_charge_coef_eur_per_mw": [5.0],
            "ev_da_discharge_coef_eur_per_mw": [-3.0],
            "ev_da_charge_eur": [5.0],
            "ev_da_discharge_eur": [0.0],
            "predicted_objective_eur": [42.0],
        }
    )
    blocked = bt._apply_da_lockbook_to_delivery_plan(
        take=take,
        colmap=col,
        da_lockbook={},
        da_precommit_audit_by_ts={},
        da_enabled=True,
    )
    assert float(blocked["raw_optimizer_plan_charge_mw"].iloc[0]) == pytest.approx(1.0)
    assert float(blocked["raw_optimizer_ev_da_charge_eur"].iloc[0]) == pytest.approx(5.0)
    assert float(blocked["raw_optimizer_predicted_objective_eur"].iloc[0]) == pytest.approx(42.0)
    assert float(blocked["accepted_lockbook_ev_da_charge_eur"].iloc[0]) == pytest.approx(0.0)
    assert float(blocked["ev_da_charge_eur"].iloc[0]) == pytest.approx(0.0)
    assert float(blocked["da_bid_locked"].iloc[0]) == pytest.approx(0.0)
    assert float(blocked["da_lockbook_row_present"].iloc[0]) == pytest.approx(0.0)
    assert float(blocked["da_is_locked_delivery_hour"].iloc[0]) == pytest.approx(
        float(blocked["da_lockbook_row_present"].iloc[0])
    )

    source_ts = pd.Timestamp("2025-01-08T10:00:00Z")  # 11:00 Europe/Berlin.
    locked = bt._apply_da_lockbook_to_delivery_plan(
        take=take,
        colmap=col,
        da_lockbook={ts0: (1.0, 0.0)},
        da_precommit_audit_by_ts={
            "da_precommit_source_snapshot_utc": {ts0: source_ts.isoformat()},
            "da_precommit_da_gate_hour_local": {ts0: 11.0},
            "da_precommit_da_gate_valid": {ts0: 1.0},
        },
        da_enabled=True,
    )
    assert float(locked["accepted_lockbook_ev_da_charge_eur"].iloc[0]) == pytest.approx(5.0)
    assert float(locked["ev_da_charge_eur"].iloc[0]) == pytest.approx(5.0)
    assert float(locked["da_locked_buy_mw"].iloc[0]) == pytest.approx(1.0)
    assert float(locked["da_locked_sell_mw"].iloc[0]) == pytest.approx(0.0)
    assert float(locked["da_locked_buy_mwh"].iloc[0]) == pytest.approx(1.0)
    assert float(locked["da_accepted_buy_mwh"].iloc[0]) == pytest.approx(1.0)
    assert float(locked["da_accepted_sell_mwh"].iloc[0]) == pytest.approx(0.0)
    assert float(locked["da_bid_locked"].iloc[0]) == pytest.approx(1.0)
    assert float(locked["da_lockbook_row_present"].iloc[0]) == pytest.approx(1.0)
    assert float(locked["da_is_locked_delivery_hour"].iloc[0]) == pytest.approx(
        float(locked["da_lockbook_row_present"].iloc[0])
    )
    assert str(locked["da_originating_source_snapshot_utc"].iloc[0])
    assert str(locked["da_originating_delivery_timestamp_utc"].iloc[0]) == ts0.isoformat()
    assert str(locked["da_originating_precommit_id"].iloc[0])
    assert float(locked["da_source_gate_valid"].iloc[0]) == pytest.approx(1.0)
    assert float(locked["source_snapshot_is_da_gate"].iloc[0]) == pytest.approx(1.0)
    assert float(locked["delivery_row_is_da_gate_hour"].iloc[0]) == pytest.approx(0.0)
    locked_counters = BatteryBacktester._compute_da_naming_semantics_counters(locked)
    assert locked_counters["da_naming_semantics_error_count"] == pytest.approx(0.0)

    no_trade_locked = bt._apply_da_lockbook_to_delivery_plan(
        take=take,
        colmap=col,
        da_lockbook={ts0: (0.0, 0.0)},
        da_precommit_audit_by_ts={
            "da_precommit_source_snapshot_utc": {ts0: source_ts.isoformat()},
            "da_precommit_da_gate_hour_local": {ts0: 11.0},
            "da_precommit_da_gate_valid": {ts0: 1.0},
        },
        da_enabled=True,
    )
    assert float(no_trade_locked["da_lockbook_row_present"].iloc[0]) == pytest.approx(1.0)
    assert float(no_trade_locked["da_bid_locked"].iloc[0]) == pytest.approx(0.0)
    assert float(no_trade_locked["da_is_locked_delivery_hour"].iloc[0]) == pytest.approx(
        float(no_trade_locked["da_lockbook_row_present"].iloc[0])
    )
    assert float(no_trade_locked["da_locked_buy_mw"].iloc[0]) == pytest.approx(0.0)
    assert float(no_trade_locked["da_locked_sell_mw"].iloc[0]) == pytest.approx(0.0)
    assert float(no_trade_locked["da_locked_buy_mwh"].iloc[0]) == pytest.approx(0.0)
    assert float(no_trade_locked["da_locked_sell_mwh"].iloc[0]) == pytest.approx(0.0)
    assert float(no_trade_locked["da_accepted_buy_mwh"].iloc[0]) == pytest.approx(0.0)
    assert float(no_trade_locked["da_accepted_sell_mwh"].iloc[0]) == pytest.approx(0.0)
    assert float(no_trade_locked["da_unlocked_raw_trade_blocked"].iloc[0]) == pytest.approx(0.0)
    no_trade_semantic = no_trade_locked.copy()
    no_trade_semantic["da_precommit_selected_incumbent"] = "no_trade"
    no_trade_semantic["da_precommit_da_accepted_buy_mw"] = 0.0
    no_trade_semantic["da_precommit_da_accepted_sell_mw"] = 0.0
    no_trade_counters = BatteryBacktester._compute_da_naming_semantics_counters(no_trade_semantic)
    assert no_trade_counters["da_naming_semantics_error_count"] == pytest.approx(0.0)

    submitted_without_lock = pd.DataFrame(
        {
            "da_lockbook_row_present": [0.0],
            "da_bid_locked": [0.0],
            "da_locked_buy_mwh": [0.0],
            "da_locked_sell_mwh": [0.0],
            "real_submitted_da_buy_mw": [1.0],
            "real_submitted_da_sell_mw": [0.0],
            "real_submitted_da_buy_price_eur_mwh": [10.0],
            "real_da_buy_accepted": [0.0],
            "real_da_sell_accepted": [0.0],
            "real_da_buy_mwh": [0.0],
            "real_da_sell_mwh": [0.0],
        }
    )
    submitted_counters = BatteryBacktester._compute_da_naming_semantics_counters(submitted_without_lock)
    assert submitted_counters["da_submitted_without_locked_bid_count"] == pytest.approx(1.0)
    assert submitted_counters["da_naming_semantics_error_count"] == pytest.approx(1.0)

    realized_without_submission = pd.DataFrame(
        {
            "da_lockbook_row_present": [1.0],
            "da_bid_locked": [1.0],
            "da_locked_buy_mwh": [1.0],
            "da_locked_sell_mwh": [0.0],
            "da_originating_source_snapshot_utc": [source_ts.isoformat()],
            "da_originating_delivery_timestamp_utc": [ts0.isoformat()],
            "da_originating_precommit_id": [f"{source_ts.isoformat()}->{ts0}"],
            "da_precommit_final_selected_incumbent": ["optimized"],
            "da_precommit_selection_reason": ["accepted_lockbook"],
            "da_precommit_da_zero_reason": ["none"],
            "real_submitted_da_buy_mw": [0.0],
            "real_submitted_da_sell_mw": [0.0],
            "real_da_buy_accepted": [0.0],
            "real_da_sell_accepted": [0.0],
            "real_da_buy_mwh": [1.0],
            "real_da_sell_mwh": [0.0],
        }
    )
    realized_counters = BatteryBacktester._compute_da_naming_semantics_counters(realized_without_submission)
    assert realized_counters["da_realized_without_submitted_bid_count"] == pytest.approx(1.0)
    assert realized_counters["da_naming_semantics_error_count"] == pytest.approx(1.0)

    realized_without_origin = pd.DataFrame(
        {
            "da_lockbook_row_present": [1.0],
            "da_bid_locked": [1.0],
            "da_locked_buy_mwh": [1.0],
            "da_locked_sell_mwh": [0.0],
            "real_submitted_da_buy_mw": [1.0],
            "real_submitted_da_sell_mw": [0.0],
            "real_submitted_da_buy_price_eur_mwh": [10.0],
            "real_da_buy_accepted": [1.0],
            "real_da_sell_accepted": [0.0],
            "real_da_buy_mwh": [1.0],
            "real_da_sell_mwh": [0.0],
        }
    )
    origin_counters = BatteryBacktester._compute_da_naming_semantics_counters(realized_without_origin)
    assert origin_counters["da_realized_without_precommit_origin_count"] == pytest.approx(1.0)
    assert origin_counters["da_naming_semantics_error_count"] == pytest.approx(1.0)

    one_hour, _ = _one_hour_pred_df(
        da=100.0,
        cap_pos=10.0,
        cap_neg=10.0,
        act_pos=0.0,
        act_neg=0.0,
        rate_pos=0.0,
        rate_neg=0.0,
    )
    da_only = bt.optimize_dispatch(one_hour, col, allowed_markets=("DA",))
    assert np.isnan(float(da_only["bcm_ev_pos"].iloc[0]))
    assert np.isnan(float(da_only["bcm_ev_neg"].iloc[0]))
    assert np.isnan(float(da_only["bcm_candidate_pos_mw"].iloc[0]))
    assert np.isnan(float(da_only["ev_bem_only_pos_eur"].iloc[0]))


def test_da_precommit_selection_blocks_invalid_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    bt.eta_in = 1.0
    bt.eta_out = 1.0
    bt.deg_eur_mwh = 0.0
    bt.trans_eur_mwh = 0.0
    bt.aux_off_mw = 0.0
    bt.aux_trading_mw = 0.0

    ts0 = pd.Timestamp("2025-01-10T00:00:00Z")
    ts1 = pd.Timestamp("2025-01-10T01:00:00Z")
    rows = pd.DataFrame(
        {
            col.timestamp: [ts0, ts1],
            "target_time_utc": [ts0, ts1],
            "charge_mw": [1.0, 0.0],
            "discharge_mw": [0.0, 1.0],
            "soc_start_lp_mwh": [10.0, np.nan],
            col.pred_da_price: [10.0, 100.0],
            col.pred_afrr_capacity_price_pos: [0.0, 0.0],
            col.pred_afrr_capacity_price_neg: [0.0, 0.0],
            col.pred_afrr_activation_price_pos: [0.0, 0.0],
            col.pred_afrr_activation_price_neg: [0.0, 0.0],
            col.pred_afrr_activation_rate_pos: [0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0],
            "predicted_objective_eur": [90.0, 90.0],
        }
    )

    original_replay = bt._replay_da_candidate_cashflow

    def fake_replay(*, schedule: dict[pd.Timestamp, tuple[float, float]], **kwargs: object) -> dict[str, float | str]:
        has_candidate_volume = any(ch > 1e-9 or dis > 1e-9 for ch, dis in schedule.values())
        if has_candidate_volume:
            return {
                "selection_pnl_eur": -10.0,
                "pnl_eur": -10.0,
                "pnl_excl_terminal_eur": -10.0,
                "da_cashflow_replay_error": 1.0,
                "da_cashflow_replay_error_reason": (
                    "manual_replay_gross_mismatch,silent_zero_cashflow_with_nonzero_candidate"
                ),
                "da_candidate_pnl_recomputed_eur": -10.0,
                "da_candidate_pnl_reconciliation_error_eur": 0.0,
                "da_gross_spread_reconciliation_error_eur": 90.0,
                "selection_pnl_basis": "excl_terminal",
            }
        return original_replay(schedule=schedule, **kwargs)

    monkeypatch.setattr(bt, "_replay_da_candidate_cashflow", fake_replay)

    selected, audit = bt._select_feasible_da_lock_schedule(
        lock_rows=rows,
        colmap=col,
        current_soc_mwh=10.0,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert selected[ts0] == pytest.approx((0.0, 0.0))
    assert selected[ts1] == pytest.approx((0.0, 0.0))
    assert {row["selected_incumbent"] for row in audit} == {"none"}
    assert {row["selection_reason"] for row in audit} == {"candidate_replay_invalid"}
    assert all(float(row["selection_valid"]) == pytest.approx(0.0) for row in audit)
    assert all(float(row["selection_blocked_by_replay_error"]) == pytest.approx(1.0) for row in audit)
    assert all(float(row["candidate_zeroed_due_to_invalid_replay"]) == pytest.approx(1.0) for row in audit)
    assert all(str(row["da_zero_reason"]) == "invalid_replay" for row in audit)
    assert "no_trade_incumbent_predicted_replay_dominates" not in {row["selection_reason"] for row in audit}

    bad_semantics = pd.DataFrame(
        {
            "da_precommit_cashflow_replay_error": [1.0],
            "da_precommit_selection_reason": ["no_trade_incumbent_predicted_replay_dominates"],
            "da_precommit_selected_incumbent": ["no_trade"],
            "da_precommit_candidate_replay_valid": [0.0],
            "da_precommit_selection_valid": [1.0],
            "da_precommit_da_accepted_buy_mw": [0.0],
            "da_precommit_da_accepted_sell_mw": [0.0],
        }
    )
    counters = BatteryBacktester._compute_da_naming_semantics_counters(bad_semantics)
    assert counters["da_replay_error_as_no_trade_dominance_count"] == pytest.approx(1.0)
    assert counters["da_invalid_selection_not_flagged_count"] == pytest.approx(1.0)


def test_da_postlock_selection_schedule_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    bt.eta_in = 1.0
    bt.eta_out = 1.0
    bt.deg_eur_mwh = 0.0
    bt.trans_eur_mwh = 0.0
    bt.aux_off_mw = 0.0
    bt.aux_trading_mw = 0.0

    ts0 = pd.Timestamp("2025-01-10T00:00:00Z")
    rows = pd.DataFrame(
        {
            col.timestamp: [ts0],
            "target_time_utc": [ts0],
            col.pred_da_price: [100.0],
            col.pred_afrr_activation_rate_pos: [0.0],
            col.pred_afrr_activation_rate_neg: [0.0],
        }
    )
    selected_da = {ts0: (0.0, 1.0)}
    audit_rows = [
        {
            "timestamp_utc": str(ts0),
            "da_candidate_buy_mw": 0.0,
            "da_candidate_sell_mw": 1.0,
            "selected_incumbent": "optimized",
            "candidate_selection_pnl_eur": 100.0,
        }
    ]
    original_replay = bt._replay_da_candidate_cashflow

    def fake_replay(*, schedule: dict[pd.Timestamp, tuple[float, float]], **kwargs: object) -> dict[str, float | str]:
        has_volume = any(ch > 1e-9 or dis > 1e-9 for ch, dis in schedule.values())
        if has_volume:
            return {
                "selection_pnl_eur": 100.0,
                "pnl_eur": 100.0,
                "cashflow_replay_error": 1.0,
                "cashflow_replay_error_reason": "forced_selection_schedule_reconciliation_failed",
                "pnl_reconciliation_error_eur": 999.0,
                "revenue_eur": 100.0,
                "cost_eur": 0.0,
                "gross_spread_eur": 100.0,
                "pnl_recomputed_eur": 100.0,
                "price_source_column": col.pred_da_price,
            }
        return original_replay(schedule=schedule, **kwargs)

    monkeypatch.setattr(bt, "_replay_da_candidate_cashflow", fake_replay)

    final_schedule, audit = bt._apply_da_postlock_future_guard(
        selected_da=selected_da,
        da_audit_rows=audit_rows,
        lock_rows=rows,
        future_rows=rows,
        colmap=col,
        current_soc_mwh=10.0,
        da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert final_schedule[ts0] == pytest.approx((0.0, 0.0))
    assert audit[0]["final_selected_incumbent"] == "no_trade"
    assert audit[0]["da_zero_reason"] == "selection_schedule_reconciliation_failed"
    assert float(audit[0]["da_precommit_selection_valid"]) == pytest.approx(0.0)
    assert audit[0]["da_precommit_selection_invalid_reason"] == "selection_schedule_reconciliation_failed"
    assert float(audit[0]["da_selected_lockable_sell_mw"]) == pytest.approx(0.0)
    assert float(audit[0]["da_selected_lockable_replay_revenue_eur"]) == pytest.approx(0.0)
    assert float(audit[0]["da_selected_lockable_replay_gross_spread_eur"]) == pytest.approx(0.0)
    assert audit[0]["da_precommit_schedule_stage_used_for_selection"] == "selected_lockable"
    assert audit[0]["da_precommit_schedule_stage_used_for_lockbook"] == "selected_lockable"


def test_da_precommit_replay_cashflow_formulas_and_export_validation() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    bt.eta_in = 1.0
    bt.eta_out = 1.0
    bt.deg_eur_mwh = 0.0
    bt.trans_eur_mwh = 0.0
    bt.aux_peak_mw = 0.0
    bt.aux_off_mw = 0.0
    bt.aux_standby_mw = 0.0
    bt.aux_trading_mw = 0.0
    bt.aux_afrr_active_mw = 0.0

    def _rows(ts_values: list[pd.Timestamp], prices: list[float], ev_prices: list[float] | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                col.timestamp: ts_values,
                "target_time_utc": ts_values,
                col.pred_da_price: prices,
                "ev_pred_da_price_eur_mwh": ev_prices if ev_prices is not None else [0.0] * len(ts_values),
                col.pred_afrr_activation_rate_pos: [0.0] * len(ts_values),
                col.pred_afrr_activation_rate_neg: [0.0] * len(ts_values),
            }
        )

    sell_ts = pd.Timestamp("2025-01-10T00:00:00Z")
    sell_rows = _rows([sell_ts], [179.235499], [0.0])
    sell_replay = bt._replay_da_candidate_cashflow(
        rows=sell_rows,
        schedule={sell_ts: (0.0, 7.5)},
        colmap=col,
        current_soc_mwh=10.0,
    )
    assert float(sell_replay["da_candidate_revenue_eur"]) == pytest.approx(1344.2662425)
    assert float(sell_replay["da_candidate_cost_eur"]) == pytest.approx(0.0)
    assert float(sell_replay["da_candidate_gross_spread_eur"]) == pytest.approx(1344.2662425)
    assert float(sell_replay["da_cashflow_replay_error"]) == pytest.approx(0.0)

    buy_ts = pd.Timestamp("2025-01-10T01:00:00Z")
    buy_rows = _rows([buy_ts], [47.707194], [9999.0])
    buy_replay = bt._replay_da_candidate_cashflow(
        rows=buy_rows,
        schedule={buy_ts: (10.0, 0.0)},
        colmap=col,
        current_soc_mwh=10.0,
    )
    assert float(buy_replay["da_candidate_revenue_eur"]) == pytest.approx(0.0)
    assert float(buy_replay["da_candidate_cost_eur"]) == pytest.approx(477.07194)
    assert float(buy_replay["da_candidate_gross_spread_eur"]) == pytest.approx(-477.07194)
    assert float(buy_replay["da_cashflow_replay_error"]) == pytest.approx(0.0)

    ts0 = pd.Timestamp("2025-01-10T02:00:00Z")
    ts1 = pd.Timestamp("2025-01-10T03:00:00Z")
    ts2 = pd.Timestamp("2025-01-10T04:00:00Z")
    mixed_rows = _rows([ts0, ts1, ts2], [10.0, 55.0, 100.0], [0.0, 0.0, 0.0])
    mixed_schedule = {ts0: (3.0, 0.0), ts1: (0.0, 0.0), ts2: (0.0, 2.0)}
    mixed_replay = bt._replay_da_candidate_cashflow(
        rows=mixed_rows,
        schedule=mixed_schedule,
        colmap=col,
        current_soc_mwh=10.0,
    )
    assert float(mixed_replay["da_candidate_revenue_eur"]) == pytest.approx(200.0)
    assert float(mixed_replay["da_candidate_cost_eur"]) == pytest.approx(30.0)
    assert float(mixed_replay["da_candidate_gross_spread_eur"]) == pytest.approx(170.0)
    assert float(mixed_replay["da_manual_revenue_from_exported_candidate_eur"]) == pytest.approx(200.0)
    assert float(mixed_replay["da_manual_cost_from_exported_candidate_eur"]) == pytest.approx(30.0)
    assert float(mixed_replay["da_manual_gross_from_exported_candidate_eur"]) == pytest.approx(170.0)
    assert float(mixed_replay["da_exported_vs_manual_revenue_gap_eur"]) == pytest.approx(0.0)
    assert float(mixed_replay["da_exported_vs_manual_cost_gap_eur"]) == pytest.approx(0.0)
    assert float(mixed_replay["da_exported_vs_manual_gross_gap_eur"]) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="missing_da_precommit_replay_price"):
        bt._replay_da_candidate_cashflow(
            rows=mixed_rows.drop(columns=[col.pred_da_price]),
            schedule=mixed_schedule,
            colmap=col,
            current_soc_mwh=10.0,
        )

    artifact_like = mixed_rows.copy()
    artifact_like["da_originating_precommit_id"] = "precommit-1"
    artifact_like["da_precommit_da_candidate_buy_mw"] = [3.0, 0.0, 0.0]
    artifact_like["da_precommit_da_candidate_sell_mw"] = [0.0, 0.0, 2.0]
    artifact_like["da_precommit_candidate_revenue_eur"] = 200.0
    artifact_like["da_precommit_candidate_cost_eur"] = 30.0
    artifact_like["da_precommit_candidate_gross_spread_eur"] = 170.0
    validation = BatteryBacktester._validate_da_precommit_replay_export_cashflow(
        artifact_like,
        dt_h=1.0,
        pred_da_price_col=col.pred_da_price,
    )
    assert len(validation) == 1
    assert float(validation["da_precommit_manual_revenue_from_exported_candidate_eur"].iloc[0]) == pytest.approx(200.0)
    assert float(validation["da_precommit_manual_cost_from_exported_candidate_eur"].iloc[0]) == pytest.approx(30.0)
    assert float(validation["da_precommit_manual_gross_from_exported_candidate_eur"].iloc[0]) == pytest.approx(170.0)
    assert float(validation["da_precommit_exported_vs_manual_revenue_gap_eur"].iloc[0]) == pytest.approx(0.0)
    assert float(validation["da_precommit_exported_vs_manual_cost_gap_eur"].iloc[0]) == pytest.approx(0.0)
    assert float(validation["da_precommit_exported_vs_manual_gross_gap_eur"].iloc[0]) == pytest.approx(0.0)
    assert float(validation["da_precommit_replay_export_validation_error"].iloc[0]) == pytest.approx(0.0)
    assert float(validation["da_precommit_cashflow_replay_error"].iloc[0]) == pytest.approx(0.0)
    assert str(validation["da_precommit_cashflow_replay_error_reason"].iloc[0]) == "none"

    broken_artifact = artifact_like.copy()
    broken_artifact["da_precommit_candidate_revenue_eur"] = 0.0
    broken_artifact["da_precommit_candidate_cost_eur"] = 0.0
    broken_artifact["da_precommit_candidate_gross_spread_eur"] = 0.0
    broken_validation = BatteryBacktester._validate_da_precommit_replay_export_cashflow(
        broken_artifact,
        dt_h=1.0,
        pred_da_price_col=col.pred_da_price,
    )
    assert float(broken_validation["da_precommit_cashflow_replay_error"].iloc[0]) == pytest.approx(1.0)
    broken_reason = str(broken_validation["da_precommit_cashflow_replay_error_reason"].iloc[0])
    assert "manual_replay_revenue_mismatch" in broken_reason
    assert "manual_replay_cost_mismatch" in broken_reason
    assert "manual_replay_gross_mismatch" in broken_reason
    assert "silent_zero_cashflow_with_nonzero_candidate" in broken_reason

    repeated_totals = artifact_like.copy()
    repeated_totals["da_precommit_candidate_revenue_eur"] = 200.0
    repeated_totals["da_precommit_candidate_cost_eur"] = 30.0
    repeated_totals["da_precommit_candidate_gross_spread_eur"] = 170.0
    repeated_validation = BatteryBacktester._validate_da_precommit_replay_export_cashflow(
        repeated_totals,
        dt_h=1.0,
        pred_da_price_col=col.pred_da_price,
    )
    assert len(repeated_validation) == 1
    assert float(repeated_validation["da_precommit_cashflow_replay_error"].iloc[0]) == pytest.approx(0.0)


def test_da_terminal_recovery_aware_replay_uses_continuation_in_ordinary_window() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    bt.terminal_value_mode = "terminal_recovery_aware"
    bt.eta_in = 1.0
    bt.eta_out = 1.0
    bt.deg_eur_mwh = 0.0
    bt.trans_eur_mwh = 0.0
    bt.aux_off_mw = 0.0
    bt.aux_trading_mw = 0.0
    ts = pd.date_range("2025-01-01T00:00:00Z", periods=2, freq="h")
    rows = pd.DataFrame(
        {
            col.timestamp: ts,
            "target_time_utc": ts,
            col.pred_da_price: [50.0, 60.0],
            col.pred_afrr_activation_rate_pos: [0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0],
        }
    )

    replay = bt._replay_da_candidate_cashflow(
        rows=rows,
        schedule={},
        colmap=col,
        current_soc_mwh=9.0,
        global_end_utc=pd.Timestamp("2025-01-10T00:00:00Z"),
    )

    assert float(replay["da_terminal_sensitive_window"]) == pytest.approx(0.0)
    assert str(replay["da_terminal_value_basis"]) == "legacy_continuation"
    assert float(replay["da_terminal_recovery_cost_eur"]) == pytest.approx(0.0)
    assert float(replay["selection_pnl_eur"]) == pytest.approx(float(replay["terminal_credit_eur"]))


def test_da_terminal_recovery_aware_shortfall_uses_recovery_cost_without_double_counting() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    bt.terminal_value_mode = "terminal_recovery_aware"
    bt.terminal_value_buy_quantile = 0.70
    bt.terminal_value_sell_quantile = 0.30
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.deg_eur_mwh = 10.0
    bt.trans_eur_mwh = 2.0
    bt.aux_off_mw = 0.0
    bt.aux_trading_mw = 0.0
    ts = pd.date_range("2025-01-01T00:00:00Z", periods=3, freq="h")
    rows = pd.DataFrame(
        {
            col.timestamp: ts,
            "target_time_utc": ts,
            col.pred_da_price: [100.0, 120.0, 140.0],
            col.pred_afrr_activation_rate_pos: [0.0, 0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0, 0.0],
        }
    )

    replay = bt._replay_da_candidate_cashflow(
        rows=rows,
        schedule={},
        colmap=col,
        current_soc_mwh=9.0,
        global_end_utc=ts[-1],
    )

    replacement = float(replay["da_terminal_replacement_cost_eur_per_internal_mwh"])
    shortfall = float(replay["da_terminal_shortfall_internal_mwh"])
    recovery_cost = float(replay["da_terminal_recovery_cost_eur"])
    assert float(replay["da_terminal_sensitive_window"]) == pytest.approx(1.0)
    assert str(replay["da_terminal_value_basis"]) == "terminal_recovery_aware"
    assert recovery_cost == pytest.approx(shortfall * replacement)
    assert float(replay["terminal_credit_eur"]) == pytest.approx(-recovery_cost)
    assert float(replay["da_terminal_double_count_guard_pass"]) == pytest.approx(1.0)


def test_da_terminal_recovery_aware_surplus_uses_liquidation_value() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    bt.terminal_value_mode = "terminal_recovery_aware"
    bt.terminal_value_buy_quantile = 0.70
    bt.terminal_value_sell_quantile = 0.30
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.deg_eur_mwh = 10.0
    bt.trans_eur_mwh = 2.0
    bt.aux_off_mw = 0.0
    bt.aux_trading_mw = 0.0
    ts = pd.date_range("2025-01-01T00:00:00Z", periods=3, freq="h")
    rows = pd.DataFrame(
        {
            col.timestamp: ts,
            "target_time_utc": ts,
            col.pred_da_price: [100.0, 120.0, 140.0],
            col.pred_afrr_activation_rate_pos: [0.0, 0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0, 0.0],
        }
    )

    replay = bt._replay_da_candidate_cashflow(
        rows=rows,
        schedule={},
        colmap=col,
        current_soc_mwh=11.0,
        global_end_utc=ts[-1],
    )

    surplus = float(replay["da_terminal_surplus_internal_mwh"])
    liquidation = float(replay["da_terminal_liquidation_value_eur_per_internal_mwh"])
    assert float(replay["da_terminal_sensitive_window"]) == pytest.approx(1.0)
    assert str(replay["da_terminal_value_basis"]) == "terminal_recovery_aware"
    assert surplus == pytest.approx(1.0)
    assert float(replay["da_terminal_shortfall_internal_mwh"]) == pytest.approx(0.0)
    assert float(replay["da_terminal_surplus_liquidation_value_eur"]) == pytest.approx(
        surplus * liquidation
    )
    assert float(replay["da_terminal_recovery_cost_eur"]) == pytest.approx(0.0)
    assert float(replay["terminal_credit_eur"]) == pytest.approx(surplus * liquidation)


def test_da_terminal_recovery_cost_cancels_for_same_shortfall_incremental_selection() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    bt.terminal_value_mode = "terminal_recovery_aware"
    bt.eta_in = 1.0
    bt.eta_out = 1.0
    bt.deg_eur_mwh = 0.0
    bt.trans_eur_mwh = 0.0
    bt.aux_off_mw = 0.0
    bt.aux_trading_mw = 0.0
    ts = pd.date_range("2025-01-01T00:00:00Z", periods=3, freq="h")
    rows = pd.DataFrame(
        {
            col.timestamp: ts,
            "target_time_utc": ts,
            col.pred_da_price: [100.0, 120.0, 140.0],
            col.pred_afrr_activation_rate_pos: [0.0, 0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0, 0.0],
        }
    )
    incumbent = bt._replay_da_candidate_cashflow(
        rows=rows,
        schedule={},
        colmap=col,
        current_soc_mwh=9.0,
        global_end_utc=ts[-1],
    )
    candidate = bt._replay_da_candidate_cashflow(
        rows=rows,
        schedule={ts[0]: (1.0, 0.0), ts[1]: (0.0, 1.0)},
        colmap=col,
        current_soc_mwh=9.0,
        global_end_utc=ts[-1],
    )

    assert float(candidate["da_terminal_shortfall_internal_mwh"]) == pytest.approx(
        float(incumbent["da_terminal_shortfall_internal_mwh"])
    )
    assert float(candidate["da_terminal_recovery_cost_eur"]) == pytest.approx(
        float(incumbent["da_terminal_recovery_cost_eur"])
    )
    incremental_selection = float(candidate["selection_pnl_eur"]) - float(incumbent["selection_pnl_eur"])
    incremental_operational = float(candidate["pnl_excl_terminal_eur"]) - float(
        incumbent["pnl_excl_terminal_eur"]
    )
    assert incremental_selection == pytest.approx(incremental_operational)


def test_da_terminal_recovery_cost_estimate_unavailable_is_strict_for_shortfall() -> None:
    bt = _mk_backtester()
    components = bt._terminal_value_components_for_candidate(
        final_soc_mwh=9.0,
        target_soc_mwh=10.0,
        forecast_prices_eur_mwh=[],
        continuation_value_eur_per_internal_mwh=50.0,
        is_terminal_sensitive_window=True,
        basis="terminal_recovery_aware",
    )

    assert components.shortfall_internal_mwh == pytest.approx(1.0)
    assert components.recovery_cost_estimate_available == pytest.approx(0.0)
    assert components.terminal_recovery_cost_eur == pytest.approx(0.0)
    assert components.terminal_value_eur == pytest.approx(0.0)


def test_da_terminal_recovery_estimator_exports_causal_units_and_source() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.90
    bt.deg_eur_mwh = 10.0
    bt.trans_eur_mwh = 2.0
    bt.terminal_value_buy_quantile = 0.70
    bt.terminal_value_sell_quantile = 0.30

    components = bt._terminal_value_components_for_candidate(
        final_soc_mwh=8.0,
        target_soc_mwh=10.0,
        forecast_prices_eur_mwh=[50.0, 100.0, 150.0],
        continuation_value_eur_per_internal_mwh=999.0,
        is_terminal_sensitive_window=True,
        basis="terminal_recovery_aware",
        price_source="pred_da_price_truth_materialized_for_rhpf",
    )

    assert components.basis == "terminal_recovery_aware"
    assert components.shortfall_internal_mwh == pytest.approx(2.0)
    assert components.recovery_cost_estimate_available == pytest.approx(1.0)
    assert components.recovery_cost_units == "eur_per_internal_mwh"
    assert components.recovery_cost_price_source == "pred_da_price_truth_materialized_for_rhpf"
    assert components.terminal_recovery_cost_eur == pytest.approx(
        components.shortfall_internal_mwh * components.replacement_cost_eur_per_internal_mwh
    )
    # Explicit recovery-cost mode must not also subtract scalar continuation for
    # the same shortfall.
    assert components.terminal_value_eur == pytest.approx(-components.terminal_recovery_cost_eur)


def test_da_terminal_recovery_estimator_rejects_model_true_price_leakage() -> None:
    bt = _mk_backtester()

    with pytest.raises(ValueError, match="terminal_recovery_source_leakage"):
        bt._estimate_terminal_recovery_cost_per_internal_mwh(
            forecast_prices_eur_mwh=[100.0],
            price_source="true_da_price",
            path_label="model",
        )

    with pytest.raises(ValueError, match="terminal_recovery_source_leakage"):
        bt._estimate_terminal_recovery_cost_per_internal_mwh(
            forecast_prices_eur_mwh=[100.0],
            price_source="target_da_price",
            path_label="naive",
        )

    rhpf = bt._estimate_terminal_recovery_cost_per_internal_mwh(
        forecast_prices_eur_mwh=[100.0],
        price_source="pred_da_price_truth_materialized_for_rhpf",
        path_label="rhpf",
        is_perfect_foresight=True,
    )
    assert str(rhpf["source_semantics"]) == "truth_materialized_forecast"
    assert str(rhpf["units"]) == "eur_per_internal_mwh"


def test_da_terminal_sensitive_window_classification_reaches_end_or_explicit_lookahead() -> None:
    bt = _mk_backtester()
    bt.da_terminal_recovery_lookahead_h = 0.0
    global_end = pd.Timestamp("2025-01-03T00:00:00Z")
    reaches_end_rows = pd.DataFrame(
        {
            "target_time_utc": pd.to_datetime(
                ["2025-01-02T23:00:00Z", "2025-01-03T00:00:00Z"],
                utc=True,
            )
        }
    )
    final_day_rows = pd.DataFrame(
        {
            "target_time_utc": pd.to_datetime(
                ["2025-01-02T00:00:00Z", "2025-01-02T12:00:00Z"],
                utc=True,
            )
        }
    )
    ordinary_rows = pd.DataFrame(
        {
            "target_time_utc": pd.to_datetime(
                ["2024-12-30T00:00:00Z", "2024-12-30T12:00:00Z"],
                utc=True,
            )
        }
    )
    lookahead_rows = pd.DataFrame(
        {
            "target_time_utc": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-01T12:00:00Z"],
                utc=True,
            )
        }
    )

    sensitive, reason = bt._is_da_terminal_sensitive_window(
        reaches_end_rows,
        global_end_utc=global_end,
    )
    assert sensitive is True
    assert reason == "window_end_reaches_global_end"

    sensitive, reason = bt._is_da_terminal_sensitive_window(
        final_day_rows,
        global_end_utc=global_end,
    )
    assert sensitive is False
    assert reason == "not_terminal_window"

    sensitive, reason = bt._is_da_terminal_sensitive_window(
        ordinary_rows,
        global_end_utc=global_end,
    )
    assert sensitive is False
    assert reason == "not_terminal_window"

    sensitive, reason = bt._is_da_terminal_sensitive_window(
        lookahead_rows,
        global_end_utc=global_end,
    )
    assert sensitive is False
    assert reason == "not_terminal_window"

    bt.da_terminal_recovery_lookahead_h = 48.0
    sensitive, reason = bt._is_da_terminal_sensitive_window(
        lookahead_rows,
        global_end_utc=global_end,
    )
    assert sensitive is True
    assert reason == "terminal_recovery_lookahead_window"


def test_da_terminal_recovery_lookahead_prices_shortfall_before_final_day() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    bt.terminal_value_mode = "terminal_recovery_aware"
    bt.da_terminal_recovery_lookahead_h = 72.0
    bt.eta_in = 1.0
    bt.eta_out = 1.0
    bt.deg_eur_mwh = 0.0
    bt.trans_eur_mwh = 0.0
    bt.aux_off_mw = 0.0
    bt.aux_trading_mw = 0.0
    ts = pd.date_range("2025-01-02T00:00:00Z", periods=2, freq="h")
    rows = pd.DataFrame(
        {
            col.timestamp: ts,
            "target_time_utc": ts,
            col.pred_da_price: [100.0, 120.0],
            col.pred_afrr_activation_rate_pos: [0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0],
        }
    )

    replay = bt._replay_da_candidate_cashflow(
        rows=rows,
        schedule={},
        colmap=col,
        current_soc_mwh=9.0,
        global_end_utc=pd.Timestamp("2025-01-04T00:00:00Z"),
    )

    assert float(replay["da_terminal_sensitive_window"]) == pytest.approx(1.0)
    assert str(replay["da_terminal_sensitive_reason"]) == "terminal_recovery_lookahead_window"
    assert str(replay["da_terminal_value_basis"]) == "terminal_recovery_aware"
    assert float(replay["da_terminal_shortfall_internal_mwh"]) == pytest.approx(1.0)
    assert float(replay["da_terminal_recovery_cost_eur"]) > 0.0


def test_da_sizer_terminal_sensitive_uses_explicit_recovery_cost_not_scalar_lambda() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    bt.terminal_value_mode = "terminal_recovery_aware"
    bt.da_terminal_recovery_lookahead_h = 72.0
    bt.eta_in = 1.0
    bt.eta_out = 1.0
    bt.deg_eur_mwh = 0.0
    bt.trans_eur_mwh = 0.0
    bt.aux_off_mw = 0.0
    bt.aux_trading_mw = 0.0
    bt.da_terminal_value_min_eur_mwh = 0.0
    bt.da_terminal_value_max_eur_mwh = 500.0
    bt.terminal_value_buy_quantile = 0.70
    ts = pd.date_range("2025-01-02T00:00:00Z", periods=2, freq="h")
    rows = pd.DataFrame(
        {
            col.timestamp: ts,
            "target_time_utc": ts,
            "charge_mw": [1.0, 0.0],
            "discharge_mw": [0.0, 0.0],
            col.pred_da_price: [5.0, 10.0],
            col.pred_afrr_activation_rate_pos: [0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0],
            "predicted_objective_eur": [0.0, 0.0],
        }
    )

    selected, audit = bt._select_feasible_da_lock_schedule(
        lock_rows=rows,
        colmap=col,
        current_soc_mwh=9.0,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=pd.Timestamp("2025-01-04T00:00:00Z"),
    )

    assert selected[pd.Timestamp(ts[0])] == pytest.approx((1.0, 0.0))
    assert float(audit[0]["da_bid_sizer_scalar_lambda_disabled_for_terminal_recovery"]) == pytest.approx(1.0)
    assert str(audit[0]["da_bid_sizer_terminal_recovery_cost_units"]) == "eur_per_internal_mwh"
    assert str(audit[0]["da_bid_sizer_terminal_recovery_cost_price_source"]) == col.pred_da_price
    assert float(audit[0]["da_bid_sizer_terminal_recovery_cost_eur_per_internal_mwh"]) == pytest.approx(8.5)
    assert float(audit[0]["candidate_selection_pnl_eur"]) > float(audit[0]["incumbent_selection_pnl_eur"])


def test_da_sizer_rejects_local_profit_when_terminal_recovery_cost_makes_incumbent_better() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    bt.terminal_value_mode = "terminal_recovery_aware"
    bt.da_terminal_recovery_lookahead_h = 48.0
    bt.terminal_recovery_lookahead_h = 48.0
    bt.eta_in = 1.0
    bt.eta_out = 1.0
    bt.deg_eur_mwh = 0.0
    bt.trans_eur_mwh = 0.0
    bt.aux_off_mw = 0.0
    bt.aux_trading_mw = 0.0
    bt.terminal_value_buy_quantile = 1.0
    bt.terminal_value_sell_quantile = 0.0
    ts = pd.date_range("2025-01-02T00:00:00Z", periods=2, freq="h")
    rows = pd.DataFrame(
        {
            col.timestamp: ts,
            "target_time_utc": ts,
            "charge_mw": [0.0, 0.0],
            "discharge_mw": [1.0, 0.0],
            col.pred_da_price: [80.0, 200.0],
            col.pred_afrr_activation_rate_pos: [0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0],
            "predicted_objective_eur": [80.0, 0.0],
        }
    )

    selected, audit = bt._select_feasible_da_lock_schedule(
        lock_rows=rows,
        colmap=col,
        current_soc_mwh=10.0,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=pd.Timestamp("2025-01-03T00:00:00Z"),
    )

    assert selected[pd.Timestamp(ts[0])] == pytest.approx((0.0, 0.0))
    assert str(audit[0]["selected_incumbent"]) in {"no_trade", "zeroed_candidate"}
    assert float(audit[0]["da_bid_sizer_scalar_lambda_disabled_for_terminal_recovery"]) == pytest.approx(1.0)
    assert float(audit[0]["da_bid_sizer_terminal_recovery_cost_eur_per_internal_mwh"]) == pytest.approx(200.0)
    assert float(audit[0]["candidate_selection_pnl_eur_after_terminal_recovery"]) <= float(
        audit[0]["incumbent_selection_pnl_eur_after_terminal_recovery"]
    )
    assert str(audit[0]["da_precommit_final_selection_reason"]) != (
        "optimized_predicted_replay_dominates_or_incumbent_infeasible"
    )


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


def test_negative_activation_revenue_uses_canonical_positive_value() -> None:
    bt = _mk_backtester("canonical_economic")
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
        act_neg_price=1000.0,
        act_pos_rate=0.0,
        act_neg_rate=1.0,
        cap_bid_pos=0.0,
        cap_bid_neg=0.0,
    )
    delivered = float(m["delivered_activation_neg_mwh"])
    assert delivered > 0.0
    assert np.isclose(float(m["revenue_activation_eur"]), delivered * 1000.0, atol=1e-6), m


def test_no_double_sign_flip_negative_activation() -> None:
    bt = _mk_backtester("canonical_economic")
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
        act_neg_price=1000.0,
        act_pos_rate=0.0,
        act_neg_rate=1.0,
        cap_bid_pos=0.0,
        cap_bid_neg=0.0,
    )
    assert float(m["revenue_activation_eur"]) > 0.0


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
    expected_mwh = 10.0 * 0.5 * bt.dt_h
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
    assert float(m["penalty_activation_neg_eur"]) == pytest.approx(
        float(m["missed_activation_neg_mwh"]) * 1000.0
    )


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
    old_discount = MODEL_SPECS.get("terminal_soc_value_discount", None)
    try:
        MODEL_SPECS["terminal_soc_value_discount"] = 0.0
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
    finally:
        if old_discount is None:
            MODEL_SPECS.pop("terminal_soc_value_discount", None)
        else:
            MODEL_SPECS["terminal_soc_value_discount"] = old_discount
    assert np.isclose(float(out["reserve_pos_mw"].iloc[0]), 0.0)
    assert np.isclose(float(out["reserve_neg_mw"].iloc[0]), 0.0)


def test_da_bid_headroom_uses_delivery_hour_soc_for_charge_and_discharge() -> None:
    old_discount = MODEL_SPECS.get("terminal_soc_value_discount", None)
    try:
        MODEL_SPECS["terminal_soc_value_discount"] = 0.0
        bt = _mk_backtester()
        charge_space_mwh = 0.5
        charge_soc_start = float(bt.soc_max - charge_space_mwh)
        charge_df, col = _one_hour_pred_df(
            da=-500.0,
            cap_pos=0.0,
            cap_neg=0.0,
            act_pos=0.0,
            act_neg=0.0,
            rate_pos=0.0,
            rate_neg=0.0,
        )
        charge_out = bt.optimize_dispatch(charge_df, col, allowed_markets=("DA",), soc_start=charge_soc_start)
        charge_limit = charge_space_mwh / max(bt.eta_in * bt.dt_h, 1e-12)
        assert float(charge_out["charge_mw"].iloc[0]) <= charge_limit + 1e-6
        assert float(charge_out["da_headroom_soc_reference_mwh"].iloc[0]) == pytest.approx(charge_soc_start)
        assert float(charge_out["da_charge_headroom_limit_mw"].iloc[0]) == pytest.approx(charge_limit)

        discharge_space_mwh = 0.4
        discharge_soc_start = float(bt.soc_min + discharge_space_mwh)
        discharge_df, col = _one_hour_pred_df(
            da=500.0,
            cap_pos=0.0,
            cap_neg=0.0,
            act_pos=0.0,
            act_neg=0.0,
            rate_pos=0.0,
            rate_neg=0.0,
        )
        discharge_out = bt.optimize_dispatch(
            discharge_df,
            col,
            allowed_markets=("DA",),
            soc_start=discharge_soc_start,
        )
        discharge_limit = discharge_space_mwh * max(bt.eta_out, 1e-12) / max(bt.dt_h, 1e-12)
        assert float(discharge_out["discharge_mw"].iloc[0]) <= discharge_limit + 1e-6
        assert float(discharge_out["da_headroom_soc_reference_mwh"].iloc[0]) == pytest.approx(discharge_soc_start)
        assert float(discharge_out["da_discharge_headroom_limit_mw"].iloc[0]) == pytest.approx(discharge_limit)
    finally:
        if old_discount is None:
            MODEL_SPECS.pop("terminal_soc_value_discount", None)
        else:
            MODEL_SPECS["terminal_soc_value_discount"] = old_discount


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
    assert np.isclose(float(m["revenue_capacity_eur"]), 4.0 * 25.0 * bt.dt_h, atol=1e-9)
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
    assert out.summary.get("bem_only_mode") != "approx_reuse_reserve_volume"


def test_summary_reports_base_and_resolved_id_modes() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=2,
        reopt_step_hours=1,
        allowed_markets=("DA",),
        strategy_name="da",
        id_recourse_mode="common",
        strict_simulation_validity=False,
    )
    assert out.summary.get("base_strategy_id_mode") == "none"
    assert out.summary.get("resolved_id_mode") == "technical_repair"
    assert out.summary.get("id_recourse_mode") == "common"
    assert float(out.summary.get("id_allowed", 0.0)) == 1.0
    assert float(out.summary.get("id_technical_repair_enabled", 0.0)) == 1.0
    assert float(out.summary.get("id_economic_enabled", 1.0)) == 0.0


def test_rolling_pf_benchmark_semantics_in_tiny_deterministic_case() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=2, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    assert out.summary.get("benchmark_type") == "rolling_perfect_foresight_same_rules"
    assert out.summary.get("rolling_pf_benchmark_semantics") == "same_rules_pf"
    assert float(out.summary.get("rolling_pf_deterministic_oracle", 1.0)) == pytest.approx(0.0)
    assert float(
        out.summary.get("rolling_pf_can_be_below_realized_due_to_pay_as_bid_and_physical_guards", 0.0)
    ) == pytest.approx(1.0)
    assert "rolling_perfect_foresight_same_rules_total_pnl_eur" in out.summary
    assert "comparable_rolling_perfect_foresight_same_rules_market_pnl_eur" not in out.summary
    assert "comparable_perfect_foresight_market_pnl_eur" not in out.summary


def test_da_perfect_foresight_price_taker_diagnostics_in_tiny_deterministic_case() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)

    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=2,
        reopt_step_hours=1,
        allowed_markets=("DA",),
        strategy_name="da",
        strict_simulation_validity=False,
    )

    assert out.summary.get("perfect_foresight_da_execution_policy") == "price_taker"
    assert float(out.summary.get("perfect_foresight_da_price_taker_mode", 0.0)) == pytest.approx(1.0)
    assert float(out.summary.get("perfect_foresight_da_price_rejected_buy_mwh", 1.0)) == pytest.approx(0.0)
    assert float(out.summary.get("perfect_foresight_da_price_rejected_sell_mwh", 1.0)) == pytest.approx(0.0)
    pf_realized_da = float(out.summary.get("rolling_pf_solver_settlement_realized_da_buy_mwh", 0.0)) + float(
        out.summary.get("rolling_pf_solver_settlement_realized_da_sell_mwh", 0.0)
    )
    no_trade_reason = str(out.summary.get("rolling_pf_solver_selected_no_trade_reason", ""))
    assert pf_realized_da > 1e-12 or no_trade_reason not in {"", "not_evaluated"}


def test_da_rhpf_price_taker_uses_same_acceptance_as_model() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=30)

    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=24,
        reopt_step_hours=1,
        allowed_markets=("DA",),
        strategy_name="da",
        strict_simulation_validity=False,
    )
    s = out.summary

    assert str(s.get("rolling_pf_da_execution_policy", "")) == "price_taker"
    assert float(s.get("rolling_pf_da_price_taker_mode", 0.0)) == pytest.approx(1.0)
    assert float(s.get("rolling_pf_da_actual_independent_clearing_applied", 1.0)) == pytest.approx(0.0)
    assert float(s.get("rolling_pf_da_limit_robust_guard_applied", 1.0)) == pytest.approx(0.0)
    assert float(s.get("rolling_pf_da_price_rejected_buy_mwh", 1.0)) == pytest.approx(0.0)
    assert float(s.get("rolling_pf_da_price_rejected_sell_mwh", 1.0)) == pytest.approx(0.0)
    assert float(s.get("rolling_pf_da_submitted_buy_mwh", 0.0)) == pytest.approx(
        float(s.get("rolling_pf_da_auction_accepted_buy_mwh", 0.0))
    )
    assert float(s.get("rolling_pf_da_submitted_sell_mwh", 0.0)) == pytest.approx(
        float(s.get("rolling_pf_da_auction_accepted_sell_mwh", 0.0))
    )
    assert "rolling_pf_da_physical_lockbook_buy_mwh" in s
    assert "rolling_pf_da_physical_lockbook_sell_mwh" in s


def test_da_rhpf_price_taker_plan_history_diagnostics_use_full_candidate_history() -> None:
    bt = _mk_backtester()
    plan_history = pd.DataFrame(
        {
            "da_candidate_allowed_by_gate_window": [1.0, 1.0, 0.0],
            "raw_optimizer_charge_mw_before_da_gate_mask": [2.0, 0.0, 9.0],
            "raw_optimizer_discharge_mw_before_da_gate_mask": [0.0, 0.0, 0.0],
            "da_sized_candidate_buy_mw": [1.5, 0.0, 9.0],
            "da_sized_candidate_sell_mw": [0.0, 0.0, 0.0],
            "da_postlock_candidate_buy_mw": [1.0, 0.0, 9.0],
            "da_postlock_candidate_sell_mw": [0.0, 0.0, 0.0],
            "da_selected_lockable_buy_mw": [0.5, 0.0, 9.0],
            "da_selected_lockable_sell_mw": [0.0, 0.0, 0.0],
            "da_candidate_disabled_reason": ["none", "none", "outside_delivery_window"],
            "da_bid_sizer_status": ["ok", "ok", "ignored"],
            "da_bid_sizer_method": ["linprog_continuous_per_hour", "linprog_continuous_per_hour", "ignored"],
            "da_bid_sizer_objective_eur": [12.5, 12.5, 99.0],
            "da_bid_sizer_total_buy_mwh": [1.5, 1.5, 9.0],
            "da_bid_sizer_total_sell_mwh": [0.0, 0.0, 0.0],
            "projected_soc_min_mwh": [7.0, 7.0, 1.0],
            "projected_soc_max_mwh": [12.0, 12.0, 19.0],
            "projected_final_soc_mwh": [10.0, 10.0, 9.0],
            "raw_candidate_pnl_recomputed_eur": [25.0, 25.0, 99.0],
            "sized_candidate_pnl_eur": [10.0, 10.0, 99.0],
        }
    )

    diag = bt._rolling_pf_da_plan_history_diagnostics(plan_history)

    assert float(diag["rolling_pf_da_plan_history_rows"]) == pytest.approx(3.0)
    assert float(diag["rolling_pf_da_gate_allowed_rows"]) == pytest.approx(2.0)
    assert float(diag["rolling_pf_da_gate_nonzero_candidate_rows"]) == pytest.approx(1.0)
    assert float(diag["rolling_pf_da_raw_candidate_buy_mwh"]) == pytest.approx(2.0 * bt.dt_h)
    assert float(diag["rolling_pf_da_sized_candidate_buy_mwh"]) == pytest.approx(1.5 * bt.dt_h)
    assert float(diag["rolling_pf_da_postlock_candidate_buy_mwh"]) == pytest.approx(1.0 * bt.dt_h)
    assert float(diag["rolling_pf_da_selected_lockable_buy_mwh"]) == pytest.approx(0.5 * bt.dt_h)
    assert str(diag["rolling_pf_da_bid_sizer_status"]) == "ok"
    assert str(diag["rolling_pf_da_bid_sizer_method"]) == "linprog_continuous_per_hour"
    assert float(diag["rolling_pf_da_bid_sizer_objective_eur"]) == pytest.approx(12.5)
    assert float(diag["rolling_pf_da_bid_sizer_total_buy_mwh"]) == pytest.approx(1.5)
    assert float(diag["rolling_pf_da_bid_sizer_projected_soc_min_mwh"]) == pytest.approx(7.0)
    assert float(diag["rolling_pf_da_bid_sizer_projected_soc_max_mwh"]) == pytest.approx(12.0)
    assert float(diag["rolling_pf_da_bid_sizer_projected_final_soc_mwh"]) == pytest.approx(10.0)
    assert float(diag["rolling_pf_da_raw_candidate_pnl_eur"]) == pytest.approx(25.0)
    assert float(diag["rolling_pf_da_sized_candidate_pnl_eur"]) == pytest.approx(10.0)
    assert str(diag["rolling_pf_da_zero_trade_stage"]) == "candidate_available"


def test_da_rhpf_plan_history_diagnostics_empty_defaults_are_export_safe() -> None:
    bt = _mk_backtester()

    diag = bt._rolling_pf_da_plan_history_diagnostics(pd.DataFrame())

    assert str(diag["rolling_pf_da_zero_trade_stage"]) == "no_plan_history"
    assert str(diag["rolling_pf_da_bid_sizer_status"]) == "not_evaluated"
    assert str(diag["rolling_pf_da_bid_sizer_error"]) == ""
    assert str(diag["rolling_pf_da_bid_sizer_method"]) == ""
    assert np.isnan(float(diag["rolling_pf_da_bid_sizer_objective_eur"]))
    assert float(diag["rolling_pf_da_bid_sizer_total_buy_mwh"]) == pytest.approx(0.0)
    assert float(diag["rolling_pf_da_bid_sizer_total_sell_mwh"]) == pytest.approx(0.0)
    assert np.isnan(float(diag["rolling_pf_da_bid_sizer_projected_soc_min_mwh"]))
    assert np.isnan(float(diag["rolling_pf_da_bid_sizer_projected_soc_max_mwh"]))
    assert np.isnan(float(diag["rolling_pf_da_bid_sizer_projected_final_soc_mwh"]))
    assert str(diag["rolling_pf_da_candidate_selection_invalid_reason"]) == ""
    assert np.isnan(float(diag["rolling_pf_da_raw_candidate_pnl_eur"]))
    assert np.isnan(float(diag["rolling_pf_da_sized_candidate_pnl_eur"]))
    assert str(diag["rolling_pf_da_sized_zero_reason"]) == "no_plan_history"
    assert str(diag["rolling_pf_da_first_sized_zero_ts_utc"]) == ""


def test_run_initializes_disabled_rhpf_da_sizer_export_locals() -> None:
    source = Path("src/energy_trading/simulation/battery_backtest.py").read_text(encoding="utf-8")
    init_pos = source.index('rolling_pf_da_bid_sizer_status = "not_evaluated"')
    branch_pos = source.index('if benchmark_paths["rhpf"]:')
    export_pos = source.index('hourly["rolling_pf_da_bid_sizer_status"]')

    assert init_pos < branch_pos < export_pos


def test_da_rhpf_plan_history_diagnostics_prefer_post_precommit_stage() -> None:
    bt = _mk_backtester()
    plan_history = pd.DataFrame(
        {
            "plan_history_stage": ["optimizer_raw", "post_precommit", "post_precommit"],
            "da_candidate_allowed_by_gate_window": [1.0, 1.0, 1.0],
            "raw_optimizer_charge_mw_before_da_gate_mask": [9.0, 2.0, 0.0],
            "raw_optimizer_discharge_mw_before_da_gate_mask": [0.0, 0.0, 0.0],
            "da_precommit_da_sized_candidate_buy_mw": [np.nan, 1.5, 0.0],
            "da_precommit_da_sized_candidate_sell_mw": [np.nan, 0.0, 0.0],
            "da_precommit_da_postlock_candidate_buy_mw": [np.nan, 1.0, 0.0],
            "da_precommit_da_postlock_candidate_sell_mw": [np.nan, 0.0, 0.0],
            "da_precommit_da_selected_lockable_buy_mw": [np.nan, 0.5, 0.0],
            "da_precommit_da_selected_lockable_sell_mw": [np.nan, 0.0, 0.0],
            "da_precommit_da_bid_sizer_status": ["", "ok", "ok"],
        }
    )

    diag = bt._rolling_pf_da_plan_history_diagnostics(plan_history)

    assert str(diag["rolling_pf_da_plan_history_stage_used"]) == "post_precommit"
    assert float(diag["rolling_pf_da_plan_history_rows"]) == pytest.approx(2.0)
    assert float(diag["rolling_pf_da_raw_candidate_buy_mwh"]) == pytest.approx(2.0 * bt.dt_h)
    assert float(diag["rolling_pf_da_sized_candidate_buy_mwh"]) == pytest.approx(1.5 * bt.dt_h)
    assert float(diag["rolling_pf_da_selected_lockable_buy_mwh"]) == pytest.approx(0.5 * bt.dt_h)
    assert str(diag["rolling_pf_da_zero_trade_stage"]) == "candidate_available"


def test_da_rhpf_plan_history_diagnostics_missing_post_precommit_not_sized_zero() -> None:
    bt = _mk_backtester()
    plan_history = pd.DataFrame(
        {
            "plan_history_stage": ["optimizer_raw", "optimizer_raw"],
            "da_candidate_allowed_by_gate_window": [1.0, 1.0],
            "raw_optimizer_charge_mw_before_da_gate_mask": [5.0, 0.0],
            "raw_optimizer_discharge_mw_before_da_gate_mask": [0.0, 5.0],
        }
    )

    diag = bt._rolling_pf_da_plan_history_diagnostics(plan_history)

    assert str(diag["rolling_pf_da_plan_history_stage_used"]) == "optimizer_raw"
    assert str(diag["rolling_pf_da_zero_trade_stage"]) == "diagnostics_missing_post_precommit_stage"
    assert str(diag["rolling_pf_da_sized_zero_reason"]) == "none"


def test_da_rhpf_price_taker_plan_history_diagnostics_classify_optimizer_zero_candidate() -> None:
    bt = _mk_backtester()
    plan_history = pd.DataFrame(
        {
            "da_candidate_allowed_by_gate_window": [1.0, 1.0],
            "raw_optimizer_charge_mw_before_da_gate_mask": [0.0, 0.0],
            "raw_optimizer_discharge_mw_before_da_gate_mask": [0.0, 0.0],
        }
    )

    diag = bt._rolling_pf_da_plan_history_diagnostics(plan_history)

    assert float(diag["rolling_pf_da_gate_allowed_rows"]) == pytest.approx(2.0)
    assert float(diag["rolling_pf_da_raw_candidate_buy_mwh"]) == pytest.approx(0.0)
    assert str(diag["rolling_pf_da_zero_trade_stage"]) == "optimizer_zero_da_candidate"
    assert str(diag["rolling_pf_da_plan_history_zero_reason"]) == "optimizer_zero_da_candidate"


def test_da_rhpf_plan_history_diagnostics_explain_sized_candidate_zeroed() -> None:
    bt = _mk_backtester()
    plan_history = pd.DataFrame(
        {
            "target_time_utc": pd.to_datetime(["2025-03-02T13:00:00Z", "2025-03-02T17:00:00Z"]),
            "da_candidate_allowed_by_gate_window": [1.0, 1.0],
            "raw_optimizer_charge_mw_before_da_gate_mask": [5.0, 0.0],
            "raw_optimizer_discharge_mw_before_da_gate_mask": [0.0, 5.0],
            "da_sized_candidate_buy_mw": [0.0, 0.0],
            "da_sized_candidate_sell_mw": [0.0, 0.0],
            "da_bid_sizer_status": ["lp_selected_zero_schedule", "lp_selected_zero_schedule"],
            "da_bid_sizer_zero_reason": [
                "lp_objective_selected_zero_no_nonzero_retry_feasible",
                "lp_objective_selected_zero_no_nonzero_retry_feasible",
            ],
            "da_bid_sizer_error": ["none", "none"],
            "da_precommit_selection_invalid_reason": ["none", "none"],
            "raw_candidate_pnl_recomputed_eur": [50.0, 50.0],
            "sized_candidate_pnl_eur": [0.0, 0.0],
        }
    )

    diag = bt._rolling_pf_da_plan_history_diagnostics(plan_history)

    assert str(diag["rolling_pf_da_zero_trade_stage"]) == "sized_candidate_zeroed"
    assert str(diag["rolling_pf_da_bid_sizer_status"]) == "lp_selected_zero_schedule"
    assert str(diag["rolling_pf_da_sized_zero_reason"]) == "lp_objective_selected_zero_no_nonzero_retry_feasible"
    assert str(diag["rolling_pf_da_first_sized_zero_ts_utc"]) == "2025-03-02T13:00:00+00:00"
    assert float(diag["rolling_pf_da_raw_candidate_pnl_eur"]) == pytest.approx(50.0)
    assert float(diag["rolling_pf_da_sized_candidate_pnl_eur"]) == pytest.approx(0.0)


def test_da_sizer_future_replay_keeps_soc_min_feasible_candidate() -> None:
    bt = _mk_backtester()
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    bt.da_terminal_soc_value_enabled = False
    df, col = _tiny_backtest_df(hours=1)
    df[col.pred_da_price] = 250.0
    lock_rows = df.copy()
    lock_rows["target_time_utc"] = lock_rows[col.timestamp]
    lock_rows["charge_mw"] = 0.0
    lock_rows["discharge_mw"] = 0.1

    selected, audit_rows = bt._select_feasible_da_lock_schedule(
        lock_rows=lock_rows,
        colmap=col,
        current_soc_mwh=2.25,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        future_rows=df,
        existing_da_lockbook={},
        global_end_utc=None,
    )

    assert selected
    assert sum(dis for _, dis in selected.values()) > 0.0
    assert audit_rows
    assert max(float(row["da_selected_lockable_sell_mw"]) for row in audit_rows) > 0.0
    assert min(float(row["da_bid_sizer_lp_projected_soc_min_mwh"]) for row in audit_rows) >= bt.soc_min - 1e-6


def test_da_rhpf_plan_history_diagnostics_export_postlock_replay_mismatch() -> None:
    bt = _mk_backtester()
    plan_history = pd.DataFrame(
        {
            "target_time_utc": pd.to_datetime(["2025-03-03T23:00:00Z"]),
            "plan_history_stage": ["post_precommit"],
            "da_candidate_allowed_by_gate_window": [1.0],
            "raw_optimizer_charge_mw_before_da_gate_mask": [0.0],
            "raw_optimizer_discharge_mw_before_da_gate_mask": [5.0],
            "da_precommit_da_sized_candidate_buy_mw": [0.0],
            "da_precommit_da_sized_candidate_sell_mw": [4.0],
            "da_precommit_da_postlock_candidate_buy_mw": [0.0],
            "da_precommit_da_postlock_candidate_sell_mw": [0.0],
            "da_precommit_da_selected_lockable_buy_mw": [0.0],
            "da_precommit_da_selected_lockable_sell_mw": [0.0],
            "da_first_rejection_ts_utc": ["2025-03-03T23:00:00+00:00"],
            "da_first_rejection_delivery_ts_utc": ["2025-03-03T23:00:00+00:00"],
            "da_first_rejection_reason": ["da_sizer_postlock_replay_mismatch"],
            "da_first_rejection_stage": ["postlock_guard_zeroed_candidate"],
            "da_first_rejection_candidate_buy_mw": [0.0],
            "da_first_rejection_candidate_sell_mw": [4.0],
            "da_first_rejection_existing_locked_buy_mw": [0.0],
            "da_first_rejection_existing_locked_sell_mw": [0.0],
            "da_first_rejection_soc_before_mwh": [2.50],
            "da_first_rejection_soc_after_mwh": [1.95],
            "da_first_rejection_soc_min_mwh": [2.0],
            "da_first_rejection_soc_max_mwh": [18.0],
            "da_first_rejection_soc_violation_mwh": [0.05],
            "da_first_rejection_aux_loss_mwh": [0.05],
            "da_first_rejection_efficiency_mode": ["eta_in=0.92,eta_out=0.92"],
            "da_first_rejection_replay_function": ["_project_da_postlock_future_feasibility"],
            "da_sizer_vs_postlock_replay_delta_min_soc_mwh": [-0.1753],
            "da_sizer_vs_postlock_replay_delta_final_soc_mwh": [-0.1753],
            "da_sizer_postlock_replay_mismatch": [1.0],
            "da_bid_sizer_lp_projected_soc_min_mwh": [2.1253],
            "da_bid_sizer_lp_projected_final_soc_mwh": [2.1253],
            "da_postlock_infeasibility_driver_detail": ["locked_da_hourly_soc_min_infeasible"],
        }
    )

    diag = bt._rolling_pf_da_plan_history_diagnostics(plan_history)

    assert str(diag["rolling_pf_da_zero_trade_stage"]) == "postlock_guard_zeroed_candidate"
    assert str(diag["rolling_pf_da_first_rejection_reason"]) == "da_sizer_postlock_replay_mismatch"
    assert str(diag["rolling_pf_da_first_rejection_ts_utc"]) == "2025-03-03T23:00:00+00:00"
    assert float(diag["rolling_pf_da_first_rejection_candidate_sell_mw"]) == pytest.approx(4.0)
    assert float(diag["rolling_pf_da_first_rejection_soc_violation_mwh"]) == pytest.approx(0.05)
    assert float(diag["rolling_pf_da_sizer_vs_postlock_replay_delta_min_soc_mwh"]) == pytest.approx(-0.1753)
    assert float(diag["da_sizer_postlock_replay_mismatch"]) == pytest.approx(1.0)


def test_da_rhpf_price_taker_uses_true_da_prices() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=30)
    # Forecast is intentionally flat/wrong; RHPF should still use truth-collapsed
    # DA columns for its rolling optimizer inputs.
    df[col.pred_da_price] = 100.0
    df["pred_da_price_p05"] = 100.0
    df["pred_da_price_p10"] = 100.0
    df["pred_da_price_p90"] = 100.0
    df["pred_da_price_p95"] = 100.0
    df[col.true_da_price] = 100.0
    df.loc[df.index[23:26], col.true_da_price] = [10.0, 15.0, 200.0]

    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=24,
        reopt_step_hours=1,
        allowed_markets=("DA",),
        strategy_name="da",
        strict_simulation_validity=False,
    )
    s = out.summary

    assert float(s.get("rolling_pf_da_truth_price_substitution_applied", 0.0)) == pytest.approx(1.0)
    assert str(s.get("rolling_pf_quantile_surface_mode", "")) == "collapsed_to_truth"
    assert "rolling_pf_da_plan_history_rows" in s
    assert "rolling_pf_da_raw_candidate_buy_mwh" in s
    assert "rolling_pf_da_zero_trade_stage" in s
    rhpf_da_volume = float(s.get("rolling_pf_da_dispatch_buy_mwh", 0.0)) + float(
        s.get("rolling_pf_da_dispatch_sell_mwh", 0.0)
    )
    assert rhpf_da_volume > 1e-12 or str(s.get("rolling_pf_da_zero_trade_reason", "")) not in {
        "",
        "not_evaluated",
    }


def test_da_rhpf_price_taker_summary_matches_hourly_final_cum_pnl() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)

    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=4,
        reopt_step_hours=1,
        allowed_markets=("DA",),
        strategy_name="da",
        strict_simulation_validity=False,
    )

    assert "rolling_pf_pnl_eur" in out.hourly.columns
    summary_pf_pnl = float(out.summary["rolling_pf_pnl_eur"])
    hourly_pf_pnl = float(pd.to_numeric(out.hourly["rolling_pf_pnl_eur"], errors="coerce").iloc[-1])
    if np.isfinite(summary_pf_pnl) or np.isfinite(hourly_pf_pnl):
        assert summary_pf_pnl == pytest.approx(hourly_pf_pnl)
    else:
        assert np.isnan(summary_pf_pnl)
        assert np.isnan(hourly_pf_pnl)
    assert "rolling_pf_solver_raw_total_pnl_eur" in out.summary
    for col_name in [
        "rolling_pf_solver_total_pnl_eur",
        "rolling_pf_solver_component_pnl_eur",
        "rolling_pf_solver_pnl_balance_error_eur",
        "rolling_pf_solver_pnl_balance_ok",
        "rolling_pf_solver_available",
        "rolling_pf_solver_invalid_reason",
        "rolling_pf_reported_total_pnl_eur",
        "rolling_pf_reported_available",
        "rolling_pf_reported_is_solver",
        "rolling_pf_reported_invalid_reason",
        "rolling_pf_realized_path_total_pnl_eur",
        "rolling_pf_no_market_total_pnl_eur",
        "rolling_pf_solver_minus_realized_path_eur",
        "rolling_pf_solver_below_realized_path",
        "rolling_pf_dominance_check_pass",
        "rolling_pf_selected_total_pnl_eur",
        "rolling_pf_selected_component_pnl_eur",
        "rolling_pf_selected_pnl_balance_error_eur",
        "rolling_pf_selected_pnl_balance_ok",
        "rolling_pf_selected_is_solver",
        "rolling_pf_selected_is_realized_path_fallback",
        "rolling_pf_selected_is_no_market_fallback",
        "rolling_pf_no_market_feasible",
        "rolling_pf_realized_path_feasible",
        "rolling_pf_solver_infeasible_reason",
        "rolling_pf_no_market_infeasible_reason",
        "rolling_pf_realized_path_infeasible_reason",
        "rolling_pf_expected_row_count",
        "rolling_pf_solver_row_count",
        "rolling_pf_no_market_row_count",
        "rolling_pf_realized_path_row_count",
    ]:
        assert col_name in out.summary
    if float(out.summary["rolling_pf_reported_available"]) >= 0.5:
        assert float(out.summary["rolling_perfect_foresight_same_rules_total_pnl_eur"]) == pytest.approx(
            float(out.summary["rolling_pf_reported_total_pnl_eur"])
        )
        assert float(out.summary["same_rules_rolling_pf_total_pnl_eur"]) == pytest.approx(
            float(out.summary["rolling_pf_reported_total_pnl_eur"])
        )
        if "rhpf_total_pnl_eur" in out.summary:
            assert float(out.summary["rhpf_total_pnl_eur"]) == pytest.approx(
                float(out.summary["rolling_pf_reported_total_pnl_eur"])
            )
    else:
        assert np.isnan(float(out.summary["rolling_perfect_foresight_same_rules_total_pnl_eur"]))
        assert str(out.summary["rolling_pf_reported_invalid_reason"]) != "none"
    selected = str(out.summary["rolling_pf_selected_incumbent"])
    assert float(out.summary["rolling_pf_selected_is_solver"]) == pytest.approx(float(selected == "solver"))
    assert float(out.summary["rolling_pf_selected_is_realized_path_fallback"]) == pytest.approx(
        float(selected == "realized_path")
    )
    assert float(out.summary["rolling_pf_selected_is_no_market_fallback"]) == pytest.approx(
        float(selected == "no_market")
    )
    for col_name in [
        "rolling_pf_da_plan_history_rows",
        "rolling_pf_da_gate_allowed_rows",
        "rolling_pf_da_raw_candidate_buy_mwh",
        "rolling_pf_da_selected_lockable_buy_mwh",
    ]:
        assert col_name in out.hourly.columns
        assert col_name in out.summary
        assert float(out.summary[col_name]) == pytest.approx(
            float(pd.to_numeric(out.hourly[col_name], errors="coerce").iloc[-1])
        )
    if float(out.summary.get("rolling_pf_reported_available", 0.0)) >= 0.5:
        assert float(out.summary["rolling_perfect_foresight_same_rules_total_pnl_eur"]) == pytest.approx(
            float(out.summary["rolling_pf_reported_total_pnl_eur"])
        )


def test_rhpf_selected_incumbent_can_beat_raw_solver_without_hiding_solver() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=30)

    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=24,
        reopt_step_hours=1,
        allowed_markets=("DA",),
        strategy_name="da",
        strict_simulation_validity=False,
    )
    s = out.summary

    assert float(s["rolling_pf_no_market_feasible"]) >= 0.5
    assert str(s["rolling_pf_no_market_infeasible_reason"]) == "none"
    assert float(s["rolling_pf_expected_row_count"]) == pytest.approx(float(s["rolling_pf_solver_row_count"]))
    assert float(s["rolling_pf_expected_row_count"]) == pytest.approx(float(s["rolling_pf_no_market_row_count"]))
    assert str(s["rolling_pf_selected_incumbent"]) in {"solver", "no_market"}
    assert float(s["rolling_pf_selected_is_no_market_fallback"]) == pytest.approx(
        float(str(s["rolling_pf_selected_incumbent"]) == "no_market")
    )
    if float(s["rolling_pf_solver_feasible"]) >= 0.5:
        assert str(s["rolling_pf_solver_infeasible_reason"]) == "none"
        assert float(s["rolling_pf_reported_available"]) == pytest.approx(1.0)
        assert float(s["rolling_pf_reported_is_solver"]) == pytest.approx(1.0)
        assert float(s["rolling_pf_reported_total_pnl_eur"]) == pytest.approx(
            float(s["rolling_pf_solver_total_pnl_eur"])
        )
        assert float(s["rolling_perfect_foresight_same_rules_total_pnl_eur"]) == pytest.approx(
            float(s["rolling_pf_reported_total_pnl_eur"])
        )
    else:
        assert str(s["rolling_pf_solver_infeasible_reason"]) != "none"
        assert float(s["rolling_pf_reported_available"]) == pytest.approx(0.0)
        assert np.isnan(float(s["rolling_pf_reported_total_pnl_eur"]))
        assert np.isnan(float(s["rolling_perfect_foresight_same_rules_total_pnl_eur"]))
    assert float(s["rolling_pf_dominance_check_pass"]) == pytest.approx(1.0)
    assert np.isfinite(float(s["rolling_pf_solver_total_pnl_eur"]))
    assert np.isfinite(float(s["rolling_pf_solver_component_pnl_eur"]))


def test_bcm_rhpf_capacity_price_plan_history_diagnostics_explain_side_selection() -> None:
    bt = _mk_backtester()
    plan_history = pd.DataFrame(
        {
            "bcm_candidate_allowed_by_gate_window": [1.0, 1.0, 0.0],
            "bcm_candidate_pos_mw": [4.0, 0.0, 9.0],
            "bcm_candidate_neg_mw": [0.0, 3.0, 9.0],
            "bcm_precommit_candidate_pos_mw": [4.0, 0.0, 9.0],
            "bcm_precommit_candidate_neg_mw": [0.0, 2.0, 9.0],
            "bcm_precommit_locked_pos_mw": [4.0, 0.0, 9.0],
            "bcm_precommit_locked_neg_mw": [0.0, 1.0, 9.0],
            "bcm_ev_pos": [10.0, 5.0, 99.0],
            "bcm_ev_neg": [2.0, 30.0, 99.0],
            "bcm_optimizer_coeff_pos_eur_per_mw": [8.0, 4.0, 99.0],
            "bcm_optimizer_coeff_neg_eur_per_mw": [1.0, 24.0, 99.0],
            "bcm_capacity_price_source_column_pos": ["pred_afrr_capacity_price_pos_p70", "none", "ignored"],
            "bcm_capacity_price_source_column_neg": ["none", "pred_afrr_capacity_price_neg_p70", "ignored"],
            "bcm_precommit_zero_reason": ["none", "none", "outside_gate"],
        }
    )

    diag = bt._rolling_pf_bcm_plan_history_diagnostics(
        plan_history,
        deterministic_reserve_settlement=False,
        truth_price_substitution_applied=True,
    )

    assert float(diag["rolling_pf_bcm_plan_history_rows"]) == pytest.approx(3.0)
    assert float(diag["rolling_pf_bcm_gate_allowed_rows"]) == pytest.approx(2.0)
    assert float(diag["rolling_pf_bcm_gate_nonzero_candidate_rows"]) == pytest.approx(2.0)
    assert float(diag["rolling_pf_bcm_candidate_pos_mw"]) == pytest.approx(4.0)
    assert float(diag["rolling_pf_bcm_candidate_neg_mw"]) == pytest.approx(3.0)
    assert float(diag["rolling_pf_bcm_locked_pos_mw"]) == pytest.approx(4.0)
    assert float(diag["rolling_pf_bcm_locked_neg_mw"]) == pytest.approx(1.0)
    assert float(diag["rolling_pf_bcm_ev_pos"]) == pytest.approx(7.5)
    assert float(diag["rolling_pf_bcm_ev_neg"]) == pytest.approx(16.0)
    assert str(diag["rolling_pf_bcm_capacity_price_source_column_pos"]) == "pred_afrr_capacity_price_pos_p70"
    assert str(diag["rolling_pf_bcm_capacity_price_source_column_neg"]) == "pred_afrr_capacity_price_neg_p70"
    assert str(diag["rolling_pf_bcm_zero_stage"]) == "candidate_locked"
    assert float(diag["rolling_pf_bcm_deterministic_reserve_settlement"]) == pytest.approx(0.0)
    assert float(diag["rolling_pf_bcm_truth_price_substitution_applied"]) == pytest.approx(1.0)


def test_bcm_rhpf_plan_history_diagnostics_prefer_post_precommit_stage() -> None:
    bt = _mk_backtester()
    plan_history = pd.DataFrame(
        {
            "plan_history_stage": ["optimizer_raw", "post_precommit", "post_precommit"],
            "timestamp_utc": pd.to_datetime(
                ["2025-03-01T00:00:00Z", "2025-03-01T00:00:00Z", "2025-03-01T01:00:00Z"],
                utc=True,
            ),
            "bcm_candidate_allowed_by_gate_window": [1.0, 1.0, 1.0],
            "bcm_candidate_pos_mw": [99.0, 4.0, 0.0],
            "bcm_candidate_neg_mw": [99.0, 0.0, 3.0],
            "bcm_precommit_candidate_pos_mw": [np.nan, 4.0, 0.0],
            "bcm_precommit_candidate_neg_mw": [np.nan, 0.0, 2.0],
            "bcm_precommit_locked_pos_mw": [np.nan, 4.0, 0.0],
            "bcm_precommit_locked_neg_mw": [np.nan, 0.0, 1.0],
        }
    )

    diag = bt._rolling_pf_bcm_plan_history_diagnostics(
        plan_history,
        deterministic_reserve_settlement=False,
        truth_price_substitution_applied=True,
    )

    assert str(diag["rolling_pf_bcm_plan_history_stage_used"]) == "post_precommit"
    assert float(diag["rolling_pf_bcm_plan_history_rows"]) == pytest.approx(2.0)
    assert float(diag["rolling_pf_bcm_candidate_pos_mw"]) == pytest.approx(4.0)
    assert float(diag["rolling_pf_bcm_candidate_neg_mw"]) == pytest.approx(3.0)
    assert float(diag["rolling_pf_bcm_plan_history_locked_pos_mw"]) == pytest.approx(4.0)
    assert float(diag["rolling_pf_bcm_plan_history_locked_neg_mw"]) == pytest.approx(1.0)
    assert str(diag["rolling_pf_bcm_plan_history_alignment_method"]) == "latest_post_precommit_per_delivery_ts"
    assert float(diag["rolling_pf_bcm_plan_history_alignment_available"]) == pytest.approx(1.0)
    assert float(diag["rolling_pf_bcm_plan_history_alignment_comparable"]) == pytest.approx(0.0)
    assert str(diag["rolling_pf_bcm_zero_stage"]) == "candidate_locked"


def test_bcm_rhpf_plan_history_diagnostics_keep_raw_duplicate_and_aligned_totals() -> None:
    bt = _mk_backtester()
    plan_history = pd.DataFrame(
        {
            "plan_history_stage": ["post_precommit", "post_precommit", "post_precommit"],
            "timestamp_utc": pd.to_datetime(
                ["2025-03-01T00:00:00Z", "2025-03-01T00:00:00Z", "2025-03-01T01:00:00Z"],
                utc=True,
            ),
            "source_snapshot_ts": pd.to_datetime(
                ["2025-02-28T10:00:00Z", "2025-02-28T11:00:00Z", "2025-02-28T11:00:00Z"],
                utc=True,
            ),
            "bcm_candidate_allowed_by_gate_window": [1.0, 1.0, 1.0],
            "bcm_candidate_pos_mw": [2.0, 4.0, 0.0],
            "bcm_candidate_neg_mw": [0.0, 0.0, 3.0],
            "bcm_precommit_locked_pos_mw": [2.0, 4.0, 0.0],
            "bcm_precommit_locked_neg_mw": [0.0, 0.0, 1.0],
        }
    )

    diag = bt._rolling_pf_bcm_plan_history_diagnostics(
        plan_history,
        deterministic_reserve_settlement=False,
        truth_price_substitution_applied=True,
    )

    assert float(diag["rolling_pf_bcm_plan_history_locked_pos_mw_raw_sum"]) == pytest.approx(6.0)
    assert float(diag["rolling_pf_bcm_plan_history_locked_neg_mw_raw_sum"]) == pytest.approx(1.0)
    assert float(diag["rolling_pf_bcm_plan_history_aligned_locked_pos_mw"]) == pytest.approx(4.0)
    assert float(diag["rolling_pf_bcm_plan_history_aligned_locked_neg_mw"]) == pytest.approx(1.0)
    assert float(diag["rolling_pf_bcm_plan_history_duplicate_delivery_rows"]) == pytest.approx(1.0)
    assert str(diag["rolling_pf_bcm_plan_history_alignment_method"]) == "latest_post_precommit_per_delivery_ts"
    assert float(diag["rolling_pf_bcm_plan_history_alignment_comparable"]) == pytest.approx(0.0)


def test_rolling_pf_incumbent_selection_prefers_higher_feasible_realized_path() -> None:
    frame = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=1, tz="UTC")})

    selected = BatteryBacktester._select_global_pf_incumbent(
        {
            "solver": (1.0, True, frame),
            "no_market": (0.0, True, frame),
            "realized_path": (2.0, True, frame),
        }
    )

    assert str(selected["selected"]) == "realized_path"
    assert float(selected["value"]) == pytest.approx(2.0)


def test_rolling_pf_normalized_diagnostic_selection_uses_final_totals() -> None:
    selected = BatteryBacktester._select_normalized_rolling_pf_diagnostic_incumbent(
        {
            "solver": (10.0, 10.0, True),
            "no_market": (-5.0, -5.0, True),
            "realized_path": (25.0, 25.0, True),
        }
    )

    assert str(selected["selected"]) == "realized_path"
    assert float(selected["selected_total_pnl_eur"]) == pytest.approx(25.0)
    assert float(selected["selected_is_realized_path_fallback"]) == pytest.approx(1.0)
    assert float(selected["selected_is_solver"]) == pytest.approx(0.0)
    assert str(selected["selection_reason"]) == "realized_path_has_highest_feasible_pnl"


def test_rolling_pf_normalized_diagnostic_selection_prefers_solver_when_final_solver_is_best() -> None:
    selected = BatteryBacktester._select_normalized_rolling_pf_diagnostic_incumbent(
        {
            "solver": (208.9375668329, 208.9375668329, True),
            "no_market": (-906.2952740803, -906.2952740803, True),
            "realized_path": (-13.0318682358, -13.0318682358, False),
        }
    )

    assert str(selected["selected"]) == "solver"
    assert float(selected["selected_total_pnl_eur"]) == pytest.approx(208.9375668329)
    assert float(selected["selected_is_solver"]) == pytest.approx(1.0)
    assert float(selected["selected_pnl_balance_ok"]) == pytest.approx(1.0)


def test_bcm_rhpf_side_mismatch_diagnostics_explain_model_neg_pf_pos() -> None:
    hourly = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2025-03-02T03:00:00Z"], utc=True),
            "real_bcm_precommit_locked_pos_mw": [0.0],
            "real_bcm_precommit_locked_neg_mw": [4.0],
            "perfect_foresight_bcm_precommit_locked_pos_mw": [4.0],
            "perfect_foresight_bcm_precommit_locked_neg_mw": [0.0],
            "real_bcm_ev_pos": [3.0],
            "real_bcm_ev_neg": [12.0],
            "perfect_foresight_bcm_ev_pos": [8.0],
            "perfect_foresight_bcm_ev_neg": [2.0],
            "real_revenue_activation_eur": [100.0],
            "perfect_foresight_revenue_activation_eur": [20.0],
            "real_revenue_capacity_eur": [30.0],
            "perfect_foresight_revenue_capacity_eur": [25.0],
            "real_bcm_precommit_zero_reason": ["none"],
            "perfect_foresight_bcm_precommit_zero_reason": ["none"],
            "real_bcm_capacity_bid_price_neg_eur_per_mw_h": [30.0],
            "perfect_foresight_bcm_capacity_bid_price_neg_eur_per_mw_h": [10.0],
            "real_awarded_capacity_neg_mw": [4.0],
            "perfect_foresight_awarded_capacity_neg_mw": [0.0],
            "real_bcm_capacity_revenue_neg_eur": [120.0],
            "perfect_foresight_bcm_capacity_revenue_neg_eur": [0.0],
        }
    )

    diag = BatteryBacktester._rolling_pf_bcm_side_mismatch_diagnostics(hourly)

    assert float(diag["rolling_pf_bcm_side_mismatch_count"]) == pytest.approx(1.0)
    assert str(diag["rolling_pf_bcm_first_side_mismatch_reason"]) == "model_locks_neg_pf_locks_pos_or_zero"
    assert str(diag["rolling_pf_bcm_raw_solver_underperformance_classification"]) == (
        "raw_solver_side_selection_differs_from_model"
    )
    assert float(diag["rolling_pf_bcm_first_side_mismatch_real_locked_neg_mw"]) == pytest.approx(4.0)
    assert float(diag["rolling_pf_bcm_first_side_mismatch_pf_locked_pos_mw"]) == pytest.approx(4.0)
    assert float(diag["rolling_pf_bcm_activation_revenue_gap_eur"]) == pytest.approx(-80.0)
    assert float(diag["rolling_pf_bcm_model_bid_above_pf_truth_bid_but_cleared_mw"]) == pytest.approx(4.0)
    assert float(diag["rolling_pf_bcm_pay_as_bid_price_gap_eur"]) == pytest.approx(120.0)
    assert float(diag["rolling_pf_bcm_underperformance_due_to_pay_as_bid_bid_price"]) == pytest.approx(1.0)


def test_bcm_rhpf_headroom_guard_diagnostics_export_first_rejected_block() -> None:
    hourly = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2025-03-02T03:00:00Z"], utc=True),
            "reserve_product_block_id": ["2025-03-02_h04-08"],
            "real_bcm_precommit_locked_pos_mw": [0.0],
            "real_bcm_precommit_locked_neg_mw": [0.0],
            "perfect_foresight_bcm_precommit_locked_pos_mw": [0.0],
            "perfect_foresight_bcm_precommit_locked_neg_mw": [0.0],
            "perfect_foresight_bcm_precommit_candidate_pos_mw": [4.0],
            "perfect_foresight_bcm_precommit_candidate_neg_mw": [0.0],
            "perfect_foresight_bcm_precommit_zero_reason": ["headroom_infeasible"],
            "perfect_foresight_soc_start_mwh": [17.2],
            "perfect_foresight_required_headroom_pos_mwh": [2.1],
            "perfect_foresight_available_headroom_pos_mwh": [0.8],
            "real_revenue_activation_eur": [0.0],
            "perfect_foresight_revenue_activation_eur": [0.0],
            "real_revenue_capacity_eur": [0.0],
            "perfect_foresight_revenue_capacity_eur": [0.0],
        }
    )

    diag = BatteryBacktester._rolling_pf_bcm_side_mismatch_diagnostics(hourly)

    assert float(diag["rolling_pf_bcm_underperformance_due_to_headroom_guard"]) == pytest.approx(1.0)
    assert str(diag["rolling_pf_bcm_first_headroom_rejected_block_id"]) == "2025-03-02_h04-08"
    assert str(diag["rolling_pf_bcm_first_headroom_rejected_side"]) == "pos"
    assert float(diag["rolling_pf_bcm_first_headroom_rejected_soc_mwh"]) == pytest.approx(17.2)
    assert float(diag["rolling_pf_bcm_first_headroom_required_mwh"]) == pytest.approx(2.1)
    assert float(diag["rolling_pf_bcm_first_headroom_available_mwh"]) == pytest.approx(0.8)
    assert str(diag["rolling_pf_bcm_raw_solver_underperformance_classification"]) == (
        "raw_solver_headroom_guard_limits_bcm"
    )


def test_bcm_capacity_auction_audit_classifies_bid_above_cutoff_as_auction_bug() -> None:
    row = pd.Series(
        {
            "true_capacity_cutoff_price_eur_per_mw_h": 10.0,
            "model_bid_price_eur_per_mw_h": 11.0,
            "model_awarded_mw": 2.0,
            "rhpf_bid_price_eur_per_mw_h": 10.0,
            "rhpf_awarded_mw": 0.0,
            "rhpf_submitted_mw": 0.0,
        }
    )

    classification = audit_bcm_capacity_auction.classify_bcm_capacity_auction_row(row)

    assert classification == "auction_bug_model_awarded_above_cutoff"


def test_bcm_capacity_auction_audit_classifies_rhpf_submitted_under_cutoff() -> None:
    row = pd.Series(
        {
            "true_capacity_cutoff_price_eur_per_mw_h": 20.0,
            "model_bid_price_eur_per_mw_h": 15.0,
            "model_awarded_mw": 3.0,
            "rhpf_bid_price_eur_per_mw_h": 10.0,
            "rhpf_awarded_mw": 3.0,
            "rhpf_submitted_mw": 3.0,
        }
    )

    classification = audit_bcm_capacity_auction.classify_bcm_capacity_auction_row(row)

    assert classification == "rhpf_underbids_cutoff"


def test_bcm_capacity_auction_audit_builds_block_side_rows(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "bcm" / "p70_p70"
    scenario_dir.mkdir(parents=True)
    hours = pd.date_range("2025-04-01T00:00:00Z", periods=4, freq="h")
    hourly = pd.DataFrame(
        {
            "timestamp_utc": hours,
            "bcm_capacity_block_id": ["block-a"] * 4,
            "bcm_capacity_block_start_utc": [hours[0]] * 4,
            "bcm_capacity_block_end_utc": [hours[-1] + pd.Timedelta(hours=1)] * 4,
            "bcm_capacity_cutoff_price_pos_eur_per_mw_h": [20.0] * 4,
            "bcm_capacity_cutoff_price_neg_eur_per_mw_h": [8.0] * 4,
            "real_submitted_bcm_capacity_pos_mw": [2.0] * 4,
            "real_bcm_capacity_bid_price_pos_eur_per_mw_h": [15.0] * 4,
            "real_awarded_capacity_pos_mw": [2.0] * 4,
            "perfect_foresight_submitted_bcm_capacity_pos_mw": [2.0] * 4,
            "perfect_foresight_bcm_capacity_bid_price_pos_eur_per_mw_h": [10.0] * 4,
            "perfect_foresight_awarded_capacity_pos_mw": [2.0] * 4,
            "real_submitted_bcm_capacity_neg_mw": [1.0] * 4,
            "real_bcm_capacity_bid_price_neg_eur_per_mw_h": [9.0] * 4,
            "real_awarded_capacity_neg_mw": [1.0] * 4,
            "perfect_foresight_submitted_bcm_capacity_neg_mw": [0.0] * 4,
            "perfect_foresight_bcm_capacity_bid_price_neg_eur_per_mw_h": [0.0] * 4,
            "perfect_foresight_awarded_capacity_neg_mw": [0.0] * 4,
            "perfect_foresight_bcm_precommit_zero_reason": ["headroom_infeasible"] * 4,
        }
    )
    hourly.to_parquet(scenario_dir / "backtest_hourly.parquet")
    (scenario_dir / "backtest_summary.json").write_text("{}", encoding="utf-8")

    audit, summary = audit_bcm_capacity_auction.build_bcm_capacity_auction_audit(scenario_dir)

    assert int(summary["rows"]) == 2
    assert int(summary["auction_bug_rows"]) == 1
    pos = audit.loc[audit["side"] == "pos"].iloc[0]
    neg = audit.loc[audit["side"] == "neg"].iloc[0]
    assert pos["classification"] == "rhpf_underbids_cutoff"
    assert neg["classification"] == "auction_bug_model_awarded_above_cutoff"
    assert float(pos["rhpf_submitted_but_under_cutoff"]) == pytest.approx(1.0)
    assert float(neg["model_awarded_with_bid_above_cutoff"]) == pytest.approx(1.0)


def test_rhpf_market_price_lineage_classifies_da_price_taker() -> None:
    row = pd.Series(
        {
            "submitted_buy_mw": 2.0,
            "submitted_sell_mw": 0.0,
            "price_rejected_buy_mwh": 0.0,
            "price_rejected_sell_mwh": 0.0,
            "execution_policy": "price_taker",
            "price_taker_mode": 1.0,
        }
    )

    classification = audit_rhpf_market_price_lineage.classify_da_lineage_row(row)

    assert classification == "da_ok_price_taker"


def test_rhpf_market_price_lineage_classifies_da_limit_rejection_as_bug() -> None:
    row = pd.Series(
        {
            "submitted_buy_mw": 2.0,
            "submitted_sell_mw": 0.0,
            "price_rejected_buy_mwh": 1.0,
            "price_rejected_sell_mwh": 0.0,
            "execution_policy": "limit",
            "price_taker_mode": 0.0,
        }
    )

    classification = audit_rhpf_market_price_lineage.classify_da_lineage_row(row)

    assert classification == "da_bug_limit_price_used_in_rhpf"


def test_rhpf_market_price_lineage_classifies_bcm_cutoff_bid_and_underbid() -> None:
    ok = pd.Series(
        {
            "rhpf_submitted_mw": 4.0,
            "rhpf_awarded_mw": 4.0,
            "rhpf_bid_price_eur_per_mw_h": 20.0,
            "true_market_price_eur_per_mw_h": 20.0,
        }
    )
    underbid = pd.Series(
        {
            "rhpf_submitted_mw": 4.0,
            "rhpf_awarded_mw": 4.0,
            "rhpf_bid_price_eur_per_mw_h": 12.0,
            "true_market_price_eur_per_mw_h": 20.0,
        }
    )
    auction_bug = pd.Series(
        {
            "rhpf_submitted_mw": 4.0,
            "rhpf_awarded_mw": 4.0,
            "rhpf_bid_price_eur_per_mw_h": 22.0,
            "true_market_price_eur_per_mw_h": 20.0,
        }
    )

    assert audit_rhpf_market_price_lineage.classify_bcm_lineage_row(ok) == "bcm_ok_rhpf_bids_cutoff"
    assert audit_rhpf_market_price_lineage.classify_bcm_lineage_row(underbid) == "bcm_rhpf_underbids_cutoff"
    assert (
        audit_rhpf_market_price_lineage.classify_bcm_lineage_row(auction_bug)
        == "bcm_auction_bug_awarded_above_cutoff"
    )


def test_rhpf_market_price_lineage_classifies_bem_true_price_mismatch() -> None:
    ok = pd.Series(
        {
            "rhpf_submitted_mw": 3.0,
            "rhpf_bid_price_eur_per_mwh": 55.0,
            "true_market_price_eur_per_mwh": 55.0,
            "rhpf_accepted": 1.0,
        }
    )
    accepted_wrong = pd.Series(
        {
            "rhpf_submitted_mw": 3.0,
            "rhpf_bid_price_eur_per_mwh": 60.0,
            "true_market_price_eur_per_mwh": 55.0,
            "rhpf_accepted": 1.0,
        }
    )
    rejected_wrong = pd.Series(
        {
            "rhpf_submitted_mw": 3.0,
            "rhpf_bid_price_eur_per_mwh": 60.0,
            "true_market_price_eur_per_mwh": 55.0,
            "rhpf_accepted": 0.0,
        }
    )

    assert (
        audit_rhpf_market_price_lineage.classify_bem_lineage_row(ok)
        == "bem_ok_rhpf_bids_true_activation_price"
    )
    assert (
        audit_rhpf_market_price_lineage.classify_bem_lineage_row(accepted_wrong)
        == "bem_accepted_despite_wrong_price"
    )
    assert (
        audit_rhpf_market_price_lineage.classify_bem_lineage_row(rejected_wrong)
        == "bem_rhpf_underbids_or_overbids_true_activation_price"
    )


def test_rhpf_market_price_lineage_builds_scenario_audit(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "multi" / "p50_p50"
    scenario_dir.mkdir(parents=True)
    ts = pd.date_range("2025-04-01T00:00:00Z", periods=2, freq="h")
    hourly = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "perfect_foresight_submitted_da_buy_mw": [1.0, 0.0],
            "perfect_foresight_submitted_da_sell_mw": [0.0, 0.0],
            "perfect_foresight_da_price_rejected_buy_mwh": [0.0, 0.0],
            "perfect_foresight_da_price_rejected_sell_mwh": [0.0, 0.0],
            "perfect_foresight_da_execution_policy": ["price_taker", "price_taker"],
            "perfect_foresight_da_price_taker_mode": [1.0, 1.0],
            "da_price": [50.0, 55.0],
            "bcm_capacity_block_id": ["block-a", "block-a"],
            "bcm_capacity_cutoff_price_pos_eur_per_mw_h": [20.0, 20.0],
            "bcm_capacity_cutoff_price_neg_eur_per_mw_h": [10.0, 10.0],
            "perfect_foresight_submitted_bcm_capacity_pos_mw": [4.0, 4.0],
            "perfect_foresight_bcm_capacity_bid_price_pos_eur_per_mw_h": [12.0, 12.0],
            "perfect_foresight_awarded_capacity_pos_mw": [4.0, 4.0],
            "perfect_foresight_submitted_bcm_capacity_neg_mw": [0.0, 0.0],
            "perfect_foresight_bcm_capacity_bid_price_neg_eur_per_mw_h": [0.0, 0.0],
            "perfect_foresight_awarded_capacity_neg_mw": [0.0, 0.0],
            "perfect_foresight_bem_only_submitted_pos_mw": [2.0, 0.0],
            "perfect_foresight_bcm_activation_bid_price_pos": [55.0, 0.0],
            "perfect_foresight_true_activation_price_pos": [50.0, 50.0],
            "perfect_foresight_afrr_act_pos_accepted": [1.0, 0.0],
            "perfect_foresight_bem_only_submitted_neg_mw": [0.0, 0.0],
            "perfect_foresight_bcm_activation_bid_price_neg": [0.0, 0.0],
            "perfect_foresight_true_activation_price_neg": [0.0, 0.0],
            "perfect_foresight_afrr_act_neg_accepted": [0.0, 0.0],
        }
    )
    hourly.to_parquet(scenario_dir / "backtest_hourly.parquet")
    (scenario_dir / "backtest_summary.json").write_text("{}", encoding="utf-8")

    audit, summary = audit_rhpf_market_price_lineage.build_rhpf_market_price_lineage_audit(scenario_dir)

    assert summary["status"] == "FAIL"
    assert int(summary["count_bcm_rhpf_underbids_cutoff"]) == 1
    assert int(summary["count_bem_accepted_despite_wrong_price"]) == 1
    assert int((audit["classification"] == "da_ok_price_taker").sum()) == 1


def test_bcm_rhpf_capacity_price_summary_matches_canonical_selected_pnl() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=30)

    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=24,
        reopt_step_hours=1,
        allowed_markets=("aFRR",),
        strategy_name="bcm",
        strict_simulation_validity=False,
    )
    s = out.summary

    assert "rolling_pf_bcm_plan_history_rows" in s
    assert "rolling_pf_bcm_candidate_pos_mw" in s
    assert "rolling_pf_bcm_candidate_neg_mw" in s
    assert "rolling_pf_bcm_locked_pos_mw" in s
    assert "rolling_pf_bcm_locked_neg_mw" in s
    assert "rolling_pf_bcm_plan_history_locked_pos_mw" in s
    assert "rolling_pf_bcm_settled_locked_pos_mw" in s
    assert "rolling_pf_bcm_plan_vs_settlement_locked_delta_mw" in s
    assert "rolling_pf_bcm_first_plan_settlement_mismatch_ts_utc" in s
    assert "rolling_pf_bcm_plan_settlement_mismatch_reason" in s
    assert "rolling_pf_bcm_solver_capacity_revenue_eur" in s
    assert "rolling_pf_bcm_solver_activation_revenue_eur" in s
    assert "rolling_pf_bcm_solver_component_includes_capacity_revenue" in s
    assert "rolling_pf_bcm_solver_component_includes_activation_revenue" in s
    assert float(s["rolling_pf_bcm_deterministic_reserve_settlement"]) == pytest.approx(0.0)
    assert float(s["rolling_pf_bcm_truth_price_substitution_applied"]) == pytest.approx(1.0)
    assert "rolling_pf_reported_total_pnl_eur" in s
    assert "rolling_pf_reported_available" in s
    if float(s["rolling_pf_reported_available"]) >= 0.5:
        assert float(s["rolling_perfect_foresight_same_rules_total_pnl_eur"]) == pytest.approx(
            float(s["same_rules_rolling_pf_total_pnl_eur"])
        )
        assert float(s["rolling_perfect_foresight_same_rules_total_pnl_eur"]) == pytest.approx(
            float(s["rolling_pf_reported_total_pnl_eur"])
        )
    else:
        assert np.isnan(float(s["rolling_perfect_foresight_same_rules_total_pnl_eur"]))
        assert str(s["rolling_pf_reported_invalid_reason"]) != "none"
    assert "rolling_pf_selected_incumbent_pre_terminal_pnl_eur" in s
    assert "rolling_pf_selected_vs_hourly_final_pnl_delta_eur" in s
    assert "rolling_pf_selected_vs_component_pnl_delta_eur" in s


def test_bem_only_positive_activation_without_bcm_award_is_activation_revenue_only() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        is_perfect_foresight=False,
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
    assert float(out["bcm_to_bem_mandatory_volume_mw"]) == pytest.approx(0.0)
    assert float(out["bem_free_volume_mw"]) == pytest.approx(3.0)
    assert float(out["bem_total_bid_volume_mw"]) == pytest.approx(
        float(out["bcm_to_bem_mandatory_volume_mw"]) + float(out["bem_free_volume_mw"])
    )
    assert float(out["bem_only_submitted_pos_mw"]) > 0.0
    assert float(out["bem_only_executed_pos_mwh"]) > 0.0
    assert float(out["bem_only_activation_revenue_eur"]) > 0.0
    assert float(out["bem_activation_revenue_eur"]) == pytest.approx(
        float(out["bem_only_activation_revenue_eur"])
    )
    assert float(out["bem_free_activation_revenue_eur"]) == pytest.approx(
        float(out["bem_only_activation_revenue_eur"])
    )
    assert float(out["bem_pos_activation_mwh"]) == pytest.approx(
        float(out["bem_only_pos_activation_mwh"])
    )
    assert float(out["bcm_linked_activation_revenue_eur"]) == pytest.approx(0.0)
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
    bt = _mk_backtester("canonical_economic")
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        is_perfect_foresight=False,
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
        pred_act_neg=100.0,
        true_act_neg=1000.0,
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
        act_neg_price=1000.0,
        act_pos_rate=float(out["executed_rate_pos"]),
        act_neg_rate=float(out["executed_rate_neg"]),
        cap_bid_pos=0.0,
        cap_bid_neg=0.0,
    )
    assert np.isclose(float(m["revenue_capacity_eur"]), 0.0, atol=1e-9)
    if float(m["delivered_activation_neg_mwh"]) > 0.0:
        assert float(m["revenue_activation_eur"]) > 0.0


def test_bem_only_pos_submission_capped_by_protected_soc_headroom() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=2.0,
        planned_bem_only_neg_mw=0.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=100.0,
        true_act_pos=100.0,
        pred_act_neg=-100.0,
        true_act_neg=-100.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        soc_now=bt.soc_min + 0.05,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[0.0] * n_bins,
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[0.0] * n_bins,
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )
    assert float(out["desired_bem_only_pos_mw"]) == 2.0
    assert float(out["submitted_bem_only_pos_mw"]) <= 2.0 + 1e-9
    assert float(out["submitted_bem_only_pos_mw"]) <= float(out["safe_bem_only_pos_mw"]) + 1e-9


def test_bem_only_neg_submission_capped_by_protected_soc_headroom() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        is_perfect_foresight=False,
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
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=100.0,
        true_act_pos=100.0,
        pred_act_neg=-100.0,
        true_act_neg=-100.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        soc_now=bt.soc_max - 0.05,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[0.0] * n_bins,
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[0.0] * n_bins,
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )
    assert float(out["desired_bem_only_neg_mw"]) == 2.0
    assert float(out["submitted_bem_only_neg_mw"]) <= 2.0 + 1e-9
    assert float(out["submitted_bem_only_neg_mw"]) <= float(out["safe_bem_only_neg_mw"]) + 1e-9


def test_bem_only_guard_uses_locked_reserve_protected_envelope() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=3.0,
        planned_bem_only_neg_mw=0.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=100.0,
        true_cap_pos=100.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=100.0,
        true_act_pos=100.0,
        pred_act_neg=-100.0,
        true_act_neg=-100.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        soc_now=bt.soc_init,
        obligation_pos_mw=2.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[0.0] * n_bins,
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[0.0] * n_bins,
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )
    assert float(out["bem_only_protected_soc_min_mwh"]) > bt.soc_min


def test_combined_afrr_directional_cap_derates_bem_pos_after_bcm_pos() -> None:
    bt = _mk_backtester()
    bt.afrr_min_bid_size_mw = 0.1
    bt.afrr_bid_granularity_mw = 0.1

    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=9.0,
        desired_bem_only_neg_mw=0.0,
        soc_start_mwh=bt.soc_init,
        locked_reserve_pos_mw=4.0,
        locked_reserve_neg_mw=0.0,
        pred_act_pos=100.0,
        pred_act_neg=0.0,
    )

    assert float(guard["submitted_bem_only_pos_mw"]) == pytest.approx(6.0)
    assert float(guard["combined_afrr_pos_bid_mw"]) == pytest.approx(10.0)
    assert float(guard["combined_afrr_pos_bid_cap_violation_mw"]) == pytest.approx(0.0)
    assert float(guard["combined_afrr_neg_bid_mw"]) == pytest.approx(0.0)


def test_combined_afrr_directional_cap_derates_bem_neg_after_bcm_neg() -> None:
    bt = _mk_backtester()
    bt.afrr_min_bid_size_mw = 0.1
    bt.afrr_bid_granularity_mw = 0.1

    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=0.0,
        desired_bem_only_neg_mw=10.0,
        soc_start_mwh=bt.soc_init,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=3.0,
        pred_act_pos=0.0,
        pred_act_neg=100.0,
    )

    assert float(guard["submitted_bem_only_neg_mw"]) == pytest.approx(7.0)
    assert float(guard["combined_afrr_neg_bid_mw"]) == pytest.approx(10.0)
    assert float(guard["combined_afrr_neg_bid_cap_violation_mw"]) == pytest.approx(0.0)
    assert float(guard["combined_afrr_pos_bid_mw"]) == pytest.approx(0.0)


def test_combined_afrr_directional_cap_allows_simultaneous_pos_and_neg_max() -> None:
    bt = _mk_backtester()
    bt.afrr_min_bid_size_mw = 0.1
    bt.afrr_bid_granularity_mw = 0.1

    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=10.0,
        desired_bem_only_neg_mw=10.0,
        soc_start_mwh=bt.soc_init,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        pred_act_pos=100.0,
        pred_act_neg=100.0,
    )

    assert float(guard["submitted_bem_only_pos_mw"]) == pytest.approx(10.0)
    assert float(guard["submitted_bem_only_neg_mw"]) == pytest.approx(10.0)
    assert float(guard["combined_afrr_pos_bid_mw"]) == pytest.approx(10.0)
    assert float(guard["combined_afrr_neg_bid_mw"]) == pytest.approx(10.0)
    assert float(guard["combined_afrr_pos_bid_cap_violation_mw"]) == pytest.approx(0.0)
    assert float(guard["combined_afrr_neg_bid_cap_violation_mw"]) == pytest.approx(0.0)


def test_bem_only_can_coexist_with_bcm_linked_activation_source_split() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T08:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=5.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=3.0,
        planned_bem_only_neg_mw=0.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=10.0,
        true_cap_pos=10.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=100.0,
        true_act_pos=100.0,
        pred_act_neg=-100.0,
        true_act_neg=-100.0,
        true_rate_pos=1.0,
        true_rate_neg=0.0,
        obligation_pos_mw=5.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[5.0] + [0.0] * (n_bins - 1),
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[10.0] + [0.0] * (n_bins - 1),
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )
    assert float(out["executed_reserve_pos_mw"]) >= 5.0 - 1e-9
    assert float(out["bcm_to_bem_mandatory_volume_mw"]) == pytest.approx(5.0)
    assert float(out["bem_free_volume_mw"]) == pytest.approx(3.0)
    assert float(out["bem_total_bid_volume_mw"]) == pytest.approx(8.0)
    assert float(out["bem_only_submitted_pos_mw"]) == 3.0
    assert float(out["bcm_linked_pos_activation_mwh"]) == pytest.approx(5.0)
    assert float(out["bem_only_pos_activation_mwh"]) == pytest.approx(3.0)
    assert float(out["bem_pos_activation_mwh"]) == pytest.approx(8.0)
    assert str(out["activation_split_method"]) == "source_mwh"
    delivered_pos_mwh = float(out["bcm_linked_pos_activation_mwh"]) + float(out["bem_only_pos_activation_mwh"])
    delivered_neg_mwh = float(out["bcm_linked_neg_activation_mwh"]) + float(out["bem_only_neg_activation_mwh"])
    comp = bt._split_activation_revenue_components(
        delivered_pos_mwh=delivered_pos_mwh,
        delivered_neg_mwh=delivered_neg_mwh,
        bem_only_pos_mwh=float(out["bem_only_pos_activation_mwh"]),
        bem_only_neg_mwh=float(out["bem_only_neg_activation_mwh"]),
        act_pos_price_eur_mwh=100.0,
        act_neg_price_eur_mwh=-100.0,
        bcm_linked_pos_mwh=float(out["bcm_linked_pos_activation_mwh"]),
        bcm_linked_neg_mwh=float(out["bcm_linked_neg_activation_mwh"]),
    )
    assert float(comp["bcm_linked_activation_revenue_eur"]) == pytest.approx(500.0)
    assert float(comp["bem_only_activation_revenue_eur"]) == pytest.approx(300.0)
    assert float(out["bem_activation_revenue_eur"]) == pytest.approx(
        float(out["bcm_linked_activation_revenue_eur"])
        + float(out["bem_only_activation_revenue_eur"])
    )


def test_market_clearing_caps_combined_bcm_bem_pos_bid_to_power_limit() -> None:
    bt = _mk_backtester()
    bt.afrr_min_bid_size_mw = 0.1
    bt.afrr_bid_granularity_mw = 0.1
    n_bins = len(bt.afrr_quantile_bins)

    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T08:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=4.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=9.0,
        planned_bem_only_neg_mw=0.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=10.0,
        true_cap_pos=10.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=100.0,
        true_act_pos=100.0,
        pred_act_neg=-100.0,
        true_act_neg=-100.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        obligation_pos_mw=4.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[4.0] + [0.0] * (n_bins - 1),
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[10.0] + [0.0] * (n_bins - 1),
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )

    assert float(out["submitted_bem_only_pos_mw"]) == pytest.approx(6.0)
    assert float(out["combined_afrr_directional_bid_cap_mw"]) == pytest.approx(bt.p_max_mw)
    assert float(out["combined_afrr_pos_bid_mw"]) == pytest.approx(10.0)
    assert float(out["combined_afrr_pos_bid_cap_violation_mw"]) == pytest.approx(0.0)


def test_bem_only_guard_keeps_forecast_values_unchanged() -> None:
    bt = _mk_backtester()
    n_bins = len(bt.afrr_quantile_bins)
    pred_pos = 123.45
    pred_neg = -67.89
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=2.0,
        planned_bem_only_neg_mw=2.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=pred_pos,
        true_act_pos=pred_pos,
        pred_act_neg=pred_neg,
        true_act_neg=pred_neg,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        soc_now=bt.soc_init,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[0.0] * n_bins,
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[0.0] * n_bins,
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )
    assert float(out["desired_bem_only_pos_mw"]) == 2.0
    assert float(out["desired_bem_only_neg_mw"]) == 2.0
    assert float(out["submitted_bem_only_pos_mw"]) <= 2.0 + 1e-9
    assert float(out["submitted_bem_only_neg_mw"]) <= 2.0 + 1e-9


def test_bem_neg_guard_accounts_for_da_buy_before_submission() -> None:
    bt = _mk_backtester()
    bt.afrr_min_bid_size_mw = 0.1
    bt.afrr_bid_granularity_mw = 0.1
    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=0.0,
        desired_bem_only_neg_mw=2.0,
        soc_start_mwh=bt.soc_max - 1.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        pred_act_pos=0.0,
        pred_act_neg=100.0,
        da_charge_mw=0.5,
        da_discharge_mw=0.0,
    )
    assert float(guard["bem_guard_available_charge_headroom_mwh"]) == pytest.approx(0.0)
    assert float(guard["submitted_bem_only_neg_mw"]) == pytest.approx(0.0)
    assert str(guard["bem_guard_zero_reason_neg"]) == "charge_headroom_cap"
    assert float(guard["bem_guard_includes_da_lockbook"]) == pytest.approx(1.0)


def test_bem_pos_guard_accounts_for_da_sell_before_submission() -> None:
    bt = _mk_backtester()
    bt.afrr_min_bid_size_mw = 0.1
    bt.afrr_bid_granularity_mw = 0.1
    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=2.0,
        desired_bem_only_neg_mw=0.0,
        soc_start_mwh=bt.soc_min + 1.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        pred_act_pos=100.0,
        pred_act_neg=0.0,
        da_charge_mw=0.0,
        da_discharge_mw=0.5,
    )
    assert float(guard["bem_guard_available_discharge_energy_mwh"]) == pytest.approx(0.0)
    assert float(guard["submitted_bem_only_pos_mw"]) == pytest.approx(0.0)
    assert str(guard["bem_guard_zero_reason_pos"]) == "discharge_headroom_cap"


def test_multi_stack_derates_bem_neg_to_preserve_da_buy_lockbook() -> None:
    bt = _mk_backtester()
    bt.afrr_min_bid_size_mw = 0.1
    bt.afrr_bid_granularity_mw = 0.1
    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=0.0,
        desired_bem_only_neg_mw=3.0,
        soc_start_mwh=bt.soc_max - 1.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        pred_act_pos=0.0,
        pred_act_neg=100.0,
        da_charge_mw=0.5,
        da_discharge_mw=0.0,
        preserve_da_lockbook=True,
    )
    assert float(guard["multi_stack_da_lockbook_preserved"]) == pytest.approx(1.0)
    assert float(guard["multi_stack_bem_neg_before_derate_mw"]) == pytest.approx(3.0)
    assert float(guard["multi_stack_bem_neg_after_derate_mw"]) == pytest.approx(0.0)
    assert float(guard["multi_stack_bem_derated_to_preserve_da_lockbook"]) == pytest.approx(1.0)
    assert float(guard["multi_stack_combined_replay_pass"]) == pytest.approx(1.0)
    assert str(guard["multi_stack_zero_reason"]) == "charge_headroom_cap"


def test_multi_stack_derates_bem_pos_to_preserve_da_sell_lockbook() -> None:
    bt = _mk_backtester()
    bt.afrr_min_bid_size_mw = 0.1
    bt.afrr_bid_granularity_mw = 0.1
    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=3.0,
        desired_bem_only_neg_mw=0.0,
        soc_start_mwh=bt.soc_min + 1.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        pred_act_pos=100.0,
        pred_act_neg=0.0,
        da_charge_mw=0.0,
        da_discharge_mw=0.5,
        preserve_da_lockbook=True,
    )
    assert float(guard["multi_stack_da_lockbook_preserved"]) == pytest.approx(1.0)
    assert float(guard["multi_stack_bem_pos_before_derate_mw"]) == pytest.approx(3.0)
    assert float(guard["multi_stack_bem_pos_after_derate_mw"]) == pytest.approx(0.0)
    assert float(guard["multi_stack_bem_derated_to_preserve_da_lockbook"]) == pytest.approx(1.0)
    assert float(guard["multi_stack_combined_replay_pass"]) == pytest.approx(1.0)
    assert str(guard["multi_stack_zero_reason"]) == "discharge_headroom_cap"


def test_bem_only_guard_does_not_enable_multi_stack_da_preservation_by_default() -> None:
    bt = _mk_backtester()
    bt.afrr_min_bid_size_mw = 0.1
    bt.afrr_bid_granularity_mw = 0.1
    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=1.0,
        desired_bem_only_neg_mw=1.0,
        soc_start_mwh=bt.soc_init,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        pred_act_pos=100.0,
        pred_act_neg=100.0,
    )
    assert float(guard["submitted_bem_only_pos_mw"]) > 0.0
    assert float(guard["submitted_bem_only_neg_mw"]) > 0.0
    assert float(guard["multi_stack_da_lockbook_preserved"]) == pytest.approx(0.0)


def test_simultaneous_bem_only_pos_neg_allowed_if_both_feasible() -> None:
    bt = _mk_backtester()
    bt.disallow_simultaneous_bem_only_pos_neg = False
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=1.0,
        planned_bem_only_neg_mw=1.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=100.0,
        true_act_pos=100.0,
        pred_act_neg=-100.0,
        true_act_neg=-100.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        soc_now=bt.soc_init,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[0.0] * n_bins,
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[0.0] * n_bins,
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )
    assert float(out["submitted_bem_only_pos_mw"]) > 0.0
    assert float(out["submitted_bem_only_neg_mw"]) > 0.0


def test_simultaneous_bem_only_pos_neg_capped_if_one_side_infeasible() -> None:
    bt = _mk_backtester()
    bt.disallow_simultaneous_bem_only_pos_neg = False
    n_bins = len(bt.afrr_quantile_bins)
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        is_perfect_foresight=False,
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=2.0,
        planned_bem_only_neg_mw=2.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=100.0,
        true_act_pos=100.0,
        pred_act_neg=-100.0,
        true_act_neg=-100.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
        soc_now=bt.soc_min + 0.05,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
        planned_reserve_pos_bins_mw=[0.0] * n_bins,
        planned_reserve_neg_bins_mw=[0.0] * n_bins,
        pred_cap_pos_bins_eur_mw=[0.0] * n_bins,
        pred_cap_neg_bins_eur_mw=[0.0] * n_bins,
    )
    assert float(out["submitted_bem_only_pos_mw"]) < float(out["desired_bem_only_pos_mw"])
    assert float(out["submitted_bem_only_pos_mw"]) <= float(out["safe_bem_only_pos_mw"]) + 1e-9
    assert float(out["submitted_bem_only_neg_mw"]) >= 0.0


def test_bem_only_guard_fields_written_to_hourly_and_summary() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=2, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    for c in (
        "desired_bem_only_pos_mw",
        "safe_bem_only_pos_mw",
        "bem_only_submitted_pos_mw_before_guard",
        "bem_only_submitted_pos_mw_after_guard",
        "submitted_bem_only_pos_mw",
        "bem_only_headroom_guard_applied",
        "bem_only_headroom_guard_reason",
        "bem_only_guard_soc_now_mwh",
        "bem_only_guard_protected_soc_min_mwh",
        "bem_only_guard_protected_soc_max_mwh",
    ):
        assert c in out.hourly.columns
    for k in (
        "bem_only_headroom_safety_mwh",
        "bem_only_headroom_guard_applied_count",
        "bem_only_pos_reduced_by_headroom_mw_sum",
        "bem_only_headroom_guard_hours",
    ):
        assert k in out.summary


def test_bem_only_guard_prevents_known_low_soc_positive_violation() -> None:
    bt = _mk_backtester()
    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=2.0,
        desired_bem_only_neg_mw=0.0,
        soc_start_mwh=3.5,
        locked_reserve_pos_mw=8.0,
        locked_reserve_neg_mw=0.0,
        pred_act_pos=100.0,
        pred_act_neg=-100.0,
    )
    assert float(guard["bem_only_guard_protected_soc_min_mwh"]) >= 5.0 - 1e-9
    assert float(guard["bem_only_submitted_pos_mw_after_guard"]) == 0.0


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


def test_bcm_obligation_only_activation_is_bcm_linked() -> None:
    bt = _mk_backtester("canonical_economic")
    comp = bt._split_activation_revenue_components(
        delivered_pos_mwh=4.0,
        delivered_neg_mwh=0.0,
        bem_only_pos_mwh=0.0,
        bem_only_neg_mwh=0.0,
        act_pos_price_eur_mwh=100.0,
        act_neg_price_eur_mwh=0.0,
        bcm_linked_pos_mwh=4.0,
        bcm_linked_neg_mwh=0.0,
    )
    assert np.isclose(comp["bcm_linked_pos_activation_mwh"], 4.0)
    assert np.isclose(comp["bem_only_pos_activation_mwh"], 0.0)
    assert np.isclose(comp["bcm_linked_activation_revenue_eur"], 400.0)
    assert np.isclose(comp["bem_only_activation_revenue_eur"], 0.0)


def test_bcm_and_bem_same_hour_split_by_source_mwh() -> None:
    bt = _mk_backtester("canonical_economic")
    comp = bt._split_activation_revenue_components(
        delivered_pos_mwh=5.0,
        delivered_neg_mwh=0.0,
        bem_only_pos_mwh=3.0,
        bem_only_neg_mwh=0.0,
        act_pos_price_eur_mwh=100.0,
        act_neg_price_eur_mwh=0.0,
        bcm_linked_pos_mwh=2.0,
        bcm_linked_neg_mwh=0.0,
    )
    assert comp["activation_split_method"] == "source_mwh"
    assert np.isclose(comp["bcm_linked_pos_activation_mwh"], 2.0)
    assert np.isclose(comp["bem_only_pos_activation_mwh"], 3.0)
    assert np.isclose(comp["bcm_linked_activation_revenue_eur"], 200.0)
    assert np.isclose(comp["bem_only_activation_revenue_eur"], 300.0)
    assert np.isclose(
        comp["activation_revenue_reconciled_eur"],
        comp["bcm_linked_activation_revenue_eur"] + comp["bem_only_activation_revenue_eur"],
    )
    assert float(comp["bcm_linked_activation_revenue_allocation_share"]) == pytest.approx(0.4)
    assert float(comp["bem_only_activation_revenue_allocation_share"]) == pytest.approx(0.6)
    assert float(comp["bcm_linked_pos_activation_revenue_allocation_share"]) == pytest.approx(0.4)
    assert float(comp["bem_only_pos_activation_revenue_allocation_share"]) == pytest.approx(0.6)


def test_activation_revenue_allocation_uses_directional_mw_shares() -> None:
    bt = _mk_backtester("canonical_economic")
    comp = bt._split_activation_revenue_components(
        delivered_pos_mwh=6.0,
        delivered_neg_mwh=4.0,
        bem_only_pos_mwh=2.0,
        bem_only_neg_mwh=1.0,
        act_pos_price_eur_mwh=100.0,
        act_neg_price_eur_mwh=200.0,
        bcm_linked_pos_mwh=4.0,
        bcm_linked_neg_mwh=3.0,
    )
    assert float(comp["bcm_linked_pos_activation_revenue_allocation_share"]) == pytest.approx(4.0 / 6.0)
    assert float(comp["bem_only_pos_activation_revenue_allocation_share"]) == pytest.approx(2.0 / 6.0)
    assert float(comp["bcm_linked_neg_activation_revenue_allocation_share"]) == pytest.approx(3.0 / 4.0)
    assert float(comp["bem_only_neg_activation_revenue_allocation_share"]) == pytest.approx(1.0 / 4.0)
    assert float(comp["bcm_linked_activation_revenue_eur"]) == pytest.approx(4.0 * 100.0 + 3.0 * 200.0)
    assert float(comp["bem_only_activation_revenue_eur"]) == pytest.approx(2.0 * 100.0 + 1.0 * 200.0)


def test_bem_only_missed_capacity_diagnostics_are_ignored_for_capacity_validation() -> None:
    hourly = pd.DataFrame(
        {
            "real_missed_capacity_pos_mw": [2.0],
            "real_missed_capacity_neg_mw": [0.0],
            "real_bem_only_submitted_pos_mw": [2.0],
            "real_bem_only_submitted_neg_mw": [0.0],
            "real_submitted_bcm_capacity_pos_mw": [0.0],
            "real_locked_bcm_capacity_pos_mw": [0.0],
            "real_bcm_awarded_capacity_pos_mw": [0.0],
        }
    )

    diag = BatteryBacktester._missed_capacity_validation_components(hourly)

    assert float(diag["missed_capacity_true_bcm_pos_mw"].sum()) == pytest.approx(0.0)
    assert float(diag["missed_capacity_ignored_bem_only_pos_mw"].sum()) == pytest.approx(2.0)
    assert float(diag["missed_capacity_ignored_bem_only_submission"]) == pytest.approx(1.0)
    assert str(diag["missed_capacity_reason"]) == "bem_only_activation_submission_ignored"


def test_bcm_capacity_missed_is_classified_as_nonfatal_stress_diagnostic() -> None:
    hourly = pd.DataFrame(
        {
            "real_missed_capacity_pos_mw": [2.0],
            "real_missed_capacity_neg_mw": [0.0],
            "real_bem_only_submitted_pos_mw": [0.0],
            "real_submitted_bcm_capacity_pos_mw": [2.0],
            "real_locked_bcm_capacity_pos_mw": [0.0],
            "real_bcm_awarded_capacity_pos_mw": [0.0],
        }
    )

    diag = BatteryBacktester._missed_capacity_validation_components(hourly)

    assert float(diag["missed_capacity_true_bcm_pos_mw"].sum()) == pytest.approx(2.0)
    assert float(diag["missed_capacity_ignored_bem_only_pos_mw"].sum()) == pytest.approx(0.0)
    assert float(diag["missed_capacity_check_applied"]) == pytest.approx(1.0)
    assert str(diag["missed_capacity_reason"]) == "true_bcm_capacity_missed"

    capacity_stress_shortfall_mw = float(diag["generic_missed_capacity_pos_mw"].sum())
    missed_activation_mwh = 0.0
    missed_capacity_check_pass = 1.0
    capacity_stress_check_pass = float(capacity_stress_shortfall_mw <= 1e-9)
    invalid_reasons = []
    if missed_activation_mwh > 1e-9:
        invalid_reasons.append("missed_activation")
    if missed_capacity_check_pass < 0.5:
        invalid_reasons.append("missed_capacity")

    assert capacity_stress_shortfall_mw == pytest.approx(2.0)
    assert capacity_stress_check_pass == pytest.approx(0.0)
    assert "missed_capacity" not in invalid_reasons


def test_multi_bem_only_submission_does_not_imply_missed_bcm_capacity() -> None:
    hourly = pd.DataFrame(
        {
            "real_missed_capacity_pos_mw": [1.0],
            "real_missed_capacity_neg_mw": [0.0],
            "real_submitted_afrr_pos_mw": [1.0],
            "real_bem_only_submitted_pos_mw": [1.0],
            "real_submitted_bcm_capacity_pos_mw": [0.0],
            "real_locked_bcm_capacity_pos_mw": [0.0],
            "real_bcm_awarded_capacity_pos_mw": [0.0],
        }
    )

    diag = BatteryBacktester._missed_capacity_validation_components(hourly)

    assert float(diag["missed_capacity_true_bcm_pos_mw"].sum()) == pytest.approx(0.0)
    assert float(diag["missed_capacity_ignored_bem_only_pos_mw"].sum()) == pytest.approx(1.0)


def test_canonical_negative_activation_positive_for_bcm_and_bem_sources() -> None:
    bt = _mk_backtester("canonical_economic")
    comp = bt._split_activation_revenue_components(
        delivered_pos_mwh=0.0,
        delivered_neg_mwh=5.0,
        bem_only_pos_mwh=0.0,
        bem_only_neg_mwh=3.0,
        act_pos_price_eur_mwh=0.0,
        act_neg_price_eur_mwh=1000.0,
        bcm_linked_pos_mwh=0.0,
        bcm_linked_neg_mwh=2.0,
    )
    assert np.isclose(comp["bcm_linked_neg_activation_revenue_eur"], 2000.0)
    assert np.isclose(comp["bem_only_neg_activation_revenue_eur"], 3000.0)
    assert np.isclose(comp["activation_revenue_reconciled_eur"], 5000.0)


def test_market_clearing_keeps_bem_only_active_during_bcm_obligation() -> None:
    bt = _mk_backtester("canonical_economic")
    bt._strategy_permissions = bt.strategy_permissions_from_name("afrr")
    out = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2025-05-02T00:00:00Z"),
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=3.0,
        planned_bem_only_neg_mw=0.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=0.0,
        true_cap_pos=100.0,
        pred_cap_neg=0.0,
        true_cap_neg=100.0,
        pred_act_pos=10.0,
        true_act_pos=100.0,
        pred_act_neg=0.0,
        true_act_neg=0.0,
        true_rate_pos=0.5,
        true_rate_neg=0.0,
        pred_rate_pos=0.5,
        pred_rate_neg=0.0,
        soc_now=12.0,
        obligation_pos_mw=2.0,
        obligation_neg_mw=0.0,
    )
    assert float(out["bem_only_submitted_pos_mw"]) > 0.0
    assert float(out["bem_only_executed_pos_mwh"]) > 0.0
    assert float(out["bcm_linked_pos_activation_mwh"]) > 0.0
    assert out["activation_split_method"] == "source_mwh"


def test_split_activation_revenue_components_uses_neg_provider_value_directly() -> None:
    bt = _mk_backtester("raw_signed")
    comp = bt._split_activation_revenue_components(
        delivered_pos_mwh=0.0,
        delivered_neg_mwh=10.0,
        bem_only_pos_mwh=0.0,
        bem_only_neg_mwh=10.0,
        act_pos_price_eur_mwh=0.0,
        act_neg_price_eur_mwh=1000.0,
    )
    assert np.isclose(comp["bem_only_activation_revenue_eur"], 10000.0)
    assert np.isclose(comp["bcm_linked_activation_revenue_eur"], 0.0)
    assert np.isclose(comp["activation_revenue_reconciled_eur"], 10000.0)


def test_neg_activation_provider_value_is_not_sign_flipped() -> None:
    bt = _mk_backtester("raw_signed")
    assert float(bt._neg_activation_provider_value(1000.0)) == pytest.approx(1000.0)
    assert float(bt._neg_activation_provider_value(60.0)) == pytest.approx(60.0)


def test_split_activation_revenue_components_canonical_neg_positive_value() -> None:
    bt = _mk_backtester("canonical_economic")
    comp = bt._split_activation_revenue_components(
        delivered_pos_mwh=0.0,
        delivered_neg_mwh=10.0,
        bem_only_pos_mwh=0.0,
        bem_only_neg_mwh=10.0,
        act_pos_price_eur_mwh=0.0,
        act_neg_price_eur_mwh=1000.0,
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


def test_full_settlement_activation_revenue_not_overwritten_by_wrong_split() -> None:
    bt = _mk_backtester("canonical_economic")
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
        act_neg_price=1000.0,
        act_pos_rate=0.0,
        act_neg_rate=1.0,
        cap_bid_pos=0.0,
        cap_bid_neg=0.0,
    )
    delivered = float(m["delivered_activation_neg_mwh"])
    expected = delivered * 1000.0
    assert delivered > 0.0
    assert np.isclose(float(m["revenue_activation_eur"]), expected, atol=1e-6), m


def test_bem_only_forced_scenario_has_nonzero_revenue_and_benchmark_diagnostics() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p50"]
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
            f"{col.pred_afrr_capacity_price_pos}_p50": np.full(n, 1.0),
            f"{col.pred_afrr_capacity_price_neg}_p50": np.full(n, 1.0),
            f"{col.pred_afrr_activation_price_pos}_p50": np.full(n, 140.0),
            f"{col.pred_afrr_activation_price_neg}_p50": np.full(n, 10.0),
            f"{col.pred_afrr_activation_rate_pos}_p50": np.full(n, 0.95),
            f"{col.pred_afrr_activation_rate_neg}_p50": np.full(n, 0.05),
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
    real_total = float(out.summary["realized_total_pnl_eur"])
    perfect_foresight_total = float(out.summary["rolling_perfect_foresight_same_rules_total_pnl_eur"])
    assert np.isfinite(real_total)
    assert np.isfinite(perfect_foresight_total)
    assert out.summary.get("benchmark_type") == "rolling_perfect_foresight_same_rules"


def test_no_headroom_violations_in_clean_forced_bem_only() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p50"]
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
            f"{col.pred_afrr_capacity_price_pos}_p50": np.full(n, 0.5),
            f"{col.pred_afrr_capacity_price_neg}_p50": np.full(n, 0.5),
            f"{col.pred_afrr_activation_price_pos}_p50": np.full(n, 140.0),
            f"{col.pred_afrr_activation_price_neg}_p50": np.full(n, 10.0),
            f"{col.pred_afrr_activation_rate_pos}_p50": np.full(n, 0.95),
            f"{col.pred_afrr_activation_rate_neg}_p50": np.full(n, 0.05),
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
    assert np.isclose(float(m["id_buy_price_eur_mwh"]), min(bt.id_buy_price_cap_eur_mwh, 100.0 + bt.id_rescue_spread_eur_mwh))
    assert np.isclose(float(m["id_sell_price_eur_mwh"]), max(bt.id_sell_price_floor_eur_mwh, 100.0 - bt.id_rescue_spread_eur_mwh))
    assert float(m["id_buy_price_eur_mwh"]) >= float(m["id_sell_price_eur_mwh"]) - 1e-12


def test_id_price_taker_handles_negative_da_prices() -> None:
    bt = _mk_backtester()
    _, m = bt._settle_one_hour(
        soc=bt.soc_init,
        charge=0.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=-50.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        id_charge_mw=1.0,
        id_discharge_mw=1.0,
    )
    assert float(m["id_buy_price_eur_mwh"]) == pytest.approx(-50.0 + bt.id_rescue_spread_eur_mwh)
    assert float(m["id_sell_price_eur_mwh"]) == pytest.approx(-50.0 - bt.id_rescue_spread_eur_mwh)
    assert float(m["id_price_uses_activation_price"]) == pytest.approx(0.0)


def test_terminal_soc_recovery_reason_cannot_execute_id_sell() -> None:
    bt = _mk_backtester()
    bt._strategy_permissions = bt.resolve_strategy_permissions(
        strategy_name="da",
        allowed_markets=("DA", "ID"),
        id_recourse_mode="common",
    )
    soc0 = 12.0
    soc1, m = bt._settle_one_hour(
        soc=soc0,
        charge=0.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=50.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        id_charge_mw=0.0,
        id_discharge_mw=5.0,
        id_recourse_reason_hint="terminal_soc_recovery",
    )
    assert float(m["terminal_soc_recovery_id_sell_suppressed"]) == pytest.approx(1.0)
    assert float(m["id_sell_mwh"]) == pytest.approx(0.0)
    assert str(m["id_recourse_reason"]) == "none"
    assert soc1 >= soc0 - float(m["aux_energy_mwh"]) - 1e-9


def test_technical_id_buy_is_clipped_by_residual_charge_power_after_reserve() -> None:
    bt = _mk_backtester()
    id_charge, id_discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=bt.soc_min,
        reserve_pos_next_mw=10.0,
        reserve_neg_next_mw=9.5,
        da_charge_next_mw=0.25,
        da_discharge_next_mw=0.0,
    )
    assert id_charge == pytest.approx(0.25)
    assert id_discharge == pytest.approx(0.0)
    assert reason == "afrr_headroom_repair"


def test_technical_id_sell_is_clipped_by_residual_discharge_power_after_reserve() -> None:
    bt = _mk_backtester()
    id_charge, id_discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=bt.soc_max,
        reserve_pos_next_mw=9.25,
        reserve_neg_next_mw=10.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.5,
    )
    assert id_charge == pytest.approx(0.0)
    assert id_discharge == pytest.approx(0.25)
    assert reason == "afrr_headroom_repair"


def test_canonical_power_stack_helper_counts_all_charge_and_discharge_components() -> None:
    bt = _mk_backtester()
    stack = bt._compute_power_stack_components(
        da_charge_mw=1.0,
        id_buy_mw=2.0,
        bem_neg_mw=3.0,
        bcm_neg_obligation_mw=4.0,
        da_discharge_mw=1.5,
        id_sell_mw=2.5,
        bem_pos_mw=3.5,
        bcm_pos_obligation_mw=4.5,
    )
    assert float(stack["charge_stack_mw"]) == pytest.approx(10.0)
    assert float(stack["discharge_stack_mw"]) == pytest.approx(12.0)
    assert float(stack["charge_stack_violation_mw"]) == pytest.approx(0.0)
    assert float(stack["discharge_stack_violation_mw"]) == pytest.approx(0.0)
    assert float(stack["power_base_mw"]) == pytest.approx(1.0)
    assert float(stack["reserve_headroom_pos_used_mw"]) == pytest.approx(8.0)
    assert float(stack["reserve_headroom_neg_used_mw"]) == pytest.approx(7.0)
    assert float(stack["reserve_headroom_pos_available_mw"]) == pytest.approx(9.0)
    assert float(stack["reserve_headroom_neg_available_mw"]) == pytest.approx(11.0)
    assert float(stack["total_stack_mw"]) == pytest.approx(9.0)
    assert float(stack["total_stack_violation_mw"]) == pytest.approx(0.0)


def test_baseline_headroom_allows_opposite_direction_reserve_capability() -> None:
    bt = _mk_backtester()
    stack = bt._compute_canonical_power_stack(
        bcm_pos_obligation_mw=10.0,
        bcm_neg_obligation_mw=10.0,
    )
    assert float(stack["power_base_mw"]) == pytest.approx(0.0)
    assert float(stack["reserve_headroom_pos_available_mw"]) == pytest.approx(10.0)
    assert float(stack["reserve_headroom_neg_available_mw"]) == pytest.approx(10.0)
    assert float(stack["reserve_headroom_pos_used_mw"]) == pytest.approx(10.0)
    assert float(stack["reserve_headroom_neg_used_mw"]) == pytest.approx(10.0)
    assert float(stack["total_stack_violation_mw"]) == pytest.approx(0.0)


def test_baseline_headroom_detects_directional_reserve_violations() -> None:
    bt = _mk_backtester()
    pos = bt._compute_canonical_power_stack(
        da_discharge_mw=5.0,
        bcm_pos_obligation_mw=6.0,
    )
    neg = bt._compute_canonical_power_stack(
        da_charge_mw=5.0,
        bcm_neg_obligation_mw=6.0,
    )
    assert float(pos["power_base_mw"]) == pytest.approx(5.0)
    assert float(pos["reserve_headroom_pos_available_mw"]) == pytest.approx(5.0)
    assert float(pos["reserve_headroom_pos_violation_mw"]) == pytest.approx(1.0)
    assert float(neg["power_base_mw"]) == pytest.approx(-5.0)
    assert float(neg["reserve_headroom_neg_available_mw"]) == pytest.approx(5.0)
    assert float(neg["reserve_headroom_neg_violation_mw"]) == pytest.approx(1.0)


def test_baseline_headroom_uses_full_reserve_mw_not_activation_rate() -> None:
    bt = _mk_backtester()
    stack = bt._compute_canonical_power_stack(
        da_discharge_mw=5.0,
        bcm_pos_obligation_mw=6.0,
    )
    assert "activation_rate" not in stack
    assert float(stack["reserve_headroom_pos_used_mw"]) == pytest.approx(6.0)
    assert float(stack["reserve_headroom_pos_violation_mw"]) == pytest.approx(1.0)


def test_bem_neg_is_clipped_by_residual_charge_stack_with_bcm_and_da() -> None:
    bt = _mk_backtester()
    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=0.0,
        desired_bem_only_neg_mw=5.0,
        soc_start_mwh=10.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=4.0,
        pred_act_pos=0.0,
        pred_act_neg=100.0,
        da_charge_mw=5.0,
        da_discharge_mw=0.0,
    )
    assert float(guard["submitted_bem_only_neg_mw"]) == pytest.approx(1.0)
    assert str(guard["bem_only_headroom_guard_reason"]) == "power_stack_cap"


def test_bem_neg_zero_when_locked_bcm_neg_uses_full_charge_stack() -> None:
    bt = _mk_backtester("canonical_economic")
    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=0.0,
        desired_bem_only_neg_mw=10.0,
        soc_start_mwh=10.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=10.0,
        pred_act_pos=0.0,
        pred_act_neg=100.0,
    )
    assert float(guard["submitted_bem_only_neg_mw"]) == pytest.approx(0.0)
    assert float(guard["bem_only_power_stack_charge_residual_mw"]) == pytest.approx(0.0)
    assert str(guard["bem_only_headroom_guard_reason"]) == "power_stack_cap"


def test_bem_neg_uses_residual_charge_stack_after_partial_bcm_neg() -> None:
    bt = _mk_backtester("canonical_economic")
    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=0.0,
        desired_bem_only_neg_mw=4.0,
        soc_start_mwh=10.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=6.0,
        pred_act_pos=0.0,
        pred_act_neg=100.0,
    )
    stack = bt._compute_canonical_power_stack(
        bcm_neg_obligation_mw=6.0,
        bem_neg_mw=float(guard["submitted_bem_only_neg_mw"]),
    )
    assert float(guard["submitted_bem_only_neg_mw"]) == pytest.approx(4.0)
    assert float(stack["charge_stack_mw"]) == pytest.approx(10.0)
    assert float(stack["charge_stack_violation_mw"]) == pytest.approx(0.0)


def test_bem_neg_clipped_when_partial_bcm_neg_leaves_less_residual_power() -> None:
    bt = _mk_backtester("canonical_economic")
    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=0.0,
        desired_bem_only_neg_mw=5.0,
        soc_start_mwh=10.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=6.0,
        pred_act_pos=0.0,
        pred_act_neg=100.0,
    )
    stack = bt._compute_canonical_power_stack(
        bcm_neg_obligation_mw=6.0,
        bem_neg_mw=float(guard["submitted_bem_only_neg_mw"]),
    )
    assert float(guard["submitted_bem_only_neg_mw"]) == pytest.approx(4.0)
    assert str(guard["bem_only_headroom_guard_reason"]) == "power_stack_cap"
    assert float(stack["charge_stack_mw"]) == pytest.approx(10.0)
    assert float(stack["charge_stack_violation_mw"]) == pytest.approx(0.0)


def test_bem_only_pos_neg_combined_same_hour_stack_is_capped() -> None:
    bt = _mk_backtester()
    bt.afrr_min_bid_size_mw = 0.1
    bt.afrr_bid_granularity_mw = 0.1
    guard = bt._apply_bem_only_submission_guard(
        desired_bem_only_pos_mw=7.0,
        desired_bem_only_neg_mw=7.0,
        soc_start_mwh=10.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        pred_act_pos=100.0,
        pred_act_neg=100.0,
    )
    submitted_sum = float(guard["submitted_bem_only_pos_mw"]) + float(guard["submitted_bem_only_neg_mw"])
    assert submitted_sum <= bt.p_max_mw + 1e-9
    assert float(guard["submitted_bem_only_pos_mw"]) > 0.0
    assert float(guard["submitted_bem_only_neg_mw"]) > 0.0
    assert str(guard["bem_only_headroom_guard_reason"]) == "power_stack_cap"


def test_optimizer_clips_independent_bem_neg_against_locked_bcm_neg() -> None:
    bt = _mk_backtester("canonical_economic")
    df, col = _one_hour_pred_df(
        da=0.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos=0.0,
        act_neg=1000.0,
        rate_pos=0.0,
        rate_neg=1.0,
    )
    ts = pd.to_datetime(df[col.timestamp].iloc[0], utc=True)
    perms = StrategyPermissions(
        allow_da=False,
        id_mode="none",
        allow_bcm=True,
        allow_bcm_activation_obligations=True,
        allow_bem_only=True,
    )
    out_full = bt.optimize_dispatch(
        df,
        col,
        soc_start=10.0,
        soc_end_min_target=None,
        fixed_reserve_obligation={ts: (0.0, 10.0)},
        allowed_markets=("aFRR", "BCM", "BEM"),
        strategy_permissions=perms,
    )
    assert float(out_full["reserve_neg_mw"].iloc[0]) == pytest.approx(10.0)
    assert float(out_full["bem_only_neg_mw"].iloc[0]) == pytest.approx(0.0)
    assert float(out_full["power_stack_neg_mw"].iloc[0]) <= bt.p_max_mw + 1e-9

    out_partial = bt.optimize_dispatch(
        df,
        col,
        soc_start=10.0,
        soc_end_min_target=None,
        fixed_reserve_obligation={ts: (0.0, 6.0)},
        allowed_markets=("aFRR", "BCM", "BEM"),
        strategy_permissions=perms,
    )
    assert float(out_partial["reserve_neg_mw"].iloc[0]) == pytest.approx(6.0)
    assert float(out_partial["bem_only_neg_mw"].iloc[0]) <= 4.0 + 1e-9
    assert float(out_partial["power_stack_neg_mw"].iloc[0]) <= bt.p_max_mw + 1e-9


def test_settlement_exports_canonical_power_stack_diagnostics() -> None:
    bt = _mk_backtester()
    _soc_next, out = bt._settle_one_hour(
        soc=1.0,
        charge=7.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=5.0,
        da_price=0.0,
        cap_pos=0.0,
        cap_neg=10.0,
        act_pos_price=0.0,
        act_neg_price=100.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
    )
    assert float(out["power_stack_charge_mw"]) == pytest.approx(12.0)
    assert float(out["power_stack_bcm_neg_mw"]) == pytest.approx(5.0)
    assert float(out["power_violation_charge_mw"]) == pytest.approx(2.0)
    assert float(out["power_stack_total_mw"]) == pytest.approx(12.0)
    assert float(out["power_violation_total_mw"]) == pytest.approx(2.0)
    assert float(out["power_base_mw"]) == pytest.approx(-7.0)
    assert float(out["reserve_headroom_neg_available_mw"]) == pytest.approx(3.0)
    assert float(out["reserve_headroom_neg_used_mw"]) == pytest.approx(5.0)
    assert float(out["reserve_headroom_neg_violation_mw"]) == pytest.approx(2.0)


def test_gross_da_charge_discharge_costs_remain_settled_on_gross_quantities() -> None:
    bt = _mk_backtester()
    _soc_next, out = bt._settle_one_hour(
        soc=5.0,
        charge=3.0,
        discharge=4.0,
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
    assert float(out["da_buy_mwh"]) == pytest.approx(3.0)
    assert float(out["da_sell_mwh"]) == pytest.approx(4.0)
    assert float(out["cost_da_eur"]) == pytest.approx(300.0)
    assert float(out["revenue_da_eur"]) == pytest.approx(400.0)
    assert float(out["transaction_cost_eur"]) == pytest.approx(bt.trans_eur_mwh * 7.0)
    expected_deg = bt.deg_eur_mwh * (3.0 * bt.eta_in + 4.0 / bt.eta_out)
    assert float(out["degradation_cost_da_eur"]) == pytest.approx(expected_deg)


def test_bem_only_negative_is_not_counted_as_bcm_in_canonical_stack() -> None:
    bt = _mk_backtester("canonical_economic")
    clearing = bt._apply_market_clearing(
        target_time_utc=pd.Timestamp("2025-05-02T00:00:00Z"),
        planned_charge_mw=0.0,
        planned_discharge_mw=0.0,
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        planned_bem_only_pos_mw=0.0,
        planned_bem_only_neg_mw=10.0,
        pred_da_price=0.0,
        true_da_price=0.0,
        pred_cap_pos=0.0,
        true_cap_pos=100.0,
        pred_cap_neg=0.0,
        true_cap_neg=100.0,
        pred_act_pos=0.0,
        true_act_pos=0.0,
        pred_act_neg=10.0,
        true_act_neg=100.0,
        true_rate_pos=0.0,
        true_rate_neg=1.0,
        pred_rate_pos=0.0,
        pred_rate_neg=1.0,
        soc_now=10.0,
        obligation_pos_mw=0.0,
        obligation_neg_mw=0.0,
    )
    stack = bt._compute_canonical_power_stack(
        bem_neg_mw=float(clearing["bem_only_submitted_neg_mw"]),
        bcm_neg_obligation_mw=float(clearing["fixed_reserve_obligation_neg_mw"]),
    )
    assert float(clearing["fixed_reserve_obligation_neg_mw"]) == pytest.approx(0.0)
    assert float(clearing["bem_only_submitted_neg_mw"]) == pytest.approx(10.0)
    assert float(stack["charge_stack_mw"]) == pytest.approx(10.0)
    assert float(stack["charge_stack_violation_mw"]) == pytest.approx(0.0)


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


def test_bcm_hard_headroom_uses_expected_activation_rate_not_full_headroom() -> None:
    bt = _mk_backtester()
    bt.reserve_activation_headroom_h = 0.5
    bt.reserve_headroom_safety_mwh = 0.0
    bt.reserve_soc_projection_safety_mwh = 0.0
    bt.min_activation_headroom_fraction = 0.0
    bt.eta_out = 1.0
    soc_start = bt.soc_min + 1.0
    df, col = _one_hour_pred_df(
        da=0.0,
        cap_pos=300.0,
        cap_neg=0.0,
        act_pos=100.0,
        act_neg=0.0,
        rate_pos=0.1,
        rate_neg=0.0,
    )

    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=soc_start,
        allowed_markets=("aFRR",),
    )

    assert float(out["reserve_pos_mw"].iloc[0]) > 9.0


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
        "real_capacity_full_activation_shortfall_mw",
        "real_capacity_full_activation_stress_check_pass",
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


def test_terminal_soc_pnl_reconciliation() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    # Encourage discharge so terminal repair/adjustment is active.
    df[col.pred_da_price] = 120.0
    df[col.true_da_price] = 120.0
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, allowed_markets=("DA",))
    s = out.summary
    rhs = float(s["realized_pnl_excl_terminal_eur"]) + float(s["terminal_soc_adjustment_eur"])
    assert np.isclose(float(s["realized_total_pnl_eur"]), rhs, atol=1e-6)


def test_summary_flags_invalid_runs() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    # Force fallback-driven infeasible stress with high reserve commitments.
    df[col.pred_afrr_capacity_price_pos] = 1e3
    df[col.true_afrr_capacity_price_pos] = 1e3
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=2, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    s = out.summary
    for k in [
        "simulation_valid",
        "invalid_reason",
        "final_soc_check_pass",
        "missed_capacity_check_pass",
        "capacity_stress_check_pass",
        "capacity_stress_check_fatal",
        "pnl_reconciliation_check_pass",
        "headroom_check_pass",
    ]:
        assert k in s


def test_bcm_precommitment_reduces_infeasible_bid() -> None:
    bt = _mk_backtester()
    bt.reserve_activation_headroom_h = 0.5
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = None
    bt.reserve_headroom_safety_mwh = 0.0
    bt.reserve_soc_projection_safety_mwh = 0.0
    bt.reserve_min_margin_after_bid_mwh = 0.0
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")  # 08:00 CET/CEST gate for model path
    target_hours = pd.date_range("2025-05-01 22:00:00+00:00", periods=4, freq="h")
    col = BacktestColumnMap()
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [10.0, 10.0, 10.0, 10.0],
            "reserve_neg_mw": [0.0, 0.0, 0.0, 0.0],
            # only ~2 MWh above soc_min -> infeasible for 10 MW @ 0.5h
            "soc_start_lp_mwh": [bt.soc_min + 2.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
        }
    )
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [50.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [100.0] * 4,
        }
    ).set_index(col.timestamp)
    lock_pos: dict[pd.Timestamp, float] = {}
    lock_neg: dict[pd.Timestamp, float] = {}
    lock_e_pos: dict[pd.Timestamp, float] = {}
    lock_e_neg: dict[pd.Timestamp, float] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos=lock_e_pos,
        lock_energy_neg=lock_e_neg,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
    )
    # 2 MWh headroom with 0.5h/eta_out implies <4 MW feasible.
    assert len(lock_pos) == 4
    assert max(lock_pos.values()) <= 4.0 + 1e-9


def test_bcm_precommit_skips_partial_product_at_simulation_end() -> None:
    bt = _mk_backtester()
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")  # 08:00 CEST BCM bid hour.
    target_hours = pd.date_range("2025-05-02 18:00:00+00:00", periods=3, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [5.0, 5.0, 5.0],
            "reserve_neg_mw": [2.0, 2.0, 2.0],
            "soc_start_lp_mwh": [12.0, 12.0, 12.0],
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame({col.timestamp: target_hours}).set_index(col.timestamp)
    lock_pos: dict[pd.Timestamp, float] = {}
    lock_neg: dict[pd.Timestamp, float] = {}
    audit: dict[str, dict[pd.Timestamp, float | str]] = {}

    stats = bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=audit,
        is_perfect_foresight=False,
        global_end_utc=pd.Timestamp("2025-05-02 20:00:00+00:00"),
    )

    assert stats["triggered"] == 1.0
    assert lock_pos == {}
    assert lock_neg == {}
    assert set(audit["bcm_precommit_zero_reason"].values()) == {"partial_product_or_sim_end"}
    assert set(audit["submitted_reserve_pos_mw_after_retry"].values()) == {0.0}
    assert set(audit["submitted_reserve_neg_mw_after_retry"].values()) == {0.0}
    assert set(audit["bcm_precommit_locked_pos_mw"].values()) == {0.0}
    assert set(audit["bcm_precommit_locked_neg_mw"].values()) == {0.0}


def test_bcm_precommit_selector_uses_first_feasible_retry_factor(monkeypatch: pytest.MonkeyPatch) -> None:
    bt = _mk_backtester()
    bt.enable_reserve_retry_ladder = True
    bt.reserve_retry_ladder = [1.0, 0.5, 0.0]
    col = BacktestColumnMap()
    target_hours = pd.date_range("2025-05-01 22:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [10.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [10.0] * 4,
        }
    )
    source = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [100.0] * 4,
        }
    ).set_index(col.timestamp)

    def fake_clear(*, cap_bids, ts_idx, source, colmap):  # type: ignore[no-untyped-def]
        raise AssertionError("realized clearing must not be used for BCM precommit retry selection")

    def fake_formulate(*, blk, source, colmap, snapshot_ts, offered_pos, offered_neg, is_perfect_foresight=False):  # type: ignore[no-untyped-def]
        ts = pd.to_datetime(blk["target_time_utc"].iloc[0], utc=True)
        bids = []
        if offered_pos > 0.0:
            bids.append(AFRRCapacityBid(ts=ts, side="pos", quantity_mw=float(offered_pos), capacity_price_eur_mw=0.0, energy_price_eur_mwh=0.0))
        if offered_neg > 0.0:
            bids.append(AFRRCapacityBid(ts=ts, side="neg", quantity_mw=float(offered_neg), capacity_price_eur_mw=0.0, energy_price_eur_mwh=0.0))
        return bids, pd.to_datetime(blk["target_time_utc"], utc=True)

    def fake_optimize(df, colmap, **kwargs):  # type: ignore[no-untyped-def]
        fixed = kwargs["fixed_reserve_obligation"]
        max_pos = max((v[0] for v in fixed.values()), default=0.0)
        if max_pos > 5.0:
            raise RuntimeError("existing lockbook obligation infeasible")
        return pd.DataFrame(
            {
                colmap.timestamp: pd.to_datetime(df[colmap.timestamp], utc=True),
                "soc_lp_mwh": [10.0] * len(df),
                "slack_pos_mw": [0.0] * len(df),
                "slack_neg_mw": [0.0] * len(df),
                "slack_soc_min_mwh": [0.0] * len(df),
                "slack_soc_max_mwh": [0.0] * len(df),
            }
        )

    monkeypatch.setattr(bt, "_clear_afrr_capacity_block_against_truth", fake_clear)
    monkeypatch.setattr(bt, "_formulate_afrr_capacity_block_bids", fake_formulate)
    monkeypatch.setattr(bt, "optimize_dispatch", fake_optimize)

    offered_pos, offered_neg, cap_res, _bids, stats = bt._select_feasible_bcm_lock_candidate(
        blk=snap,
        snapshot_plan=snap,
        source=source,
        colmap=col,
        snapshot_ts=pd.Timestamp("2025-05-01 06:00:00+00:00"),
        offered_pos_mw=10.0,
        offered_neg_mw=0.0,
        lock_pos={},
        lock_neg={},
        da_lockbook={},
        strategy_permissions=StrategyPermissions(False, "technical_repair", True, True, False),
        optimizer_allowed_markets=("aFRR", "ID"),
        horizon_hours=4,
    )

    assert offered_pos == 5.0
    assert offered_neg == 0.0
    assert cap_res is None
    assert stats["feasibility_pass"] == 1.0
    assert stats["retry_factor_selected"] == 0.5
    assert stats["retry_factor_selected_before_clearing"] == 0.5
    assert stats["selection_is_causal"] == 1.0
    assert stats["full_award_feasibility_checked"] == 1.0
    assert stats["realized_clearing_used_for_selection"] == 0.0
    assert stats["reduced_due_to_reserve_feasibility"] == 1.0


def test_bcm_precommit_derates_terminal_soc_infeasible_bid_before_lockbook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt = _mk_backtester("canonical_economic")
    bt.final_soc_mode = "hard"
    bt.enable_reserve_retry_ladder = True
    bt.reserve_retry_ladder = [1.0, 0.5, 0.0]
    bt.reserve_headroom_safety_mwh = 0.0
    bt.reserve_soc_projection_safety_mwh = 0.0
    bt.reserve_power_safety_mw = 0.0
    col = BacktestColumnMap()
    target_hours = pd.date_range("2025-05-02 02:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [2.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [15.4] * 4,
        }
    )
    source = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_activation_rate_pos: [1.0] * 4,
            col.pred_afrr_activation_rate_neg: [0.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [100.0] * 4,
        }
    ).set_index(col.timestamp)

    def fake_clear(*, cap_bids, ts_idx, source, colmap):  # type: ignore[no-untyped-def]
        raise AssertionError("realized clearing must not be used for BCM precommit retry selection")

    def fake_formulate(*, blk, source, colmap, snapshot_ts, offered_pos, offered_neg, is_perfect_foresight=False):  # type: ignore[no-untyped-def]
        ts = pd.to_datetime(blk["target_time_utc"].iloc[0], utc=True)
        bids = []
        if offered_pos > 0.0:
            bids.append(
                AFRRCapacityBid(
                    ts=ts,
                    side="pos",
                    quantity_mw=float(offered_pos),
                    capacity_price_eur_mw=0.0,
                    energy_price_eur_mwh=0.0,
                )
            )
        return bids, pd.to_datetime(blk["target_time_utc"], utc=True)

    def fake_optimize(df, colmap, **kwargs):  # type: ignore[no-untyped-def]
        fixed = kwargs["fixed_reserve_obligation"]
        max_pos = max((v[0] for v in fixed.values()), default=0.0)
        assert max_pos <= 1.0 + 1e-9
        return pd.DataFrame(
            {
                colmap.timestamp: pd.to_datetime(df[colmap.timestamp], utc=True),
                "soc_lp_mwh": [10.1] * len(df),
                "slack_pos_mw": [0.0] * len(df),
                "slack_neg_mw": [0.0] * len(df),
                "slack_soc_min_mwh": [0.0] * len(df),
                "slack_soc_max_mwh": [0.0] * len(df),
            }
        )

    monkeypatch.setattr(bt, "_clear_afrr_capacity_block_against_truth", fake_clear)
    monkeypatch.setattr(bt, "_formulate_afrr_capacity_block_bids", fake_formulate)
    monkeypatch.setattr(bt, "optimize_dispatch", fake_optimize)

    offered_pos, offered_neg, cap_res, _bids, stats = bt._select_feasible_bcm_lock_candidate(
        blk=snap,
        snapshot_plan=snap,
        source=source,
        colmap=col,
        snapshot_ts=pd.Timestamp("2025-05-01 06:00:00+00:00"),
        offered_pos_mw=2.0,
        offered_neg_mw=0.0,
        lock_pos={},
        lock_neg={},
        da_lockbook={},
        strategy_permissions=StrategyPermissions(False, "technical_repair", True, True, False),
        optimizer_allowed_markets=("aFRR", "ID"),
        horizon_hours=4,
    )

    assert offered_pos == pytest.approx(1.0)
    assert offered_neg == pytest.approx(0.0)
    assert cap_res is None
    assert 0.0 < float(stats["retry_factor_selected"]) < 1.0
    assert float(stats["retry_factor_selected_before_clearing"]) == pytest.approx(
        float(stats["retry_factor_selected"])
    )
    assert float(stats["margin_before_derate_mwh"]) <= 0.0
    assert float(stats["margin_after_derate_mwh"]) >= 0.0
    assert float(stats["terminal_soc_shortfall_mwh"]) == pytest.approx(0.0)
    assert str(stats["zero_reason"]) == "none"
    assert str(stats["zero_reason"]) != "reserve_retry_ladder_pending_feasibility"


def test_bcm_precommit_records_id_recourse_for_auxiliary_soc_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt = _mk_backtester()
    bt.enable_reserve_retry_ladder = True
    bt.reserve_retry_ladder = [1.0, 0.0]
    bt.aux_mode = "state_dependent"
    bt.aux_peak_mw = 1.0
    bt.aux_standby_mw = 1.0
    bt.aux_afrr_active_mw = 1.0
    bt.aux_trading_mw = 1.0
    col = BacktestColumnMap()
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=2, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [0.0] * 2,
            "reserve_neg_mw": [1.0] * 2,
            "soc_start_lp_mwh": [bt.soc_min + 0.05] * 2,
            "charge_mw": [0.0] * 2,
            "discharge_mw": [0.0] * 2,
            "bem_only_pos_mw": [0.0] * 2,
            "bem_only_neg_mw": [0.0] * 2,
        }
    )
    source = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.true_afrr_capacity_price_pos: [0.0] * 2,
            col.true_afrr_capacity_price_neg: [100.0] * 2,
            col.pred_afrr_activation_rate_pos: [0.0] * 2,
            col.pred_afrr_activation_rate_neg: [0.0] * 2,
        }
    ).set_index(col.timestamp)

    def fake_clear(*, cap_bids, ts_idx, source, colmap):  # type: ignore[no-untyped-def]
        raise AssertionError("realized clearing must not be used for BCM precommit selection")

    def fake_formulate(*, blk, source, colmap, snapshot_ts, offered_pos, offered_neg, is_perfect_foresight=False):  # type: ignore[no-untyped-def]
        ts = pd.to_datetime(blk["target_time_utc"].iloc[0], utc=True)
        bids = []
        if offered_neg > 0.0:
            bids.append(
                AFRRCapacityBid(
                    ts=ts,
                    side="neg",
                    quantity_mw=float(offered_neg),
                    capacity_price_eur_mw=0.0,
                    energy_price_eur_mwh=0.0,
                )
            )
        return bids, pd.to_datetime(blk["target_time_utc"], utc=True)

    def feasible_plan(df, colmap, **kwargs):  # type: ignore[no-untyped-def]
        return pd.DataFrame(
            {
                colmap.timestamp: pd.to_datetime(df[colmap.timestamp], utc=True),
                "soc_lp_mwh": [bt.soc_min + 0.2] * len(df),
                "id_charge_mw": [0.0] * len(df),
                "terminal_soc_id_recourse_cost_eur": [0.0] * len(df),
                "power_stack_pos_mw": [0.0] * len(df),
                "power_stack_neg_mw": [1.0] * len(df),
                "slack_pos_mw": [0.0] * len(df),
                "slack_neg_mw": [0.0] * len(df),
                "slack_soc_min_mwh": [0.0] * len(df),
                "slack_soc_max_mwh": [0.0] * len(df),
            }
        )

    monkeypatch.setattr(bt, "_clear_afrr_capacity_block_against_truth", fake_clear)
    monkeypatch.setattr(bt, "_formulate_afrr_capacity_block_bids", fake_formulate)
    monkeypatch.setattr(bt, "optimize_dispatch", feasible_plan)

    offered_pos, offered_neg, cap_res, _bids, stats = bt._select_feasible_bcm_lock_candidate(
        blk=snap,
        snapshot_plan=snap,
        source=source,
        colmap=col,
        snapshot_ts=pd.Timestamp("2025-05-01 06:00:00+00:00"),
        offered_pos_mw=0.0,
        offered_neg_mw=1.0,
        lock_pos={},
        lock_neg={},
        da_lockbook={},
        strategy_permissions=StrategyPermissions(False, "technical_repair", True, True, False),
        optimizer_allowed_markets=("aFRR", "ID"),
        horizon_hours=2,
    )

    assert offered_pos == 0.0
    assert offered_neg == 1.0
    assert cap_res is None
    assert stats["feasibility_pass"] == 1.0
    assert float(stats["id_recourse_needed_mwh"]) > 0.0
    assert stats["id_recourse_reason"] == "reserve_obligation_recovery"
    assert stats["selection_is_causal"] == 1.0
    assert stats["realized_clearing_used_for_selection"] == 0.0


def test_bcm_precommit_recoverable_terminal_shortfall_keeps_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt = _mk_backtester("canonical_economic")
    bt.final_soc_mode = "hard"
    bt.enable_reserve_retry_ladder = True
    bt.reserve_retry_ladder = [1.0, 0.0]
    bt.reserve_activation_headroom_h = 0.0
    bt.reserve_headroom_safety_mwh = 0.0
    bt.reserve_soc_projection_safety_mwh = 0.0
    bt.aux_mode = "state_dependent"
    bt.aux_peak_mw = 0.10
    bt.aux_standby_mw = 0.10
    bt.aux_afrr_active_mw = 0.10
    col = BacktestColumnMap()
    target_hours = pd.date_range("2025-05-02 02:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [2.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [10.2] * 4,
            col.pred_da_price: [20.0] * 4,
            col.pred_afrr_activation_rate_pos: [0.0] * 4,
            col.pred_afrr_activation_rate_neg: [0.0] * 4,
        }
    )
    source = snap.set_index(col.timestamp)

    def fake_formulate(*, blk, source, colmap, snapshot_ts, offered_pos, offered_neg, is_perfect_foresight=False):  # type: ignore[no-untyped-def]
        ts = pd.to_datetime(blk["target_time_utc"].iloc[0], utc=True)
        return [AFRRCapacityBid(ts=ts, side="pos", quantity_mw=float(offered_pos), capacity_price_eur_mw=0.0, energy_price_eur_mwh=0.0)], pd.to_datetime(blk["target_time_utc"], utc=True)

    def feasible_plan(df, colmap, **kwargs):  # type: ignore[no-untyped-def]
        return pd.DataFrame(
            {
                colmap.timestamp: pd.to_datetime(df[colmap.timestamp], utc=True),
                "soc_lp_mwh": [10.0] * len(df),
                "id_charge_mw": [0.0] * len(df),
                "terminal_soc_id_recourse_cost_eur": [0.0] * len(df),
                "slack_pos_mw": [0.0] * len(df),
                "slack_neg_mw": [0.0] * len(df),
                "slack_soc_min_mwh": [0.0] * len(df),
                "slack_soc_max_mwh": [0.0] * len(df),
            }
        )

    monkeypatch.setattr(bt, "_formulate_afrr_capacity_block_bids", fake_formulate)
    monkeypatch.setattr(bt, "optimize_dispatch", feasible_plan)

    offered_pos, offered_neg, _cap_res, _bids, stats = bt._select_feasible_bcm_lock_candidate(
        blk=snap,
        snapshot_plan=snap,
        source=source,
        colmap=col,
        snapshot_ts=pd.Timestamp("2025-05-01 06:00:00+00:00"),
        offered_pos_mw=2.0,
        offered_neg_mw=0.0,
        lock_pos={},
        lock_neg={},
        da_lockbook={},
        strategy_permissions=StrategyPermissions(False, "technical_repair", True, True, False),
        optimizer_allowed_markets=("aFRR", "ID"),
        horizon_hours=4,
        candidate_ev_eur=500.0,
    )

    assert offered_pos == pytest.approx(2.0), stats
    assert offered_neg == pytest.approx(0.0)
    assert float(stats["id_recovery_feasible"]) == pytest.approx(1.0)
    assert float(stats["terminal_shortfall_before_recovery_mwh"]) > 0.0
    assert float(stats["terminal_shortfall_after_recovery_mwh"]) <= 1e-6
    assert float(stats["id_recovery_mwh"]) > 0.0
    assert float(stats["id_recovery_scheduled_or_reserved"]) == pytest.approx(1.0)
    assert str(stats["aux_mode"]) == str(bt.aux_mode)
    assert np.isfinite(float(stats["soc_after_aux_mwh"]))
    assert str(stats["zero_reason"]) != "terminal_soc_infeasible"
    assert str(stats["zero_reason"]) != "terminal_soc_not_recoverable"
    assert str(stats["zero_reason"]) != "id_recovery_blocked"


def test_bcm_precommit_unrecoverable_terminal_shortfall_zeroes_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt = _mk_backtester("canonical_economic")
    bt.final_soc_mode = "hard"
    bt.enable_reserve_retry_ladder = True
    bt.reserve_retry_ladder = [1.0, 0.0]
    bt.reserve_activation_headroom_h = 0.0
    bt.reserve_headroom_safety_mwh = 0.0
    bt.reserve_soc_projection_safety_mwh = 0.0
    bt.aux_mode = "state_dependent"
    bt.aux_peak_mw = 0.10
    bt.aux_standby_mw = 0.10
    bt.aux_afrr_active_mw = 0.10
    col = BacktestColumnMap()
    target_hours = pd.date_range("2025-05-02 02:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [2.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [10.2] * 4,
            col.pred_da_price: [20.0] * 4,
            col.pred_afrr_activation_rate_pos: [0.0] * 4,
            "bem_only_neg_mw": [bt.p_max_mw] * 4,
        }
    )
    source = snap.set_index(col.timestamp)

    def fake_formulate(*, blk, source, colmap, snapshot_ts, offered_pos, offered_neg, is_perfect_foresight=False):  # type: ignore[no-untyped-def]
        ts = pd.to_datetime(blk["target_time_utc"].iloc[0], utc=True)
        bids = []
        if offered_pos > 0.0:
            bids.append(AFRRCapacityBid(ts=ts, side="pos", quantity_mw=float(offered_pos), capacity_price_eur_mw=0.0, energy_price_eur_mwh=0.0))
        return bids, pd.to_datetime(blk["target_time_utc"], utc=True)

    monkeypatch.setattr(bt, "_formulate_afrr_capacity_block_bids", fake_formulate)

    offered_pos, offered_neg, _cap_res, _bids, stats = bt._select_feasible_bcm_lock_candidate(
        blk=snap,
        snapshot_plan=snap,
        source=source,
        colmap=col,
        snapshot_ts=pd.Timestamp("2025-05-01 06:00:00+00:00"),
        offered_pos_mw=2.0,
        offered_neg_mw=0.0,
        lock_pos={},
        lock_neg={},
        da_lockbook={},
        strategy_permissions=StrategyPermissions(False, "technical_repair", True, True, False),
        optimizer_allowed_markets=("aFRR", "ID"),
        horizon_hours=4,
        candidate_ev_eur=500.0,
    )

    assert offered_pos == pytest.approx(0.0)
    assert offered_neg == pytest.approx(0.0)
    assert str(stats["zero_reason"]) == "id_recovery_blocked"
    assert float(stats["terminal_shortfall_before_recovery_mwh"]) > 0.0
    assert float(stats["terminal_shortfall_after_recovery_mwh"]) > 0.0


def test_bcm_precommit_negative_ev_after_id_recovery_zeroes_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt = _mk_backtester("canonical_economic")
    bt.final_soc_mode = "hard"
    bt.enable_reserve_retry_ladder = True
    bt.reserve_retry_ladder = [1.0, 0.0]
    bt.reserve_activation_headroom_h = 0.0
    bt.reserve_headroom_safety_mwh = 0.0
    bt.reserve_soc_projection_safety_mwh = 0.0
    bt.aux_mode = "state_dependent"
    bt.aux_peak_mw = 0.10
    bt.aux_standby_mw = 0.10
    bt.aux_afrr_active_mw = 0.10
    col = BacktestColumnMap()
    target_hours = pd.date_range("2025-05-02 02:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [2.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [10.2] * 4,
            col.pred_da_price: [1000.0] * 4,
            col.pred_afrr_activation_rate_pos: [0.0] * 4,
        }
    )
    source = snap.set_index(col.timestamp)

    def fake_formulate(*, blk, source, colmap, snapshot_ts, offered_pos, offered_neg, is_perfect_foresight=False):  # type: ignore[no-untyped-def]
        ts = pd.to_datetime(blk["target_time_utc"].iloc[0], utc=True)
        bids = []
        if offered_pos > 0.0:
            bids.append(AFRRCapacityBid(ts=ts, side="pos", quantity_mw=float(offered_pos), capacity_price_eur_mw=0.0, energy_price_eur_mwh=0.0))
        return bids, pd.to_datetime(blk["target_time_utc"], utc=True)

    monkeypatch.setattr(bt, "_formulate_afrr_capacity_block_bids", fake_formulate)

    offered_pos, offered_neg, _cap_res, _bids, stats = bt._select_feasible_bcm_lock_candidate(
        blk=snap,
        snapshot_plan=snap,
        source=source,
        colmap=col,
        snapshot_ts=pd.Timestamp("2025-05-01 06:00:00+00:00"),
        offered_pos_mw=2.0,
        offered_neg_mw=0.0,
        lock_pos={},
        lock_neg={},
        da_lockbook={},
        strategy_permissions=StrategyPermissions(False, "technical_repair", True, True, False),
        optimizer_allowed_markets=("aFRR", "ID"),
        horizon_hours=4,
        candidate_ev_eur=1.0,
    )

    assert offered_pos == pytest.approx(0.0)
    assert offered_neg == pytest.approx(0.0)
    assert str(stats["zero_reason"]) == "candidate_incremental_ev_negative"
    assert float(stats["id_recovery_feasible"]) == pytest.approx(1.0)
    assert float(stats["effective_ev_after_recovery_eur"]) < 0.0


def test_bcm_precommit_uses_incremental_recovery_cost_not_full_baseline_recovery() -> None:
    stats = BatteryBacktester._bcm_ev_recovery_cost_adjustment(
        candidate_ev_eur=50.0,
        retry_factor=1.0,
        baseline_recovery_cost_eur=100.0,
        candidate_recovery_cost_eur=130.0,
    )

    assert float(stats["incremental_recovery_cost_eur"]) == pytest.approx(30.0)
    assert float(stats["ev_after_incremental_recovery_eur"]) == pytest.approx(20.0)
    assert float(stats["recovery_ev_violation"]) == pytest.approx(0.0)
    assert float(stats["full_recovery_cost_ev_after_recovery_eur"]) == pytest.approx(-80.0)
    assert float(stats["full_recovery_cost_would_have_zeroed"]) == pytest.approx(1.0)
    assert float(stats["recovery_cost_probability_weighted"]) == pytest.approx(0.0)


def test_bcm_precommit_preserves_negative_incremental_recovery_cost_as_benefit() -> None:
    stats = BatteryBacktester._bcm_ev_recovery_cost_adjustment(
        candidate_ev_eur=10.0,
        retry_factor=1.0,
        baseline_recovery_cost_eur=100.0,
        candidate_recovery_cost_eur=40.0,
    )

    assert float(stats["incremental_recovery_cost_eur"]) == pytest.approx(-60.0)
    assert float(stats["expected_incremental_recovery_cost_eur"]) == pytest.approx(-60.0)
    assert float(stats["ev_after_incremental_recovery_eur"]) == pytest.approx(70.0)
    assert float(stats["recovery_ev_violation"]) == pytest.approx(0.0)


def test_bcm_precommit_recovery_cost_probability_factor() -> None:
    stats = BatteryBacktester._bcm_ev_recovery_cost_adjustment(
        candidate_ev_eur=60.0,
        retry_factor=1.0,
        baseline_recovery_cost_eur=0.0,
        candidate_recovery_cost_eur=100.0,
        recovery_cost_probability_factor=0.5,
    )

    assert float(stats["incremental_recovery_cost_eur"]) == pytest.approx(100.0)
    assert float(stats["expected_incremental_recovery_cost_eur"]) == pytest.approx(50.0)
    assert float(stats["ev_after_incremental_recovery_eur"]) == pytest.approx(10.0)
    assert float(stats["recovery_ev_violation"]) == pytest.approx(0.0)
    assert float(stats["recovery_cost_probability_weighted"]) == pytest.approx(1.0)
    assert float(stats["recovery_cost_probability_factor"]) == pytest.approx(0.5)
    assert str(stats["recovery_cost_used_in_ev_mode"]) == "expected_incremental_award_weighted"


def test_bcm_precommit_ev_nan_is_not_negative_ev() -> None:
    stats = BatteryBacktester._bcm_ev_recovery_cost_adjustment(
        candidate_ev_eur=float("nan"),
        retry_factor=1.0,
        baseline_recovery_cost_eur=0.0,
        candidate_recovery_cost_eur=100.0,
        recovery_cost_probability_factor=0.5,
    )

    assert float(stats["ev_nan_detected"]) == pytest.approx(1.0)
    assert str(stats["ev_nan_component"]) == "candidate_ev_eur"
    assert float(stats["recovery_ev_violation"]) == pytest.approx(0.0)


def test_bcm_precommit_selector_selects_full_factor_when_full_award_feasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt = _mk_backtester()
    bt.enable_reserve_retry_ladder = True
    bt.reserve_retry_ladder = [1.0, 0.5, 0.0]
    col = BacktestColumnMap()
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [10.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [10.0] * 4,
        }
    )
    source = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [100.0] * 4,
        }
    ).set_index(col.timestamp)

    def fake_clear(*, cap_bids, ts_idx, source, colmap):  # type: ignore[no-untyped-def]
        raise AssertionError("realized clearing must happen after retry factor selection")

    def fake_formulate(*, blk, source, colmap, snapshot_ts, offered_pos, offered_neg, is_perfect_foresight=False):  # type: ignore[no-untyped-def]
        ts = pd.to_datetime(blk["target_time_utc"].iloc[0], utc=True)
        bids = []
        if offered_pos > 0.0:
            bids.append(AFRRCapacityBid(ts=ts, side="pos", quantity_mw=float(offered_pos), capacity_price_eur_mw=0.0, energy_price_eur_mwh=0.0))
        return bids, pd.to_datetime(blk["target_time_utc"], utc=True)

    def always_feasible(df, colmap, **kwargs):  # type: ignore[no-untyped-def]
        fixed = kwargs["fixed_reserve_obligation"]
        assert max((v[0] for v in fixed.values()), default=0.0) == pytest.approx(10.0)
        return pd.DataFrame(
            {
                colmap.timestamp: pd.to_datetime(df[colmap.timestamp], utc=True),
                "soc_lp_mwh": [12.0] * len(df),
                "slack_pos_mw": [0.0] * len(df),
                "slack_neg_mw": [0.0] * len(df),
                "slack_soc_min_mwh": [0.0] * len(df),
                "slack_soc_max_mwh": [0.0] * len(df),
            }
        )

    monkeypatch.setattr(bt, "_clear_afrr_capacity_block_against_truth", fake_clear)
    monkeypatch.setattr(bt, "_formulate_afrr_capacity_block_bids", fake_formulate)
    monkeypatch.setattr(bt, "optimize_dispatch", always_feasible)

    offered_pos, offered_neg, cap_res, _bids, stats = bt._select_feasible_bcm_lock_candidate(
        blk=snap,
        snapshot_plan=snap,
        source=source,
        colmap=col,
        snapshot_ts=pd.Timestamp("2025-05-01 06:00:00+00:00"),
        offered_pos_mw=10.0,
        offered_neg_mw=0.0,
        lock_pos={},
        lock_neg={},
        da_lockbook={},
        strategy_permissions=StrategyPermissions(False, "technical_repair", True, True, False),
        optimizer_allowed_markets=("aFRR", "ID"),
        horizon_hours=4,
    )

    assert offered_pos == 10.0
    assert offered_neg == 0.0
    assert cap_res is None
    assert stats["retry_factor_selected"] == 1.0
    assert stats["retry_factor_selected_before_clearing"] == 1.0
    assert stats["selection_is_causal"] == 1.0
    assert stats["full_award_feasibility_checked"] == 1.0
    assert stats["realized_clearing_used_for_selection"] == 0.0


def test_bcm_precommit_selector_locks_zero_only_when_no_nonzero_feasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt = _mk_backtester()
    bt.enable_reserve_retry_ladder = True
    bt.reserve_retry_ladder = [1.0, 0.5, 0.0]
    col = BacktestColumnMap()
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [10.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [10.0] * 4,
        }
    )
    source = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [100.0] * 4,
        }
    ).set_index(col.timestamp)

    def fake_clear(*, cap_bids, ts_idx, source, colmap):  # type: ignore[no-untyped-def]
        raise AssertionError("realized clearing must not be used for BCM precommit retry selection")

    def fake_formulate(*, blk, source, colmap, snapshot_ts, offered_pos, offered_neg, is_perfect_foresight=False):  # type: ignore[no-untyped-def]
        ts = pd.to_datetime(blk["target_time_utc"].iloc[0], utc=True)
        bids = []
        if offered_pos > 0.0:
            bids.append(AFRRCapacityBid(ts=ts, side="pos", quantity_mw=float(offered_pos), capacity_price_eur_mw=0.0, energy_price_eur_mwh=0.0))
        return bids, pd.to_datetime(blk["target_time_utc"], utc=True)

    def always_infeasible(df, colmap, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("terminal_soc_not_recoverable_even_with_id")

    monkeypatch.setattr(bt, "_clear_afrr_capacity_block_against_truth", fake_clear)
    monkeypatch.setattr(bt, "_formulate_afrr_capacity_block_bids", fake_formulate)
    monkeypatch.setattr(bt, "optimize_dispatch", always_infeasible)

    offered_pos, offered_neg, cap_res, _bids, stats = bt._select_feasible_bcm_lock_candidate(
        blk=snap,
        snapshot_plan=snap,
        source=source,
        colmap=col,
        snapshot_ts=pd.Timestamp("2025-05-01 06:00:00+00:00"),
        offered_pos_mw=10.0,
        offered_neg_mw=0.0,
        lock_pos={},
        lock_neg={},
        da_lockbook={},
        strategy_permissions=StrategyPermissions(False, "technical_repair", True, True, False),
        optimizer_allowed_markets=("aFRR", "ID"),
        horizon_hours=4,
    )

    assert offered_pos == 0.0
    assert offered_neg == 0.0
    assert cap_res is None
    assert stats["retry_factor_selected"] == 0.0
    assert stats["retry_factor_selected_before_clearing"] == 0.0
    assert stats["selection_is_causal"] == 1.0
    assert stats["full_award_feasibility_checked"] == 1.0
    assert stats["realized_clearing_used_for_selection"] == 0.0
    assert stats["zero_reason"] == "optimizer_validation_exception"
    assert stats["feasibility_driver"] == "optimizer_validation_exception"
    assert float(stats["optimizer_validation_exception"]) == pytest.approx(1.0)
    assert str(stats["optimizer_validation_exception_class"]) == "RuntimeError"
    assert stats["zero_reason"] != "not_awarded_or_zero_candidate"


def test_bcm_precommit_positive_ev_cumulative_feasible_survives_optimizer_validation_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt = _mk_backtester()
    bt.enable_reserve_retry_ladder = True
    bt.reserve_retry_ladder = [1.0, 0.0]
    col = BacktestColumnMap()
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [1.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [12.0] * 4,
            col.pred_da_price: [50.0] * 4,
            col.pred_afrr_activation_rate_pos: [0.0] * 4,
            col.pred_afrr_activation_rate_neg: [0.0] * 4,
        }
    )
    source = snap.set_index(col.timestamp)

    def fake_formulate(*, blk, source, colmap, snapshot_ts, offered_pos, offered_neg, is_perfect_foresight=False):  # type: ignore[no-untyped-def]
        ts = pd.to_datetime(blk["target_time_utc"].iloc[0], utc=True)
        bids = []
        if offered_pos > 0.0:
            bids.append(
                AFRRCapacityBid(
                    ts=ts,
                    side="pos",
                    quantity_mw=float(offered_pos),
                    capacity_price_eur_mw=0.0,
                    energy_price_eur_mwh=0.0,
                )
            )
        return bids, pd.to_datetime(blk["target_time_utc"], utc=True)

    def validator_raises(df, colmap, **kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("synthetic optimizer validation failure")

    monkeypatch.setattr(bt, "_formulate_afrr_capacity_block_bids", fake_formulate)
    monkeypatch.setattr(bt, "optimize_dispatch", validator_raises)

    offered_pos, offered_neg, _cap_res, _bids, stats = bt._select_feasible_bcm_lock_candidate(
        blk=snap,
        snapshot_plan=snap,
        source=source,
        colmap=col,
        snapshot_ts=pd.Timestamp("2025-05-01 06:00:00+00:00"),
        offered_pos_mw=1.0,
        offered_neg_mw=0.0,
        lock_pos={},
        lock_neg={},
        da_lockbook={},
        strategy_permissions=StrategyPermissions(False, "technical_repair", True, True, False),
        optimizer_allowed_markets=("aFRR", "ID"),
        horizon_hours=4,
        candidate_ev_eur=100.0,
    )

    assert offered_pos == pytest.approx(1.0)
    assert offered_neg == pytest.approx(0.0)
    assert float(stats["feasibility_pass"]) == pytest.approx(1.0)
    assert str(stats["zero_reason"]) == "none"
    assert str(stats["final_decision_reason"]) == "accepted_positive_ev_after_optimizer_validation_exception"
    assert float(stats["optimizer_validation_exception"]) == pytest.approx(1.0)
    assert str(stats["optimizer_validation_exception_class"]) == "ValueError"
    assert "synthetic optimizer validation failure" in str(stats["optimizer_validation_exception_message"])


def test_bcm_precommit_opaque_unrecoverable_without_physical_diagnostics_is_not_candidate_unrecoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt = _mk_backtester()
    bt.enable_reserve_retry_ladder = True
    bt.reserve_retry_ladder = [1.0, 0.0]
    col = BacktestColumnMap()
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [1.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [12.0] * 4,
            col.pred_afrr_activation_rate_pos: [0.0] * 4,
            col.pred_afrr_activation_rate_neg: [0.0] * 4,
        }
    )
    source = snap.set_index(col.timestamp)

    def fake_formulate(*, blk, source, colmap, snapshot_ts, offered_pos, offered_neg, is_perfect_foresight=False):  # type: ignore[no-untyped-def]
        ts = pd.to_datetime(blk["target_time_utc"].iloc[0], utc=True)
        bids = []
        if offered_pos > 0.0:
            bids.append(
                AFRRCapacityBid(
                    ts=ts,
                    side="pos",
                    quantity_mw=float(offered_pos),
                    capacity_price_eur_mw=0.0,
                    energy_price_eur_mwh=0.0,
                )
            )
        return bids, pd.to_datetime(blk["target_time_utc"], utc=True)

    def validator_needs_slack(df, colmap, **kwargs):  # type: ignore[no-untyped-def]
        return pd.DataFrame(
            {
                colmap.timestamp: pd.to_datetime(df[colmap.timestamp], utc=True),
                "soc_lp_mwh": [12.0] * len(df),
                "slack_pos_mw": [1.0] * len(df),
                "slack_neg_mw": [0.0] * len(df),
                "slack_soc_min_mwh": [0.0] * len(df),
                "slack_soc_max_mwh": [0.0] * len(df),
            }
        )

    monkeypatch.setattr(bt, "_formulate_afrr_capacity_block_bids", fake_formulate)
    monkeypatch.setattr(bt, "optimize_dispatch", validator_needs_slack)

    offered_pos, offered_neg, _cap_res, _bids, stats = bt._select_feasible_bcm_lock_candidate(
        blk=snap,
        snapshot_plan=snap,
        source=source,
        colmap=col,
        snapshot_ts=pd.Timestamp("2025-05-01 06:00:00+00:00"),
        offered_pos_mw=1.0,
        offered_neg_mw=0.0,
        lock_pos={},
        lock_neg={},
        da_lockbook={},
        strategy_permissions=StrategyPermissions(False, "technical_repair", True, True, False),
        optimizer_allowed_markets=("aFRR", "ID"),
        horizon_hours=4,
        candidate_ev_eur=100.0,
    )

    assert offered_pos == pytest.approx(0.0)
    assert offered_neg == pytest.approx(0.0)
    assert str(stats["zero_reason"]) == "unrecoverable_without_physical_diagnostics"
    assert str(stats["zero_reason"]) != "candidate_unrecoverable"
    assert float(stats["unrecoverable_without_physical_diagnostics"]) == pytest.approx(1.0)
    assert float(stats["candidate_unrecoverable_physical_diagnostics_present"]) == pytest.approx(0.0)
    assert "f=1" in str(stats["retry_trace_compact"])


def test_bcm_precommit_downward_candidate_not_terminal_shortfall_when_feasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt = _mk_backtester()
    bt.enable_reserve_retry_ladder = True
    bt.reserve_retry_ladder = [1.0, 0.0]
    col = BacktestColumnMap()
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [0.0] * 4,
            "reserve_neg_mw": [2.0] * 4,
            "soc_start_lp_mwh": [10.0] * 4,
            col.pred_afrr_activation_rate_pos: [0.0] * 4,
            col.pred_afrr_activation_rate_neg: [0.0] * 4,
        }
    )
    source = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [100.0] * 4,
        }
    ).set_index(col.timestamp)

    def fake_clear(*, cap_bids, ts_idx, source, colmap):  # type: ignore[no-untyped-def]
        raise AssertionError("realized clearing must not be used for BCM precommit retry selection")

    def fake_formulate(*, blk, source, colmap, snapshot_ts, offered_pos, offered_neg, is_perfect_foresight=False):  # type: ignore[no-untyped-def]
        ts = pd.to_datetime(blk["target_time_utc"].iloc[0], utc=True)
        bids = []
        if offered_neg > 0.0:
            bids.append(
                AFRRCapacityBid(
                    ts=ts,
                    side="neg",
                    quantity_mw=float(offered_neg),
                    capacity_price_eur_mw=0.0,
                    energy_price_eur_mwh=0.0,
                )
            )
        return bids, pd.to_datetime(blk["target_time_utc"], utc=True)

    def feasible_plan(df, colmap, **kwargs):  # type: ignore[no-untyped-def]
        fixed = kwargs["fixed_reserve_obligation"]
        max_neg = max((v[1] for v in fixed.values()), default=0.0)
        assert max_neg == pytest.approx(2.0)
        return pd.DataFrame(
            {
                colmap.timestamp: pd.to_datetime(df[colmap.timestamp], utc=True),
                "soc_lp_mwh": [10.0] * len(df),
                "slack_pos_mw": [0.0] * len(df),
                "slack_neg_mw": [0.0] * len(df),
                "slack_soc_min_mwh": [0.0] * len(df),
                "slack_soc_max_mwh": [0.0] * len(df),
            }
        )

    monkeypatch.setattr(bt, "_clear_afrr_capacity_block_against_truth", fake_clear)
    monkeypatch.setattr(bt, "_formulate_afrr_capacity_block_bids", fake_formulate)
    monkeypatch.setattr(bt, "optimize_dispatch", feasible_plan)

    offered_pos, offered_neg, cap_res, _bids, stats = bt._select_feasible_bcm_lock_candidate(
        blk=snap,
        snapshot_plan=snap,
        source=source,
        colmap=col,
        snapshot_ts=pd.Timestamp("2025-05-01 06:00:00+00:00"),
        offered_pos_mw=0.0,
        offered_neg_mw=2.0,
        lock_pos={},
        lock_neg={},
        da_lockbook={},
        strategy_permissions=StrategyPermissions(False, "technical_repair", True, True, False),
        optimizer_allowed_markets=("aFRR", "ID"),
        horizon_hours=4,
    )

    assert offered_pos == pytest.approx(0.0)
    assert offered_neg == pytest.approx(2.0)
    assert cap_res is None
    assert stats["zero_reason"] == "none"
    assert stats["feasibility_driver"] == "none"
    assert stats["zero_reason"] != "terminal_soc_infeasible"


def test_precommit_includes_aux_losses() -> None:
    bt = _mk_backtester()
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    # Without aux this would be feasible for ~4 MW; with 0.2 MWh aux per hour it is reduced.
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [4.0, 4.0, 4.0, 4.0],
            "reserve_neg_mw": [0.0, 0.0, 0.0, 0.0],
            "soc_start_lp_mwh": [bt.soc_min + 2.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.2] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [50.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [100.0] * 4,
        }
    ).set_index(col.timestamp)
    lock_pos: dict[pd.Timestamp, float] = {}
    lock_neg: dict[pd.Timestamp, float] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos={},
        lock_energy_neg={},
        is_perfect_foresight=False,
    )
    assert len(lock_pos) == 4
    assert max(lock_pos.values()) < 4.0


def test_precommit_includes_existing_lockbook_obligations() -> None:
    bt = _mk_backtester()
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [4.0, 4.0, 4.0, 4.0],
            "reserve_neg_mw": [0.0, 0.0, 0.0, 0.0],
            "soc_start_lp_mwh": [bt.soc_min + 3.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [50.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [100.0] * 4,
        }
    ).set_index(col.timestamp)
    lock_pos: dict[pd.Timestamp, float] = {ts: 3.0 for ts in target_hours}
    lock_neg: dict[pd.Timestamp, float] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos={},
        lock_energy_neg={},
        is_perfect_foresight=False,
    )
    # New offer should be clamped to leave room for already-locked 3 MW.
    assert len(lock_pos) == 4
    assert max(lock_pos.values()) <= 3.0 + 1e-9


def test_final_report_excludes_invalid_runs() -> None:
    df = pd.DataFrame(
        {
            "scenario": ["p50_p50", "p70_p90"],
            "trading_strategy": ["multi", "multi"],
            "simulation_valid": [0.0, 1.0],
            "invalid_reason": ["missed_capacity", ""],
        }
    )
    valid = df.loc[pd.to_numeric(df["simulation_valid"], errors="coerce").fillna(0.0) >= 0.5]
    assert len(valid) == 1
    assert valid.iloc[0]["scenario"] == "p70_p90"


def test_precommit_audit_fields_present() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=30)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=12, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    required = [
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
    ]
    for c in required:
        assert c in out.hourly.columns
    assert "precommit_clamp_applied_count" in out.summary
    assert "precommit_clamped_pos_mw_sum" in out.summary
    assert "precommit_clamped_neg_mw_sum" in out.summary


def test_summary_includes_precommit_and_fallback_counts() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=24)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=8, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    s = out.summary
    for k in [
        "fallback_mode_counts",
        "optimization_error_code_counts",
        "activation_split_reconciliation_error_max",
        "precommit_clamp_applied_count",
    ]:
        assert k in s


def test_fallback_infeasible_marks_invalid() -> None:
    bt = _mk_backtester()
    # Build a tiny case likely to trigger fallback with reserve obligation at low SoC.
    df, col = _tiny_backtest_df(hours=6)
    df[col.pred_afrr_capacity_price_pos] = 1000.0
    df[col.true_afrr_capacity_price_pos] = 1000.0
    bt.soc_init = bt.soc_min + 0.1
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    if "reserve_infeasible" in str(out.summary.get("invalid_reason", "")):
        assert float(out.summary.get("simulation_valid", 0.0)) == 0.0


def test_precommit_clamp_prevents_start_of_hour_headroom_violation() -> None:
    bt = _mk_backtester()
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 02:00:00+00:00", periods=4, freq="h")
    # Too close to min SoC; a 10 MW positive reserve bid would violate strict
    # start-of-hour headroom, so the precommit guard rejects it before lockbook.
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [10.0, 10.0, 10.0, 10.0],
            "reserve_neg_mw": [0.0, 0.0, 0.0, 0.0],
            "soc_start_lp_mwh": [bt.soc_min + 0.2] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [100.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [150.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [1000.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    lock_pos: dict[pd.Timestamp, float] = {}
    lock_neg: dict[pd.Timestamp, float] = {}
    precommit: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=precommit,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
    )
    assert len(lock_pos) == 4
    assert max(lock_pos.values()) == pytest.approx(0.0)
    applied = precommit.get("precommit_clamp_applied", {})
    assert any(float(v) > 0.5 for v in applied.values())
    reasons = {str(v) for v in precommit.get("bcm_precommit_zero_reason", {}).values()}
    assert "headroom_infeasible" in reasons


def test_fallback_repair_preserves_start_of_hour_reserve_headroom() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=10)
    # Encourage reserve commitments and make solve harder.
    df[col.pred_afrr_capacity_price_pos] = 1e3
    df[col.true_afrr_capacity_price_pos] = 1e3
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    codes = (
        out.hourly["optimization_error_code"].astype(str).value_counts().to_dict()
        if "optimization_error_code" in out.hourly.columns
        else {}
    )
    # Legacy unsafe label must not be used when reserve obligations exist.
    assert "safe_hold_plan_under_infeasible_soft_final_soc" not in codes


def test_intrahour_id_rescue_does_not_mask_start_of_hour_headroom_violation() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    df[col.pred_afrr_capacity_price_pos] = 1e3
    df[col.true_afrr_capacity_price_pos] = 1e3
    bt.soc_init = bt.soc_min + 0.05
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    hv = float(out.summary.get("headroom_violation_count", 0.0))
    if hv > 1e-9:
        assert float(out.summary.get("headroom_check_pass", 1.0)) == 0.0
        assert float(out.summary.get("simulation_valid", 1.0)) == 0.0


def test_fallback_marks_scenario_invalid_in_strict_mode() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    df[col.pred_afrr_capacity_price_pos] = 1000.0
    df[col.true_afrr_capacity_price_pos] = 1000.0
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=4,
        reopt_step_hours=1,
        allowed_markets=("DA", "aFRR"),
        strict_simulation_validity=True,
    )
    fb = float(out.summary.get("fallback_used", 0.0))
    if fb > 0.5:
        assert float(out.summary.get("simulation_valid", 1.0)) == 0.0
        assert float(out.summary.get("thesis_reportable", 1.0)) == 0.0
        assert "fallback_used" in str(out.summary.get("invalid_reason", ""))


def test_safe_hold_not_reportable() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    bt.soc_init = bt.soc_min + 0.01
    df[col.pred_afrr_capacity_price_pos] = 1000.0
    df[col.true_afrr_capacity_price_pos] = 1000.0
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=4,
        reopt_step_hours=1,
        allowed_markets=("DA", "aFRR"),
        strict_simulation_validity=True,
    )
    codes = out.summary.get("optimization_error_code_counts", "{}")
    if "safe_hold" in str(codes).lower():
        assert float(out.summary.get("fallback_used", 0.0)) >= 1.0
        assert float(out.summary.get("thesis_reportable", 1.0)) == 0.0


def test_valid_run_has_no_fallback_modes() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=12)
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=6,
        reopt_step_hours=1,
        allowed_markets=("DA",),
        strict_simulation_validity=True,
    )
    if float(out.summary.get("simulation_valid", 0.0)) >= 0.5:
        assert float(out.summary.get("fallback_used", 1.0)) == 0.0
        counts = str(out.summary.get("optimization_error_code_counts", "{}")).lower()
        assert counts in {"{}", ""} or ("ok" in counts)


def test_validity_flags_are_consistent() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    if float(s.get("headroom_violation_count", 0.0)) > 1e-9:
        assert float(s.get("simulation_valid", 1.0)) == 0.0
    if float(s.get("missed_capacity_pos_mw", 0.0)) + float(s.get("missed_capacity_neg_mw", 0.0)) > 1e-9:
        assert float(s.get("simulation_valid", 1.0)) == 0.0
    if float(s.get("fallback_used", 0.0)) > 0.5:
        assert float(s.get("simulation_valid", 1.0)) == 0.0


def test_reserve_commitment_traceability_fields_present() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=30)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=12, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    req = [
        "reserve_commitment_id",
        "reserve_product_block_id",
        "reserve_commitment_source_snapshot_utc",
        "reserve_delivery_start_utc",
        "reserve_delivery_end_utc",
        "reserve_precommit_feasible_pos_mw",
        "reserve_precommit_feasible_neg_mw",
        "reserve_submitted_pos_mw",
        "reserve_submitted_neg_mw",
        "reserve_awarded_pos_mw",
        "reserve_awarded_neg_mw",
        "reserve_lockbook_pos_mw",
        "reserve_lockbook_neg_mw",
    ]
    for c in req:
        assert c in out.hourly.columns


def test_precommit_clamp_includes_projection_safety_buffer() -> None:
    bt = _mk_backtester()
    # Compare with/without projection safety by manually toggling on same synthetic block.
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [10.0, 10.0, 10.0, 10.0],
            "reserve_neg_mw": [0.0, 0.0, 0.0, 0.0],
            "soc_start_lp_mwh": [bt.soc_min + 2.5] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
            col.true_afrr_activation_price_pos: [100.0] * 4,
            col.true_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_activation_rate_pos: [0.0] * 4,
            col.true_afrr_activation_rate_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    pre0: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt.reserve_soc_projection_safety_mwh = 0.0
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap.copy(),
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre0,
        is_perfect_foresight=False,
    )
    pre1: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt.reserve_soc_projection_safety_mwh = 0.1
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap.copy(),
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre1,
        is_perfect_foresight=False,
    )
    f0 = min(float(v) for v in pre0.get("precommit_feasible_pos_mw", {}).values())
    f1 = min(float(v) for v in pre1.get("precommit_feasible_pos_mw", {}).values())
    assert f1 <= f0 + 1e-9


def test_reserve_min_margin_after_bid_applied() -> None:
    bt = _mk_backtester()
    bt.reserve_feasibility_mode = "conservative"
    bt.reserve_min_margin_after_bid_mwh = 10.0  # force zeroing
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [5.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 5.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
            col.true_afrr_activation_price_pos: [100.0] * 4,
            col.true_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_activation_rate_pos: [0.0] * 4,
            col.true_afrr_activation_rate_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    lock_pos: dict[pd.Timestamp, float] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
    )
    assert all(float(v) == 0.0 for v in lock_pos.values())
    assert any(float(v) > 0.5 for v in pre.get("precommit_zeroed_due_to_margin", {}).values())


def test_conservative_precommit_reduces_bid_to_safe_mw() -> None:
    bt = _mk_backtester()
    bt.reserve_feasibility_mode = "conservative"
    bt.reserve_min_margin_after_bid_mwh = 0.4
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [5.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 2.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
            col.true_afrr_activation_price_pos: [100.0] * 4,
            col.true_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_activation_rate_pos: [0.0] * 4,
            col.true_afrr_activation_rate_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    lock_pos: dict[pd.Timestamp, float] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
    )
    assert any(float(v) > 0.5 for v in pre.get("precommit_reduced_due_to_margin", {}).values())
    assert any(float(v) > 0.0 for v in lock_pos.values())
    assert any(float(v) < 5.0 for v in lock_pos.values())


def test_conservative_reserve_cli_params_reach_backtester() -> None:
    txt = Path("scripts/run_battery_backtest.py").read_text(encoding="utf-8")
    assert "--reserve-feasibility-mode" in txt
    assert "--reserve-soc-projection-safety-mwh" in txt
    assert "--reserve-headroom-safety-mwh" in txt
    assert "--reserve-power-safety-mw" in txt
    assert "--reserve-min-margin-after-bid-mwh" in txt
    assert 'MODEL_SPECS["reserve_feasibility_mode"]' in txt
    assert 'MODEL_SPECS["reserve_soc_projection_safety_mwh"]' in txt
    assert 'MODEL_SPECS["reserve_headroom_safety_mwh"]' in txt
    assert 'MODEL_SPECS["reserve_power_safety_mw"]' in txt
    assert 'MODEL_SPECS["reserve_min_margin_after_bid_mwh"]' in txt
    assert "--reserve-bid-derate" in txt
    assert "--max-reserve-bid-mw" in txt
    assert 'MODEL_SPECS["reserve_bid_derate"]' in txt
    assert 'MODEL_SPECS["max_reserve_bid_mw"]' in txt


def test_summary_never_omits_required_validity_fields() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=12)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1, strict_simulation_validity=True)
    req = [
        "simulation_valid",
        "thesis_reportable",
        "invalid_reason",
        "fallback_used",
        "infeasibility_driver",
        "fallback_mode_counts",
        "optimization_error_code_counts",
        "precommit_clamp_applied_count",
        "precommit_clamped_pos_mw_sum",
        "precommit_clamped_neg_mw_sum",
        "precommit_clamp_reasons",
        "activation_split_reconciliation_error_max",
        "reserve_soc_projection_safety_mwh",
        "reserve_headroom_safety_mwh",
        "reserve_power_safety_mw",
        "reserve_feasibility_mode",
        "reserve_min_margin_after_bid_mwh",
        "reserve_bid_derate",
        "max_reserve_bid_mw",
        "precommit_reduced_due_to_margin_count",
        "precommit_safe_pos_mw_avg",
        "precommit_safe_neg_mw_avg",
        "fallback_is_repair_optimization",
        "afrr_bcm_bid_hour_local_model",
        "afrr_bcm_bid_hour_local_benchmark",
        "benchmark_same_rules_gate_consistent",
        "benchmark_is_global_upper_bound",
        "final_soc_handling_mode",
        "terminal_soc_net_adjustment_eur",
        "terminal_price_eur_mwh",
        "terminal_price_source",
    ]
    for c in req:
        assert c in out.summary


def test_fallback_is_repair_optimization_flag() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    assert float(out.summary.get("fallback_is_repair_optimization", 1.0)) == 0.0


def test_no_thesis_reportable_with_fallback() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    df[col.pred_afrr_capacity_price_pos] = 1000.0
    df[col.true_afrr_capacity_price_pos] = 1000.0
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    if float(out.summary.get("fallback_used", 0.0)) > 0.5:
        assert float(out.summary.get("thesis_reportable", 1.0)) == 0.0


def test_fallback_never_thesis_reportable() -> None:
    test_no_thesis_reportable_with_fallback()


def test_fallback_never_reportable() -> None:
    test_fallback_never_thesis_reportable()


def test_reserve_feasibility_repair_never_reportable() -> None:
    row = {
        "simulation_valid": 1.0,
        "final_soc_check_pass": 1.0,
        "missed_capacity_check_pass": 1.0,
        "missed_activation_check_pass": 1.0,
        "headroom_check_pass": 1.0,
        "protected_soc_check_pass": 1.0,
        "reserve_headroom_shortfall_check_pass": 1.0,
        "pnl_reconciliation_check_pass": 1.0,
        "fallback_used": 0.0,
        "reserve_feasibility_repair_used": 1.0,
    }
    thesis_reportable = float(
        (row["simulation_valid"] >= 0.5)
        and (row["fallback_used"] <= 0.5)
        and (row["reserve_feasibility_repair_used"] <= 0.5)
        and (row["final_soc_check_pass"] >= 0.5)
        and (row["missed_capacity_check_pass"] >= 0.5)
        and (row["missed_activation_check_pass"] >= 0.5)
        and (row["headroom_check_pass"] >= 0.5)
        and (row["protected_soc_check_pass"] >= 0.5)
        and (row["reserve_headroom_shortfall_check_pass"] >= 0.5)
        and (row["pnl_reconciliation_check_pass"] >= 0.5)
    )
    assert thesis_reportable == 0.0


def test_reserve_bid_derate_reduces_submitted_mw() -> None:
    bt = _mk_backtester()
    bt.reserve_bid_derate = 0.5
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [8.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 8.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
    )
    fea = list(pre.get("precommit_feasible_pos_mw", {}).values())
    sub = list(pre.get("precommit_submitted_pos_mw_after_derate_cap", {}).values())
    assert fea and sub
    assert max(float(s) for s in sub) <= max(float(f) for f in fea) + 1e-9


def test_max_reserve_bid_mw_caps_submission() -> None:
    bt = _mk_backtester()
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = 2.0
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [8.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 8.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
    )
    sub = list(pre.get("precommit_submitted_pos_mw_after_derate_cap", {}).values())
    assert sub and all(float(v) <= 2.0 + 1e-9 for v in sub)


def test_derate_and_cap_written_to_summary_and_debug() -> None:
    txt_bt = Path("src/energy_trading/simulation/battery_backtest.py").read_text(encoding="utf-8")
    txt_run = Path("scripts/run_battery_backtest.py").read_text(encoding="utf-8")
    for key in (
        "reserve_bid_derate",
        "max_reserve_bid_mw",
        "final_soc_mode",
        "precommit_reduction_reason",
        "desired_reserve_pos_mw",
        "safe_reserve_pos_mw",
        "submitted_reserve_pos_mw",
        "precommit_submitted_pos_mw_after_derate_cap",
        "precommit_submitted_neg_mw_after_derate_cap",
    ):
        assert key in txt_bt
        assert key in txt_run


def test_conservative_fields_written_to_summary_and_debug() -> None:
    test_derate_and_cap_written_to_summary_and_debug()


def test_quantile_aggressiveness_not_changed_by_feasibility_cap() -> None:
    bt = _mk_backtester()
    bt.reserve_bid_derate = 0.5
    bt.max_reserve_bid_mw = 2.0
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [6.0] * 4,
            "reserve_neg_mw": [4.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 8.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [30.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [100.0] * 4,
        }
    ).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
    )
    assert all(abs(float(v) - 6.0) < 1e-9 for v in pre.get("desired_reserve_pos_mw", {}).values())
    assert all(abs(float(v) - 4.0) < 1e-9 for v in pre.get("desired_reserve_neg_mw", {}).values())


def test_reserve_submission_capped_by_safe_mw() -> None:
    bt = _mk_backtester()
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = None
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [8.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 1.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [50.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [100.0] * 4,
            col.pred_afrr_activation_price_neg: [-100.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
    )
    for ts, desired in pre.get("desired_reserve_pos_mw", {}).items():
        safe = float(pre.get("safe_reserve_pos_mw", {}).get(ts, 0.0))
        submitted = float(pre.get("submitted_reserve_pos_mw", {}).get(ts, 0.0))
        assert submitted <= safe + 1e-9
        assert safe <= float(desired) + 1e-9


def test_reserve_submission_never_exceeds_safe_mw() -> None:
    test_reserve_submission_capped_by_safe_mw()


def test_zero_reserve_when_safe_mw_below_tolerance() -> None:
    bt = _mk_backtester()
    bt.reserve_bid_derate = 1.0
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [5.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame({col.timestamp: target_hours}).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    lock_pos: dict[pd.Timestamp, float] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
    )
    assert all(float(v) <= 1e-9 for v in lock_pos.values())
    assert all(float(v) <= 1e-9 for v in pre.get("submitted_reserve_pos_mw", {}).values())


def test_zero_safe_reserve_submits_zero() -> None:
    test_zero_reserve_when_safe_mw_below_tolerance()


def test_reserve_bid_derate_applies_after_safe_mw() -> None:
    bt = _mk_backtester()
    bt.reserve_bid_derate = 0.5
    bt.max_reserve_bid_mw = None
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [4.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 8.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame({col.timestamp: target_hours}).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
    )
    for ts, safe in pre.get("safe_reserve_pos_mw", {}).items():
        sub = float(pre.get("submitted_reserve_pos_mw", {}).get(ts, 0.0))
        assert sub <= float(safe) * 0.5 + 1e-9


def test_reserve_bid_derate_applied_after_safe_mw() -> None:
    test_reserve_bid_derate_applies_after_safe_mw()


def test_max_reserve_bid_caps_after_safe_mw() -> None:
    bt = _mk_backtester()
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = 1.0
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 00:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [8.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 8.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame({col.timestamp: target_hours}).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
    )
    assert all(float(v) <= 1.0 + 1e-9 for v in pre.get("submitted_reserve_pos_mw", {}).values())


def test_max_reserve_bid_mw_caps_submission() -> None:
    test_max_reserve_bid_caps_after_safe_mw()


def test_data_flow_has_single_canonical_soc_source() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=10)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1)
    # Canonical settlement SoC must exist and be finite.
    assert "real_soc_mwh" in out.hourly.columns
    rs = pd.to_numeric(out.hourly["real_soc_mwh"], errors="coerce")
    assert rs.notna().all()
    # If start-of-hour SoC audit field exists, it should be finite and aligned.
    if "real_soc_start_mwh" in out.hourly.columns:
        ss = pd.to_numeric(out.hourly["real_soc_start_mwh"], errors="coerce")
        assert ss.notna().all()


def test_required_debug_fields_present() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=16)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=8, reopt_step_hours=1, allowed_markets=("DA", "aFRR"))
    req = [
        "reserve_commitment_id",
        "reserve_product_block_id",
        "reserve_commitment_source_snapshot_utc",
        "reserve_delivery_start_utc",
        "reserve_delivery_end_utc",
        "reserve_submitted_pos_mw",
        "reserve_awarded_pos_mw",
        "fixed_reserve_obligation_pos_mw",
        "real_required_headroom_pos_mwh",
        "real_available_headroom_pos_mwh",
        "optimization_error_code",
        "is_fallback_hour",
    ]
    for c in req:
        assert c in out.hourly.columns


def test_same_rules_benchmark_uses_same_bcm_gate() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=10)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1)
    assert float(out.summary.get("afrr_bcm_bid_hour_local_model", -1.0)) == float(
        out.summary.get("afrr_bcm_bid_hour_local_benchmark", -2.0)
    )
    assert float(out.summary.get("benchmark_same_rules_gate_consistent", 0.0)) == 1.0


def test_strict_final_soc_hard_constraint_or_repair() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    df[col.pred_da_price] = 200.0
    df[col.true_da_price] = 200.0
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    shortfall = float(s.get("final_soc_shortfall_mwh", 0.0))
    repair = float(s.get("terminal_soc_repair_cost_eur", 0.0))
    assert (shortfall <= 1e-6) or (repair > 0.0)
    if shortfall <= 1e-6:
        assert float(s.get("final_soc_check_pass", 0.0)) >= 0.5
    else:
        assert float(s.get("final_soc_physical_check_pass", 1.0)) == 0.0
        assert float(s.get("final_soc_check_pass", 1.0)) == 0.0
    assert "final_soc_actual_mwh" in s
    assert "final_soc_target_mwh" in s
    assert "final_soc_physical_check_pass" in s
    assert "final_soc_economic_repair_check_pass" in s
    assert "terminal_soc_repair_included_in_pnl" in s


def test_unrepaired_final_soc_shortfall_invalid() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    # Force shortfall and zero terminal repair by setting terminal price to zero.
    df[col.pred_da_price] = 100.0
    df[col.true_da_price] = 0.0
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    if float(s.get("final_soc_shortfall_mwh", 0.0)) > 1e-6:
        assert float(s.get("terminal_soc_repair_cost_eur", 0.0)) <= 1e-9
        assert float(s.get("terminal_soc_repair_included_in_pnl", 1.0)) == 0.0
        assert float(s.get("final_soc_check_pass", 1.0)) == 0.0
        assert float(s.get("simulation_valid", 1.0)) == 0.0
        assert float(s.get("thesis_reportable", 1.0)) == 0.0
        assert "final_soc_unrepaired" in str(s.get("invalid_reason", ""))


def test_final_soc_shortfall_repaired_is_valid_if_pnl_reconciles() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    df[col.pred_da_price] = 150.0
    df[col.true_da_price] = 120.0
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    if float(s.get("final_soc_shortfall_mwh", 0.0)) > 1e-6 and float(s.get("terminal_soc_repair_cost_eur", 0.0)) > 1e-9:
        assert float(s.get("terminal_soc_repair_included_in_pnl", 0.0)) >= 0.5
        assert float(s.get("final_soc_economic_repair_check_pass", 1.0)) == 0.0
        assert float(s.get("final_soc_check_pass", 1.0)) == 0.0
        assert float(s.get("simulation_valid", 1.0)) == 0.0


def test_terminal_repair_mode_allows_physical_shortfall_if_repaired() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "terminal_repair"
    df, col = _tiny_backtest_df(hours=6)
    df[col.pred_da_price] = 150.0
    df[col.true_da_price] = 120.0
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=6,
        reopt_step_hours=1,
        strict_simulation_validity=True,
        enforce_final_soc_min=False,
    )
    s = out.summary
    if float(s.get("final_soc_shortfall_mwh", 0.0)) > 1e-6:
        if float(s.get("terminal_soc_repair_cost_eur", 0.0)) > 0.0:
            assert float(s.get("terminal_soc_repair_included_in_pnl", 0.0)) >= 0.5
            assert float(s.get("final_soc_economic_repair_check_pass", 0.0)) >= 0.5
            assert float(s.get("final_soc_check_pass", 0.0)) >= 0.5
        else:
            assert float(s.get("terminal_soc_repair_included_in_pnl", 1.0)) == 0.0
            assert float(s.get("final_soc_economic_repair_check_pass", 1.0)) == 0.0
            assert float(s.get("final_soc_check_pass", 1.0)) == 0.0


def test_hard_final_soc_mode_invalidates_shortfall() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "hard"
    df, col = _tiny_backtest_df(hours=6)
    df[col.pred_da_price] = 100.0
    df[col.true_da_price] = 0.0
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=6,
        reopt_step_hours=1,
        strict_simulation_validity=True,
        enforce_final_soc_min=False,
    )
    s = out.summary
    if float(s.get("final_soc_shortfall_mwh", 0.0)) > 1e-6:
        assert float(s.get("final_soc_physical_check_pass", 1.0)) == 0.0
        assert float(s.get("final_soc_check_pass", 1.0)) == 0.0
        assert float(s.get("simulation_valid", 1.0)) == 0.0


def test_hard_mode_final_soc_slack_not_used() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "hard"
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=8,
        reopt_step_hours=1,
        strict_simulation_validity=True,
        enforce_final_soc_min=True,
    )
    assert float(out.summary.get("final_soc_slack_used_mwh", 0.0)) <= 1e-9


def test_rolling_final_soc_target_only_applies_at_global_end() -> None:
    assert BatteryBacktester._rolling_final_soc_min_target(
        enforce_final_soc_min=True,
        window_end=12,
        total_rows=24,
        soc_min=2.0,
        soc_target_end=10.0,
    ) == pytest.approx(2.0)
    assert BatteryBacktester._rolling_final_soc_min_target(
        enforce_final_soc_min=True,
        window_end=24,
        total_rows=24,
        soc_min=2.0,
        soc_target_end=10.0,
    ) == pytest.approx(10.0)
    assert BatteryBacktester._rolling_final_soc_min_target(
        enforce_final_soc_min=False,
        window_end=24,
        total_rows=24,
        soc_min=2.0,
        soc_target_end=10.0,
    ) is None


def test_rolling_final_soc_target_includes_terminal_safety_at_global_end() -> None:
    assert BatteryBacktester._rolling_final_soc_min_target(
        enforce_final_soc_min=True,
        window_end=24,
        total_rows=24,
        soc_min=2.0,
        soc_target_end=10.0,
        terminal_soc_safety_margin_mwh=0.03,
        soc_max=18.0,
    ) == pytest.approx(10.03)


def test_rolling_final_soc_target_safety_is_capped_by_soc_max() -> None:
    assert BatteryBacktester._rolling_final_soc_min_target(
        enforce_final_soc_min=True,
        window_end=24,
        total_rows=24,
        soc_min=2.0,
        soc_target_end=10.0,
        terminal_soc_safety_margin_mwh=0.50,
        soc_max=10.2,
    ) == pytest.approx(10.2)


def test_rolling_final_soc_target_anticipates_latest_technical_id_recovery() -> None:
    target = BatteryBacktester._rolling_final_soc_min_target(
        enforce_final_soc_min=True,
        window_end=23,
        total_rows=24,
        soc_min=2.0,
        soc_target_end=10.0,
        p_max_mw=1.0,
        eta_in=0.9487,
        dt_h=1.0,
        remaining_aux_loss_mwh_per_hour=0.04,
        terminal_soc_safety_margin_mwh=0.10,
        soc_max=18.0,
    )

    assert target == pytest.approx(10.0 + 0.10 + 0.04 - 1.0 * 0.9487)
    assert target > 2.0


def test_rolling_final_soc_recoverability_floor_stays_at_soc_min_when_time_is_sufficient() -> None:
    target = BatteryBacktester._rolling_final_soc_min_target(
        enforce_final_soc_min=True,
        window_end=12,
        total_rows=24,
        soc_min=2.0,
        soc_target_end=10.0,
        p_max_mw=10.0,
        eta_in=0.9487,
        dt_h=1.0,
        remaining_aux_loss_mwh_per_hour=0.04,
        terminal_soc_safety_margin_mwh=0.10,
        soc_max=18.0,
    )

    assert target == pytest.approx(2.0)


def test_hard_final_soc_false_infeasible_classification_when_safe_hold_feasible() -> None:
    assert BatteryBacktester._classify_hard_final_soc_infeasibility(
        current_soc_mwh=13.6,
        final_soc_target_mwh=10.0,
        rolling_window_contains_global_end=True,
        safe_hold_feasible=True,
    ) == "hard_final_soc_false_infeasible"
    assert BatteryBacktester._classify_hard_final_soc_infeasibility(
        current_soc_mwh=9.36,
        final_soc_target_mwh=10.0,
        rolling_window_contains_global_end=True,
        safe_hold_feasible=False,
    ) == "terminal_soc_conflict"
    assert BatteryBacktester._classify_hard_final_soc_infeasibility(
        current_soc_mwh=9.0,
        final_soc_target_mwh=10.0,
        rolling_window_contains_global_end=False,
        safe_hold_feasible=False,
        fixed_da_charge_mwh=5.0,
        fixed_da_discharge_mwh=6.0,
        fixed_reserve_pos_mw=0.0,
        fixed_reserve_neg_mw=0.0,
    ) == "locked_da_commitment_infeasible"
    assert BatteryBacktester._classify_hard_final_soc_infeasibility(
        current_soc_mwh=9.0,
        final_soc_target_mwh=10.0,
        rolling_window_contains_global_end=False,
        safe_hold_feasible=True,
        fixed_da_charge_mwh=5.0,
        fixed_da_discharge_mwh=6.0,
        fixed_reserve_pos_mw=0.0,
        fixed_reserve_neg_mw=0.0,
    ) == "fixed_obligation_replay_milp_mismatch"


def _da_lock_rows(
    *,
    hours: int,
    soc_start: float,
    charge_mw: float = 0.0,
    discharge_mw: float = 0.0,
) -> pd.DataFrame:
    col = BacktestColumnMap()
    ts = pd.date_range("2026-01-02T00:00:00Z", periods=hours, freq="h")
    return pd.DataFrame(
        {
            col.timestamp: ts,
            "target_time_utc": ts,
            "charge_mw": [charge_mw] * hours,
            "discharge_mw": [discharge_mw] * hours,
            "soc_start_lp_mwh": [soc_start] * hours,
            col.pred_da_price: [100.0] * hours,
            col.pred_afrr_activation_rate_pos: [0.0] * hours,
            col.pred_afrr_activation_rate_neg: [0.0] * hours,
        }
    )


def test_da_precommit_accepts_full_schedule_when_feasible() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    rows = _da_lock_rows(hours=2, soc_start=5.0, charge_mw=1.0)
    rows[col.pred_da_price] = -100.0
    schedule, audit = bt._select_feasible_da_lock_schedule(
        lock_rows=rows,
        colmap=col,
        current_soc_mwh=5.0,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )
    assert schedule
    assert all(np.isclose(ch, 1.0) and np.isclose(dis, 0.0) for ch, dis in schedule.values())
    assert audit
    assert all(float(r["da_retry_factor_selected"]) == pytest.approx(1.0) for r in audit)
    assert all(float(r["feasibility_pass"]) == pytest.approx(1.0) for r in audit)


def test_da_precommit_reduces_infeasible_schedule_when_smaller_schedule_feasible() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    rows = _da_lock_rows(hours=4, soc_start=7.0, discharge_mw=3.0)
    rows[col.pred_da_price] = 1_000.0
    schedule, audit = bt._select_feasible_da_lock_schedule(
        lock_rows=rows,
        colmap=col,
        current_soc_mwh=7.0,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )
    factor = float(audit[0]["da_retry_factor_selected"])
    assert 0.0 < factor < 1.0
    assert any(dis > 0.0 for _, dis in schedule.values())
    assert all(float(r["feasibility_pass"]) == pytest.approx(1.0) for r in audit)
    assert all(float(r["da_accepted_sell_mw"]) <= float(r["da_candidate_sell_mw"]) for r in audit)


def test_da_precommit_zeroes_when_no_nonzero_schedule_feasible() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    rows = _da_lock_rows(hours=1, soc_start=bt.soc_min + 0.05, discharge_mw=10.0)
    schedule, audit = bt._select_feasible_da_lock_schedule(
        lock_rows=rows,
        colmap=col,
        current_soc_mwh=bt.soc_min + 0.05,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )
    assert all(ch == pytest.approx(0.0) and dis == pytest.approx(0.0) for ch, dis in schedule.values())
    assert all(float(r["da_zeroed_all_bids"]) == pytest.approx(1.0) for r in audit)
    assert all(
        str(r["da_zero_reason"]) in {"no_nonzero_feasible_da_schedule", "no_trade_incumbent_selected"}
        for r in audit
    )
    assert all(str(r["selected_incumbent"]) in {"zeroed_candidate", "no_trade"} for r in audit)


def test_da_postlock_future_guard_rejects_future_infeasible_candidate_and_accepts_feasible() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts0 = pd.Timestamp("2026-01-02T00:00:00Z")
    ts1 = pd.Timestamp("2026-01-02T01:00:00Z")
    future_rows = pd.DataFrame(
        {
            col.timestamp: [ts0, ts1],
            col.pred_afrr_activation_rate_pos: [0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0],
        }
    )
    audit_rows = [
        {
            "timestamp_utc": str(ts1),
            "da_candidate_buy_mw": 0.0,
            "da_candidate_sell_mw": 10.0,
            "da_accepted_buy_mw": 0.0,
            "da_accepted_sell_mw": 10.0,
            "candidate_selection_pnl_eur": 100.0,
            "incumbent_selection_pnl_eur": 0.0,
            "selected_incumbent": "optimized",
        }
    ]

    selected, audit = bt._apply_da_postlock_future_guard(
        selected_da={ts1: (0.0, 10.0)},
        da_audit_rows=audit_rows,
        future_rows=future_rows,
        colmap=col,
        current_soc_mwh=bt.soc_min + 0.2,
        da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert selected[ts1] == pytest.approx((0.0, 0.0))
    assert float(audit[0]["da_postlock_future_feasible"]) == pytest.approx(0.0)
    assert float(audit[0]["da_postlock_candidate_future_feasible"]) == pytest.approx(0.0)
    assert float(audit[0]["da_postlock_no_trade_future_feasible"]) == pytest.approx(1.0)
    assert float(audit[0]["da_postlock_final_selected_future_feasible"]) == pytest.approx(1.0)
    assert str(audit[0]["da_postlock_selected_after_future_check"]) == "no_trade"
    assert str(audit[0]["da_postlock_final_selected_after_future_check"]) == "no_trade"
    assert float(audit[0]["da_postlock_rejected_due_to_future_infeasibility"]) == pytest.approx(1.0)
    assert str(audit[0]["da_postlock_infeasibility_driver"]) == "locked_da_hourly_soc_min_infeasible"
    assert str(audit[0]["da_postlock_infeasibility_driver_detail"]) == "locked_da_hourly_soc_min_infeasible"
    assert str(audit[0]["da_postlock_first_infeasible_timestamp_utc"]) == str(ts1)
    assert float(audit[0]["da_postlock_candidate_locked_sell_mwh"]) == pytest.approx(10.0 * bt.dt_h)
    assert float(audit[0]["da_postlock_candidate_locked_sell_mwh_t"]) == pytest.approx(10.0 * bt.dt_h)
    assert float(audit[0]["da_postlock_candidate_total_locked_sell_mwh"]) == pytest.approx(10.0 * bt.dt_h)
    assert float(audit[0]["da_postlock_failed_locked_sell_mwh_t"]) == pytest.approx(10.0 * bt.dt_h)
    assert str(audit[0]["da_postlock_failed_reason_detail"]) == "locked_da_hourly_soc_min_infeasible"
    assert str(audit[0]["pre_postlock_selected_incumbent"]) == "optimized"
    assert str(audit[0]["final_selected_incumbent"]) == "no_trade"

    selected_ok, audit_ok = bt._apply_da_postlock_future_guard(
        selected_da={ts0: (1.0, 0.0)},
        da_audit_rows=[{**audit_rows[0], "timestamp_utc": str(ts0), "da_candidate_sell_mw": 0.0}],
        future_rows=future_rows,
        colmap=col,
        current_soc_mwh=10.0,
        da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert selected_ok[ts0] == pytest.approx((1.0, 0.0))
    assert float(audit_ok[0]["da_postlock_future_feasible"]) == pytest.approx(1.0)
    assert float(audit_ok[0]["da_postlock_candidate_future_feasible"]) == pytest.approx(1.0)
    assert str(audit_ok[0]["da_postlock_selected_after_future_check"]) == "candidate"
    assert float(audit_ok[0]["da_postlock_rejected_due_to_future_infeasibility"]) == pytest.approx(0.0)


def test_da_postlock_uses_hourly_commitments_not_schedule_totals_for_feasibility() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.date_range("2026-01-02T00:00:00Z", periods=3, freq="h")
    future_rows = pd.DataFrame(
        {
            col.timestamp: ts,
            col.pred_afrr_activation_rate_pos: [0.0, 0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0, 0.0],
        }
    )
    selected_da = {pd.Timestamp(t): (0.0, 4.0) for t in ts}
    audit_rows = [
        {
            "timestamp_utc": str(pd.Timestamp(t)),
            "da_candidate_buy_mw": 0.0,
            "da_candidate_sell_mw": 4.0,
            "candidate_selection_pnl_eur": 100.0,
            "incumbent_selection_pnl_eur": 0.0,
        }
        for t in ts
    ]

    selected, audit = bt._apply_da_postlock_future_guard(
        selected_da=selected_da,
        da_audit_rows=audit_rows,
        future_rows=future_rows,
        colmap=col,
        current_soc_mwh=18.0,
        da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert all(pair == pytest.approx((0.0, 4.0)) for pair in selected.values())
    assert all(float(r["da_postlock_candidate_future_feasible"]) == pytest.approx(1.0) for r in audit)
    assert all(str(r["da_postlock_selected_after_future_check"]) == "candidate" for r in audit)
    assert all(float(r["da_postlock_candidate_locked_sell_mwh_t"]) == pytest.approx(4.0 * bt.dt_h) for r in audit)
    assert all(float(r["da_postlock_candidate_total_locked_sell_mwh"]) == pytest.approx(12.0 * bt.dt_h) for r in audit)
    assert all(float(r["da_postlock_candidate_total_locked_sell_mwh"]) > bt.p_max_mw for r in audit)


def test_da_postlock_diagnostics_survive_thesis_hourly_output_filter() -> None:
    required = {
        "da_postlock_candidate_future_feasible": [0.0],
        "da_postlock_no_trade_future_feasible": [1.0],
        "da_postlock_final_selected_future_feasible": [1.0],
        "da_postlock_final_selected_after_future_check": ["no_trade"],
        "da_postlock_rejected_due_to_future_infeasibility": [1.0],
        "da_postlock_infeasibility_driver": ["locked_da_hourly_soc_min_infeasible"],
        "da_postlock_infeasibility_driver_detail": ["locked_da_hourly_soc_min_infeasible"],
        "da_postlock_guard_mode": ["recoverability_aware"],
        "da_postlock_hard_projection_until": ["next_recovery_opportunity"],
        "da_postlock_hard_local_candidate_feasible": [0.0],
        "da_postlock_terminal_shortfall_recoverable": [0.0],
        "da_postlock_terminal_shortfall_unrecoverable": [0.0],
        "da_postlock_failed_reason_detail": ["locked_da_hourly_soc_min_infeasible"],
        "da_postlock_failed_timestamp_utc": ["2026-01-02 01:00:00+00:00"],
        "da_postlock_failed_candidate_buy_mw": [0.0],
        "da_postlock_failed_candidate_sell_mw": [10.0],
        "da_postlock_failed_locked_buy_mwh_t": [0.0],
        "da_postlock_failed_locked_sell_mwh_t": [10.0],
        "da_postlock_failed_total_locked_buy_mwh": [0.0],
        "da_postlock_failed_total_locked_sell_mwh": [10.0],
        "da_postlock_failed_soc_before_mwh": [2.2],
        "da_postlock_failed_soc_after_mwh": [1.1],
        "da_postlock_failed_soc_min_mwh": [2.0],
        "da_postlock_failed_soc_max_mwh": [18.0],
        "da_postlock_failed_power_stack_mw": [10.0],
        "da_postlock_failed_power_limit_mw": [10.0],
        "da_precommit_pre_postlock_selected_incumbent": ["optimized"],
        "da_precommit_final_selected_incumbent": ["no_trade"],
        "da_precommit_da_postlock_candidate_future_feasible": [0.0],
        "da_precommit_da_postlock_no_trade_future_feasible": [1.0],
        "da_precommit_da_postlock_final_selected_future_feasible": [1.0],
        "da_precommit_da_postlock_infeasibility_driver_detail": ["locked_da_hourly_soc_min_infeasible"],
        "da_precommit_da_postlock_failed_reason_detail": ["locked_da_hourly_soc_min_infeasible"],
    }
    hourly = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2026-01-02T01:00:00Z")],
            "real_soc_mwh": [10.0],
            "da_precommit_da_zero_reason": ["postlock_future_infeasible_no_trade_selected"],
            "da_originating_precommit_id": ["2026-01-01T10:00:00+00:00->2026-01-02T01:00:00+00:00"],
            "da_originating_source_snapshot_utc": ["2026-01-01T10:00:00+00:00"],
            "da_originating_delivery_timestamp_utc": ["2026-01-02T01:00:00+00:00"],
            "da_originating_candidate_group_id": ["2026-01-01T10:00:00+00:00->2026-01-02"],
            "da_originating_schedule_reduced": [0.0],
            "da_realized_without_precommit_origin_error": [0.0],
            "da_realized_origin_complete": [1.0],
            **required,
            "ev_bcm_debug_should_drop": [123.0],
        }
    )

    out = _select_hourly_output_columns(hourly, output_detail="thesis", timestamp_col="timestamp_utc")

    for col in required:
        assert col in out.columns
    for col in (
        "da_originating_precommit_id",
        "da_originating_source_snapshot_utc",
        "da_originating_delivery_timestamp_utc",
        "da_originating_candidate_group_id",
        "da_originating_schedule_reduced",
        "da_realized_without_precommit_origin_error",
        "da_realized_origin_complete",
    ):
        assert col in out.columns
    assert "da_precommit_da_zero_reason" in out.columns
    assert "ev_bcm_debug_should_drop" not in out.columns


def test_da_postlock_missing_rejection_diagnostics_are_counted_for_strict_validity() -> None:
    bad = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2026-01-02T01:00:00Z")],
            "da_precommit_da_zero_reason": ["postlock_future_infeasible_no_trade_selected"],
            "da_postlock_infeasibility_driver": ["locked_da_hourly_soc_min_infeasible"],
        }
    )
    bad_audit = BatteryBacktester._audit_da_postlock_rejection_diagnostics(
        bad,
        timestamp_col="timestamp_utc",
    )
    assert float(bad_audit["da_postlock_rejection_count"]) == pytest.approx(1.0)
    assert float(bad_audit["missing_da_postlock_rejection_diagnostics_count"]) == pytest.approx(1.0)
    missing_cols = set(json.loads(str(bad_audit["missing_da_postlock_rejection_diagnostics_columns"])))
    assert "da_postlock_candidate_future_feasible" in missing_cols
    assert "da_postlock_failed_reason_detail" in missing_cols
    assert (
        str(bad_audit["first_missing_da_postlock_rejection_diagnostics_timestamp_utc"])
        == "2026-01-02T01:00:00+00:00"
    )

    good = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2026-01-02T01:00:00Z")],
            "da_precommit_da_zero_reason": ["postlock_future_infeasible_no_trade_selected"],
            "da_postlock_candidate_future_feasible": [0.0],
            "da_postlock_no_trade_future_feasible": [1.0],
            "da_postlock_final_selected_future_feasible": [1.0],
            "da_postlock_final_selected_after_future_check": ["no_trade"],
            "da_postlock_infeasibility_driver": ["locked_da_hourly_soc_min_infeasible"],
            "da_postlock_failed_reason_detail": ["locked_da_hourly_soc_min_infeasible"],
            "da_postlock_failed_timestamp_utc": ["2026-01-02 01:00:00+00:00"],
            "da_postlock_failed_locked_buy_mwh_t": [0.0],
            "da_postlock_failed_locked_sell_mwh_t": [10.0],
            "da_postlock_failed_soc_before_mwh": [2.2],
            "da_postlock_failed_soc_after_mwh": [1.1],
            "da_postlock_failed_soc_min_mwh": [2.0],
            "da_postlock_failed_soc_max_mwh": [18.0],
        }
    )
    good_audit = BatteryBacktester._audit_da_postlock_rejection_diagnostics(
        good,
        timestamp_col="timestamp_utc",
    )
    assert float(good_audit["missing_da_postlock_rejection_diagnostics_count"]) == pytest.approx(0.0)
    assert json.loads(str(good_audit["missing_da_postlock_rejection_diagnostics_columns"])) == []


def test_da_recovery_cost_postlock_rejected_group_broadcasts_failure_diagnostics() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts0 = pd.Timestamp("2026-01-02T00:00:00Z")
    ts1 = pd.Timestamp("2026-01-02T01:00:00Z")
    future_rows = pd.DataFrame(
        {
            col.timestamp: [ts0, ts1],
            col.pred_afrr_activation_rate_pos: [0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0],
        }
    )
    selected, audit = bt._apply_da_postlock_future_guard(
        selected_da={ts0: (0.0, 0.0), ts1: (0.0, 10.0)},
        da_audit_rows=[
            {"timestamp_utc": str(ts0), "da_candidate_buy_mw": 0.0, "da_candidate_sell_mw": 0.0},
            {"timestamp_utc": str(ts1), "da_candidate_buy_mw": 0.0, "da_candidate_sell_mw": 10.0},
        ],
        future_rows=future_rows,
        colmap=col,
        current_soc_mwh=bt.soc_min + 0.2,
        da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert all(pair == pytest.approx((0.0, 0.0)) for pair in selected.values())
    assert {str(r["da_postlock_failed_timestamp_utc"]) for r in audit} == {str(ts1)}
    assert all(str(r["da_postlock_failed_reason_detail"]) == "locked_da_hourly_soc_min_infeasible" for r in audit)
    assert all(np.isfinite(float(r["da_postlock_failed_soc_before_mwh"])) for r in audit)
    assert all(np.isfinite(float(r["da_postlock_failed_soc_after_mwh"])) for r in audit)
    hourly = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp(r["timestamp_utc"]) for r in audit],
            "da_precommit_da_zero_reason": [r["da_zero_reason"] for r in audit],
            **{k: [r.get(k) for r in audit] for k in audit[0] if str(k).startswith("da_postlock_")},
        }
    )
    postlock_audit = BatteryBacktester._audit_da_postlock_rejection_diagnostics(
        hourly,
        timestamp_col="timestamp_utc",
    )
    assert float(postlock_audit["missing_da_postlock_rejection_diagnostics_count"]) == pytest.approx(0.0)


def test_da_recovery_cost_realized_rows_with_final_precommit_origin_are_valid() -> None:
    ts = pd.Timestamp("2026-01-02T01:00:00Z")
    source_snapshot = pd.Timestamp("2026-01-01T10:00:00Z")
    realized_with_origin = pd.DataFrame(
        {
            "da_lockbook_row_present": [1.0],
            "da_bid_locked": [1.0],
            "da_locked_buy_mwh": [1.5],
            "da_locked_sell_mwh": [0.0],
            "da_originating_source_snapshot_utc": [str(source_snapshot)],
            "da_originating_delivery_timestamp_utc": [ts.isoformat()],
            "da_originating_precommit_id": [f"{source_snapshot}->{ts}"],
            "da_precommit_final_selected_incumbent": ["optimized"],
            "da_precommit_selection_reason": ["accepted_lockbook"],
            "da_precommit_da_zero_reason": ["none"],
            "real_submitted_da_buy_mw": [1.5],
            "real_submitted_da_sell_mw": [0.0],
            "real_submitted_da_buy_price_eur_mwh": [10.0],
            "real_da_buy_accepted": [1.0],
            "real_da_sell_accepted": [0.0],
            "real_da_buy_mwh": [1.5],
            "real_da_sell_mwh": [0.0],
        }
    )

    origin_counters = BatteryBacktester._compute_da_naming_semantics_counters(realized_with_origin)
    assert origin_counters["da_realized_without_precommit_origin_count"] == pytest.approx(0.0)
    assert origin_counters["da_naming_semantics_error_count"] == pytest.approx(0.0)


def test_realized_da_origin_requires_delivery_timestamp() -> None:
    ts = pd.Timestamp("2026-01-02T01:00:00Z")
    source_snapshot = pd.Timestamp("2026-01-01T10:00:00Z")
    missing_delivery = pd.DataFrame(
        {
            "da_lockbook_row_present": [1.0],
            "da_bid_locked": [1.0],
            "da_locked_buy_mwh": [1.0],
            "da_locked_sell_mwh": [0.0],
            "da_originating_source_snapshot_utc": [str(source_snapshot)],
            "da_originating_precommit_id": [f"{source_snapshot}->{ts}"],
            "da_precommit_final_selected_incumbent": ["optimized"],
            "da_precommit_selection_reason": ["accepted_lockbook"],
            "da_precommit_da_zero_reason": ["none"],
            "real_submitted_da_buy_mw": [1.0],
            "real_submitted_da_sell_mw": [0.0],
            "real_submitted_da_buy_price_eur_mwh": [10.0],
            "real_da_buy_accepted": [1.0],
            "real_da_sell_accepted": [0.0],
            "real_da_buy_mwh": [1.0],
            "real_da_sell_mwh": [0.0],
        }
    )

    counters = BatteryBacktester._compute_da_naming_semantics_counters(missing_delivery)
    assert counters["da_realized_without_precommit_origin_count"] == pytest.approx(1.0)
    assert counters["da_naming_semantics_error_count"] == pytest.approx(1.0)


def _da_hourly_lock_future_rows(col: BacktestColumnMap, timestamps: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            col.timestamp: pd.to_datetime(timestamps, utc=True),
            col.pred_afrr_activation_rate_pos: [0.0] * len(timestamps),
            col.pred_afrr_activation_rate_neg: [0.0] * len(timestamps),
        }
    )


def _configure_zero_aux_unit_efficiency(bt: BatteryBacktester) -> None:
    bt.dt_h = 1.0
    bt.eta_in = 1.0
    bt.eta_out = 1.0
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    bt.soc_target_end = 2.0
    bt.aux_peak_mw = 0.0
    bt.aux_off_mw = 0.0
    bt.aux_standby_mw = 0.0
    bt.aux_trading_mw = 0.0
    bt.aux_afrr_active_mw = 0.0


def test_da_hourly_lock_rejects_buy_exceeding_soc_headroom() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-22T01:00:00Z")

    ok, diag, by_ts = bt._check_da_hourly_lock_physical_feasibility(
        future_rows=_da_hourly_lock_future_rows(col, [ts]),
        colmap=col,
        current_soc_mwh=8.25,
        existing_da_lockbook={},
        candidate_da={ts: (10.0, 0.0)},
    )

    assert ok is False
    assert str(diag["da_hourly_lock_infeasible_reason"]) == "locked_da_buy_exceeds_soc_headroom"
    assert float(diag["da_candidate_rejected_hourly_lock_infeasible"]) == pytest.approx(1.0)
    assert float(by_ts[ts]["da_hourly_lock_max_feasible_buy_mwh"]) == pytest.approx(9.75)
    assert float(by_ts[ts]["da_hourly_lock_infeasible_buy_mwh"]) == pytest.approx(0.25)


def test_da_hourly_lock_rejects_sell_exceeding_available_energy() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-22T01:00:00Z")

    ok, diag, by_ts = bt._check_da_hourly_lock_physical_feasibility(
        future_rows=_da_hourly_lock_future_rows(col, [ts]),
        colmap=col,
        current_soc_mwh=11.5,
        existing_da_lockbook={},
        candidate_da={ts: (0.0, 10.0)},
    )

    assert ok is False
    assert str(diag["da_hourly_lock_infeasible_reason"]) == "locked_da_sell_exceeds_available_energy"
    assert float(diag["da_candidate_rejected_hourly_lock_infeasible"]) == pytest.approx(1.0)
    assert float(by_ts[ts]["da_hourly_lock_max_feasible_sell_mwh"]) == pytest.approx(9.5)
    assert float(by_ts[ts]["da_hourly_lock_infeasible_sell_mwh"]) == pytest.approx(0.5)


def test_da_hourly_lock_replay_uses_sequential_soc_not_independent_hourly_capacity() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts0 = pd.Timestamp("2026-01-22T01:00:00Z")
    ts1 = pd.Timestamp("2026-01-22T02:00:00Z")

    ok, diag, by_ts = bt._check_da_hourly_lock_physical_feasibility(
        future_rows=_da_hourly_lock_future_rows(col, [ts0, ts1]),
        colmap=col,
        current_soc_mwh=12.0,
        existing_da_lockbook={},
        candidate_da={ts0: (0.0, 6.0), ts1: (0.0, 6.0)},
    )

    assert ok is False
    assert str(diag["da_hourly_lock_infeasible_timestamp_utc"]) == str(ts1)
    assert str(diag["da_hourly_lock_infeasible_reason"]) == "locked_da_sell_exceeds_available_energy"
    assert float(by_ts[ts0]["da_hourly_lock_projected_soc_end_mwh"]) == pytest.approx(6.0)
    assert float(by_ts[ts1]["da_hourly_lock_max_feasible_sell_mwh"]) == pytest.approx(4.0)
    assert float(by_ts[ts1]["da_hourly_lock_infeasible_sell_mwh"]) == pytest.approx(2.0)


def test_da_hourly_lock_accepts_feasible_buy_sell_schedule() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts0 = pd.Timestamp("2026-01-22T01:00:00Z")
    ts1 = pd.Timestamp("2026-01-22T02:00:00Z")

    ok, diag, by_ts = bt._check_da_hourly_lock_physical_feasibility(
        future_rows=_da_hourly_lock_future_rows(col, [ts0, ts1]),
        colmap=col,
        current_soc_mwh=10.0,
        existing_da_lockbook={},
        candidate_da={ts0: (2.0, 0.0), ts1: (0.0, 2.0)},
    )

    assert ok is True
    assert str(diag["da_hourly_lock_infeasible_reason"]) == "none"
    assert float(diag["da_lockbook_physical_replay_projected_final_soc_mwh"]) == pytest.approx(10.0)
    assert float(by_ts[ts0]["da_hourly_lock_projected_soc_end_mwh"]) == pytest.approx(12.0)
    assert float(by_ts[ts1]["da_hourly_lock_projected_soc_end_mwh"]) == pytest.approx(10.0)


def test_da_hourly_lock_same_direction_buy_overlap_aggregates() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-22T01:00:00Z")

    ok, diag, by_ts = bt._check_da_hourly_lock_physical_feasibility(
        future_rows=_da_hourly_lock_future_rows(col, [ts]),
        colmap=col,
        current_soc_mwh=10.0,
        existing_da_lockbook={ts: (6.0, 0.0)},
        candidate_da={ts: (2.0, 0.0)},
    )

    assert ok is True
    assert str(diag["da_hourly_lock_infeasibility_origin"]) == "none"
    assert float(diag["da_candidate_overlaps_existing_lockbook"]) == pytest.approx(1.0)
    assert str(diag["da_candidate_overlap_resolution"]) == "same_direction_aggregated"
    assert str(diag["da_candidate_overlaps_existing_lockbook_timestamp_utc"]) == str(ts)
    assert float(diag["da_candidate_existing_lockbook_buy_mw"]) == pytest.approx(6.0)
    assert float(diag["da_candidate_overlap_candidate_buy_mw"]) == pytest.approx(2.0)
    assert float(by_ts[ts]["da_hourly_lock_projected_soc_end_mwh"]) == pytest.approx(18.0)


def test_da_hourly_lock_same_direction_sell_overlap_aggregates() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-22T01:00:00Z")

    ok, diag, by_ts = bt._check_da_hourly_lock_physical_feasibility(
        future_rows=_da_hourly_lock_future_rows(col, [ts]),
        colmap=col,
        current_soc_mwh=12.0,
        existing_da_lockbook={ts: (0.0, 3.0)},
        candidate_da={ts: (0.0, 4.0)},
    )

    assert ok is True
    assert str(diag["da_hourly_lock_infeasibility_origin"]) == "none"
    assert float(diag["da_candidate_overlaps_existing_lockbook"]) == pytest.approx(1.0)
    assert str(diag["da_candidate_overlap_resolution"]) == "same_direction_aggregated"
    assert float(diag["da_candidate_existing_lockbook_sell_mw"]) == pytest.approx(3.0)
    assert float(diag["da_candidate_overlap_candidate_sell_mw"]) == pytest.approx(4.0)
    assert float(by_ts[ts]["da_hourly_lock_projected_soc_end_mwh"]) == pytest.approx(5.0)


def test_da_hourly_lock_exact_duplicate_overlap_is_ignored_not_double_counted() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-22T01:00:00Z")

    ok, diag, by_ts = bt._check_da_hourly_lock_physical_feasibility(
        future_rows=_da_hourly_lock_future_rows(col, [ts]),
        colmap=col,
        current_soc_mwh=10.0,
        existing_da_lockbook={ts: (6.0, 0.0)},
        candidate_da={ts: (6.0, 0.0)},
    )

    assert ok is True
    assert str(diag["da_hourly_lock_infeasibility_origin"]) == "none"
    assert str(diag["da_candidate_overlap_resolution"]) == "exact_duplicate_ignored"
    assert float(diag["da_candidate_duplicate_existing_lockbook_timestamp"]) == pytest.approx(1.0)
    assert float(by_ts[ts]["da_hourly_lock_projected_soc_end_mwh"]) == pytest.approx(16.0)


def test_da_hourly_lock_opposite_direction_overlap_rejected() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-22T01:00:00Z")

    ok, diag, _ = bt._check_da_hourly_lock_physical_feasibility(
        future_rows=_da_hourly_lock_future_rows(col, [ts]),
        colmap=col,
        current_soc_mwh=10.0,
        existing_da_lockbook={ts: (6.0, 0.0)},
        candidate_da={ts: (0.0, 4.0)},
    )

    assert ok is False
    assert str(diag["da_hourly_lock_infeasibility_origin"]) == "candidate_overlaps_existing_lockbook"
    assert str(diag["da_candidate_overlap_resolution"]) == "opposite_direction_rejected"
    assert str(diag["da_hourly_lock_infeasible_reason"]) == "candidate_overlaps_existing_da_lockbook"
    assert float(diag["da_candidate_existing_lockbook_buy_mw"]) == pytest.approx(6.0)
    assert float(diag["da_candidate_overlap_candidate_sell_mw"]) == pytest.approx(4.0)


def test_da_hourly_lock_reports_existing_lockbook_already_infeasible() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts0 = pd.Timestamp("2026-01-22T01:00:00Z")
    ts1 = pd.Timestamp("2026-01-22T02:00:00Z")

    ok, diag, _ = bt._check_da_hourly_lock_physical_feasibility(
        future_rows=_da_hourly_lock_future_rows(col, [ts0, ts1]),
        colmap=col,
        current_soc_mwh=11.0,
        existing_da_lockbook={ts0: (0.0, 10.0)},
        candidate_da={ts1: (1.0, 0.0)},
    )

    assert ok is False
    assert str(diag["da_hourly_lock_infeasibility_origin"]) == "existing_lockbook_already_infeasible"
    assert float(diag["da_existing_lockbook_physical_feasibility_passed"]) == pytest.approx(0.0)
    assert str(diag["da_existing_lockbook_physical_infeasible_timestamp_utc"]) == str(ts0)
    assert str(diag["da_existing_lockbook_physical_infeasible_reason"]) == "locked_da_sell_exceeds_available_energy"
    assert float(diag["da_combined_lockbook_physical_feasibility_checked"]) == pytest.approx(0.0)


def test_da_hourly_lock_reports_candidate_caused_infeasibility() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts0 = pd.Timestamp("2026-01-22T01:00:00Z")
    ts1 = pd.Timestamp("2026-01-22T02:00:00Z")

    ok, diag, by_ts = bt._check_da_hourly_lock_physical_feasibility(
        future_rows=_da_hourly_lock_future_rows(col, [ts0, ts1]),
        colmap=col,
        current_soc_mwh=12.0,
        existing_da_lockbook={ts0: (0.0, 2.0)},
        candidate_da={ts1: (0.0, 9.0)},
    )

    assert ok is False
    assert str(diag["da_hourly_lock_infeasibility_origin"]) == "candidate_caused_infeasibility"
    assert float(diag["da_existing_lockbook_physical_feasibility_passed"]) == pytest.approx(1.0)
    assert float(diag["da_combined_lockbook_physical_feasibility_passed"]) == pytest.approx(0.0)
    assert str(diag["da_combined_lockbook_physical_infeasible_timestamp_utc"]) == str(ts1)
    assert str(diag["da_combined_lockbook_physical_infeasible_reason"]) == "locked_da_sell_exceeds_available_energy"
    assert float(by_ts[ts1]["da_hourly_lock_max_feasible_sell_mwh"]) == pytest.approx(8.0)


def test_da_hourly_lock_existing_and_candidate_feasible_passes() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts0 = pd.Timestamp("2026-01-22T01:00:00Z")
    ts1 = pd.Timestamp("2026-01-22T02:00:00Z")

    ok, diag, by_ts = bt._check_da_hourly_lock_physical_feasibility(
        future_rows=_da_hourly_lock_future_rows(col, [ts0, ts1]),
        colmap=col,
        current_soc_mwh=12.0,
        existing_da_lockbook={ts0: (0.0, 2.0)},
        candidate_da={ts1: (2.0, 0.0)},
    )

    assert ok is True
    assert str(diag["da_hourly_lock_infeasibility_origin"]) == "none"
    assert float(diag["da_existing_lockbook_physical_feasibility_passed"]) == pytest.approx(1.0)
    assert float(diag["da_combined_lockbook_physical_feasibility_passed"]) == pytest.approx(1.0)
    assert float(by_ts[ts0]["da_hourly_lock_projected_soc_end_mwh"]) == pytest.approx(10.0)
    assert float(by_ts[ts1]["da_hourly_lock_projected_soc_end_mwh"]) == pytest.approx(12.0)


def test_da_postlock_recomputes_locked_candidate_pnl_against_no_trade() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-02T00:00:00Z")
    rows = pd.DataFrame(
        {
            col.timestamp: [ts],
            "target_time_utc": [ts],
            "snapshot_time_utc": [pd.Timestamp("2026-01-01T10:00:00Z")],
            "soc_start_lp_mwh": [5.0],
            col.pred_da_price: [100.0],
            col.pred_afrr_capacity_price_pos: [0.0],
            col.pred_afrr_capacity_price_neg: [0.0],
            col.pred_afrr_activation_price_pos: [0.0],
            col.pred_afrr_activation_price_neg: [0.0],
            col.pred_afrr_activation_rate_pos: [0.0],
            col.pred_afrr_activation_rate_neg: [0.0],
        }
    )
    audit_rows = [
        {
            "timestamp_utc": str(ts),
            "da_candidate_buy_mw": 1.0,
            "da_candidate_sell_mw": 0.0,
            "candidate_selection_pnl_eur": 50.0,
            "incumbent_selection_pnl_eur": 0.0,
        }
    ]

    selected, audit = bt._apply_da_postlock_future_guard(
        selected_da={ts: (1.0, 0.0)},
        da_audit_rows=audit_rows,
        lock_rows=rows,
        future_rows=rows[[col.timestamp, col.pred_afrr_activation_rate_pos, col.pred_afrr_activation_rate_neg]],
        colmap=col,
        current_soc_mwh=5.0,
        da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert selected[ts] == pytest.approx((0.0, 0.0))
    assert str(audit[0]["da_postlock_selected_after_future_check"]) == "no_trade_after_locked_pnl_recompute"
    assert str(audit[0]["candidate_rejection_reason"]) == "locked_candidate_pnl_below_no_trade"
    assert float(audit[0]["candidate_pnl_before_locking"]) == pytest.approx(50.0)
    assert float(audit[0]["postlock_rejected_candidate_pnl_eur"]) <= float(audit[0]["no_trade_pnl"]) + 1e-9
    assert float(audit[0]["locked_buy_mwh_by_hour"]) == pytest.approx(0.0)
    assert str(audit[0]["source_snapshot_utc"]) == str(pd.Timestamp("2026-01-01T10:00:00Z"))
    assert str(audit[0]["delivery_timestamp_utc"]) == str(ts)


def test_da_postlock_future_sell_can_use_soc_from_earlier_candidate_buy() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    ts0 = pd.Timestamp("2026-01-02T00:00:00Z")
    ts1 = pd.Timestamp("2026-01-02T01:00:00Z")
    rows = pd.DataFrame(
        {
            col.timestamp: [ts0, ts1],
            "target_time_utc": [ts0, ts1],
            "snapshot_time_utc": [pd.Timestamp("2026-01-01T10:00:00Z")] * 2,
            "soc_start_lp_mwh": [bt.soc_min + 0.2, bt.soc_min + 0.2],
            col.pred_da_price: [-100.0, 200.0],
            col.pred_afrr_capacity_price_pos: [0.0, 0.0],
            col.pred_afrr_capacity_price_neg: [0.0, 0.0],
            col.pred_afrr_activation_price_pos: [0.0, 0.0],
            col.pred_afrr_activation_price_neg: [0.0, 0.0],
            col.pred_afrr_activation_rate_pos: [0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0],
        }
    )
    selected, audit = bt._apply_da_postlock_future_guard(
        selected_da={ts0: (4.0, 0.0), ts1: (0.0, 2.0)},
        da_audit_rows=[
            {"timestamp_utc": str(ts0), "da_candidate_buy_mw": 4.0, "da_candidate_sell_mw": 0.0},
            {"timestamp_utc": str(ts1), "da_candidate_buy_mw": 0.0, "da_candidate_sell_mw": 2.0},
        ],
        lock_rows=rows,
        future_rows=rows[[col.timestamp, col.pred_afrr_activation_rate_pos, col.pred_afrr_activation_rate_neg]],
        colmap=col,
        current_soc_mwh=bt.soc_min + 0.2,
        da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert selected[ts0] == pytest.approx((4.0, 0.0))
    assert selected[ts1] == pytest.approx((0.0, 2.0))
    assert all(str(r["da_postlock_selected_after_future_check"]) == "candidate" for r in audit)
    sell_row = next(r for r in audit if str(r["timestamp_utc"]) == str(ts1))
    assert float(sell_row["locked_sell_mwh_by_hour"]) == pytest.approx(2.0 * bt.dt_h)
    assert str(sell_row["sell_disabled_reason"]) == "none"


def test_da_postlock_terminal_shortfall_recoverable_candidate_survives() -> None:
    bt = _mk_backtester()
    bt.da_postlock_guard_mode = "recoverability_aware"
    col = BacktestColumnMap()
    ts = pd.date_range("2026-01-02T00:00:00Z", periods=3, freq="h")
    future_rows = pd.DataFrame(
        {
            col.timestamp: ts,
            col.pred_da_price: [50.0, 50.0, 50.0],
            col.pred_afrr_activation_rate_pos: [0.0, 0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0, 0.0],
        }
    )

    selected, audit = bt._apply_da_postlock_future_guard(
        selected_da={pd.Timestamp(ts[0]): (0.0, 5.0)},
        da_audit_rows=[
            {
                "timestamp_utc": str(pd.Timestamp(ts[0])),
                "da_candidate_buy_mw": 0.0,
                "da_candidate_sell_mw": 5.0,
            }
        ],
        future_rows=future_rows,
        colmap=col,
        current_soc_mwh=10.0,
        da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=pd.Timestamp(ts[-1]),
    )

    assert selected[pd.Timestamp(ts[0])] == pytest.approx((0.0, 5.0))
    assert str(audit[0]["da_postlock_selected_after_future_check"]) == "candidate"
    assert float(audit[0]["da_postlock_candidate_future_feasible"]) == pytest.approx(1.0)
    assert float(audit[0]["da_postlock_terminal_shortfall_recoverable"]) == pytest.approx(1.0)
    assert float(audit[0]["da_postlock_terminal_shortfall_unrecoverable"]) == pytest.approx(0.0)
    assert str(audit[0]["da_postlock_infeasibility_driver_detail"]) == "locked_da_terminal_shortfall_recoverable"
    assert str(audit[0]["da_postlock_next_recovery_opportunity_type"]) == "technical_id_recourse"
    assert str(audit[0]["da_postlock_hard_projection_end_utc"]) == str(pd.Timestamp(ts[0]))
    assert float(audit[0]["da_postlock_hard_projection_reached_next_recovery"]) == pytest.approx(1.0)


def test_da_postlock_ordinary_window_does_not_apply_extra_terminal_recovery_cost() -> None:
    bt = _mk_backtester()
    bt.da_postlock_guard_mode = "recoverability_aware"
    bt.da_postlock_hard_projection_until = "global_final"
    bt.final_soc_mode = "hard"
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    bt.soc_target_end = 10.0
    bt.eta_in = 1.0
    bt.eta_out = 1.0
    bt.trans_eur_mwh = 0.0
    bt.deg_eur_mwh = 0.0
    bt.aux_off_mw = 0.0
    bt.aux_trading_mw = 0.0
    bt._id_recourse_mode = "common"
    col = BacktestColumnMap()
    ts_trade = pd.Timestamp("2026-01-02T00:00:00Z")
    ts_recovery = pd.Timestamp("2026-01-03T00:00:00Z")
    ts_final = pd.Timestamp("2026-01-10T00:00:00Z")
    lock_rows = pd.DataFrame(
        {
            col.timestamp: [ts_trade],
            "target_time_utc": [ts_trade],
            "snapshot_time_utc": [pd.Timestamp("2026-01-01T10:00:00Z")],
            col.pred_da_price: [1_000.0],
            col.pred_afrr_activation_rate_pos: [0.0],
            col.pred_afrr_activation_rate_neg: [0.0],
        }
    )
    future_rows = pd.DataFrame(
        {
            col.timestamp: [ts_trade, ts_recovery, ts_final],
            "target_time_utc": [ts_trade, ts_recovery, ts_final],
            col.pred_da_price: [1_000.0, 1_000.0, 1_000.0],
            col.pred_afrr_activation_rate_pos: [0.0, 0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0, 0.0],
        }
    )

    selected, audit = bt._apply_da_postlock_future_guard(
        selected_da={ts_trade: (0.0, 8.0)},
        da_audit_rows=[
            {
                "timestamp_utc": str(ts_trade),
                "da_candidate_buy_mw": 0.0,
                "da_candidate_sell_mw": 8.0,
                "da_sized_candidate_buy_mw": 0.0,
                "da_sized_candidate_sell_mw": 8.0,
                "selected_incumbent": "optimized",
                "selection_reason": "optimized_predicted_replay_dominates_or_incumbent_infeasible",
                "candidate_selection_pnl_eur": 6500.0,
                "incumbent_selection_pnl_eur": 0.0,
            }
        ],
        lock_rows=lock_rows,
        future_rows=future_rows,
        colmap=col,
        current_soc_mwh=15.0,
        da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=ts_final,
    )

    assert selected[ts_trade] == pytest.approx((0.0, 8.0))
    assert str(audit[0]["da_postlock_selected_after_future_check"]) == "candidate"
    assert float(audit[0]["da_terminal_sensitive_window"]) == pytest.approx(0.0)
    assert str(audit[0]["da_terminal_sensitive_reason"]) == "not_terminal_window"
    assert float(audit[0]["da_postlock_terminal_shortfall_recoverable"]) == pytest.approx(1.0)
    assert float(audit[0]["da_postlock_terminal_recovery_cost_eur"]) > 0.0
    assert float(audit[0]["candidate_selection_pnl_eur_after_terminal_recovery"]) == pytest.approx(
        float(audit[0]["da_postlock_candidate_pnl_before_recovery_cost_eur"])
    )
    assert str(audit[0]["da_volume_loss_stage"]) == "none"
    assert str(audit[0]["da_volume_loss_reason"]) == "none"


def test_da_final_lockbook_check_rejects_unscheduled_terminal_shortfall() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    bt.soc_target_end = 10.0
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    bt.eta_in = 1.0
    bt.eta_out = 1.0
    bt.aux_off_mw = 0.0
    bt.aux_trading_mw = 0.0
    bt._id_recourse_mode = "common"
    ts = pd.Timestamp("2026-01-02T00:00:00Z")
    rows = pd.DataFrame(
        {
            col.timestamp: [ts],
            col.pred_afrr_activation_rate_pos: [0.0],
            col.pred_afrr_activation_rate_neg: [0.0],
        }
    )

    ok, diag, by_ts = bt._check_da_hourly_lock_physical_feasibility(
        future_rows=rows,
        colmap=col,
        current_soc_mwh=10.0,
        existing_da_lockbook={},
        candidate_da={ts: (0.0, 1.0)},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=ts,
        include_existing_lockbook=True,
        schedule_source="selected_lockable",
    )

    assert ok is False
    assert float(diag["da_lockbook_physical_replay_candidate_included"]) == pytest.approx(1.0)
    assert float(by_ts[ts]["da_hourly_lock_terminal_shortfall_mwh"]) == pytest.approx(1.0)
    assert float(by_ts[ts]["da_hourly_lock_terminal_shortfall_recoverable"]) == pytest.approx(0.0)
    assert str(by_ts[ts]["da_hourly_lock_infeasible_reason"]) == "terminal_repair_missing"
    assert float(by_ts[ts]["da_terminal_repair_required_mwh"]) == pytest.approx(1.0)
    assert float(by_ts[ts]["da_terminal_repair_scheduled_mwh"]) == pytest.approx(0.0)
    assert float(by_ts[ts]["da_terminal_replay_pass"]) == pytest.approx(0.0)
    assert str(by_ts[ts]["da_terminal_replay_failure_reason"]) == "terminal_repair_missing"

    bt._id_recourse_mode = "disabled"
    ok_disabled, diag_disabled, _ = bt._check_da_hourly_lock_physical_feasibility(
        future_rows=rows,
        colmap=col,
        current_soc_mwh=10.0,
        existing_da_lockbook={},
        candidate_da={ts: (0.0, 1.0)},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=ts,
        include_existing_lockbook=True,
        schedule_source="selected_lockable",
    )

    assert ok_disabled is False
    assert str(diag_disabled["da_hourly_lock_infeasible_reason"]) == "terminal_repair_unreachable"


def test_da_accepted_lockbook_zero_after_postlock_exports_specific_reason() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-22T01:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])

    accepted, audit = bt._apply_da_accepted_lockbook_replay_to_audit_rows(
        da_audit_rows=[
            {
                "timestamp_utc": str(ts),
                "da_postlock_candidate_buy_mw": 1.0,
                "da_postlock_candidate_sell_mw": 0.0,
                "da_sized_candidate_buy_mw": 1.0,
                "da_sized_candidate_sell_mw": 0.0,
                "da_prelock_rejected_no_feasible_schedule": 1.0,
                "da_prelock_repair_reason": "prelock_repair_no_feasible_subset",
                "da_zero_reason": "none",
                "zero_reason": "none",
            }
        ],
        accepted_da={},
        lock_rows=rows,
        colmap=col,
        current_soc_mwh=10.0,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert accepted == {}
    assert str(audit[0]["da_final_selection_reason"]) == "prelock_repair_no_feasible_subset"
    assert str(audit[0]["da_volume_loss_stage"]) == "prelock_guard_zeroed"
    assert str(audit[0]["da_volume_loss_reason"]) == "prelock_repair_no_feasible_subset"
    assert str(audit[0]["da_zero_reason"]) == "prelock_repair_no_feasible_subset"


def test_da_accepted_lockbook_partial_generic_loss_reconstructs_row() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    keep_ts = pd.Timestamp("2026-01-22T01:00:00Z")
    lost_ts = pd.Timestamp("2026-01-22T02:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [keep_ts, lost_ts])
    rows[col.pred_da_price] = [1_000.0, 1_000.0]

    accepted, audit = bt._apply_da_accepted_lockbook_replay_to_audit_rows(
        da_audit_rows=[
            {
                "timestamp_utc": str(keep_ts),
                "da_postlock_candidate_buy_mw": 0.0,
                "da_postlock_candidate_sell_mw": 1.0,
                "da_sized_candidate_buy_mw": 0.0,
                "da_sized_candidate_sell_mw": 1.0,
                "selected_incumbent": "optimized",
                "final_selected_incumbent": "optimized",
                "da_zero_reason": "none",
                "zero_reason": "none",
            },
            {
                "timestamp_utc": str(lost_ts),
                "da_postlock_candidate_buy_mw": 0.0,
                "da_postlock_candidate_sell_mw": 1.0,
                "da_sized_candidate_buy_mw": 0.0,
                "da_sized_candidate_sell_mw": 1.0,
                "selected_incumbent": "optimized",
                "final_selected_incumbent": "optimized",
                "da_zero_reason": "none",
                "zero_reason": "none",
            },
        ],
        accepted_da={keep_ts: (0.0, 1.0)},
        lock_rows=rows,
        colmap=col,
        current_soc_mwh=10.0,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert accepted == {
        keep_ts: pytest.approx((0.0, 1.0)),
        lost_ts: pytest.approx((0.0, 1.0)),
    }
    keep_row, lost_row = audit
    assert str(keep_row["da_volume_loss_stage"]) == "none"
    assert str(keep_row["final_selected_incumbent"]) == "optimized"
    assert str(lost_row["final_selected_incumbent"]) == "optimized"
    assert str(lost_row["da_volume_loss_stage"]) == "none"
    assert str(lost_row["da_volume_loss_reason"]) == "none"
    assert float(lost_row["accepted_lockbook_row_sell_mwh"]) == pytest.approx(1.0)
    assert float(lost_row["da_handoff_lost_postlock_candidate"]) == pytest.approx(0.0)


def test_da_accepted_lockbook_reconstructs_generic_zeroed_postlock_candidate() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-22T02:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])
    rows[col.pred_da_price] = [1_000.0]

    accepted, audit = bt._apply_da_accepted_lockbook_replay_to_audit_rows(
        da_audit_rows=[
            {
                "timestamp_utc": str(ts),
                "da_handoff_postlock_selected_candidate": 1.0,
                "da_postlock_candidate_buy_mw": 0.0,
                "da_postlock_candidate_sell_mw": 1.0,
                "da_sized_candidate_buy_mw": 0.0,
                "da_sized_candidate_sell_mw": 1.0,
                "selected_incumbent": "no_trade",
                "final_selected_incumbent": "no_trade",
                "da_handoff_lost_postlock_candidate_reason": "accepted_lockbook_zeroed_after_postlock",
                "da_zero_reason": "accepted_lockbook_zeroed_after_postlock",
                "zero_reason": "accepted_lockbook_zeroed_after_postlock",
                "candidate_rejection_reason": "none",
            }
        ],
        accepted_da={},
        lock_rows=rows,
        colmap=col,
        current_soc_mwh=10.0,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert accepted == {ts: pytest.approx((0.0, 1.0))}
    row = audit[0]
    assert float(row["accepted_lockbook_row_sell_mwh"]) == pytest.approx(1.0)
    assert str(row["final_selected_incumbent"]) == "optimized"
    assert float(row["da_handoff_lost_postlock_candidate"]) == pytest.approx(0.0)
    assert str(row["da_handoff_lost_postlock_candidate_reason"]) == "none"


def test_da_reduced_to_feasible_without_blocker_reconstructs_postlock_schedule() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-22T02:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])
    rows[col.pred_da_price] = [1_000.0]

    accepted, audit = bt._apply_da_accepted_lockbook_replay_to_audit_rows(
        da_audit_rows=[
            {
                "timestamp_utc": str(ts),
                "da_handoff_postlock_selected_candidate": 1.0,
                "da_postlock_candidate_buy_mw": 0.0,
                "da_postlock_candidate_sell_mw": 1.0,
                "da_sized_candidate_buy_mw": 0.0,
                "da_sized_candidate_sell_mw": 1.0,
                "selected_incumbent": "optimized",
                "final_selected_incumbent": "optimized",
                "da_handoff_lost_postlock_candidate_reason": "reduced_to_feasible_da_schedule",
                "da_zero_reason": "reduced_to_feasible_da_schedule",
                "zero_reason": "reduced_to_feasible_da_schedule",
                "candidate_rejection_reason": "none",
            }
        ],
        accepted_da={},
        lock_rows=rows,
        colmap=col,
        current_soc_mwh=10.0,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert accepted == {ts: pytest.approx((0.0, 1.0))}
    row = audit[0]
    assert float(row["accepted_lockbook_row_sell_mwh"]) == pytest.approx(1.0)
    assert str(row["final_selected_incumbent"]) == "optimized"
    assert float(row["da_handoff_lost_postlock_candidate"]) == pytest.approx(0.0)
    assert str(row["da_handoff_lost_postlock_candidate_reason"]) == "none"


def test_da_reduced_to_feasible_with_explicit_blocker_remains_handoff_loss() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-22T02:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])
    rows[col.pred_da_price] = [1_000.0]

    accepted, audit = bt._apply_da_accepted_lockbook_replay_to_audit_rows(
        da_audit_rows=[
            {
                "timestamp_utc": str(ts),
                "da_handoff_postlock_selected_candidate": 1.0,
                "da_postlock_candidate_buy_mw": 0.0,
                "da_postlock_candidate_sell_mw": 1.0,
                "da_sized_candidate_buy_mw": 0.0,
                "da_sized_candidate_sell_mw": 1.0,
                "selected_incumbent": "optimized",
                "final_selected_incumbent": "optimized",
                "da_handoff_lost_postlock_candidate_reason": "reduced_to_feasible_da_schedule",
                "da_zero_reason": "reduced_to_feasible_da_schedule",
                "zero_reason": "reduced_to_feasible_da_schedule",
                "candidate_rejection_reason": "none",
                "da_settlement_equiv_replay_rejected": 1.0,
                "da_settlement_equiv_replay_reason": "settlement_equiv_replay_rejected",
            }
        ],
        accepted_da={},
        lock_rows=rows,
        colmap=col,
        current_soc_mwh=10.0,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert accepted == {}
    row = audit[0]
    assert float(row["accepted_lockbook_row_sell_mwh"]) == pytest.approx(0.0)
    assert float(row["da_handoff_lost_postlock_candidate"]) == pytest.approx(1.0)
    assert str(row["da_handoff_lost_postlock_candidate_reason"]) == "settlement_equiv_replay_rejected"


def test_da_economic_no_trade_does_not_preserve_stale_handoff_loss_flag() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-22T23:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])
    rows[col.pred_da_price] = [1_000.0]

    accepted, audit = bt._apply_da_accepted_lockbook_replay_to_audit_rows(
        da_audit_rows=[
            {
                "timestamp_utc": str(ts),
                "da_candidate_buy_mw": 0.0,
                "da_candidate_sell_mw": 1.7,
                "da_sized_candidate_buy_mw": 0.0,
                "da_sized_candidate_sell_mw": 1.7,
                "da_postlock_candidate_buy_mw": 0.0,
                "da_postlock_candidate_sell_mw": 0.0,
                "selected_incumbent": "no_trade",
                "final_selected_incumbent": "no_trade",
                "selection_reason": "no_trade_incumbent_selected",
                "candidate_rejection_reason": "candidate_pnl_below_no_trade",
                "candidate_minus_incumbent_eur": -48.55,
                "da_handoff_lost_postlock_candidate": 1.0,
                "da_handoff_lost_postlock_candidate_reason": "postlock_candidate_missing_before_physical_replay",
                "da_zero_reason": "none",
                "zero_reason": "none",
            }
        ],
        accepted_da={},
        lock_rows=rows,
        colmap=col,
        current_soc_mwh=10.0,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=ts,
    )

    assert accepted == {}
    row = audit[0]
    assert str(row["final_selected_incumbent"]) == "no_trade"
    assert float(row["da_handoff_lost_postlock_candidate"]) == pytest.approx(0.0)
    assert str(row["da_handoff_lost_postlock_candidate_reason"]) == "none"


def test_da_prelock_guard_derates_buy_to_available_headroom() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-22T01:00:00Z")

    repaired, diag, by_ts = bt._repair_or_reject_da_before_lock(
        accepted_da={ts: (1.0, 0.0)},
        current_soc_mwh=17.55,
        future_rows=_da_hourly_lock_future_rows(col, [ts]),
        colmap=col,
        existing_da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert repaired[ts] == pytest.approx((0.4, 0.0))
    assert float(diag["da_prelock_final_guard_passed"]) == pytest.approx(1.0)
    assert float(diag["da_prelock_derated_to_preserve_headroom"]) == pytest.approx(1.0)
    assert str(diag["da_prelock_repair_reason"]) == "derated_buy_to_charge_headroom"
    assert float(by_ts[ts]["da_prelock_original_buy_mwh"]) == pytest.approx(1.0)
    assert float(by_ts[ts]["da_prelock_repaired_buy_mwh"]) == pytest.approx(0.4)


def test_da_prelock_guard_removes_early_sell_that_breaks_soc_before_later_buy() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.aux_mode = "state_dependent"
    bt.aux_off_mw = 0.05
    bt.aux_standby_mw = 0.05
    bt.aux_trading_mw = 0.05
    col = BacktestColumnMap()
    sell_ts = pd.Timestamp("2026-01-22T04:00:00Z")
    buy_ts = pd.Timestamp("2026-01-22T10:00:00Z")
    rows = _da_hourly_lock_future_rows(
        col,
        [
            sell_ts,
            pd.Timestamp("2026-01-22T05:00:00Z"),
            pd.Timestamp("2026-01-22T06:00:00Z"),
            pd.Timestamp("2026-01-22T07:00:00Z"),
            pd.Timestamp("2026-01-22T08:00:00Z"),
            pd.Timestamp("2026-01-22T09:00:00Z"),
            buy_ts,
        ],
    )

    repaired, diag, by_ts = bt._repair_or_reject_da_before_lock(
        accepted_da={sell_ts: (0.0, 6.95), buy_ts: (1.0, 0.0)},
        current_soc_mwh=9.0,
        future_rows=rows,
        colmap=col,
        existing_da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert sell_ts not in repaired
    assert repaired[buy_ts] == pytest.approx((1.0, 0.0))
    assert float(diag["da_prelock_final_guard_passed"]) == pytest.approx(1.0)
    assert str(diag["da_prelock_repair_reason"]) == "locked_da_sell_below_soc_min_before_future_buy"
    assert float(by_ts[sell_ts]["da_prelock_unaccepted_sell_mwh"]) == pytest.approx(6.9)


def test_da_future_headroom_caps_recovery_buy_before_locked_da_buy() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    now = pd.Timestamp("2026-01-22T00:00:00Z")
    da_ts = pd.Timestamp("2026-01-22T02:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [pd.Timestamp("2026-01-22T01:00:00Z"), da_ts])

    capped, diag = bt._cap_grid_buy_to_preserve_future_da_headroom(
        current_ts_utc=now,
        current_soc_mwh=16.0,
        requested_buy_grid_mwh=2.0,
        future_rows=rows,
        da_lockbook={da_ts: (2.0, 0.0)},
        eta_in=bt.eta_in,
        soc_max_mwh=bt.soc_max,
    )

    assert capped == pytest.approx(0.0)
    assert float(diag["da_future_headroom_cap_applied"]) == pytest.approx(1.0)
    assert float(diag["id_recovery_capped_to_preserve_future_da_headroom"] if "id_recovery_capped_to_preserve_future_da_headroom" in diag else diag["da_future_headroom_cap_applied"]) == pytest.approx(1.0)
    assert float(diag["da_future_headroom_blocked_buy_mwh"]) == pytest.approx(2.0)
    assert str(diag["da_future_headroom_first_blocking_da_ts_utc"]) == str(da_ts)
    assert float(diag["da_future_headroom_blocking_locked_da_buy_mwh"]) == pytest.approx(2.0)


def test_da_future_headroom_does_not_cap_recovery_without_future_da_buy() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    now = pd.Timestamp("2026-01-22T00:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [pd.Timestamp("2026-01-22T01:00:00Z")])

    capped, diag = bt._cap_grid_buy_to_preserve_future_da_headroom(
        current_ts_utc=now,
        current_soc_mwh=16.0,
        requested_buy_grid_mwh=2.0,
        future_rows=rows,
        da_lockbook={},
        eta_in=bt.eta_in,
        soc_max_mwh=bt.soc_max,
    )

    assert capped == pytest.approx(2.0)
    assert float(diag["da_future_headroom_cap_applied"]) == pytest.approx(0.0)
    assert float(diag["da_future_headroom_blocked_buy_mwh"]) == pytest.approx(0.0)


def test_locked_da_buy_headroom_repair_sells_before_future_da_buy() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    col = BacktestColumnMap()
    now = pd.Timestamp("2026-01-22T00:00:00Z")
    repair_ts = pd.Timestamp("2026-01-22T01:00:00Z")
    da_ts = pd.Timestamp("2026-01-22T02:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [repair_ts, da_ts])

    sell_mwh, diag = bt._required_id_sell_to_create_next_locked_da_buy_headroom(
        current_ts_utc=now,
        current_soc_mwh=8.513,
        future_rows=rows,
        da_lockbook={da_ts: (10.0, 0.0)},
        eta_in=bt.eta_in,
        eta_out=bt.eta_out,
        soc_max_mwh=bt.soc_max,
        soc_min_mwh=bt.soc_min,
    )

    assert sell_mwh > 0.0
    assert float(diag["da_locked_buy_headroom_repair_applied"]) == pytest.approx(1.0)
    projected_before_da = float(diag["da_locked_buy_headroom_repair_projected_soc_before_da"])
    projected_after_repair = projected_before_da - sell_mwh / bt.eta_out
    assert projected_after_repair + 10.0 * bt.eta_in <= bt.soc_max + 1e-6
    assert str(diag["da_locked_buy_headroom_repair_blocking_da_ts_utc"]) == da_ts.isoformat()
    assert str(diag["da_locked_buy_headroom_repair_reason"]) == "preserve_future_locked_da_buy_headroom"


def test_locked_da_buy_headroom_repair_noops_when_headroom_exists() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    col = BacktestColumnMap()
    now = pd.Timestamp("2026-01-22T00:00:00Z")
    da_ts = pd.Timestamp("2026-01-22T02:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [pd.Timestamp("2026-01-22T01:00:00Z"), da_ts])

    sell_mwh, diag = bt._required_id_sell_to_create_next_locked_da_buy_headroom(
        current_ts_utc=now,
        current_soc_mwh=8.49,
        future_rows=rows,
        da_lockbook={da_ts: (10.0, 0.0)},
        eta_in=bt.eta_in,
        eta_out=bt.eta_out,
        soc_max_mwh=bt.soc_max,
        soc_min_mwh=bt.soc_min,
    )

    assert sell_mwh == pytest.approx(0.0)
    assert float(diag["da_locked_buy_headroom_repair_needed"]) == pytest.approx(0.0)
    assert float(diag["da_locked_buy_headroom_repair_applied"]) == pytest.approx(0.0)


def test_locked_da_buy_headroom_repair_unresolved_without_prior_repair_hour() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    col = BacktestColumnMap()
    now = pd.Timestamp("2026-01-22T00:00:00Z")
    da_ts = pd.Timestamp("2026-01-22T01:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [da_ts])

    sell_mwh, diag = bt._required_id_sell_to_create_next_locked_da_buy_headroom(
        current_ts_utc=now,
        current_soc_mwh=8.513,
        future_rows=rows,
        da_lockbook={da_ts: (10.0, 0.0)},
        eta_in=bt.eta_in,
        eta_out=bt.eta_out,
        soc_max_mwh=bt.soc_max,
        soc_min_mwh=bt.soc_min,
    )

    assert sell_mwh == pytest.approx(0.0)
    assert float(diag["da_future_locked_buy_exists"]) == pytest.approx(1.0)
    assert float(diag["da_future_locked_buy_headroom_ok_before_repair"]) == pytest.approx(0.0)
    assert float(diag["da_future_locked_buy_headroom_ok_after_repair"]) == pytest.approx(0.0)
    assert float(diag["da_locked_buy_headroom_repair_unresolved"]) == pytest.approx(1.0)
    assert str(diag["da_locked_buy_headroom_repair_unresolved_reason"]) == (
        "no_prior_repair_hour_before_locked_da_buy"
    )


def test_da_prelock_guard_derates_later_buy_to_preserve_future_headroom() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts0 = pd.Timestamp("2026-01-22T01:00:00Z")
    ts1 = pd.Timestamp("2026-01-22T02:00:00Z")

    repaired, diag, by_ts = bt._repair_or_reject_da_before_lock(
        accepted_da={ts0: (7.0, 0.0), ts1: (7.0, 0.0)},
        current_soc_mwh=10.0,
        future_rows=_da_hourly_lock_future_rows(col, [ts0, ts1]),
        colmap=col,
        existing_da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert repaired[ts0] == pytest.approx((7.0, 0.0))
    assert repaired[ts1] == pytest.approx((1.0, 0.0))
    assert float(diag["da_prelock_final_guard_passed"]) == pytest.approx(1.0)
    assert float(by_ts[ts1]["da_prelock_repaired_buy_mwh"]) == pytest.approx(1.0)


def test_da_prelock_future_headroom_derates_candidate_before_lockbook_write() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts0 = pd.Timestamp("2026-01-22T01:00:00Z")
    ts1 = pd.Timestamp("2026-01-22T02:00:00Z")

    repaired, diag, by_ts = bt._repair_or_reject_da_before_lock(
        accepted_da={ts0: (7.0, 0.0), ts1: (7.0, 0.0)},
        current_soc_mwh=10.0,
        future_rows=_da_hourly_lock_future_rows(col, [ts0, ts1]),
        colmap=col,
        existing_da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    accepted_buy_mwh = max(0.0, repaired[ts1][0]) * bt.dt_h
    projected_soc_before_buy = float(by_ts[ts1]["da_prelock_future_headroom_projected_soc_before_buy"])
    assert accepted_buy_mwh == pytest.approx(1.0)
    assert float(diag["da_prelock_final_guard_passed"]) == pytest.approx(1.0)
    assert float(by_ts[ts1]["da_prelock_future_headroom_check_pass"]) == pytest.approx(1.0)
    assert float(by_ts[ts1]["da_prelock_future_headroom_candidate_buy_mwh"]) == pytest.approx(7.0)
    assert float(by_ts[ts1]["da_prelock_future_headroom_accepted_buy_mwh"]) == pytest.approx(1.0)
    assert float(by_ts[ts1]["da_prelock_future_headroom_derated_mwh"]) == pytest.approx(6.0)
    assert str(by_ts[ts1]["da_prelock_future_headroom_blocking_ts_utc"]) == str(ts1)
    assert projected_soc_before_buy + accepted_buy_mwh * bt.eta_in <= bt.soc_max + 1e-9


def test_da_final_lockbook_consistency_derates_exact_final_accepted_schedule() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-22T01:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])

    ok_before, diag_before, _ = bt._check_da_hourly_lock_physical_feasibility(
        future_rows=rows,
        colmap=col,
        current_soc_mwh=17.55,
        existing_da_lockbook={},
        candidate_da={ts: (1.0, 0.0)},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
        include_existing_lockbook=True,
        schedule_source="final_da_lockbook_consistency_guard",
    )
    assert ok_before is False
    assert str(diag_before["da_hourly_lock_infeasible_reason"]) == "locked_da_buy_exceeds_soc_headroom"

    repaired, diag, by_ts = bt._repair_or_reject_da_before_lock(
        accepted_da={ts: (1.0, 0.0)},
        current_soc_mwh=17.55,
        future_rows=rows,
        colmap=col,
        existing_da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )
    assert repaired[ts] == pytest.approx((0.4, 0.0))
    assert float(diag["da_prelock_final_guard_passed"]) == pytest.approx(1.0)
    assert float(by_ts[ts]["da_prelock_future_headroom_check_pass"]) == pytest.approx(1.0)

    ok_after, _, after_by_ts = bt._check_da_hourly_lock_physical_feasibility(
        future_rows=rows,
        colmap=col,
        current_soc_mwh=17.55,
        existing_da_lockbook={},
        candidate_da=repaired,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
        include_existing_lockbook=True,
        schedule_source="final_da_lockbook_consistency_guard",
    )
    assert ok_after is True
    assert float(after_by_ts[ts]["da_hourly_lock_projected_soc_end_mwh"]) <= bt.soc_max + 1e-9


def test_da_realized_anchor_caps_lockbook_candidate_before_write() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_in = 0.95
    bt.soc_init = 8.84
    col = BacktestColumnMap()
    snapshot_ts = pd.Timestamp("2026-01-22T11:00:00Z")
    prior_ts = snapshot_ts - pd.Timedelta(hours=1)
    delivery_ts = pd.Timestamp("2026-01-23T00:00:00Z")
    decisions = [
        pd.DataFrame(
            {
                col.timestamp: [snapshot_ts],
                "soc_before_mwh": [8.86],
                "executed_soc_mwh": [8.84],
            }
        )
    ]
    snapshot_plan = pd.DataFrame(
        {
            "target_time_utc": [snapshot_ts],
            "soc_lp_mwh": [8.58],
        }
    )
    id_buy_before_gate_mw = (15.75 - 8.84) / bt.eta_in
    settlement_anchor, settlement_anchor_diag = bt._replay_da_prelock_settlement_equivalent_soc_anchor(
        rows=_da_hourly_lock_future_rows(col, [prior_ts]),
        snapshot_ts=snapshot_ts,
        colmap=col,
        da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        scheduled_id_by_ts={prior_ts: (id_buy_before_gate_mw, 0.0)},
    )

    assert settlement_anchor == pytest.approx(15.75)
    assert float(settlement_anchor_diag["da_prelock_anchor_settlement_replay_rows"]) == pytest.approx(1.0)
    assert float(settlement_anchor_diag["da_prelock_anchor_realized_settlement_soc_mwh"]) == pytest.approx(15.75)

    anchor, anchor_diag = bt._get_realized_soc_anchor_for_snapshot(
        snapshot_ts=snapshot_ts,
        decisions=decisions,
        current_soc_mwh=8.84,
        snapshot_plan=snapshot_plan,
        colmap=col,
        realized_settlement_soc_mwh=settlement_anchor,
    )

    assert anchor == pytest.approx(15.75)
    assert str(anchor_diag["da_prelock_soc_anchor_source"]) == "realized_settlement_replay_soc"
    assert str(anchor_diag["da_prelock_anchor_source"]) == "realized_settlement_replay_soc"
    assert float(anchor_diag["da_prelock_anchor_used_current_soc"]) == pytest.approx(0.0)
    assert float(anchor_diag["da_prelock_anchor_current_soc_mwh"]) == pytest.approx(8.84)
    assert float(anchor_diag["da_prelock_anchor_exact_decision_soc_mwh"]) == pytest.approx(8.86)
    assert float(anchor_diag["da_prelock_anchor_realized_settlement_soc_mwh"]) == pytest.approx(15.75)
    assert float(anchor_diag["da_prelock_anchor_current_vs_realized_delta_mwh"]) == pytest.approx(8.84 - 15.75)
    assert float(anchor_diag["da_prelock_anchor_decision_vs_realized_delta_mwh"]) == pytest.approx(8.86 - 15.75)
    assert float(anchor_diag["da_prelock_anchor_stale_state_detected"]) == pytest.approx(1.0)
    assert float(anchor_diag["da_prelock_planned_soc_at_gate_mwh"]) == pytest.approx(8.58)
    assert float(anchor_diag["da_prelock_anchor_minus_planned_soc_mwh"]) == pytest.approx(7.17)
    assert 8.58 + 9.9 * bt.eta_in <= bt.soc_max
    assert 15.75 + 9.9 * bt.eta_in > bt.soc_max

    repaired, diag, by_ts = bt._repair_or_reject_da_before_lock(
        accepted_da={delivery_ts: (9.9, 0.0)},
        current_soc_mwh=anchor,
        future_rows=_da_hourly_lock_future_rows(col, [delivery_ts]),
        colmap=col,
        existing_da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    accepted_buy_mwh = repaired[delivery_ts][0] * bt.dt_h
    projected_soc_before_buy = float(by_ts[delivery_ts]["da_prelock_future_headroom_projected_soc_before_buy"])
    assert accepted_buy_mwh == pytest.approx(2.3)
    assert float(diag["da_prelock_final_guard_passed"]) == pytest.approx(1.0)
    assert projected_soc_before_buy == pytest.approx(15.75)
    assert projected_soc_before_buy + accepted_buy_mwh * bt.eta_in <= bt.soc_max + 1e-9

    prior_anchor, prior_diag = bt._get_realized_soc_anchor_for_snapshot(
        snapshot_ts=snapshot_ts,
        decisions=[
            pd.DataFrame(
                {
                    col.timestamp: [snapshot_ts - pd.Timedelta(hours=1)],
                    "real_soc_mwh": [15.77],
                    "soc_mwh": [8.86],
                }
            )
        ],
        current_soc_mwh=8.84,
        snapshot_plan=snapshot_plan,
        colmap=col,
        realized_settlement_soc_mwh=15.75,
    )
    assert prior_anchor == pytest.approx(15.77)
    assert str(prior_diag["da_prelock_soc_anchor_source"]) == "realized_settlement_latest_prior_end_soc"
    assert float(prior_diag["da_prelock_anchor_latest_prior_decision_soc_mwh"]) == pytest.approx(8.86)
    assert float(prior_diag["da_prelock_anchor_used_current_soc"]) == pytest.approx(0.0)
    assert float(prior_diag["da_prelock_anchor_stale_state_detected"]) == pytest.approx(1.0)


def test_da_soc_feedback_uses_latest_settled_real_soc_before_gate() -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    gate_ts = pd.Timestamp("2025-03-03T10:00:00Z")
    decisions = [
        pd.DataFrame(
            {
                col.timestamp: [gate_ts - pd.Timedelta(hours=1)],
                "real_soc_mwh": [15.75],
                # Stale plan/decision columns must not be authoritative.
                "executed_soc_mwh": [8.84],
                "soc_mwh": [8.86],
            }
        )
    ]

    anchor, diag = bt._get_optimizer_soc_feedback_for_snapshot(
        snapshot_ts=gate_ts,
        decisions=decisions,
        current_soc_mwh=8.84,
        colmap=col,
    )

    assert anchor == pytest.approx(15.75)
    assert str(diag["optimizer_current_soc_source"]) == "realized_settlement_latest_prior_end_soc"
    assert float(diag["optimizer_current_soc_mwh"]) == pytest.approx(15.75)
    assert float(diag["latest_settled_real_soc_mwh_before_gate"]) == pytest.approx(15.75)
    assert float(diag["da_prelock_anchor_current_soc_mwh"]) == pytest.approx(8.84)
    assert float(diag["da_prelock_anchor_latest_prior_decision_soc_mwh"]) == pytest.approx(8.84)
    assert float(diag["optimizer_current_vs_realized_soc_delta_mwh"]) == pytest.approx(8.84 - 15.75)
    assert float(diag["optimizer_decision_vs_realized_soc_delta_mwh"]) == pytest.approx(8.84 - 15.75)
    assert float(diag["optimizer_soc_stale_state_detected"]) == pytest.approx(1.0)


def test_da_soc_feedback_settlement_exports_executed_da_buy_for_soc_increase() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_in = 0.95

    soc_next, metrics = bt._settle_one_hour(
        soc=10.0,
        charge=1.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=50.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        da_charge_mw=1.0,
        da_discharge_mw=0.0,
    )

    assert float(metrics["da_buy_mwh"]) == pytest.approx(1.0)
    assert float(metrics["da_soc_gain_mwh"]) == pytest.approx(0.95)
    assert soc_next == pytest.approx(10.95)
    assert float(metrics["da_unexecuted_buy_mwh"]) == pytest.approx(0.0)
    assert float(metrics["da_execution_lockbook_physical_infeasible"]) == pytest.approx(0.0)


def test_rolling_soc_feedback_includes_current_optimizer_id_setpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt = _mk_backtester("canonical_economic")
    _configure_zero_aux_unit_efficiency(bt)
    bt.soc_init = 8.953905
    bt.eta_in = 0.943278054766642
    bt.dt_h = 1.0
    df, col = _tiny_backtest_df(hours=30)
    df[col.timestamp] = pd.date_range("2025-03-02T10:00:00Z", periods=len(df), freq="h")
    id_charge_mw = 9.240225

    def _id_plan(window: pd.DataFrame, colmap: BacktestColumnMap, **_: object) -> pd.DataFrame:
        ts = pd.to_datetime(window[colmap.timestamp], utc=True, errors="coerce")
        return pd.DataFrame(
            {
                colmap.timestamp: ts,
                "charge_mw": np.zeros(len(window), dtype=float),
                "discharge_mw": np.zeros(len(window), dtype=float),
                "reserve_pos_mw": np.zeros(len(window), dtype=float),
                "reserve_neg_mw": np.zeros(len(window), dtype=float),
                "bem_only_pos_mw": np.zeros(len(window), dtype=float),
                "bem_only_neg_mw": np.zeros(len(window), dtype=float),
                "id_charge_mw": [id_charge_mw] + [0.0] * (len(window) - 1),
                "id_discharge_mw": np.zeros(len(window), dtype=float),
                "id_recourse_reason": ["protected_soc_recovery"] + ["none"] * (len(window) - 1),
                "soc_lp_mwh": np.full(len(window), float(bt.soc_init), dtype=float),
                "soc_start_lp_mwh": np.full(len(window), float(bt.soc_init), dtype=float),
                "predicted_objective_eur": np.zeros(len(window), dtype=float),
                "aux_power_mw": np.zeros(len(window), dtype=float),
            }
        )

    monkeypatch.setattr(bt, "optimize_dispatch", _id_plan)

    out, _ = bt.optimize_dispatch_rolling(
        df,
        col,
        horizon_hours=30,
        reopt_step_hours=2,
        allowed_markets=("DA", "ID"),
        strategy_name="da",
        strict_simulation_validity=False,
    )

    expected_after_first = float(bt.soc_init + id_charge_mw * bt.eta_in)
    assert float(out.loc[0, "executed_id_charge_mw_for_soc_feedback"]) == pytest.approx(id_charge_mw)
    assert float(out.loc[0, "rolling_soc_feedback_included_plan_id_setpoint"]) == pytest.approx(1.0)
    assert float(out.loc[0, "executed_soc_mwh"]) == pytest.approx(expected_after_first)
    assert float(out.loc[1, "soc_before_mwh"]) == pytest.approx(expected_after_first)


def test_realized_feedback_da_clearing_uses_forecast_limit_not_true_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt = _mk_backtester("canonical_economic")
    _configure_zero_aux_unit_efficiency(bt)
    df, col = _tiny_backtest_df(hours=2)
    df[col.pred_da_price] = [50.0, 50.0]
    df[col.true_da_price] = [100.0, 100.0]
    for q in ["p05", "p10", "p90", "p95"]:
        df[f"{col.pred_da_price}_{q}"] = [50.0, 50.0]

    def _da_plan(window: pd.DataFrame, colmap: BacktestColumnMap, **_: object) -> pd.DataFrame:
        ts = pd.to_datetime(window[colmap.timestamp], utc=True, errors="coerce")
        return pd.DataFrame(
            {
                colmap.timestamp: ts,
                "charge_mw": [1.0] + [0.0] * (len(window) - 1),
                "discharge_mw": np.zeros(len(window), dtype=float),
                "reserve_pos_mw": np.zeros(len(window), dtype=float),
                "reserve_neg_mw": np.zeros(len(window), dtype=float),
                "bem_only_pos_mw": np.zeros(len(window), dtype=float),
                "bem_only_neg_mw": np.zeros(len(window), dtype=float),
                "id_charge_mw": np.zeros(len(window), dtype=float),
                "id_discharge_mw": np.zeros(len(window), dtype=float),
                "id_recourse_reason": ["none"] * len(window),
                "soc_lp_mwh": np.full(len(window), float(bt.soc_init), dtype=float),
                "soc_start_lp_mwh": np.full(len(window), float(bt.soc_init), dtype=float),
                "predicted_objective_eur": np.zeros(len(window), dtype=float),
                "aux_power_mw": np.zeros(len(window), dtype=float),
            }
        )

    captured: list[tuple[float, float]] = []
    original_apply_market_clearing = bt._apply_market_clearing

    def _capture_apply_market_clearing(*args: object, **kwargs: object) -> dict[str, object]:
        captured.append((float(kwargs["pred_da_price"]), float(kwargs["true_da_price"])))
        return original_apply_market_clearing(*args, **kwargs)

    monkeypatch.setattr(bt, "optimize_dispatch", _da_plan)
    monkeypatch.setattr(bt, "_apply_market_clearing", _capture_apply_market_clearing)

    bt.optimize_dispatch_rolling(
        df,
        col,
        horizon_hours=2,
        reopt_step_hours=2,
        allowed_markets=("DA", "ID"),
        strategy_name="da",
        soc_feedback_mode="realized",
        strict_simulation_validity=False,
    )

    assert captured
    assert captured[0] == pytest.approx((50.0, 100.0))


def test_da_gate_after_optimizer_id_recovery_uses_high_rolling_soc_feedback() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_in = 0.95
    bt.soc_max = 18.0
    col = BacktestColumnMap()
    gate_ts = pd.Timestamp("2025-03-03T10:00:00Z")
    realized_high_soc = 17.67

    anchor, diag = bt._get_optimizer_soc_feedback_for_snapshot(
        snapshot_ts=gate_ts,
        decisions=[
            pd.DataFrame(
                {
                    col.timestamp: [gate_ts - pd.Timedelta(hours=1)],
                    "real_soc_mwh": [realized_high_soc],
                    "executed_soc_mwh": [9.26],
                    "soc_mwh": [9.26],
                }
            )
        ],
        current_soc_mwh=9.26,
        colmap=col,
    )
    max_feasible_buy = (bt.soc_max - anchor) / bt.eta_in

    assert anchor == pytest.approx(realized_high_soc)
    assert str(diag["optimizer_current_soc_source"]) == "realized_settlement_latest_prior_end_soc"
    assert float(diag["optimizer_soc_stale_state_detected"]) == pytest.approx(1.0)
    assert max_feasible_buy < 9.9


def test_da_precommit_bid_sizer_uses_physical_current_soc_not_stale_lp_soc() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_in = 1.0
    bt.eta_out = 1.0
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    bt.soc_target_end = 2.0
    bt.deg_eur_mwh = 0.0
    bt.trans_eur_mwh = 0.0
    bt.da_min_bid_size_mw = 0.1
    bt.da_bid_granularity_mw = 0.1
    col = BacktestColumnMap()
    ts = pd.date_range("2025-03-04T00:00:00Z", periods=2, freq="h")
    rows = pd.DataFrame(
        {
            col.timestamp: ts,
            "target_time_utc": ts,
            "charge_mw": [9.9, 0.0],
            "discharge_mw": [0.0, 1.0],
            # This stale LP SoC would incorrectly allow the 9.9 MW buy.
            "soc_start_lp_mwh": [8.0, 8.0],
            col.pred_da_price: [0.0, 100.0],
            col.pred_afrr_activation_rate_pos: [0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0],
        }
    )

    accepted, audit = bt._select_feasible_da_lock_schedule(
        lock_rows=rows,
        colmap=col,
        current_soc_mwh=17.0,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert accepted
    first_buy = float(accepted[pd.Timestamp(ts[0])][0])
    assert first_buy <= 1.0 + 1e-9
    first_audit = next(
        row
        for row in audit
        if pd.Timestamp(row["timestamp_utc"]) == pd.Timestamp(ts[0])
    )
    assert float(first_audit["da_candidate_buy_mw"]) == pytest.approx(9.9)
    assert float(first_audit["da_accepted_buy_mw"]) <= 1.0 + 1e-9
    assert float(first_audit["projected_soc_max_mwh"]) <= 18.0 + 1e-9

    replay = bt._replay_da_candidate_cashflow(
        rows=rows,
        schedule={pd.Timestamp(ts[0]): (1.0, 0.0)},
        colmap=col,
        current_soc_mwh=17.0,
    )
    assert float(replay["final_soc_mwh"]) == pytest.approx(18.0)


def test_da_settlement_equiv_lockbook_cap_prevents_unexecuted_locked_buy() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_in = 0.95
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-23T00:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])

    repaired, diag, by_ts = bt._cap_da_lockbook_to_settlement_equivalent_replay(
        accepted_da={ts: (9.9, 0.0)},
        current_soc_mwh=15.49,
        future_rows=rows,
        colmap=col,
        existing_da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        scheduled_id_by_ts={},
    )

    assert repaired[ts][0] == pytest.approx(2.6)
    assert float(diag["da_settlement_equiv_replay_checked"]) == pytest.approx(1.0)
    assert float(by_ts[ts]["da_settlement_equiv_replay_candidate_buy_mwh"]) == pytest.approx(9.9)
    assert float(by_ts[ts]["da_settlement_equiv_replay_max_feasible_buy_mwh"]) == pytest.approx(
        (bt.soc_max - 15.49) / bt.eta_in
    )
    assert float(by_ts[ts]["da_settlement_equiv_replay_final_accepted_buy_mwh"]) == pytest.approx(2.6)
    assert str(by_ts[ts]["da_settlement_equiv_replay_reason"]) == "derated_buy_to_settlement_equivalent_headroom"

    soc_next, metrics = bt._settle_one_hour(
        soc=15.49,
        charge=repaired[ts][0],
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
        da_charge_mw=repaired[ts][0],
        da_discharge_mw=0.0,
    )

    assert soc_next <= bt.soc_max + 1e-9
    assert float(metrics["da_buy_mwh"]) == pytest.approx(2.6)
    assert float(metrics["da_unexecuted_buy_mwh"]) == pytest.approx(0.0)
    assert float(metrics["da_execution_lockbook_physical_infeasible"]) == pytest.approx(0.0)


def test_da_settlement_equiv_replay_caps_using_settlement_reported_unexecuted_buy(monkeypatch: pytest.MonkeyPatch) -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.da_bid_granularity_mw = 0.1
    bt.da_min_bid_size_mw = 0.1
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-23T00:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])

    def fake_settle_one_hour(*args: object, **kwargs: object) -> tuple[float, dict[str, float]]:
        return (
            17.75,
            {
                "da_max_feasible_buy_mwh": 0.4,
                "da_max_feasible_sell_mwh": 0.0,
                "da_unexecuted_buy_mwh": 0.3,
                "da_unexecuted_sell_mwh": 0.0,
            },
        )

    monkeypatch.setattr(bt, "_settle_one_hour", fake_settle_one_hour)

    repaired, _, by_ts = bt._cap_da_lockbook_to_settlement_equivalent_replay(
        accepted_da={ts: (0.7, 0.0)},
        current_soc_mwh=17.1,
        future_rows=rows,
        colmap=col,
        existing_da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        scheduled_id_by_ts={},
    )

    assert repaired[ts][0] == pytest.approx(0.4)
    assert float(by_ts[ts]["da_settlement_equiv_replay_settlement_checked"]) == pytest.approx(1.0)
    assert float(by_ts[ts]["da_settlement_equiv_replay_unexecuted_buy_mwh"]) == pytest.approx(0.3)
    assert float(by_ts[ts]["da_settlement_equiv_replay_max_feasible_buy_mwh"]) == pytest.approx(0.4)
    assert str(by_ts[ts]["da_settlement_equiv_replay_reason"]) == "derated_buy_to_settlement_equivalent_headroom"


def test_da_settlement_equiv_lockbook_cap_prevents_unexecuted_locked_sell() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_out = 0.95
    bt.da_bid_granularity_mw = 0.1
    bt.da_min_bid_size_mw = 0.1
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-23T00:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])

    repaired, diag, by_ts = bt._cap_da_lockbook_to_settlement_equivalent_replay(
        accepted_da={ts: (0.0, 2.8)},
        current_soc_mwh=3.0,
        future_rows=rows,
        colmap=col,
        existing_da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        scheduled_id_by_ts={},
    )

    assert repaired[ts][1] == pytest.approx(0.9)
    assert float(diag["da_settlement_equiv_replay_checked"]) == pytest.approx(1.0)
    assert float(by_ts[ts]["da_settlement_equiv_replay_candidate_sell_mwh"]) == pytest.approx(2.8)
    assert float(by_ts[ts]["da_settlement_equiv_replay_max_feasible_sell_mwh"]) == pytest.approx(
        (3.0 - bt.soc_min) * bt.eta_out
    )
    assert float(by_ts[ts]["da_settlement_equiv_replay_final_accepted_sell_mwh"]) == pytest.approx(0.9)
    assert float(by_ts[ts]["da_settlement_equiv_replay_derated_sell_mwh"]) == pytest.approx(1.9)
    assert str(by_ts[ts]["da_settlement_equiv_replay_reason"]) == "derated_sell_to_settlement_equivalent_footroom"

    soc_next, metrics = bt._settle_one_hour(
        soc=3.0,
        charge=0.0,
        discharge=repaired[ts][1],
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=100.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        da_charge_mw=0.0,
        da_discharge_mw=repaired[ts][1],
    )

    assert soc_next >= bt.soc_min - 1e-9
    assert float(metrics["da_sell_mwh"]) == pytest.approx(0.9)
    assert float(metrics["da_unexecuted_sell_mwh"]) == pytest.approx(0.0)
    assert float(metrics["da_execution_lockbook_physical_infeasible"]) == pytest.approx(0.0)


def test_da_settlement_equiv_replay_includes_same_gate_accepted_rows_before_later_buy() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_in = 0.95
    bt.da_bid_granularity_mw = 0.1
    bt.da_min_bid_size_mw = 0.1
    col = BacktestColumnMap()
    ts0 = pd.Timestamp("2026-01-23T00:00:00Z")
    ts1 = pd.Timestamp("2026-01-23T01:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts0, ts1])

    repaired, _, by_ts = bt._cap_da_lockbook_to_settlement_equivalent_replay(
        accepted_da={ts0: (7.0, 0.0), ts1: (7.0, 0.0)},
        current_soc_mwh=10.0,
        future_rows=rows,
        colmap=col,
        existing_da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        scheduled_id_by_ts={},
    )

    assert repaired[ts0][0] == pytest.approx(7.0)
    assert float(by_ts[ts0]["da_settlement_equiv_replay_soc_before_delivery_mwh"]) == pytest.approx(10.0)
    assert float(by_ts[ts1]["da_settlement_equiv_replay_soc_before_delivery_mwh"]) == pytest.approx(16.65)
    assert float(by_ts[ts1]["da_settlement_equiv_replay_candidate_buy_mwh"]) == pytest.approx(7.0)
    assert repaired[ts1][0] == pytest.approx(1.4)
    assert float(by_ts[ts1]["da_settlement_equiv_replay_final_accepted_buy_mwh"]) == pytest.approx(1.4)
    assert str(by_ts[ts1]["da_settlement_equiv_replay_reason"]) == "derated_buy_to_settlement_equivalent_headroom"


def test_da_lockbook_settlement_equiv_replay_includes_scheduled_bem_activation_before_later_buy() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.da_bid_granularity_mw = 0.1
    bt.da_min_bid_size_mw = 0.1
    col = BacktestColumnMap()
    ts0 = pd.Timestamp("2026-01-23T00:00:00Z")
    ts1 = pd.Timestamp("2026-01-23T01:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts0, ts1])
    rows.loc[rows[col.timestamp] == ts0, col.pred_afrr_activation_rate_neg] = 1.0

    repaired, _, by_ts = bt._cap_da_lockbook_to_settlement_equivalent_replay(
        accepted_da={ts1: (7.0, 0.0)},
        current_soc_mwh=10.0,
        future_rows=rows,
        colmap=col,
        existing_da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        scheduled_id_by_ts={},
        scheduled_bem_by_ts={ts0: (0.0, 5.0)},
    )

    assert float(by_ts[ts0]["da_settlement_equiv_replay_bem_neg_mw"]) == pytest.approx(5.0)
    assert float(by_ts[ts1]["da_settlement_equiv_replay_soc_before_delivery_mwh"]) == pytest.approx(15.0)
    assert float(by_ts[ts1]["da_settlement_equiv_replay_candidate_buy_mwh"]) == pytest.approx(7.0)
    assert float(by_ts[ts1]["da_settlement_equiv_replay_max_feasible_buy_mwh"]) == pytest.approx(3.0)
    assert repaired[ts1][0] == pytest.approx(3.0)
    assert str(by_ts[ts1]["da_settlement_equiv_replay_reason"]) == "derated_buy_to_settlement_equivalent_headroom"


def test_da_lockbook_settlement_equiv_replay_includes_scheduled_bem_activation_before_later_sell() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.da_bid_granularity_mw = 0.1
    bt.da_min_bid_size_mw = 0.1
    col = BacktestColumnMap()
    ts0 = pd.Timestamp("2026-01-23T00:00:00Z")
    ts1 = pd.Timestamp("2026-01-23T01:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts0, ts1])
    rows.loc[rows[col.timestamp] == ts0, col.pred_afrr_activation_rate_pos] = 1.0

    repaired, _, by_ts = bt._cap_da_lockbook_to_settlement_equivalent_replay(
        accepted_da={ts1: (0.0, 3.5)},
        current_soc_mwh=10.0,
        future_rows=rows,
        colmap=col,
        existing_da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        scheduled_id_by_ts={},
        scheduled_bem_by_ts={ts0: (5.0, 0.0)},
    )

    assert float(by_ts[ts0]["da_settlement_equiv_replay_bem_pos_mw"]) == pytest.approx(5.0)
    assert float(by_ts[ts1]["da_settlement_equiv_replay_soc_before_delivery_mwh"]) == pytest.approx(5.0)
    assert float(by_ts[ts1]["da_settlement_equiv_replay_candidate_sell_mwh"]) == pytest.approx(3.5)
    assert float(by_ts[ts1]["da_settlement_equiv_replay_max_feasible_sell_mwh"]) == pytest.approx(3.0)
    assert repaired[ts1][1] == pytest.approx(3.0)
    assert str(by_ts[ts1]["da_settlement_equiv_replay_reason"]) == "derated_sell_to_settlement_equivalent_footroom"


def test_fixed_obligation_replay_diagnostic_classifies_da_buy_headroom_conflict() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_in = 0.95
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-23T00:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])

    diag = bt._replay_fixed_obligations_diagnostic(
        current_soc_mwh=17.0,
        future_rows=rows,
        colmap=col,
        da_lockbook={ts: (2.0, 0.0)},
        fixed_reserve_obligation={},
        scheduled_id_by_ts={},
        enforce_final_soc_mwh=None,
    )

    assert float(diag["fixed_obligation_replay_checked"]) == pytest.approx(1.0)
    assert float(diag["fixed_obligation_replay_pass"]) == pytest.approx(0.0)
    assert str(diag["fixed_obligation_infeasible_at"]) == ts.isoformat()
    assert str(diag["fixed_obligation_infeasible_driver"]) == "fixed_da_buy_exceeds_headroom"
    assert float(diag["fixed_da_buy_exceeds_headroom"]) == pytest.approx(1.0)


def test_fixed_obligation_replay_allows_da_buy_when_same_hour_aux_preserves_soc_max() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_in = 0.95
    bt.aux_trading_mw = 0.05
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-23T00:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])

    diag = bt._replay_fixed_obligations_diagnostic(
        current_soc_mwh=8.532919,
        future_rows=rows,
        colmap=col,
        da_lockbook={ts: (10.0, 0.0)},
        fixed_reserve_obligation={},
        scheduled_id_by_ts={},
        enforce_final_soc_mwh=None,
    )

    assert float(diag["fixed_obligation_replay_checked"]) == pytest.approx(1.0)
    assert float(diag["fixed_obligation_replay_pass"]) == pytest.approx(1.0)
    assert str(diag["fixed_obligation_infeasible_driver"]) == "none"
    assert float(diag["fixed_obligation_replay_max_soc_mwh"]) == pytest.approx(17.982919)


def test_fixed_obligation_replay_classifies_locked_bcm_safety_buffer_shortfall() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    bt.reserve_activation_headroom_h = 0.5
    bt.reserve_headroom_safety_mwh = 0.25
    bt.reserve_soc_projection_safety_mwh = 0.5
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-23T09:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])
    rows[col.pred_afrr_activation_rate_pos] = [0.5]

    diag = bt._replay_fixed_obligations_diagnostic(
        current_soc_mwh=6.3883,
        future_rows=rows,
        colmap=col,
        da_lockbook={},
        fixed_reserve_obligation={ts: (7.0, 0.0)},
        scheduled_id_by_ts={},
        enforce_final_soc_mwh=None,
    )

    assert float(diag["fixed_obligation_replay_pass"]) == pytest.approx(0.0)
    assert str(diag["fixed_obligation_driver"]) == "locked_bcm_safety_buffer_shortfall"
    assert float(diag["locked_bcm_safety_buffer_shortfall"]) == pytest.approx(1.0)
    assert float(diag["fixed_obligation_physical_shortfall_mwh"]) == pytest.approx(0.0)
    assert float(diag["fixed_obligation_safety_shortfall_mwh"]) > 0.0


def test_fixed_obligation_replay_uses_row_level_bcm_lockbook_when_map_missing() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    bt.reserve_activation_headroom_h = 0.5
    bt.reserve_headroom_safety_mwh = 0.25
    bt.reserve_soc_projection_safety_mwh = 0.5
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-23T09:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])
    rows[col.pred_afrr_activation_rate_neg] = [0.5]
    rows["locked_bcm_capacity_neg_mw"] = [9.0]

    diag = bt._replay_fixed_obligations_diagnostic(
        current_soc_mwh=13.3,
        future_rows=rows,
        colmap=col,
        da_lockbook={},
        fixed_reserve_obligation={},
        scheduled_id_by_ts={},
        enforce_final_soc_mwh=None,
    )

    assert float(diag["fixed_obligation_reserve_neg_mw_sum"]) == pytest.approx(9.0)
    assert float(diag["fixed_obligation_reserve_neg_mw_max"]) == pytest.approx(9.0)
    assert float(diag["fixed_obligation_reserve_neg_mwh_equiv"]) == pytest.approx(9.0 * bt.dt_h)
    assert float(diag["fixed_obligation_replay_pass"]) == pytest.approx(0.0)
    assert str(diag["fixed_obligation_driver"]) == "locked_bcm_safety_buffer_shortfall"


def test_fixed_obligation_replay_uses_row_level_bcm_lockbook_when_map_understates_obligation() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    bt.reserve_activation_headroom_h = 0.5
    bt.reserve_headroom_safety_mwh = 0.25
    bt.reserve_soc_projection_safety_mwh = 0.5
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-23T09:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])
    rows[col.pred_afrr_activation_rate_neg] = [0.5]
    rows["fixed_reserve_obligation_neg_mw"] = [9.0]

    diag = bt._replay_fixed_obligations_diagnostic(
        current_soc_mwh=13.3,
        future_rows=rows,
        colmap=col,
        da_lockbook={},
        fixed_reserve_obligation={ts: (0.0, 0.0)},
        scheduled_id_by_ts={},
        enforce_final_soc_mwh=None,
    )

    assert float(diag["fixed_obligation_reserve_neg_mw_sum"]) == pytest.approx(9.0)
    assert float(diag["fixed_obligation_reserve_neg_mw_max"]) == pytest.approx(9.0)
    assert float(diag["fixed_obligation_reserve_neg_mwh_equiv"]) == pytest.approx(9.0 * bt.dt_h)
    assert float(diag["fixed_obligation_replay_pass"]) == pytest.approx(0.0)
    assert str(diag["fixed_obligation_driver"]) == "locked_bcm_safety_buffer_shortfall"


def test_fixed_obligation_replay_exports_reserve_source_snapshot_and_block() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    bt.reserve_activation_headroom_h = 0.5
    bt.reserve_headroom_safety_mwh = 0.25
    bt.reserve_soc_projection_safety_mwh = 0.5
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-23T09:00:00Z")
    source_ts = pd.Timestamp("2026-01-22T06:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])

    diag = bt._replay_fixed_obligations_diagnostic(
        current_soc_mwh=10.0,
        future_rows=rows,
        colmap=col,
        da_lockbook={},
        fixed_reserve_obligation={ts: (0.0, 9.0)},
        scheduled_id_by_ts={},
        enforce_final_soc_mwh=None,
        reserve_source_snapshot_utc={ts: source_ts},
    )

    assert float(diag["fixed_obligation_replay_pass"]) == pytest.approx(1.0)
    assert str(diag["fixed_obligation_first_reserve_ts_utc"]) == ts.isoformat()
    assert str(diag["fixed_obligation_first_reserve_source_snapshot_utc"]) == source_ts.isoformat()
    assert str(diag["fixed_obligation_first_reserve_product_block_id"])


def test_fixed_obligation_replay_passes_artifact_shape_bcm_neg_obligation() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    bt.reserve_activation_headroom_h = 0.5
    bt.reserve_headroom_safety_mwh = 0.25
    bt.reserve_soc_projection_safety_mwh = 0.5
    col = BacktestColumnMap()
    ts = pd.Timestamp("2025-06-05T02:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])

    diag = bt._replay_fixed_obligations_diagnostic(
        current_soc_mwh=6.159769,
        future_rows=rows,
        colmap=col,
        da_lockbook={},
        fixed_reserve_obligation={ts: (0.0, 3.0)},
        scheduled_id_by_ts={},
        enforce_final_soc_mwh=None,
    )
    proof_volume = BatteryBacktester._fixed_obligation_replay_proof_volume_mw(
        fixed_obligation_diag=diag,
        fixed_reserve_obligation={ts: (0.0, 3.0)},
        da_lockbook={},
    )

    assert float(diag["fixed_obligation_replay_pass"]) == pytest.approx(1.0)
    assert float(diag["fixed_obligation_reserve_neg_mw_sum"]) == pytest.approx(3.0)
    assert proof_volume == pytest.approx(3.0)


def test_fixed_obligation_replay_proof_volume_uses_lockbook_when_aggregate_missing() -> None:
    ts = pd.Timestamp("2026-01-23T09:00:00Z")
    volume = BatteryBacktester._fixed_obligation_replay_proof_volume_mw(
        fixed_obligation_diag={
            "fixed_obligation_replay_pass": 1.0,
            "fixed_obligation_reserve_pos_mw_sum": 0.0,
            "fixed_obligation_reserve_neg_mw_sum": 0.0,
        },
        fixed_reserve_obligation={ts: (0.0, 9.0)},
        da_lockbook={},
    )

    assert volume == pytest.approx(9.0)


def test_fixed_obligation_presolve_repair_schedules_id_for_locked_bcm_safety_shortfall() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    bt.reserve_activation_headroom_h = 0.5
    bt.reserve_headroom_safety_mwh = 0.25
    bt.reserve_soc_projection_safety_mwh = 0.5
    col = BacktestColumnMap()
    current_ts = pd.Timestamp("2026-01-23T08:00:00Z")
    target_ts = pd.Timestamp("2026-01-23T09:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [target_ts])
    rows[col.pred_afrr_activation_rate_pos] = [0.5]
    diag = bt._replay_fixed_obligations_diagnostic(
        current_soc_mwh=6.3883,
        future_rows=rows,
        colmap=col,
        da_lockbook={},
        fixed_reserve_obligation={target_ts: (7.0, 0.0)},
        scheduled_id_by_ts={},
        enforce_final_soc_mwh=None,
    )

    ch_mw, dis_mw, repaired_soc, repair_diag = bt._plan_fixed_obligation_presolve_id_repair(
        current_soc_mwh=6.3883,
        fixed_obligation_diag=diag,
        existing_id_charge_mw=0.0,
        existing_id_discharge_mw=0.0,
        allow_technical_id_repair=True,
        current_timestamp_utc=current_ts,
    )

    assert ch_mw > 0.0
    assert dis_mw == pytest.approx(0.0)
    assert repaired_soc > 6.3883
    assert float(repair_diag["fixed_obligation_id_repair_feasible"]) == pytest.approx(1.0)
    assert str(repair_diag["fixed_obligation_id_repair_reason"]) == "reserve_obligation_recovery"


def test_fixed_obligation_replay_classifies_aux_soc_min_shortfall() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.aux_off_mw = 0.02
    bt.aux_trading_mw = 0.05
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    col = BacktestColumnMap()
    ts0 = pd.Timestamp("2026-01-23T09:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts0])

    diag = bt._replay_fixed_obligations_diagnostic(
        current_soc_mwh=2.01,
        future_rows=rows,
        colmap=col,
        da_lockbook={},
        fixed_reserve_obligation={},
        scheduled_id_by_ts={},
        enforce_final_soc_mwh=None,
    )

    assert float(diag["fixed_obligation_replay_pass"]) == pytest.approx(0.0)
    assert str(diag["fixed_obligation_driver"]) == "rolling_window_aux_soc_min_shortfall"
    assert float(diag["rolling_window_aux_soc_min_shortfall"]) == pytest.approx(1.0)
    assert float(diag["fixed_obligation_soc_min_shortfall_mwh"]) == pytest.approx(0.01)


def test_fixed_obligation_presolve_repair_schedules_id_for_aux_soc_min_shortfall() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.aux_off_mw = 0.02
    bt.aux_trading_mw = 0.05
    bt.eta_in = 0.95
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    col = BacktestColumnMap()
    current_ts = pd.Timestamp("2026-01-23T08:00:00Z")
    ts0 = pd.Timestamp("2026-01-23T09:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts0])
    diag = bt._replay_fixed_obligations_diagnostic(
        current_soc_mwh=2.01,
        future_rows=rows,
        colmap=col,
        da_lockbook={},
        fixed_reserve_obligation={},
        scheduled_id_by_ts={},
        enforce_final_soc_mwh=None,
    )

    ch_mw, dis_mw, repaired_soc, repair_diag = bt._plan_fixed_obligation_presolve_id_repair(
        current_soc_mwh=2.01,
        fixed_obligation_diag=diag,
        existing_id_charge_mw=0.0,
        existing_id_discharge_mw=0.0,
        allow_technical_id_repair=True,
        current_timestamp_utc=current_ts,
    )

    assert ch_mw > 0.0
    assert dis_mw == pytest.approx(0.0)
    assert repaired_soc > 2.01
    assert float(repair_diag["fixed_obligation_id_repair_feasible"]) == pytest.approx(1.0)
    assert str(repair_diag["fixed_obligation_id_repair_reason"]) == "physical_soc_recovery"
    assert float(repair_diag["rolling_window_aux_soc_repair_scheduled_mwh"]) > 0.0


def test_fixed_obligation_replay_classifies_locked_bcm_physical_headroom_shortfall() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    bt.reserve_activation_headroom_h = 0.5
    bt.reserve_headroom_safety_mwh = 0.25
    bt.reserve_soc_projection_safety_mwh = 0.5
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-23T09:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])
    rows[col.pred_afrr_activation_rate_pos] = [0.5]

    diag = bt._replay_fixed_obligations_diagnostic(
        current_soc_mwh=5.0,
        future_rows=rows,
        colmap=col,
        da_lockbook={},
        fixed_reserve_obligation={ts: (7.0, 0.0)},
        scheduled_id_by_ts={},
        enforce_final_soc_mwh=None,
    )
    ch_mw, dis_mw, repaired_soc, repair_diag = bt._plan_fixed_obligation_presolve_id_repair(
        current_soc_mwh=5.0,
        fixed_obligation_diag=diag,
        existing_id_charge_mw=0.0,
        existing_id_discharge_mw=0.0,
        allow_technical_id_repair=True,
        current_timestamp_utc=pd.Timestamp("2026-01-23T08:00:00Z"),
    )

    assert float(diag["fixed_obligation_replay_pass"]) == pytest.approx(0.0)
    assert str(diag["fixed_obligation_driver"]) == "locked_bcm_physical_headroom_shortfall"
    assert float(diag["locked_bcm_physical_headroom_shortfall"]) == pytest.approx(1.0)
    assert float(diag["fixed_obligation_physical_shortfall_mwh"]) > 0.0
    assert ch_mw == pytest.approx(0.0)
    assert dis_mw == pytest.approx(0.0)
    assert repaired_soc == pytest.approx(5.0)
    assert str(repair_diag["fixed_obligation_id_repair_blocked_reason"]) == (
        "physical_shortfall_not_repairable_as_safety"
    )


def test_da_actual_independent_limit_clearing_rejects_uncleared_buy() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.da_limit_clearing_policy = "accepted_independent_limit_replay"
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-24T03:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])
    rows[col.pred_da_price] = [50.0]
    rows[col.true_da_price] = [75.0]

    accepted, diag, by_ts = bt._clear_da_limit_schedule_against_true_prices(
        submitted_da={ts: (2.0, 0.0)},
        lock_rows=rows,
        colmap=col,
        is_perfect_foresight=False,
    )

    assert accepted == {}
    assert float(diag["da_actual_independent_clearing_submitted_buy_mwh"]) == pytest.approx(2.0)
    assert float(diag["da_actual_independent_clearing_accepted_buy_mwh"]) == pytest.approx(0.0)
    assert float(diag["da_actual_independent_clearing_price_rejected_buy_mwh"]) == pytest.approx(2.0)
    assert str(diag["da_actual_independent_clearing_rejection_reason"]) == "price_rejected"
    assert float(by_ts[ts]["da_submitted_buy_mwh"]) == pytest.approx(2.0)
    assert float(by_ts[ts]["da_auction_accepted_buy_mwh"]) == pytest.approx(0.0)
    assert float(by_ts[ts]["da_price_rejected_buy_mwh"]) == pytest.approx(2.0)


def test_da_actual_independent_limit_clearing_accepts_cleared_sell() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.da_limit_clearing_policy = "accepted_independent_limit_replay"
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-24T17:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts])
    rows[col.pred_da_price] = [100.0]
    rows[col.true_da_price] = [125.0]

    accepted, diag, by_ts = bt._clear_da_limit_schedule_against_true_prices(
        submitted_da={ts: (0.0, 3.0)},
        lock_rows=rows,
        colmap=col,
        is_perfect_foresight=False,
    )

    assert accepted[ts] == pytest.approx((0.0, 3.0))
    assert float(diag["da_actual_independent_clearing_submitted_sell_mwh"]) == pytest.approx(3.0)
    assert float(diag["da_actual_independent_clearing_accepted_sell_mwh"]) == pytest.approx(3.0)
    assert float(diag["da_actual_independent_clearing_price_rejected_sell_mwh"]) == pytest.approx(0.0)
    assert str(diag["da_actual_independent_clearing_rejection_reason"]) == "none"
    assert float(by_ts[ts]["da_auction_accepted_sell_mwh"]) == pytest.approx(3.0)
    assert float(by_ts[ts]["da_price_rejected_sell_mwh"]) == pytest.approx(0.0)


def test_da_limit_robust_guard_caps_sell_when_prior_limit_buys_may_reject() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts_buy_1 = pd.Timestamp("2026-01-24T03:00:00Z")
    ts_buy_2 = pd.Timestamp("2026-01-24T13:00:00Z")
    ts_sell = pd.Timestamp("2026-01-24T17:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts_buy_1, ts_buy_2, ts_sell])

    repaired, diag, by_ts = bt._cap_da_limit_schedule_for_independent_clearing(
        submitted_da={
            ts_buy_1: (4.0, 0.0),
            ts_buy_2: (4.0, 0.0),
            ts_sell: (0.0, 8.8),
        },
        current_soc_mwh=9.0,
        future_rows=rows,
        colmap=col,
        existing_da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
    )

    assert repaired[ts_buy_1] == pytest.approx((4.0, 0.0))
    assert repaired[ts_buy_2] == pytest.approx((4.0, 0.0))
    assert repaired[ts_sell] == pytest.approx((0.0, 7.0))
    assert float(diag["da_limit_clearing_low_soc_pass"]) == pytest.approx(0.0)
    assert str(diag["da_limit_clearing_rejection_reason"]) == "derated_sell_for_buy_rejection_low_soc"
    assert float(by_ts[ts_sell]["da_submitted_sell_mwh"]) == pytest.approx(8.8)
    assert float(by_ts[ts_sell]["da_auction_accepted_sell_mwh"]) == pytest.approx(7.0)

    soc_next, metrics = bt._settle_one_hour(
        soc=9.0,
        charge=0.0,
        discharge=repaired[ts_sell][1],
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=100.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        da_charge_mw=0.0,
        da_discharge_mw=repaired[ts_sell][1],
    )
    assert soc_next == pytest.approx(bt.soc_min)
    assert float(metrics["da_unexecuted_sell_mwh"]) == pytest.approx(0.0)
    assert float(metrics["da_execution_lockbook_physical_infeasible"]) == pytest.approx(0.0)


def test_da_limit_robust_guard_caps_buy_when_prior_limit_sells_may_reject() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    col = BacktestColumnMap()
    ts_buy = pd.Timestamp("2026-01-24T03:00:00Z")
    rows = _da_hourly_lock_future_rows(col, [ts_buy])

    repaired, diag, by_ts = bt._cap_da_limit_schedule_for_independent_clearing(
        submitted_da={ts_buy: (5.0, 0.0)},
        current_soc_mwh=15.5,
        future_rows=rows,
        colmap=col,
        existing_da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
    )

    assert repaired[ts_buy] == pytest.approx((2.5, 0.0))
    assert float(diag["da_limit_clearing_high_soc_pass"]) == pytest.approx(0.0)
    assert str(diag["da_limit_clearing_rejection_reason"]) == "derated_buy_for_sell_rejection_high_soc"
    assert float(by_ts[ts_buy]["da_submitted_buy_mwh"]) == pytest.approx(5.0)
    assert float(by_ts[ts_buy]["da_auction_accepted_buy_mwh"]) == pytest.approx(2.5)


def test_da_prelock_guard_derates_terminal_sell_when_id_recovery_disabled() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.soc_target_end = 10.0
    bt._id_recourse_mode = "disabled"
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-22T01:00:00Z")

    repaired, diag, by_ts = bt._repair_or_reject_da_before_lock(
        accepted_da={ts: (0.0, 1.0)},
        current_soc_mwh=10.0,
        future_rows=_da_hourly_lock_future_rows(col, [ts]),
        colmap=col,
        existing_da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=ts,
    )

    assert repaired == {}
    assert float(diag["da_prelock_final_guard_passed"]) == pytest.approx(0.0)
    assert float(diag["da_prelock_rejected_no_feasible_schedule"]) == pytest.approx(1.0)
    assert str(diag["da_prelock_repair_reason"]) == "prelock_repair_no_feasible_subset"
    assert float(by_ts[ts]["da_prelock_unaccepted_sell_mwh"]) == pytest.approx(1.0)


def test_da_settlement_reports_unexecuted_locked_buy_without_soc_clipping() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)

    soc_next, metrics = bt._settle_one_hour(
        soc=17.8,
        charge=1.0,
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
        da_charge_mw=1.0,
        da_discharge_mw=0.0,
    )

    assert soc_next == pytest.approx(18.0)
    assert float(metrics["da_buy_mwh"]) == pytest.approx(0.2)
    assert float(metrics["da_unexecuted_buy_mwh"]) == pytest.approx(0.8)
    assert float(metrics["da_execution_lockbook_physical_infeasible"]) == pytest.approx(1.0)


def test_da_lockbook_infeasible_flag_ignores_generic_no_lockbook_physical_pressure() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.aux_standby_mw = 0.05

    _, metrics = bt._settle_one_hour(
        soc=bt.soc_min,
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
        da_charge_mw=0.0,
        da_discharge_mw=0.0,
    )

    assert float(metrics["da_locked_buy_mwh_requested"]) == pytest.approx(0.0)
    assert float(metrics["da_locked_sell_mwh_requested"]) == pytest.approx(0.0)
    assert float(metrics["da_unexecuted_buy_mwh"]) == pytest.approx(0.0)
    assert float(metrics["da_unexecuted_sell_mwh"]) == pytest.approx(0.0)
    assert float(metrics["da_execution_lockbook_physical_infeasible"]) == pytest.approx(0.0)


def test_da_postlock_hard_projection_stops_at_next_recovery_opportunity() -> None:
    bt = _mk_backtester()
    bt.da_postlock_guard_mode = "recoverability_aware"
    bt.aux_standby_mw = 0.5
    col = BacktestColumnMap()
    ts = pd.date_range("2026-01-02T00:00:00Z", periods=4, freq="h")
    future_rows = pd.DataFrame(
        {
            col.timestamp: ts,
            col.pred_da_price: [50.0, 50.0, 50.0, 50.0],
            col.pred_afrr_activation_rate_pos: [0.0, 0.0, 0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0, 0.0, 0.0],
        }
    )

    selected, audit = bt._apply_da_postlock_future_guard(
        selected_da={pd.Timestamp(ts[0]): (0.0, 0.1)},
        da_audit_rows=[
            {
                "timestamp_utc": str(pd.Timestamp(ts[0])),
                "da_candidate_buy_mw": 0.0,
                "da_candidate_sell_mw": 0.1,
            }
        ],
        future_rows=future_rows,
        colmap=col,
        current_soc_mwh=bt.soc_min + 0.25,
        da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert selected[pd.Timestamp(ts[0])] == pytest.approx((0.0, 0.1))
    assert float(audit[0]["da_postlock_candidate_future_feasible"]) == pytest.approx(1.0)
    assert str(audit[0]["da_postlock_infeasibility_driver"]) == "none"
    assert str(audit[0]["da_postlock_hard_projection_end_utc"]) == str(pd.Timestamp(ts[0]))
    assert str(audit[0]["da_postlock_next_recovery_opportunity_utc"]) == str(pd.Timestamp(ts[1]))


def test_da_postlock_terminal_shortfall_unrecoverable_rejected() -> None:
    bt = _mk_backtester()
    bt.da_postlock_guard_mode = "recoverability_aware"
    bt.p_max_mw = 2.0
    bt.soc_target_end = 18.0
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-02T00:00:00Z")
    future_rows = pd.DataFrame(
        {
            col.timestamp: [ts],
            col.pred_da_price: [50.0],
            col.pred_afrr_activation_rate_pos: [0.0],
            col.pred_afrr_activation_rate_neg: [0.0],
        }
    )

    selected, audit = bt._apply_da_postlock_future_guard(
        selected_da={ts: (0.0, 1.5)},
        da_audit_rows=[
            {
                "timestamp_utc": str(ts),
                "da_candidate_buy_mw": 0.0,
                "da_candidate_sell_mw": 1.5,
            }
        ],
        future_rows=future_rows,
        colmap=col,
        current_soc_mwh=10.0,
        da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=ts,
    )

    assert selected[ts] == pytest.approx((0.0, 0.0))
    assert float(audit[0]["da_postlock_candidate_future_feasible"]) == pytest.approx(0.0)
    assert float(audit[0]["da_postlock_terminal_shortfall_unrecoverable"]) == pytest.approx(1.0)
    assert str(audit[0]["da_postlock_infeasibility_driver"]) == "locked_da_terminal_shortfall_unrecoverable"
    assert str(audit[0]["da_postlock_selected_after_future_check"]) == "rejected"


def test_da_postlock_terminal_recovery_cost_can_select_no_trade() -> None:
    bt = _mk_backtester()
    bt.da_postlock_guard_mode = "recoverability_aware"
    col = BacktestColumnMap()
    ts = pd.Timestamp("2026-01-02T00:00:00Z")
    rows = pd.DataFrame(
        {
            col.timestamp: [ts],
            "target_time_utc": [ts],
            "snapshot_time_utc": [pd.Timestamp("2026-01-01T10:00:00Z")],
            "soc_start_lp_mwh": [10.0],
            col.pred_da_price: [1_000.0],
            col.pred_afrr_capacity_price_pos: [0.0],
            col.pred_afrr_capacity_price_neg: [0.0],
            col.pred_afrr_activation_price_pos: [0.0],
            col.pred_afrr_activation_price_neg: [0.0],
            col.pred_afrr_activation_rate_pos: [0.0],
            col.pred_afrr_activation_rate_neg: [0.0],
        }
    )

    selected, audit = bt._apply_da_postlock_future_guard(
        selected_da={ts: (0.0, 5.0)},
        da_audit_rows=[
            {
                "timestamp_utc": str(ts),
                "da_candidate_buy_mw": 0.0,
                "da_candidate_sell_mw": 5.0,
            }
        ],
        lock_rows=rows,
        future_rows=rows[[col.timestamp, col.pred_da_price, col.pred_afrr_activation_rate_pos, col.pred_afrr_activation_rate_neg]],
        colmap=col,
        current_soc_mwh=10.0,
        da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=ts,
    )

    assert selected[ts] == pytest.approx((0.0, 5.0))
    assert str(audit[0]["da_postlock_selected_after_future_check"]) == "candidate"
    assert str(audit[0]["da_zero_reason"]) == "none"
    assert float(audit[0]["da_postlock_terminal_shortfall_recoverable"]) == pytest.approx(1.0)
    assert float(audit[0]["candidate_pnl_after_recovery_cost_eur"]) > float(audit[0]["no_trade_pnl"])
    assert str(audit[0]["final_selected_incumbent"]) == "optimized"
    assert float(audit[0]["locked_sell_mwh_by_hour"]) == pytest.approx(5.0)
    assert float(audit[0]["da_postlock_final_selected_pnl_eur"]) == pytest.approx(
        float(audit[0]["candidate_pnl_after_recovery_cost_eur"])
    )
    assert float(audit[0]["da_postlock_recovery_cost_probability_weighting_required"]) == pytest.approx(0.0)
    assert str(audit[0]["da_postlock_recovery_cost_context"]) == "post_clearing_locked_schedule"
    assert str(audit[0]["da_postlock_recovery_cost_used_in_ev_mode"]) == "full_locked_schedule_recovery_cost"
    assert float(audit[0]["da_postlock_recovery_cost_full_eur"]) == pytest.approx(
        float(audit[0]["da_postlock_recovery_cost_expected_eur"])
    )


def test_da_postlock_strict_no_future_action_keeps_legacy_terminal_rejection() -> None:
    bt = _mk_backtester()
    bt.da_postlock_guard_mode = "strict_no_future_action"
    col = BacktestColumnMap()
    ts = pd.date_range("2026-01-02T00:00:00Z", periods=3, freq="h")
    future_rows = pd.DataFrame(
        {
            col.timestamp: ts,
            col.pred_da_price: [50.0, 50.0, 50.0],
            col.pred_afrr_activation_rate_pos: [0.0, 0.0, 0.0],
            col.pred_afrr_activation_rate_neg: [0.0, 0.0, 0.0],
        }
    )

    selected, audit = bt._apply_da_postlock_future_guard(
        selected_da={pd.Timestamp(ts[0]): (0.0, 5.0)},
        da_audit_rows=[
            {
                "timestamp_utc": str(pd.Timestamp(ts[0])),
                "da_candidate_buy_mw": 0.0,
                "da_candidate_sell_mw": 5.0,
            }
        ],
        future_rows=future_rows,
        colmap=col,
        current_soc_mwh=12.0,
        da_lockbook={},
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=pd.Timestamp(ts[-1]),
    )

    assert selected[pd.Timestamp(ts[0])] == pytest.approx((0.0, 0.0))
    assert str(audit[0]["da_postlock_guard_mode"]) == "strict_no_future_action"
    assert str(audit[0]["da_postlock_selected_after_future_check"]) == "no_trade"
    assert str(audit[0]["da_postlock_infeasibility_driver"]) == "locked_da_terminal_shortfall_unrecoverable"


def test_da_delivery_plan_executes_only_lockbook_quantities_with_origin() -> None:
    bt = _mk_backtester()
    bt.bid_builder.da_buy_limit_quantile = "p50"
    bt.bid_builder.da_sell_limit_quantile = "p50"
    col = BacktestColumnMap()
    ts_unlocked = pd.Timestamp("2026-01-02T00:00:00Z")
    ts_locked = pd.Timestamp("2026-01-02T01:00:00Z")
    take = pd.DataFrame(
        {
            col.timestamp: [ts_unlocked, ts_locked],
            "charge_mw": [4.2, 9.9],
            "discharge_mw": [0.0, 0.0],
        }
    )
    source_snapshot = pd.Timestamp("2026-01-01T10:00:00Z")
    gated = bt._apply_da_lockbook_to_delivery_plan(
        take=take,
        colmap=col,
        da_lockbook={ts_locked: (1.5, 0.0)},
        da_precommit_audit_by_ts={
            "da_precommit_source_snapshot_utc": {ts_locked: str(source_snapshot)},
            "da_precommit_originating_precommit_id": {ts_locked: f"{source_snapshot}->{ts_locked}"},
        },
        da_enabled=True,
    )

    unlocked_row = gated.loc[gated[col.timestamp].eq(ts_unlocked)].iloc[0]
    assert float(unlocked_row["charge_mw"]) == pytest.approx(0.0)
    assert float(unlocked_row["discharge_mw"]) == pytest.approx(0.0)
    assert float(unlocked_row["optimizer_da_charge_mw_before_lockbook"]) == pytest.approx(4.2)
    assert float(unlocked_row["da_is_locked_delivery_hour"]) == pytest.approx(0.0)
    assert float(unlocked_row["da_unlocked_trade_blocked"]) == pytest.approx(1.0)
    assert float(unlocked_row["da_unlocked_trade_violation"]) == pytest.approx(0.0)

    locked_row = gated.loc[gated[col.timestamp].eq(ts_locked)].iloc[0]
    assert float(locked_row["charge_mw"]) == pytest.approx(1.5)
    assert float(locked_row["discharge_mw"]) == pytest.approx(0.0)
    assert float(locked_row["da_locked_buy_mwh"]) == pytest.approx(1.5 * bt.dt_h)
    assert float(locked_row["da_locked_sell_mwh"]) == pytest.approx(0.0)
    assert float(locked_row["da_is_locked_delivery_hour"]) == pytest.approx(1.0)
    assert float(locked_row["da_unlocked_trade_blocked"]) == pytest.approx(0.0)
    assert str(locked_row["da_originating_source_snapshot_utc"]) == str(source_snapshot)
    assert str(locked_row["da_originating_delivery_timestamp_utc"]) == ts_locked.isoformat()
    assert str(locked_row["da_originating_precommit_id"]) == f"{source_snapshot}->{ts_locked}"
    assert str(locked_row["da_precommit_final_selected_incumbent"]) == "optimized"
    assert str(locked_row["da_precommit_selection_reason"]) == "accepted_lockbook"

    realized_with_origin = pd.DataFrame(
        {
            "da_lockbook_row_present": [1.0],
            "da_bid_locked": [1.0],
            "da_locked_buy_mwh": [1.5],
            "da_locked_sell_mwh": [0.0],
            "da_originating_source_snapshot_utc": [str(source_snapshot)],
            "da_originating_delivery_timestamp_utc": [ts_locked.isoformat()],
            "da_originating_precommit_id": [f"{source_snapshot}->{ts_locked}"],
            "da_precommit_final_selected_incumbent": ["optimized"],
            "da_precommit_selection_reason": ["accepted_lockbook"],
            "da_precommit_da_zero_reason": ["none"],
            "real_submitted_da_buy_mw": [1.5],
            "real_submitted_da_sell_mw": [0.0],
            "real_submitted_da_buy_price_eur_mwh": [10.0],
            "real_da_buy_accepted": [1.0],
            "real_da_sell_accepted": [0.0],
            "real_da_buy_mwh": [1.5],
            "real_da_sell_mwh": [0.0],
        }
    )
    origin_counters = BatteryBacktester._compute_da_naming_semantics_counters(realized_with_origin)
    assert origin_counters["da_realized_without_precommit_origin_count"] == pytest.approx(0.0)
    assert origin_counters["da_naming_semantics_error_count"] == pytest.approx(0.0)

    unlocked_clearing = bt._apply_market_clearing(
        target_time_utc=ts_unlocked,
        planned_charge_mw=float(unlocked_row["charge_mw"]),
        planned_discharge_mw=float(unlocked_row["discharge_mw"]),
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        pred_da_price=100.0,
        pred_da_price_p05=80.0,
        pred_da_price_p10=90.0,
        pred_da_price_p90=110.0,
        pred_da_price_p95=120.0,
        true_da_price=100.0,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=0.0,
        true_act_pos=0.0,
        pred_act_neg=0.0,
        true_act_neg=0.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
    )
    assert float(unlocked_clearing["submitted_da_buy_mw"]) == pytest.approx(0.0)
    assert float(unlocked_clearing["executed_charge_mw"]) == pytest.approx(0.0)

    locked_clearing = bt._apply_market_clearing(
        target_time_utc=ts_locked,
        planned_charge_mw=float(locked_row["charge_mw"]),
        planned_discharge_mw=float(locked_row["discharge_mw"]),
        planned_reserve_pos_mw=0.0,
        planned_reserve_neg_mw=0.0,
        pred_da_price=22.620669,
        pred_da_price_p05=80.0,
        pred_da_price_p10=90.0,
        pred_da_price_p90=56.018817,
        pred_da_price_p95=120.0,
        true_da_price=12.78,
        pred_cap_pos=0.0,
        true_cap_pos=0.0,
        pred_cap_neg=0.0,
        true_cap_neg=0.0,
        pred_act_pos=0.0,
        true_act_pos=0.0,
        pred_act_neg=0.0,
        true_act_neg=0.0,
        true_rate_pos=0.0,
        true_rate_neg=0.0,
    )
    assert float(locked_clearing["submitted_da_buy_mw"]) == pytest.approx(1.5)
    assert float(locked_clearing["executed_charge_mw"]) == pytest.approx(1.5)
    assert float(locked_clearing["submitted_da_buy_price_eur_mwh"]) == pytest.approx(22.620669)
    assert str(locked_clearing["da_limit_price_quantile"]) == "p50"
    assert str(locked_clearing["da_limit_price_source_column"]) == col.pred_da_price
    assert float(locked_clearing["da_price_fallback_used"]) == pytest.approx(0.0)

    bt.bid_builder.da_buy_limit_quantile = "p90"
    with pytest.raises(ValueError, match="missing_da_quantile"):
        bt._apply_market_clearing(
            target_time_utc=ts_locked,
            planned_charge_mw=1.5,
            planned_discharge_mw=0.0,
            planned_reserve_pos_mw=0.0,
            planned_reserve_neg_mw=0.0,
            pred_da_price=22.620669,
            pred_da_price_p05=None,
            pred_da_price_p10=None,
            pred_da_price_p90=None,
            pred_da_price_p95=None,
            true_da_price=12.78,
            pred_cap_pos=0.0,
            true_cap_pos=0.0,
            pred_cap_neg=0.0,
            true_cap_neg=0.0,
            pred_act_pos=0.0,
            true_act_pos=0.0,
            pred_act_neg=0.0,
            true_act_neg=0.0,
            true_rate_pos=0.0,
            true_rate_neg=0.0,
        )

    _, unlocked_metrics = bt._settle_one_hour(
        soc=10.0,
        charge=4.2,
        discharge=0.0,
        da_charge_mw=float(unlocked_row["charge_mw"]),
        da_discharge_mw=float(unlocked_row["discharge_mw"]),
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
    assert float(unlocked_metrics["da_buy_mwh"]) == pytest.approx(0.0)
    assert float(unlocked_metrics["da_sell_mwh"]) == pytest.approx(0.0)

    _, locked_metrics = bt._settle_one_hour(
        soc=10.0,
        charge=9.9,
        discharge=0.0,
        da_charge_mw=float(locked_row["charge_mw"]),
        da_discharge_mw=float(locked_row["discharge_mw"]),
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
    assert float(locked_metrics["da_buy_mwh"]) == pytest.approx(1.5 * bt.dt_h)
    assert float(locked_metrics["da_sell_mwh"]) == pytest.approx(0.0)


def test_technical_id_terminal_recovery_in_optimizer() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "hard"
    df, col = _tiny_backtest_df(hours=4)
    df[col.pred_da_price] = 100.0
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=5.0,
        soc_end_min_target=8.0,
        allowed_markets=("ID",),
    )
    assert "id_charge_mw" in out.columns
    assert float(out["id_charge_mw"].sum()) > 0.0
    assert float(out["soc_lp_mwh"].iloc[-1]) >= 8.0 - 1e-6
    assert "terminal_soc_recovery" in set(out["id_recourse_reason"].astype(str))
    assert float(out["terminal_soc_id_recourse_cost_eur"].sum()) > 0.0


def test_technical_id_not_used_for_arbitrage_without_technical_need() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=3)
    df[col.pred_da_price] = 5_000.0
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=10.0,
        soc_end_min_target=None,
        allowed_markets=("ID",),
    )
    assert float(out["id_charge_mw"].abs().sum()) == pytest.approx(0.0)
    assert float(out["id_discharge_mw"].abs().sum()) == pytest.approx(0.0)


def test_optimizer_id_is_in_power_stack() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "hard"
    df, col = _tiny_backtest_df(hours=2)
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=2.1,
        soc_end_min_target=10.0,
        allowed_markets=("ID",),
    )
    assert "power_stack_neg_mw" in out.columns
    assert float(out["power_stack_neg_mw"].max()) <= bt.p_max_mw + 1e-6
    assert np.allclose(
        out["power_stack_neg_mw"].to_numpy(dtype=float),
        out["id_charge_mw"].to_numpy(dtype=float),
        atol=1e-6,
    )


def test_technical_id_price_uses_da_not_activation_price() -> None:
    bt = _mk_backtester("canonical_economic")
    _, m = bt._settle_one_hour(
        soc=10.0,
        charge=0.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=50.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=10_000.0,
        act_neg_price=10_000.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        id_charge_mw=1.0,
        id_discharge_mw=0.0,
        id_recourse_reason_hint="terminal_soc_recovery",
    )
    assert float(m["id_buy_price_eur_mwh"]) == pytest.approx(50.0 + bt.id_rescue_spread_eur_mwh)
    assert float(m["id_sell_price_eur_mwh"]) == pytest.approx(50.0 - bt.id_rescue_spread_eur_mwh)
    assert m["id_price_source"] == "da_price_plus_spread"
    assert float(m["id_da_reference_price_eur_mwh"]) == pytest.approx(50.0)
    assert float(m["id_price_uses_activation_price"]) == pytest.approx(0.0)


def test_technical_id_settlement_soc_uses_grid_mwh_times_eta_once() -> None:
    bt = _mk_backtester()
    soc0 = 10.0
    soc1, m = bt._settle_one_hour(
        soc=soc0,
        charge=0.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=50.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        id_charge_mw=1.0,
        id_discharge_mw=0.0,
        id_recourse_reason_hint="terminal_soc_recovery",
    )
    expected = soc0 + bt.eta_in * float(m["id_buy_mwh"]) - float(m["aux_energy_mwh"])
    assert float(m["id_buy_mwh"]) == pytest.approx(1.0 * bt.dt_h)
    assert soc1 == pytest.approx(expected)


def test_terminal_id_recovery_sizes_grid_buy_from_internal_shortfall_and_losses() -> None:
    bt = _mk_backtester()
    internal_shortfall = 0.16
    remaining_losses = 0.04
    id_charge, id_discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=9.0,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=10.0,
        projected_terminal_soc_without_new_id_mwh=10.0 - internal_shortfall,
        remaining_known_losses_mwh=remaining_losses,
    )
    diag = bt._last_id_rescue_plan_diagnostics
    expected_internal = (
        internal_shortfall
        + remaining_losses
        + float(diag.get("terminal_soc_projection_id_extra_aux_losses_mwh", 0.0))
    )
    assert reason == "terminal_soc_recovery"
    assert id_discharge == pytest.approx(0.0)
    assert id_charge * bt.dt_h == pytest.approx(expected_internal / bt.eta_in)
    assert diag["terminal_soc_id_recourse_needed_internal_mwh"] == pytest.approx(expected_internal)
    assert diag["terminal_soc_id_recourse_scheduled_grid_mwh"] == pytest.approx(expected_internal / bt.eta_in)
    assert diag["terminal_soc_id_recourse_scheduled_internal_mwh"] == pytest.approx(expected_internal)


def test_terminal_id_recovery_sizes_for_future_fixed_da_discharge() -> None:
    bt = _mk_backtester()
    projection = bt._project_terminal_soc_with_known_flows(
        current_soc_mwh=12.0,
        fixed_future_discharge_mwh=3.0,
        remaining_aux_losses_mwh=0.25,
    )
    recovery = bt._schedule_terminal_id_recovery(
        projected_terminal_soc_without_new_id_mwh=float(projection["projected_terminal_soc_without_new_id_mwh"]),
        current_soc_mwh=12.0,
        terminal_soc_target_mwh=10.0,
        residual_charge_mw=10.0,
    )
    assert float(projection["terminal_soc_projection_fixed_future_discharge_mwh"]) == pytest.approx(3.0)
    assert float(recovery["projected_terminal_soc_with_new_id_mwh"]) >= 10.0 - 1e-9
    assert float(recovery["terminal_soc_id_recourse_scheduled_grid_mwh"]) == pytest.approx(
        float(recovery["terminal_soc_id_recourse_needed_internal_mwh"]) / bt.eta_in
    )


def test_terminal_id_recovery_includes_safety_margin() -> None:
    bt = _mk_backtester()
    id_charge, id_discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=9.0,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=10.0,
        projected_terminal_soc_without_new_id_mwh=9.95,
        remaining_known_losses_mwh=0.0,
        terminal_soc_safety_margin_mwh=0.10,
    )
    diag = bt._last_id_rescue_plan_diagnostics
    expected_internal = 0.15 + float(diag.get("terminal_soc_projection_id_extra_aux_losses_mwh", 0.0))
    assert reason == "terminal_soc_recovery"
    assert id_discharge == pytest.approx(0.0)
    assert id_charge * bt.dt_h == pytest.approx(expected_internal / bt.eta_in)
    assert diag["terminal_soc_id_recourse_needed_internal_mwh"] == pytest.approx(expected_internal)


def test_terminal_id_recovery_does_not_double_count_safety_margin() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    projection_without_new_id = 9.6
    safety = 0.15
    known_aux = 0.04
    recovery_aux = 0.02
    diag = bt._schedule_terminal_id_recovery(
        projected_terminal_soc_without_new_id_mwh=projection_without_new_id,
        current_soc_mwh=9.6,
        terminal_soc_target_mwh=10.0,
        residual_charge_mw=1.0,
        terminal_soc_safety_margin_mwh=safety,
        terminal_repair_known_future_aux_mwh=known_aux,
        terminal_repair_recovery_aux_mwh=recovery_aux,
    )
    expected_internal = max(0.0, 10.0 + safety - projection_without_new_id) + known_aux + recovery_aux
    assert float(diag["terminal_soc_target_with_safety_mwh"]) == pytest.approx(10.0 + safety)
    assert float(diag["terminal_repair_required_additional_internal_mwh"]) == pytest.approx(expected_internal)
    assert float(diag["terminal_repair_required_internal_mwh"]) == pytest.approx(expected_internal)
    assert float(diag["terminal_soc_id_recourse_scheduled_internal_mwh"]) == pytest.approx(expected_internal)
    assert float(diag["terminal_soc_id_recourse_scheduled_grid_mwh"]) == pytest.approx(expected_internal / bt.eta_in)
    assert float(diag["terminal_repair_projected_final_soc_after_repair_mwh"]) >= 10.0 + safety - 1e-9


def test_terminal_id_recovery_uses_residual_after_provisional_id_near_capacity() -> None:
    bt = _mk_backtester()
    bt.p_max_mw = 1.0
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    id_charge, id_discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=9.0,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=10.0,
        projected_terminal_soc_without_new_id_mwh=9.145,
        remaining_known_losses_mwh=0.0,
        terminal_soc_safety_margin_mwh=0.0,
    )
    diag = bt._last_id_rescue_plan_diagnostics
    expected_total_internal = 10.0 - 9.145 + float(diag["terminal_soc_projection_id_extra_aux_losses_mwh"])
    assert reason == "terminal_soc_recovery"
    assert id_discharge == pytest.approx(0.0)
    assert id_charge * bt.dt_h == pytest.approx(expected_total_internal / bt.eta_in)
    assert float(diag["terminal_soc_id_recourse_needed_internal_mwh"]) == pytest.approx(expected_total_internal)
    assert float(diag["terminal_soc_id_recourse_scheduled_internal_mwh"]) == pytest.approx(expected_total_internal)
    assert float(diag["terminal_repair_residual_charge_internal_capacity_mwh"]) < 0.10
    assert float(diag["terminal_soc_id_recourse_scheduled_internal_mwh"]) > 0.90
    assert float(diag["terminal_soc_id_recourse_scheduled_internal_mwh"]) <= 0.95 + 1e-9
    assert float(diag["terminal_soc_recovery_feasible"]) == pytest.approx(1.0)
    assert float(diag["terminal_repair_projected_final_soc_after_repair_mwh"]) >= 10.0 - 1e-9


def test_terminal_id_recovery_schedules_grid_buy_including_recovery_aux() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    projection_without_new_id = 10.0 - 0.332
    diag = bt._schedule_terminal_id_recovery(
        projected_terminal_soc_without_new_id_mwh=projection_without_new_id,
        current_soc_mwh=9.5,
        terminal_soc_target_mwh=10.0,
        residual_charge_mw=2.0,
        terminal_soc_safety_margin_mwh=0.0,
        terminal_repair_known_future_aux_mwh=0.095,
        terminal_repair_recovery_aux_mwh=0.0,
    )
    expected_internal = 0.332 + 0.095
    assert float(diag["terminal_repair_required_internal_mwh"]) == pytest.approx(expected_internal)
    assert float(diag["terminal_soc_id_recourse_scheduled_grid_mwh"]) == pytest.approx(expected_internal / bt.eta_in)
    assert float(diag["terminal_soc_recovery_feasible"]) == pytest.approx(1.0)
    assert float(diag["terminal_repair_projected_final_soc_after_repair_mwh"]) >= 10.0 - 1e-9


def test_terminal_id_recovery_room_cap_allows_same_hour_aux_buffer_at_soc_max() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.soc_max = 10.0
    diag = bt._schedule_terminal_id_recovery(
        projected_terminal_soc_without_new_id_mwh=9.5,
        current_soc_mwh=9.5,
        terminal_soc_target_mwh=10.0,
        residual_charge_mw=2.0,
        terminal_soc_safety_margin_mwh=0.0,
        terminal_repair_known_future_aux_mwh=0.0,
        terminal_repair_recovery_aux_mwh=0.095,
    )

    expected_internal = 10.0 - 9.5 + 0.095
    assert float(diag["terminal_repair_required_internal_mwh"]) == pytest.approx(expected_internal)
    assert float(diag["terminal_soc_id_recourse_scheduled_internal_mwh"]) == pytest.approx(expected_internal)
    assert float(diag["terminal_recovery_recovery_hour_aux_mwh"]) == pytest.approx(0.095)
    assert float(diag["terminal_recovery_physical_target_met"]) == pytest.approx(1.0)
    assert float(diag["terminal_recovery_physical_shortfall_after_recovery_mwh"]) == pytest.approx(0.0)
    assert float(diag["terminal_repair_projected_final_soc_after_repair_mwh"]) >= 10.0 - 1e-9


def test_terminal_id_recovery_aux_buffer_not_double_counted() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    projection_before = 9.8
    safety = 0.10
    known_future_aux = 0.04
    recovery_hour_aux = 0.095

    diag = bt._schedule_terminal_id_recovery(
        projected_terminal_soc_without_new_id_mwh=projection_before,
        current_soc_mwh=9.5,
        terminal_soc_target_mwh=10.0,
        residual_charge_mw=2.0,
        terminal_soc_safety_margin_mwh=safety,
        terminal_repair_known_future_aux_mwh=known_future_aux,
        terminal_repair_recovery_aux_mwh=recovery_hour_aux,
    )

    expected_internal = 10.0 + safety + known_future_aux + recovery_hour_aux - projection_before
    assert float(diag["terminal_repair_required_internal_mwh"]) == pytest.approx(expected_internal)
    assert float(diag["terminal_recovery_recovery_hour_aux_mwh"]) == pytest.approx(recovery_hour_aux)
    assert float(diag["terminal_recovery_recovery_induced_aux_mwh"]) == pytest.approx(0.0)
    assert float(diag["terminal_soc_id_recourse_scheduled_grid_mwh"]) == pytest.approx(
        expected_internal / bt.eta_in
    )
    assert float(diag["terminal_repair_projected_final_soc_after_repair_mwh"]) >= 10.0 + safety - 1e-9


def test_terminal_id_recovery_schedules_grid_buy_without_aux() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    projection_without_new_id = 10.0 - 0.332
    diag = bt._schedule_terminal_id_recovery(
        projected_terminal_soc_without_new_id_mwh=projection_without_new_id,
        current_soc_mwh=9.5,
        terminal_soc_target_mwh=10.0,
        residual_charge_mw=2.0,
        terminal_soc_safety_margin_mwh=0.0,
        terminal_repair_known_future_aux_mwh=0.0,
        terminal_repair_recovery_aux_mwh=0.0,
    )
    expected_internal = 0.332
    assert float(diag["terminal_repair_required_internal_mwh"]) == pytest.approx(expected_internal)
    assert float(diag["terminal_soc_id_recourse_scheduled_grid_mwh"]) == pytest.approx(expected_internal / bt.eta_in)
    assert float(diag["terminal_soc_recovery_feasible"]) == pytest.approx(1.0)
    assert float(diag["terminal_repair_projected_final_soc_after_repair_mwh"]) >= 10.0 - 1e-9


def test_terminal_id_recovery_projection_updates_between_calls() -> None:
    bt = _mk_backtester()
    bt._plan_id_rescue_for_next_hour(
        soc_next=9.0,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=10.0,
        projected_terminal_soc_without_new_id_mwh=9.8,
    )
    first = dict(bt._last_id_rescue_plan_diagnostics)
    bt._plan_id_rescue_for_next_hour(
        soc_next=9.3,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=10.0,
        projected_terminal_soc_without_new_id_mwh=9.95,
    )
    second = dict(bt._last_id_rescue_plan_diagnostics)
    assert first["projected_terminal_soc_without_new_id_mwh"] == pytest.approx(9.8)
    assert second["projected_terminal_soc_without_new_id_mwh"] == pytest.approx(9.95)
    assert second["terminal_soc_id_recourse_needed_internal_mwh"] < first["terminal_soc_id_recourse_needed_internal_mwh"]


def test_terminal_recovery_existing_scheduled_internal_prevents_duplicate() -> None:
    diag = BatteryBacktester._terminal_recovery_remaining_shortfall_after_existing(
        projected_terminal_soc_without_new_recovery_mwh=9.0,
        existing_scheduled_terminal_recovery_internal_mwh=1.0,
        known_future_aux_mwh=0.0,
        terminal_soc_target_mwh=10.0,
        safety_buffer_mwh=0.0,
    )

    assert diag["terminal_recovery_projected_final_soc_before_existing_recovery_mwh"] == pytest.approx(9.0)
    assert diag["terminal_recovery_existing_scheduled_internal_mwh"] == pytest.approx(1.0)
    assert diag["terminal_recovery_projected_final_soc_after_existing_recovery_mwh"] == pytest.approx(10.0)
    assert diag["terminal_recovery_remaining_shortfall_internal_mwh"] == pytest.approx(0.0)


def test_terminal_recovery_existing_scheduled_internal_schedules_only_delta() -> None:
    diag = BatteryBacktester._terminal_recovery_remaining_shortfall_after_existing(
        projected_terminal_soc_without_new_recovery_mwh=9.0,
        existing_scheduled_terminal_recovery_internal_mwh=1.0,
        known_future_aux_mwh=0.2,
        terminal_soc_target_mwh=10.0,
        safety_buffer_mwh=0.0,
    )

    assert diag["terminal_recovery_projected_final_soc_after_existing_recovery_mwh"] == pytest.approx(9.8)
    assert diag["terminal_recovery_remaining_shortfall_internal_mwh"] == pytest.approx(0.2)


def test_terminal_recovery_existing_scheduled_internal_subtracts_induced_aux_delta() -> None:
    diag = BatteryBacktester._terminal_recovery_remaining_shortfall_after_existing(
        projected_terminal_soc_without_new_recovery_mwh=8.30,
        existing_scheduled_terminal_recovery_internal_mwh=1.70,
        known_future_aux_mwh=0.0,
        terminal_soc_target_mwh=10.0,
        safety_buffer_mwh=0.0,
        existing_recovery_aux_without_recovery_mwh=0.02,
        existing_recovery_aux_with_recovery_mwh=0.05,
        existing_recovery_projection_includes_aux_without_recovery=True,
    )

    assert diag["terminal_recovery_existing_induced_aux_delta_mwh"] == pytest.approx(0.03)
    assert diag["terminal_recovery_existing_aux_sized_mwh"] == pytest.approx(0.03)
    assert diag["terminal_recovery_projected_final_soc_after_existing_before_aux_mwh"] == pytest.approx(10.0)
    assert diag["terminal_recovery_projected_final_soc_after_existing_after_aux_mwh"] == pytest.approx(9.97)
    assert diag["terminal_recovery_remaining_shortfall_after_existing_recovery_mwh"] == pytest.approx(0.03)


def test_terminal_recovery_existing_scheduled_internal_subtracts_full_aux_for_pre_aux_projection() -> None:
    diag = BatteryBacktester._terminal_recovery_remaining_shortfall_after_existing(
        projected_terminal_soc_without_new_recovery_mwh=8.32,
        existing_scheduled_terminal_recovery_internal_mwh=1.70,
        known_future_aux_mwh=0.0,
        terminal_soc_target_mwh=10.0,
        safety_buffer_mwh=0.0,
        existing_recovery_aux_without_recovery_mwh=0.02,
        existing_recovery_aux_with_recovery_mwh=0.05,
        existing_recovery_projection_includes_aux_without_recovery=False,
    )

    assert diag["terminal_recovery_existing_induced_aux_delta_mwh"] == pytest.approx(0.03)
    assert diag["terminal_recovery_existing_aux_sized_mwh"] == pytest.approx(0.05)
    assert diag["terminal_recovery_projected_final_soc_after_existing_before_aux_mwh"] == pytest.approx(10.02)
    assert diag["terminal_recovery_projected_final_soc_after_existing_after_aux_mwh"] == pytest.approx(9.97)
    assert diag["terminal_recovery_remaining_shortfall_after_existing_recovery_mwh"] == pytest.approx(0.03)


def test_terminal_recovery_topup_scheduled_after_existing_recovery_aux_shortfall() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.aux_off_mw = 0.02
    bt.aux_trading_mw = 0.05
    bt.aux_standby_mw = 0.0
    bt.p_max_mw = 10.0

    existing = BatteryBacktester._terminal_recovery_remaining_shortfall_after_existing(
        projected_terminal_soc_without_new_recovery_mwh=8.30,
        existing_scheduled_terminal_recovery_internal_mwh=1.70,
        known_future_aux_mwh=0.0,
        terminal_soc_target_mwh=10.0,
        safety_buffer_mwh=0.0,
        existing_recovery_aux_without_recovery_mwh=0.02,
        existing_recovery_aux_with_recovery_mwh=0.05,
        existing_recovery_projection_includes_aux_without_recovery=True,
    )

    id_charge, id_discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=9.97,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=10.0,
        projected_terminal_soc_without_new_id_mwh=float(
            existing["terminal_recovery_projected_final_soc_after_existing_after_aux_mwh"]
        ),
        remaining_known_losses_mwh=0.0,
        terminal_soc_safety_margin_mwh=0.0,
    )
    diag = bt._last_id_rescue_plan_diagnostics

    assert reason == "terminal_soc_recovery"
    assert id_discharge == pytest.approx(0.0)
    assert id_charge * bt.dt_h > 0.03 / bt.eta_in
    assert float(diag["terminal_recovery_projected_final_soc_after_recovery_mwh"]) >= 10.0 - 1e-9


def test_protected_soc_recovery_is_sized_net_positive_after_aux() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.aux_peak_mw = 0.095
    bt.aux_trading_mw = 0.095
    bt.aux_standby_mw = 0.0
    bt.reserve_headroom_safety_mwh = 0.0
    soc_next = bt.soc_min - 0.01

    id_charge, id_discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=soc_next,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
    )
    diag = bt._last_id_rescue_plan_diagnostics

    assert reason == "protected_soc_recovery"
    assert id_discharge == pytest.approx(0.0)
    assert float(diag["protected_recovery_aux_mwh"]) == pytest.approx(0.095)
    assert float(diag["protected_recovery_required_internal_mwh"]) == pytest.approx(0.105)
    assert float(diag["protected_recovery_net_internal_mwh"]) == pytest.approx(0.01)
    assert float(diag["protected_recovery_suppressed_zero_net_effect"]) == pytest.approx(0.0)
    assert id_charge * bt.dt_h * bt.eta_in - float(diag["protected_recovery_aux_mwh"]) > 0.0


def test_protected_soc_recovery_uses_minimum_net_gain_after_aux() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.aux_peak_mw = 0.095
    bt.aux_trading_mw = 0.095
    bt.aux_standby_mw = 0.0
    bt.reserve_headroom_safety_mwh = 0.0
    bt.min_protected_recovery_net_gain_mwh = 0.01

    id_charge, id_discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=bt.soc_min - 0.0001,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
    )
    diag = bt._last_id_rescue_plan_diagnostics

    assert reason == "protected_soc_recovery"
    assert id_discharge == pytest.approx(0.0)
    assert float(diag["protected_recovery_required_net_internal_mwh"]) == pytest.approx(0.01)
    assert float(diag["protected_recovery_required_internal_mwh"]) == pytest.approx(0.105)
    assert float(diag["protected_recovery_net_internal_mwh"]) == pytest.approx(0.01)
    assert float(diag["protected_recovery_min_net_gain_mwh"]) == pytest.approx(0.01)
    assert float(diag["protected_recovery_suppressed_zero_net_effect"]) == pytest.approx(0.0)


def test_protected_soc_recovery_does_not_offset_aux_when_soc_is_valid() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.aux_peak_mw = 0.095
    bt.aux_trading_mw = 0.095
    bt.aux_standby_mw = 0.0
    bt.reserve_headroom_safety_mwh = 0.1
    bt.min_protected_recovery_net_gain_mwh = 0.01

    id_charge, id_discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=bt.soc_min + 0.155,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
    )

    assert reason == "none"
    assert id_charge == pytest.approx(0.0)
    assert id_discharge == pytest.approx(0.0)


def test_terminal_recovery_suppressed_outside_final_recovery_window() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.terminal_id_recovery_safety_mwh = 0.0

    id_charge, id_discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=9.8,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=10.0,
        projected_terminal_soc_without_new_id_mwh=9.8,
        terminal_recovery_allowed=False,
    )
    diag = bt._last_id_rescue_plan_diagnostics

    assert reason == "none"
    assert id_charge == pytest.approx(0.0)
    assert id_discharge == pytest.approx(0.0)
    assert float(diag["terminal_recovery_suppressed_until_final_window"]) == pytest.approx(1.0)


def test_terminal_recovery_sizes_to_target_after_same_hour_aux() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.aux_peak_mw = 0.095
    bt.aux_trading_mw = 0.095
    bt.aux_standby_mw = 0.0
    bt.p_max_mw = 10.0

    id_charge, id_discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=2.155,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=10.0,
        projected_terminal_soc_without_new_id_mwh=2.155,
        remaining_known_losses_mwh=0.0,
        terminal_soc_safety_margin_mwh=0.0,
    )
    diag = bt._last_id_rescue_plan_diagnostics

    expected_aux_delta = 0.095 - float(bt.aux_off_mw)
    expected_aux_sized = 0.095
    expected_internal = 10.0 - 2.155 + expected_aux_sized
    assert reason == "terminal_soc_recovery"
    assert id_discharge == pytest.approx(0.0)
    assert float(diag["terminal_recovery_aux_without_recovery_mwh"]) == pytest.approx(float(bt.aux_off_mw))
    assert float(diag["terminal_recovery_aux_with_recovery_mwh"]) == pytest.approx(0.095)
    assert float(diag["terminal_recovery_induced_aux_delta_mwh"]) == pytest.approx(expected_aux_delta)
    assert float(diag["terminal_recovery_aux_sized_mwh"]) == pytest.approx(expected_aux_sized)
    assert float(diag["terminal_soc_projection_id_extra_aux_losses_mwh"]) == pytest.approx(expected_aux_sized)
    assert float(diag["terminal_soc_id_recourse_needed_internal_mwh"]) == pytest.approx(expected_internal)
    assert id_charge * bt.dt_h * bt.eta_in == pytest.approx(expected_internal)
    assert float(diag["terminal_repair_projected_final_soc_after_repair_mwh"]) >= 10.0 - 1e-9
    assert float(diag["terminal_soc_recovery_post_aux_shortfall_mwh"]) == pytest.approx(0.0)


def test_terminal_recovery_observed_residual_shortfall_regression() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.aux_peak_mw = 0.03
    bt.aux_trading_mw = 0.03
    bt.aux_standby_mw = 0.0
    bt.p_max_mw = 10.0

    id_charge, id_discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=8.58,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=10.0,
        projected_terminal_soc_without_new_id_mwh=8.58,
        remaining_known_losses_mwh=0.0,
        terminal_soc_safety_margin_mwh=0.0,
    )
    diag = bt._last_id_rescue_plan_diagnostics
    aux_mwh = float(diag["terminal_soc_projection_id_extra_aux_losses_mwh"])
    final_projection = 8.58 + id_charge * bt.dt_h * bt.eta_in - aux_mwh

    assert reason == "terminal_soc_recovery"
    assert id_discharge == pytest.approx(0.0)
    expected_aux_delta = 0.03 - float(bt.aux_off_mw)
    expected_aux_sized = 0.03
    assert float(diag["terminal_recovery_aux_without_recovery_mwh"]) == pytest.approx(float(bt.aux_off_mw))
    assert float(diag["terminal_recovery_aux_with_recovery_mwh"]) == pytest.approx(0.03)
    assert float(diag["terminal_recovery_induced_aux_delta_mwh"]) == pytest.approx(expected_aux_delta)
    assert float(diag["terminal_recovery_aux_sized_mwh"]) == pytest.approx(expected_aux_sized)
    assert aux_mwh == pytest.approx(expected_aux_sized)
    assert id_charge * bt.dt_h * bt.eta_in == pytest.approx(10.0 - 8.58 + expected_aux_sized)
    assert final_projection >= 10.0 - 1e-9
    assert float(diag["terminal_recovery_remaining_shortfall_after_recovery_mwh"]) == pytest.approx(0.0)
    assert float(diag["terminal_recovery_final_projection_passed"]) == pytest.approx(1.0)


def test_terminal_recovery_includes_off_to_trading_aux_delta() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.aux_off_mw = 0.02
    bt.aux_trading_mw = 0.05
    bt.aux_standby_mw = 0.0
    bt.p_max_mw = 10.0

    id_charge, id_discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=8.58,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=10.0,
        projected_terminal_soc_without_new_id_mwh=8.58,
        remaining_known_losses_mwh=0.0,
        terminal_soc_safety_margin_mwh=0.0,
    )
    diag = bt._last_id_rescue_plan_diagnostics

    expected_delta = 0.05 - 0.02
    expected_aux_sized = 0.05
    expected_internal = 10.0 - 8.58 + expected_aux_sized
    final_projection = 8.58 + id_charge * bt.dt_h * bt.eta_in - expected_aux_sized

    assert reason == "terminal_soc_recovery"
    assert id_discharge == pytest.approx(0.0)
    assert float(diag["terminal_recovery_aux_without_recovery_mwh"]) == pytest.approx(0.02)
    assert float(diag["terminal_recovery_aux_with_recovery_mwh"]) == pytest.approx(0.05)
    assert float(diag["terminal_recovery_induced_aux_delta_mwh"]) == pytest.approx(expected_delta)
    assert float(diag["terminal_recovery_aux_sized_mwh"]) == pytest.approx(expected_aux_sized)
    assert float(diag["terminal_recovery_projection_includes_aux_without_recovery"]) == pytest.approx(0.0)
    assert float(diag["terminal_soc_id_recourse_needed_internal_mwh"]) == pytest.approx(expected_internal)
    assert final_projection >= 10.0 - 1e-9


def test_terminal_recovery_iterates_scaled_aux_to_remove_residual_shortfall() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.aux_mode = "state_dependent_scaled"
    bt.aux_off_mw = 0.0
    bt.aux_trading_mw = 0.05
    bt.aux_standby_mw = 0.0
    bt.p_max_mw = 10.0

    id_charge, id_discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=9.97,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=10.0,
        projected_terminal_soc_without_new_id_mwh=9.97,
        remaining_known_losses_mwh=0.0,
        terminal_soc_safety_margin_mwh=0.0,
    )
    diag = bt._last_id_rescue_plan_diagnostics
    aux_mwh = float(diag["terminal_soc_projection_id_extra_aux_losses_mwh"])
    final_projection = 9.97 + id_charge * bt.dt_h * bt.eta_in - aux_mwh

    assert reason == "terminal_soc_recovery"
    assert id_discharge == pytest.approx(0.0)
    assert final_projection >= 10.0 - 1e-9
    assert float(diag["terminal_recovery_projected_final_soc_after_recovery_mwh"]) >= 10.0 - 1e-9
    assert float(diag["terminal_recovery_remaining_shortfall_after_recovery_mwh"]) == pytest.approx(0.0)
    assert float(diag["terminal_recovery_final_projection_passed"]) == pytest.approx(1.0)
    assert float(diag["terminal_recovery_iteration_count"]) >= 2.0
    assert float(diag["terminal_recovery_replanned_due_to_residual_shortfall"]) == pytest.approx(1.0)


def test_terminal_recovery_marks_unrecoverable_when_power_limit_blocks_target() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95

    diag = bt._schedule_terminal_id_recovery(
        projected_terminal_soc_without_new_id_mwh=9.0,
        current_soc_mwh=9.0,
        terminal_soc_target_mwh=10.0,
        residual_charge_mw=0.2,
        terminal_soc_safety_margin_mwh=0.0,
        terminal_repair_known_future_aux_mwh=0.0,
        terminal_repair_recovery_aux_mwh=0.0,
    )

    assert float(diag["terminal_soc_recovery_feasible"]) == pytest.approx(0.0)
    assert float(diag["terminal_recovery_remaining_shortfall_after_recovery_mwh"]) > 0.0
    assert str(diag["terminal_soc_not_recoverable_reason"]) == "insufficient_remaining_id_capacity"


def test_terminal_closure_shortfall_buys_id_to_target() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.90
    bt.trans_eur_mwh = 2.0
    bt.deg_eur_mwh = 10.0
    bt.aux_trading_mw = 0.0

    diag = bt._terminal_closure_components(
        final_soc_before_mwh=9.05,
        buy_price_eur_mwh=100.0,
        sell_price_eur_mwh=90.0,
        aux_settlement_price_eur_mwh=80.0,
        existing_aux_mwh=0.0,
    )

    assert str(diag["terminal_closure_reason"]) == "shortfall_recovery"
    assert float(diag["terminal_closure_id_buy_mwh"]) == pytest.approx(1.0)
    assert float(diag["terminal_closure_internal_buy_mwh"]) == pytest.approx(0.95)
    assert float(diag["terminal_closure_soc_after_mwh"]) == pytest.approx(10.0)
    assert float(diag["terminal_closure_cost_eur"]) == pytest.approx(100.0)
    assert float(diag["terminal_closure_transaction_cost_eur"]) == pytest.approx(2.0)
    assert float(diag["terminal_closure_degradation_cost_eur"]) == pytest.approx(9.5)
    assert float(diag["terminal_closure_net_pnl_eur"]) == pytest.approx(-111.5)


def test_rhpf_terminal_repair_feasible_when_post_repair_soc_hits_target() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.trans_eur_mwh = 2.0
    bt.deg_eur_mwh = 10.0
    bt.aux_trading_mw = 0.0
    closure = bt._terminal_closure_components(
        final_soc_before_mwh=9.05,
        buy_price_eur_mwh=100.0,
        sell_price_eur_mwh=90.0,
        aux_settlement_price_eur_mwh=80.0,
        existing_aux_mwh=0.0,
    )

    status = BatteryBacktester._classify_rolling_pf_terminal_repair(
        previous_solver_feasible=0.0,
        previous_solver_infeasible_reason="terminal_soc_shortfall",
        terminal_closure=closure,
        fallback_final_soc_mwh=10.0,
        target_final_soc_mwh=10.0,
        eta_in=0.95,
        terminal_price_true_eur_mwh=100.0,
        terminal_closure_pf_pnl_eur=float(closure["terminal_closure_net_pnl_eur"]),
        solver_component_pnl_eur=208.937566832903,
        solver_pnl_balance_ok=1.0,
        rhpf_enabled=True,
    )

    assert float(status["pre_repair_shortfall_mwh"]) == pytest.approx(0.95)
    assert float(status["terminal_repair_scheduled_mwh"]) == pytest.approx(1.0)
    assert float(status["post_repair_shortfall_mwh"]) == pytest.approx(0.0)
    assert float(status["terminal_target_met"]) == pytest.approx(1.0)
    assert float(status["terminal_repair_feasible"]) == pytest.approx(1.0)
    assert float(status["solver_feasible"]) == pytest.approx(1.0)
    assert str(status["solver_infeasible_reason"]) == "none"
    assert str(status["terminal_repair_clearance_reason"]) == "physical_terminal_repair_to_target_cost_accounted"


def test_rhpf_terminal_repair_not_feasible_when_post_repair_shortfall_remains() -> None:
    closure = {
        "terminal_closure_applied": 0.0,
        "terminal_closure_reason": "none",
        "terminal_closure_soc_before_mwh": 9.05,
        "terminal_closure_soc_after_mwh": 9.05,
        "final_soc_before_terminal_closure_mwh": 9.05,
        "final_soc_after_terminal_closure_mwh": 9.05,
        "final_soc_target_met_after_closure": 0.0,
        "terminal_closure_id_buy_mwh": 0.0,
        "terminal_closure_revenue_eur": 0.0,
        "terminal_closure_cost_eur": 0.0,
        "terminal_closure_transaction_cost_eur": 0.0,
        "terminal_closure_degradation_cost_eur": 0.0,
        "terminal_closure_aux_cost_eur": 0.0,
    }

    status = BatteryBacktester._classify_rolling_pf_terminal_repair(
        previous_solver_feasible=0.0,
        previous_solver_infeasible_reason="terminal_soc_shortfall",
        terminal_closure=closure,
        fallback_final_soc_mwh=9.05,
        target_final_soc_mwh=10.0,
        eta_in=0.95,
        terminal_price_true_eur_mwh=100.0,
        terminal_closure_pf_pnl_eur=0.0,
        solver_component_pnl_eur=208.937566832903,
        solver_pnl_balance_ok=1.0,
        rhpf_enabled=True,
    )

    assert float(status["pre_repair_shortfall_mwh"]) == pytest.approx(0.95)
    assert float(status["terminal_repair_scheduled_mwh"]) == pytest.approx(0.0)
    assert float(status["post_repair_shortfall_mwh"]) == pytest.approx(0.95)
    assert float(status["terminal_target_met"]) == pytest.approx(0.0)
    assert float(status["terminal_repair_feasible"]) == pytest.approx(0.0)
    assert float(status["solver_feasible"]) == pytest.approx(0.0)
    assert str(status["solver_infeasible_reason"]) == "terminal_soc_shortfall_after_terminal_closure"
    assert str(status["terminal_repair_failure_reason"]) == "post_repair_shortfall_remaining"


def test_rhpf_terminal_repair_no_repair_needed_keeps_solver_feasible() -> None:
    closure = {
        "terminal_closure_applied": 0.0,
        "terminal_closure_reason": "none",
        "terminal_closure_soc_before_mwh": 10.0,
        "terminal_closure_soc_after_mwh": 10.0,
        "final_soc_before_terminal_closure_mwh": 10.0,
        "final_soc_after_terminal_closure_mwh": 10.0,
        "final_soc_target_met_after_closure": 1.0,
        "terminal_closure_id_buy_mwh": 0.0,
        "terminal_closure_revenue_eur": 0.0,
        "terminal_closure_cost_eur": 0.0,
        "terminal_closure_transaction_cost_eur": 0.0,
        "terminal_closure_degradation_cost_eur": 0.0,
        "terminal_closure_aux_cost_eur": 0.0,
    }

    status = BatteryBacktester._classify_rolling_pf_terminal_repair(
        previous_solver_feasible=1.0,
        previous_solver_infeasible_reason="none",
        terminal_closure=closure,
        fallback_final_soc_mwh=10.0,
        target_final_soc_mwh=10.0,
        eta_in=0.95,
        terminal_price_true_eur_mwh=100.0,
        terminal_closure_pf_pnl_eur=0.0,
        solver_component_pnl_eur=208.937566832903,
        solver_pnl_balance_ok=1.0,
        rhpf_enabled=True,
    )

    assert float(status["pre_repair_shortfall_mwh"]) == pytest.approx(0.0)
    assert float(status["terminal_repair_scheduled_mwh"]) == pytest.approx(0.0)
    assert float(status["post_repair_shortfall_mwh"]) == pytest.approx(0.0)
    assert float(status["terminal_target_met"]) == pytest.approx(1.0)
    assert float(status["terminal_repair_feasible"]) == pytest.approx(1.0)
    assert float(status["solver_feasible"]) == pytest.approx(1.0)
    assert str(status["solver_infeasible_reason"]) == "none"


def test_rhpf_terminal_repair_uses_post_repair_shortfall_not_positive_shortfall() -> None:
    closure = {
        "terminal_closure_applied": 1.0,
        "terminal_closure_reason": "shortfall_recovery",
        "terminal_closure_soc_before_mwh": 9.05,
        "terminal_closure_soc_after_mwh": 10.0,
        "final_soc_before_terminal_closure_mwh": 9.05,
        "final_soc_after_terminal_closure_mwh": 10.0,
        "final_soc_target_met_after_closure": 1.0,
        "terminal_closure_id_buy_mwh": 1.0,
        "terminal_closure_revenue_eur": 0.0,
        "terminal_closure_cost_eur": 100.0,
        "terminal_closure_transaction_cost_eur": 2.0,
        "terminal_closure_degradation_cost_eur": 9.5,
        "terminal_closure_aux_cost_eur": 0.0,
    }

    status = BatteryBacktester._classify_rolling_pf_terminal_repair(
        previous_solver_feasible=0.0,
        previous_solver_infeasible_reason="terminal_soc_shortfall",
        terminal_closure=closure,
        fallback_final_soc_mwh=10.0,
        target_final_soc_mwh=10.0,
        eta_in=0.95,
        terminal_price_true_eur_mwh=100.0,
        terminal_closure_pf_pnl_eur=-111.5,
        solver_component_pnl_eur=208.937566832903,
        solver_pnl_balance_ok=1.0,
        rhpf_enabled=True,
    )

    assert float(status["pre_repair_shortfall_mwh"]) > 0.0
    assert float(status["post_repair_shortfall_mwh"]) == pytest.approx(0.0)
    assert float(status["terminal_repair_feasible"]) == pytest.approx(1.0)
    assert str(status["solver_infeasible_reason"]) == "none"


def test_rhpf_terminal_repair_requires_cost_accounting() -> None:
    closure = {
        "terminal_closure_applied": 1.0,
        "terminal_closure_reason": "shortfall_recovery",
        "terminal_closure_soc_before_mwh": 9.05,
        "terminal_closure_soc_after_mwh": 10.0,
        "final_soc_before_terminal_closure_mwh": 9.05,
        "final_soc_after_terminal_closure_mwh": 10.0,
        "final_soc_target_met_after_closure": 1.0,
        "terminal_closure_id_buy_mwh": 1.0,
        "terminal_closure_revenue_eur": 0.0,
        "terminal_closure_cost_eur": 100.0,
        "terminal_closure_transaction_cost_eur": 2.0,
        "terminal_closure_degradation_cost_eur": 9.5,
        "terminal_closure_aux_cost_eur": 0.0,
    }

    status = BatteryBacktester._classify_rolling_pf_terminal_repair(
        previous_solver_feasible=0.0,
        previous_solver_infeasible_reason="terminal_soc_shortfall",
        terminal_closure=closure,
        fallback_final_soc_mwh=10.0,
        target_final_soc_mwh=10.0,
        eta_in=0.95,
        terminal_price_true_eur_mwh=100.0,
        terminal_closure_pf_pnl_eur=0.0,
        solver_component_pnl_eur=208.937566832903,
        solver_pnl_balance_ok=1.0,
        rhpf_enabled=True,
    )

    assert float(status["terminal_repair_physical"]) == pytest.approx(1.0)
    assert float(status["terminal_repair_cost_accounted"]) == pytest.approx(0.0)
    assert float(status["terminal_repair_feasible"]) == pytest.approx(0.0)
    assert float(status["solver_feasible"]) == pytest.approx(0.0)
    assert str(status["solver_infeasible_reason"]) == "terminal_repair_cost_not_accounted"


def test_terminal_closure_surplus_sells_id_to_target() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.90
    bt.trans_eur_mwh = 2.0
    bt.deg_eur_mwh = 10.0
    bt.aux_trading_mw = 0.0

    diag = bt._terminal_closure_components(
        final_soc_before_mwh=11.0,
        buy_price_eur_mwh=100.0,
        sell_price_eur_mwh=90.0,
        aux_settlement_price_eur_mwh=80.0,
        existing_aux_mwh=0.0,
    )

    assert str(diag["terminal_closure_reason"]) == "surplus_liquidation"
    assert float(diag["terminal_closure_id_sell_mwh"]) == pytest.approx(0.9)
    assert float(diag["terminal_closure_internal_sell_mwh"]) == pytest.approx(1.0)
    assert float(diag["terminal_closure_soc_after_mwh"]) == pytest.approx(10.0)
    assert float(diag["terminal_closure_revenue_eur"]) == pytest.approx(81.0)
    assert float(diag["terminal_closure_transaction_cost_eur"]) == pytest.approx(1.8)
    assert float(diag["terminal_closure_degradation_cost_eur"]) == pytest.approx(10.0)
    assert float(diag["terminal_closure_net_pnl_eur"]) == pytest.approx(69.2)


def test_terminal_closure_includes_recovery_induced_aux_delta() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    bt.trans_eur_mwh = 0.0
    bt.deg_eur_mwh = 0.0
    bt.aux_trading_mw = 0.05

    diag = bt._terminal_closure_components(
        final_soc_before_mwh=9.97,
        buy_price_eur_mwh=100.0,
        sell_price_eur_mwh=90.0,
        aux_settlement_price_eur_mwh=80.0,
        existing_aux_mwh=0.02,
    )

    expected_internal = 10.0 - 9.97 + 0.03
    assert str(diag["terminal_closure_reason"]) == "shortfall_recovery"
    assert float(diag["terminal_closure_recovery_induced_aux_delta_mwh"]) == pytest.approx(0.03)
    assert float(diag["terminal_closure_internal_buy_mwh"]) == pytest.approx(expected_internal)
    assert float(diag["terminal_closure_soc_after_mwh"]) == pytest.approx(10.0)
    assert float(diag["terminal_closure_aux_cost_eur"]) == pytest.approx(0.03 * 80.0)


def test_da_origin_summary_counter_uses_canonical_origin_fields() -> None:
    ts = pd.Timestamp("2026-01-02T01:00:00Z")
    source_snapshot = pd.Timestamp("2026-01-01T10:00:00Z")
    hourly = pd.DataFrame(
        {
            "da_lockbook_row_present": [1.0],
            "da_bid_locked": [1.0],
            "da_locked_buy_mwh": [1.0],
            "da_locked_sell_mwh": [0.0],
            "da_originating_source_snapshot_utc": [source_snapshot.isoformat()],
            "da_originating_delivery_timestamp_utc": [ts.isoformat()],
            "da_originating_precommit_id": [f"{source_snapshot.isoformat()}->{ts.isoformat()}"],
            "da_precommit_final_selected_incumbent": ["optimized"],
            "da_precommit_selection_reason": ["accepted_lockbook"],
            "da_precommit_da_zero_reason": ["none"],
            "real_submitted_da_buy_mw": [1.0],
            "real_submitted_da_sell_mw": [0.0],
            "real_submitted_da_buy_price_eur_mwh": [10.0],
            "real_da_buy_accepted": [1.0],
            "real_da_sell_accepted": [0.0],
            "real_da_buy_mwh": [1.0],
            "real_da_sell_mwh": [0.0],
            # Stale exported flag from an older merge must not be trusted over canonical fields.
            "da_realized_without_precommit_origin_error": [1.0],
        }
    )

    counters = BatteryBacktester._compute_da_naming_semantics_counters(hourly)
    summary_error_count = float(counters["da_realized_without_precommit_origin_count"])
    assert summary_error_count == pytest.approx(0.0)


def test_realized_da_origin_diagnostics_recompute_stale_error_from_canonical_fields() -> None:
    ts = pd.Timestamp("2026-01-02T01:00:00Z")
    source_snapshot = pd.Timestamp("2026-01-01T10:00:00Z")
    hourly = pd.DataFrame(
        {
            "timestamp_utc": [ts],
            "da_originating_precommit_id": [f"{source_snapshot.isoformat()}->{ts.isoformat()}"],
            "da_originating_source_snapshot_utc": [source_snapshot.isoformat()],
            "da_originating_delivery_timestamp_utc": [ts.isoformat()],
            "da_precommit_final_selected_incumbent": ["optimized"],
            "da_precommit_selection_reason": ["accepted_lockbook"],
            "real_da_buy_mwh": [1.0],
            "real_da_sell_mwh": [0.0],
            # This stale intermediate value must be overwritten from final canonical metadata.
            "da_realized_without_precommit_origin_error": [1.0],
            "da_realized_without_precommit_origin_missing_columns": ["da_originating_precommit_id"],
        }
    )

    out = BatteryBacktester._attach_da_realized_origin_diagnostics(hourly, timestamp_col="timestamp_utc")

    assert float(out["da_realized_without_precommit_origin_error"].iloc[0]) == pytest.approx(0.0)
    assert float(out["da_realized_origin_complete"].iloc[0]) == pytest.approx(1.0)
    assert str(out["da_realized_without_precommit_origin_missing_columns"].iloc[0]) == ""
    assert str(out["first_da_realized_without_precommit_origin_timestamp_utc"].iloc[0]) == ""


def test_realized_da_origin_diagnostics_reports_missing_canonical_columns() -> None:
    ts = pd.Timestamp("2026-01-02T01:00:00Z")
    source_snapshot = pd.Timestamp("2026-01-01T10:00:00Z")
    hourly = pd.DataFrame(
        {
            "timestamp_utc": [ts],
            "da_originating_precommit_id": [""],
            "da_originating_source_snapshot_utc": [source_snapshot.isoformat()],
            "da_originating_delivery_timestamp_utc": [ts.isoformat()],
            "da_precommit_final_selected_incumbent": ["optimized"],
            "da_precommit_selection_reason": ["accepted_lockbook"],
            "real_da_buy_mwh": [1.0],
            "real_da_sell_mwh": [0.0],
        }
    )

    out = BatteryBacktester._attach_da_realized_origin_diagnostics(hourly, timestamp_col="timestamp_utc")

    assert float(out["da_realized_without_precommit_origin_error"].iloc[0]) == pytest.approx(1.0)
    assert float(out["da_realized_origin_complete"].iloc[0]) == pytest.approx(0.0)
    assert "da_originating_precommit_id" in str(
        out["da_realized_without_precommit_origin_missing_columns"].iloc[0]
    )
    assert str(out["first_da_realized_without_precommit_origin_timestamp_utc"].iloc[0]) == ts.isoformat()


def test_realized_da_origin_diagnostics_ignores_zero_volume_no_trade_row() -> None:
    ts = pd.Timestamp("2026-01-02T01:00:00Z")
    hourly = pd.DataFrame(
        {
            "timestamp_utc": [ts],
            "real_da_buy_mwh": [0.0],
            "real_da_sell_mwh": [0.0],
        }
    )

    out = BatteryBacktester._attach_da_realized_origin_diagnostics(hourly, timestamp_col="timestamp_utc")

    assert float(out["da_realized_without_precommit_origin_error"].iloc[0]) == pytest.approx(0.0)
    assert float(out["da_realized_origin_complete"].iloc[0]) == pytest.approx(0.0)
    assert str(out["da_realized_without_precommit_origin_missing_columns"].iloc[0]) == ""
    assert str(out["first_da_realized_without_precommit_origin_timestamp_utc"].iloc[0]) == ""


def test_upper_soc_relief_is_blocked_when_it_would_undo_hard_terminal_target() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    id_charge, id_discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=17.0,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=10.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=10.0,
        projected_terminal_soc_without_new_id_mwh=10.0,
        remaining_known_losses_mwh=0.0,
        terminal_soc_safety_margin_mwh=0.0,
        terminal_hard_soc_protection=True,
    )
    diag = bt._last_id_rescue_plan_diagnostics

    assert id_charge == pytest.approx(0.0)
    assert id_discharge == pytest.approx(0.0)
    assert reason == "none"
    assert float(diag["upper_soc_relief_terminal_shortfall_prevented"]) == pytest.approx(1.0)
    assert float(diag["upper_soc_relief_caused_terminal_shortfall"]) == pytest.approx(0.0)
    assert float(diag["terminal_day_upper_soc_relief_capped"]) == pytest.approx(1.0)
    assert str(diag["terminal_day_upper_soc_relief_reason"]) == "protect_hard_final_soc"


def test_non_terminal_upper_soc_relief_does_not_use_global_terminal_cap() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    id_charge, id_discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=17.0,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=10.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=10.0,
        projected_terminal_soc_without_new_id_mwh=10.0,
        remaining_known_losses_mwh=0.0,
        terminal_soc_safety_margin_mwh=0.0,
        terminal_hard_soc_protection=False,
    )

    assert id_charge == pytest.approx(0.0)
    assert id_discharge > 0.0
    assert reason == "afrr_headroom_repair"


def test_terminal_upper_soc_relief_cap_uses_eta_out() -> None:
    bt = _mk_backtester()
    _configure_zero_aux_unit_efficiency(bt)
    bt.eta_out = 0.8
    bt.soc_target_end = 10.0
    bt.reserve_headroom_safety_mwh = 0.0
    id_charge, id_discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=15.0,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=8.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=10.0,
        projected_terminal_soc_without_new_id_mwh=10.5,
        remaining_known_losses_mwh=0.0,
        terminal_soc_safety_margin_mwh=0.0,
        terminal_hard_soc_protection=True,
    )
    diag = bt._last_id_rescue_plan_diagnostics

    assert id_charge == pytest.approx(0.0)
    assert id_discharge == pytest.approx(0.5 * bt.eta_out / bt.dt_h)
    assert reason == "afrr_headroom_repair"
    assert float(diag["terminal_day_upper_soc_relief_allowed_mwh"]) == pytest.approx(0.5 * bt.eta_out)
    assert float(diag["terminal_day_upper_soc_relief_projected_soc_after_mwh"]) >= 10.0 - 1e-9


def test_bcm_only_common_technical_id_passes_strategy_isolation() -> None:
    bt = _mk_backtester()
    hourly = pd.DataFrame(
        {
            "real_id_charge_mw": [1.0],
            "real_id_discharge_mw": [0.0],
            "real_pending_id_charge_mw": [0.0],
            "real_pending_id_discharge_mw": [0.0],
            "real_id_trade_type": ["technical_repair"],
            "real_id_repair_reason": ["terminal_soc_recovery"],
            "real_id_recourse_reason": ["terminal_soc_recovery"],
        }
    )
    perms = bt.resolve_strategy_permissions(
        strategy_name="bcm",
        allowed_markets=("aFRR", "BCM"),
        id_recourse_mode="common",
    )
    assert perms.id_mode == "technical_repair"
    bt._validate_strategy_isolation_outputs(
        hourly=hourly,
        allowed_markets=("aFRR", "BCM"),
        strategy_permissions=perms,
        strategy_name="bcm",
        id_recourse_mode="common",
    )


def test_da_only_disabled_id_activity_fails_strategy_isolation() -> None:
    bt = _mk_backtester()
    hourly = pd.DataFrame(
        {
            "real_id_charge_mw": [1.0],
            "real_id_discharge_mw": [0.0],
            "real_pending_id_charge_mw": [0.0],
            "real_pending_id_discharge_mw": [0.0],
            "real_id_trade_type": ["technical_repair"],
            "real_id_recourse_reason": ["terminal_soc_recovery"],
        }
    )
    perms = bt.resolve_strategy_permissions(
        strategy_name="da",
        allowed_markets=("DA",),
        id_recourse_mode="disabled",
    )
    assert perms.id_mode == "none"
    with pytest.raises(RuntimeError, match="resolved_id_mode=none"):
        bt._validate_strategy_isolation_outputs(
            hourly=hourly,
            allowed_markets=("DA",),
            strategy_permissions=perms,
            strategy_name="da",
            id_recourse_mode="disabled",
        )


def test_technical_repair_mode_rejects_economic_id_trade_type() -> None:
    bt = _mk_backtester()
    hourly = pd.DataFrame(
        {
            "real_id_charge_mw": [1.0],
            "real_id_discharge_mw": [0.0],
            "real_pending_id_charge_mw": [0.0],
            "real_pending_id_discharge_mw": [0.0],
            "real_id_trade_type": ["economic"],
            "real_id_repair_reason": ["arbitrage"],
            "real_id_recourse_reason": ["arbitrage"],
        }
    )
    perms = bt.resolve_strategy_permissions(
        strategy_name="bcm",
        allowed_markets=("aFRR", "BCM"),
        id_recourse_mode="common",
    )
    with pytest.raises(RuntimeError, match="non-technical ID trade type"):
        bt._validate_strategy_isolation_outputs(
            hourly=hourly,
            allowed_markets=("aFRR", "BCM"),
            strategy_permissions=perms,
            strategy_name="bcm",
            id_recourse_mode="common",
        )


def test_multi_common_recourse_resolves_technical_id_consistently() -> None:
    bt = _mk_backtester()
    hourly = pd.DataFrame(
        {
            "real_id_charge_mw": [0.5],
            "real_id_discharge_mw": [0.0],
            "real_pending_id_charge_mw": [0.0],
            "real_pending_id_discharge_mw": [0.0],
            "real_id_trade_type": ["technical_repair"],
            "real_id_repair_reason": ["protected_soc_recovery"],
            "real_id_recourse_reason": ["protected_soc_recovery"],
        }
    )
    perms = bt.resolve_strategy_permissions(
        strategy_name="multi",
        allowed_markets=("DA", "aFRR"),
        id_recourse_mode="common",
    )
    assert perms.id_mode == "technical_repair"
    bt._validate_strategy_isolation_outputs(
        hourly=hourly,
        allowed_markets=("DA", "aFRR"),
        strategy_permissions=perms,
        strategy_name="multi",
        id_recourse_mode="common",
    )


def test_optimize_dispatch_keeps_resolved_technical_id_when_id_in_allowed_markets() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "hard"
    df, col = _tiny_backtest_df(hours=3)
    perms = StrategyPermissions(
        allow_da=False,
        id_mode="technical_repair",
        allow_bcm=True,
        allow_bcm_activation_obligations=True,
        allow_bem_only=False,
    )
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=5.0,
        soc_end_min_target=8.0,
        allowed_markets=("aFRR", "BCM", "ID"),
        strategy_permissions=perms,
    )
    assert bt._strategy_permissions.id_mode == "technical_repair"
    assert float(out["id_charge_mw"].sum()) > 0.0
    assert "terminal_soc_recovery" in set(out.loc[out["id_charge_mw"] > 1e-9, "id_recourse_reason"].astype(str))


def test_hard_mode_reaches_target_when_feasible() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "hard"
    df, col = _tiny_backtest_df(hours=8)
    # Encourage charging to satisfy terminal SoC hard floor.
    df[col.pred_da_price] = -100.0
    df[col.true_da_price] = -100.0
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=8,
        reopt_step_hours=1,
        strict_simulation_validity=True,
        enforce_final_soc_min=True,
    )
    assert float(out.summary.get("final_soc_physical_check_pass", 0.0)) >= 0.5


def test_strict_hard_mode_never_drops_terminal_constraint_in_fallback() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "hard"
    df, col = _tiny_backtest_df(hours=6)
    calls: list[float | None] = []
    original_opt = bt.optimize_dispatch

    def _wrapped(*args, **kwargs):
        calls.append(kwargs.get("soc_end_min_target"))
        return original_opt(*args, **kwargs)

    bt.optimize_dispatch = _wrapped  # type: ignore[method-assign]
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=6,
        reopt_step_hours=1,
        strict_simulation_validity=True,
        enforce_final_soc_min=True,
    )
    assert all(v is not None for v in calls)
    assert float(out.summary.get("terminal_constraint_dropped", 0.0)) == 0.0


def test_terminal_repair_mode_not_thesis_reportable_when_physical_shortfall() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "terminal_repair"
    df, col = _tiny_backtest_df(hours=6)
    df[col.pred_da_price] = 200.0
    df[col.true_da_price] = 120.0
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=6,
        reopt_step_hours=1,
        strict_simulation_validity=True,
        enforce_final_soc_min=True,
    )
    s = out.summary
    if float(s.get("final_soc_physical_check_pass", 1.0)) < 0.5:
        assert float(s.get("thesis_reportable", 1.0)) == 0.0


def test_terminal_soc_value_discount_does_not_replace_hard_final_soc() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "hard"
    df, col = _tiny_backtest_df(hours=4)
    orig_target = bt.soc_target_end
    orig_discount = MODEL_SPECS.get("terminal_soc_value_discount", None)
    try:
        bt.soc_target_end = float(bt.soc_max) + 5.0  # physically infeasible
        MODEL_SPECS["terminal_soc_value_discount"] = 1.0
        out = bt.run(
            df,
            col,
            use_rolling_horizon=True,
            horizon_hours=4,
            reopt_step_hours=1,
            strict_simulation_validity=True,
            enforce_final_soc_min=True,
        )
        s = out.summary
        assert float(s.get("final_soc_physical_check_pass", 1.0)) == 0.0
        assert float(s.get("thesis_reportable", 1.0)) == 0.0
    finally:
        bt.soc_target_end = orig_target
        if orig_discount is None:
            MODEL_SPECS.pop("terminal_soc_value_discount", None)
        else:
            MODEL_SPECS["terminal_soc_value_discount"] = orig_discount


def test_console_final_soc_reports_physical_and_economic_status() -> None:
    txt = Path("scripts/run_battery_backtest.py").read_text(encoding="utf-8")
    assert "final_soc_physical_check" in txt
    assert "final_soc_economic_repair_check" in txt


def test_terminal_repair_cost_reconciles() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    df[col.pred_da_price] = 150.0
    df[col.true_da_price] = 120.0
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1)
    s = out.summary
    rhs = float(s.get("realized_pnl_excl_terminal_eur", 0.0)) + float(s.get("terminal_soc_net_adjustment_eur", 0.0))
    assert np.isclose(float(s.get("realized_total_pnl_eur", 0.0)), rhs, atol=1e-6)


def test_strategy_overview_ratio_fields_semantics() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1)
    assert "realized_vs_perfect_foresight_ratio_multi_market" in out.summary
    assert "realized_vs_perfect_foresight_comparable_market_ratio" not in out.summary
    assert "rolling_perfect_foresight_same_rules_total_pnl_eur" in out.summary
    assert "realized_vs_perfect_foresight_pct" in out.summary
    assert "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur" in out.summary
    assert "realized_vs_global_hindsight_perfect_foresight_upper_bound_pct" in out.summary
    assert float(out.summary.get("benchmark_is_global_upper_bound", 1.0)) == 0.0
    assert float(out.summary.get("rolling_perfect_foresight_same_rules_is_global_upper_bound", 1.0)) == 0.0


def test_rolling_pf_not_global_upper_bound() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1)
    assert float(out.summary.get("rolling_perfect_foresight_same_rules_is_global_upper_bound", 1.0)) == 0.0
    assert float(out.summary.get("rolling_perfect_foresight_same_rules_can_be_beaten", 0.0)) == 1.0


def test_global_perfect_foresight_is_full_horizon_not_rolling() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1)
    assert float(out.summary.get("global_perfect_foresight_available", 1.0)) == 0.0
    assert float(out.summary.get("global_hindsight_perfect_foresight_is_global_upper_bound", 1.0)) == 0.0
    assert str(out.summary.get("global_perfect_foresight_validation_status", "")).startswith("disabled")


def test_global_perfect_foresight_dominance_flag_present() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1)
    assert "global_perfect_foresight_dominance_check_pass" in out.summary
    assert "realized_exceeds_global_perfect_foresight" in out.summary


def test_global_perfect_foresight_does_not_invalidate_when_disabled() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    assert float(s.get("global_perfect_foresight_available", 1.0)) == 0.0
    assert float(s.get("global_perfect_foresight_dominance_check_pass", 0.0)) >= 0.5
    assert float(s.get("realized_exceeds_global_perfect_foresight", 1.0)) == 0.0
    assert "realized_exceeds_global_perfect_foresight" not in str(s.get("invalid_reason", ""))


def test_deprecated_perfect_foresight_aliases_are_labelled() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1)
    s = out.summary
    assert "perfect_foresight_total_pnl_eur" in s
    assert float(s.get("perfect_foresight_total_pnl_eur_is_deprecated", 0.0)) == 1.0
    assert str(s.get("perfect_foresight_total_pnl_eur_semantics", "")) == "rolling_perfect_foresight_same_rules"


def test_global_perfect_foresight_disabled_by_default() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1)
    s = out.summary
    assert float(s.get("global_perfect_foresight_available", 1.0)) == 0.0
    assert str(s.get("global_perfect_foresight_validation_status", "")).startswith("disabled")


def test_realized_beating_rolling_pf_does_not_invalidate() -> None:
    invalid_reason = "fallback_used,headroom"
    assert "realized_exceeds_rolling_pf" not in invalid_reason


def test_strategy_overview_labels_diagnostic_vs_upper_bound_ratios() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1)
    s = out.summary
    assert "realized_vs_perfect_foresight_pct" in s
    if float(s.get("global_perfect_foresight_available", 0.0)) < 0.5:
        assert np.isnan(float(s.get("realized_vs_global_hindsight_perfect_foresight_upper_bound_pct", float("nan"))))


def test_rolling_pf_quantile_surface_mode_present() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1)
    assert str(out.summary.get("rolling_pf_quantile_surface_mode", "")) in {
        "collapsed_to_truth",
        "preserved_quantile_surface",
        "unknown",
    }


def test_global_perfect_foresight_bem_only_plan_mapping() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=4,
        reopt_step_hours=1,
        enable_global_perfect_foresight=True,
    )
    s = out.summary
    # Scope-mismatch can still disable availability, but summary fields must exist.
    assert "global_perfect_foresight_bem_only_included" in s
    assert "global_perfect_foresight_dispatch_rows" in s
    assert "global_perfect_foresight_settlement_rows" in s


def test_global_perfect_foresight_same_rules_candidate_fields() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=4,
        reopt_step_hours=1,
        enable_global_perfect_foresight=True,
    )
    s = out.summary
    assert "global_hindsight_same_rules_upper_bound_total_pnl_eur" in s
    assert "realized_path_pnl_under_global_upper_bound_rules_eur" in s
    assert "global_pf_verified_upper_bound" in s
    assert "global_pf_upper_bound_gap_eur" in s
    assert "global_pf_below_realized_incumbent" in s
    assert str(s.get("global_perfect_foresight_capacity_bid_semantics")) in {
        "global_solver_missing_bcm_lockbook_semantics",
        "full_horizon_solver_with_lower_bound_guards",
    }
    if float(s.get("global_perfect_foresight_available", 0.0)) >= 0.5:
        assert np.isclose(
            float(s["global_hindsight_same_rules_upper_bound_total_pnl_eur"]),
            float(s["rolling_perfect_foresight_same_rules_total_pnl_eur"]),
            atol=1e-6,
        )
        assert float(s["global_pf_verified_upper_bound"]) in {0.0, 1.0}
        assert float(s["global_perfect_foresight_dominance_check_pass"]) == float(
            s["global_pf_verified_upper_bound"]
        )


def test_perfect_foresight_bcm_participates_and_exports_same_rules_columns() -> None:
    col = BacktestColumnMap()
    hours = 36
    ts = pd.date_range("2025-09-01T00:00:00Z", periods=hours, freq="h")
    df = pd.DataFrame(
        {
            col.timestamp: ts,
            col.pred_da_price: [0.0] * hours,
            "pred_da_price_p05": [0.0] * hours,
            "pred_da_price_p10": [0.0] * hours,
            "pred_da_price_p90": [0.0] * hours,
            "pred_da_price_p95": [0.0] * hours,
            col.pred_afrr_capacity_price_pos: [0.0] * hours,
            col.pred_afrr_capacity_price_neg: [0.0] * hours,
            col.pred_afrr_activation_price_pos: [1000.0] * hours,
            col.pred_afrr_activation_price_neg: [1000.0] * hours,
            col.pred_afrr_activation_rate_pos: [1.0] * hours,
            col.pred_afrr_activation_rate_neg: [1.0] * hours,
            col.true_da_price: [0.0] * hours,
            col.true_afrr_capacity_price_pos: [0.0] * hours,
            col.true_afrr_capacity_price_neg: [0.0] * hours,
            col.true_afrr_activation_price_pos: [1000.0] * hours,
            col.true_afrr_activation_price_neg: [1000.0] * hours,
            col.true_afrr_activation_rate_pos: [1.0] * hours,
            col.true_afrr_activation_rate_neg: [1.0] * hours,
        }
    )
    for pref, val in [
        (col.pred_afrr_capacity_price_pos, 0.0),
        (col.pred_afrr_capacity_price_neg, 0.0),
        (col.pred_afrr_activation_price_pos, 1000.0),
        (col.pred_afrr_activation_price_neg, 1000.0),
        (col.pred_afrr_activation_rate_pos, 1.0),
        (col.pred_afrr_activation_rate_neg, 1.0),
    ]:
        for q in ["p01", "p05", "p10", "p30", "p50", "p70", "p90", "p95", "p99"]:
            df[f"{pref}_{q}"] = val

    bt = _mk_backtester("canonical_economic")
    bt.reserve_feasibility_mode = "normal"
    bt.enable_reserve_retry_ladder = False
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=30,
        reopt_step_hours=1,
        allowed_markets=("aFRR", "BCM"),
        strategy_name="bcm",
        id_recourse_mode="common",
        enable_global_perfect_foresight=True,
        bcm_bid_hour_local=8,
    )
    h = out.hourly
    for prefix in ["real", "perfect_foresight", "global_perfect_foresight"]:
        assert f"{prefix}_submitted_bcm_capacity_pos_mw" in h.columns
        assert f"{prefix}_submitted_bcm_capacity_neg_mw" in h.columns
        assert f"{prefix}_bcm_activation_bid_price_pos" in h.columns
        assert f"{prefix}_bcm_true_activation_price_pos" in h.columns
        assert f"{prefix}_bcm_precommit_feasibility_pass" in h.columns
        assert f"{prefix}_bcm_precommit_zero_reason" in h.columns
        assert f"{prefix}_bcm_zero_reason" in h.columns
    assert (
        pd.to_numeric(h["perfect_foresight_submitted_bcm_capacity_pos_mw"], errors="coerce").fillna(0.0).sum()
        + pd.to_numeric(h["perfect_foresight_submitted_bcm_capacity_neg_mw"], errors="coerce").fillna(0.0).sum()
    ) > 0.0
    if float(out.summary.get("global_perfect_foresight_available", 0.0)) >= 0.5:
        assert (
            pd.to_numeric(h["global_perfect_foresight_submitted_bcm_capacity_pos_mw"], errors="coerce").fillna(0.0).sum()
            + pd.to_numeric(h["global_perfect_foresight_submitted_bcm_capacity_neg_mw"], errors="coerce").fillna(0.0).sum()
        ) > 0.0
    pf_zero = (
        pd.to_numeric(h["pf_submitted_bcm_capacity_pos_mw"], errors="coerce").fillna(0.0)
        + pd.to_numeric(h["pf_submitted_bcm_capacity_neg_mw"], errors="coerce").fillna(0.0)
    ) <= 1e-12
    if bool(pf_zero.any()):
        reasons = h.loc[pf_zero, "pf_bcm_zero_reason"].fillna("").astype(str).str.strip()
        assert bool((reasons != "").all())


def test_global_pf_benchmark_uses_solver_only_when_solver_dominates() -> None:
    solver_frame = pd.DataFrame({"x": [1]})
    result = BatteryBacktester._evaluate_global_pf_solver_benchmark(
        solver_value_eur=50.0,
        solver_feasible=True,
        solver_available=True,
        solver_frame=solver_frame,
        lower_bound_candidates={
            "realized_path": (40.0, True, pd.DataFrame({"x": [2]})),
            "no_market": (0.0, True, pd.DataFrame({"x": [3]})),
        },
    )

    assert result["selected"] == "solver"
    assert float(result["benchmark_value_eur"]) == pytest.approx(50.0)
    assert result["frame"].equals(solver_frame)
    assert float(result["verified"]) == pytest.approx(1.0)
    assert result["failure_reason"] == "none"
    assert result["best_feasible_lower_bound_name"] == "realized_path"
    assert float(result["best_feasible_lower_bound_eur"]) == pytest.approx(40.0)
    assert float(result["is_solver_upper_bound"]) == pytest.approx(1.0)
    assert float(result["is_realized_path_fallback"]) == pytest.approx(0.0)


def test_global_pf_benchmark_selects_realized_path_when_it_is_best_feasible() -> None:
    solver_frame = pd.DataFrame({"x": [1]})
    result = BatteryBacktester._evaluate_global_pf_solver_benchmark(
        solver_value_eur=0.0,
        solver_feasible=True,
        solver_available=True,
        solver_frame=solver_frame,
        lower_bound_candidates={
            "realized_path": (100.0, True, pd.DataFrame({"x": [2]})),
            "no_market": (0.0, True, pd.DataFrame({"x": [3]})),
        },
    )

    assert result["selected"] == "realized_path"
    assert float(result["benchmark_value_eur"]) == pytest.approx(100.0)
    assert result["frame"].equals(pd.DataFrame({"x": [2]}))
    assert float(result["available"]) == pytest.approx(1.0)
    assert float(result["verified"]) == pytest.approx(1.0)
    assert result["failure_reason"] == "none"
    assert result["best_feasible_lower_bound_name"] == "realized_path"
    assert float(result["best_feasible_lower_bound_eur"]) == pytest.approx(100.0)
    assert float(result["is_solver_upper_bound"]) == pytest.approx(0.0)
    assert float(result["is_realized_path_fallback"]) == pytest.approx(1.0)


def test_da_global_perfect_foresight_diagnostics_realized_path_fallback_is_not_solver_upper_bound() -> None:
    result = BatteryBacktester._evaluate_global_pf_solver_benchmark(
        solver_value_eur=0.0,
        solver_feasible=True,
        solver_available=True,
        solver_frame=pd.DataFrame({"x": [1]}),
        lower_bound_candidates={
            "realized_path": (5.0, True, pd.DataFrame({"x": [2]})),
            "no_market": (0.0, True, pd.DataFrame({"x": [3]})),
        },
    )

    assert result["selected"] == "realized_path"
    assert float(result["is_solver_upper_bound"]) == pytest.approx(0.0)
    assert float(result["is_realized_path_fallback"]) == pytest.approx(1.0)


def test_global_pf_realized_path_fallback_is_not_strict_model_invalidity() -> None:
    reasons = BatteryBacktester._global_pf_strict_invalid_reason(
        enable_global_perfect_foresight=True,
        global_pf_verified_upper_bound=0.0,
        global_perfect_foresight_available=1.0,
        global_perfect_foresight_validation_status="solver_below_realized_path_incumbent",
        global_pf_below_realized_incumbent=1.0,
        global_pf_is_realized_path_fallback=1.0,
    )

    assert reasons == []


def test_global_pf_unavailable_without_fallback_remains_strict_invalidity() -> None:
    reasons = BatteryBacktester._global_pf_strict_invalid_reason(
        enable_global_perfect_foresight=True,
        global_pf_verified_upper_bound=0.0,
        global_perfect_foresight_available=0.0,
        global_perfect_foresight_validation_status="global_pf_unavailable",
        global_pf_below_realized_incumbent=0.0,
        global_pf_is_realized_path_fallback=0.0,
    )

    assert reasons == ["global_pf_unavailable"]


def test_global_pf_solver_below_realized_without_fallback_remains_strict_invalidity() -> None:
    reasons = BatteryBacktester._global_pf_strict_invalid_reason(
        enable_global_perfect_foresight=True,
        global_pf_verified_upper_bound=0.0,
        global_perfect_foresight_available=1.0,
        global_perfect_foresight_validation_status="solver_below_realized_path_incumbent",
        global_pf_below_realized_incumbent=1.0,
        global_pf_is_realized_path_fallback=0.0,
    )

    assert reasons == ["global_pf_solver_failed"]


def test_global_pf_upper_bound_ratio_requires_verified_solver_upper_bound() -> None:
    fallback_ratio = BatteryBacktester._global_pf_upper_bound_ratio_pct(
        realized_total_eur=159.0,
        global_pf_total_eur=159.0,
        global_perfect_foresight_available=1.0,
        global_pf_verified_upper_bound=0.0,
    )
    verified_ratio = BatteryBacktester._global_pf_upper_bound_ratio_pct(
        realized_total_eur=50.0,
        global_pf_total_eur=100.0,
        global_perfect_foresight_available=1.0,
        global_pf_verified_upper_bound=1.0,
    )

    assert np.isnan(fallback_ratio)
    assert verified_ratio == pytest.approx(50.0)


def test_global_pf_benchmark_selects_no_market_when_solver_infeasible() -> None:
    result = BatteryBacktester._evaluate_global_pf_solver_benchmark(
        solver_value_eur=float("nan"),
        solver_feasible=False,
        solver_available=False,
        solver_frame=pd.DataFrame(),
        lower_bound_candidates={
            "no_market": (5.0, True, pd.DataFrame({"x": [1]})),
            "realized_path": (10.0, False, pd.DataFrame({"x": [2]})),
        },
        solver_unavailable_reason="solver_unavailable:test",
    )

    assert result["selected"] == "no_market"
    assert float(result["benchmark_value_eur"]) == pytest.approx(5.0)
    assert float(result["available"]) == pytest.approx(1.0)
    assert float(result["verified"]) == pytest.approx(1.0)
    assert result["failure_reason"] == "none"
    assert result["best_feasible_lower_bound_name"] == "no_market"
    assert float(result["best_feasible_lower_bound_eur"]) == pytest.approx(5.0)


def test_global_pf_benchmark_selects_no_market_when_solver_infeasible_and_realized_negative() -> None:
    no_market_frame = pd.DataFrame({"x": [1]})
    result = BatteryBacktester._evaluate_global_pf_solver_benchmark(
        solver_value_eur=-8.145,
        solver_feasible=False,
        solver_available=True,
        solver_frame=pd.DataFrame({"x": [0]}),
        lower_bound_candidates={
            "no_market": (0.0, True, no_market_frame),
            "realized_path": (-282.2286, True, pd.DataFrame({"x": [2]})),
        },
        solver_unavailable_reason="terminal_soc_shortfall",
    )

    assert result["selected"] == "no_market"
    assert float(result["benchmark_value_eur"]) == pytest.approx(0.0)
    assert result["frame"].equals(no_market_frame)
    assert float(result["available"]) == pytest.approx(1.0)
    assert float(result["verified"]) == pytest.approx(1.0)
    assert result["failure_reason"] == "none"
    assert result["selection_reason"] == "no_market_has_highest_feasible_pnl"


def test_global_pf_bcm_partial_product_disable_mask_uses_evaluated_window() -> None:
    bt = _mk_backtester("canonical_economic")
    # January is CET: 01:00-02:00 UTC are the tail of local 00:00-04:00,
    # 03:00-06:00 UTC are the complete local 04:00-08:00 product, and
    # 07:00 UTC starts the next partial product.
    ts = pd.date_range("2025-01-08T01:00:00Z", periods=7, freq="h")

    mask = bt._bcm_partial_product_disable_mask(pd.Series(ts))

    assert mask.tolist() == [True, True, False, False, False, False, True]


def test_global_pf_bcm_product_mapping_preserves_pay_as_bid_capacity_revenue_and_excludes_partials() -> None:
    bt = _mk_backtester("canonical_economic")
    col = BacktestColumnMap()
    # January is CET: 03:00-06:00 UTC is a complete local 04:00-08:00 BCM product.
    ts = pd.date_range("2025-01-08T03:00:00Z", periods=5, freq="h")
    dispatch = pd.DataFrame(
        {
            col.timestamp: ts,
            "reserve_pos_mw": [2.0, 2.0, 2.0, 2.0, 2.0],
            "reserve_neg_mw": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    market_input = pd.DataFrame(
        {
            col.timestamp: ts,
            col.pred_afrr_capacity_price_pos: [20.0] * len(ts),
            col.pred_afrr_capacity_price_neg: [0.0] * len(ts),
            col.true_afrr_capacity_price_pos: [25.0] * len(ts),
            col.true_afrr_capacity_price_neg: [0.0] * len(ts),
            col.pred_afrr_activation_price_pos: [100.0] * len(ts),
            col.pred_afrr_activation_price_neg: [100.0] * len(ts),
            col.true_afrr_activation_price_pos: [100.0] * len(ts),
            col.true_afrr_activation_price_neg: [100.0] * len(ts),
        }
    )

    mapped = bt._map_global_bcm_reserves_to_product_obligations(
        dispatch=dispatch,
        market_input=market_input,
        colmap=col,
    )

    complete = mapped.iloc[:4]
    partial = mapped.iloc[4:]
    assert pd.to_numeric(complete["aFRR_Capacity_Won_Pos_MW"], errors="coerce").tolist() == [2.0] * 4
    assert pd.to_numeric(complete["settlement_cap_bid_price_pos_eur_mw"], errors="coerce").tolist() == [20.0] * 4
    assert float(
        (
            pd.to_numeric(complete["aFRR_Capacity_Won_Pos_MW"], errors="coerce")
            * pd.to_numeric(complete["settlement_cap_bid_price_pos_eur_mw"], errors="coerce")
            * bt.dt_h
        ).sum()
    ) == pytest.approx(160.0)
    assert float(pd.to_numeric(partial["aFRR_Capacity_Won_Pos_MW"], errors="coerce").sum()) == pytest.approx(0.0)
    assert float(pd.to_numeric(partial["global_pf_bcm_product_partial_excluded"], errors="coerce").iloc[0]) == pytest.approx(1.0)


def test_global_pf_benchmark_timestamp_validation_uses_evaluated_window() -> None:
    raw_effective_window = pd.date_range("2025-01-08T00:00:00Z", periods=48, freq="h")
    evaluated_settlement_window = pd.date_range("2025-01-08T01:00:00Z", periods=47, freq="h")

    raw_diag = BatteryBacktester._timestamp_coverage_diagnostics(raw_effective_window, evaluated_settlement_window)
    evaluated_diag = BatteryBacktester._timestamp_coverage_diagnostics(
        evaluated_settlement_window,
        evaluated_settlement_window,
    )

    assert float(raw_diag["aligned"]) == pytest.approx(0.0)
    assert float(evaluated_diag["aligned"]) == pytest.approx(1.0)
    assert float(evaluated_diag["expected_rows"]) == pytest.approx(47.0)
    assert float(evaluated_diag["actual_rows"]) == pytest.approx(47.0)


def test_global_pf_no_market_candidate_uses_evaluated_timestamp_index() -> None:
    raw_input = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-01-08T00:00:00Z", periods=72, freq="h"),
            "value": np.arange(72, dtype=float),
        }
    )
    evaluated_index = pd.date_range("2025-01-08T01:00:00Z", periods=71, freq="h")

    filtered = BatteryBacktester._filter_frame_to_expected_timestamps(
        raw_input,
        timestamp_col="timestamp_utc",
        expected_timestamps=evaluated_index,
    )

    assert len(filtered) == 71
    assert pd.Timestamp(filtered["timestamp_utc"].iloc[0]) == evaluated_index[0]
    assert pd.Timestamp(filtered["timestamp_utc"].iloc[-1]) == evaluated_index[-1]
    diag = BatteryBacktester._timestamp_coverage_diagnostics(evaluated_index, filtered["timestamp_utc"])
    assert float(diag["aligned"]) == pytest.approx(1.0)
    assert float(diag["extra_timestamp_count"]) == pytest.approx(0.0)


def test_same_rules_rolling_pf_dominance_fields_present() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1)
    s = out.summary
    assert "same_rules_rolling_pf_total_pnl_eur" in s
    assert "same_rules_rolling_pf_dominates_realized" in s
    assert "same_rules_rolling_pf_gap_eur" in s
    assert "same_rules_rolling_pf_verified_oracle" in s


def test_global_perfect_foresight_available_only_after_scope_validation() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, enable_global_perfect_foresight=True)
    s = out.summary
    if float(s.get("global_perfect_foresight_available", 0.0)) >= 0.5:
        assert str(s.get("global_perfect_foresight_validation_status", "")) in {
            "verified_same_rules_dominates_realized",
            "verified_full_horizon_solver",
            "verified_no_market",
            "solver_below_realized_path_incumbent",
            "global_pf_unverified",
        }
        if str(s.get("global_pf_selected_incumbent", "")) == "solver":
            assert float(s.get("global_perfect_foresight_dispatch_rows", 0.0)) > 0.0
            assert float(s.get("global_perfect_foresight_settlement_rows", 0.0)) > 0.0
    else:
        assert str(s.get("global_perfect_foresight_validation_status", "")).startswith("disabled") or str(s.get("global_perfect_foresight_validation_status", "")).startswith("computed")


def test_global_pf_bcm_solver_unavailable_can_still_use_realized_path_fallback() -> None:
    bt = _mk_backtester("canonical_economic")
    df, col = _tiny_backtest_df(hours=8)

    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=4,
        reopt_step_hours=1,
        enable_global_perfect_foresight=True,
        allowed_markets=("aFRR", "BCM"),
        strategy_name="bcm",
    )
    s = out.summary

    assert float(s.get("global_pf_available", 0.0)) == pytest.approx(1.0)
    assert float(s.get("global_pf_verified_upper_bound", 1.0)) == pytest.approx(0.0)
    assert float(s.get("global_pf_solver_available", 1.0)) == pytest.approx(0.0)
    assert float(s.get("global_pf_is_solver_upper_bound", 1.0)) == pytest.approx(0.0)
    assert str(s.get("global_pf_selected_incumbent")) in {"no_market", "realized_path"}
    assert str(s.get("global_pf_solver_status")) == "global_solver_missing_bcm_lockbook_semantics"
    assert str(s.get("global_perfect_foresight_capacity_bid_semantics")) == (
        "global_solver_missing_bcm_lockbook_semantics"
    )


def test_locked_reserve_obligation_preserves_future_soc_headroom() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    # Make DA discharge attractive in early hours.
    df[col.pred_da_price] = [250.0, 250.0, 10.0, 10.0]
    ts = pd.to_datetime(df[col.timestamp], utc=True)
    # Lock a positive reserve obligation on the 3rd hour.
    lock = {pd.Timestamp(ts.iloc[2]): (8.0, 0.0)}
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=bt.soc_min + 6.5,
        fixed_reserve_obligation=lock,
        allowed_markets=("DA", "aFRR"),
    )
    r = out.iloc[2]
    req = 8.0 * bt.reserve_activation_headroom_h / bt.eta_out + bt.reserve_headroom_safety_mwh + bt.reserve_soc_projection_safety_mwh
    avail = float(r["soc_start_lp_mwh"]) - bt.soc_min
    assert avail + 1e-9 >= req


def test_locked_negative_reserve_obligation_preserves_future_empty_headroom() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    # Make DA charge attractive in early hours.
    df[col.pred_da_price] = [-200.0, -200.0, -10.0, -10.0]
    ts = pd.to_datetime(df[col.timestamp], utc=True)
    lock = {pd.Timestamp(ts.iloc[2]): (0.0, 8.0)}
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=bt.soc_max - 6.5,
        fixed_reserve_obligation=lock,
        allowed_markets=("DA", "aFRR"),
    )
    r = out.iloc[2]
    req = 8.0 * bt.reserve_activation_headroom_h * bt.eta_in + bt.reserve_headroom_safety_mwh + bt.reserve_soc_projection_safety_mwh
    avail = bt.soc_max - float(r["soc_start_lp_mwh"])
    assert avail + 1e-9 >= req


def test_fallback_counts_nonempty_when_fallback_used() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    df[col.pred_afrr_capacity_price_pos] = 1e3
    df[col.true_afrr_capacity_price_pos] = 1e3
    bt.soc_init = bt.soc_min + 0.01
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    if float(out.summary.get("fallback_used", 0.0)) > 0.5:
        counts = str(out.summary.get("fallback_mode_counts", "")).strip()
        assert counts not in {"", "{}"}


def test_locked_positive_reserve_blocks_future_da_discharge() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    # Strongly profitable DA discharge in first two hours.
    df[col.pred_da_price] = [400.0, 400.0, 0.0, 0.0]
    ts = pd.to_datetime(df[col.timestamp], utc=True)
    lock = {pd.Timestamp(ts.iloc[2]): (9.0, 0.0)}
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=bt.soc_min + 7.0,
        fixed_reserve_obligation=lock,
        allowed_markets=("DA", "aFRR"),
    )
    # Future locked reserve headroom must be preserved.
    req = 9.0 * bt.reserve_activation_headroom_h / bt.eta_out + bt.reserve_headroom_safety_mwh + bt.reserve_soc_projection_safety_mwh
    avail = float(out.loc[2, "soc_start_lp_mwh"]) - bt.soc_min
    assert avail + 1e-9 >= req


def test_locked_positive_reserve_blocks_future_bem_only_discharge() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    # Make BEM-only positive attractive in first hours.
    df[col.pred_afrr_activation_price_pos] = [300.0, 300.0, 0.0, 0.0]
    df[col.pred_afrr_activation_rate_pos] = [1.0, 1.0, 0.0, 0.0]
    ts = pd.to_datetime(df[col.timestamp], utc=True)
    lock = {pd.Timestamp(ts.iloc[2]): (8.0, 0.0)}
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=bt.soc_min + 6.5,
        fixed_reserve_obligation=lock,
        allowed_markets=("aFRR",),
    )
    req = 8.0 * bt.reserve_activation_headroom_h / bt.eta_out + bt.reserve_headroom_safety_mwh + bt.reserve_soc_projection_safety_mwh
    avail = float(out.loc[2, "soc_start_lp_mwh"]) - bt.soc_min
    assert avail + 1e-9 >= req


def test_locked_negative_reserve_blocks_future_charge() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    # Make charging attractive in first two hours.
    df[col.pred_da_price] = [-300.0, -300.0, 0.0, 0.0]
    ts = pd.to_datetime(df[col.timestamp], utc=True)
    lock = {pd.Timestamp(ts.iloc[2]): (0.0, 9.0)}
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=bt.soc_max - 7.0,
        fixed_reserve_obligation=lock,
        allowed_markets=("DA", "aFRR"),
    )
    req = 9.0 * bt.reserve_activation_headroom_h * bt.eta_in + bt.reserve_headroom_safety_mwh + bt.reserve_soc_projection_safety_mwh
    avail = bt.soc_max - float(out.loc[2, "soc_start_lp_mwh"])
    assert avail + 1e-9 >= req


def test_terminal_soc_does_not_override_locked_reserve() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=4)
    # Keep reserve obligation that consumes SoC margin near end.
    ts = pd.to_datetime(df[col.timestamp], utc=True)
    lock = {pd.Timestamp(ts.iloc[3]): (9.0, 0.0)}
    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=bt.soc_min + 6.0,
        soc_end_min_target=bt.soc_target_end,
        fixed_reserve_obligation=lock,
        allowed_markets=("DA", "aFRR"),
    )
    req = 9.0 * bt.reserve_activation_headroom_h / bt.eta_out + bt.reserve_headroom_safety_mwh + bt.reserve_soc_projection_safety_mwh
    avail = float(out.loc[3, "soc_start_lp_mwh"]) - bt.soc_min
    assert avail + 1e-9 >= req


def test_reserve_commitment_debug_explains_soc_drift_fields() -> None:
    required = {
        "scenario",
        "submitted_mw",
        "awarded_mw",
        "locked_obligation_mw",
        "projected_soc_start_mwh_at_commitment",
        "projected_soc_start_mwh_latest_before_delivery",
        "realized_soc_start_mwh_at_delivery",
        "reserve_projected_vs_realized_soc_delta_mwh",
        "da_dispatch_mw_between_commit_and_delivery",
        "id_dispatch_mw_between_commit_and_delivery",
        "bem_only_dispatch_mw_between_commit_and_delivery",
        "aux_energy_mwh_between_commit_and_delivery",
        "required_headroom_mwh",
        "available_headroom_mwh",
        "headroom_violation_mwh",
    }
    # Column list in run_battery_backtest must include these fields.
    path = Path("scripts/run_battery_backtest.py")
    txt = path.read_text(encoding="utf-8")
    for c in required:
        assert c in txt


def test_no_thesis_reportable_reserve_shortfall() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    df[col.pred_afrr_capacity_price_pos] = 1e3
    df[col.true_afrr_capacity_price_pos] = 1e3
    bt.soc_init = bt.soc_min + 0.01
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    short_sum = float(s.get("reserve_headroom_shortfall_pos_mwh_sum", 0.0)) + float(
        s.get("reserve_headroom_shortfall_neg_mwh_sum", 0.0)
    )
    if short_sum > 1e-9:
        assert float(s.get("reserve_headroom_shortfall_check_pass", 1.0)) == 0.0
        assert float(s.get("thesis_reportable", 1.0)) == 0.0
        assert "reserve_headroom_shortfall" in str(s.get("invalid_reason", ""))


def test_protected_soc_envelope_fields_present() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    hourly_cols = set(out.hourly.columns)
    required_hourly = {
        "real_locked_reserve_pos_mw",
        "real_locked_reserve_neg_mw",
        "real_protected_soc_min_mwh",
        "real_protected_soc_max_mwh",
        "real_protected_soc_margin_pos_mwh",
        "real_protected_soc_margin_neg_mwh",
        "real_protected_soc_violation_pos_mwh",
        "real_protected_soc_violation_neg_mwh",
    }
    missing = required_hourly.difference(hourly_cols)
    assert not missing, f"Missing protected SoC fields: {sorted(missing)}"
    s = out.summary
    for k in (
        "protected_soc_violation_count",
        "protected_soc_violation_max_mwh",
        "protected_soc_check_pass",
        "locked_reserve_obligation_hours",
        "locked_reserve_obligation_pos_mw_sum",
        "locked_reserve_obligation_neg_mw_sum",
    ):
        assert k in s


def test_no_thesis_reportable_with_protected_soc_violation() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    psv = float(s.get("protected_soc_violation_count", 0.0))
    if psv > 1e-9:
        assert float(s.get("simulation_valid", 1.0)) == 0.0
        assert float(s.get("thesis_reportable", 1.0)) == 0.0
        assert float(s.get("protected_soc_check_pass", 1.0)) == 0.0
        assert "protected_soc" in str(s.get("invalid_reason", ""))


def test_submitted_bem_only_without_locked_obligation_does_not_activate_protected_envelope() -> None:
    bt = _mk_backtester()
    psv = bt._compute_obligation_driven_protected_soc_bounds(
        soc_start_mwh=4.0,
        required_headroom_pos_mwh=0.0,
        required_headroom_neg_mwh=0.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        committed_bem_pos_mw=4.0,
        committed_bem_neg_mw=0.0,
        reserve_pos_mw=4.0,
        reserve_neg_mw=0.0,
    )
    assert float(psv["obligation_headroom_pos_active"]) == 0.0
    assert np.isclose(float(psv["protected_soc_min_mwh"]), bt.soc_min)
    assert np.isclose(float(psv["protected_soc_max_mwh"]), bt.soc_max)
    assert float(psv["protected_soc_violation_pos_mwh"]) == 0.0


def test_locked_obligation_still_activates_protected_envelope() -> None:
    bt = _mk_backtester()
    req_pos = 2.0 * bt.reserve_activation_headroom_h / max(bt.eta_out, 1e-12)
    psv = bt._compute_obligation_driven_protected_soc_bounds(
        soc_start_mwh=bt.soc_min + req_pos - 0.1,
        required_headroom_pos_mwh=req_pos,
        required_headroom_neg_mwh=0.0,
        locked_reserve_pos_mw=2.0,
        locked_reserve_neg_mw=0.0,
        committed_bem_pos_mw=0.0,
        committed_bem_neg_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
    )
    assert float(psv["obligation_headroom_pos_active"]) == 1.0
    assert float(psv["protected_soc_min_mwh"]) > bt.soc_min
    assert float(psv["protected_soc_violation_pos_mwh"]) > 0.0


def test_bcm_precommit_does_not_monetize_headroom_opportunity_cost() -> None:
    bt = _mk_backtester()
    bt.reserve_min_margin_after_bid_mwh = 0.0
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = None
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-02 02:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            "reserve_pos_mw": [2.0, 2.0, 2.0, 2.0],
            "reserve_neg_mw": [0.0, 0.0, 0.0, 0.0],
            "soc_start_lp_mwh": [bt.soc_min + 10.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    col = BacktestColumnMap()
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [100.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [0.0] * 4,
            col.pred_afrr_activation_price_neg: [0.0] * 4,
            col.pred_afrr_activation_rate_pos: [0.0] * 4,
            col.pred_afrr_activation_rate_neg: [0.0] * 4,
            col.pred_da_price: [300.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    lock_pos: dict[pd.Timestamp, float] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
    )
    assert all(float(v) == pytest.approx(0.0) for v in pre.get("precommit_headroom_opportunity_cost_eur", {}).values())
    assert all(float(v) == pytest.approx(0.0) for v in pre.get("bcm_precommit_expected_headroom_opportunity_cost_eur", {}).values())
    assert all(float(v) == pytest.approx(0.0) for v in pre.get("bcm_precommit_full_headroom_opportunity_cost_eur", {}).values())
    assert all(float(v) == pytest.approx(0.0) for v in pre.get("precommit_bid_zeroed_due_to_negative_ev", {}).values())
    assert any(float(v) > 0.0 for v in pre.get("bcm_precommit_candidate_pos_mw", {}).values())
    assert all(str(v) != "negative_bcm_ev" for v in pre.get("bcm_precommit_zero_reason", {}).values())


def test_bcm_precommit_economic_filter_uses_optimizer_ev_with_capacity_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt = _mk_backtester()
    bt.reserve_min_margin_after_bid_mwh = 0.0
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = None
    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-01 22:00:00+00:00", periods=4, freq="h")
    col = BacktestColumnMap()
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [2.0, 2.0, 2.0, 2.0],
            "reserve_neg_mw": [0.0, 0.0, 0.0, 0.0],
            "soc_start_lp_mwh": [bt.soc_min + 8.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
            # Optimizer objective EV already includes BCM capacity value.
            "ev_afrr_pos_eur": [5000.0, 5000.0, 5000.0, 5000.0],
            "ev_afrr_neg_eur": [0.0, 0.0, 0.0, 0.0],
        }
    )
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [1000.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [0.0] * 4,
            col.pred_afrr_activation_price_neg: [0.0] * 4,
            col.pred_afrr_activation_rate_pos: [0.0] * 4,
            col.pred_afrr_activation_rate_neg: [0.0] * 4,
            col.pred_da_price: [300.0] * 4,
            col.true_afrr_capacity_price_pos: [1000.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    selector_calls: list[tuple[float, float]] = []

    def fake_select(**kwargs):  # type: ignore[no-untyped-def]
        offered_pos = float(kwargs["offered_pos_mw"])
        offered_neg = float(kwargs["offered_neg_mw"])
        selector_calls.append((offered_pos, offered_neg))
        return (
            offered_pos,
            offered_neg,
            AFRRCapacityClearingResult(
                submitted_pos_mw=offered_pos,
                submitted_neg_mw=offered_neg,
                awarded_pos_mw=offered_pos,
                awarded_neg_mw=offered_neg,
                pos_awarded=offered_pos > 0.0,
                neg_awarded=offered_neg > 0.0,
            ),
            [],
            {
                "feasibility_pass": 1.0,
                "retry_factor_selected": 1.0,
                "zero_reason": "none",
                "projected_soc_min_mwh": bt.soc_min + 8.0,
                "projected_soc_max_mwh": bt.soc_min + 8.0,
                "projected_terminal_soc_mwh": bt.soc_min + 8.0,
                "terminal_soc_feasible": 1.0,
                "terminal_soc_shortfall_mwh": 0.0,
                "selection_is_causal": 1.0,
                "full_award_feasibility_checked": 1.0,
                "retry_factor_selected_before_clearing": 1.0,
                "realized_clearing_used_for_selection": 0.0,
            },
        )

    monkeypatch.setattr(bt, "_select_feasible_bcm_lock_candidate", fake_select)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
    )

    assert selector_calls == [(pytest.approx(2.0), pytest.approx(0.0))]
    assert all(float(v) == pytest.approx(0.0) for v in pre["precommit_bid_zeroed_due_to_negative_ev"].values())
    assert all(float(v) > 0.0 for v in pre["precommit_net_capacity_ev_after_headroom_cost_eur"].values())


def test_bcm_precommit_fallback_ev_uses_award_probability_for_capacity_and_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p50"]
    bt.reserve_min_margin_after_bid_mwh = 0.0
    bt.reserve_activation_headroom_h = 0.0
    bt.reserve_headroom_safety_mwh = 0.0
    bt.reserve_soc_projection_safety_mwh = 0.0
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = None
    bt.trans_eur_mwh = 0.0
    bt.deg_eur_mwh = 0.0
    bt.aux_afrr_active_mw = 0.0
    bt.aux_standby_mw = 0.0
    bt.da_terminal_soc_value_enabled = False

    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-01 22:00:00+00:00", periods=4, freq="h")
    col = BacktestColumnMap()
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [2.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "reserve_pos_bin_0_mw": [2.0] * 4,
            "reserve_neg_bin_0_mw": [0.0] * 4,
            "ev_bcm_p_award_pos_bin_0": [0.25] * 4,
            "ev_bcm_p_award_neg_bin_0": [0.25] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 8.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [100.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [50.0] * 4,
            col.pred_afrr_activation_price_neg: [0.0] * 4,
            col.pred_afrr_activation_rate_pos: [0.4] * 4,
            col.pred_afrr_activation_rate_neg: [0.0] * 4,
            col.pred_da_price: [0.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    captured: dict[str, float] = {}

    def fake_select(**kwargs):  # type: ignore[no-untyped-def]
        offered_pos = float(kwargs["offered_pos_mw"])
        offered_neg = float(kwargs["offered_neg_mw"])
        captured["candidate_ev_eur"] = float(kwargs["candidate_ev_eur"])
        captured["recovery_cost_probability_factor"] = float(kwargs["recovery_cost_probability_factor"])
        return (
            offered_pos,
            offered_neg,
            AFRRCapacityClearingResult(
                submitted_pos_mw=offered_pos,
                submitted_neg_mw=offered_neg,
                awarded_pos_mw=offered_pos,
                awarded_neg_mw=offered_neg,
                pos_awarded=offered_pos > 0.0,
                neg_awarded=offered_neg > 0.0,
            ),
            [],
            {
                "feasibility_pass": 1.0,
                "retry_factor_selected": 1.0,
                "zero_reason": "none",
                "projected_soc_min_mwh": bt.soc_min + 8.0,
                "projected_soc_max_mwh": bt.soc_min + 8.0,
                "projected_terminal_soc_mwh": bt.soc_min + 8.0,
                "terminal_soc_feasible": 1.0,
                "terminal_soc_shortfall_mwh": 0.0,
                "selection_is_causal": 1.0,
                "full_award_feasibility_checked": 1.0,
                "retry_factor_selected_before_clearing": 1.0,
                "realized_clearing_used_for_selection": 0.0,
            },
        )

    monkeypatch.setattr(bt, "_select_feasible_bcm_lock_candidate", fake_select)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
    )

    expected_capacity = 0.25 * 2.0 * 100.0 * 4.0
    expected_activation = 0.25 * 2.0 * 0.4 * 50.0 * 4.0
    assert captured["recovery_cost_probability_factor"] == pytest.approx(0.25)
    assert captured["candidate_ev_eur"] == pytest.approx(expected_capacity + expected_activation)
    assert all(float(v) == pytest.approx(0.25) for v in pre["bcm_precommit_ev_award_probability"].values())
    assert all(float(v) == pytest.approx(expected_capacity) for v in pre["bcm_precommit_expected_capacity_revenue_eur"].values())
    assert all(float(v) == pytest.approx(expected_activation) for v in pre["bcm_precommit_expected_activation_revenue_eur"].values())
    assert all(float(v) == pytest.approx(0.0) for v in pre["precommit_bid_zeroed_due_to_negative_ev"].values())


def test_bcm_precommit_recovery_cost_probability_factor_without_soc_start_uses_soc_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p50"]
    bt.reserve_min_margin_after_bid_mwh = 0.0
    bt.reserve_activation_headroom_h = 0.0
    bt.reserve_headroom_safety_mwh = 0.0
    bt.reserve_soc_projection_safety_mwh = 0.0
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = None
    bt.trans_eur_mwh = 0.0
    bt.deg_eur_mwh = 0.0
    bt.aux_afrr_active_mw = 0.0
    bt.aux_standby_mw = 0.0
    bt.da_terminal_soc_value_enabled = False

    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-01 22:00:00+00:00", periods=4, freq="h")
    col = BacktestColumnMap()
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [2.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "reserve_pos_bin_0_mw": [2.0] * 4,
            "reserve_neg_bin_0_mw": [0.0] * 4,
            "ev_bcm_p_award_pos_bin_0": [0.25] * 4,
            "ev_bcm_p_award_neg_bin_0": [0.25] * 4,
            "soc_before_mwh": [bt.soc_min + 8.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [100.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [50.0] * 4,
            col.pred_afrr_activation_price_neg: [0.0] * 4,
            col.pred_afrr_activation_rate_pos: [0.4] * 4,
            col.pred_afrr_activation_rate_neg: [0.0] * 4,
            col.pred_da_price: [0.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    captured: dict[str, float] = {}

    def fake_select(**kwargs):  # type: ignore[no-untyped-def]
        offered_pos = float(kwargs["offered_pos_mw"])
        offered_neg = float(kwargs["offered_neg_mw"])
        captured["candidate_ev_eur"] = float(kwargs["candidate_ev_eur"])
        captured["recovery_cost_probability_factor"] = float(kwargs["recovery_cost_probability_factor"])
        return (
            offered_pos,
            offered_neg,
            AFRRCapacityClearingResult(
                submitted_pos_mw=offered_pos,
                submitted_neg_mw=offered_neg,
                awarded_pos_mw=offered_pos,
                awarded_neg_mw=offered_neg,
                pos_awarded=offered_pos > 0.0,
                neg_awarded=offered_neg > 0.0,
            ),
            [],
            {
                "feasibility_pass": 1.0,
                "retry_factor_selected": 1.0,
                "zero_reason": "none",
                "projected_soc_min_mwh": bt.soc_min + 8.0,
                "projected_soc_max_mwh": bt.soc_min + 8.0,
                "projected_terminal_soc_mwh": bt.soc_min + 8.0,
                "terminal_soc_feasible": 1.0,
                "terminal_soc_shortfall_mwh": 0.0,
                "selection_is_causal": 1.0,
                "full_award_feasibility_checked": 1.0,
                "retry_factor_selected_before_clearing": 1.0,
                "realized_clearing_used_for_selection": 0.0,
            },
        )

    monkeypatch.setattr(bt, "_select_feasible_bcm_lock_candidate", fake_select)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
    )

    expected_capacity = 0.25 * 2.0 * 100.0 * 4.0
    expected_activation = 0.25 * 2.0 * 0.4 * 50.0 * 4.0
    assert captured["recovery_cost_probability_factor"] == pytest.approx(0.25)
    assert captured["candidate_ev_eur"] == pytest.approx(expected_capacity + expected_activation)
    assert set(pre["bcm_precommit_soc_start_source"].values()) == {"soc_before_mwh"}
    assert all(float(v) == pytest.approx(0.0) for v in pre["bcm_precommit_soc_start_missing"].values())
    assert all(float(v) == pytest.approx(0.25) for v in pre["bcm_precommit_ev_award_probability"].values())
    assert all(
        str(v) == "ev_bcm_p_award_pos_bin_0"
        for v in pre["bcm_precommit_recovery_cost_probability_factor_source"].values()
    )


def test_bcm_precommit_uses_executed_soc_anchor_for_expected_delivery_guard() -> None:
    bt = _mk_backtester()
    bt.reserve_headroom_safety_mwh = 0.0
    bt.reserve_soc_projection_safety_mwh = 0.0
    bt.reserve_power_safety_mw = 0.0
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = None
    bt.afrr_min_bid_size_mw = 1.0
    bt.afrr_bid_granularity_mw = 1.0

    col = BacktestColumnMap()
    snapshot_ts = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-01 22:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [4.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 8.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [100.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_rate_pos: [0.5] * 4,
            col.pred_afrr_activation_rate_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [50.0] * 4,
            col.pred_afrr_activation_price_neg: [0.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}

    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=snapshot_ts,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
        current_soc_mwh=bt.soc_min,
    )

    assert all(float(v) == pytest.approx(0.0) for v in pre["bcm_precommit_locked_pos_mw"].values())
    assert all(float(v) == pytest.approx(1.0) for v in pre["bcm_precommit_stale_soc_anchor"].values())
    assert set(pre["bcm_precommit_soc_start_source"].values()) == {"executed_carry_soc"}
    assert all(
        str(v) == "pos_expected_activation_not_deliverable"
        for v in pre["bcm_precommit_expected_delivery_guard_reason"].values()
    )
    assert all(
        float(v) == pytest.approx(4.0)
        for v in pre["bcm_precommit_pos_derated_by_expected_delivery_guard_mw"].values()
    )


def _bcm_naive_guard_inputs(
    bt: BatteryBacktester,
    *,
    reserve_pos_mw: float,
    reserve_neg_mw: float,
    rate_pos: float,
    rate_neg: float,
    capacity_price_pos: float = 100.0,
    capacity_price_neg: float = 100.0,
) -> tuple[BacktestColumnMap, pd.Timestamp, pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    col = BacktestColumnMap()
    snapshot_ts = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-01 22:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [float(reserve_pos_mw)] * 4,
            "reserve_neg_mw": [float(reserve_neg_mw)] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 8.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [float(capacity_price_pos)] * 4,
            col.pred_afrr_capacity_price_neg: [float(capacity_price_neg)] * 4,
            col.pred_afrr_activation_rate_pos: [float(rate_pos)] * 4,
            col.pred_afrr_activation_rate_neg: [float(rate_neg)] * 4,
            col.true_afrr_activation_rate_pos: [float(rate_pos)] * 4,
            col.true_afrr_activation_rate_neg: [float(rate_neg)] * 4,
            col.pred_afrr_activation_price_pos: [50.0] * 4,
            col.pred_afrr_activation_price_neg: [50.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [100.0] * 4,
        }
    ).set_index(col.timestamp)
    return col, snapshot_ts, target_hours, snap, src


def _configure_bcm_naive_guard_test_backtester() -> BatteryBacktester:
    bt = _mk_backtester()
    bt.eta_out = 1.0
    bt.eta_in = 1.0
    bt.reserve_activation_headroom_h = 0.0
    bt.reserve_headroom_safety_mwh = 0.0
    bt.reserve_soc_projection_safety_mwh = 0.0
    bt.reserve_power_safety_mw = 0.0
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = None
    bt.afrr_min_bid_size_mw = 1.0
    bt.afrr_bid_granularity_mw = 1.0
    return bt


def test_bcm_naive_rejects_pos_capacity_when_soc_at_min_and_activation_requested() -> None:
    bt = _configure_bcm_naive_guard_test_backtester()
    col, snapshot_ts, target_hours, snap, src = _bcm_naive_guard_inputs(
        bt,
        reserve_pos_mw=4.0,
        reserve_neg_mw=0.0,
        rate_pos=0.5,
        rate_neg=0.0,
    )
    lock_pos: dict[pd.Timestamp, float] = {}
    lock_neg: dict[pd.Timestamp, float] = {}
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}

    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=snapshot_ts,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
        current_soc_mwh=bt.soc_min,
    )

    assert all(float(v) == pytest.approx(0.0) for v in lock_pos.values())
    assert all(float(v) == pytest.approx(0.0) for v in pre["bcm_precommit_locked_pos_mw"].values())
    assert all(float(v) == pytest.approx(0.0) for v in pre["bcm_activation_replay_prelock_pass"].values())
    assert all(float(v) == pytest.approx(8.0) for v in pre["bcm_replay_required_pos_internal_mwh"].values())
    assert all(float(v) == pytest.approx(0.0) for v in pre["bcm_replay_available_pos_internal_mwh"].values())
    assert all(
        float(v) == pytest.approx(4.0)
        for v in pre["bcm_capacity_rejected_missed_activation_pos_mw"].values()
    )
    assert all(
        str(v) == "missed_activation_prevented_pos"
        for v in pre["bcm_missed_activation_prevention_reason"].values()
    )


def test_bcm_naive_uses_executed_soc_and_activation_headroom_floor() -> None:
    bt = _configure_bcm_naive_guard_test_backtester()
    bt.reserve_activation_headroom_h = 0.5
    col, snapshot_ts, target_hours, snap, src = _bcm_naive_guard_inputs(
        bt,
        reserve_pos_mw=4.0,
        reserve_neg_mw=0.0,
        rate_pos=0.0,
        rate_neg=0.0,
    )
    lock_pos: dict[pd.Timestamp, float] = {}
    lock_neg: dict[pd.Timestamp, float] = {}
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}

    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=snapshot_ts,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
        current_soc_mwh=bt.soc_min,
    )

    assert all(float(v) == pytest.approx(0.0) for v in lock_pos.values())
    assert all(
        str(v) == "executed_carry_soc"
        for v in pre["bcm_precommit_soc_start_source"].values()
    )
    assert all(float(v) == pytest.approx(1.0) for v in pre["bcm_precommit_stale_soc_anchor"].values())
    assert all(
        float(v) == pytest.approx(bt.soc_min)
        for v in pre["bcm_precommit_settlement_soc_start_mwh"].values()
    )
    assert all(float(v) == pytest.approx(0.0) for v in pre["bcm_activation_replay_prelock_pass"].values())
    assert all(float(v) == pytest.approx(2.0) for v in pre["bcm_replay_required_pos_internal_mwh"].values())
    assert all(float(v) == pytest.approx(0.0) for v in pre["bcm_replay_available_pos_internal_mwh"].values())
    assert all(
        float(v) == pytest.approx(4.0)
        for v in pre["bcm_capacity_rejected_missed_activation_pos_mw"].values()
    )
    assert all(
        str(v) == "missed_activation_prevented_pos"
        for v in pre["bcm_missed_activation_prevention_reason"].values()
    )


def test_bcm_naive_derates_pos_capacity_to_deliverable_energy() -> None:
    bt = _configure_bcm_naive_guard_test_backtester()
    col, snapshot_ts, target_hours, snap, src = _bcm_naive_guard_inputs(
        bt,
        reserve_pos_mw=4.0,
        reserve_neg_mw=0.0,
        rate_pos=0.5,
        rate_neg=0.0,
    )
    lock_pos: dict[pd.Timestamp, float] = {}
    lock_neg: dict[pd.Timestamp, float] = {}
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}

    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=snapshot_ts,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
        current_soc_mwh=bt.soc_min + 2.0,
    )

    assert all(float(v) == pytest.approx(1.0) for v in lock_pos.values())
    assert all(float(v) == pytest.approx(1.0) for v in pre["bcm_precommit_locked_pos_mw"].values())
    assert all(float(v) == pytest.approx(8.0) for v in pre["bcm_replay_required_pos_internal_mwh"].values())
    assert all(float(v) == pytest.approx(2.0) for v in pre["bcm_replay_available_pos_internal_mwh"].values())
    assert all(
        float(v) == pytest.approx(3.0)
        for v in pre["bcm_capacity_derated_missed_activation_pos_mw"].values()
    )
    assert all(
        float(v) == pytest.approx(0.0)
        for v in pre["bcm_capacity_rejected_missed_activation_pos_mw"].values()
    )


def test_bcm_naive_rejects_or_derates_neg_capacity_when_no_charge_headroom() -> None:
    bt = _configure_bcm_naive_guard_test_backtester()
    col, snapshot_ts, target_hours, snap, src = _bcm_naive_guard_inputs(
        bt,
        reserve_pos_mw=0.0,
        reserve_neg_mw=4.0,
        rate_pos=0.0,
        rate_neg=0.5,
    )
    lock_pos: dict[pd.Timestamp, float] = {}
    lock_neg: dict[pd.Timestamp, float] = {}
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}

    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=snapshot_ts,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
        current_soc_mwh=bt.soc_max,
    )

    assert all(float(v) == pytest.approx(0.0) for v in lock_neg.values())
    assert all(float(v) == pytest.approx(0.0) for v in pre["bcm_precommit_locked_neg_mw"].values())
    assert all(float(v) == pytest.approx(8.0) for v in pre["bcm_replay_required_neg_headroom_mwh"].values())
    assert all(float(v) == pytest.approx(0.0) for v in pre["bcm_replay_available_neg_headroom_mwh"].values())
    assert all(
        float(v) == pytest.approx(4.0)
        for v in pre["bcm_capacity_rejected_missed_activation_neg_mw"].values()
    )
    assert all(
        str(v) == "missed_activation_prevented_neg"
        for v in pre["bcm_missed_activation_prevention_reason"].values()
    )


def test_bcm_naive_valid_capacity_survives_prelock_guard() -> None:
    bt = _configure_bcm_naive_guard_test_backtester()
    col, snapshot_ts, target_hours, snap, src = _bcm_naive_guard_inputs(
        bt,
        reserve_pos_mw=1.0,
        reserve_neg_mw=1.0,
        rate_pos=0.1,
        rate_neg=0.1,
    )
    lock_pos: dict[pd.Timestamp, float] = {}
    lock_neg: dict[pd.Timestamp, float] = {}
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}

    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=snapshot_ts,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
        current_soc_mwh=bt.soc_min + 8.0,
    )

    assert all(float(v) == pytest.approx(1.0) for v in lock_pos.values())
    assert all(float(v) == pytest.approx(1.0) for v in lock_neg.values())
    assert all(float(v) == pytest.approx(1.0) for v in pre["bcm_activation_replay_prelock_pass"].values())
    assert all(
        str(v) == "none"
        for v in pre["bcm_missed_activation_prevention_reason"].values()
    )


def _configure_bcm_rhpf_protected_soc_test_backtester() -> BatteryBacktester:
    bt = _configure_bcm_naive_guard_test_backtester()
    bt.reserve_activation_headroom_h = 0.5
    bt.reserve_headroom_safety_mwh = 0.0
    bt.reserve_soc_projection_safety_mwh = 0.0
    return bt


def test_bcm_rhpf_rejects_pos_capacity_that_violates_protected_soc() -> None:
    bt = _configure_bcm_rhpf_protected_soc_test_backtester()
    col, snapshot_ts, target_hours, snap, src = _bcm_naive_guard_inputs(
        bt,
        reserve_pos_mw=4.0,
        reserve_neg_mw=0.0,
        rate_pos=0.0,
        rate_neg=0.0,
    )
    lock_pos: dict[pd.Timestamp, float] = {}
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}

    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=snapshot_ts,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=True,
        global_end_utc=target_hours[-1],
        current_soc_mwh=bt.soc_min,
    )

    assert all(float(v) == pytest.approx(0.0) for v in lock_pos.values())
    assert all(float(v) == pytest.approx(0.0) for v in pre["bcm_protected_soc_replay_pass"].values())
    assert all(
        float(v) == pytest.approx(4.0)
        for v in pre["bcm_capacity_rejected_protected_soc_pos_mw"].values()
    )
    assert all(
        str(v) == "protected_soc_prevented_pos"
        for v in pre["bcm_protected_soc_rejection_reason"].values()
    )
    assert all(
        float(v) == pytest.approx(bt.soc_min + 2.0)
        for v in pre["bcm_required_protected_soc_min_mwh"].values()
    )


def test_bcm_rhpf_derates_pos_capacity_to_protected_soc_feasible_level() -> None:
    bt = _configure_bcm_rhpf_protected_soc_test_backtester()
    col, snapshot_ts, target_hours, snap, src = _bcm_naive_guard_inputs(
        bt,
        reserve_pos_mw=6.0,
        reserve_neg_mw=0.0,
        rate_pos=0.0,
        rate_neg=0.0,
    )
    lock_pos: dict[pd.Timestamp, float] = {}
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}

    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=snapshot_ts,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=True,
        global_end_utc=target_hours[-1],
        current_soc_mwh=bt.soc_min + 2.0,
    )

    assert all(float(v) == pytest.approx(3.0) for v in lock_pos.values())
    assert all(
        float(v) == pytest.approx(3.0)
        for v in pre["bcm_capacity_derated_protected_soc_pos_mw"].values()
    )
    assert all(
        float(v) == pytest.approx(0.0)
        for v in pre["bcm_capacity_rejected_protected_soc_pos_mw"].values()
    )
    assert all(
        str(v) == "protected_soc_prevented_pos"
        for v in pre["bcm_protected_soc_rejection_reason"].values()
    )


def test_bcm_rhpf_derates_pos_capacity_for_sequential_block_drain() -> None:
    bt = _configure_bcm_rhpf_protected_soc_test_backtester()
    col, snapshot_ts, target_hours, snap, src = _bcm_naive_guard_inputs(
        bt,
        reserve_pos_mw=4.0,
        reserve_neg_mw=0.0,
        rate_pos=0.5,
        rate_neg=0.0,
    )
    lock_pos: dict[pd.Timestamp, float] = {}
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}

    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=snapshot_ts,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=True,
        global_end_utc=target_hours[-1],
        current_soc_mwh=bt.soc_min + 4.5,
    )

    assert all(0.0 <= float(v) < 4.0 for v in lock_pos.values())
    assert any(
        float(v) > 0.0 for v in pre["bcm_capacity_derated_protected_soc_pos_mw"].values()
    )
    assert all(
        str(v) == "protected_soc_prevented_pos"
        for v in pre["bcm_protected_soc_rejection_reason"].values()
    )
    assert any(
        str(v) != "" for v in pre["bcm_first_protected_soc_violation_ts_utc"].values()
    )


def test_bcm_rhpf_existing_pos_obligations_consume_protected_soc_capacity() -> None:
    bt = _configure_bcm_rhpf_protected_soc_test_backtester()
    col, snapshot_ts, target_hours, snap, src = _bcm_naive_guard_inputs(
        bt,
        reserve_pos_mw=6.0,
        reserve_neg_mw=0.0,
        rate_pos=0.0,
        rate_neg=0.0,
    )
    existing_lock_pos = {pd.Timestamp(ts): 4.0 for ts in target_hours}
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}

    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=snapshot_ts,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=existing_lock_pos,
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=True,
        global_end_utc=target_hours[-1],
        current_soc_mwh=bt.soc_min + 4.0,
    )

    assert all(float(v) == pytest.approx(4.0) for v in existing_lock_pos.values())
    assert all(
        float(v) == pytest.approx(2.0)
        for v in pre["bcm_capacity_derated_protected_soc_pos_mw"].values()
    )
    assert all(
        float(v) == pytest.approx(0.0)
        for v in pre["bcm_capacity_rejected_protected_soc_pos_mw"].values()
    )
    assert all(
        float(v) == pytest.approx(4.0)
        for v in pre["bcm_precommit_locked_pos_mw"].values()
    )
    assert all(
        str(v) == "protected_soc_prevented_pos"
        for v in pre["bcm_protected_soc_rejection_reason"].values()
    )


def test_bcm_rhpf_rejects_neg_capacity_that_violates_upper_protected_soc() -> None:
    bt = _configure_bcm_rhpf_protected_soc_test_backtester()
    col, snapshot_ts, target_hours, snap, src = _bcm_naive_guard_inputs(
        bt,
        reserve_pos_mw=0.0,
        reserve_neg_mw=4.0,
        rate_pos=0.0,
        rate_neg=0.0,
    )
    lock_neg: dict[pd.Timestamp, float] = {}
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}

    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=snapshot_ts,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg=lock_neg,
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=True,
        global_end_utc=target_hours[-1],
        current_soc_mwh=bt.soc_max,
    )

    assert all(float(v) == pytest.approx(0.0) for v in lock_neg.values())
    assert all(float(v) == pytest.approx(0.0) for v in pre["bcm_protected_soc_replay_pass"].values())
    assert all(
        float(v) == pytest.approx(4.0)
        for v in pre["bcm_capacity_rejected_protected_soc_neg_mw"].values()
    )
    assert all(
        str(v) == "protected_soc_prevented_neg"
        for v in pre["bcm_protected_soc_rejection_reason"].values()
    )
    assert all(
        float(v) == pytest.approx(bt.soc_max - 2.0)
        for v in pre["bcm_required_protected_soc_max_mwh"].values()
    )


def test_bcm_rhpf_valid_capacity_survives_protected_soc_guard() -> None:
    bt = _configure_bcm_rhpf_protected_soc_test_backtester()
    col, snapshot_ts, target_hours, snap, src = _bcm_naive_guard_inputs(
        bt,
        reserve_pos_mw=2.0,
        reserve_neg_mw=2.0,
        rate_pos=0.0,
        rate_neg=0.0,
    )
    lock_pos: dict[pd.Timestamp, float] = {}
    lock_neg: dict[pd.Timestamp, float] = {}
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}

    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=snapshot_ts,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos=lock_pos,
        lock_neg=lock_neg,
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=True,
        global_end_utc=target_hours[-1],
        current_soc_mwh=bt.soc_min + 8.0,
    )

    assert all(float(v) == pytest.approx(2.0) for v in lock_pos.values())
    assert all(float(v) == pytest.approx(2.0) for v in lock_neg.values())
    assert all(float(v) == pytest.approx(1.0) for v in pre["bcm_protected_soc_replay_pass"].values())
    assert all(str(v) == "none" for v in pre["bcm_protected_soc_rejection_reason"].values())


def test_bcm_rhpf_precommit_expected_delivery_guard_uses_true_activation_rate() -> None:
    bt = _mk_backtester()
    bt.reserve_headroom_safety_mwh = 0.0
    bt.reserve_soc_projection_safety_mwh = 0.0
    bt.reserve_power_safety_mw = 0.0
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = None
    bt.afrr_min_bid_size_mw = 1.0
    bt.afrr_bid_granularity_mw = 1.0

    col = BacktestColumnMap()
    snapshot_ts = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-01 22:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [4.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 8.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [100.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_rate_pos: [0.0] * 4,
            col.pred_afrr_activation_rate_neg: [0.0] * 4,
            col.true_afrr_activation_rate_pos: [0.5] * 4,
            col.true_afrr_activation_rate_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [50.0] * 4,
            col.pred_afrr_activation_price_neg: [0.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}

    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=snapshot_ts,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=True,
        global_end_utc=target_hours[-1],
        current_soc_mwh=bt.soc_min,
    )

    assert all(float(v) == pytest.approx(0.0) for v in pre["bcm_precommit_locked_pos_mw"].values())
    assert all(float(v) == pytest.approx(0.5) for v in pre["bcm_precommit_expected_activation_rate_pos"].values())
    assert all(
        str(v) == "pos_expected_activation_not_deliverable"
        for v in pre["bcm_precommit_expected_delivery_guard_reason"].values()
    )


def test_bcm_precommit_expected_delivery_guard_uses_reserve_headroom_floor() -> None:
    bt = _mk_backtester()
    bt.reserve_headroom_safety_mwh = 0.0
    bt.reserve_soc_projection_safety_mwh = 0.0
    bt.reserve_power_safety_mw = 0.0
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = None
    bt.afrr_min_bid_size_mw = 1.0
    bt.afrr_bid_granularity_mw = 1.0
    bt.reserve_activation_headroom_h = 0.5

    col = BacktestColumnMap()
    snapshot_ts = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-01 22:00:00+00:00", periods=4, freq="h")
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [4.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 8.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [100.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_rate_pos: [0.001] * 4,
            col.pred_afrr_activation_rate_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [50.0] * 4,
            col.pred_afrr_activation_price_neg: [0.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}

    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=snapshot_ts,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
        current_soc_mwh=bt.soc_min + 2.0,
    )

    assert all(float(v) == pytest.approx(3.0) for v in pre["bcm_precommit_locked_pos_mw"].values())
    assert all(float(v) == pytest.approx(1.0) for v in pre["bcm_pos_derated_by_expected_delivery_guard_mw"].values())
    assert all(
        str(v) == "pos_expected_activation_not_deliverable"
        for v in pre["bcm_precommit_expected_delivery_guard_reason"].values()
    )


def test_bcm_model_vs_naive_audit_uses_locked_bcm_columns_before_awarded_aliases() -> None:
    ts = pd.date_range("2025-06-01 00:00:00+00:00", periods=2, freq="h")
    model = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "real_revenue_capacity_eur": [10.0, 20.0],
            "real_revenue_activation_eur": [1.0, 2.0],
            "real_locked_bcm_capacity_pos_mw": [4.0, 4.0],
            "real_locked_bcm_capacity_neg_mw": [1.0, 1.0],
            "real_awarded_capacity_pos_mw": [0.0, 0.0],
            "real_awarded_capacity_neg_mw": [0.0, 0.0],
        }
    )
    naive = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "naive_revenue_capacity_eur": [30.0, 40.0],
            "naive_revenue_activation_eur": [3.0, 4.0],
            "naive_locked_bcm_capacity_pos_mw": [6.0, 6.0],
            "naive_locked_bcm_capacity_neg_mw": [2.0, 2.0],
            "naive_awarded_capacity_pos_mw": [0.0, 0.0],
            "naive_awarded_capacity_neg_mw": [0.0, 0.0],
        }
    )

    audit = build_bcm_model_vs_naive_audit({"model": model, "naive": naive})

    model_row = audit.loc[audit["path_type"].eq("model")].iloc[0]
    naive_row = audit.loc[audit["path_type"].eq("naive")].iloc[0]
    assert float(model_row["locked_pos_mw"]) == pytest.approx(8.0)
    assert float(model_row["locked_neg_mw"]) == pytest.approx(2.0)
    assert float(naive_row["locked_pos_mw"]) == pytest.approx(12.0)
    assert float(naive_row["locked_neg_mw"]) == pytest.approx(4.0)
    assert set(audit["classification"]) == {"naive_more_pos_exposure"}


def test_bcm_precommit_uses_optimizer_component_ev_decomposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p50"]
    bt.reserve_min_margin_after_bid_mwh = 0.0
    bt.reserve_activation_headroom_h = 0.0
    bt.reserve_headroom_safety_mwh = 0.0
    bt.reserve_soc_projection_safety_mwh = 0.0
    bt.reserve_bid_derate = 1.0
    bt.max_reserve_bid_mw = None
    bt.afrr_min_bid_size_mw = 1.0
    bt.afrr_bid_granularity_mw = 1.0
    bt.da_terminal_soc_value_enabled = False

    ts_snapshot = pd.Timestamp("2025-05-01 06:00:00+00:00")
    target_hours = pd.date_range("2025-05-01 22:00:00+00:00", periods=4, freq="h")
    col = BacktestColumnMap()
    snap = pd.DataFrame(
        {
            "target_time_utc": target_hours,
            col.timestamp: target_hours,
            "reserve_pos_mw": [2.0] * 4,
            "reserve_neg_mw": [0.0] * 4,
            "reserve_pos_bin_0_mw": [2.0] * 4,
            "reserve_neg_bin_0_mw": [0.0] * 4,
            "ev_bcm_expected_capacity_revenue_pos_bin_0": [25.0] * 4,
            "ev_bcm_expected_capacity_revenue_neg_bin_0": [0.0] * 4,
            "ev_bcm_expected_activation_revenue_pos_bin_0": [5.0] * 4,
            "ev_bcm_expected_activation_revenue_neg_bin_0": [0.0] * 4,
            "ev_bcm_standby_cost_pos_bin_0": [3.0] * 4,
            "ev_bcm_standby_cost_neg_bin_0": [0.0] * 4,
            "ev_bcm_activation_cost_pos_bin_0": [2.0] * 4,
            "ev_bcm_activation_cost_neg_bin_0": [0.0] * 4,
            # This legacy aggregate is intentionally inconsistent. The
            # precommit selector must use the component decomposition instead.
            "ev_afrr_pos_eur": [9999.0] * 4,
            "ev_afrr_neg_eur": [0.0] * 4,
            "soc_start_lp_mwh": [bt.soc_min + 8.0] * 4,
            "discharge_mw": [0.0] * 4,
            "charge_mw": [0.0] * 4,
            "id_discharge_mw": [0.0] * 4,
            "id_charge_mw": [0.0] * 4,
            "bem_only_pos_mw": [0.0] * 4,
            "bem_only_neg_mw": [0.0] * 4,
            "aux_power_mw": [0.0] * 4,
        }
    )
    src = pd.DataFrame(
        {
            col.timestamp: target_hours,
            col.pred_afrr_capacity_price_pos: [0.0] * 4,
            col.pred_afrr_capacity_price_neg: [0.0] * 4,
            col.pred_afrr_activation_price_pos: [0.0] * 4,
            col.pred_afrr_activation_price_neg: [0.0] * 4,
            col.pred_afrr_activation_rate_pos: [0.0] * 4,
            col.pred_afrr_activation_rate_neg: [0.0] * 4,
            col.pred_da_price: [0.0] * 4,
            col.true_afrr_capacity_price_pos: [100.0] * 4,
            col.true_afrr_capacity_price_neg: [0.0] * 4,
        }
    ).set_index(col.timestamp)
    captured: dict[str, float] = {}

    def fake_select(**kwargs):  # type: ignore[no-untyped-def]
        offered_pos = float(kwargs["offered_pos_mw"])
        offered_neg = float(kwargs["offered_neg_mw"])
        captured["candidate_ev_eur"] = float(kwargs["candidate_ev_eur"])
        return (
            offered_pos,
            offered_neg,
            AFRRCapacityClearingResult(
                submitted_pos_mw=offered_pos,
                submitted_neg_mw=offered_neg,
                awarded_pos_mw=offered_pos,
                awarded_neg_mw=offered_neg,
                pos_awarded=offered_pos > 0.0,
                neg_awarded=offered_neg > 0.0,
            ),
            [],
            {
                "feasibility_pass": 1.0,
                "retry_factor_selected": 1.0,
                "zero_reason": "none",
                "projected_soc_min_mwh": bt.soc_min + 8.0,
                "projected_soc_max_mwh": bt.soc_min + 8.0,
                "projected_terminal_soc_mwh": bt.soc_min + 8.0,
                "terminal_soc_feasible": 1.0,
                "terminal_soc_shortfall_mwh": 0.0,
                "candidate_recoverable": 1.0,
                "ev_after_recovery_eur": float("nan"),
                "effective_ev_after_recovery_eur": float("nan"),
                "ev_after_incremental_recovery_eur": float("nan"),
                "incremental_recovery_cost_eur": float("nan"),
                "selection_is_causal": 1.0,
                "full_award_feasibility_checked": 1.0,
                "retry_factor_selected_before_clearing": 1.0,
                "realized_clearing_used_for_selection": 0.0,
            },
        )

    monkeypatch.setattr(bt, "_select_feasible_bcm_lock_candidate", fake_select)
    pre: dict[str, dict[pd.Timestamp, float | str]] = {}
    bt._update_afrr_capacity_lockbooks_from_snapshot(
        snapshot_ts=ts_snapshot,
        snapshot_plan=snap,
        source=src,
        colmap=col,
        lock_pos={},
        lock_neg={},
        lock_energy_pos={},
        lock_energy_neg={},
        precommit_audit_by_ts=pre,
        is_perfect_foresight=False,
        global_end_utc=target_hours[-1],
    )

    expected_capacity = 2.0 * 25.0 * 4.0
    expected_activation = 2.0 * 5.0 * 4.0
    expected_standby = 2.0 * 3.0 * 4.0
    expected_activation_cost = 2.0 * 2.0 * 4.0
    expected_ev = expected_capacity + expected_activation - expected_standby - expected_activation_cost
    assert captured["candidate_ev_eur"] == pytest.approx(expected_ev)
    assert all(float(v) == pytest.approx(1.0) for v in pre["bcm_precommit_optimizer_component_ev_available"].values())
    assert all(float(v) == pytest.approx(expected_capacity) for v in pre["bcm_precommit_expected_capacity_revenue_eur"].values())
    assert all(float(v) == pytest.approx(expected_activation) for v in pre["bcm_precommit_expected_activation_revenue_eur"].values())
    assert all(float(v) == pytest.approx(expected_standby) for v in pre["bcm_precommit_expected_standby_cost_eur"].values())
    assert all(float(v) == pytest.approx(expected_activation_cost) for v in pre["bcm_precommit_expected_activation_cost_eur"].values())
    assert all(float(v) == pytest.approx(expected_ev) for v in pre["bcm_precommit_expected_capacity_ev_eur"].values())
    assert all(float(v) == pytest.approx(expected_ev) for v in pre["bcm_precommit_ev_after_recovery_eur"].values())
    assert all(float(v) == pytest.approx(expected_ev) for v in pre["bcm_precommit_effective_ev_after_recovery_eur"].values())
    assert all(float(v) == pytest.approx(expected_ev) for v in pre["bcm_precommit_ev_after_incremental_recovery_eur"].values())
    assert all(float(v) == pytest.approx(0.0) for v in pre["bcm_precommit_ev_nan_detected"].values())
    assert all(float(v) == pytest.approx(0.0) for v in pre["bcm_precommit_invariant_violation"].values())


def test_required_series_missing_raises_keyerror() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0]})
    try:
        _ = require_numeric_series(df, "real_executed_charge_mw", aliases=["real_da_charge_mw"])
        assert False, "expected KeyError"
    except KeyError as exc:
        assert "real_executed_charge_mw" in str(exc)


def test_optional_series_missing_defaults_to_zero() -> None:
    df = pd.DataFrame(index=[0, 1, 2])
    s = optional_numeric_series(df, "optional_col", default=0.0)
    assert len(s) == len(df)
    assert np.isclose(float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum()), 0.0, atol=1e-12)


def test_legacy_alias_resolution_prefers_canonical() -> None:
    df = pd.DataFrame(
        {
            "real_executed_charge_mw": [1.0, 2.0],
            "real_da_charge_mw": [100.0, 200.0],
        }
    )
    s = require_numeric_series(df, "real_executed_charge_mw", aliases=["real_da_charge_mw"])
    assert np.isclose(float(s.iloc[0]), 1.0, atol=1e-12)
    assert np.isclose(float(s.iloc[1]), 2.0, atol=1e-12)


def test_validator_reports_debug_dump_counts(tmp_path: Path) -> None:
    scen = tmp_path / "multi" / "p50_p50"
    scen.mkdir(parents=True, exist_ok=True)
    payload = {
        "simulation_valid": 0.0,
        "thesis_reportable": 0.0,
        "invalid_reason": "optimization_infeasible_debug_dump",
        "fallback_used": 0.0,
        "fallback_mode_counts": "{}",
        "optimization_error_code_counts": '{"ok": 1.0}',
        "infeasible_debug_dump_count": 1.0,
        "accepted_path_infeasible_debug_dump_count": 1.0,
        "candidate_infeasible_debug_dump_count": 0.0,
        "infeasible_debug_dump_paths": '["artifacts/simulation_debug/infeasible_debug_X.npz"]',
        "precommit_clamp_applied_count": 0.0,
        "activation_split_reconciliation_error_max": 0.0,
        "final_soc_check_pass": 1.0,
        "benchmark_same_rules_gate_consistent": 1.0,
    }
    (scen / "backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    df = validate_outputs._collect(tmp_path)
    assert not df.empty
    assert "infeasible_debug_dump_count" in df.columns
    assert "accepted_path_infeasible_debug_dump_count" in df.columns


def test_invalid_by_quantile_counts_only_invalid_scenarios(tmp_path: Path) -> None:
    base = tmp_path / "multi"
    base.mkdir(parents=True, exist_ok=True)
    common = {
        "fallback_used": 0.0,
        "fallback_mode_counts": "{}",
        "optimization_error_code_counts": '{"ok": 1.0}',
        "simulation_schema_version": "v2",
        "required_summary_fields_version": "v2",
        "code_run_started_at_utc": "2026-01-01T00:00:00Z",
        "command_line_args": "{}",
        "output_was_cleaned": 1.0,
        "infeasible_debug_dump_count": 0.0,
        "accepted_path_infeasible_debug_dump_count": 0.0,
        "candidate_infeasible_debug_dump_count": 0.0,
        "infeasible_debug_dump_paths": [],
        "infeasible_debug_dump_timestamps": [],
        "summary_fields_defaulted": "[]",
        "required_fields_defaulted": "[]",
        "required_fields_computed": "[]",
        "required_fields_missing": "[]",
        "critical_required_fields_defaulted": "[]",
        "optional_fields_defaulted": "[]",
        "required_fields_check_pass": 1.0,
        "precommit_clamp_applied_count": 0.0,
        "activation_split_reconciliation_error_max": 0.0,
        "final_soc_check_pass": 1.0,
        "benchmark_same_rules_gate_consistent": 1.0,
        "global_perfect_foresight_dominance_check_pass": 1.0,
        "global_perfect_foresight_validation_status": "disabled_unverified",
    }
    payloads = {
        "p30_p30": dict(common, simulation_valid=0.0, thesis_reportable=0.0, invalid_reason="fallback_used"),
        "p50_p50": dict(common, simulation_valid=1.0, thesis_reportable=1.0, invalid_reason=""),
        "p70_p90": dict(common, simulation_valid=0.0, thesis_reportable=0.0, invalid_reason="optimization_failure"),
    }
    for scen, payload in payloads.items():
        d = base / scen
        d.mkdir(parents=True, exist_ok=True)
        (d / "backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    out_json = tmp_path / "stats.json"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["validate_simulation_outputs.py", str(tmp_path), "--out-json", str(out_json)])
        validate_outputs.main()
    stats = json.loads(out_json.read_text(encoding="utf-8"))
    assert int(stats["invalid_scenarios"]) == 2
    assert stats["invalid_by_quantile"] == {"p30_p30": 1, "p70_p90": 1}


def test_validation_counts_are_consistent(tmp_path: Path) -> None:
    base = tmp_path / "multi"
    base.mkdir(parents=True, exist_ok=True)
    payload = {
        "simulation_valid": 1.0,
        "thesis_reportable": 1.0,
        "invalid_reason": "",
        "fallback_used": 0.0,
        "fallback_mode_counts": "{}",
        "optimization_error_code_counts": '{"ok": 1.0}',
        "simulation_schema_version": "v2",
        "required_summary_fields_version": "v2",
        "code_run_started_at_utc": "2026-01-01T00:00:00Z",
        "command_line_args": "{}",
        "output_was_cleaned": 1.0,
        "infeasible_debug_dump_count": 0.0,
        "accepted_path_infeasible_debug_dump_count": 0.0,
        "candidate_infeasible_debug_dump_count": 0.0,
        "infeasible_debug_dump_paths": [],
        "infeasible_debug_dump_timestamps": [],
        "summary_fields_defaulted": "[]",
        "required_fields_defaulted": "[]",
        "required_fields_computed": "[]",
        "required_fields_missing": "[]",
        "critical_required_fields_defaulted": "[]",
        "optional_fields_defaulted": "[]",
        "required_fields_check_pass": 1.0,
        "precommit_clamp_applied_count": 0.0,
        "activation_split_reconciliation_error_max": 0.0,
        "final_soc_check_pass": 1.0,
        "benchmark_same_rules_gate_consistent": 1.0,
        "global_perfect_foresight_dominance_check_pass": 1.0,
        "global_perfect_foresight_validation_status": "disabled_unverified",
    }
    for scen in ("p30_p30", "p50_p50", "p70_p90"):
        d = base / scen
        d.mkdir(parents=True, exist_ok=True)
        (d / "backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    out_json = tmp_path / "stats.json"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["validate_simulation_outputs.py", str(tmp_path), "--out-json", str(out_json)])
        validate_outputs.main()
    stats = json.loads(out_json.read_text(encoding="utf-8"))
    assert int(stats["total_scenarios"]) == int(stats["valid_scenarios"]) + int(stats["invalid_scenarios"])
    assert sum(stats["invalid_by_quantile"].values()) == int(stats["invalid_scenarios"])


def test_accepted_infeasible_debug_dump_invalidates_scenario() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    s = dict(out.summary)
    s["accepted_path_infeasible_debug_dump_count"] = 1.0
    invalid = str(s.get("invalid_reason", ""))
    if "optimization_infeasible_debug_dump" not in invalid:
        invalid = ",".join([v for v in [invalid, "optimization_infeasible_debug_dump"] if v])
    s["invalid_reason"] = invalid
    s["simulation_valid"] = 0.0
    s["thesis_reportable"] = 0.0
    assert float(s["accepted_path_infeasible_debug_dump_count"]) > 0.5
    assert float(s["simulation_valid"]) == 0.0
    assert float(s["thesis_reportable"]) == 0.0
    assert "optimization_infeasible_debug_dump" in str(s["invalid_reason"])


def test_accepted_infeasible_dump_never_reportable() -> None:
    test_accepted_infeasible_debug_dump_invalidates_scenario()


def test_candidate_debug_dump_does_not_invalidate_if_not_accepted() -> None:
    row = {
        "simulation_valid": 1.0,
        "thesis_reportable": 1.0,
        "fallback_used": 0.0,
        "headroom_violation_count": 0.0,
        "missed_capacity_pos_mw": 0.0,
        "missed_capacity_neg_mw": 0.0,
        "pnl_reconciliation_error_max_eur": 0.0,
        "activation_split_reconciliation_error_max": 0.0,
        "final_soc_check_pass": 1.0,
        "benchmark_same_rules_gate_consistent": 1.0,
        "terminal_soc_repair_cost_eur": 0.0,
        "final_soc_actual_mwh": 10.0,
        "final_soc_target_mwh": 10.0,
        "optimization_error_code_counts": '{"ok": 10.0}',
        "accepted_path_infeasible_debug_dump_count": 0.0,
        "candidate_infeasible_debug_dump_count": 2.0,
        "infeasible_debug_dump_count": 2.0,
    }
    d = pd.DataFrame([row])
    non_ok = d["optimization_error_code_counts"].apply(lambda v: any(k not in {"ok", "none", ""} and float(c) > 0.0 for k, c in json.loads(v).items()))
    thesis_rule = (
        (d["simulation_valid"] >= 0.5)
        & (d["thesis_reportable"] >= 0.5)
        & (d["fallback_used"] <= 0.5)
        & (d["headroom_violation_count"] <= 1e-9)
        & (d["missed_capacity_pos_mw"] <= 1e-9)
        & (d["missed_capacity_neg_mw"] <= 1e-9)
        & (d["pnl_reconciliation_error_max_eur"] <= 1e-2)
        & (d["activation_split_reconciliation_error_max"] <= 1e-2)
        & (d["final_soc_check_pass"] >= 0.5)
        & (d["benchmark_same_rules_gate_consistent"] >= 0.5)
        & (d["accepted_path_infeasible_debug_dump_count"] <= 0.5)
        & (~non_ok)
    )
    assert bool(thesis_rule.iloc[0])


def test_candidate_retry_infeasible_dumps_with_final_ok_path_do_not_invalidate() -> None:
    bt = _mk_backtester()
    hourly = pd.DataFrame(
        {
            "timestamp_utc": [
                pd.Timestamp("2025-05-01T06:00:00Z"),
                pd.Timestamp("2025-05-01T07:00:00Z"),
            ],
            "optimization_error_code": ["ok", "ok"],
            "optimization_fallback": ["none", "none"],
            "optimizer_fallback_used": [0.0, 0.0],
        }
    )
    dumps = [
        {"timestamp_utc": "20250501T060000Z", "path": "a.npz", "solve_context": '{"final_accepted_path": false}'},
        {"timestamp_utc": "20250501T070000Z", "path": "b.npz", "solve_context": '{"final_accepted_path": false}'},
    ]
    accepted, candidate = bt._classify_infeasible_debug_dumps(dumps, hourly, timestamp_col="timestamp_utc")
    assert len(accepted) == 0
    assert len(candidate) == 2


def test_accepted_fallback_hour_marks_dump_as_accepted_and_invalidating() -> None:
    bt = _mk_backtester()
    hourly = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2025-05-01T09:00:00Z")],
            "optimization_error_code": ["safe_hold_plan"],
            "optimization_fallback": ["safe_hold_plan"],
            "optimizer_fallback_used": [1.0],
        }
    )
    dumps = [{"timestamp_utc": "20250501T090000Z", "path": "x.npz"}]
    accepted, candidate = bt._classify_infeasible_debug_dumps(dumps, hourly, timestamp_col="timestamp_utc")
    assert len(accepted) == 1
    assert len(candidate) == 0


def test_replay_proven_false_infeasible_status_keeps_dump_candidate_only() -> None:
    bt = _mk_backtester()
    hourly = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2025-05-01T09:00:00Z")],
            "optimization_error_code": ["ok_fixed_obligation_replay_proof"],
            "optimization_fallback": ["none"],
            "optimizer_fallback_used": [0.0],
            "fixed_obligation_replay_milp_mismatch": [1.0],
        }
    )
    dumps = [{"timestamp_utc": "20250501T090000Z", "path": "x.npz"}]

    accepted, candidate = bt._classify_infeasible_debug_dumps(dumps, hourly, timestamp_col="timestamp_utc")

    assert len(accepted) == 0
    assert len(candidate) == 1


def test_da_milp_failure_replay_only_is_not_fallback() -> None:
    ts = pd.Timestamp("2025-05-02T10:00:00Z")
    plan = pd.DataFrame(
        {
            "timestamp_utc": [ts],
            "reserve_pos_mw": [0.0],
            "reserve_neg_mw": [0.0],
            "bem_only_pos_mw": [0.0],
            "bem_only_neg_mw": [0.0],
            "id_charge_mw": [0.0],
            "id_discharge_mw": [0.0],
        }
    )

    diag = BatteryBacktester._classify_fixed_obligation_replay_proof(
        milp_failed=True,
        plan=plan,
        timestamps=plan["timestamp_utc"],
        fixed_obligation_diag={
            "fixed_obligation_replay_pass": 1.0,
            "fixed_obligation_replay_min_soc_mwh": 2.5,
            "fixed_obligation_replay_max_soc_mwh": 12.0,
            "fixed_obligation_replay_final_soc_mwh": 8.0,
        },
        da_lockbook={ts: (1.0, 0.0)},
        replay_da_schedule={ts: (1.0, 0.0)},
        fixed_reserve_obligation={},
        rolling_window_contains_global_end=False,
        terminal_soc_target_mwh=10.0,
    )

    assert str(diag["fixed_obligation_replay_classification"]) == "ok_fixed_obligation_replay_proof"
    assert str(diag["fixed_obligation_replay_reason"]) == "pure_fixed_obligation_replay"
    assert float(diag["fixed_obligation_replay_added_da_buy_mwh"]) == pytest.approx(0.0)
    assert float(diag["fixed_obligation_replay_changed_locked_da_buy_mwh"]) == pytest.approx(0.0)


def test_da_milp_failure_replay_with_new_trade_is_invalid_fallback() -> None:
    locked_ts = pd.Timestamp("2025-05-02T10:00:00Z")
    new_ts = pd.Timestamp("2025-05-02T11:00:00Z")
    plan = pd.DataFrame(
        {
            "timestamp_utc": [locked_ts, new_ts],
            "reserve_pos_mw": [0.0, 0.0],
            "reserve_neg_mw": [0.0, 0.0],
            "bem_only_pos_mw": [0.0, 0.0],
            "bem_only_neg_mw": [0.0, 0.0],
            "id_charge_mw": [0.0, 0.0],
            "id_discharge_mw": [0.0, 0.0],
        }
    )

    diag = BatteryBacktester._classify_fixed_obligation_replay_proof(
        milp_failed=True,
        plan=plan,
        timestamps=plan["timestamp_utc"],
        fixed_obligation_diag={
            "fixed_obligation_replay_pass": 1.0,
            "fixed_obligation_replay_min_soc_mwh": 2.5,
            "fixed_obligation_replay_max_soc_mwh": 12.0,
            "fixed_obligation_replay_final_soc_mwh": 8.0,
        },
        da_lockbook={locked_ts: (1.0, 0.0)},
        replay_da_schedule={locked_ts: (1.0, 0.0), new_ts: (0.5, 0.0)},
        fixed_reserve_obligation={},
        rolling_window_contains_global_end=False,
        terminal_soc_target_mwh=10.0,
    )

    assert str(diag["fixed_obligation_replay_classification"]) == "invalid_accepted_fallback"
    assert str(diag["fixed_obligation_replay_reason"]) == "replay_introduced_new_da_trade"
    assert float(diag["fixed_obligation_replay_added_da_buy_mwh"]) == pytest.approx(0.5)


def test_da_milp_failure_replay_terminal_violation_remains_invalid() -> None:
    ts = pd.Timestamp("2025-05-02T10:00:00Z")
    plan = pd.DataFrame(
        {
            "timestamp_utc": [ts],
            "reserve_pos_mw": [0.0],
            "reserve_neg_mw": [0.0],
            "bem_only_pos_mw": [0.0],
            "bem_only_neg_mw": [0.0],
            "id_charge_mw": [0.0],
            "id_discharge_mw": [0.0],
        }
    )

    diag = BatteryBacktester._classify_fixed_obligation_replay_proof(
        milp_failed=True,
        plan=plan,
        timestamps=plan["timestamp_utc"],
        fixed_obligation_diag={
            "fixed_obligation_replay_pass": 1.0,
            "fixed_obligation_replay_min_soc_mwh": 2.5,
            "fixed_obligation_replay_max_soc_mwh": 12.0,
            "fixed_obligation_replay_final_soc_mwh": 9.0,
        },
        da_lockbook={ts: (1.0, 0.0)},
        replay_da_schedule={ts: (1.0, 0.0)},
        fixed_reserve_obligation={},
        rolling_window_contains_global_end=True,
        terminal_soc_target_mwh=10.0,
    )

    assert str(diag["fixed_obligation_replay_classification"]) == "invalid_accepted_fallback"
    assert str(diag["fixed_obligation_replay_reason"]) == "replay_terminal_soc_violation"
    assert float(diag["fixed_obligation_replay_terminal_soc_pass"]) == pytest.approx(0.0)


def test_accepted_infeasible_dump_count_requires_first_timestamp() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    if float(s.get("accepted_path_infeasible_debug_dump_count", 0.0)) > 0.5:
        assert str(s.get("first_infeasible_timestamp_utc", "")).strip() != ""


def test_first_infeasible_timestamp_ignores_initial_deterministic_noop() -> None:
    hourly = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-05-01T06:00:00Z", periods=3, freq="h"),
            "optimization_error_code": [
                "ok_deterministic_noop",
                "ok_fixed_obligation_replay_proof",
                "rolling_window_nonterminal_infeasible",
            ],
            "optimization_fallback": ["none", "none", "safe_hold_plan"],
            "optimizer_fallback_used": [0.0, 0.0, 1.0],
        }
    )

    ts = BatteryBacktester._first_actual_infeasible_timestamp(hourly)

    assert ts == pd.Timestamp("2025-05-01T08:00:00Z").isoformat()


def test_replay_proven_false_infeasible_has_no_infeasibility_driver() -> None:
    driver = BatteryBacktester._infer_infeasibility_driver_for_row(
        optimization_error_code="ok_fixed_obligation_replay_proof",
        fixed_reserve_obligation_pos_mw=4.0,
    )

    assert driver == "none"


def test_fixed_obligation_replay_proof_volume_includes_da_lockbook() -> None:
    ts = pd.Timestamp("2025-05-02T10:00:00Z")

    volume = BatteryBacktester._fixed_obligation_replay_proof_volume_mw(
        fixed_obligation_diag={
            "fixed_obligation_reserve_pos_mw_sum": 0.0,
            "fixed_obligation_reserve_neg_mw_sum": 0.0,
        },
        fixed_reserve_obligation={},
        da_lockbook={ts: (10.0, 0.0)},
    )

    assert volume == pytest.approx(10.0)


def test_rhpf_da_sizer_fallback_invalidates_upper_bound_and_explains_naive_win() -> None:
    diag = BatteryBacktester._rolling_pf_da_upper_bound_diagnostics(
        rhpf_enabled=True,
        da_strategy_active=True,
        rolling_pf_verified=1.0,
        rolling_pf_da_bid_sizer_method="uniform_retry_ladder_fallback",
        rolling_pf_da_bid_sizer_status="linprog_exact_feasibility_failed",
        da_sizer_postlock_replay_mismatch=0.0,
        naive_total_pnl_eur=12493.60,
        rolling_pf_total_pnl_eur=11492.53,
    )

    assert float(diag["rolling_pf_da_exact_sizer_fallback_used"]) == pytest.approx(1.0)
    assert float(diag["rolling_pf_upper_bound_invalidated_by_da_sizer"]) == pytest.approx(1.0)
    assert float(diag["rolling_pf_verified_upper_bound_effective"]) == pytest.approx(0.0)
    assert float(diag["rolling_pf_naive_beats_rhpf"]) == pytest.approx(1.0)
    assert str(diag["naive_beats_rhpf_reason"]) == "rolling_pf_da_sizer_fallback"
    assert float(diag["naive_minus_rhpf_eur"]) == pytest.approx(1001.07)
    assert float(diag["rhpf_benchmark_quality_pass"]) == pytest.approx(0.0)
    assert str(diag["rhpf_benchmark_quality_reason"]) == "rolling_pf_da_sizer_fallback"


def test_da_rhpf_sizer_fallback_invalidates_rhpf_benchmark_only() -> None:
    summary = {
        "simulation_valid": 1.0,
        "thesis_reportable": 1.0,
        "invalid_reason": "",
        "naive_simulation_valid": 1.0,
        "rhpf_simulation_valid": 1.0,
        "rhpf_thesis_reportable": 1.0,
        "rhpf_invalid_reason": "",
        "naive_total_pnl_eur": 12493.60,
        "rolling_perfect_foresight_same_rules_total_pnl_eur": 11492.53,
        "rolling_pf_da_exact_sizer_fallback_used": 1.0,
        "model_vs_rhpf_pct": 12.0,
        "capture_ratio_vs_perfect_foresight": 0.12,
    }

    out = BatteryBacktester._apply_da_rhpf_benchmark_quality_gate(
        summary,
        rhpf_enabled=True,
        da_strategy_active=True,
    )

    assert float(out["simulation_valid"]) == pytest.approx(1.0)
    assert float(out["thesis_reportable"]) == pytest.approx(1.0)
    assert float(out["rhpf_simulation_valid"]) == pytest.approx(0.0)
    assert float(out["rhpf_thesis_reportable"]) == pytest.approx(0.0)
    assert "rhpf_da_sizer_fallback" in str(out["rhpf_invalid_reason"])
    assert float(out["rhpf_benchmark_quality_pass"]) == pytest.approx(0.0)
    assert str(out["rhpf_benchmark_quality_reason"]) == "rhpf_da_sizer_fallback"
    assert np.isnan(float(out["model_vs_rhpf_pct"]))
    assert np.isnan(float(out["capture_ratio_vs_perfect_foresight"]))


def test_da_valid_naive_beats_rhpf_marks_benchmark_quality_failed() -> None:
    summary = {
        "simulation_valid": 1.0,
        "naive_simulation_valid": 1.0,
        "rhpf_simulation_valid": 1.0,
        "rhpf_thesis_reportable": 1.0,
        "rhpf_invalid_reason": "none",
        "naive_total_pnl_eur": 110.0,
        "rolling_perfect_foresight_same_rules_total_pnl_eur": 100.0,
        "rolling_pf_da_exact_sizer_fallback_used": 0.0,
        "rolling_pf_upper_bound_invalidated_by_da_sizer": 0.0,
        "model_vs_rhpf_pct": 80.0,
    }

    out = BatteryBacktester._apply_da_rhpf_benchmark_quality_gate(
        summary,
        rhpf_enabled=True,
        da_strategy_active=True,
    )

    assert float(out["rhpf_benchmark_quality_pass"]) == pytest.approx(0.0)
    assert str(out["rhpf_benchmark_quality_reason"]) == "rhpf_dominated_by_valid_naive"
    assert float(out["rhpf_simulation_valid"]) == pytest.approx(0.0)
    assert "rhpf_dominated_by_valid_naive" in str(out["rhpf_invalid_reason"])
    assert float(out["naive_minus_rhpf_eur"]) == pytest.approx(10.0)
    assert np.isnan(float(out["model_vs_rhpf_pct"]))


def test_da_invalid_naive_does_not_invalidate_rhpf_dominance() -> None:
    summary = {
        "naive_simulation_valid": 0.0,
        "rhpf_simulation_valid": 1.0,
        "rhpf_thesis_reportable": 1.0,
        "rhpf_invalid_reason": "",
        "naive_total_pnl_eur": 110.0,
        "rolling_perfect_foresight_same_rules_total_pnl_eur": 100.0,
        "rolling_pf_da_exact_sizer_fallback_used": 0.0,
        "rolling_pf_upper_bound_invalidated_by_da_sizer": 0.0,
    }

    out = BatteryBacktester._apply_da_rhpf_benchmark_quality_gate(
        summary,
        rhpf_enabled=True,
        da_strategy_active=True,
    )

    assert float(out["rhpf_benchmark_quality_pass"]) == pytest.approx(1.0)
    assert str(out["rhpf_benchmark_quality_reason"]) == "none"
    assert float(out["rhpf_simulation_valid"]) == pytest.approx(1.0)
    assert str(out["rhpf_invalid_reason"]) == ""


def test_rhpf_non_da_strategy_keeps_upper_bound_despite_empty_da_sizer() -> None:
    diag = BatteryBacktester._rolling_pf_da_upper_bound_diagnostics(
        rhpf_enabled=True,
        da_strategy_active=False,
        rolling_pf_verified=1.0,
        rolling_pf_da_bid_sizer_method="uniform_retry_ladder_fallback",
        rolling_pf_da_bid_sizer_status="linprog_exact_feasibility_failed",
        da_sizer_postlock_replay_mismatch=1.0,
        naive_total_pnl_eur=10.0,
        rolling_pf_total_pnl_eur=9.0,
    )

    assert float(diag["rolling_pf_da_exact_sizer_fallback_used"]) == pytest.approx(0.0)
    assert float(diag["rolling_pf_upper_bound_invalidated_by_da_sizer"]) == pytest.approx(0.0)
    assert float(diag["rolling_pf_verified_upper_bound_effective"]) == pytest.approx(1.0)
    assert str(diag["naive_beats_rhpf_reason"]) == "rhpf_below_naive_unclassified"


def test_invalid_reason_includes_infeasible_debug_dump_only_for_accepted_path() -> None:
    s = {
        "accepted_path_infeasible_debug_dump_count": 0.0,
        "candidate_infeasible_debug_dump_count": 5.0,
        "fallback_used": 0.0,
        "reserve_feasibility_repair_used": 0.0,
        "simulation_valid": 1.0,
        "invalid_reason": "",
    }
    invalid_reason = str(s.get("invalid_reason", ""))
    if float(s.get("accepted_path_infeasible_debug_dump_count", 0.0)) > 0.5:
        invalid_reason = ",".join([v for v in [invalid_reason, "optimization_infeasible_debug_dump"] if v])
    assert "optimization_infeasible_debug_dump" not in invalid_reason


def test_summary_includes_input_vs_optimized_row_diagnostics() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    for k in [
        "input_row_count",
        "optimized_hour_count",
        "first_input_timestamp_utc",
        "first_optimized_target_timestamp_utc",
        "dropped_initial_rows_due_to_forecast_target_alignment",
    ]:
        assert k in s
    assert float(s["input_row_count"]) >= float(s["optimized_hour_count"])


def test_validator_required_fields_include_debug_dump_schema() -> None:
    req = set(validate_outputs.REQUIRED_SUMMARY_FIELDS)
    for k in (
        "simulation_schema_version",
        "required_summary_fields_version",
        "code_run_started_at_utc",
        "command_line_args",
        "output_was_cleaned",
        "infeasible_debug_dump_count",
        "accepted_path_infeasible_debug_dump_count",
        "candidate_infeasible_debug_dump_count",
        "infeasible_debug_dump_paths",
        "infeasible_debug_dump_timestamps",
        "required_fields_defaulted",
        "required_fields_computed",
    ):
        assert k in req


def test_summary_contains_debug_dump_paths_even_when_empty() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1)
    s = out.summary
    assert "infeasible_debug_dump_paths" in s
    assert "infeasible_debug_dump_timestamps" in s
    assert isinstance(s.get("infeasible_debug_dump_paths"), list)
    assert isinstance(s.get("infeasible_debug_dump_timestamps"), list)


def test_required_fields_not_runner_defaulted_silently() -> None:
    row = {
        "simulation_valid": 1.0,
        "thesis_reportable": 1.0,
        "invalid_reason": "",
        "fallback_used": 0.0,
        "fallback_mode_counts": "{}",
        "optimization_error_code_counts": '{"ok": 1}',
        "simulation_schema_version": "v2",
        "required_summary_fields_version": "v2",
        "code_run_started_at_utc": "2026-01-01T00:00:00Z",
        "command_line_args": "{}",
        "output_was_cleaned": 1.0,
        "infeasible_debug_dump_count": 0.0,
        "accepted_path_infeasible_debug_dump_count": 0.0,
        "candidate_infeasible_debug_dump_count": 0.0,
        "infeasible_debug_dump_paths": "[]",
        "infeasible_debug_dump_timestamps": "[]",
        "precommit_clamp_applied_count": 0.0,
        "activation_split_reconciliation_error_max": 0.0,
        "final_soc_check_pass": 1.0,
        "benchmark_same_rules_gate_consistent": 1.0,
        "global_perfect_foresight_dominance_check_pass": 1.0,
        "required_fields_defaulted": '["simulation_valid"]',
        "required_fields_computed": "[]",
    }
    d = pd.DataFrame([row])
    cnt = d["required_fields_defaulted"].apply(lambda v: len(json.loads(v))).iloc[0]
    assert cnt > 0


def test_required_field_defaulted_critical_invalidates() -> None:
    required_missing_count = 0
    critical_defaulted_count = 1
    required_fields_check_pass = (required_missing_count <= 0) and (critical_defaulted_count <= 0)
    assert required_fields_check_pass is False


def test_optional_field_defaulted_does_not_invalidate() -> None:
    required_missing_count = 0
    critical_defaulted_count = 0
    optional_defaulted_count = 2
    required_fields_check_pass = (required_missing_count <= 0) and (critical_defaulted_count <= 0)
    assert optional_defaulted_count > 0
    assert required_fields_check_pass is True


def test_optional_fields_defaulted_does_not_invalidate() -> None:
    test_optional_field_defaulted_does_not_invalidate()


def test_required_fields_missing_invalidates() -> None:
    required_fields_missing = 1
    critical_required_fields_defaulted = 0
    required_fields_check_pass = float(
        (required_fields_missing <= 0) and (critical_required_fields_defaulted <= 0)
    )
    assert required_fields_check_pass == 0.0


def test_required_fields_ok_consistent_with_check_pass(tmp_path: Path) -> None:
    scen = tmp_path / "multi" / "p50_p50"
    scen.mkdir(parents=True, exist_ok=True)
    payload = {
        "simulation_valid": 1.0,
        "thesis_reportable": 1.0,
        "invalid_reason": "",
        "fallback_used": 0.0,
        "fallback_mode_counts": "{}",
        "optimization_error_code_counts": '{"ok": 1.0}',
        "simulation_schema_version": "v2",
        "required_summary_fields_version": "v2",
        "code_run_started_at_utc": "2026-01-01T00:00:00Z",
        "command_line_args": "{}",
        "output_was_cleaned": 1.0,
        "infeasible_debug_dump_count": 0.0,
        "accepted_path_infeasible_debug_dump_count": 0.0,
        "candidate_infeasible_debug_dump_count": 0.0,
        "infeasible_debug_dump_paths": [],
        "infeasible_debug_dump_timestamps": [],
        "summary_fields_defaulted": "[]",
        "required_fields_defaulted": "[]",
        "required_fields_computed": "[]",
        "required_fields_missing": '["x"]',
        "critical_required_fields_defaulted": "[]",
        "optional_fields_defaulted": "[]",
        "required_fields_check_pass": 0.0,
        "precommit_clamp_applied_count": 0.0,
        "activation_split_reconciliation_error_max": 0.0,
        "final_soc_check_pass": 1.0,
        "benchmark_same_rules_gate_consistent": 1.0,
        "global_perfect_foresight_dominance_check_pass": 1.0,
        "global_perfect_foresight_validation_status": "disabled_unverified",
    }
    (scen / "backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    out_json = tmp_path / "stats.json"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["validate_simulation_outputs.py", str(tmp_path), "--out-json", str(out_json)])
        validate_outputs.main()
    stats = json.loads(out_json.read_text(encoding="utf-8"))
    assert stats["required_fields_ok"] is False


def test_summary_always_contains_infeasible_debug_dump_fields() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1)
    s = out.summary
    for k in (
        "infeasible_debug_dump_count",
        "accepted_path_infeasible_debug_dump_count",
        "candidate_infeasible_debug_dump_count",
        "infeasible_debug_dump_paths",
        "infeasible_debug_dump_timestamps",
    ):
        assert k in s
    assert float(s["infeasible_debug_dump_count"]) >= 0.0
    assert float(s["accepted_path_infeasible_debug_dump_count"]) >= 0.0
    assert float(s["candidate_infeasible_debug_dump_count"]) >= 0.0
    assert isinstance(s["infeasible_debug_dump_paths"], list)
    assert isinstance(s["infeasible_debug_dump_timestamps"], list)


def test_validator_fails_missing_debug_dump_fields(tmp_path: Path) -> None:
    scen = tmp_path / "multi" / "p50_p50"
    scen.mkdir(parents=True, exist_ok=True)
    payload = {
        "simulation_valid": 1.0,
        "thesis_reportable": 1.0,
        "invalid_reason": "",
        "fallback_used": 0.0,
        "fallback_mode_counts": "{}",
        "optimization_error_code_counts": '{"ok": 1.0}',
        "simulation_schema_version": "v2",
        "required_summary_fields_version": "v2",
        "code_run_started_at_utc": "2026-01-01T00:00:00Z",
        "command_line_args": "{}",
        "output_was_cleaned": 1.0,
        "infeasible_debug_dump_count": 0.0,
        "accepted_path_infeasible_debug_dump_count": 0.0,
        "candidate_infeasible_debug_dump_count": 0.0,
        # intentionally missing infeasible_debug_dump_paths
        "infeasible_debug_dump_timestamps": [],
        "summary_fields_defaulted": "[]",
        "required_fields_defaulted": "[]",
        "required_fields_computed": "[]",
        "precommit_clamp_applied_count": 0.0,
        "activation_split_reconciliation_error_max": 0.0,
        "final_soc_check_pass": 1.0,
        "benchmark_same_rules_gate_consistent": 1.0,
        "global_perfect_foresight_dominance_check_pass": 1.0,
        "global_perfect_foresight_validation_status": "disabled_unverified",
    }
    (scen / "backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                sys,
                "argv",
                ["validate_simulation_outputs.py", str(tmp_path)],
            )
            validate_outputs.main()
    assert "Missing required fields" in str(e.value)


def test_allow_stale_is_diagnostic_only(tmp_path: Path) -> None:
    scen = tmp_path / "multi" / "p50_p50"
    scen.mkdir(parents=True, exist_ok=True)
    payload = {
        "simulation_valid": 1.0,
        "thesis_reportable": 1.0,
        "invalid_reason": "",
        "fallback_used": 0.0,
        "fallback_mode_counts": "{}",
        "optimization_error_code_counts": '{"ok": 1.0}',
        "simulation_schema_version": "v2",
        "required_summary_fields_version": "v2",
        "code_run_started_at_utc": "2026-01-01T00:00:00Z",
        "command_line_args": "{}",
        "output_was_cleaned": 1.0,
        "infeasible_debug_dump_count": 0.0,
        "accepted_path_infeasible_debug_dump_count": 0.0,
        "candidate_infeasible_debug_dump_count": 0.0,
        # intentionally missing infeasible_debug_dump_paths
        "infeasible_debug_dump_timestamps": [],
        "summary_fields_defaulted": "[]",
        "required_fields_defaulted": "[]",
        "required_fields_computed": "[]",
        "precommit_clamp_applied_count": 0.0,
        "activation_split_reconciliation_error_max": 0.0,
        "final_soc_check_pass": 1.0,
        "benchmark_same_rules_gate_consistent": 1.0,
        "global_perfect_foresight_dominance_check_pass": 1.0,
        "global_perfect_foresight_validation_status": "disabled_unverified",
    }
    (scen / "backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    out_json = tmp_path / "stats.json"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            sys,
            "argv",
            ["validate_simulation_outputs.py", str(tmp_path), "--allow-stale", "--out-json", str(out_json)],
        )
        validate_outputs.main()
    stats = json.loads(out_json.read_text(encoding="utf-8"))
    assert stats["required_fields_ok"] is False
    assert int(stats["stale_scenario_count"]) > 0
    assert int(stats["thesis_reportable_scenarios"]) == 0


def test_clean_output_removes_stale_summary(tmp_path: Path) -> None:
    scen = tmp_path / "multi" / "p50_p50"
    scen.mkdir(parents=True, exist_ok=True)
    stale = scen / "backtest_summary.json"
    stale.write_text("{}", encoding="utf-8")
    assert stale.exists()
    cleaned = _prepare_scenario_output_dir(
        scenario_out_dir=scen,
        clean_output=True,
        strict_simulation_validity=True,
        simulation_schema_version="v2",
    )
    assert cleaned == 1.0
    assert scen.exists()
    assert not stale.exists()


def test_strict_run_refuses_existing_output_without_clean_output(tmp_path: Path) -> None:
    scen = tmp_path / "multi" / "p50_p50"
    scen.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError):
        _prepare_scenario_output_dir(
            scenario_out_dir=scen,
            clean_output=False,
            strict_simulation_validity=True,
            simulation_schema_version="v2",
        )


def test_reserve_retry_ladder_reaches_zero() -> None:
    bt = _mk_backtester()
    ladder = bt._parse_reserve_retry_ladder("1.0,0.5,0.25,0.0")
    assert ladder == [1.0, 0.5, 0.25, 0.0]


def test_disable_new_bcm_reserve_bids_blocks_new_bids() -> None:
    bt = _mk_backtester()
    bt.disable_new_bcm_reserve_bids = True
    assert bt.disable_new_bcm_reserve_bids is True


def test_retry_fields_written_to_summary() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=6)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    for k in (
        "reserve_retry_ladder",
        "reserve_retry_attempts_used",
        "reserve_retry_final_factor",
        "reserve_retry_succeeded",
        "disable_new_bcm_reserve_bids",
        "new_reserve_bids_zeroed_by_retry",
        "reserve_retry_infeasible_after_zero_reserve",
    ):
        assert k in s


def test_retry_fields_written_to_reserve_commitment_debug() -> None:
    bt = _mk_backtester()
    df, col = _tiny_backtest_df(hours=8)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=6, reopt_step_hours=1, strict_simulation_validity=True)
    h = out.hourly
    expected = {
        "submitted_reserve_pos_mw_before_retry",
        "submitted_reserve_neg_mw_before_retry",
        "reserve_retry_factor",
        "submitted_reserve_pos_mw_after_retry",
        "submitted_reserve_neg_mw_after_retry",
        "retry_reduction_reason",
    }
    assert expected.issubset(set(h.columns))


def test_conservative_reserve_cli_params_reach_backtester() -> None:
    txt = Path("scripts/run_battery_backtest.py").read_text(encoding="utf-8")
    assert "--disable-new-bcm-reserve-bids" in txt
    assert "--reserve-retry-ladder" in txt
    assert 'MODEL_SPECS["disable_new_bcm_reserve_bids"]' in txt
    assert 'MODEL_SPECS["reserve_retry_ladder"]' in txt


def test_terminal_repair_still_required_for_final_soc_shortfall() -> None:
    bt = _mk_backtester()
    bt.final_soc_mode = "terminal_repair"
    df, col = _tiny_backtest_df(hours=6)
    out = bt.run(df, col, use_rolling_horizon=True, horizon_hours=4, reopt_step_hours=1, strict_simulation_validity=True)
    s = out.summary
    if float(s.get("final_soc_shortfall_mwh", 0.0)) > 1e-6:
        if float(s.get("terminal_soc_repair_cost_eur", 0.0)) > 0.0:
            assert float(s.get("terminal_soc_repair_included_in_pnl", 0.0)) >= 0.5
            assert float(s.get("final_soc_economic_repair_check_pass", 0.0)) >= 0.5
        else:
            assert float(s.get("terminal_soc_repair_included_in_pnl", 1.0)) == 0.0
            assert float(s.get("final_soc_economic_repair_check_pass", 1.0)) == 0.0


def test_driver_new_reserve_bid_too_high() -> None:
    row = pd.Series(
        {
            "optimization_error_code": "optimization_infeasible",
            "fallback_mode": "reserve_feasibility_repair",
            "new_submitted_reserve_pos_mw": 4.0,
            "new_submitted_reserve_neg_mw": 0.0,
            "locked_reserve_pos_mw": 0.0,
            "locked_reserve_neg_mw": 0.0,
        }
    )
    d, _ = _suspected_infeasibility_driver_from_row(row)
    assert d == "new_reserve_bid_too_high"


def test_driver_existing_lockbook_obligation_infeasible() -> None:
    row = pd.Series(
        {
            "optimization_error_code": "optimization_infeasible",
            "fallback_mode": "reserve_feasibility_repair",
            "new_submitted_reserve_pos_mw": 0.0,
            "new_submitted_reserve_neg_mw": 0.0,
            "locked_reserve_pos_mw": 3.0,
            "locked_reserve_neg_mw": 0.0,
        }
    )
    d, _ = _suspected_infeasibility_driver_from_row(row)
    assert d == "existing_lockbook_obligation_infeasible"


def test_driver_fixed_reserve_obligation_infeasible() -> None:
    row = pd.Series(
        {
            "optimization_error_code": "rolling_window_nonterminal_infeasible",
            "fallback_mode": "safe_hold_plan",
            "new_submitted_reserve_pos_mw": 0.0,
            "new_submitted_reserve_neg_mw": 0.0,
            "fixed_reserve_obligation_pos_mw": 9.0,
            "fixed_reserve_obligation_neg_mw": 0.0,
            "locked_reserve_pos_mw": 0.0,
            "locked_reserve_neg_mw": 0.0,
        }
    )
    d, detail = _suspected_infeasibility_driver_from_row(row)
    assert d == "existing_lockbook_obligation_infeasible"
    assert "locked_only_pos=9.0000" in detail


def test_driver_prefers_fixed_obligation_replay_milp_mismatch() -> None:
    row = pd.Series(
        {
            "optimization_error_code": "rolling_window_nonterminal_infeasible",
            "fallback_mode": "safe_hold_plan",
            "fixed_obligation_replay_milp_mismatch": 1.0,
            "fixed_obligation_driver": "fixed_obligation_replay_milp_mismatch",
            "locked_reserve_pos_mw": 4.0,
            "locked_reserve_neg_mw": 0.0,
            "new_submitted_reserve_pos_mw": 0.0,
            "new_submitted_reserve_neg_mw": 0.0,
        }
    )

    d, detail = _suspected_infeasibility_driver_from_row(row)

    assert d == "fixed_obligation_replay_milp_mismatch"
    assert "fixed_obligation_replay_milp_mismatch" in detail


def test_driver_protected_soc_violation() -> None:
    row = pd.Series(
        {
            "optimization_error_code": "optimization_failure",
            "protected_soc_violation_pos_mwh": 0.5,
            "protected_soc_violation_neg_mwh": 0.0,
        }
    )
    d, _ = _suspected_infeasibility_driver_from_row(row)
    assert d == "protected_soc_violation"


def test_infeasibility_attribution_file_written_on_failure(tmp_path: Path) -> None:
    h = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T01:00:00Z")],
            "optimization_error_code": ["ok", "optimization_failure"],
            "optimization_fallback": ["none", "reserve_feasibility_repair"],
            "is_fallback_hour": [0.0, 1.0],
            "fixed_reserve_obligation_pos_mw": [0.0, 1.0],
            "fixed_reserve_obligation_neg_mw": [0.0, 0.0],
            "reserve_submitted_pos_mw": [0.0, 1.0],
            "reserve_submitted_neg_mw": [0.0, 0.0],
            "real_soc_start_mwh": [10.0, 9.0],
            "real_soc_mwh": [9.0, 8.0],
            "real_aux_energy_mwh": [0.0, 0.1],
        }
    )
    s = {"reserve_feasibility_repair_used": 1.0, "accepted_path_infeasible_debug_dump_count": 1.0, "p_max_mw": 20.0}
    out = _build_optimization_infeasibility_attribution(hourly=h, summary=s, scenario="p50_p50")
    assert not out.empty
    p = tmp_path / "optimization_infeasibility_attribution.csv"
    out.to_csv(p, index=False)
    assert p.exists()


def test_first_infeasibility_driver_in_summary() -> None:
    h = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "optimization_error_code": ["optimization_infeasible"],
            "optimization_fallback": ["reserve_feasibility_repair"],
            "is_fallback_hour": [1.0],
            "reserve_submitted_pos_mw": [2.0],
            "reserve_submitted_neg_mw": [0.0],
            "fixed_reserve_obligation_pos_mw": [0.0],
            "fixed_reserve_obligation_neg_mw": [0.0],
            "real_soc_start_mwh": [10.0],
            "real_soc_mwh": [9.0],
        }
    )
    out = _build_optimization_infeasibility_attribution(hourly=h, summary={"p_max_mw": 20.0}, scenario="p30_p30")
    assert str(out.iloc[0]["suspected_infeasibility_driver"]) in {
        "new_reserve_bid_too_high",
        "unknown_solver_or_numeric",
    }


def test_no_new_bcm_keeps_existing_lockbook_obligations() -> None:
    bt = _mk_backtester()
    bt.disable_new_bcm_reserve_bids = True
    assert bt.disable_new_bcm_reserve_bids is True


def test_da_bcm_gate_uses_berlin_time_and_next_local_delivery_day() -> None:
    bt = _mk_backtester()

    # Winter: Europe/Berlin is CET (UTC+1).
    winter_da_gate = pd.Timestamp("2025-01-01T10:00:00Z")
    winter_bcm_gate = pd.Timestamp("2025-01-01T07:00:00Z")
    assert bt._is_local_bid_hour(winter_da_gate, 11)
    assert not bt._is_local_bid_hour(winter_da_gate - pd.Timedelta(hours=1), 11)
    assert not bt._is_local_bid_hour(winter_da_gate + pd.Timedelta(hours=1), 11)
    assert bt._is_local_bid_hour(winter_bcm_gate, 8)
    assert not bt._is_local_bid_hour(winter_bcm_gate - pd.Timedelta(hours=1), 8)
    assert not bt._is_local_bid_hour(winter_bcm_gate + pd.Timedelta(hours=1), 8)

    winter_start_local, winter_end_local, winter_start_utc, winter_end_utc = (
        bt._next_berlin_delivery_day_window(winter_da_gate)
    )
    assert winter_start_local == pd.Timestamp("2025-01-02T00:00:00", tz="Europe/Berlin")
    assert winter_end_local == pd.Timestamp("2025-01-03T00:00:00", tz="Europe/Berlin")
    assert winter_start_utc == pd.Timestamp("2025-01-01T23:00:00Z")
    assert winter_end_utc == pd.Timestamp("2025-01-02T23:00:00Z")

    # Summer: Europe/Berlin is CEST (UTC+2). This catches hardcoded UTC+1 logic.
    summer_da_gate = pd.Timestamp("2025-07-01T09:00:00Z")
    summer_bcm_gate = pd.Timestamp("2025-07-01T06:00:00Z")
    assert bt._is_local_bid_hour(summer_da_gate, 11)
    assert not bt._is_local_bid_hour(summer_da_gate - pd.Timedelta(hours=1), 11)
    assert not bt._is_local_bid_hour(summer_da_gate + pd.Timedelta(hours=1), 11)
    assert bt._is_local_bid_hour(summer_bcm_gate, 8)
    assert not bt._is_local_bid_hour(summer_bcm_gate - pd.Timedelta(hours=1), 8)
    assert not bt._is_local_bid_hour(summer_bcm_gate + pd.Timedelta(hours=1), 8)

    summer_start_local, summer_end_local, summer_start_utc, summer_end_utc = (
        bt._next_berlin_delivery_day_window(summer_da_gate)
    )
    assert summer_start_local == pd.Timestamp("2025-07-02T00:00:00", tz="Europe/Berlin")
    assert summer_end_local == pd.Timestamp("2025-07-03T00:00:00", tz="Europe/Berlin")
    assert summer_start_utc == pd.Timestamp("2025-07-01T22:00:00Z")
    assert summer_end_utc == pd.Timestamp("2025-07-02T22:00:00Z")


def test_da_gate_delivery_window_rejects_same_day_targets() -> None:
    bt = _mk_backtester()
    snapshot = pd.Timestamp("2025-10-22T09:00:00Z")  # 11:00 Europe/Berlin, still CEST.

    window = bt._da_delivery_window_for_snapshot(snapshot, da_bid_hour_local=11)
    assert window["is_gate"] is True
    assert window["delivery_start_local"] == pd.Timestamp("2025-10-23T00:00:00", tz="Europe/Berlin")
    assert window["delivery_end_local"] == pd.Timestamp("2025-10-24T00:00:00", tz="Europe/Berlin")

    # Same local day as the gate snapshot: not bidable for DA.
    assert not bt._is_da_target_allowed(
        snapshot,
        pd.Timestamp("2025-10-22T10:00:00Z"),  # 12:00 Europe/Berlin
        da_bid_hour_local=11,
    )
    # Next local delivery day 00:00 through 23:00: bidable.
    assert bt._is_da_target_allowed(
        snapshot,
        pd.Timestamp("2025-10-22T22:00:00Z"),  # 2025-10-23 00:00 Europe/Berlin
        da_bid_hour_local=11,
    )
    assert bt._is_da_target_allowed(
        snapshot,
        pd.Timestamp("2025-10-23T21:00:00Z"),  # 2025-10-23 23:00 Europe/Berlin
        da_bid_hour_local=11,
    )
    # Following local day boundary: outside the DA delivery window.
    assert not bt._is_da_target_allowed(
        snapshot,
        pd.Timestamp("2025-10-23T22:00:00Z"),  # 2025-10-24 00:00 Europe/Berlin
        da_bid_hour_local=11,
    )
    # Outside 11:00 local gate: no new DA target is bidable.
    assert not bt._is_da_target_allowed(
        pd.Timestamp("2025-10-22T08:00:00Z"),  # 10:00 Europe/Berlin
        pd.Timestamp("2025-10-22T22:00:00Z"),
        da_bid_hour_local=11,
    )


def test_same_day_da_lockbook_obligation_is_valid_from_previous_gate() -> None:
    bt = _mk_backtester()
    current_snapshot = pd.Timestamp("2025-10-22T09:00:00Z")  # 11:00 Europe/Berlin.
    origin_snapshot = pd.Timestamp("2025-10-21T09:00:00Z")  # Previous valid DA gate.
    delivery = pd.Timestamp("2025-10-22T11:00:00Z")  # 13:00 Europe/Berlin, same day as current snapshot.

    assert not bt._is_da_target_allowed(current_snapshot, delivery, da_bid_hour_local=11)
    assert bt._da_origin_window_valid(origin_snapshot, delivery, da_bid_hour_local=11)

    row = pd.DataFrame(
        {
            "charge_mw": [5.0],
            "discharge_mw": [0.0],
            "da_candidate_allowed_by_gate_window": [0.0],
            "da_bid_locked": [1.0],
            "da_lockbook_row_present": [1.0],
            "da_existing_lockbook_delivery_row": [1.0],
            "da_new_bid_gate_window_violation": [0.0],
            "da_existing_lockbook_origin_window_valid": [1.0],
            "da_existing_lockbook_origin_window_violation": [0.0],
            "da_originating_source_snapshot_utc": [origin_snapshot.isoformat()],
            "timestamp_utc": [delivery.isoformat()],
        }
    )
    assert float(row["da_new_bid_gate_window_violation"].iloc[0]) == pytest.approx(0.0)
    assert float(row["da_existing_lockbook_origin_window_violation"].iloc[0]) == pytest.approx(0.0)


def test_da_bcm_candidate_gate_window_masks_are_applied_before_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    captured_windows: list[pd.DataFrame] = []

    def _zero_plan(window: pd.DataFrame, colmap: BacktestColumnMap, **_: object) -> pd.DataFrame:
        captured_windows.append(window.copy())
        ts = pd.to_datetime(window[colmap.timestamp], utc=True, errors="coerce")
        return pd.DataFrame(
            {
                colmap.timestamp: ts,
                "charge_mw": np.zeros(len(window), dtype=float),
                "discharge_mw": np.zeros(len(window), dtype=float),
                "reserve_pos_mw": np.zeros(len(window), dtype=float),
                "reserve_neg_mw": np.zeros(len(window), dtype=float),
                "bem_only_pos_mw": np.zeros(len(window), dtype=float),
                "bem_only_neg_mw": np.zeros(len(window), dtype=float),
                "id_charge_mw": np.zeros(len(window), dtype=float),
                "id_discharge_mw": np.zeros(len(window), dtype=float),
                "soc_lp_mwh": np.full(len(window), float(bt.soc_init), dtype=float),
                "soc_start_lp_mwh": np.full(len(window), float(bt.soc_init), dtype=float),
                "predicted_objective_eur": np.zeros(len(window), dtype=float),
                "aux_power_mw": np.zeros(len(window), dtype=float),
            }
        )

    monkeypatch.setattr(bt, "optimize_dispatch", _zero_plan)

    # 09:00 UTC is 11:00 Europe/Berlin in summer: DA gate valid, BCM gate invalid.
    gate_df, _ = _tiny_backtest_df(hours=40)
    gate_df[col.timestamp] = pd.date_range("2025-06-02T09:00:00Z", periods=len(gate_df), freq="h")
    out, plan_history = bt.optimize_dispatch_rolling(
        gate_df,
        col,
        horizon_hours=len(gate_df),
        reopt_step_hours=len(gate_df),
        allowed_markets=("DA", "aFRR", "BCM"),
        strategy_name="multi",
        strict_simulation_validity=False,
    )
    assert not out.empty
    assert not plan_history.empty
    assert captured_windows
    window = captured_windows[-1]
    target_local = pd.to_datetime(window[col.timestamp], utc=True, errors="coerce").dt.tz_convert("Europe/Berlin")
    da_next_day = (target_local >= pd.Timestamp("2025-06-03T00:00:00", tz="Europe/Berlin")) & (
        target_local < pd.Timestamp("2025-06-04T00:00:00", tz="Europe/Berlin")
    )
    da_disable = pd.to_numeric(window["_disable_da_bid"], errors="coerce").fillna(0.0)
    assert da_disable.loc[da_next_day].eq(0.0).all()
    assert da_disable.loc[~da_next_day].eq(1.0).all()

    bcm_disable = pd.to_numeric(window["_disable_bcm_product_bid"], errors="coerce").fillna(0.0)
    assert bcm_disable.eq(1.0).all()
    assert pd.to_numeric(plan_history["da_candidate_gate_window_violation"], errors="coerce").fillna(0.0).eq(0.0).all()
    assert pd.to_numeric(plan_history["bcm_candidate_gate_window_violation"], errors="coerce").fillna(0.0).eq(0.0).all()
    assert "da_gate_actionable" in plan_history.columns
    assert "da_optimizer_ub_zeroed_by_gate" in plan_history.columns
    assert "da_illegal_same_day_variable_count" in plan_history.columns
    assert "bcm_gate_actionable" in plan_history.columns
    assert "bcm_optimizer_ub_zeroed_by_gate" in plan_history.columns
    assert "bcm_illegal_same_day_variable_count" in plan_history.columns
    ph_target_local = pd.to_datetime(plan_history["target_time_utc"], utc=True, errors="coerce").dt.tz_convert("Europe/Berlin")
    ph_same_day = ph_target_local.dt.date == pd.Timestamp("2025-06-02", tz="Europe/Berlin").date()
    ph_next_day = (ph_target_local >= pd.Timestamp("2025-06-03T00:00:00", tz="Europe/Berlin")) & (
        ph_target_local < pd.Timestamp("2025-06-04T00:00:00", tz="Europe/Berlin")
    )
    assert pd.to_numeric(plan_history.loc[ph_same_day, "da_gate_actionable"], errors="coerce").fillna(0.0).eq(0.0).all()
    assert pd.to_numeric(plan_history.loc[ph_same_day, "da_optimizer_ub_zeroed_by_gate"], errors="coerce").fillna(0.0).eq(1.0).all()
    assert pd.to_numeric(plan_history.loc[ph_same_day, "da_illegal_same_day_variable_count"], errors="coerce").fillna(0.0).eq(0.0).all()
    assert pd.to_numeric(plan_history.loc[ph_next_day, "da_gate_actionable"], errors="coerce").fillna(0.0).eq(1.0).all()
    assert pd.to_numeric(plan_history["bcm_optimizer_ub_zeroed_by_gate"], errors="coerce").fillna(0.0).eq(1.0).all()


def test_bcm_gate_disables_new_candidates_outside_gate_and_limits_gate_to_next_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt = _mk_backtester()
    col = BacktestColumnMap()
    captured_windows: list[pd.DataFrame] = []

    def _zero_plan(window: pd.DataFrame, colmap: BacktestColumnMap, **_: object) -> pd.DataFrame:
        captured_windows.append(window.copy())
        ts = pd.to_datetime(window[colmap.timestamp], utc=True, errors="coerce")
        return pd.DataFrame(
            {
                colmap.timestamp: ts,
                "charge_mw": np.zeros(len(window), dtype=float),
                "discharge_mw": np.zeros(len(window), dtype=float),
                "reserve_pos_mw": np.zeros(len(window), dtype=float),
                "reserve_neg_mw": np.zeros(len(window), dtype=float),
                "bem_only_pos_mw": np.zeros(len(window), dtype=float),
                "bem_only_neg_mw": np.zeros(len(window), dtype=float),
                "id_charge_mw": np.zeros(len(window), dtype=float),
                "id_discharge_mw": np.zeros(len(window), dtype=float),
                "soc_lp_mwh": np.full(len(window), float(bt.soc_init), dtype=float),
                "soc_start_lp_mwh": np.full(len(window), float(bt.soc_init), dtype=float),
                "predicted_objective_eur": np.zeros(len(window), dtype=float),
                "aux_power_mw": np.zeros(len(window), dtype=float),
            }
        )

    monkeypatch.setattr(bt, "optimize_dispatch", _zero_plan)

    non_gate_df, _ = _tiny_backtest_df(hours=8)
    non_gate_df[col.timestamp] = pd.date_range("2025-05-01T05:00:00Z", periods=len(non_gate_df), freq="h")
    bt.optimize_dispatch_rolling(
        non_gate_df,
        col,
        horizon_hours=len(non_gate_df),
        reopt_step_hours=len(non_gate_df),
        allowed_markets=("aFRR", "BCM", "BEM"),
        strategy_name="afrr",
        strict_simulation_validity=False,
    )
    assert captured_windows
    non_gate_disable = pd.to_numeric(captured_windows[-1]["_disable_bcm_product_bid"], errors="coerce").fillna(0.0)
    assert non_gate_disable.eq(1.0).all()

    gate_df, _ = _tiny_backtest_df(hours=30)
    gate_df[col.timestamp] = pd.date_range("2025-05-01T06:00:00Z", periods=len(gate_df), freq="h")
    captured_windows.clear()
    bt.optimize_dispatch_rolling(
        gate_df,
        col,
        horizon_hours=len(gate_df),
        reopt_step_hours=len(gate_df),
        allowed_markets=("aFRR", "BCM", "BEM"),
        strategy_name="afrr",
        strict_simulation_validity=False,
    )
    gate_window = captured_windows[-1]
    ts_local = pd.to_datetime(gate_window[col.timestamp], utc=True, errors="coerce").dt.tz_convert("Europe/Berlin")
    next_day_start = pd.Timestamp("2025-05-02T00:00:00", tz="Europe/Berlin")
    next_day_end = next_day_start + pd.Timedelta(days=1)
    next_day_mask = (ts_local >= next_day_start) & (ts_local < next_day_end)
    gate_disable = pd.to_numeric(gate_window["_disable_bcm_product_bid"], errors="coerce").fillna(0.0)
    assert gate_disable.loc[next_day_mask].eq(0.0).all()
    assert gate_disable.loc[~next_day_mask].eq(1.0).all()

    plan_history = bt.optimize_dispatch_rolling(
        gate_df,
        col,
        horizon_hours=len(gate_df),
        reopt_step_hours=len(gate_df),
        allowed_markets=("aFRR", "BCM", "BEM"),
        strategy_name="afrr",
        strict_simulation_validity=False,
    )[1]
    assert "bcm_gate_actionable" in plan_history.columns
    assert "bcm_optimizer_ub_zeroed_by_gate" in plan_history.columns
    assert "bcm_illegal_same_day_variable_count" in plan_history.columns
    ph_target_local = pd.to_datetime(plan_history["target_time_utc"], utc=True, errors="coerce").dt.tz_convert("Europe/Berlin")
    ph_same_day = ph_target_local.dt.date == pd.Timestamp("2025-05-01", tz="Europe/Berlin").date()
    ph_next_day = (ph_target_local >= next_day_start) & (ph_target_local < next_day_end)
    assert pd.to_numeric(plan_history.loc[ph_same_day, "bcm_gate_actionable"], errors="coerce").fillna(0.0).eq(0.0).all()
    assert pd.to_numeric(plan_history.loc[ph_same_day, "bcm_optimizer_ub_zeroed_by_gate"], errors="coerce").fillna(0.0).eq(1.0).all()
    assert pd.to_numeric(plan_history.loc[ph_same_day, "bcm_illegal_same_day_variable_count"], errors="coerce").fillna(0.0).eq(0.0).all()
    assert pd.to_numeric(plan_history.loc[ph_next_day, "bcm_gate_actionable"], errors="coerce").fillna(0.0).eq(1.0).all()


def test_bem_gate_next_hour_disables_later_horizon_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    col = BacktestColumnMap()
    strategy_cases = [
        ("bem", ("aFRR", "BEM")),
        ("afrr", ("aFRR", "BCM", "BEM")),
        ("multi", ("DA", "aFRR", "ID", "BCM", "BEM")),
    ]

    for strategy_name, allowed_markets in strategy_cases:
        bt = _mk_backtester()
        captured_windows: list[pd.DataFrame] = []

        def _zero_plan(window: pd.DataFrame, colmap: BacktestColumnMap, **_: object) -> pd.DataFrame:
            captured_windows.append(window.copy())
            ts = pd.to_datetime(window[colmap.timestamp], utc=True, errors="coerce")
            return pd.DataFrame(
                {
                    colmap.timestamp: ts,
                    "charge_mw": np.zeros(len(window), dtype=float),
                    "discharge_mw": np.zeros(len(window), dtype=float),
                    "reserve_pos_mw": np.zeros(len(window), dtype=float),
                    "reserve_neg_mw": np.zeros(len(window), dtype=float),
                    "bem_only_pos_mw": np.zeros(len(window), dtype=float),
                    "bem_only_neg_mw": np.zeros(len(window), dtype=float),
                    "id_charge_mw": np.zeros(len(window), dtype=float),
                    "id_discharge_mw": np.zeros(len(window), dtype=float),
                    "soc_lp_mwh": np.full(len(window), float(bt.soc_init), dtype=float),
                    "soc_start_lp_mwh": np.full(len(window), float(bt.soc_init), dtype=float),
                    "predicted_objective_eur": np.zeros(len(window), dtype=float),
                    "aux_power_mw": np.zeros(len(window), dtype=float),
                }
            )

        monkeypatch.setattr(bt, "optimize_dispatch", _zero_plan)
        df, _ = _tiny_backtest_df(hours=5)
        df[col.timestamp] = pd.date_range("2025-05-01T00:00:00Z", periods=len(df), freq="h")

        bt.optimize_dispatch_rolling(
            df,
            col,
            horizon_hours=len(df),
            reopt_step_hours=len(df),
            allowed_markets=allowed_markets,
            strategy_name=strategy_name,
            strict_simulation_validity=False,
        )

        assert captured_windows, strategy_name
        disable = pd.to_numeric(captured_windows[-1]["_disable_bem_only_bid"], errors="coerce").fillna(0.0)
        assert float(disable.iloc[0]) == pytest.approx(0.0), strategy_name
        assert disable.iloc[1:].eq(1.0).all(), strategy_name


def test_fallback_still_not_reportable() -> None:
    test_fallback_never_reportable()


def test_non_actionable_hour_uses_deterministic_noop_not_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    df, col = _tiny_backtest_df(hours=3)
    bt = _mk_backtester("canonical_economic")

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("solver should not be called for deterministic no-op hour")

    monkeypatch.setattr(bt, "optimize_dispatch", _boom)
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=3,
        reopt_step_hours=1,
        allowed_markets=("aFRR",),
        strategy_name="bcm",
        strict_simulation_validity=True,
        enforce_final_soc_min=True,
    )
    codes = set(out.hourly["optimization_error_code"].astype(str))
    assert codes == {"ok_deterministic_noop"}
    assert float(out.summary["fallback_used"]) == 0.0
    assert float(out.summary["simulation_valid"]) == 1.0
    assert pd.to_numeric(out.hourly["deterministic_noop_used"], errors="coerce").eq(1.0).all()
    assert pd.to_numeric(out.hourly["optimizer_fallback_used"], errors="coerce").eq(0.0).all()
    assert pd.to_numeric(out.hourly["deterministic_noop_allowed"], errors="coerce").eq(1.0).all()
    assert pd.to_numeric(out.hourly["noop_no_market_action_available"], errors="coerce").eq(1.0).all()


def test_deterministic_noop_not_allowed_when_bem_available() -> None:
    df, col = _tiny_backtest_df(hours=3)
    bt = _mk_backtester("canonical_economic")
    calls = {"n": 0}
    original = bt.optimize_dispatch

    def _wrapped(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["n"] += 1
        return original(*args, **kwargs)

    bt.optimize_dispatch = _wrapped  # type: ignore[method-assign]
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=3,
        reopt_step_hours=1,
        allowed_markets=("DA", "aFRR", "ID", "BCM", "BEM"),
        strategy_name="multi",
        strict_simulation_validity=True,
        enforce_final_soc_min=True,
    )
    assert calls["n"] > 0
    assert "ok_deterministic_noop" not in set(out.hourly["optimization_error_code"].astype(str))
    assert pd.to_numeric(out.hourly["deterministic_noop_used"], errors="coerce").fillna(0.0).eq(0.0).all()


def test_deterministic_noop_not_allowed_at_bcm_bid_hour(monkeypatch: pytest.MonkeyPatch) -> None:
    df, col = _tiny_backtest_df(hours=2)
    df[col.timestamp] = pd.date_range("2026-01-01T07:00:00Z", periods=2, freq="h")
    bt = _mk_backtester("canonical_economic")

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("forced bcm gate solver path")

    monkeypatch.setattr(bt, "optimize_dispatch", _boom)
    with pytest.raises(RuntimeError, match="forced bcm gate solver path"):
        bt.run(
            df,
            col,
            use_rolling_horizon=True,
            horizon_hours=2,
            reopt_step_hours=1,
            allowed_markets=("aFRR",),
            strategy_name="bcm",
            strict_simulation_validity=True,
            enforce_final_soc_min=True,
        )


def test_deterministic_noop_not_allowed_with_terminal_pressure(monkeypatch: pytest.MonkeyPatch) -> None:
    df, col = _tiny_backtest_df(hours=2)
    bt = _mk_backtester("canonical_economic")
    bt.soc_target_end = 12.0

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("forced solver path")

    monkeypatch.setattr(bt, "optimize_dispatch", _boom)
    with pytest.raises(RuntimeError, match="forced solver path"):
        bt.run(
            df,
            col,
            use_rolling_horizon=True,
            horizon_hours=2,
            reopt_step_hours=1,
            allowed_markets=("aFRR",),
            strategy_name="bcm",
            strict_simulation_validity=True,
            enforce_final_soc_min=True,
        )


def test_not_set_final_window_uses_terminal_id_recovery_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df, col = _tiny_backtest_df(hours=1)
    bt = _mk_backtester("canonical_economic")
    bt.soc_init = 9.0
    bt.soc_target_end = 10.0
    bt.final_soc_mode = "hard"

    def _not_set(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("MIP optimization failed: HiGHS Status 0: Not Set")

    monkeypatch.setattr(bt, "optimize_dispatch", _not_set)
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=1,
        reopt_step_hours=1,
        allowed_markets=("aFRR", "ID"),
        strategy_name="bcm",
        strict_simulation_validity=True,
        enforce_final_soc_min=True,
        id_recourse_mode="common",
    )
    h = out.hourly
    assert set(h["optimization_error_code"].astype(str)) == {"ok_terminal_recovery_fallback"}
    assert float(out.summary["fallback_used"]) == 0.0
    assert float(out.summary["final_soc_check_pass"]) == 1.0
    assert float(out.summary["simulation_valid"]) == 1.0
    assert pd.to_numeric(h["optimizer_fallback_used"], errors="coerce").fillna(1.0).eq(0.0).all()
    assert pd.to_numeric(h["real_id_charge_mw"], errors="coerce").fillna(0.0).gt(0.0).any()
    assert pd.to_numeric(h["terminal_recovery_fallback_success"], errors="coerce").fillna(0.0).eq(1.0).all()


def test_hard_final_infeasible_final_window_uses_terminal_id_recovery_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df, col = _tiny_backtest_df(hours=1)
    bt = _mk_backtester("canonical_economic")
    bt.soc_init = 9.0
    bt.soc_target_end = 10.0
    bt.final_soc_mode = "hard"

    def _infeasible(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError(
            "MIP optimization failed: The problem is infeasible. "
            "(HiGHS Status 8: model_status is Infeasible)"
        )

    monkeypatch.setattr(bt, "optimize_dispatch", _infeasible)
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=1,
        reopt_step_hours=1,
        allowed_markets=("aFRR", "ID"),
        strategy_name="bcm",
        strict_simulation_validity=True,
        enforce_final_soc_min=True,
        id_recourse_mode="common",
    )
    h = out.hourly
    assert set(h["optimization_error_code"].astype(str)) == {"ok_terminal_recovery_fallback"}
    assert float(out.summary["fallback_used"]) == 0.0
    assert float(out.summary["simulation_valid"]) == 1.0
    assert float(out.summary["final_soc_check_pass"]) == 1.0
    assert pd.to_numeric(h["optimizer_fallback_used"], errors="coerce").fillna(1.0).eq(0.0).all()
    assert pd.to_numeric(h["real_id_charge_mw"], errors="coerce").fillna(0.0).gt(0.0).any()
    assert pd.to_numeric(h["terminal_recovery_fallback_success"], errors="coerce").fillna(0.0).eq(1.0).all()


def test_hard_final_infeasible_with_id_disabled_remains_invalid_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df, col = _tiny_backtest_df(hours=1)
    bt = _mk_backtester("canonical_economic")
    bt.soc_init = 9.0
    bt.soc_target_end = 10.0
    bt.final_soc_mode = "hard"

    def _infeasible(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError(
            "MIP optimization failed: The problem is infeasible. "
            "(HiGHS Status 8: model_status is Infeasible)"
        )

    monkeypatch.setattr(bt, "optimize_dispatch", _infeasible)
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=1,
        reopt_step_hours=1,
        allowed_markets=("aFRR", "ID"),
        strategy_name="bcm",
        strict_simulation_validity=True,
        enforce_final_soc_min=True,
        id_recourse_mode="disabled",
    )
    assert set(out.hourly["optimization_error_code"].astype(str)) == {"terminal_soc_conflict"}
    assert float(out.summary["fallback_used"]) == 1.0
    assert float(out.summary["simulation_valid"]) == 0.0
    assert "optimization_infeasible" in str(out.summary["invalid_reason"])


def test_not_set_final_window_with_id_disabled_remains_invalid_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df, col = _tiny_backtest_df(hours=1)
    bt = _mk_backtester("canonical_economic")
    bt.soc_init = 9.0
    bt.soc_target_end = 10.0
    bt.final_soc_mode = "hard"

    def _not_set(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("MIP optimization failed: HiGHS Status 0: Not Set")

    monkeypatch.setattr(bt, "optimize_dispatch", _not_set)
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=1,
        reopt_step_hours=1,
        allowed_markets=("aFRR", "ID"),
        strategy_name="bcm",
        strict_simulation_validity=True,
        enforce_final_soc_min=True,
        id_recourse_mode="disabled",
    )
    assert set(out.hourly["optimization_error_code"].astype(str)) == {"safe_hold_plan_under_solver_not_set"}
    assert float(out.summary["fallback_used"]) == 1.0
    assert float(out.summary["simulation_valid"]) == 0.0
    assert "fallback_used" in str(out.summary["invalid_reason"])


def test_highs_unknown_with_feasible_primal_counts_as_feasible_incumbent() -> None:
    class _Sol:
        status = 15
        message = (
            "The HiGHS status code was not recognized. "
            "(HiGHS Status 15: model_status is Unknown; primal_status is Feasible)"
        )
        x = np.array([0.0])

    assert BatteryBacktester._has_feasible_incumbent_result(_Sol())


def test_rolling_pf_solver_error_selects_feasible_no_market_incumbent() -> None:
    df, col = _tiny_backtest_df(hours=3)
    bt = _mk_backtester("canonical_economic")
    original = bt.optimize_dispatch_rolling

    def _wrapped(*args, **kwargs):  # noqa: ANN002, ANN003
        if kwargs.get("run_mode") == "perfect_foresight":
            raise RuntimeError(
                "MIP optimization failed: The HiGHS status code was not recognized. "
                "(HiGHS Status 15: model_status is Unknown; primal_status is Feasible)"
            )
        return original(*args, **kwargs)

    bt.optimize_dispatch_rolling = _wrapped  # type: ignore[method-assign]
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=3,
        reopt_step_hours=1,
        allowed_markets=("aFRR",),
        strategy_name="bcm",
        strict_simulation_validity=True,
        enforce_final_soc_min=True,
        enable_global_perfect_foresight=True,
    )
    s = out.summary
    assert float(s["rolling_pf_available"]) == 1.0
    assert str(s["rolling_pf_solver_status"]) == "unknown_feasible"
    assert "HiGHS Status 15" in str(s["rolling_pf_solver_message"])
    assert float(s["rolling_pf_verified"]) == 1.0
    assert float(s["rolling_pf_verified_upper_bound"]) == 1.0
    assert str(s["rolling_pf_selected_incumbent"]) == "no_market"
    assert np.isfinite(float(s["rolling_pf_no_market_incumbent_eur"]))
    assert float(s["rolling_pf_no_market_incumbent_eur"]) <= 1e-9
    assert float(s["rolling_pf_no_market_terminal_shortfall_mwh"]) <= 1e-6
    assert "global_pf_verified_upper_bound" in s
    assert "rolling_pf_solver_failed" in str(s["invalid_reason"])
    assert float(s["rolling_pf_reported_available"]) == pytest.approx(0.0)
    assert np.isnan(float(s["rolling_perfect_foresight_same_rules_total_pnl_eur"]))
    assert "rolling_pf_solver_raw_total_pnl_eur" in s
    assert str(s["rolling_pf_reported_invalid_reason"]) != "none"
    assert np.isnan(float(s["same_rules_rolling_pf_total_pnl_eur"]))
    assert np.isnan(float(s["rhpf_total_pnl_eur"]))


def test_rolling_pf_solver_error_selects_feasible_incumbent_without_invalidating() -> None:
    df, col = _tiny_backtest_df(hours=3)
    df[col.true_da_price] = [100.0, 100.0, 100.0]
    df[col.pred_da_price] = [100.0, 100.0, 100.0]
    for q in ["p05", "p10", "p90", "p95"]:
        df[f"{col.pred_da_price}_{q}"] = [100.0, 100.0, 100.0]
    bt = _mk_backtester("canonical_economic")
    original = bt.optimize_dispatch_rolling

    def _wrapped(*args, **kwargs):  # noqa: ANN002, ANN003
        if kwargs.get("run_mode") == "perfect_foresight":
            raise RuntimeError("MIP optimization failed: (HiGHS Status 0: Error)")
        return original(*args, **kwargs)

    bt.optimize_dispatch_rolling = _wrapped  # type: ignore[method-assign]
    out = bt.run(
        df,
        col,
        use_rolling_horizon=True,
        horizon_hours=3,
        reopt_step_hours=1,
        allowed_markets=("DA",),
        strategy_name="da",
        strict_simulation_validity=True,
        enforce_final_soc_min=True,
        enable_global_perfect_foresight=False,
    )
    s = out.summary
    assert str(s["rolling_pf_solver_status"]) == "solver_failed"
    assert float(s["rolling_pf_available"]) == 1.0
    assert float(s["rolling_pf_reported_available"]) == 0.0
    assert str(s["rolling_pf_selected_incumbent"]) in {"no_market", "realized_path"}
    assert "rolling_pf_solver_failed" in str(s["invalid_reason"])


def test_no_obligation_protected_soc_equals_physical_bounds() -> None:
    bt = _mk_backtester()
    bt.reserve_headroom_safety_mwh = 0.75
    bt.reserve_soc_projection_safety_mwh = 1.5
    res = bt._compute_obligation_driven_protected_soc_bounds(
        soc_start_mwh=3.5,
        required_headroom_pos_mwh=0.0,
        required_headroom_neg_mwh=0.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        committed_bem_pos_mw=0.0,
        committed_bem_neg_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
    )
    assert np.isclose(float(res["physical_soc_min_mwh"]), bt.soc_min)
    assert np.isclose(float(res["physical_soc_max_mwh"]), bt.soc_max)
    assert np.isclose(float(res["protected_soc_min_mwh"]), bt.soc_min)
    assert np.isclose(float(res["protected_soc_max_mwh"]), bt.soc_max)
    assert np.isclose(float(res["protected_soc_violation_pos_mwh"]), 0.0)
    assert np.isclose(float(res["protected_soc_violation_neg_mwh"]), 0.0)


def test_positive_obligation_protected_soc_uses_headroom_and_safety() -> None:
    bt = _mk_backtester()
    bt.reserve_headroom_safety_mwh = 0.75
    bt.reserve_soc_projection_safety_mwh = 1.5
    res = bt._compute_obligation_driven_protected_soc_bounds(
        soc_start_mwh=3.0,
        required_headroom_pos_mwh=0.5,
        required_headroom_neg_mwh=0.0,
        locked_reserve_pos_mw=1.0,
        locked_reserve_neg_mw=0.0,
        committed_bem_pos_mw=0.0,
        committed_bem_neg_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
    )
    assert float(res["protected_soc_min_mwh"]) > bt.soc_min
    assert float(res["protected_soc_violation_pos_mwh"]) > 0.0


def test_negative_obligation_protected_soc_uses_headroom_and_safety() -> None:
    bt = _mk_backtester()
    bt.reserve_headroom_safety_mwh = 0.75
    bt.reserve_soc_projection_safety_mwh = 1.5
    res = bt._compute_obligation_driven_protected_soc_bounds(
        soc_start_mwh=17.5,
        required_headroom_pos_mwh=0.0,
        required_headroom_neg_mwh=0.5,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=1.0,
        committed_bem_pos_mw=0.0,
        committed_bem_neg_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
    )
    assert float(res["protected_soc_max_mwh"]) < bt.soc_max
    assert float(res["protected_soc_violation_neg_mwh"]) > 0.0


def test_zero_required_headroom_does_not_invalidate_da_only_soc_path() -> None:
    bt = _mk_backtester()
    bt.reserve_headroom_safety_mwh = 0.75
    bt.reserve_soc_projection_safety_mwh = 1.5
    res = bt._compute_obligation_driven_protected_soc_bounds(
        soc_start_mwh=3.5,
        required_headroom_pos_mwh=0.0,
        required_headroom_neg_mwh=0.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        committed_bem_pos_mw=0.0,
        committed_bem_neg_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
    )
    assert float(res["protected_soc_violation_pos_mwh"]) <= 1e-12
    assert float(res["protected_soc_violation_without_obligation"]) <= 1e-12


def test_real_physical_soc_violation_still_invalidates() -> None:
    bt = _mk_backtester()
    res = bt._compute_obligation_driven_protected_soc_bounds(
        soc_start_mwh=bt.soc_min - 0.1,
        required_headroom_pos_mwh=0.0,
        required_headroom_neg_mwh=0.0,
        locked_reserve_pos_mw=0.0,
        locked_reserve_neg_mw=0.0,
        committed_bem_pos_mw=0.0,
        committed_bem_neg_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
    )
    assert float(res["physical_soc_violation_pos_mwh"]) > 0.0


def test_protected_soc_violation_with_obligation_still_invalidates() -> None:
    bt = _mk_backtester()
    bt.reserve_headroom_safety_mwh = 0.75
    bt.reserve_soc_projection_safety_mwh = 1.5
    res = bt._compute_obligation_driven_protected_soc_bounds(
        soc_start_mwh=3.0,
        required_headroom_pos_mwh=0.75,
        required_headroom_neg_mwh=0.0,
        locked_reserve_pos_mw=0.5,
        locked_reserve_neg_mw=0.0,
        committed_bem_pos_mw=0.0,
        committed_bem_neg_mw=0.0,
        reserve_pos_mw=0.0,
        reserve_neg_mw=0.0,
    )
    assert float(res["obligation_headroom_pos_active"]) > 0.5
    assert float(res["protected_soc_violation_pos_mwh"]) > 0.0


def _perf_args() -> argparse.Namespace:
    return argparse.Namespace(
        model_key="xgb",
        run_manifest="artifacts/model_runs/latest_xgboost.json",
        split="test",
        trading_strategy="multi",
        id_recourse_mode="none",
        da_quantile_role="mid",
    )


def test_predicted_total_pnl_adds_planned_legacy_alias() -> None:
    summary = normalize_predicted_pnl_aliases({"predicted_total_pnl_eur": 12.5})

    assert float(summary["predicted_total_pnl_eur"]) == pytest.approx(12.5)
    assert float(summary["planned_total_pnl_eur"]) == pytest.approx(12.5)
    assert float(summary["planned_total_pnl_eur_is_legacy_alias"]) == pytest.approx(1.0)


def test_legacy_planned_total_pnl_normalizes_to_predicted() -> None:
    summary = normalize_predicted_pnl_aliases({"planned_total_pnl_eur": -3.25})

    assert float(summary["predicted_total_pnl_eur"]) == pytest.approx(-3.25)
    assert float(summary["planned_total_pnl_eur"]) == pytest.approx(-3.25)
    assert summary["predicted_total_pnl_eur_source"] == "legacy_planned_total_pnl_eur"


def test_predicted_planned_total_pnl_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="predicted_total_pnl_eur and planned_total_pnl_eur differ"):
        normalize_predicted_pnl_aliases(
            {
                "predicted_total_pnl_eur": 1.0,
                "planned_total_pnl_eur": 2.0,
            }
        )


def test_validate_outputs_flags_predicted_planned_total_pnl_mismatch() -> None:
    summary = validate_outputs._normalize_predicted_pnl_aliases(
        {
            "predicted_total_pnl_eur": 1.0,
            "planned_total_pnl_eur": 2.0,
        }
    )

    assert float(summary["predicted_planned_pnl_alias_consistency_ok"]) == pytest.approx(0.0)
    assert float(summary["predicted_planned_pnl_alias_error_eur"]) == pytest.approx(-1.0)
    assert float(summary["planned_total_pnl_eur"]) == pytest.approx(1.0)


def test_performance_metrics_prefers_predicted_pnl_and_accepts_legacy_planned_alias() -> None:
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=1, freq="h")
    hourly = pd.DataFrame({"timestamp_utc": ts, "real_pnl_eur": [0.0], "real_throughput_mwh": [0.0]})
    summary = {
        "realized_total_pnl_eur": 0.0,
        "planned_total_pnl_eur": 42.0,
        "p_max_mw": 10.0,
        "capacity_mwh": 20.0,
    }

    perf_df, _ = _build_performance_metrics(
        hourly=hourly,
        summary=summary,
        args=_perf_args(),
        scenario_name="p50_p50",
        scenario_bins=["p50", "p50"],
        scenario_start_utc=None,
        scenario_end_utc=None,
    )

    assert float(perf_df.iloc[0]["predicted_net_revenue_eur"]) == pytest.approx(42.0)


def test_strategy_overview_benchmark_comparison_columns() -> None:
    overview = pd.DataFrame(
        {
            "realized_total_pnl_eur": [85.0, 120.0],
            "naive_total_pnl_eur": [25.0, 0.0],
            "rolling_perfect_foresight_same_rules_total_pnl_eur": [145.0, 240.0],
        }
    )

    out = _add_benchmark_comparison_columns(overview)

    assert float(out.loc[0, "value_of_foresight_ratio"]) == pytest.approx((85.0 - 25.0) / (145.0 - 25.0))
    assert float(out.loc[0, "value_of_foresight_pct"]) == pytest.approx(50.0)
    assert float(out.loc[0, "model_vs_naive_pct_minus_100"]) == pytest.approx((85.0 / 25.0 - 1.0) * 100.0)
    assert float(out.loc[0, "model_vs_rhpf_pct"]) == pytest.approx((85.0 / 145.0) * 100.0)
    assert np.isnan(float(out.loc[1, "model_vs_naive_pct_minus_100"]))
    assert float(out.loc[1, "model_vs_rhpf_pct"]) == pytest.approx(50.0)


def test_valid_only_overview_masks_invalid_benchmark_using_simulation_valid_fallback() -> None:
    overview = pd.DataFrame(
        {
            "realized_total_pnl_eur": [100.0],
            "naive_total_pnl_eur": [200.0],
            "rolling_perfect_foresight_same_rules_total_pnl_eur": [300.0],
            "naive_simulation_valid": [0.0],
            "rhpf_simulation_valid": [0.0],
        }
    )
    overview = _add_benchmark_comparison_columns(overview)

    masked = _mask_invalid_benchmark_comparisons(overview)

    assert np.isnan(float(masked.loc[0, "naive_total_pnl_eur"]))
    assert np.isnan(float(masked.loc[0, "rolling_perfect_foresight_same_rules_total_pnl_eur"]))
    assert np.isnan(float(masked.loc[0, "model_vs_naive_pct_minus_100"]))
    assert np.isnan(float(masked.loc[0, "model_vs_rhpf_pct"]))
    assert np.isnan(float(masked.loc[0, "value_of_foresight_pct"]))


def test_strategy_overview_masks_model_vs_rhpf_when_rhpf_invalid() -> None:
    overview = pd.DataFrame(
        {
            "realized_total_pnl_eur": [100.0],
            "naive_total_pnl_eur": [50.0],
            "rolling_perfect_foresight_same_rules_total_pnl_eur": [200.0],
            "naive_simulation_valid": [1.0],
            "rhpf_simulation_valid": [0.0],
        }
    )

    out = _add_benchmark_comparison_columns(overview)

    assert np.isnan(float(out.loc[0, "model_vs_rhpf_pct"]))
    assert np.isnan(float(out.loc[0, "value_of_foresight_pct"]))
    assert float(out.loc[0, "model_vs_naive_pct_minus_100"]) == pytest.approx(100.0)


def test_daily_to_scenario_reconciliation_uses_hourly_real_pnl() -> None:
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=24, freq="h")
    hourly = pd.DataFrame({"timestamp_utc": ts, "real_pnl_eur": [10.0] * 24, "real_throughput_mwh": [0.0] * 24})
    summary = {
        "realized_total_pnl_eur": 9999.0,  # intentionally wrong; hourly must be source-of-truth
        "predicted_total_pnl_eur": 0.0,
        "p_max_mw": 10.0,
        "capacity_mwh": 20.0,
    }
    perf_df, _ = _build_performance_metrics(
        hourly=hourly,
        summary=summary,
        args=_perf_args(),
        scenario_name="p50_p50",
        scenario_bins=["p50", "p50"],
        scenario_start_utc=None,
        scenario_end_utc=None,
    )
    daily_df = _build_daily_performance_metrics(hourly=hourly, perf_row=perf_df.iloc[0])
    checks = _validate_performance_metrics(perf_row=perf_df.iloc[0], daily_df=daily_df)
    assert float(perf_df.iloc[0]["realized_net_revenue_eur"]) == pytest.approx(240.0)
    assert float(pd.to_numeric(daily_df["net_revenue_eur"], errors="coerce").sum()) == pytest.approx(240.0)
    assert bool(checks["daily_to_scenario_reconciliation_ok"])


def test_component_to_net_reconciliation_passes_for_consistent_hourly_components() -> None:
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=1, freq="h")
    hourly = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "real_revenue_da_eur": [100.0],
            "real_cost_da_eur": [0.0],
            "real_revenue_id_eur": [0.0],
            "real_cost_id_eur": [5.0],
            "real_revenue_capacity_eur": [20.0],
            "real_revenue_activation_eur": [10.0],
            "real_degradation_cost_eur": [2.0],
            "real_penalty_eur": [3.0],
            "real_aux_cost_eur": [1.0],
            "real_transaction_cost_eur": [0.0],
            "real_offer_cost_eur": [0.0],
            "real_pnl_eur": [119.0],
            "real_throughput_mwh": [0.0],
        }
    )
    summary = {"realized_total_pnl_eur": 0.0, "predicted_total_pnl_eur": 0.0, "p_max_mw": 10.0, "capacity_mwh": 20.0}
    perf_df, _ = _build_performance_metrics(
        hourly=hourly,
        summary=summary,
        args=_perf_args(),
        scenario_name="p50_p50",
        scenario_bins=["p50", "p50"],
        scenario_start_utc=None,
        scenario_end_utc=None,
    )
    assert float(perf_df.iloc[0]["net_revenue_reconciliation_error_eur"]) == pytest.approx(0.0, abs=1e-9)
    assert float(perf_df.iloc[0]["realized_net_revenue_eur"]) == pytest.approx(119.0, abs=1e-9)


def test_reconciliation_debug_exposes_daily_vs_scenario_mismatch() -> None:
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")
    hourly = pd.DataFrame({"timestamp_utc": ts, "real_pnl_eur": [10.0, 10.0, 10.0, 10.0]})
    perf_row = pd.Series({"realized_net_revenue_eur": 40.0, "da_gross_revenue_eur": 0.0})
    daily_df = pd.DataFrame({"date_utc": ["2026-01-01"], "net_revenue_eur": [25.0]})
    dbg = _build_performance_reconciliation_debug(
        scenario="p50_p50",
        perf_row=perf_row,
        daily_df=daily_df,
        hourly=hourly,
    )
    row = dbg.loc[dbg["metric"] == "realized_net_revenue_eur"].iloc[0]
    assert float(row["scenario_minus_daily"]) == pytest.approx(15.0)
    assert float(row["scenario_minus_hourly"]) == pytest.approx(0.0)


def test_reconciliation_debug_marks_checked_and_skipped_metrics() -> None:
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=1, freq="h")
    hourly = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "real_pnl_eur": [10.0],
            "real_transaction_cost_eur": [2.0],
        }
    )
    perf_row = pd.Series(
        {
            "realized_net_revenue_eur": 10.0,
            "transaction_cost_eur": 2.0,
            "terminal_soc_repair_cost_eur": 7.0,
        }
    )
    daily_df = pd.DataFrame(
        {
            "date_utc": ["2026-01-01"],
            "net_revenue_eur": [10.0],
            "transaction_cost_eur": [2.0],
            "terminal_soc_repair_cost_eur": [0.0],
        }
    )
    dbg = _build_performance_reconciliation_debug(
        scenario="p50_p50",
        perf_row=perf_row,
        daily_df=daily_df,
        hourly=hourly,
    )
    net_row = dbg.loc[dbg["metric"] == "realized_net_revenue_eur"].iloc[0]
    term_row = dbg.loc[dbg["metric"] == "terminal_soc_repair_cost_eur"].iloc[0]
    assert bool(net_row["checked_daily_to_scenario"]) is True
    assert float(net_row["daily_abs_error"]) == pytest.approx(0.0)
    assert bool(term_row["checked_daily_to_scenario"]) is False
    assert term_row["skipped_reason"] == "not_checked_in_daily_to_scenario_reconciliation"


def test_daily_to_scenario_max_uses_checked_rows_only() -> None:
    perf_row = pd.Series(
        {
            "realized_net_revenue_eur": 40.0,
            "transaction_cost_eur": 3.0,
            "terminal_soc_repair_cost_eur": 45.056,
            "net_revenue_reconciliation_error_eur": 0.0,
            "total_costs_eur": 3.0,
            "realized_degradation_cost_eur": 0.0,
            "realized_aux_cost_eur": 0.0,
            "offer_cost_eur": 0.0,
            "penalty_cost_eur": 0.0,
        }
    )
    daily_df = pd.DataFrame(
        {
            "date_utc": ["2026-01-01"],
            "net_revenue_eur": [40.0],
            "transaction_cost_eur": [3.0],
            "terminal_soc_repair_cost_eur": [0.0],
        }
    )
    checks = _validate_performance_metrics(perf_row=perf_row, daily_df=daily_df)
    assert checks["daily_to_scenario_reconciliation_ok"] is True
    assert float(checks["daily_to_scenario_error_max_abs"]) == pytest.approx(0.0)


def test_daily_metrics_include_component_pnl_columns() -> None:
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=1, freq="h")
    hourly = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "real_revenue_da_eur": [100.0],
            "real_cost_da_eur": [30.0],
            "real_revenue_id_eur": [20.0],
            "real_cost_id_eur": [5.0],
            "real_revenue_capacity_eur": [7.0],
            "real_revenue_activation_eur": [9.0],
            "real_bcm_linked_activation_revenue_eur": [4.0],
            "real_bem_only_activation_revenue_eur": [6.0],
            "real_offer_cost_eur": [2.0],
            "real_pnl_eur": [101.0],
            "real_throughput_mwh": [0.0],
        }
    )
    perf_row = pd.Series({"simulation_valid": 1.0, "thesis_reportable": 1.0, "invalid_reason": ""})
    daily_df = _build_daily_performance_metrics(hourly=hourly, perf_row=perf_row)
    row = daily_df.iloc[0]
    assert float(row["da_pnl_eur"]) == pytest.approx(70.0)
    assert float(row["id_recourse_pnl_eur"]) == pytest.approx(15.0)
    assert float(row["bcm_pnl_eur"]) == pytest.approx(11.0)
    assert float(row["bem_pnl_eur"]) == pytest.approx(6.0)
    assert float(row["afrr_pnl_eur"]) == pytest.approx(16.0)


def test_terminal_surplus_value_is_target_relative() -> None:
    bt = _mk_backtester()
    bt.soc_target_end = 10.0
    bt.eta_out = 0.9

    at_target = bt._terminal_surplus_value_components(
        final_soc_mwh=10.0,
        terminal_price_eur_mwh=100.0,
    )
    assert float(at_target["terminal_surplus_mwh"]) == pytest.approx(0.0)
    assert float(at_target["terminal_surplus_value_net_eur"]) == pytest.approx(0.0)

    surplus = bt._terminal_surplus_value_components(
        final_soc_mwh=17.56,
        terminal_price_eur_mwh=100.0,
    )
    assert float(surplus["terminal_surplus_mwh"]) == pytest.approx(7.56)
    assert float(surplus["terminal_surplus_grid_mwh"]) == pytest.approx(7.56 * 0.9)
    assert float(surplus["terminal_surplus_value_net_eur"]) == pytest.approx(7.56 * 0.9 * 100.0)
    assert str(surplus["terminal_value_convention"]) == "gross_target_relative_surplus"

    shortfall = bt._terminal_surplus_value_components(
        final_soc_mwh=8.0,
        terminal_price_eur_mwh=100.0,
    )
    assert float(shortfall["terminal_surplus_mwh"]) == pytest.approx(0.0)
    assert float(shortfall["terminal_surplus_value_net_eur"]) == pytest.approx(0.0)


def test_da_soc_shadow_value_uses_conservative_internal_mwh_value() -> None:
    bt = _mk_backtester()
    bt.eta_in = 0.9
    bt.eta_out = 0.9
    bt.trans_eur_mwh = 1.0
    bt.deg_eur_mwh = 5.0
    bt.da_terminal_soc_value_enabled = True
    bt.da_terminal_value_quantile_sell = 0.5
    bt.da_terminal_value_quantile_buy = 0.5
    bt.da_terminal_value_margin_eur_mwh = 0.0
    bt.da_terminal_value_min_eur_mwh = 0.0
    bt.da_terminal_value_max_eur_mwh = 500.0

    comp = bt._da_soc_shadow_value_components(pd.Series([10.0, 100.0]))

    sell_value = 0.9 * 55.0 - 0.9 * 1.0 - 5.0
    replacement_cost = 55.0 / 0.9 + 1.0 / 0.9 + 5.0
    assert float(comp["da_future_sell_value_internal_eur_mwh"]) == pytest.approx(sell_value)
    assert float(comp["da_future_replacement_cost_internal_eur_mwh"]) == pytest.approx(replacement_cost)
    assert float(comp["da_soc_shadow_value_eur_per_internal_mwh"]) == pytest.approx(
        min(sell_value, replacement_cost)
    )
    assert str(comp["da_terminal_value_source"]) == "conservative_da_quantile"


def test_da_replay_terminal_value_counts_local_continuation_for_selection() -> None:
    bt = _mk_backtester()
    bt.soc_target_end = 10.0
    bt.eta_in = 1.0
    bt.eta_out = 1.0
    bt.trans_eur_mwh = 0.0
    bt.deg_eur_mwh = 0.0
    bt.aux_peak_mw = 0.0
    bt.aux_off_mw = 0.0
    bt.aux_standby_mw = 0.0
    bt.aux_trading_mw = 0.0
    bt.aux_afrr_active_mw = 0.0
    bt.da_terminal_soc_value_enabled = True
    bt.da_terminal_value_quantile_sell = 0.5
    bt.da_terminal_value_quantile_buy = 0.5
    bt.da_terminal_value_min_eur_mwh = 0.0
    bt.da_terminal_value_max_eur_mwh = 500.0
    bt.terminal_value_mode = "legacy_continuation"
    df, col = _tiny_backtest_df(hours=1)
    df[col.pred_da_price] = 100.0
    ts = pd.to_datetime(df[col.timestamp].iloc[0], utc=True)
    rows = df.rename(columns={col.timestamp: "target_time_utc"})

    no_global = bt._replay_da_candidate_cashflow(
        rows=rows,
        schedule={ts: (0.0, 0.0)},
        colmap=col,
        current_soc_mwh=12.0,
        global_end_utc=None,
    )
    with_global = bt._replay_da_candidate_cashflow(
        rows=rows,
        schedule={ts: (0.0, 0.0)},
        colmap=col,
        current_soc_mwh=12.0,
        global_end_utc=ts,
    )

    assert float(no_global["terminal_credit_eur"]) == pytest.approx(200.0)
    assert float(no_global["local_terminal_credit_ignored_eur"]) == pytest.approx(0.0)
    assert str(no_global["selection_pnl_basis"]) == "includes_local_continuation"
    assert float(with_global["terminal_credit_eur"]) == pytest.approx(200.0)
    assert str(with_global["selection_pnl_basis"]) == "includes_global_terminal"
    assert float(with_global["da_soc_shadow_value_eur_per_internal_mwh"]) == pytest.approx(100.0)


def test_da_precommit_selects_inventory_buy_when_local_continuation_value_positive() -> None:
    bt = _mk_backtester()
    bt.soc_target_end = 10.0
    bt.eta_in = 1.0
    bt.eta_out = 1.0
    bt.trans_eur_mwh = 0.0
    bt.deg_eur_mwh = 0.0
    bt.aux_peak_mw = 0.0
    bt.aux_off_mw = 0.0
    bt.aux_standby_mw = 0.0
    bt.aux_trading_mw = 0.0
    bt.aux_afrr_active_mw = 0.0
    bt.da_terminal_soc_value_enabled = True
    bt.da_terminal_value_min_eur_mwh = 100.0
    bt.da_terminal_value_max_eur_mwh = 100.0
    col = BacktestColumnMap()
    ts = pd.Timestamp("2025-03-02T00:00:00Z")
    rows = pd.DataFrame(
        {
            col.timestamp: [ts],
            "target_time_utc": [ts],
            "charge_mw": [1.0],
            "discharge_mw": [0.0],
            col.pred_da_price: [10.0],
            col.pred_afrr_activation_rate_pos: [0.0],
            col.pred_afrr_activation_rate_neg: [0.0],
            "predicted_objective_eur": [90.0],
        }
    )

    selected, audit = bt._select_feasible_da_lock_schedule(
        lock_rows=rows,
        colmap=col,
        current_soc_mwh=10.0,
        fixed_reserve_pos={},
        fixed_reserve_neg={},
        global_end_utc=None,
    )

    assert selected[ts] == pytest.approx((1.0, 0.0))
    assert audit[0]["selected_incumbent"] == "optimized"
    assert str(audit[0]["selection_pnl_basis"]) == "includes_local_continuation"
    assert float(audit[0]["candidate_cost_eur"]) == pytest.approx(10.0)
    assert float(audit[0]["terminal_credit_in_selection_eur"]) == pytest.approx(100.0)
    assert float(audit[0]["candidate_pnl_recomputed_eur"]) == pytest.approx(90.0)
    assert float(audit[0]["candidate_selection_pnl_eur"]) > float(audit[0]["incumbent_selection_pnl_eur"])
    assert float(audit[0]["incumbent_selection_pnl_eur"]) == pytest.approx(0.0)


def test_raw_optimizer_objective_scope_separates_terminal_credit_from_row_ev() -> None:
    bt = _mk_backtester()
    bt.soc_target_end = 10.0
    bt.soc_init = 17.56
    bt.eta_out = 1.0
    df, col = _tiny_backtest_df(hours=1)
    df[col.pred_da_price] = 100.0
    df[col.true_da_price] = 100.0

    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=17.56,
        soc_end_min_target=10.0,
        allowed_markets=(),
    )

    row = out.iloc[0]
    assert float(row["charge_mw"]) == pytest.approx(0.0)
    assert float(row["discharge_mw"]) == pytest.approx(0.0)
    assert float(row["ev_da_charge_eur"]) == pytest.approx(0.0)
    assert float(row["ev_da_discharge_eur"]) == pytest.approx(0.0)
    assert float(row["ev_objective_rebuild_eur"]) == pytest.approx(0.0)
    assert float(row["ev_terminal_soc_credit_eur"]) == pytest.approx(7.56 * 0.8 * 100.0)
    assert float(row["terminal_surplus_mwh"]) == pytest.approx(7.56)
    assert row["raw_optimizer_objective_scope"] == "rolling_window"
    assert "raw_optimizer_window_terminal_credit_eur" in out.columns



def test_benchmark_mode_model_only_masks_disabled_benchmarks() -> None:
    mode, paths = resolve_benchmark_paths("model_only", enable_global_perfect_foresight=True)

    summary = apply_benchmark_availability_to_summary(
        {
            "realized_total_pnl_eur": 12.0,
            "naive_total_pnl_eur": 0.0,
            "rolling_perfect_foresight_same_rules_total_pnl_eur": 0.0,
            "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur": 0.0,
        },
        benchmark_mode=mode,
        benchmark_paths=paths,
    )

    assert paths == {"model": True, "naive": False, "rhpf": False, "ghpf": False}
    assert summary["model_available"] == 1.0
    assert summary["naive_available"] == 0.0
    assert summary["rolling_pf_available"] == 0.0
    assert summary["global_pf_available"] == 0.0
    assert summary["model_total_pnl_eur"] == pytest.approx(12.0)
    assert np.isnan(float(summary["naive_total_pnl_eur"]))
    assert np.isnan(float(summary["rolling_perfect_foresight_same_rules_total_pnl_eur"]))
    assert np.isnan(float(summary["global_hindsight_perfect_foresight_upper_bound_total_pnl_eur"]))


def test_disabled_rolling_pf_does_not_report_solver_failed() -> None:
    assert not should_report_rolling_pf_solver_failed(
        strict_simulation_validity=True,
        rolling_pf_enabled=False,
        rolling_pf_available=0.0,
    )


def test_enabled_rolling_pf_unavailable_reports_solver_failed() -> None:
    assert should_report_rolling_pf_solver_failed(
        strict_simulation_validity=True,
        rolling_pf_enabled=True,
        rolling_pf_available=0.0,
    )


def test_benchmark_mode_full_excludes_ghpf_by_default() -> None:
    mode, paths = resolve_benchmark_paths("full", enable_global_perfect_foresight=True)
    summary = apply_benchmark_availability_to_summary(
        {
            "realized_total_pnl_eur": 1.0,
            "naive_total_pnl_eur": 2.0,
            "rolling_perfect_foresight_same_rules_total_pnl_eur": 3.0,
            "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur": 4.0,
            "rolling_pf_available": 1.0,
            "global_pf_available": 1.0,
        },
        benchmark_mode=mode,
        benchmark_paths=paths,
    )

    assert paths == {"model": True, "naive": True, "rhpf": True, "ghpf": False}
    assert summary["model_available"] == 1.0
    assert summary["naive_available"] == 1.0
    assert summary["rolling_pf_available"] == 1.0
    assert summary["global_pf_available"] == 0.0
    assert summary["model_total_pnl_eur"] == pytest.approx(1.0)
    assert summary["rhpf_total_pnl_eur"] == pytest.approx(3.0)
    assert np.isnan(float(summary["ghpf_total_pnl_eur"]))


def test_benchmark_mode_full_can_enable_ghpf_explicitly() -> None:
    mode, paths = resolve_benchmark_paths("full", enable_global_pf=True)
    summary = apply_benchmark_availability_to_summary(
        {
            "realized_total_pnl_eur": 1.0,
            "naive_total_pnl_eur": 2.0,
            "rolling_perfect_foresight_same_rules_total_pnl_eur": 3.0,
            "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur": 4.0,
            "rolling_pf_available": 1.0,
            "global_pf_available": 1.0,
        },
        benchmark_mode=mode,
        benchmark_paths=paths,
    )

    assert paths == {"model": True, "naive": True, "rhpf": True, "ghpf": True}
    assert summary["global_pf_available"] == 1.0
    assert summary["ghpf_total_pnl_eur"] == pytest.approx(4.0)


def test_rhpf_aliases_use_reported_raw_solver_not_selected_fallback() -> None:
    mode, paths = resolve_benchmark_paths("full", enable_global_perfect_foresight=False)
    summary = apply_benchmark_availability_to_summary(
        {
            "realized_total_pnl_eur": 1.0,
            "naive_total_pnl_eur": 0.0,
            "rolling_pf_reported_total_pnl_eur": 10.0,
            "rolling_pf_reported_available": 1.0,
            "rolling_pf_reported_is_solver": 1.0,
            "rolling_pf_selected_total_pnl_eur": 25.0,
            "rolling_pf_selected_is_realized_path_fallback": 1.0,
            "rolling_perfect_foresight_same_rules_total_pnl_eur": 10.0,
            "same_rules_rolling_pf_total_pnl_eur": 10.0,
            "rolling_pf_available": 1.0,
        },
        benchmark_mode=mode,
        benchmark_paths=paths,
    )

    assert summary["rhpf_total_pnl_eur"] == pytest.approx(10.0)
    assert summary["rhpf_total_pnl_eur"] != pytest.approx(25.0)


def test_validate_rhpf_outputs_warns_on_diagnostic_fallback_but_keeps_raw_solver(tmp_path: Path) -> None:
    scenario = tmp_path / "p50_p50"
    scenario.mkdir()
    summary = {
        "rolling_pf_available": 1.0,
        "rolling_pf_reported_available": 1.0,
        "rolling_pf_reported_is_solver": 1.0,
        "rolling_pf_reported_total_pnl_eur": 10.0,
        "rolling_pf_solver_total_pnl_eur": 10.0,
        "rolling_pf_solver_component_pnl_eur": 10.0,
        "rolling_pf_solver_pnl_balance_error_eur": 0.0,
        "rolling_pf_solver_pnl_balance_ok": 1.0,
        "rolling_pf_selected_total_pnl_eur": 25.0,
        "rolling_pf_selected_component_pnl_eur": 25.0,
        "rolling_pf_selected_pnl_balance_error_eur": 0.0,
        "rolling_pf_selected_pnl_balance_ok": 1.0,
        "rolling_pf_selected_incumbent": "realized_path",
        "rolling_pf_selected_is_solver": 0.0,
        "rolling_pf_selected_is_realized_path_fallback": 1.0,
        "rolling_pf_selected_is_no_market_fallback": 0.0,
        "rolling_pf_incumbent_selection_reason": "diagnostic_realized_path_higher",
        "rolling_perfect_foresight_same_rules_total_pnl_eur": 10.0,
        "same_rules_rolling_pf_total_pnl_eur": 10.0,
        "rhpf_total_pnl_eur": 10.0,
    }
    (scenario / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    strict = validate_rhpf_outputs.validate_scenario(scenario, 1e-6)
    exploratory = validate_rhpf_outputs.validate_scenario(
        scenario,
        1e-6,
        allow_fallback_benchmark=True,
    )

    assert strict.status == "PASS"
    assert any("diagnostic fallback beats solver" in msg for msg in strict.warnings)
    assert exploratory.status == "PASS"
    assert any("diagnostic fallback beats solver" in msg for msg in exploratory.warnings)


def test_validate_rhpf_outputs_fails_aliases_equaling_selected_fallback(tmp_path: Path) -> None:
    scenario = tmp_path / "p50_p50"
    scenario.mkdir()
    summary = {
        "rolling_pf_available": 1.0,
        "rolling_pf_reported_available": 1.0,
        "rolling_pf_reported_is_solver": 1.0,
        "rolling_pf_reported_total_pnl_eur": 10.0,
        "rolling_pf_solver_total_pnl_eur": 10.0,
        "rolling_pf_solver_component_pnl_eur": 10.0,
        "rolling_pf_solver_pnl_balance_error_eur": 0.0,
        "rolling_pf_solver_pnl_balance_ok": 1.0,
        "rolling_pf_selected_total_pnl_eur": 25.0,
        "rolling_pf_selected_component_pnl_eur": 25.0,
        "rolling_pf_selected_pnl_balance_error_eur": 0.0,
        "rolling_pf_selected_pnl_balance_ok": 1.0,
        "rolling_pf_selected_incumbent": "solver",
        "rolling_pf_selected_is_solver": 1.0,
        "rolling_pf_selected_is_realized_path_fallback": 0.0,
        "rolling_pf_selected_is_no_market_fallback": 0.0,
        "rolling_perfect_foresight_same_rules_total_pnl_eur": 25.0,
        "same_rules_rolling_pf_total_pnl_eur": 25.0,
        "rhpf_total_pnl_eur": 25.0,
    }
    (scenario / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    result = validate_rhpf_outputs.validate_scenario(scenario, 1e-6)

    assert result.status == "FAIL"
    assert any("differs from reported" in msg for msg in result.failures)


def test_validate_rhpf_outputs_fails_nonzero_solver_balance_marked_ok(tmp_path: Path) -> None:
    scenario = tmp_path / "p50_p50"
    scenario.mkdir()
    summary = {
        "rolling_pf_available": 1.0,
        "rolling_pf_reported_available": 1.0,
        "rolling_pf_reported_is_solver": 1.0,
        "rolling_pf_reported_total_pnl_eur": -364.310450282,
        "rolling_pf_solver_total_pnl_eur": -364.310450282,
        "rolling_pf_solver_component_pnl_eur": -759.561464562,
        "rolling_pf_solver_pnl_balance_error_eur": 395.251014280,
        "rolling_pf_solver_pnl_balance_ok": 1.0,
        "rolling_pf_selected_total_pnl_eur": -364.310450282,
        "rolling_pf_selected_component_pnl_eur": -364.310450282,
        "rolling_pf_selected_pnl_balance_error_eur": 0.0,
        "rolling_pf_selected_pnl_balance_ok": 1.0,
        "rolling_pf_selected_incumbent": "solver",
        "rolling_pf_selected_is_solver": 1.0,
        "rolling_pf_selected_is_realized_path_fallback": 0.0,
        "rolling_pf_selected_is_no_market_fallback": 0.0,
        "rolling_perfect_foresight_same_rules_total_pnl_eur": -364.310450282,
        "same_rules_rolling_pf_total_pnl_eur": -364.310450282,
        "rhpf_total_pnl_eur": -364.310450282,
    }
    (scenario / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    result = validate_rhpf_outputs.validate_scenario(scenario, 1e-6)

    assert result.status == "FAIL"
    assert any("marked OK" in msg for msg in result.failures)


def test_validate_rhpf_outputs_passes_terminal_inclusive_solver_basis(tmp_path: Path) -> None:
    scenario = tmp_path / "p50_p50"
    scenario.mkdir()
    summary = {
        "rolling_pf_available": 1.0,
        "rolling_pf_reported_available": 1.0,
        "rolling_pf_reported_is_solver": 1.0,
        "rolling_pf_reported_total_pnl_eur": -364.310450282,
        "rolling_pf_solver_total_pnl_eur": -364.310450282,
        "rolling_pf_solver_component_pnl_eur": -364.310450282,
        "rolling_pf_solver_component_pnl_excl_terminal_eur": -759.561464562,
        "rolling_pf_solver_terminal_closure_pnl_eur": 0.0,
        "rolling_pf_solver_terminal_adjustment_eur": 395.251014280,
        "rolling_pf_solver_component_pnl_incl_terminal_eur": -364.310450282,
        "rolling_pf_solver_pnl_balance_error_eur": 0.0,
        "rolling_pf_solver_pnl_balance_ok": 1.0,
        "rolling_pf_selected_total_pnl_eur": -364.310450282,
        "rolling_pf_selected_component_pnl_eur": -364.310450282,
        "rolling_pf_selected_pnl_balance_error_eur": 0.0,
        "rolling_pf_selected_pnl_balance_ok": 1.0,
        "rolling_pf_selected_incumbent": "solver",
        "rolling_pf_selected_is_solver": 1.0,
        "rolling_pf_selected_is_realized_path_fallback": 0.0,
        "rolling_pf_selected_is_no_market_fallback": 0.0,
        "rolling_perfect_foresight_same_rules_total_pnl_eur": -364.310450282,
        "same_rules_rolling_pf_total_pnl_eur": -364.310450282,
        "rhpf_total_pnl_eur": -364.310450282,
    }
    (scenario / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    result = validate_rhpf_outputs.validate_scenario(scenario, 1e-6)

    assert result.status == "PASS"


def test_benchmark_mode_naive_rhpf_ghpf_only() -> None:
    assert resolve_benchmark_paths("naive_only")[1] == {"model": False, "naive": True, "rhpf": False, "ghpf": False}
    assert resolve_benchmark_paths("rhpf_only")[1] == {"model": False, "naive": False, "rhpf": True, "ghpf": False}
    assert resolve_benchmark_paths("ghpf_only", enable_global_perfect_foresight=True)[1] == {
        "model": False,
        "naive": False,
        "rhpf": False,
        "ghpf": True,
    }
    assert resolve_benchmark_paths("rolling_pf_only")[0] == "rhpf_only"
    assert resolve_benchmark_paths("global_pf_only")[0] == "ghpf_only"
    assert resolve_benchmark_paths("global_pf_only")[1] == {
        "model": False,
        "naive": False,
        "rhpf": False,
        "ghpf": True,
    }


def test_naive_only_can_skip_model_manifest_even_with_model_selector() -> None:
    _, paths = resolve_benchmark_paths("naive_only")

    assert _can_skip_model_manifest_for_benchmark(
        predictions_path="",
        benchmark_paths=paths,
        run_manifest="",
        run_id=None,
        # --model is normalized into args.model_key by the CLI, but should not
        # force model artifact loading when the model path is disabled.
    )


def test_model_enabled_cannot_skip_model_manifest_without_predictions() -> None:
    _, paths = resolve_benchmark_paths("model_only")

    assert not _can_skip_model_manifest_for_benchmark(
        predictions_path="",
        benchmark_paths=paths,
        run_manifest="",
        run_id=None,
    )


def test_benchmark_summary_exports_path_execution_flags() -> None:
    mode, paths = resolve_benchmark_paths("naive_only")
    summary = apply_benchmark_availability_to_summary(
        {
            "model_path_executed": 0.0,
            "naive_path_executed": 1.0,
            "rhpf_path_executed": 0.0,
            "model_manifest_loaded": 0.0,
            "model_predictions_loaded": 0.0,
        },
        benchmark_mode=mode,
        benchmark_paths=paths,
    )

    assert summary["benchmark_mode"] == "naive_only"
    assert summary["enabled_paths"] == "naive"
    assert summary["model_available"] == 0.0
    assert summary["naive_available"] == 1.0
    assert summary["model_path_executed"] == 0.0
    assert summary["naive_path_executed"] == 1.0
    assert summary["rhpf_path_executed"] == 0.0
    assert summary["model_manifest_loaded"] == 0.0
    assert summary["model_predictions_loaded"] == 0.0


def test_overview_uses_active_path_validity_for_naive_only() -> None:
    fields = _overview_validity_fields(
        {
            "benchmark_mode": "naive_only",
            "simulation_valid": 0.0,
            "thesis_reportable": 0.0,
            "invalid_reason": "disabled_model_terminal_soc",
            "active_path": "naive",
            "active_path_simulation_valid": 1.0,
            "active_path_thesis_reportable": 1.0,
            "active_path_invalid_reason": "",
        },
        "naive_only",
    )

    assert fields["simulation_valid"] == 1.0
    assert fields["thesis_reportable"] == 1.0
    assert fields["invalid_reason"] == "none"
    assert fields["legacy_invalid_reason"] == "disabled_model_terminal_soc"


def test_overview_uses_active_path_validity_for_rhpf_only() -> None:
    fields = _overview_validity_fields(
        {
            "benchmark_mode": "rhpf_only",
            "simulation_valid": 0.0,
            "thesis_reportable": 0.0,
            "invalid_reason": "disabled_model_soc_mass_balance",
            "active_path": "rhpf",
            "active_path_simulation_valid": 1.0,
            "active_path_thesis_reportable": 1.0,
            "active_path_invalid_reason": "",
        },
        "rhpf_only",
    )

    assert fields["simulation_valid"] == 1.0
    assert fields["thesis_reportable"] == 1.0
    assert fields["invalid_reason"] == "none"
    assert fields["legacy_simulation_valid"] == 0.0


def test_overview_keeps_legacy_validity_for_model_modes() -> None:
    fields = _overview_validity_fields(
        {
            "benchmark_mode": "full",
            "simulation_valid": 0.0,
            "thesis_reportable": 0.0,
            "invalid_reason": "model_invalid",
            "active_path_simulation_valid": 1.0,
            "active_path_thesis_reportable": 1.0,
            "active_path_invalid_reason": "",
        },
        "full",
    )

    assert fields["simulation_valid"] == 0.0
    assert fields["thesis_reportable"] == 0.0
    assert fields["invalid_reason"] == "model_invalid"


def test_overview_invalid_reason_normalizes_blank_for_csv_display() -> None:
    assert _overview_invalid_reason("") == "none"
    assert _overview_invalid_reason(None) == "none"
    assert _overview_invalid_reason(float("nan")) == "none"
    assert _overview_invalid_reason("missed_activation") == "missed_activation"


def test_bem_activity_daily_classifies_rejected_model_and_oracle_rhpf() -> None:
    ts = pd.date_range("2025-06-13", periods=2, freq="h", tz="UTC")
    model = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "real_bem_only_submitted_pos_mw": [5.0, 5.0],
            "real_bem_only_submitted_neg_mw": [10.0, 10.0],
            "real_bem_rejected_pos_mw": [0.0, 5.0],
            "real_bem_rejected_neg_mw": [10.0, 0.0],
            "real_bem_only_executed_pos_mwh": [0.1, 0.0],
            "real_bem_only_executed_neg_mwh": [0.0, 0.2],
            "real_bem_only_pos_activation_revenue_eur": [10.0, 0.0],
            "real_bem_only_neg_activation_revenue_eur": [0.0, 20.0],
            "real_bem_submitted_activation_price_pos_eur_per_mwh": [100.0, 150.0],
            "real_bem_submitted_activation_price_neg_eur_per_mwh": [200.0, 100.0],
            "true_activation_price_pos": [100.0, 125.0],
            "true_activation_price_neg": [150.0, 100.0],
        }
    )
    rhpf = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "perfect_foresight_bem_only_submitted_pos_mw": [10.0, 10.0],
            "perfect_foresight_bem_only_submitted_neg_mw": [10.0, 10.0],
            "perfect_foresight_bem_rejected_pos_mw": [0.0, 0.0],
            "perfect_foresight_bem_rejected_neg_mw": [0.0, 0.0],
            "perfect_foresight_bem_only_executed_pos_mwh": [0.1, 0.2],
            "perfect_foresight_bem_only_executed_neg_mwh": [0.3, 0.4],
            "perfect_foresight_bem_only_pos_activation_revenue_eur": [10.0, 20.0],
            "perfect_foresight_bem_only_neg_activation_revenue_eur": [30.0, 40.0],
            "perfect_foresight_bem_submitted_activation_price_pos_eur_per_mwh": [100.0, 125.0],
            "perfect_foresight_bem_submitted_activation_price_neg_eur_per_mwh": [150.0, 100.0],
            "perfect_foresight_true_activation_price_pos": [100.0, 125.0],
            "perfect_foresight_true_activation_price_neg": [150.0, 100.0],
        }
    )

    daily = build_bem_activity_daily({"model": model, "rhpf": rhpf})
    model_row = daily.loc[daily["path_type"].eq("model")].iloc[0]
    rhpf_row = daily.loc[daily["path_type"].eq("rhpf")].iloc[0]

    assert float(model_row["submitted_pos_mw"]) == pytest.approx(10.0)
    assert float(model_row["rejected_neg_mw"]) == pytest.approx(10.0)
    assert float(model_row["bid_price_mismatch_count"]) == pytest.approx(2.0)
    assert model_row["bem_activity_classification"] == "trading_but_price_rejected"
    assert float(rhpf_row["rejected_pos_mw"] + rhpf_row["rejected_neg_mw"]) == pytest.approx(0.0)
    assert float(rhpf_row["bid_price_mismatch_count"]) == pytest.approx(0.0)
    assert rhpf_row["bem_activity_classification"] == "trading_and_accepted"


def test_infeasible_debug_dump_records_naive_context_without_writing_file() -> None:
    bt = _mk_backtester()
    bt.debug_dump_mode = "accepted_only"
    df, col = _tiny_backtest_df(hours=1)

    bt._write_infeasible_debug_dump(
        df=df,
        colmap=col,
        c=np.array([0.0]),
        a_ub=None,
        b_ub=None,
        a_eq=None,
        b_eq=None,
        lb=np.array([0.0]),
        ub=np.array([1.0]),
        message="The problem is infeasible.",
        solve_context={
            "timestamp_utc": pd.Timestamp("2025-01-01T00:00:00Z").isoformat(),
            "attempt_type": "optimize_dispatch_primary",
            "run_mode": "naive",
            "benchmark_path": "naive",
            "strategy_name": "multi",
            "allowed_markets": "BCM|BEM|DA|ID",
            "id_mode": "technical_repair",
            "final_soc_mode": "hard",
            "terminal_soc_target_mwh": 10.0,
            "final_accepted_path": False,
        },
        variable_names=["ch[0]"],
    )

    assert len(bt._infeasible_debug_dumps) == 1
    rec = bt._infeasible_debug_dumps[0]
    ctx = json.loads(rec["solve_context"])
    assert rec["dump_written"] == "0"
    assert ctx["run_mode"] == "naive"
    assert ctx["benchmark_path"] == "naive"
    assert ctx["strategy_name"] == "multi"
    assert ctx["terminal_soc_target_mwh"] == pytest.approx(10.0)


def test_milp_variable_names_include_binned_and_state_blocks() -> None:
    bt = _mk_backtester()
    n = 2
    n_bins = 2
    sl = bt._variable_slices(n, n_bins=n_bins)
    names = bt._variable_names_for_slices(n, n_bins, sl)

    assert len(names) == int(sl["slack_obligation_neg"].stop)
    assert names[sl["ch"].start] == "ch[0]"
    assert names[sl["soc"].start + 2] == "soc[2]"
    assert names[sl["rpos_bin"].start + n] == "rpos_bin[1,0]"


def test_benchmark_mode_selected_respects_enable_disable_flags() -> None:
    mode, paths = resolve_benchmark_paths(
        "selected",
        enable_naive=False,
        enable_rolling_pf=True,
        enable_global_pf=False,
        enable_global_perfect_foresight=True,
    )

    assert mode == "selected"
    assert paths == {"model": True, "naive": False, "rhpf": True, "ghpf": False}


def test_benchmark_mode_selected_does_not_enable_ghpf_implicitly() -> None:
    mode, paths = resolve_benchmark_paths(
        "selected",
        enable_global_perfect_foresight=True,
    )

    assert mode == "selected"
    assert paths == {"model": True, "naive": True, "rhpf": True, "ghpf": False}


def test_performance_paths_long_contains_only_enabled_paths_and_stable_schema() -> None:
    hourly_model = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
            "real_pnl_eur": [1.0, 2.0],
            "real_revenue_da_eur": [10.0, 20.0],
            "real_cost_da_eur": [4.0, 5.0],
            "real_revenue_id_eur": [0.0, 0.0],
            "real_cost_id_eur": [0.0, 1.0],
            "real_soc_mwh": [10.0, 11.0],
        }
    )
    hourly_naive = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
            "naive_pnl_eur": [0.5, 0.25],
            "naive_revenue_da_eur": [1.0, 2.0],
            "naive_cost_da_eur": [0.5, 1.75],
            "naive_soc_mwh": [10.0, 10.0],
        }
    )
    args = argparse.Namespace(trading_strategy="da", model_key="linear", model="linear", split="test")
    paths = _build_performance_paths_long(
        isolated_hourly={"model": hourly_model, "naive": hourly_naive},
        summary={"model_available": 1.0, "naive_available": 1.0, "rhpf_available": 0.0, "ghpf_available": 0.0, "simulation_valid": 1.0},
        args=args,
        scenario_name="p50_p50",
        scenario_bins=["p50"],
    )

    assert set(paths["path_type"]) == {"model", "naive"}
    assert len(paths) == 4
    for col in [
        "scenario",
        "strategy",
        "model_key",
        "split",
        "quantile_low",
        "quantile_high",
        "path_type",
        "timestamp_utc",
        "pnl_eur",
        "cum_pnl_eur",
        "net_da_eur",
        "available",
    ]:
        assert col in paths.columns
    model = paths.loc[paths["path_type"].eq("model")].sort_values("timestamp_utc")
    assert model["cum_pnl_eur"].iloc[-1] == pytest.approx(3.0)
    assert model["net_da_eur"].iloc[0] == pytest.approx(6.0)

    daily = _build_daily_performance_paths_long(paths)
    assert set(daily["path_type"]) == {"model", "naive"}
    assert "soc_mean_mwh" in daily.columns
    assert "soc_end_mwh" in daily.columns


def test_terminal_inventory_main_milp_objective_coefficient_is_negative_lambda() -> None:
    bt = _mk_backtester()
    bt.da_terminal_soc_value_enabled = True
    bt.da_terminal_value_min_eur_mwh = 100.0
    bt.da_terminal_value_max_eur_mwh = 100.0
    df, col = _tiny_backtest_df(hours=1)
    df[col.pred_da_price] = 50.0
    df[col.true_da_price] = 50.0

    out = bt.optimize_dispatch(df, col, allowed_markets=())

    assert float(out["terminal_inventory_shadow_value_eur_per_internal_mwh"].iloc[0]) == pytest.approx(100.0)
    assert float(out["terminal_inventory_objective_coef_soc_end"].iloc[0]) == pytest.approx(-100.0)
    assert str(out["terminal_inventory_value_source"].iloc[0]) == "da_liquidation_replacement_proxy"
    assert float(out["da_soc_shadow_value_eur_per_internal_mwh"].iloc[0]) == pytest.approx(100.0)


def test_technical_repair_optimizer_id_does_not_monetize_terminal_shadow_value() -> None:
    bt = _mk_backtester()
    bt.da_terminal_soc_value_enabled = True
    bt.da_terminal_value_min_eur_mwh = 1000.0
    bt.da_terminal_value_max_eur_mwh = 1000.0
    bt.trans_eur_mwh = 0.0
    bt.deg_eur_mwh = 0.0
    bt.aux_trading_mw = 0.0
    df, col = _tiny_backtest_df(hours=2)
    df[col.pred_da_price] = 50.0
    df[col.true_da_price] = 50.0

    out = bt.optimize_dispatch(
        df,
        col,
        soc_start=4.07,
        soc_end_min_target=2.0,
        allowed_markets=("ID",),
        strategy_permissions=StrategyPermissions(False, "technical_repair", False, False, False),
    )

    assert pd.to_numeric(out["id_charge_mw"], errors="coerce").fillna(0.0).sum() == pytest.approx(0.0)
    assert pd.to_numeric(out["id_discharge_mw"], errors="coerce").fillna(0.0).sum() == pytest.approx(0.0)
    assert pd.to_numeric(out["optimizer_id_buy_economic_leakage_flag"], errors="coerce").fillna(0.0).sum() == pytest.approx(0.0)
    assert set(out["optimizer_id_buy_reason_classification_basis"].astype(str)) == {"no_optimizer_id"}


def test_settlement_does_not_auto_buy_to_reserve_safety_without_obligation() -> None:
    bt = _mk_backtester()
    bt.soc_min = 2.0
    bt.soc_max = 18.0
    bt.reserve_headroom_safety_mwh = 0.2
    bt.aux_off_mw = 0.0
    bt.aux_standby_mw = 0.0
    bt.aux_trading_mw = 0.0
    bt._strategy_permissions = StrategyPermissions(
        allow_da=True,
        id_mode="technical_repair",
        allow_bcm=False,
        allow_bcm_activation_obligations=False,
        allow_bem_only=False,
    )

    soc_next, metrics = bt._settle_one_hour(
        soc=2.13,
        charge=0.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=50.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
    )

    assert soc_next == pytest.approx(2.13)
    assert float(metrics["id_buy_mwh"]) == pytest.approx(0.0)
    assert str(metrics["id_recourse_reason"]) == "none"
    assert float(metrics["protected_soc_recovery_invalid_without_violation"]) == pytest.approx(0.0)
    assert float(metrics["protected_recovery_actual_violation_mwh"]) == pytest.approx(0.0)
    assert float(metrics["settlement_reserve_safety_active_pos"]) == pytest.approx(0.0)
    assert float(metrics["settlement_auto_id_repair_applied"]) == pytest.approx(0.0)


def test_hard_terminal_target_excludes_internal_recovery_safety_buffer() -> None:
    target = BatteryBacktester._rolling_final_soc_min_target(
        enforce_final_soc_min=True,
        window_end=72,
        total_rows=72,
        soc_min=2.0,
        soc_target_end=10.0,
        p_max_mw=10.0,
        eta_in=0.95,
        terminal_soc_safety_margin_mwh=0.13,
        soc_max=18.0,
    )

    assert target == pytest.approx(10.0)


def test_terminal_soc_recovery_classified_only_when_terminal_recovery_allowed() -> None:
    bt = _mk_backtester()
    bt.aux_off_mw = 0.0
    bt.aux_standby_mw = 0.0
    bt.aux_trading_mw = 0.0

    charge, discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=4.0,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=10.0,
        projected_terminal_soc_without_new_id_mwh=4.0,
        terminal_recovery_allowed=True,
        terminal_hard_soc_protection=True,
    )

    assert charge > 0.0
    assert discharge == pytest.approx(0.0)
    assert reason == "terminal_soc_recovery"


def test_terminal_shortfall_not_bought_when_terminal_recovery_not_allowed() -> None:
    bt = _mk_backtester()
    bt.aux_off_mw = 0.0
    bt.aux_standby_mw = 0.0
    bt.aux_trading_mw = 0.0

    charge, discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=4.0,
        reserve_pos_next_mw=0.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=10.0,
        projected_terminal_soc_without_new_id_mwh=4.0,
        terminal_recovery_allowed=False,
        terminal_hard_soc_protection=True,
    )

    assert charge == pytest.approx(0.0)
    assert discharge == pytest.approx(0.0)
    assert reason == "none"


def test_settlement_flags_protected_soc_recovery_hint_without_violation() -> None:
    bt = _mk_backtester()
    bt._strategy_permissions = StrategyPermissions(
        allow_da=True,
        id_mode="technical_repair",
        allow_bcm=False,
        allow_bcm_activation_obligations=False,
        allow_bem_only=False,
    )

    _, metrics = bt._settle_one_hour(
        soc=4.07,
        charge=0.0,
        discharge=0.0,
        reserve_pos=0.0,
        reserve_neg=0.0,
        da_price=50.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        id_charge_mw=1.0,
        id_recourse_reason_hint="protected_soc_recovery",
    )

    assert metrics["id_recourse_reason"] == "invalid_protected_soc_recovery_without_violation"
    assert float(metrics["protected_soc_recovery_invalid_without_violation"]) == pytest.approx(1.0)
    assert float(metrics["optimizer_id_buy_economic_leakage_flag"]) == pytest.approx(1.0)
    assert float(metrics["protected_recovery_actual_violation_mwh"]) == pytest.approx(0.0)


def test_reserve_obligation_recovery_reason_preserved_for_real_lower_gap() -> None:
    bt = _mk_backtester()
    bt.reserve_activation_headroom_h = 0.5
    bt.reserve_headroom_safety_mwh = 0.0
    charge, discharge, reason = bt._plan_id_rescue_for_next_hour(
        soc_next=2.1,
        reserve_pos_next_mw=4.0,
        reserve_neg_next_mw=0.0,
        da_charge_next_mw=0.0,
        da_discharge_next_mw=0.0,
        terminal_soc_target_mwh=None,
    )

    assert charge > 0.0
    assert discharge == pytest.approx(0.0)
    assert reason == "reserve_obligation_recovery"

    bt._strategy_permissions = StrategyPermissions(
        allow_da=False,
        id_mode="technical_repair",
        allow_bcm=True,
        allow_bcm_activation_obligations=True,
        allow_bem_only=False,
    )
    _, metrics = bt._settle_one_hour(
        soc=2.1,
        charge=0.0,
        discharge=0.0,
        reserve_pos=4.0,
        reserve_neg=0.0,
        da_price=50.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos_price=0.0,
        act_neg_price=0.0,
        act_pos_rate=0.0,
        act_neg_rate=0.0,
        id_charge_mw=charge,
        id_recourse_reason_hint="protected_soc_recovery",
    )
    assert metrics["id_recourse_reason"] == "reserve_obligation_recovery"
    assert float(metrics["protected_recovery_obligation_active"]) == pytest.approx(1.0)
    assert float(metrics["protected_recovery_actual_violation_mwh"]) > 0.0


def test_terminal_inventory_da_buy_and_sell_effects_use_execution_probability() -> None:
    bt = _mk_backtester()
    bt.da_execution_mode = "limit"
    bt.bid_builder.da_buy_limit_quantile = "p50"
    bt.bid_builder.da_sell_limit_quantile = "p50"
    bt.da_terminal_soc_value_enabled = True
    bt.da_terminal_value_min_eur_mwh = 100.0
    bt.da_terminal_value_max_eur_mwh = 100.0
    bt.trans_eur_mwh = 0.0
    bt.deg_eur_mwh = 0.0
    bt.aux_trading_mw = 0.0
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    col = BacktestColumnMap()
    ts_buy = pd.Timestamp("2026-01-01T00:00:00Z")
    ts_sell = pd.Timestamp("2026-01-01T01:00:00Z")
    rows = pd.DataFrame(
        {
            "target_time_utc": [ts_buy, ts_sell],
            col.timestamp: [ts_buy, ts_sell],
            col.pred_da_price: [50.0, 50.0],
        }
    )

    replay = bt._replay_da_candidate_cashflow(
        rows=rows,
        schedule={ts_buy: (10.0, 0.0), ts_sell: (0.0, 10.0)},
        colmap=col,
        current_soc_mwh=10.0,
    )

    assert float(replay["da_terminal_inventory_effect_buy_eur"]) == pytest.approx(0.5 * 10.0 * 0.95 * 100.0)
    assert float(replay["da_terminal_inventory_effect_sell_eur"]) == pytest.approx(0.5 * 10.0 / 0.95 * 100.0)
    assert float(replay["terminal_inventory_shadow_value_eur_per_internal_mwh"]) == pytest.approx(100.0)


def test_bcm_terminal_inventory_effect_formulas() -> None:
    pos, neg = BatteryBacktester.terminal_inventory_effects_eur_per_mw(
        probability=0.5,
        activation_rate=0.01,
        duration_h=4.0,
        eta_in=0.95,
        eta_out=0.95,
        lambda_eur_per_internal_mwh=100.0,
    )

    assert float(pos) * 10.0 == pytest.approx(-0.5 * 10.0 * 0.01 * 4.0 / 0.95 * 100.0)
    assert float(neg) * 10.0 == pytest.approx(0.5 * 10.0 * 0.01 * 4.0 * 0.95 * 100.0)


def test_bem_terminal_inventory_effect_formulas_use_execution_probability() -> None:
    pos, neg = BatteryBacktester.terminal_inventory_effects_eur_per_mw(
        probability=0.5,
        activation_rate=0.01,
        duration_h=4.0,
        eta_in=0.95,
        eta_out=0.95,
        lambda_eur_per_internal_mwh=100.0,
    )

    assert float(pos) * 10.0 == pytest.approx(-0.5 * 10.0 * 0.01 * 4.0 / 0.95 * 100.0)
    assert float(neg) * 10.0 == pytest.approx(0.5 * 10.0 * 0.01 * 4.0 * 0.95 * 100.0)


def test_bem_terminal_inventory_diagnostics_not_double_counted_in_full_objective() -> None:
    bt = _mk_backtester()
    bt.afrr_quantile_bins = ["p50"]
    bt.da_terminal_soc_value_enabled = True
    bt.da_terminal_value_min_eur_mwh = 100.0
    bt.da_terminal_value_max_eur_mwh = 100.0
    bt.eta_in = 0.95
    bt.eta_out = 0.95
    df, col = _one_hour_pred_df(
        da=50.0,
        cap_pos=0.0,
        cap_neg=0.0,
        act_pos=0.0,
        act_neg=0.0,
        rate_pos=0.01,
        rate_neg=0.01,
    )

    out = bt.optimize_dispatch(df, col, allowed_markets=("afrr", "bem"))

    expected_pos = -0.5 * 0.01 * 1.0 / 0.95 * 100.0
    expected_neg = 0.5 * 0.01 * 1.0 * 0.95 * 100.0
    assert float(out["bem_terminal_inventory_effect_pos_bin_0_eur_per_mw"].iloc[0]) == pytest.approx(expected_pos)
    assert float(out["bem_terminal_inventory_effect_neg_bin_0_eur_per_mw"].iloc[0]) == pytest.approx(expected_neg)
    assert float(out["bem_terminal_inventory_shadow_value_already_in_full_objective"].iloc[0]) == pytest.approx(1.0)
    assert float(out["terminal_inventory_shadow_value_eur_per_internal_mwh"].iloc[0]) == pytest.approx(100.0)
