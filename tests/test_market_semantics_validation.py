from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import check_market_semantics as market_semantics


def _base_summary(strategy: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "trading_strategy": strategy,
        "simulation_valid": 1.0,
        "thesis_reportable": 1.0,
        "invalid_reason": "",
        "base_strategy_id_mode": "none" if strategy == "da" else "technical_repair",
        "resolved_id_mode": "technical_repair",
        "id_mode": "technical_repair",
        "id_recourse_mode": "common",
        "id_allowed": 1.0,
        "id_economic_enabled": 0.0,
        "id_technical_repair_enabled": 1.0,
        "command_line_args": "{}",
        "soc_min_mwh": 2.0,
        "soc_max_mwh": 18.0,
        "p_max_mw": 10.0,
        "strict_simulation_validity": 1.0,
        "final_soc_mode": "hard",
        "final_soc_actual_mwh": 10.0,
        "final_soc_target_mwh": 10.0,
        "final_soc_constraint_satisfied": 1.0,
        "terminal_soc_repair_cost_eur": 0.0,
    }
    payload.update(overrides)
    return payload


def _base_hourly() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="h")
    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "real_pnl_eur": [0.0, 0.0],
            "real_revenue_da_eur": [0.0, 0.0],
            "real_cost_da_eur": [0.0, 0.0],
            "real_revenue_id_eur": [0.0, 0.0],
            "real_cost_id_eur": [0.0, 0.0],
            "real_id_pnl_eur": [0.0, 0.0],
            "real_revenue_capacity_eur": [0.0, 0.0],
            "real_revenue_activation_eur": [0.0, 0.0],
            "real_bcm_linked_activation_revenue_eur": [0.0, 0.0],
            "real_bem_only_activation_revenue_eur": [0.0, 0.0],
            "real_submitted_da_buy_mw": [0.0, 0.0],
            "real_submitted_da_sell_mw": [0.0, 0.0],
            "real_executed_charge_mw": [0.0, 0.0],
            "real_executed_discharge_mw": [0.0, 0.0],
            "real_da_buy_mwh": [0.0, 0.0],
            "real_da_sell_mwh": [0.0, 0.0],
            "real_submitted_afrr_pos_mw": [0.0, 0.0],
            "real_submitted_afrr_neg_mw": [0.0, 0.0],
            "real_executed_reserve_pos_mw": [0.0, 0.0],
            "real_executed_reserve_neg_mw": [0.0, 0.0],
            "real_bem_only_submitted_pos_mw": [0.0, 0.0],
            "real_bem_only_submitted_neg_mw": [0.0, 0.0],
            "real_bem_only_executed_pos_mwh": [0.0, 0.0],
            "real_bem_only_executed_neg_mwh": [0.0, 0.0],
            "real_id_buy_mwh": [0.0, 0.0],
            "real_id_sell_mwh": [0.0, 0.0],
            "real_id_charge_mw": [0.0, 0.0],
            "real_id_discharge_mw": [0.0, 0.0],
            "id_recourse_reason": ["none", "none"],
            "bcm_capacity_block_id": ["blk_a", "blk_a"],
        }
    )


def _performance_metrics(realized_net: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "realized_net_revenue_eur": realized_net,
                "da_gross_revenue_eur": 0.0,
                "da_gross_cost_eur": 0.0,
                "id_gross_revenue_eur": 0.0,
                "id_gross_cost_eur": 0.0,
                "afrr_capacity_revenue_eur": 0.0,
                "afrr_activation_revenue_eur": realized_net,
                "realized_degradation_cost_eur": 0.0,
                "realized_aux_cost_eur": 0.0,
                "transaction_cost_eur": 0.0,
                "offer_cost_eur": 0.0,
                "penalty_cost_eur": 0.0,
                "terminal_soc_repair_cost_eur": 0.0,
                "total_costs_eur": 0.0,
            }
        ]
    )


def _daily_metrics(realized_net: float) -> pd.DataFrame:
    return pd.DataFrame([{"date": "2026-01-01", "net_revenue_eur": realized_net}])


def _perf_debug() -> pd.DataFrame:
    return pd.DataFrame([{"checked_daily_to_scenario": 1.0, "daily_abs_error": 0.0, "hourly_abs_error": 0.0}])


def _write_scenario(
    root: Path,
    *,
    strategy: str,
    scenario: str,
    hourly: pd.DataFrame,
    summary_overrides: dict[str, object] | None = None,
) -> Path:
    scen = root / strategy / scenario
    scen.mkdir(parents=True, exist_ok=True)
    realized_net = float(pd.to_numeric(hourly["real_pnl_eur"], errors="coerce").fillna(0.0).sum())
    summary = _base_summary(strategy, **(summary_overrides or {}))
    (scen / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    hourly.to_csv(scen / "backtest_hourly.csv", index=False)
    _performance_metrics(realized_net).to_csv(scen / "performance_metrics.csv", index=False)
    _daily_metrics(realized_net).to_csv(scen / "daily_performance_metrics.csv", index=False)
    _perf_debug().to_csv(scen / "performance_metric_reconciliation_debug.csv", index=False)
    return scen


def test_market_validator_valid_afrr_only_with_bcm_and_bem_passes(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_submitted_afrr_pos_mw"] = [1.0, 1.0]
    hourly["real_executed_reserve_pos_mw"] = [1.0, 1.0]
    hourly["real_bem_only_submitted_pos_mw"] = [0.5, 1.0]
    hourly["real_bem_only_executed_pos_mwh"] = [0.5, 1.0]
    hourly["real_bcm_linked_activation_revenue_eur"] = [5.0, 5.0]
    hourly["real_bem_only_activation_revenue_eur"] = [1.0, 2.0]
    hourly["real_revenue_activation_eur"] = [6.0, 7.0]
    hourly["real_pnl_eur"] = [6.0, 7.0]
    scen = _write_scenario(tmp_path, strategy="afrr", scenario="p50_p50", hourly=hourly)
    row, vdf = market_semantics.validate_scenario(scen)
    assert bool(row["thesis_semantics_ok"])
    assert vdf.empty


def test_market_validator_da_activity_forbidden_in_afrr_only_fails(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_da_buy_mwh"] = [1.0, 0.0]
    hourly["real_pnl_eur"] = [0.0, 0.0]
    scen = _write_scenario(tmp_path, strategy="afrr", scenario="p50_p50", hourly=hourly)
    _, vdf = market_semantics.validate_scenario(scen)
    assert "forbidden_da_quantity" in set(vdf["check_name"])


def test_market_validator_bcm_only_with_bem_activity_fails(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_submitted_afrr_pos_mw"] = [1.0, 1.0]
    hourly["real_bem_only_submitted_pos_mw"] = [1.0, 0.0]
    scen = _write_scenario(tmp_path, strategy="bcm", scenario="p50_p50", hourly=hourly)
    _, vdf = market_semantics.validate_scenario(scen)
    assert "forbidden_bem_quantity" in set(vdf["check_name"])


def test_market_validator_bem_only_with_bcm_activity_fails(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_submitted_afrr_pos_mw"] = [1.0, 1.0]
    scen = _write_scenario(tmp_path, strategy="bem", scenario="p50_p50", hourly=hourly)
    _, vdf = market_semantics.validate_scenario(scen)
    assert "forbidden_bcm_quantity" in set(vdf["check_name"])


def test_market_validator_da_only_with_bcm_and_bem_activity_fails(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_submitted_afrr_pos_mw"] = [1.0, 1.0]
    hourly["real_bem_only_submitted_pos_mw"] = [0.5, 0.0]
    scen = _write_scenario(tmp_path, strategy="da", scenario="p50_p50", hourly=hourly)
    _, vdf = market_semantics.validate_scenario(scen)
    assert "forbidden_bcm_quantity" in set(vdf["check_name"])
    assert "forbidden_bem_quantity" in set(vdf["check_name"])


def test_market_validator_multi_with_all_markets_passes(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_submitted_da_buy_mw"] = [1.0, 0.0]
    hourly["real_da_buy_mwh"] = [1.0, 0.0]
    hourly["real_submitted_afrr_pos_mw"] = [1.0, 1.0]
    hourly["real_executed_reserve_pos_mw"] = [1.0, 1.0]
    hourly["real_bem_only_submitted_pos_mw"] = [0.5, 1.0]
    hourly["real_bem_only_executed_pos_mwh"] = [0.5, 1.0]
    hourly["real_id_buy_mwh"] = [0.0, 0.1]
    hourly["real_id_charge_mw"] = [0.0, 0.1]
    hourly["id_recourse_reason"] = ["none", "technical_repair"]
    hourly["real_bcm_linked_activation_revenue_eur"] = [4.0, 4.0]
    hourly["real_bem_only_activation_revenue_eur"] = [1.0, 2.0]
    hourly["real_revenue_activation_eur"] = [5.0, 6.0]
    hourly["real_pnl_eur"] = [5.0, 6.0]
    scen = _write_scenario(tmp_path, strategy="multi", scenario="p50_p50", hourly=hourly)
    row, vdf = market_semantics.validate_scenario(scen)
    assert bool(row["thesis_semantics_ok"])
    assert vdf.empty


def test_market_validator_id_disabled_with_activity_fails(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_id_buy_mwh"] = [1.0, 0.0]
    hourly["real_id_charge_mw"] = [1.0, 0.0]
    hourly["id_recourse_reason"] = ["technical_repair", "none"]
    scen = _write_scenario(
        tmp_path,
        strategy="multi",
        scenario="p50_p50",
        hourly=hourly,
        summary_overrides={"id_recourse_mode": "disabled", "resolved_id_mode": "none", "id_mode": "none", "id_allowed": 0.0},
    )
    _, vdf = market_semantics.validate_scenario(scen)
    assert "id_disabled_nonzero" in set(vdf["check_name"])


def test_market_validator_bem_hourly_variation_does_not_fail_bcm_blocks(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_submitted_afrr_pos_mw"] = [2.0, 2.0]
    hourly["real_executed_reserve_pos_mw"] = [2.0, 2.0]
    hourly["real_bem_only_submitted_pos_mw"] = [0.0, 1.0]
    scen = _write_scenario(tmp_path, strategy="afrr", scenario="p50_p50", hourly=hourly)
    row, vdf = market_semantics.validate_scenario(scen)
    assert bool(row["bcm_block_ok"])
    assert "bcm_intrablock_variation" not in set(vdf["check_name"])


def test_market_validator_bcm_block_variation_fails(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_submitted_afrr_pos_mw"] = [1.0, 2.0]
    scen = _write_scenario(tmp_path, strategy="afrr", scenario="p50_p50", hourly=hourly)
    _, vdf = market_semantics.validate_scenario(scen)
    assert "bcm_intrablock_variation" in set(vdf["check_name"])


def test_market_validator_activation_revenue_split_mismatch_fails(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_bcm_linked_activation_revenue_eur"] = [2.0, 2.0]
    hourly["real_bem_only_activation_revenue_eur"] = [1.0, 1.0]
    hourly["real_revenue_activation_eur"] = [10.0, 10.0]
    hourly["real_pnl_eur"] = [10.0, 10.0]
    scen = _write_scenario(tmp_path, strategy="multi", scenario="p50_p50", hourly=hourly)
    _, vdf = market_semantics.validate_scenario(scen)
    assert "activation_revenue_split_mismatch" in set(vdf["check_name"])


def test_market_validator_writes_expected_outputs(tmp_path: Path) -> None:
    hourly = _base_hourly()
    _write_scenario(tmp_path, strategy="multi", scenario="p50_p50", hourly=hourly)
    out_dir = tmp_path / "market_semantics_validation"
    report = market_semantics.run_market_semantics_validation(simulation_root=tmp_path, out_dir=out_dir)
    assert (out_dir / "market_semantics_summary.csv").exists()
    assert (out_dir / "market_semantics_violations.csv").exists()
    assert (out_dir / "market_semantics_report.json").exists()
    assert "output_paths" in report
