from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_market_semantics import discover_scenarios, validate_scenario  # noqa: E402


def _base_hourly() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-05-01", periods=2, freq="h", tz="UTC"),
            "real_pnl_eur": [1.0, 2.0],
            "real_revenue_da_eur": [1.0, 2.0],
            "real_cost_da_eur": [0.0, 0.0],
            "real_soc_mwh": [10.0, 10.5],
            "real_headroom_violation_pos_mwh": [0.0, 0.0],
            "real_headroom_violation_neg_mwh": [0.0, 0.0],
            "real_headroom_margin_pos_mwh": [1.0, 1.0],
            "real_headroom_margin_neg_mwh": [1.0, 1.0],
            "optimization_error_code": ["ok", "ok"],
            "is_fallback_hour": [0.0, 0.0],
            "real_submitted_da_buy_mw": [0.0, 0.0],
            "real_submitted_da_sell_mw": [1.0, 2.0],
            "real_executed_charge_mw": [0.0, 0.0],
            "real_executed_discharge_mw": [1.0, 2.0],
            "real_da_buy_mwh": [0.0, 0.0],
            "real_da_sell_mwh": [1.0, 2.0],
            "real_bem_only_submitted_pos_mw": [0.0, 0.0],
            "real_bem_only_submitted_neg_mw": [0.0, 0.0],
            "real_bem_only_executed_pos_mwh": [0.0, 0.0],
            "real_bem_only_executed_neg_mwh": [0.0, 0.0],
            "real_submitted_afrr_pos_mw": [0.0, 0.0],
            "real_submitted_afrr_neg_mw": [0.0, 0.0],
            "real_executed_reserve_pos_mw": [0.0, 0.0],
            "real_executed_reserve_neg_mw": [0.0, 0.0],
            "real_revenue_capacity_eur": [0.0, 0.0],
            "real_bcm_linked_activation_revenue_eur": [0.0, 0.0],
            "real_bem_only_activation_revenue_eur": [0.0, 0.0],
            "real_revenue_activation_eur": [0.0, 0.0],
            "real_revenue_id_eur": [0.0, 0.0],
            "real_cost_id_eur": [0.0, 0.0],
            "real_id_buy_mwh": [0.0, 0.0],
            "real_id_sell_mwh": [0.0, 0.0],
            "real_id_charge_mw": [0.0, 0.0],
            "real_id_discharge_mw": [0.0, 0.0],
            "id_recourse_reason": ["none", "none"],
            "bcm_capacity_block_id": ["blk0", "blk0"],
        }
    )


def _base_perf() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "realized_net_revenue_eur": 3.0,
                "da_gross_revenue_eur": 3.0,
                "da_gross_cost_eur": 0.0,
                "da_pnl_eur": 3.0,
                "id_gross_revenue_eur": 0.0,
                "id_gross_cost_eur": 0.0,
                "id_recourse_pnl_eur": 0.0,
                "bcm_capacity_revenue_eur": 0.0,
                "bcm_linked_activation_revenue_eur": 0.0,
                "bcm_pnl_eur": 0.0,
                "bem_activation_revenue_eur": 0.0,
                "bem_pnl_eur": 0.0,
                "afrr_capacity_revenue_eur": 0.0,
                "afrr_activation_revenue_eur": 0.0,
                "afrr_pnl_eur": 0.0,
                "realized_degradation_cost_eur": 0.0,
                "realized_aux_cost_eur": 0.0,
                "transaction_cost_eur": 0.0,
                "offer_cost_eur": 0.0,
                "penalty_cost_eur": 0.0,
                "terminal_soc_repair_cost_eur": 0.0,
                "total_costs_eur": 0.0,
                "trading_strategy": "multi",
                "id_recourse_mode": "common",
            }
        ]
    )


def _base_daily() -> pd.DataFrame:
    return pd.DataFrame({"date": ["2025-05-01", "2025-05-02"], "net_revenue_eur": [1.0, 2.0]})


def _base_summary(strategy: str = "multi", id_mode: str = "common") -> dict[str, object]:
    return {
        "trading_strategy": strategy,
        "simulation_valid": 1,
        "thesis_reportable": 1,
        "invalid_reason": "",
        "id_recourse_mode": id_mode,
        "soc_min_mwh": 2.0,
        "soc_max_mwh": 18.0,
        "p_max_mw": 5.0,
        "final_soc_target_mwh": 10.0,
        "final_soc_actual_mwh": 10.0,
        "final_soc_constraint_satisfied": True,
        "strict_simulation_validity": True,
        "final_soc_mode": "hard",
        "command_line_args": {"disallow_simultaneous_bem_only_pos_neg": False},
    }


def _write_scenario(
    root: Path,
    *,
    strategy: str = "multi",
    model_dir: str | None = None,
    scenario_name: str = "p50_p50",
    summary_extra: dict[str, object] | None = None,
    hourly: pd.DataFrame | None = None,
    perf: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
    perf_debug: pd.DataFrame | None = None,
) -> Path:
    if model_dir:
        scen = root / model_dir / strategy / scenario_name
    else:
        scen = root / strategy / scenario_name
    scen.mkdir(parents=True, exist_ok=True)
    summary = _base_summary(strategy=strategy)
    if summary_extra:
        summary.update(summary_extra)
    (scen / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (hourly if hourly is not None else _base_hourly()).to_parquet(scen / "backtest_hourly.parquet", index=False)
    (perf if perf is not None else _base_perf()).to_csv(scen / "performance_metrics.csv", index=False)
    (daily if daily is not None else _base_daily()).to_csv(scen / "daily_performance_metrics.csv", index=False)
    if perf_debug is not None:
        perf_debug.to_csv(scen / "performance_metric_reconciliation_debug.csv", index=False)
    return scen


def _validate(root: Path, *, include_invalid: bool = True) -> tuple[dict[str, object], pd.DataFrame]:
    row, vdf = validate_scenario(root, include_invalid=include_invalid)
    return row, vdf


def test_da_only_rejects_bcm_quantity_leakage(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_submitted_afrr_pos_mw"] = [1.0, 0.0]
    scen = _write_scenario(tmp_path, strategy="da_only", hourly=hourly)
    _, vdf = _validate(scen)
    assert "forbidden_bcm_quantity" in set(vdf["check_name"])


def test_da_only_rejects_bem_pnl_leakage(tmp_path: Path) -> None:
    perf = _base_perf()
    perf.loc[0, "bem_pnl_eur"] = 5.0
    scen = _write_scenario(tmp_path, strategy="da_only", perf=perf)
    _, vdf = _validate(scen)
    assert "forbidden_bem_pnl" in set(vdf["check_name"])


def test_bcm_only_allows_bcm_capacity_and_bcm_linked_activation(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_revenue_da_eur"] = [0.0, 0.0]
    hourly["real_submitted_da_sell_mw"] = [0.0, 0.0]
    hourly["real_executed_discharge_mw"] = [0.0, 0.0]
    hourly["real_da_sell_mwh"] = [0.0, 0.0]
    hourly["real_pnl_eur"] = [2.0, 3.0]
    hourly["real_submitted_afrr_pos_mw"] = [1.0, 1.0]
    hourly["real_executed_reserve_pos_mw"] = [1.0, 1.0]
    hourly["real_revenue_capacity_eur"] = [2.0, 2.0]
    hourly["real_bcm_linked_activation_revenue_eur"] = [0.5, 0.5]
    perf = _base_perf()
    perf.loc[0, ["realized_net_revenue_eur", "da_gross_revenue_eur", "da_pnl_eur"]] = [5.0, 0.0, 0.0]
    perf.loc[0, ["bcm_capacity_revenue_eur", "bcm_linked_activation_revenue_eur", "bcm_pnl_eur", "afrr_capacity_revenue_eur", "afrr_activation_revenue_eur"]] = [4.0, 1.0, 5.0, 4.0, 1.0]
    daily = pd.DataFrame({"date": ["2025-05-01"], "net_revenue_eur": [5.0]})
    scen = _write_scenario(tmp_path, strategy="bcm_only", hourly=hourly, perf=perf, daily=daily)
    row, vdf = _validate(scen)
    assert row["thesis_semantics_ok"] is True
    assert not {"forbidden_da_quantity", "forbidden_bem_quantity"} & set(vdf["check_name"])


def test_bcm_only_rejects_discretionary_bem_only_quantity_pnl(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_revenue_da_eur"] = [0.0, 0.0]
    hourly["real_pnl_eur"] = [1.0, 1.0]
    hourly["real_bem_only_submitted_pos_mw"] = [1.0, 0.0]
    perf = _base_perf()
    perf.loc[0, ["realized_net_revenue_eur", "da_gross_revenue_eur", "da_pnl_eur", "bem_pnl_eur"]] = [2.0, 0.0, 0.0, 2.0]
    daily = pd.DataFrame({"date": ["2025-05-01"], "net_revenue_eur": [2.0]})
    scen = _write_scenario(tmp_path, strategy="bcm_only", hourly=hourly, perf=perf, daily=daily)
    _, vdf = _validate(scen)
    assert {"forbidden_bem_quantity", "forbidden_bem_pnl"} <= set(vdf["check_name"])


def test_bem_only_allows_bem_only_and_rejects_bcm_capacity(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_revenue_da_eur"] = [0.0, 0.0]
    hourly["real_submitted_da_sell_mw"] = [0.0, 0.0]
    hourly["real_executed_discharge_mw"] = [0.0, 0.0]
    hourly["real_da_sell_mwh"] = [0.0, 0.0]
    hourly["real_pnl_eur"] = [2.0, 0.0]
    hourly["real_bem_only_submitted_pos_mw"] = [1.0, 0.0]
    hourly["real_bem_only_activation_revenue_eur"] = [2.0, 0.0]
    hourly["real_revenue_activation_eur"] = [2.0, 0.0]
    perf = _base_perf()
    perf.loc[0, ["realized_net_revenue_eur", "da_gross_revenue_eur", "da_pnl_eur", "bem_pnl_eur", "bem_activation_revenue_eur", "afrr_activation_revenue_eur"]] = [2.0, 0.0, 0.0, 2.0, 2.0, 2.0]
    daily = pd.DataFrame({"date": ["2025-05-01"], "net_revenue_eur": [2.0]})
    scen = _write_scenario(tmp_path, strategy="bem_only", hourly=hourly, perf=perf, daily=daily)
    row, vdf = _validate(scen)
    assert row["thesis_semantics_ok"] is True
    hourly["real_submitted_afrr_pos_mw"] = [1.0, 0.0]
    scen_bad = _write_scenario(tmp_path / "bad", strategy="bem_only", hourly=hourly, perf=perf, daily=daily)
    _, vdf_bad = _validate(scen_bad)
    assert "forbidden_bcm_quantity" in set(vdf_bad["check_name"])


def test_afrr_only_allows_bcm_and_bem_but_rejects_da(tmp_path: Path) -> None:
    scen = _write_scenario(tmp_path, strategy="afrr_only")
    _, vdf = _validate(scen)
    assert {"forbidden_da_quantity", "forbidden_da_pnl"} <= set(vdf["check_name"])


def test_multi_allows_all_markets_but_still_requires_accounting(tmp_path: Path) -> None:
    scen = _write_scenario(tmp_path, strategy="multi")
    row, vdf = _validate(scen)
    assert row["thesis_semantics_ok"] is True
    assert vdf.loc[vdf["severity"] == "hard"].empty


def test_id_disabled_forces_all_id_quantity_pnl_zero(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_id_buy_mwh"] = [1.0, 0.0]
    perf = _base_perf()
    perf.loc[0, "id_recourse_pnl_eur"] = 1.0
    scen = _write_scenario(tmp_path, strategy="multi", summary_extra={"id_recourse_mode": "disabled"}, hourly=hourly, perf=perf)
    _, vdf = _validate(scen)
    assert "id_disabled_nonzero" in set(vdf["check_name"])


def test_id_common_allows_id_but_requires_reason_code_when_present(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_id_buy_mwh"] = [1.0, 0.0]
    hourly["id_recourse_reason"] = ["none", "none"]
    scen = _write_scenario(tmp_path, strategy="multi", hourly=hourly)
    _, vdf = _validate(scen)
    assert "missing_id_reason_code" in set(vdf["check_name"])


def test_id_afrr_obligation_only_rejects_id_for_da_only(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_id_buy_mwh"] = [1.0, 0.0]
    hourly["id_recourse_reason"] = ["repair", "none"]
    scen = _write_scenario(tmp_path, strategy="da_only", summary_extra={"id_recourse_mode": "afrr_obligation_only"}, hourly=hourly)
    _, vdf = _validate(scen)
    assert "id_forbidden_for_da_only" in set(vdf["check_name"])


def test_bcm_block_consistency_catches_intrablock_variation(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_revenue_da_eur"] = [0.0, 0.0]
    hourly["real_pnl_eur"] = [0.0, 0.0]
    hourly["real_submitted_afrr_pos_mw"] = [1.0, 2.0]
    scen = _write_scenario(tmp_path, strategy="bcm_only", hourly=hourly, perf=_base_perf().assign(realized_net_revenue_eur=0.0, da_gross_revenue_eur=0.0, da_pnl_eur=0.0))
    _, vdf = _validate(scen)
    assert "bcm_intrablock_variation" in set(vdf["check_name"])


def test_bcm_block_check_does_not_inspect_bem_hourly_columns(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_revenue_da_eur"] = [0.0, 0.0]
    hourly["real_pnl_eur"] = [0.0, 0.0]
    hourly["real_bem_only_submitted_pos_mw"] = [1.0, 2.0]
    perf = _base_perf().assign(realized_net_revenue_eur=0.0, da_gross_revenue_eur=0.0, da_pnl_eur=0.0)
    scen = _write_scenario(tmp_path, strategy="bem_only", hourly=hourly, perf=perf)
    _, vdf = _validate(scen)
    assert "bcm_intrablock_variation" not in set(vdf["check_name"])


def test_soc_min_max_violation_is_caught(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_soc_mwh"] = [1.0, 10.0]
    scen = _write_scenario(tmp_path, strategy="multi", hourly=hourly)
    _, vdf = _validate(scen)
    assert "soc_below_min" in set(vdf["check_name"])


def test_headroom_violation_columns_gt_zero_are_caught(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_headroom_violation_pos_mwh"] = [0.1, 0.0]
    scen = _write_scenario(tmp_path, strategy="multi", hourly=hourly)
    _, vdf = _validate(scen)
    assert "headroom_violation" in set(vdf["check_name"])


def test_negative_headroom_margins_are_caught(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_headroom_margin_pos_mwh"] = [-0.1, 1.0]
    scen = _write_scenario(tmp_path, strategy="multi", hourly=hourly)
    _, vdf = _validate(scen)
    assert "negative_headroom_margin" in set(vdf["check_name"])


def test_power_stack_gt_pmax_is_caught(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["real_executed_discharge_mw"] = [6.0, 0.0]
    scen = _write_scenario(tmp_path, strategy="multi", hourly=hourly)
    _, vdf = _validate(scen)
    assert "positive_power_stack_exceeds_pmax" in set(vdf["check_name"])


def test_non_ok_optimization_error_code_is_caught(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["optimization_error_code"] = ["ok", "infeasible"]
    scen = _write_scenario(tmp_path, strategy="multi", hourly=hourly)
    _, vdf = _validate(scen)
    assert "optimization_error_code_non_ok" in set(vdf["check_name"])


def test_fallback_hour_is_caught(tmp_path: Path) -> None:
    hourly = _base_hourly()
    hourly["is_fallback_hour"] = [0.0, 1.0]
    scen = _write_scenario(tmp_path, strategy="multi", hourly=hourly)
    _, vdf = _validate(scen)
    assert "fallback_hour_present" in set(vdf["check_name"])


def test_final_soc_below_target_is_caught(tmp_path: Path) -> None:
    scen = _write_scenario(tmp_path, strategy="multi", summary_extra={"final_soc_actual_mwh": 8.0, "final_soc_target_mwh": 10.0})
    _, vdf = _validate(scen)
    assert "final_soc_below_target" in set(vdf["check_name"])


def test_hourly_daily_performance_reconciliation_failure_is_caught(tmp_path: Path) -> None:
    daily = pd.DataFrame({"date": ["2025-05-01"], "net_revenue_eur": [99.0]})
    scen = _write_scenario(tmp_path, strategy="multi", daily=daily)
    _, vdf = _validate(scen)
    assert "daily_to_hourly_net_mismatch" in set(vdf["check_name"])


def test_cost_reconciliation_failure_is_caught(tmp_path: Path) -> None:
    perf = _base_perf()
    perf.loc[0, "total_costs_eur"] = 5.0
    scen = _write_scenario(tmp_path, strategy="multi", perf=perf)
    _, vdf = _validate(scen)
    assert "cost_reconciliation_failure" in set(vdf["check_name"])


def test_valid_synthetic_multi_scenario_passes(tmp_path: Path) -> None:
    scen = _write_scenario(tmp_path, strategy="multi")
    row, vdf = _validate(scen)
    assert row["thesis_semantics_ok"] is True
    assert vdf.loc[vdf["severity"] == "hard"].empty


def test_discovery_works_for_nested_layout(tmp_path: Path) -> None:
    _write_scenario(tmp_path, strategy="multi", model_dir="xgb_multi_p50-p50")
    records = discover_scenarios(tmp_path, include_invalid=True)
    assert len(records) == 1
    assert records[0]["strategy"] == "multi"


def test_missing_optional_columns_create_skipped_rows_not_false_passes(tmp_path: Path) -> None:
    hourly = pd.DataFrame({"timestamp_utc": pd.date_range("2025-05-01", periods=1, freq="h", tz="UTC"), "real_pnl_eur": [0.0], "real_soc_mwh": [10.0]})
    perf = pd.DataFrame([{"realized_net_revenue_eur": 0.0, "trading_strategy": "multi", "id_recourse_mode": "common"}])
    scen = _write_scenario(tmp_path, strategy="multi", hourly=hourly, perf=perf, daily=pd.DataFrame({"date": ["2025-05-01"], "net_revenue_eur": [0.0]}))
    row, vdf = _validate(scen)
    assert row["thesis_semantics_ok"] is True
    assert "cost_reconciliation_skipped" in set(vdf["check_name"])
    assert "component_decomposition_skipped" in set(vdf["check_name"])


def test_cli_writes_outputs_and_exit_codes(tmp_path: Path) -> None:
    valid_root = tmp_path / "valid"
    invalid_root = tmp_path / "invalid"
    _write_scenario(valid_root, strategy="multi")
    bad_hourly = _base_hourly()
    bad_hourly["real_submitted_afrr_pos_mw"] = [1.0, 0.0]
    _write_scenario(invalid_root, strategy="da_only", hourly=bad_hourly)

    cp_valid = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_market_semantics.py"),
            "--simulation-root",
            str(valid_root),
        ],
        capture_output=True,
        text=True,
    )
    assert cp_valid.returncode == 0, cp_valid.stdout + cp_valid.stderr
    assert (valid_root / "market_semantics_validation" / "market_semantics_summary.csv").exists()

    cp_invalid = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_market_semantics.py"),
            "--simulation-root",
            str(invalid_root),
        ],
        capture_output=True,
        text=True,
    )
    assert cp_invalid.returncode == 1, cp_invalid.stdout + cp_invalid.stderr
