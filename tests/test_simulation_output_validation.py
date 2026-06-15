from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import validate_simulation_outputs as validate_outputs


def _write_scenario(
    root: Path,
    *,
    strategy: str = "multi",
    scenario: str = "p50_p50",
    realized: float = 100.0,
    rolling: float = 120.0,
    global_pf: float = 150.0,
    global_available: float = 1.0,
    global_verified: float = 1.0,
    global_component_gap: dict[str, object] | None = None,
    global_pf_failure_reason: str = "none",
    afrr: float = 30.0,
    bcm: float = 20.0,
    bem: float = 10.0,
    hourly_activation: float = 30.0,
    hourly_bcm: float = 20.0,
    hourly_bem: float = 10.0,
    include_bcm_bem_same_hour_mwh: bool = False,
    max_charge_stack: float = 8.0,
    max_discharge_stack: float = 7.0,
    max_charge_violation: float = 0.0,
    max_discharge_violation: float = 0.0,
) -> Path:
    scen = root / strategy / scenario
    scen.mkdir(parents=True, exist_ok=True)
    component_gap_payload = global_component_gap or {
        "bcm_capacity_revenue": {
            "realized_eur": bcm,
            "global_pf_eur": bcm + 10.0,
            "global_pf_minus_realized_eur": 10.0,
        }
    }
    summary = {
        "model_key": "xgb",
        "trading_strategy": strategy,
        "scenario": scenario,
        "simulation_valid": 1.0,
        "thesis_reportable": 1.0,
        "invalid_reason": "",
        "realized_total_pnl_eur": realized,
        "rolling_perfect_foresight_same_rules_total_pnl_eur": rolling,
        "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur": global_pf,
        "global_perfect_foresight_available": global_available,
        "global_pf_verified_upper_bound": global_verified,
        "global_pf_same_market_rules": global_verified,
        "global_pf_realized_path_incumbent_eur": realized,
        "global_pf_solution_eur": global_pf,
        "global_pf_minus_realized_incumbent_eur": global_pf - realized,
        "global_pf_component_gap_json": json.dumps(component_gap_payload, sort_keys=True),
        "global_pf_failure_reason": global_pf_failure_reason,
        "activation_split_reconciliation_error_max": abs(hourly_activation - hourly_bcm - hourly_bem),
    }
    (scen / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "afrr_total_net_revenue_eur": afrr,
                "bcm_strategy_total_revenue_eur": bcm,
                "bem_net_revenue_eur": bem,
            }
        ]
    ).to_csv(scen / "performance_metrics.csv", index=False)
    hourly = {
        "real_revenue_activation_eur": [hourly_activation],
        "real_bcm_linked_activation_revenue_eur": [hourly_bcm],
        "real_bem_only_activation_revenue_eur": [hourly_bem],
        "real_power_stack_charge_mw": [max_charge_stack],
        "real_power_stack_discharge_mw": [max_discharge_stack],
        "real_power_violation_charge_mw": [max_charge_violation],
        "real_power_violation_discharge_mw": [max_discharge_violation],
    }
    if include_bcm_bem_same_hour_mwh:
        hourly.update(
            {
                "real_fixed_reserve_obligation_pos_mw": [5.0],
                "real_bcm_linked_pos_activation_mwh": [0.2],
                "real_bem_only_pos_activation_mwh": [0.1],
            }
        )
    pd.DataFrame(hourly).to_csv(scen / "backtest_hourly.csv", index=False)
    return scen


def _report(root: Path, *, require_pnl: bool = True, require_afrr: bool = True) -> pd.Series:
    df = validate_outputs._collect(root)
    report = validate_outputs._build_pnl_validation_report(
        df,
        require_pnl_hierarchy=require_pnl,
        require_afrr_decomposition=require_afrr,
    )
    assert len(report) == 1
    return report.iloc[0]


def test_pnl_hierarchy_valid_passes(tmp_path: Path) -> None:
    _write_scenario(tmp_path, realized=100.0, rolling=120.0, global_pf=150.0)
    row = _report(tmp_path)
    assert float(row["pnl_hierarchy_pass"]) == 1.0
    assert str(row["invalid_reason_added"]) == ""


def test_realized_exceeds_rolling_pf_fails(tmp_path: Path) -> None:
    _write_scenario(tmp_path, realized=121.0, rolling=120.0, global_pf=150.0)
    row = _report(tmp_path)
    assert float(row["pnl_hierarchy_pass"]) == 0.0
    assert "realized_exceeds_rolling_perfect_foresight" in str(row["invalid_reason_added"])


def test_rolling_pf_exceeds_global_pf_fails(tmp_path: Path) -> None:
    _write_scenario(tmp_path, realized=100.0, rolling=151.0, global_pf=150.0)
    row = _report(tmp_path)
    assert float(row["pnl_hierarchy_pass"]) == 0.0
    assert "rolling_pf_exceeds_global_perfect_foresight" in str(row["invalid_reason_added"])


def test_realized_exceeds_verified_global_pf_fails(tmp_path: Path) -> None:
    _write_scenario(tmp_path, realized=151.0, rolling=160.0, global_pf=150.0)
    row = _report(tmp_path)
    assert float(row["pnl_hierarchy_pass"]) == 0.0
    assert "realized_exceeds_global_perfect_foresight" in str(row["invalid_reason_added"])


def test_global_pf_unavailable_or_unverified_fails_when_required(tmp_path: Path) -> None:
    _write_scenario(tmp_path, global_available=0.0, global_verified=0.0)
    row = _report(tmp_path)
    assert float(row["pnl_hierarchy_pass"]) == 0.0
    assert "global_pf_unavailable" in str(row["invalid_reason_added"])


def test_afrr_decomposition_passes_when_afrr_equals_bcm_plus_bem(tmp_path: Path) -> None:
    _write_scenario(tmp_path, afrr=30.0, bcm=20.0, bem=10.0)
    row = _report(tmp_path)
    assert float(row["afrr_decomposition_pass"]) == 1.0


def test_afrr_decomposition_fails_when_mismatch_exceeds_tolerance(tmp_path: Path) -> None:
    _write_scenario(tmp_path, afrr=31.0, bcm=20.0, bem=10.0)
    row = _report(tmp_path)
    assert float(row["afrr_decomposition_pass"]) == 0.0
    assert "afrr_decomposition_mismatch" in str(row["invalid_reason_added"])


def test_hourly_activation_split_passes_when_total_equals_bcm_plus_bem(tmp_path: Path) -> None:
    _write_scenario(tmp_path, hourly_activation=30.0, hourly_bcm=20.0, hourly_bem=10.0)
    row = _report(tmp_path)
    assert float(row["activation_split_pass"]) == 1.0
    assert float(row["max_activation_split_error_eur"]) == 0.0


def test_hourly_activation_split_fails_when_mismatch_exceeds_tolerance(tmp_path: Path) -> None:
    _write_scenario(tmp_path, hourly_activation=31.0, hourly_bcm=20.0, hourly_bem=10.0)
    row = _report(tmp_path)
    assert float(row["activation_split_pass"]) == 0.0
    assert "activation_split_mismatch" in str(row["invalid_reason_added"])


def test_bcm_and_bem_same_hour_activation_split_is_source_revenue_based(tmp_path: Path) -> None:
    _write_scenario(
        tmp_path,
        hourly_activation=30.0,
        hourly_bcm=20.0,
        hourly_bem=10.0,
        include_bcm_bem_same_hour_mwh=True,
    )
    row = _report(tmp_path)
    assert float(row["activation_split_pass"]) == 1.0
    assert float(row["afrr_decomposition_pass"]) == 1.0


def test_validation_collect_reports_exact_summary_paths_and_power_stack(tmp_path: Path) -> None:
    first = _write_scenario(
        tmp_path,
        strategy="multi",
        scenario="p50_p50",
        max_charge_stack=10.0,
        max_discharge_stack=9.5,
        max_charge_violation=0.0,
        max_discharge_violation=0.0,
    )
    second = _write_scenario(
        tmp_path,
        strategy="bcm",
        scenario="p10_p10",
        realized=80.0,
        rolling=90.0,
        global_pf=100.0,
        max_charge_stack=8.0,
        max_discharge_stack=10.0,
        max_charge_violation=0.0,
        max_discharge_violation=0.25,
    )
    other_root = tmp_path.parent / f"{tmp_path.name}_other"
    _write_scenario(other_root, strategy="multi", scenario="p90_p90")

    df = validate_outputs._collect(tmp_path).sort_values("quantile_pair").reset_index(drop=True)

    assert set(df["summary_file_path"]) == {
        str(first / "backtest_summary.json"),
        str(second / "backtest_summary.json"),
    }
    assert set(df["quantile_pair"]) == {"p10_p10", "p50_p50"}
    assert set(df["strategy"]) == {"bcm", "multi"}
    assert float(df.loc[df["quantile_pair"].eq("p50_p50"), "max_real_power_stack_charge_mw"].iloc[0]) == 10.0
    assert float(df.loc[df["quantile_pair"].eq("p10_p10"), "max_real_power_violation_discharge_mw"].iloc[0]) == 0.25


def test_collect_missing_scenario_artifacts_from_strategy_overview(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "multi" / "p90_p90"
    scenario_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "scenario": "p90_p90",
                "trading_strategy": "multi",
                "output_dir": str(scenario_dir),
                "naive_simulation_valid": 1.0,
                "naive_invalid_reason": "none",
            }
        ]
    ).to_csv(tmp_path / "strategy_overview.csv", index=False)

    rows = validate_outputs._collect_missing_scenario_artifacts(tmp_path)

    assert len(rows) == 1
    assert rows[0]["scenario_path"] == str(scenario_dir)
    assert "scenario_artifacts_missing" in rows[0]["missing_fields"]
    assert "naive_hourly.parquet" in rows[0]["missing_fields"]


def test_collect_path_validity_mismatch_flags_naive_global_validity_leak(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "scenario": "p90_p90",
                "simulation_valid": 0.0,
                "active_path_simulation_valid": 0.0,
                "naive_simulation_valid": 1.0,
                "rhpf_simulation_valid": 0.0,
            }
        ]
    ).to_csv(tmp_path / "strategy_overview.csv", index=False)
    pd.DataFrame(
        [
            {"scenario": "p90_p90", "path_type": "model", "validity_flag": 0.0},
            {"scenario": "p90_p90", "path_type": "naive", "validity_flag": 0.0},
            {"scenario": "p90_p90", "path_type": "rhpf", "validity_flag": 0.0},
        ]
    ).to_parquet(tmp_path / "performance_paths_long.parquet", index=False)

    rows = validate_outputs._collect_path_validity_mismatches(tmp_path)

    assert len(rows) == 1
    assert rows[0]["path_type"] == "naive"
    assert rows[0]["expected_validity_flag"] == 1.0
    assert rows[0]["actual_validity_flag"] == "0.0"


def test_validation_collect_surfaces_global_pf_component_gap(tmp_path: Path) -> None:
    gap = {
        "bcm_capacity_revenue": {
            "realized_eur": 100.0,
            "global_pf_eur": 0.0,
            "global_pf_minus_realized_eur": -100.0,
        }
    }
    _write_scenario(
        tmp_path,
        realized=150.0,
        rolling=140.0,
        global_pf=120.0,
        global_verified=0.0,
        global_component_gap=gap,
        global_pf_failure_reason="below_realized_incumbent",
    )

    row = validate_outputs._collect(tmp_path).iloc[0]
    parsed = json.loads(str(row["global_pf_component_gap_json"]))

    assert str(row["global_pf_failure_reason"]) == "below_realized_incumbent"
    assert float(row["global_pf_minus_realized_incumbent_eur"]) == -30.0
    assert parsed["bcm_capacity_revenue"]["global_pf_minus_realized_eur"] == -100.0


def test_clean_power_stack_still_fails_pnl_report_when_global_pf_unverified(tmp_path: Path) -> None:
    _write_scenario(
        tmp_path,
        realized=100.0,
        rolling=120.0,
        global_pf=150.0,
        global_available=1.0,
        global_verified=0.0,
        max_charge_violation=0.0,
        max_discharge_violation=0.0,
    )

    row = _report(tmp_path, require_pnl=True, require_afrr=False)

    assert float(row["pnl_hierarchy_pass"]) == 0.0
    assert "global_pf_unverified" in str(row["invalid_reason_added"])
