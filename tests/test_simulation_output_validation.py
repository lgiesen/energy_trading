from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import validate_simulation_outputs as validate_outputs


def _write_scenario(
    root: Path,
    *,
    realized: float = 100.0,
    rolling: float = 120.0,
    global_pf: float = 150.0,
    global_available: float = 1.0,
    global_verified: float = 1.0,
    afrr: float = 30.0,
    bcm: float = 20.0,
    bem: float = 10.0,
    hourly_activation: float = 30.0,
    hourly_bcm: float = 20.0,
    hourly_bem: float = 10.0,
    include_bcm_bem_same_hour_mwh: bool = False,
) -> Path:
    scen = root / "multi" / "p50_p50"
    scen.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_key": "xgb",
        "trading_strategy": "multi",
        "simulation_valid": 1.0,
        "thesis_reportable": 1.0,
        "invalid_reason": "",
        "realized_total_pnl_eur": realized,
        "comparable_rolling_perfect_foresight_same_rules_market_pnl_eur": rolling,
        "global_hindsight_perfect_foresight_upper_bound_total_pnl_eur": global_pf,
        "global_perfect_foresight_available": global_available,
        "global_pf_verified_upper_bound": global_verified,
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
