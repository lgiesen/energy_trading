from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_simulation_invalidity_severity import build_outputs, parse_bool, parse_scenario_folder


def _make_scenario(root: Path, folder: str = "xgb_p30") -> Path:
    scenario = root / folder / "multi" / "p30_p30"
    scenario.mkdir(parents=True)
    (scenario / "backtest_summary.json").write_text(
        json.dumps(
            {
                "simulation_valid": 0.0,
                "thesis_reportable": 0.0,
                "invalid_reason": "fallback_used,missed_activation,protected_soc",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "timestamp_utc": [
                "2025-01-01T00:00:00Z",
                "2025-01-01T01:00:00Z",
                "2025-01-01T02:00:00Z",
            ],
            "optimizer_fallback_used": [1, 0, 0],
            "optimization_error_code": ["infeasible", "ok", "ok"],
            "real_missed_activation_mwh": [0.4, 0.0, 0.0],
            "real_act_pos_mwh": [0.0, 0.0, 0.0],
            "real_act_neg_mwh": [0.0, 0.0, 0.0],
            "planned_soc_mwh": [-99.0, -99.0, -99.0],
            "projected_soc_mwh": [-99.0, -99.0, -99.0],
            "real_soc_mwh": [-0.2, 0.5, 1.3],
            "physical_soc_min_mwh": [0.0, 0.0, 0.0],
            "physical_soc_max_mwh": [1.0, 1.0, 1.0],
            "reserve_headroom_shortfall_mw": [0.7, 0.0, 0.0],
        }
    ).to_csv(scenario / "backtest_hourly.csv", index=False)
    pd.DataFrame(
        {
            "timestamp_utc": ["2025-01-01T00:00:00Z"],
            "da_hourly_lock_infeasible_buy_mwh": [0.2],
            "da_hourly_lock_infeasible_sell_mwh": [0.0],
        }
    ).to_csv(scenario / "da_precommit_debug.csv", index=False)
    pd.DataFrame(
        {
            "da_bid_abs_mwh_total": [0.0],
            "bem_bid_abs_mwh_total": [0.0],
            "id_abs_mwh_total": [0.0],
            "bcm_bid_abs_mwh_total": [0.0],
        }
    ).to_csv(scenario / "performance_metrics.csv", index=False)
    return scenario


def test_parse_scenario_folder_variants(tmp_path: Path) -> None:
    cases = {
        "xgb_p30": ("xgb", "XGB", "p30", False),
        "linear_p70": ("linear", "RLQR", "p70", False),
        "tft_p90": ("tft", "TFT", "p90", False),
        "benchmarks_naive": ("naive", "Naive", "benchmark", True),
        "benchmarks_rhpf": ("rhpf", "RHPF", "benchmark", True),
        "xgb_p 30": ("xgb", "XGB", "p30", False),
    }
    for folder, expected in cases.items():
        path = tmp_path / folder
        got = parse_scenario_folder(path)
        assert got[:4] == expected
        if " " in folder:
            assert got[4]


def test_parse_bool_values() -> None:
    assert parse_bool("False") is False
    assert parse_bool("false") is False
    assert parse_bool("0") is False
    assert parse_bool(0) is False
    assert parse_bool("no") is False
    assert parse_bool("True") is True
    assert parse_bool("1") is True
    assert parse_bool(1) is True
    assert parse_bool("yes") is True
    assert parse_bool("") is None
    assert parse_bool(None) is None


def test_invalidity_severity_direct_extraction(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    out_root = tmp_path / "out"
    _make_scenario(run_root)

    paths = build_outputs(run_root, out_root, label="rq2")
    summary = pd.read_csv(paths["summary"])
    hourly = pd.read_csv(paths["hourly"])
    warnings = pd.read_csv(paths["warnings"])
    metric_sources = pd.read_csv(paths["metric_sources"])

    row = summary.iloc[0]
    assert row["scenario"] == "xgb_p30"
    assert row["model_label"] == "XGB"
    assert row["quantile"] == "p30"
    assert row["total_hours"] == 3
    # Fallback, optimizer infeasible, missed activation, DA infeasible and reserve
    # all occur in the same first hour, so this counts as one combined hour.
    assert row["combined_infeasibility_hours"] == 2
    assert np.isclose(row["combined_infeasibility_hours_share"], 2 / 3)
    assert row["da_lockbook_infeasible_hours"] == 1
    assert row["fallback_optimization_count"] == 1
    assert row["missed_activation_count"] == 1
    assert np.isnan(row["missed_activation_mwh_share"])
    assert np.isnan(row["da_lockbook_infeasible_mwh_share_of_total_planned_trade"])
    assert row["max_soc_violation_mwh"] == 0.3
    assert row["sum_soc_violation_mwh"] == 0.5
    assert row["max_reserve_headroom_shortfall_mw"] == 0.7
    assert row["total_planned_trade_mwh_source"] == "planned"
    assert row["total_activation_mwh_source"] == "hourly"
    assert row["total_optimization_count_source"] == "hourly_fallback_flags"
    assert row["denominator_completeness_status"] == "complete"
    assert row["activation_count_semantics"] == "hourly_event_count"
    assert row["missed_activation_count_semantics"] == "hourly_event_count"
    assert row["coverage_share"] == 1.0
    assert row["missing_hours_count"] == 0.0
    assert row["invalidity_severity_class"] == "material_invalidity"
    assert "zero_denominator" in set(warnings["warning"])
    assert not hourly.duplicated(["scenario", "timestamp_utc"]).any()
    first_hour = hourly.sort_values("timestamp_utc").iloc[0]
    assert first_hour["combined_infeasibility_flag"] == 1
    assert {"total_hours", "da_lockbook_infeasible_mwh", "soc_violation_mwh"}.issubset(set(metric_sources["metric_name"]))
    soc_source = metric_sources.loc[metric_sources["metric_name"].eq("soc_violation_mwh"), "source_column"].iloc[0]
    assert "real_soc_mwh" in soc_source
    assert "planned_soc_mwh" not in soc_source
    assert "projected_soc_mwh" not in soc_source

    tex = paths["latex"].read_text(encoding="utf-8")
    txt = paths["limitation_summary"].read_text(encoding="utf-8")
    assert "\\toprule" in tex
    assert "Severity class" in tex
    assert "diagnostic backtest evidence" in txt
    assert "fully validated physically feasible" in txt
