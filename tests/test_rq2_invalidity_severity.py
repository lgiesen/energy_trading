from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_rq2_invalidity_severity import build_outputs
from scripts.summarize_rq2_invalidity_severity import build_outputs as build_limitation_outputs


def _make_scenario(root: Path) -> Path:
    scenario = root / "tft_p90" / "multi" / "p90_p90"
    scenario.mkdir(parents=True)
    (scenario / "backtest_summary.json").write_text(
        json.dumps(
            {
                "simulation_valid": 0.0,
                "thesis_reportable": 0.0,
                "invalid_reason": "fallback_used,missed_activation",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z", "2025-01-01T02:00:00Z"],
                utc=True,
            ),
            "optimizer_fallback_used": [1.0, 0.0, 1.0],
            "optimization_error_code": ["infeasible", "ok", "ok"],
            "real_missed_activation_mwh": [0.5, 0.0, 0.0],
            "real_protected_soc_violation_pos_mwh": [0.0, 0.2, 0.0],
            "real_protected_soc_violation_neg_mwh": [0.0, 0.0, 0.0],
        }
    ).to_csv(scenario / "backtest_hourly.csv", index=False)
    pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T02:00:00Z"], utc=True),
            "da_hourly_lock_infeasible_buy_mwh": [0.1, 0.0],
            "da_hourly_lock_infeasible_sell_mwh": [0.0, 0.0],
        }
    ).to_csv(scenario / "da_precommit_debug.csv", index=False)
    pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"], utc=True),
            "real_act_pos_mwh": [1.0, 0.0],
            "real_act_neg_mwh": [0.0, 0.0],
            "real_missed_activation_mwh": [0.5, 0.0],
        }
    ).to_csv(scenario / "realized_ledger.csv", index=False)
    pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2025-01-01T01:00:00Z"], utc=True),
            "headroom_violation_mwh": [0.3],
        }
    ).to_csv(scenario / "reserve_commitment_debug.csv", index=False)
    pd.DataFrame(
        {
            "da_bid_abs_mwh_total": [0.0],
            "bem_bid_abs_mwh_total": [0.0],
            "id_abs_mwh_total": [0.0],
            "simulation_valid": [0.0],
            "thesis_reportable": [0.0],
            "invalid_reason": ["fallback_used,missed_activation"],
        }
    ).to_csv(scenario / "performance_metrics.csv", index=False)
    return scenario


def test_invalidity_severity_outputs_and_zero_denominator_warning(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "benchmark"
    _make_scenario(run_dir)

    paths = build_outputs(run_dir, out_dir)
    summary = pd.read_csv(paths["summary"])
    hourly = pd.read_csv(paths["hourly"])
    warnings = pd.read_csv(paths["warnings"])

    row = summary.iloc[0]
    assert row["model_label"] == "TFT"
    assert row["quantile"] == "p90"
    assert row["total_hours"] == 3
    assert row["combined_infeasibility_hours"] == 3
    assert row["combined_infeasibility_hours_share"] == 1.0
    assert row["fallback_optimization_count"] == 2
    assert row["missed_activation_mwh"] == 0.5
    assert row["sum_soc_violation_mwh"] == 0.2
    assert row["sum_reserve_headroom_shortfall_mwh_or_mw_hours"] == 0.3
    assert np.isnan(row["da_lockbook_infeasible_mwh_share_of_total_planned_trade"])
    assert "zero_denominator" in set(warnings["warning"])
    assert not hourly.duplicated(["scenario", "timestamp_utc"]).any()

    limitation_paths = build_limitation_outputs(out_dir)
    compact = pd.read_csv(limitation_paths["compact"])
    tex = limitation_paths["latex"].read_text(encoding="utf-8")
    txt = limitation_paths["summary"].read_text(encoding="utf-8")

    assert compact["DA infeasible MWh / trade MWh (%)"].iloc[0] == "--"
    assert "\\toprule" in tex
    assert "\\bottomrule" in tex
    assert re.search(r"(^|[^A-Za-z])[-+]?inf([^A-Za-z]|$)", tex, flags=re.IGNORECASE) is None
    assert "diagnostic backtest evidence" in txt
    assert "fully valid trading-performance results" in txt
