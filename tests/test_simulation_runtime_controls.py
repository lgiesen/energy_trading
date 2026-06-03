from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from energy_trading.simulation.battery_backtest import BacktestColumnMap, BatteryBacktester
from scripts.run_battery_backtest import _input_cache_key, _select_hourly_output_columns
from scripts.summarize_simulation_runtime import collect_runtime


def test_output_detail_thesis_keeps_required_validation_columns() -> None:
    hourly = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-01-01", periods=1, tz="UTC"),
            "real_soc_mwh": [10.0],
            "real_pnl_eur": [1.0],
            "real_revenue_da_eur": [2.0],
            "real_cost_da_eur": [1.0],
            "real_revenue_capacity_eur": [3.0],
            "real_bcm_linked_activation_revenue_eur": [4.0],
            "real_bem_only_activation_revenue_eur": [5.0],
            "real_revenue_activation_eur": [9.0],
            "real_id_buy_mwh": [0.0],
            "real_cost_id_eur": [0.0],
            "real_id_recourse_reason": ["none"],
            "optimization_error_code": ["ok"],
            "optimizer_fallback_used": [0.0],
            "ev_bcm_p10_coef_debug": [999.0],
        }
    )
    out = _select_hourly_output_columns(hourly, output_detail="thesis", timestamp_col="timestamp_utc")
    for col in [
        "timestamp_utc",
        "real_soc_mwh",
        "real_pnl_eur",
        "real_revenue_da_eur",
        "real_revenue_capacity_eur",
        "real_bcm_linked_activation_revenue_eur",
        "real_bem_only_activation_revenue_eur",
        "real_id_recourse_reason",
        "optimization_error_code",
    ]:
        assert col in out.columns
    assert "ev_bcm_p10_coef_debug" not in out.columns


def test_output_detail_debug_preserves_debug_columns() -> None:
    hourly = pd.DataFrame({"timestamp_utc": [pd.Timestamp("2025-01-01", tz="UTC")], "ev_debug_col": [1.0]})
    out = _select_hourly_output_columns(hourly, output_detail="debug", timestamp_col="timestamp_utc")
    assert list(out.columns) == list(hourly.columns)


def test_debug_dumps_accepted_only_suppresses_candidate_dump_but_counts_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    bt = BatteryBacktester()
    bt.debug_dump_mode = "accepted_only"
    df = pd.DataFrame({"timestamp_utc": [pd.Timestamp("2025-01-01", tz="UTC")]})
    bt._write_infeasible_debug_dump(
        df=df,
        colmap=BacktestColumnMap(),
        c=np.array([1.0]),
        a_ub=None,
        b_ub=None,
        a_eq=None,
        b_eq=None,
        lb=np.array([0.0]),
        ub=np.array([1.0]),
        message="infeasible",
        solve_context={"timestamp_utc": "2025-01-01T00:00:00+00:00", "final_accepted_path": False},
    )
    assert len(bt._infeasible_debug_dumps) == 1
    assert bt._infeasible_debug_dumps[0]["path"] == ""
    assert not list(tmp_path.rglob("*.npz"))


def test_input_cache_key_changes_when_truth_mtime_changes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    pred = tmp_path / "pred.parquet"
    truth = tmp_path / "truth.parquet"
    for p in (manifest, pred, truth):
        p.write_text("x", encoding="utf-8")
    key1 = _input_cache_key(
        model_key="xgb",
        split="test",
        manifest_path=manifest,
        prediction_files=[pred],
        truth_file=truth,
        forecast_value_mode="canonical_economic",
    )
    time.sleep(0.001)
    truth.write_text("changed", encoding="utf-8")
    key2 = _input_cache_key(
        model_key="xgb",
        split="test",
        manifest_path=manifest,
        prediction_files=[pred],
        truth_file=truth,
        forecast_value_mode="canonical_economic",
    )
    assert key1 != key2


def test_runtime_summary_parser_reads_tiny_synthetic_run(tmp_path: Path) -> None:
    scen = tmp_path / "multi" / "p50_p50"
    scen.mkdir(parents=True)
    (scen / "backtest_summary.json").write_text(
        json.dumps(
            {
                "model_key": "xgb",
                "trading_strategy": "multi",
                "optimized_hours_only_rows": 24,
                "backtester_run_seconds": 12.0,
                "optimizer_total_seconds": 8.0,
                "optimizer_mean_seconds_per_step": 0.5,
                "settlement_predicted_seconds": 1.0,
                "settlement_realized_seconds": 2.0,
                "output_write_seconds": 0.25,
                "infeasible_debug_dump_count": 0,
                "simulation_valid": 1,
                "thesis_reportable": 1,
                "invalid_reason": "",
            }
        ),
        encoding="utf-8",
    )
    df = collect_runtime(tmp_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["model"] == "xgb"
    assert row["strategy"] == "multi"
    assert row["quantile"] == "p50-p50"
    assert float(row["settlement_seconds"]) == 3.0
    assert float(row["seconds_per_simulated_hour"]) == 0.5
