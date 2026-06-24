from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_battery_backtest import (  # noqa: E402
    SIMULATION_EVAL_END_UTC,
    SIMULATION_EVAL_START_UTC,
    _apply_fallback_column_map,
    _forecast_coverage_report,
    _manifest_can_resolve_long_predictions,
    _matches_model_key,
    _normalize_model_choice,
    _plot_cumulative_pnl,
    _preflight_manifest_and_quantiles,
    _resolve_simulation_eval_window,
    _resolve_long_prediction_path,
    _resolve_model_manifest,
    _validate_bcm_capacity_truth_block_consistency,
    parse_args,
)
from energy_trading.simulation.battery_backtest import BacktestColumnMap  # noqa: E402
from energy_trading.visualization.style import THESIS_PALETTE, get_backtest_line_style  # noqa: E402


def _write_long_predictions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "target_time_utc": pd.date_range("2025-05-01", periods=4, freq="h", tz="UTC"),
            "lead_time_h": [1, 1, 1, 1],
            "p50": [1.0, 2.0, 3.0, 4.0],
            "predicted_value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    df.to_parquet(path, index=False)


def _write_manifest(path: Path, pred_rel: str, *, run_id: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": run_id,
        "bundles": {
            "da": {"predictions_long": {"val": {"pred_da_price": pred_rel}, "test": {"pred_da_price": pred_rel}}},
            "afrr": {"predictions_long": {"val": {}, "test": {}}},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_model_alias_normalization() -> None:
    assert _normalize_model_choice("xgb") == ("xgb", "latest_xgboost.json")
    assert _normalize_model_choice("xgboost") == ("xgb", "latest_xgboost.json")
    assert _normalize_model_choice("tft") == ("tft", "latest_tft.json")
    assert _normalize_model_choice("linear") == ("linear", "latest_linear.json")
    assert _normalize_model_choice("rlqr") == ("linear", "latest_linear.json")


def test_cli_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_battery_backtest.py"])
    args = parse_args()
    assert args.split == "test"
    assert args.trading_strategy == "multi"
    assert args.da_quantile_role == "mid"
    assert args.quantile_pairs == "p50-p50"
    assert args.strict_simulation_validity is True
    assert args.final_soc_mode == "hard_min"
    assert args.clean_output is True
    assert args.enable_global_perfect_foresight is True


def test_cli_can_disable_default_global_perfect_foresight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_battery_backtest.py", "--no-enable-global-perfect-foresight"])
    args = parse_args()
    assert args.enable_global_perfect_foresight is False


def test_common_eval_window_clamps_requested_start_and_end() -> None:
    info = _resolve_simulation_eval_window(
        requested_start="2025-01-04T00:00:00Z",
        requested_end="2026-03-01T00:00:00Z",
    )
    assert info["effective_start_utc"] == SIMULATION_EVAL_START_UTC.isoformat()
    assert info["effective_end_utc"] == SIMULATION_EVAL_END_UTC.isoformat()
    assert info["simulation_window_clamped"] == 1.0
    assert "start_below_common_lower_bound" in str(info["simulation_window_clamp_reason"])
    assert "end_above_common_upper_bound" in str(info["simulation_window_clamp_reason"])
    assert info["simulation_window_hours"] == 8760.0
    assert info["simulation_window_days"] == 365.0


def test_common_eval_window_defaults_to_common_bounds() -> None:
    info = _resolve_simulation_eval_window(requested_start=None, requested_end=None)
    assert info["requested_start_utc"] == ""
    assert info["requested_end_utc"] == ""
    assert info["effective_start_utc"] == SIMULATION_EVAL_START_UTC.isoformat()
    assert info["effective_end_utc"] == SIMULATION_EVAL_END_UTC.isoformat()
    assert info["simulation_window_clamped"] == 0.0
    assert info["simulation_window_hours"] == 8760.0
    assert info["simulation_window_days"] == 365.0


def test_common_eval_window_is_model_independent() -> None:
    # The helper intentionally has no model-key branch; TFT, XGB, linear/RLQR,
    # and future models use the same effective thesis window.
    bounds = {
        model: _resolve_simulation_eval_window(requested_start=None, requested_end=None)
        for model in ["xgb", "tft", "linear", "rlqr", "future_model"]
    }
    assert {v["effective_start_utc"] for v in bounds.values()} == {SIMULATION_EVAL_START_UTC.isoformat()}
    assert {v["effective_end_utc"] for v in bounds.values()} == {SIMULATION_EVAL_END_UTC.isoformat()}


def _minimal_truth_frame(*, include_unshifted_capacity_truth: bool = True) -> pd.DataFrame:
    data: dict[str, object] = {
        "timestamp_utc": pd.date_range("2025-07-01T06:00:00Z", periods=4, freq="h", tz="UTC"),
        "da_price": [1.0, 2.0, 3.0, 4.0],
        "afrr_activation_price_vwap_pos": [10.0, 10.0, 10.0, 10.0],
        "afrr_activation_price_vwap_neg": [11.0, 11.0, 11.0, 11.0],
        "activation_rate_phys_pos": [0.0, 0.0, 0.0, 0.0],
        "activation_rate_phys_neg": [0.0, 0.0, 0.0, 0.0],
        "target_afrr_capacity_price_pos": [50.0, 50.0, 50.0, 60.0],
        "target_afrr_capacity_price_neg": [70.0, 70.0, 70.0, 80.0],
    }
    if include_unshifted_capacity_truth:
        data["afrr_capacity_price_pos"] = [50.0, 50.0, 50.0, 50.0]
        data["afrr_capacity_price_neg"] = [70.0, 70.0, 70.0, 70.0]
    return pd.DataFrame(data)


def test_bcm_capacity_truth_mapping_rejects_shifted_target_fallback() -> None:
    truth = _minimal_truth_frame(include_unshifted_capacity_truth=False)

    with pytest.raises(KeyError, match="Missing unshifted BCM capacity settlement truth column"):
        _apply_fallback_column_map(pd.DataFrame(), truth, BacktestColumnMap())


def test_bcm_capacity_truth_mapping_rejects_explicit_shifted_target_column() -> None:
    truth = _minimal_truth_frame(include_unshifted_capacity_truth=True)
    colmap = BacktestColumnMap(
        true_afrr_capacity_price_pos="target_afrr_capacity_price_pos",
        true_afrr_capacity_price_neg="target_afrr_capacity_price_neg",
    )

    with pytest.raises(ValueError, match="shifted target columns are ML labels"):
        _apply_fallback_column_map(pd.DataFrame(), truth, colmap)


def test_bcm_capacity_truth_block_guard_accepts_unshifted_block_prices() -> None:
    truth = _minimal_truth_frame(include_unshifted_capacity_truth=True)

    issues = _validate_bcm_capacity_truth_block_consistency(
        df=truth,
        timestamp_col="timestamp_utc",
        pos_col="afrr_capacity_price_pos",
        neg_col="afrr_capacity_price_neg",
    )

    assert issues.empty


def test_bcm_capacity_truth_block_guard_rejects_shifted_target_pattern() -> None:
    truth = _minimal_truth_frame(include_unshifted_capacity_truth=True)

    with pytest.raises(ValueError, match="Invalid BCM capacity settlement truth column"):
        _validate_bcm_capacity_truth_block_consistency(
            df=truth,
            timestamp_col="timestamp_utc",
            pos_col="target_afrr_capacity_price_pos",
            neg_col="target_afrr_capacity_price_neg",
        )


def test_bcm_capacity_truth_block_guard_rejects_intra_block_variation() -> None:
    truth = _minimal_truth_frame(include_unshifted_capacity_truth=True)
    truth["capacity_truth_pos_bad"] = [50.0, 50.0, 50.0, 60.0]
    truth["capacity_truth_neg_bad"] = [70.0, 70.0, 70.0, 80.0]

    with pytest.raises(ValueError, match="not constant within local 4h product blocks"):
        _validate_bcm_capacity_truth_block_consistency(
            df=truth,
            timestamp_col="timestamp_utc",
            pos_col="capacity_truth_pos_bad",
            neg_col="capacity_truth_neg_bad",
        )


def test_forecast_coverage_ignores_missing_snapshots_before_clamped_window() -> None:
    start = SIMULATION_EVAL_START_UTC
    end = start + pd.Timedelta(hours=1)
    rows = []
    for snapshot in pd.date_range(start, end, freq="h", tz="UTC"):
        for lead in [1, 2]:
            rows.append(
                {
                    "snapshot_time_utc": snapshot,
                    "target_time_utc": snapshot + pd.Timedelta(hours=lead),
                    "lead_time_h": lead,
                    "predicted_value": 1.0,
                    "p50": 1.0,
                }
            )
    report, summary = _forecast_coverage_report(
        forecast_warehouse={"pred_da_price": pd.DataFrame(rows)},
        effective_start_utc=start,
        effective_end_utc=end,
        horizon_hours=2,
        expected_quantiles={"p50"},
    )
    assert summary["status"] == "ok"
    assert int(report["missing_snapshot_count"].sum()) == 0


def test_forecast_coverage_fails_missing_snapshot_inside_effective_window() -> None:
    start = SIMULATION_EVAL_START_UTC
    end = start + pd.Timedelta(hours=2)
    rows = [
        {
            "snapshot_time_utc": start,
            "target_time_utc": start + pd.Timedelta(hours=1),
            "lead_time_h": 1,
            "predicted_value": 1.0,
            "p50": 1.0,
        }
    ]
    _, summary = _forecast_coverage_report(
        forecast_warehouse={"pred_da_price": pd.DataFrame(rows)},
        effective_start_utc=start,
        effective_end_utc=end,
        horizon_hours=1,
        expected_quantiles={"p50"},
    )
    assert summary["status"] == "missing_coverage"
    assert summary["first_missing_snapshot"] == (start + pd.Timedelta(hours=1)).isoformat()


def test_cumulative_pnl_plot_includes_validated_global_perfect_foresight(tmp_path: Path) -> None:
    hourly = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-05-01T00:00:00Z", periods=3, freq="h", tz="UTC"),
            "real_pnl_eur": [1.0, 2.0, 3.0],
            "naive_pnl_eur": [0.5, 1.0, 1.5],
            "perfect_foresight_pnl_eur": [1.5, 2.5, 3.5],
            "global_perfect_foresight_pnl_eur": [2.0, 3.0, 4.0],
        }
    )
    out = tmp_path / "cum_pnl.png"
    _plot_cumulative_pnl(
        hourly,
        "timestamp_utc",
        out,
        summary={"global_perfect_foresight_available": 1.0},
    )
    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_cumulative_pnl_uses_thesis_benchmark_styles() -> None:
    assert get_backtest_line_style("naive")["color"] == THESIS_PALETTE["naive"]
    assert get_backtest_line_style("rolling_perfect_foresight")["color"] == THESIS_PALETTE["perfect_foresight"]
    assert get_backtest_line_style("rolling_perfect_foresight")["linestyle"] == "--"
    assert get_backtest_line_style("global_hindsight_perfect_foresight")["color"] == THESIS_PALETTE["perfect_foresight"]
    assert get_backtest_line_style("global_hindsight_perfect_foresight")["linestyle"] == "--"


def test_latest_pointer_resolution(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "model_runs"
    run_dir = root / "xgb_123"
    pred = run_dir / "predictions" / "xgboost_da_val_pred_da_price_long.parquet"
    _write_long_predictions(pred)
    _write_manifest(run_dir / "manifest.json", "predictions/xgboost_da_val_pred_da_price_long.parquet", run_id="xgb_123")
    latest = root / "latest_xgboost.json"
    latest.write_text(json.dumps({"manifest_path": "xgb_123/manifest.json", "run_id": "xgb_123"}), encoding="utf-8")

    resolved_path, payload, run_id = _resolve_model_manifest(
        run_manifest_arg="",
        run_id=None,
        model_key="xgb",
        split="val",
        model_runs_root=root,
    )
    assert resolved_path == run_dir / "manifest.json"
    assert run_id == "xgb_123"
    usable, issues = _manifest_can_resolve_long_predictions(payload, resolved_path.parent, "val", "xgb", {"p50"})
    assert usable, issues


def test_copied_latest_manifest_resolution_switches_to_actual_run(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "model_runs"
    run_dir = root / "xgb_123"
    pred = run_dir / "predictions" / "xgboost_da_val_pred_da_price_long.parquet"
    _write_long_predictions(pred)
    _write_manifest(run_dir / "manifest.json", "predictions/xgboost_da_val_pred_da_price_long.parquet", run_id="xgb_123")
    copied_latest = root / "latest_xgboost.json"
    copied_latest.write_text(
        json.dumps(
            {
                "run_id": "xgb_123",
                "bundles": {
                    "da": {"predictions_long": {"val": {"pred_da_price": "predictions/xgboost_da_val_pred_da_price_long.parquet"}}},
                    "afrr": {"predictions_long": {"val": {}}},
                },
            }
        ),
        encoding="utf-8",
    )

    resolved_path, _, _ = _resolve_model_manifest(
        run_manifest_arg=str(copied_latest),
        run_id=None,
        model_key="xgb",
        split="val",
        model_runs_root=root,
    )
    assert resolved_path == run_dir / "manifest.json"


def test_fallback_scan_chooses_newest_usable_manifest(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "model_runs"
    older = root / "xgb_older"
    newer = root / "xgb_newer"
    _write_long_predictions(older / "predictions" / "xgboost_da_val_pred_da_price_long.parquet")
    _write_manifest(older / "manifest.json", "predictions/xgboost_da_val_pred_da_price_long.parquet", run_id="xgb_older")
    _write_long_predictions(newer / "predictions" / "xgboost_da_val_pred_da_price_long.parquet")
    _write_manifest(newer / "manifest.json", "predictions/xgboost_da_val_pred_da_price_long.parquet", run_id="xgb_newer")
    os.utime(older / "manifest.json", (1, 1))
    os.utime(newer / "manifest.json", None)

    resolved_path, _, run_id = _resolve_model_manifest(
        run_manifest_arg="",
        run_id=None,
        model_key="xgb",
        split="val",
        model_runs_root=root,
    )
    assert resolved_path == newer / "manifest.json"
    assert run_id == "xgb_newer"


def test_strict_model_matching() -> None:
    assert _matches_model_key(Path("xgboost_da_val_pred_da_price_long.parquet"), "xgb")
    assert not _matches_model_key(Path("tft_da_val_pred_da_price_long.parquet"), "xgb")
    assert _matches_model_key(Path("tft_afrr_val_pred_afrr_activation_price_pos_long.parquet"), "tft")
    assert not _matches_model_key(Path("xgboost_afrr_val_pred_afrr_activation_price_pos_long.parquet"), "tft")
    assert _matches_model_key(Path("linear_da_val_pred_da_price_long.parquet"), "linear")
    assert _matches_model_key(Path("rlqr_da_val_pred_da_price_long.parquet"), "linear")
    assert not _matches_model_key(Path("tft_da_val_pred_da_price_long.parquet"), "linear")


def test_error_message_lists_manifest_context_and_candidates(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "model_runs"
    run_dir = root / "xgb_123"
    manifest_path = run_dir / "manifest.json"
    payload = _write_manifest(manifest_path, "predictions/xgboost_da_val_pred_da_price_long.parquet", run_id="xgb_123")
    with pytest.raises(FileNotFoundError) as exc:
        _preflight_manifest_and_quantiles(
            manifest_path=manifest_path,
            manifest_payload=payload,
            split="val",
            model_key="xgb",
            manifest_dir=run_dir,
            expected_quantiles={"p50"},
        )
    msg = str(exc.value)
    assert "resolved_manifest_path=" in msg
    assert "manifest_dir=" in msg
    assert "configured_path=" in msg
    assert "model_key=xgb" in msg
    assert "split=val" in msg
    assert "exact_candidates=" in msg


def test_end_to_end_manifest_preflight_smoke(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "model_runs"
    run_dir = root / "xgb_123"
    pred = run_dir / "predictions" / "xgboost_da_val_pred_da_price_long.parquet"
    _write_long_predictions(pred)
    payload = _write_manifest(run_dir / "manifest.json", "predictions/xgboost_da_val_pred_da_price_long.parquet", run_id="xgb_123")

    _preflight_manifest_and_quantiles(
        manifest_path=run_dir / "manifest.json",
        manifest_payload=payload,
        split="val",
        model_key="xgb",
        manifest_dir=run_dir,
        expected_quantiles={"p50"},
    )


def test_exact_manifest_path_bypasses_model_token_filter(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "xgb_123"
    pred_dir = manifest_dir / "predictions"
    pred = pred_dir / "xgboost_da_val_pred_da_price_long.parquet"
    _write_long_predictions(pred)
    resolved = _resolve_long_prediction_path(
        pred_col="pred_da_price",
        configured_path="predictions/xgboost_da_val_pred_da_price_long.parquet",
        manifest_dir=manifest_dir,
        split="val",
        model_key="tft",
    )
    assert resolved == pred


def test_model_tft_fallback_does_not_pick_xgb_file(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "xgb_123"
    pred_dir = manifest_dir / "predictions"
    _write_long_predictions(pred_dir / "xgboost_da_val_pred_da_price_long.parquet")
    with pytest.raises(FileNotFoundError):
        _resolve_long_prediction_path(
            pred_col="pred_da_price",
            configured_path="predictions/missing_tft_da_val_pred_da_price_long.parquet",
            manifest_dir=manifest_dir,
            split="val",
            model_key="tft",
        )
