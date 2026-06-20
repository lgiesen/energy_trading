from __future__ import annotations

import json
from pathlib import Path
import subprocess
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

from energy_trading.evaluation.forecast_truth_mapping import resolve_truth_mapping
from energy_trading.evaluation.forecast_metrics import (
    TailConfig,
    approx_crps,
    assign_gate_context,
    assign_horizon_bucket,
    empirical_coverage,
    interval_coverage,
    mean_pinball_loss,
    pinball_loss,
    quantile_crossing_metrics,
    repair_monotone_quantiles,
    tail_event_metrics,
)
from energy_trading.evaluation.forecast_benchmark import run_benchmark
from energy_trading.evaluation.forecast_figures import (
    example_lead_for_target,
    generate_forecast_benchmark_figures,
)
from energy_trading.visualization.style import THESIS_PALETTE, get_model_color


def test_truth_mapping_resolves_canonical_columns_correctly() -> None:
    r = resolve_truth_mapping(
        prediction_target_name="pred_da_price",
        available_truth_columns=["timestamp_utc", "da_price"],
        truth_source_path="truth.parquet",
    )
    assert r.status == "ok"
    assert r.truth_column == "da_price"


def test_truth_mapping_fails_on_ambiguous_missing_truth() -> None:
    with pytest.raises(ValueError):
        resolve_truth_mapping(
            prediction_target_name="pred_afrr_capacity_price_pos",
            available_truth_columns=["afrr_capacity_price_pos", "target_afrr_capacity_price_pos"],
            truth_source_path="truth.parquet",
        )
    with pytest.raises(ValueError):
        resolve_truth_mapping(
            prediction_target_name="pred_afrr_capacity_price_pos",
            available_truth_columns=["foo"],
            truth_source_path="truth.parquet",
        )


def test_pinball_loss_known_values() -> None:
    y = np.array([0.0, 1.0])
    yq = np.array([0.0, 0.0])
    assert np.isclose(pinball_loss(y, yq, 0.5), 0.25)


def test_approx_crps_monotonic_sanity() -> None:
    y = np.array([1.0, 2.0, 3.0])
    good = {0.1: np.array([0.9, 1.9, 2.9]), 0.5: np.array([1.0, 2.0, 3.0]), 0.9: np.array([1.1, 2.1, 3.1])}
    bad = {0.1: np.array([0.0, 0.0, 0.0]), 0.5: np.array([0.0, 0.0, 0.0]), 0.9: np.array([0.0, 0.0, 0.0])}
    assert approx_crps(y, good) < approx_crps(y, bad)


def test_approx_crps_falls_back_when_numpy_trapezoid_is_unavailable(monkeypatch) -> None:
    y = np.array([1.0, 2.0, 3.0])
    preds = {0.1: np.array([0.9, 1.9, 2.9]), 0.5: np.array([1.0, 2.0, 3.0]), 0.9: np.array([1.1, 2.1, 3.1])}
    monkeypatch.delattr(np, "trapezoid", raising=False)
    assert np.isfinite(approx_crps(y, preds))


def test_quantile_coverage_known_values() -> None:
    y = np.array([1, 2, 3], dtype=float)
    yq = np.array([2, 2, 2], dtype=float)
    assert np.isclose(empirical_coverage(y, yq), 2 / 3)


def test_crossing_metrics_detect_crossing_before_repair() -> None:
    qp = {0.1: np.array([2.0]), 0.5: np.array([1.0])}
    rate, maxv = quantile_crossing_metrics(qp)
    assert rate > 0
    assert maxv > 0


def test_quantile_repair_makes_rows_monotone() -> None:
    qp = {0.1: np.array([2.0, 0.0]), 0.5: np.array([1.0, 1.0]), 0.9: np.array([3.0, 2.0])}
    repaired = repair_monotone_quantiles(qp)
    assert np.all(repaired[0.1] <= repaired[0.5])
    assert np.all(repaired[0.5] <= repaired[0.9])


def test_tail_event_metrics_known_small_fixture() -> None:
    y = np.array([0.0, 1.0, 10.0, 12.0])
    yhat = np.array([0.0, 1.0, 8.0, 9.0])
    m = tail_event_metrics(y=y, yhat_p50=yhat, cfg=TailConfig())
    assert "tail_mae" in m
    assert np.isfinite(m["tail_mae"])


def test_horizon_bucket_assignment_works() -> None:
    b = {"short": [1, 8], "medium": [9, 16], "long": [17, 48]}
    assert assign_horizon_bucket(4, b) == "short"
    assert assign_horizon_bucket(12, b) == "medium"


def test_gate_time_context_assignment_works() -> None:
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    c = assign_gate_context(ts, 2)
    assert "bem_short_horizon" in c


def test_global_ranking_uses_normalized_rank_not_raw_mae_average(tmp_path: Path) -> None:
    d = tmp_path / "bench"
    (d / "metrics").mkdir(parents=True)
    (d / "diagnostics").mkdir(parents=True)
    (d / "benchmark_manifest.json").write_text(json.dumps({"config_sha256":"x","quantiles":[0.1],"package_versions":{},"save_joined_predictions":False}))
    (d / "input_manifest.json").write_text("{}")
    (d / "benchmark_config_resolved.yaml").write_text("x: 1")
    pd.DataFrame([{"status":"ok"}]).to_csv(d / "diagnostics" / "truth_mapping_report.csv", index=False)
    pd.DataFrame([{"join_coverage":1.0}]).to_csv(d / "diagnostics" / "join_coverage_report.csv", index=False)
    pd.DataFrame([{"a":1}]).to_csv(d / "diagnostics" / "benchmark_input_inventory.csv", index=False)
    (d / "diagnostics" / "schema_report.json").write_text("[]")
    for f in [
        "metrics_overall.csv","metrics_by_target.csv","metrics_by_model.csv","metrics_by_lead.csv",
        "metrics_by_horizon_bucket.csv","metrics_gate_time.csv","metrics_tail_events.csv",
        "metrics_calibration.csv","metrics_model_ranking_by_target.csv","metrics_residual_patterns.csv",
        "metrics_volatility_regimes.csv","metrics_directional_bias.csv",
    ]:
        pd.DataFrame([{"model":"m","target":"t","split":"test","mae_p50":1.0,"mean_pinball":1.0,"approx_crps":1.0}]).to_csv(d / "metrics" / f, index=False)
    pd.DataFrame([{"model":"m","target":"t","split":"test","crossing_rate_before_repair":0.0,"max_crossing_violation_before_repair":0.0}]).to_csv(
        d / "metrics" / "metrics_crossing.csv", index=False
    )
    pd.DataFrame([{"model":"m","avg_rank_across_targets":1.0}]).to_csv(d / "metrics" / "metrics_model_ranking_global_normalized.csv", index=False)
    cp = subprocess.run([
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "scripts" / "validate_forecast_benchmark.py"),
        "--benchmark-dir", str(d),
        "--min-join-coverage", "0.9",
    ], capture_output=True, text=True)
    assert cp.returncode == 0, cp.stdout + cp.stderr


def test_output_manifest_includes_hashes_and_config(tmp_path: Path) -> None:
    truth = tmp_path / "truth.parquet"
    pred = tmp_path / "pred.parquet"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    df_truth = pd.DataFrame({"timestamp_utc": pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"), "da_price": [1,2,3,4]})
    df_truth.to_parquet(truth, index=False)
    df_pred = pd.DataFrame({
        "target_time_utc": pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
        "predicted_value": [1,2,3,4],
        "lead_time_h": [1,1,1,1],
        "p10": [0,1,2,3], "p30": [1,1.5,2.5,3.5], "p50": [1,2,3,4], "p70": [1.2,2.2,3.2,4.2], "p90":[1.5,2.5,3.5,4.5],
    })
    df_pred.to_parquet(pred, index=False)
    manifest = {
        "training": {"model_type": "xgboost"},
        "bundles": {"da": {"predictions_long": {"test": {"pred_da_price": f"../{pred.name}"}}}},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    cfg = {"quantiles":[0.1,0.3,0.5,0.7,0.9],"horizon":{"buckets":{"short":[1,8],"medium":[9,16],"long":[17,48]}},"tail_events":{}}
    out = tmp_path / "out"
    run_benchmark(
        config=cfg,
        model_run_manifests=[run_dir / "manifest.json"],
        out_dir=out,
        splits=["test"],
        truth_source=truth,
        min_join_coverage=0.9,
        fail_on_missing_truth=True,
        make_figures=False,
        save_joined_predictions=True,
        overwrite=True,
    )
    bm = json.loads((out / "benchmark_manifest.json").read_text())
    assert "config_sha256" in bm
    inp = json.loads((out / "input_manifest.json").read_text())
    assert "truth_source_sha256" in inp


def test_join_fails_if_coverage_below_threshold(tmp_path: Path) -> None:
    truth = tmp_path / "truth.parquet"
    pred = tmp_path / "pred.parquet"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame({"timestamp_utc": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"), "da_price": [1,2]}).to_parquet(truth, index=False)
    pd.DataFrame({
        "target_time_utc": pd.date_range("2025-02-01", periods=4, freq="h", tz="UTC"),
        "predicted_value": [1,2,3,4],
        "lead_time_h": [1,1,1,1],
        "p10":[0,0,0,0],"p30":[0,0,0,0],"p50":[1,2,3,4],"p70":[1,2,3,4],"p90":[1,2,3,4],
    }).to_parquet(pred, index=False)
    (run_dir / "manifest.json").write_text(json.dumps({
        "training": {"model_type": "xgb"},
        "bundles": {"da": {"predictions_long": {"test": {"pred_da_price": f"../{pred.name}"}}}},
    }))
    with pytest.raises(ValueError):
        run_benchmark(
            config={"quantiles":[0.1,0.3,0.5,0.7,0.9],"horizon":{"buckets":{"short":[1,8],"medium":[9,16],"long":[17,48]}},"tail_events":{}},
            model_run_manifests=[run_dir / "manifest.json"],
            out_dir=tmp_path / "out",
            splits=["test"],
            truth_source=truth,
            min_join_coverage=0.999,
            fail_on_missing_truth=True,
            make_figures=False,
            save_joined_predictions=True,
            overwrite=True,
        )


def test_benchmark_applies_postprocessing_to_predictions_and_truth(tmp_path: Path) -> None:
    truth = tmp_path / "truth.parquet"
    pred = tmp_path / "pred.parquet"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ts = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    pd.DataFrame(
        {
            "timestamp_utc": ts,
            "afrr_activation_price_vwap_neg": [-100.0, -200.0, -300.0, -400.0],
        }
    ).to_parquet(truth, index=False)
    pd.DataFrame(
        {
            "target_time_utc": ts,
            "lead_time_h": [1, 1, 1, 1],
            "predicted_value": [-80.0, -150.0, -260.0, -350.0],
            "p10": [-110.0, -210.0, -310.0, -410.0],
            "p30": [-100.0, -200.0, -300.0, -400.0],
            "p50": [-80.0, -150.0, -260.0, -350.0],
            "p70": [-60.0, -120.0, -220.0, -300.0],
            "p90": [-40.0, -100.0, -180.0, -260.0],
        }
    ).to_parquet(pred, index=False)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "training": {"model_type": "xgb"},
                "bundles": {
                    "afrr": {
                        "predictions_long": {
                            "test": {"pred_afrr_activation_price_neg": f"../{pred.name}"}
                        }
                    }
                },
            }
        )
    )
    out = tmp_path / "out"
    run_benchmark(
        config={"quantiles": [0.1, 0.3, 0.5, 0.7, 0.9], "horizon": {"buckets": {"short": [1, 8], "medium": [9, 16], "long": [17, 48]}}, "tail_events": {}},
        model_run_manifests=[run_dir / "manifest.json"],
        out_dir=out,
        splits=["test"],
        truth_source=truth,
        min_join_coverage=0.9,
        fail_on_missing_truth=True,
        make_figures=False,
        save_joined_predictions=True,
        overwrite=True,
    )
    joined = pd.read_parquet(out / "diagnostics" / "joined_predictions" / "xgb__test__pred_afrr_activation_price_neg.parquet")
    assert (joined["y_true"] >= 0.0).all()
    assert (joined["predicted_value"] >= 0.0).all()


def test_benchmark_writes_postprocessing_report(tmp_path: Path) -> None:
    out = _run_small_benchmark(tmp_path, make_figures=False)
    p = out / "diagnostics" / "forecast_postprocessing_report.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert {"model", "split", "target", "sign_flipped", "status", "target_value_mode"}.issubset(df.columns)


def test_validation_script_detects_missing_required_files(tmp_path: Path) -> None:
    cp = subprocess.run([
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "scripts" / "validate_forecast_benchmark.py"),
        "--benchmark-dir", str(tmp_path / "missing"),
        "--min-join-coverage", "0.9",
    ], capture_output=True, text=True)
    assert cp.returncode != 0


def _run_small_benchmark(tmp_path: Path, make_figures: bool) -> Path:
    truth = tmp_path / "truth.parquet"
    pred_xgb = tmp_path / "pred_xgb.parquet"
    pred_rlqr = tmp_path / "pred_rlqr.parquet"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ts = pd.date_range("2025-01-01", periods=24 * 10, freq="h", tz="UTC")
    truth_df = pd.DataFrame({"timestamp_utc": ts, "da_price": np.sin(np.arange(len(ts)) / 8.0) * 10 + 50})
    truth_df.to_parquet(truth, index=False)
    base = truth_df["da_price"].to_numpy(dtype=float)
    for path, shift in [(pred_xgb, 0.5), (pred_rlqr, -0.3)]:
        df_pred = pd.DataFrame({
            "target_time_utc": ts,
            "lead_time_h": np.tile(np.arange(1, 25), len(ts) // 24),
            "predicted_value": base + shift,
            "p10": base - 3 + shift,
            "p30": base - 1 + shift,
            "p50": base + shift,
            "p70": base + 1 + shift,
            "p90": base + 3 + shift,
        })
        df_pred.to_parquet(path, index=False)
    (run_dir / "manifest_xgb.json").write_text(json.dumps({
        "training": {"model_type": "xgboost"},
        "bundles": {"da": {"predictions_long": {"test": {"pred_da_price": f"../{pred_xgb.name}"}}}},
    }))
    (run_dir / "manifest_rlqr.json").write_text(json.dumps({
        "training": {"model_type": "linear"},
        "bundles": {"da": {"predictions_long": {"test": {"pred_da_price": f"../{pred_rlqr.name}"}}}},
    }))
    out = tmp_path / "out"
    cfg = {
        "quantiles": [0.1, 0.3, 0.5, 0.7, 0.9],
        "horizon": {"buckets": {"short": [1, 8], "medium": [9, 16], "long": [17, 48]}},
        "tail_events": {},
        "figures": {"enabled": make_figures, "dpi": 150, "make": {
            "leadtime_metrics": True,
            "calibration": True,
            "coverage_width": True,
            "forecast_bands": True,
            "tail_scatter": True,
            "residual_diagnostics": True,
            "volatility_diagnostics": True,
        }, "example_windows": {"window_days": 7, "min_coverage": 0.8}},
    }
    run_benchmark(
        config=cfg,
        model_run_manifests=[run_dir / "manifest_xgb.json", run_dir / "manifest_rlqr.json"],
        out_dir=out,
        splits=["test"],
        truth_source=truth,
        min_join_coverage=0.9,
        fail_on_missing_truth=True,
        make_figures=make_figures,
        save_joined_predictions=True,
        overwrite=True,
    )
    return out


def test_make_figures_creates_leadtime_metric_figures(tmp_path: Path) -> None:
    out = _run_small_benchmark(tmp_path, make_figures=True)
    assert (out / "figures" / "test" / "pred_da_price" / "leadtime_mae_p50.png").exists()
    assert (out / "figures" / "test" / "pred_da_price" / "leadtime_mean_pinball.png").exists()
    assert (out / "figures" / "test" / "pred_da_price" / "leadtime_approx_crps.png").exists()


def test_make_figures_creates_calibration_curve(tmp_path: Path) -> None:
    out = _run_small_benchmark(tmp_path, make_figures=True)
    assert (out / "figures" / "test" / "pred_da_price" / "calibration_curve.png").exists()


def test_make_figures_creates_forecast_band_examples(tmp_path: Path) -> None:
    out = _run_small_benchmark(tmp_path, make_figures=True)
    assert list((out / "figures" / "test" / "pred_da_price").glob("*/typical_week_forecast_band.png"))
    assert (out / "figures" / "test" / "pred_da_price" / "typical_week_p50_model_comparison.png").exists()


def test_thesis_palette_model_mapping_is_fixed() -> None:
    assert get_model_color("truth") == THESIS_PALETTE["perfect_foresight"]
    assert get_model_color("linear") == THESIS_PALETTE["secondary"]
    assert get_model_color("xgb") == THESIS_PALETTE["primary"]
    assert get_model_color("tft") == THESIS_PALETTE["tertiary"]


def test_example_lead_mapping_matches_expected_targets() -> None:
    assert example_lead_for_target("pred_da_price") == 24
    assert example_lead_for_target("pred_afrr_capacity_price_pos") == 24
    assert example_lead_for_target("pred_afrr_capacity_price_neg") == 24
    assert example_lead_for_target("pred_afrr_activation_price_pos") == 1
    assert example_lead_for_target("pred_afrr_activation_price_neg") == 1
    assert example_lead_for_target("pred_afrr_activation_rate_pos") == 1
    assert example_lead_for_target("pred_afrr_activation_rate_neg") == 1


def test_example_forecast_band_uses_target_specific_lead(tmp_path: Path) -> None:
    ts = pd.date_range("2025-01-01", periods=24 * 8, freq="h", tz="UTC")
    joined = pd.concat(
        [
            pd.DataFrame(
                {
                    "model": "xgb",
                    "split": "test",
                    "target": "pred_da_price",
                    "target_time_utc": ts,
                    "lead_time_h": 24,
                    "y_true": np.linspace(1.0, 2.0, len(ts)),
                    "p10": np.linspace(0.0, 1.0, len(ts)),
                    "p30": np.linspace(0.5, 1.5, len(ts)),
                    "p50": np.linspace(1.0, 2.0, len(ts)),
                    "p70": np.linspace(1.5, 2.5, len(ts)),
                    "p90": np.linspace(2.0, 3.0, len(ts)),
                }
            ),
            pd.DataFrame(
                {
                    "model": "xgb",
                    "split": "test",
                    "target": "pred_da_price",
                    "target_time_utc": ts,
                    "lead_time_h": 1,
                    "y_true": np.linspace(100.0, 101.0, len(ts)),
                    "p10": np.linspace(99.0, 100.0, len(ts)),
                    "p30": np.linspace(99.5, 100.5, len(ts)),
                    "p50": np.linspace(100.0, 101.0, len(ts)),
                    "p70": np.linspace(100.5, 101.5, len(ts)),
                    "p90": np.linspace(101.0, 102.0, len(ts)),
                }
            ),
            pd.DataFrame(
                {
                    "model": "xgb",
                    "split": "test",
                    "target": "pred_afrr_activation_price_pos",
                    "target_time_utc": ts,
                    "lead_time_h": 1,
                    "y_true": np.linspace(10.0, 11.0, len(ts)),
                    "p10": np.linspace(9.0, 10.0, len(ts)),
                    "p30": np.linspace(9.5, 10.5, len(ts)),
                    "p50": np.linspace(10.0, 11.0, len(ts)),
                    "p70": np.linspace(10.5, 11.5, len(ts)),
                    "p90": np.linspace(11.0, 12.0, len(ts)),
                }
            ),
            pd.DataFrame(
                {
                    "model": "xgb",
                    "split": "test",
                    "target": "pred_afrr_activation_price_pos",
                    "target_time_utc": ts,
                    "lead_time_h": 24,
                    "y_true": np.linspace(200.0, 201.0, len(ts)),
                    "p10": np.linspace(199.0, 200.0, len(ts)),
                    "p30": np.linspace(199.5, 200.5, len(ts)),
                    "p50": np.linspace(200.0, 201.0, len(ts)),
                    "p70": np.linspace(200.5, 201.5, len(ts)),
                    "p90": np.linspace(201.0, 202.0, len(ts)),
                }
            ),
        ],
        ignore_index=True,
    )
    by_lead = pd.DataFrame(
        [
            {"model": "xgb", "split": "test", "target": "pred_da_price", "lead_time_h": 24, "mae_p50": 1.0, "mean_pinball": 1.0, "approx_crps": 1.0, "coverage_p10_p90": 0.8, "interval_width_p10_p90": 2.0},
            {"model": "xgb", "split": "test", "target": "pred_afrr_activation_price_pos", "lead_time_h": 1, "mae_p50": 1.0, "mean_pinball": 1.0, "approx_crps": 1.0, "coverage_p10_p90": 0.8, "interval_width_p10_p90": 2.0},
        ]
    )
    calibration = pd.DataFrame(
        [
            {"model": "xgb", "split": "test", "target": "pred_da_price", "quantile": 0.1, "empirical_coverage": 0.1},
            {"model": "xgb", "split": "test", "target": "pred_afrr_activation_price_pos", "quantile": 0.1, "empirical_coverage": 0.1},
        ]
    )
    out = tmp_path / "out"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    res = generate_forecast_benchmark_figures(
        joined_df=joined,
        by_lead=by_lead,
        calibration=calibration,
        figures_dir=out / "figures",
        diagnostics_dir=out / "diagnostics",
        config={"figures": {"enabled": True, "dpi": 150, "make": {"forecast_bands": True}, "example_windows": {"window_days": 7, "min_coverage": 0.8}}},
    )
    report = res.example_window_report
    assert set(report["selected_lead_h"].dropna().astype(int)) == {1, 24}
    da_rows = report.loc[report["target"] == "pred_da_price"]
    act_rows = report.loc[report["target"] == "pred_afrr_activation_price_pos"]
    assert set(da_rows["selected_lead_h"].dropna().astype(int)) == {24}
    assert set(act_rows["selected_lead_h"].dropna().astype(int)) == {1}
    assert (out / "figures" / "test" / "pred_da_price" / "typical_week_p50_model_comparison.png").exists()
    assert (out / "figures" / "test" / "pred_afrr_activation_price_pos" / "typical_week_p50_model_comparison.png").exists()


def test_make_figures_creates_tail_scatter(tmp_path: Path) -> None:
    out = _run_small_benchmark(tmp_path, make_figures=True)
    assert list((out / "figures" / "test" / "pred_da_price").glob("*/tail_event_scatter.png"))


def test_make_figures_creates_residual_diagnostics(tmp_path: Path) -> None:
    out = _run_small_benchmark(tmp_path, make_figures=True)
    assert list((out / "figures" / "test" / "pred_da_price").glob("*/residual_by_hour_of_day.png"))
    assert (out / "metrics" / "metrics_residual_patterns.csv").exists()


def test_make_figures_creates_volatility_diagnostics(tmp_path: Path) -> None:
    out = _run_small_benchmark(tmp_path, make_figures=True)
    assert (out / "figures" / "test" / "pred_da_price" / "error_by_volatility_bucket.png").exists()
    assert (out / "metrics" / "metrics_volatility_regimes.csv").exists()


def test_figures_are_split_aware(tmp_path: Path) -> None:
    out = _run_small_benchmark(tmp_path, make_figures=True)
    assert (out / "figures" / "test").exists()


def test_joined_predictions_are_saved(tmp_path: Path) -> None:
    out = _run_small_benchmark(tmp_path, make_figures=False)
    files = list((out / "diagnostics" / "joined_predictions").glob("*.parquet"))
    assert files


def test_example_window_report_written(tmp_path: Path) -> None:
    out = _run_small_benchmark(tmp_path, make_figures=True)
    p = out / "diagnostics" / "example_window_report.csv"
    assert p.exists()
    assert p.stat().st_size > 0


def test_validator_fails_when_required_figure_missing(tmp_path: Path) -> None:
    out = _run_small_benchmark(tmp_path, make_figures=True)
    missing = out / "figures" / "test" / "pred_da_price" / "calibration_curve.png"
    missing.unlink()
    cp = subprocess.run([
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "scripts" / "validate_forecast_benchmark.py"),
        "--benchmark-dir", str(out),
        "--require-figures",
    ], capture_output=True, text=True)
    assert cp.returncode != 0


def test_validator_passes_with_required_figures(tmp_path: Path) -> None:
    out = _run_small_benchmark(tmp_path, make_figures=True)
    cp = subprocess.run([
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "scripts" / "validate_forecast_benchmark.py"),
        "--benchmark-dir", str(out),
        "--require-figures",
    ], capture_output=True, text=True)
    assert cp.returncode == 0, cp.stdout + cp.stderr


def test_no_make_figures_skips_figure_validation_unless_required(tmp_path: Path) -> None:
    out = _run_small_benchmark(tmp_path, make_figures=False)
    cp_ok = subprocess.run([
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "scripts" / "validate_forecast_benchmark.py"),
        "--benchmark-dir", str(out),
        "--no-require-figures",
    ], capture_output=True, text=True)
    assert cp_ok.returncode == 0, cp_ok.stdout + cp_ok.stderr
    cp_fail = subprocess.run([
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "scripts" / "validate_forecast_benchmark.py"),
        "--benchmark-dir", str(out),
        "--require-figures",
    ], capture_output=True, text=True)
    assert cp_fail.returncode != 0
