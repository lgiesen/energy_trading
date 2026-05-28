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
    (d / "benchmark_manifest.json").write_text(json.dumps({"config_sha256":"x","quantiles":[0.1],"package_versions":{}}))
    (d / "input_manifest.json").write_text("{}")
    (d / "benchmark_config_resolved.yaml").write_text("x: 1")
    pd.DataFrame([{"status":"ok"}]).to_csv(d / "diagnostics" / "truth_mapping_report.csv", index=False)
    pd.DataFrame([{"join_coverage":1.0}]).to_csv(d / "diagnostics" / "join_coverage_report.csv", index=False)
    pd.DataFrame([{"a":1}]).to_csv(d / "diagnostics" / "benchmark_input_inventory.csv", index=False)
    (d / "diagnostics" / "schema_report.json").write_text("[]")
    for f in [
        "metrics_overall.csv","metrics_by_target.csv","metrics_by_model.csv","metrics_by_lead.csv",
        "metrics_by_horizon_bucket.csv","metrics_gate_time.csv","metrics_tail_events.csv",
        "metrics_calibration.csv","metrics_model_ranking_by_target.csv",
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
            overwrite=True,
        )


def test_validation_script_detects_missing_required_files(tmp_path: Path) -> None:
    cp = subprocess.run([
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "scripts" / "validate_forecast_benchmark.py"),
        "--benchmark-dir", str(tmp_path / "missing"),
        "--min-join-coverage", "0.9",
    ], capture_output=True, text=True)
    assert cp.returncode != 0
