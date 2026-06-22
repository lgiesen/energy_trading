from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_final_calibration_uncertainty import (
    ModelSpec,
    _crossing_metrics,
    _interval_metrics,
    _quantile_coverage,
    aggregate_activation_reliability,
    build_calibration_outputs,
    build_main_summary_table,
    build_summary,
    plot_reliability_activation_aggregates,
    plot_reliability_by_target_group,
    write_latex_summary,
    write_outputs,
)


MODELS = [ModelSpec("tft", "TFT"), ModelSpec("xgb", "XGB"), ModelSpec("linear", "RLQR")]


def _coverage_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_time_utc": pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
            "lead_time_h": [1, 1, 1, 1],
            "y_true": [1.0, 2.0, 3.0, 4.0],
            "p10": [0.0, 2.5, 3.5, 5.0],
            "p50": [1.0, 2.0, 2.0, 5.0],
            "p90": [2.0, 3.0, 4.0, 5.0],
        }
    )


def _write_joined(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _rows(model: str, *, include_extra: bool = True, with_p50: bool = True, with_optional: bool = False) -> list[dict]:
    rows = [
        {
            "model": model,
            "split": "test",
            "target": "pred_da_price",
            "target_time_utc": "2025-01-01T00:00:00Z",
            "lead_time_h": 1,
            "y_true": 1.0,
            "p10": 0.0,
            "p50": 1.0,
            "p90": 2.0,
        },
        {
            "model": model,
            "split": "test",
            "target": "pred_da_price",
            "target_time_utc": "2025-01-01T01:00:00Z",
            "lead_time_h": 1,
            "y_true": 2.0,
            "p10": 1.0,
            "p50": 2.0,
            "p90": 3.0,
        },
    ]
    if include_extra:
        rows.append(
            {
                "model": model,
                "split": "test",
                "target": "pred_da_price",
                "target_time_utc": "2025-01-01T02:00:00Z",
                "lead_time_h": 1,
                "y_true": 3.0,
                "p10": 2.0,
                "p50": 3.0,
                "p90": 4.0,
            }
        )
    if not with_p50:
        for row in rows:
            row.pop("p50", None)
    if with_optional:
        for row in rows:
            row["p05"] = row["p10"] - 0.5
            row["p95"] = row["p90"] + 0.5
            row["p01"] = row["p10"] - 1.0
            row["p99"] = row["p90"] + 1.0
    return rows


def _small_benchmark(tmp_path: Path, *, tft_extra: bool = False, with_p50: bool = True, with_optional: bool = False) -> Path:
    joined = tmp_path / "benchmark" / "diagnostics" / "joined_predictions"
    _write_joined(joined / "tft__test__pred_da_price.parquet", _rows("tft", include_extra=tft_extra, with_p50=with_p50, with_optional=with_optional))
    _write_joined(joined / "xgb__test__pred_da_price.parquet", _rows("xgb", include_extra=True, with_optional=with_optional))
    _write_joined(joined / "linear__test__pred_da_price.parquet", _rows("linear", include_extra=True, with_optional=with_optional))
    return tmp_path / "benchmark"


def test_quantile_coverage_and_calibration_error_exact() -> None:
    df = _coverage_frame()
    cov = _quantile_coverage(
        df,
        model=ModelSpec("xgb", "XGB"),
        split="test",
        target="pred_da_price",
        qcols={0.1: "p10", 0.5: "p50", 0.9: "p90"},
    )
    p10 = cov.loc[np.isclose(cov["quantile"], 0.1)].iloc[0]
    p50 = cov.loc[np.isclose(cov["quantile"], 0.5)].iloc[0]
    assert p10["empirical_coverage"] == 0.75
    assert p10["calibration_error"] == pytest.approx(0.65)
    assert p10["abs_calibration_error"] == pytest.approx(0.65)
    assert p50["empirical_coverage"] == 0.75
    assert p50["calibration_error"] == pytest.approx(0.25)


def test_interval_coverage_width_and_score_exact() -> None:
    df = _coverage_frame()
    intervals = _interval_metrics(
        df,
        model=ModelSpec("xgb", "XGB"),
        split="test",
        target="pred_da_price",
        qcols={0.1: "p10", 0.5: "p50", 0.9: "p90"},
    )
    row = intervals.loc[intervals["interval"] == "p10-p90"].iloc[0]
    assert row["nominal_interval_coverage"] == pytest.approx(0.8)
    assert row["interval_coverage"] == 0.25
    assert row["interval_coverage_error"] == pytest.approx(-0.55)
    assert row["interval_width_mean"] == pytest.approx(0.75)
    assert row["interval_width_median"] == pytest.approx(0.5)
    assert row["interval_score"] == pytest.approx(5.75)


def test_mean_abs_calibration_error_summary() -> None:
    df = _coverage_frame()
    cov = _quantile_coverage(
        df,
        model=ModelSpec("xgb", "XGB"),
        split="test",
        target="pred_da_price",
        qcols={0.1: "p10", 0.5: "p50", 0.9: "p90"},
    )
    intervals = _interval_metrics(
        df,
        model=ModelSpec("xgb", "XGB"),
        split="test",
        target="pred_da_price",
        qcols={0.1: "p10", 0.5: "p50", 0.9: "p90"},
    )
    crossing = pd.DataFrame(
        [_crossing_metrics(df, model=ModelSpec("xgb", "XGB"), split="test", target="pred_da_price", qcols={0.1: "p10", 0.5: "p50", 0.9: "p90"})]
    )
    summary = build_summary(cov, intervals, crossing)
    expected = np.mean([0.65, 0.25, 0.10])
    assert summary["mean_abs_calibration_error"].iloc[0] == pytest.approx(expected)
    assert summary["p50_bias"].iloc[0] == pytest.approx(0.0)


def test_main_summary_table_is_compact_and_excludes_point_error_metrics() -> None:
    df = _coverage_frame()
    frames = []
    intervals = []
    crossings = []
    for model in MODELS:
        frames.append(
            _quantile_coverage(
                df,
                model=model,
                split="test",
                target="pred_da_price",
                qcols={0.1: "p10", 0.5: "p50", 0.9: "p90"},
            )
        )
        intervals.append(
            _interval_metrics(
                df,
                model=model,
                split="test",
                target="pred_da_price",
                qcols={0.1: "p10", 0.5: "p50", 0.9: "p90"},
            )
        )
        crossings.append(
            _crossing_metrics(
                df,
                model=model,
                split="test",
                target="pred_da_price",
                qcols={0.1: "p10", 0.5: "p50", 0.9: "p90"},
            )
        )
    summary = build_summary(pd.concat(frames, ignore_index=True), pd.concat(intervals, ignore_index=True), pd.DataFrame(crossings))
    main = build_main_summary_table(summary, split="test")
    assert list(main.columns) == [
        "target",
        "target_label",
        "TFT_MACE",
        "XGB_MACE",
        "RLQR_MACE",
        "best_calibrated",
        "p10_p90_coverage",
        "main_issue",
    ]
    assert "mae_p50" not in main.columns
    assert "rmse_p50" not in main.columns
    assert "mean_pinball_loss" not in main.columns


def test_quantile_crossing_rate_and_magnitude() -> None:
    df = pd.DataFrame(
        {
            "y_true": [1.0, 2.0, 3.0],
            "p10": [1.0, 3.0, 1.0],
            "p50": [2.0, 2.0, 2.0],
            "p90": [3.0, 4.0, 1.5],
        }
    )
    out = _crossing_metrics(
        df,
        model=ModelSpec("tft", "TFT"),
        split="test",
        target="pred_da_price",
        qcols={0.1: "p10", 0.5: "p50", 0.9: "p90"},
    )
    assert out["crossing_any"] is True
    assert out["crossing_rate"] == pytest.approx(2 / 3)
    assert out["num_crossings"] == 2
    assert out["mean_crossing_magnitude"] == pytest.approx(0.75)
    assert out["max_crossing_magnitude"] == pytest.approx(1.0)


def test_common_row_intersection_and_optional_quantile_warning(tmp_path: Path) -> None:
    benchmark = _small_benchmark(tmp_path, tft_extra=False, with_optional=False)
    outputs = build_calibration_outputs(
        benchmark_dir=benchmark,
        models=MODELS,
        splits=["test"],
        eval_origin_start=None,
        eval_origin_end=None,
    )
    row_counts = outputs["row_counts"]
    assert set(row_counts["retained_common_rows"]) == {2}
    assert row_counts.loc[row_counts["model"] == "xgb", "dropped_rows"].iloc[0] == 1
    assert not outputs["warnings"].empty
    assert outputs["warnings"]["message"].str.contains("Optional interval").any()


def test_missing_required_p50_fails(tmp_path: Path) -> None:
    benchmark = _small_benchmark(tmp_path, with_p50=False)
    with pytest.raises(ValueError, match="Missing required p50"):
        build_calibration_outputs(
            benchmark_dir=benchmark,
            models=MODELS,
            splits=["test"],
            eval_origin_start=None,
            eval_origin_end=None,
        )


def test_latex_and_figure_outputs_use_style(tmp_path: Path) -> None:
    benchmark = _small_benchmark(tmp_path, with_optional=True)
    outputs = build_calibration_outputs(
        benchmark_dir=benchmark,
        models=MODELS,
        splits=["test"],
        eval_origin_start=None,
        eval_origin_end=None,
    )
    written = write_outputs(
        outputs,
        out_dir=tmp_path / "calibration",
        split="test",
        legacy_flat_out_dir=tmp_path / "legacy_calibration",
    )
    names = {p.name for p in written}
    assert "rq1_4_1_2_calibration_summary_test.tex" in names
    assert "rq1_4_1_2_calibration_quantile_coverage_test_appendix.tex" in names
    assert "rq1_4_1_2_calibration_interval_quality_test_appendix.tex" in names
    assert "rq1_4_1_2_calibration_quantile_crossing_test_appendix.tex" in names
    assert "calibration_interval_coverage_by_target_group.png" in names
    assert "rq1_4_1_2_calibration_summary_test.csv" in names
    tex = (tmp_path / "calibration" / "latex" / "rq1_4_1_2_calibration_summary_test.tex").read_text(encoding="utf-8")
    assert "\\toprule" in tex
    assert "\\midrule" in tex
    assert "\\bottomrule" in tex
    assert "\\textbf{\\shortstack{" in tex
    assert "Target" in tex
    assert "TFT" in tex and "MACE" in tex
    assert "XGB" in tex
    assert "RLQR" in tex
    assert "Best" in tex and "calibrated" in tex
    assert "p10-p90" in tex and "coverage" in tex
    assert (
        "\\caption{MACE for each target variable and p10-p90 coverage for best calibrated model.}"
        in tex
    )
    assert "Mean MACE" in tex
    assert "\\textbf{" in tex and " pp" in tex
    assert "\\%" not in tex
    assert "0.8223" not in tex
    assert "Compact calibration summary" not in tex
    assert "MAE" not in tex
    assert "RMSE" not in tex
    assert "pinball" not in tex.lower()
    assert "pred_da_price" not in tex
    assert "\\label{tab:calibration_summary_test}" in tex
    assert (tmp_path / "legacy_calibration" / "calibration_quantile_coverage_test.csv").exists()
    assert (tmp_path / "legacy_calibration" / "latex" / "calibration_summary_test.tex").exists()


def test_reliability_figure_can_be_created_from_synthetic_metrics(tmp_path: Path) -> None:
    df = _coverage_frame()
    cov = _quantile_coverage(
        df,
        model=ModelSpec("linear", "RLQR"),
        split="test",
        target="pred_da_price",
        qcols={0.1: "p10", 0.5: "p50", 0.9: "p90"},
    )
    out = plot_reliability_by_target_group(cov, tmp_path)
    assert out is not None
    assert out.exists()
    source = Path("scripts/build_final_calibration_uncertainty.py").read_text(encoding="utf-8")
    assert "apply_geo_style" in source


def test_activation_reliability_aggregate_merges_pos_neg(tmp_path: Path) -> None:
    coverage = pd.DataFrame(
        [
            {
                "target": "pred_afrr_activation_price_pos",
                "target_label": "aFRR activation price +",
                "target_group": "aFRR activation price",
                "model": "xgb",
                "model_label": "XGB",
                "quantile": 0.1,
                "empirical_coverage": 0.2,
                "n_obs": 2,
            },
            {
                "target": "pred_afrr_activation_price_neg",
                "target_label": "aFRR activation price -",
                "target_group": "aFRR activation price",
                "model": "xgb",
                "model_label": "XGB",
                "quantile": 0.1,
                "empirical_coverage": 0.8,
                "n_obs": 6,
            },
            {
                "target": "pred_afrr_activation_rate_pos",
                "target_label": "aFRR activation rate +",
                "target_group": "aFRR activation rate",
                "model": "linear",
                "model_label": "RLQR",
                "quantile": 0.9,
                "empirical_coverage": 0.75,
                "n_obs": 4,
            },
            {
                "target": "pred_afrr_activation_rate_neg",
                "target_label": "aFRR activation rate -",
                "target_group": "aFRR activation rate",
                "model": "linear",
                "model_label": "RLQR",
                "quantile": 0.9,
                "empirical_coverage": 1.0,
                "n_obs": 4,
            },
        ]
    )
    agg = aggregate_activation_reliability(coverage)
    price = agg.loc[(agg["target"].eq("afrr_activation_price")) & agg["model"].eq("xgb")].iloc[0]
    assert price["empirical_coverage"] == pytest.approx((0.2 * 2 + 0.8 * 6) / 8)
    rate = agg.loc[(agg["target"].eq("afrr_activation_rate")) & agg["model"].eq("linear")].iloc[0]
    assert rate["empirical_coverage"] == pytest.approx(0.875)
    out = plot_reliability_activation_aggregates(coverage, tmp_path)
    assert out is not None
    assert out.name == "rq1_4_1_2_calibration_reliability_activation_aggregates.png"
    assert out.exists()
