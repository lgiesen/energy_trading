from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from scripts.build_final_full_forecast_metrics import (
    ModelSpec,
    _latex_table,
    build_p50_error_tolerance_outputs,
    build_detailed_table,
    build_full_metrics,
    build_primary_table,
    write_outputs,
)


def _write_joined(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _rows(model: str, *, include_third: bool = True, p90_third: float | None = 35.0, with_p30: bool = True) -> list[dict]:
    base = [
        {
            "model": model,
            "split": "test",
            "target": "pred_da_price",
            "target_time_utc": "2025-01-01T00:00:00Z",
            "lead_time_h": 1,
            "y_true": 10.0,
            "p10": 8.0,
            "p30": 9.0,
            "p50": 11.0 if model == "linear" else (9.0 if model == "xgb" else 10.0),
            "p90": 12.0,
        },
        {
            "model": model,
            "split": "test",
            "target": "pred_da_price",
            "target_time_utc": "2025-01-01T01:00:00Z",
            "lead_time_h": 2,
            "y_true": 20.0,
            "p10": 18.0,
            "p30": 19.0,
            "p50": 18.0 if model == "linear" else (19.0 if model == "xgb" else 21.0),
            "p90": 22.0,
        },
    ]
    if include_third:
        base.append(
            {
                "model": model,
                "split": "test",
                "target": "pred_da_price",
                "target_time_utc": "2025-01-01T02:00:00Z",
                "lead_time_h": 3,
                "y_true": 30.0,
                "p10": 28.0,
                "p30": 29.0,
                "p50": 33.0,
                "p90": p90_third,
            }
        )
    if not with_p30:
        for row in base:
            row.pop("p30", None)
    return base


def _build_small_metrics(tmp_path: Path, *, tft_with_p30: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = tmp_path / "benchmark" / "diagnostics" / "joined_predictions"
    _write_joined(joined / "linear__test__pred_da_price.parquet", _rows("linear", include_third=True))
    _write_joined(joined / "xgb__test__pred_da_price.parquet", _rows("xgb", include_third=True, p90_third=None))
    _write_joined(joined / "tft__test__pred_da_price.parquet", _rows("tft", include_third=False, with_p30=tft_with_p30))
    return build_full_metrics(
        benchmark_dir=tmp_path / "benchmark",
        models=[ModelSpec("tft", "TFT"), ModelSpec("xgb", "XGB"), ModelSpec("linear", "RLQR")],
        splits=["test"],
    )


def test_common_row_intersection_and_alignment_diagnostics(tmp_path: Path) -> None:
    metrics, diagnostics = _build_small_metrics(tmp_path)

    assert set(metrics["model"]) == {"tft", "xgb", "linear"}
    assert set(metrics["n_valid_rows"]) == {2}
    assert set(metrics["quantiles_used"]) == {"p10,p30,p50,p90"}
    required = {
        "split",
        "target",
        "model",
        "original_valid_rows",
        "common_intersection_rows",
        "dropped_rows",
        "quantiles_available",
        "quantiles_used",
        "lead_min",
        "lead_max",
    }
    assert required.issubset(diagnostics.columns)
    assert diagnostics.loc[diagnostics["model"] == "linear", "dropped_rows"].iloc[0] == 1
    assert diagnostics.loc[diagnostics["model"] == "xgb", "original_valid_rows"].iloc[0] == 2


def test_common_quantile_grid_is_enforced(tmp_path: Path) -> None:
    metrics, _ = _build_small_metrics(tmp_path, tft_with_p30=False)
    assert set(metrics["quantiles_used"]) == {"p10,p50,p90"}


def test_pinball_mae_rmse_bias_are_unweighted_on_common_rows(tmp_path: Path) -> None:
    metrics, _ = _build_small_metrics(tmp_path)
    linear = metrics[(metrics["model"] == "linear") & (metrics["metric"] == "mean_pinball_loss")].iloc[0]
    mae = metrics[(metrics["model"] == "linear") & (metrics["metric"] == "mae_p50")].iloc[0]
    rmse = metrics[(metrics["model"] == "linear") & (metrics["metric"] == "rmse_p50")].iloc[0]
    bias = metrics[(metrics["model"] == "linear") & (metrics["metric"] == "bias_p50")].iloc[0]

    expected_pinball = (0.2 + 0.3 + 0.75 + 0.2) / 4.0
    assert math.isclose(float(linear["value"]), expected_pinball, rel_tol=1e-9)
    assert math.isclose(float(mae["value"]), 1.5, rel_tol=1e-9)
    assert math.isclose(float(rmse["value"]), math.sqrt(2.5), rel_tol=1e-9)
    assert math.isclose(float(bias["value"]), -0.5, rel_tol=1e-9)


def test_primary_table_is_mean_pinball_only(tmp_path: Path) -> None:
    metrics, _ = _build_small_metrics(tmp_path)
    primary = build_primary_table(metrics, split="test")
    assert len(primary) == 1
    assert "metric" not in primary.columns
    assert "target_group" not in primary.columns
    assert list(primary.columns) == [
        "target",
        "RLQR",
        "XGB",
        "TFT",
        "best_model",
        "relative_improvement_vs_RLQR_pct",
        "n_obs",
        "quantiles_used",
        "lead_min",
        "lead_max",
    ]
    assert primary["best_model"].iloc[0] in {"TFT", "XGB", "RLQR"}


def test_primary_table_keeps_activation_rate_pinball_precision(tmp_path: Path) -> None:
    table = pd.DataFrame(
        [
            {
                "target": "pred_afrr_activation_rate_pos",
                "RLQR": 0.00261,
                "XGB": 0.00225,
                "TFT": 0.00226,
                "best_model": "XGB",
                "relative_improvement_vs_RLQR_pct": 13.65,
            },
            {
                "target": "pred_da_price",
                "RLQR": 14.591,
                "XGB": 8.824,
                "TFT": 10.565,
                "best_model": "XGB",
                "relative_improvement_vs_RLQR_pct": 39.53,
            },
        ]
    )
    path = tmp_path / "primary.tex"
    _latex_table(
        table,
        columns=["target", "RLQR", "XGB", "TFT", "best_model", "relative_improvement_vs_RLQR_pct"],
        headers=["Target", "RLQR", "XGB", "TFT", "Best model", r"\shortstack{Improvement\\vs RLQR (\%)}"],
        caption="Model Mean Pinball Loss",
        label="tab:forecast_metrics_full_primary",
        path=path,
        bold_best_model_values=True,
        value_decimals=2,
        activation_rate_value_decimals=5,
    )

    tex = path.read_text(encoding="utf-8")
    assert r"aFRR activation rate + & 0.00261 & \textbf{0.00225} & 0.00226" in tex
    assert r"DA price & 14.59 & \textbf{8.82} & 10.56" in tex


def test_detailed_table_contains_all_four_metrics(tmp_path: Path) -> None:
    metrics, _ = _build_small_metrics(tmp_path)
    detailed = build_detailed_table(metrics, split="test")
    assert set(detailed["metric"]) == {"mean_pinball_loss", "mae_p50", "rmse_p50", "bias_p50"}
    bias_row = detailed.loc[detailed["metric"] == "bias_p50"].iloc[0]
    assert bias_row["best_model"] == ""
    assert pd.isna(bias_row["relative_improvement_vs_RLQR_pct"])


def test_outputs_exclude_interval_metrics_from_4_1_1(tmp_path: Path) -> None:
    metrics, diagnostics = _build_small_metrics(tmp_path)
    outputs = write_outputs(metrics, diagnostics, benchmark_dir=tmp_path / "benchmark", out_dir=tmp_path / "out", split="test")
    names = {p.name for p in outputs}
    assert "rq1_4_1_1_forecast_metrics_full_long.csv" in names
    assert "rq1_4_1_1_forecast_metrics_full_primary_test.csv" in names
    assert "rq1_4_1_1_forecast_metrics_full_detailed_test.csv" in names
    assert "rq1_4_1_1_forecast_metrics_full_alignment_diagnostics_test.csv" in names
    assert "rq1_4_1_1_forecast_metrics_full_primary_test.tex" in names
    assert "rq1_4_1_1_forecast_metrics_full_detailed_test.tex" in names
    assert "rq1_4_1_1_forecast_metrics_full_relative_pinball_test.png" in names

    primary_tex = (tmp_path / "out" / "latex" / "rq1_4_1_1_forecast_metrics_full_primary_test.tex").read_text(encoding="utf-8")
    detailed_tex = (tmp_path / "out" / "latex" / "rq1_4_1_1_forecast_metrics_full_detailed_test.tex").read_text(encoding="utf-8")
    assert r"\textbf{Target} & \textbf{RLQR} & \textbf{XGB} & \textbf{TFT} & \textbf{Best model}" in primary_tex
    assert r"\shortstack{Improvement\\vs RLQR (\%)}" in primary_tex
    assert r"\textbf{Target group}" not in primary_tex
    assert r"\textbf{Quantiles}" not in primary_tex
    assert r"\textbf{Metric}" not in primary_tex
    assert "mean pinball" in primary_tex.lower()
    assert "mae" not in primary_tex.lower()
    assert "rmse" not in primary_tex.lower()
    assert "bias" not in primary_tex.lower()
    assert "Mean pinball loss" in detailed_tex
    assert "MAE p50" in detailed_tex
    assert "RMSE p50" in detailed_tex
    assert "Bias p50" in detailed_tex
    assert "Lower values are better for mean pinball loss, MAE and RMSE." in detailed_tex
    assert "Bias p50 is the mean p50 forecast error" in detailed_tex
    bias_line = next(line for line in detailed_tex.splitlines() if "Bias p50" in line and "&" in line)
    assert "-- & --" in bias_line
    assert "Winkler" not in primary_tex
    assert "PICP" not in primary_tex
    assert "PINAW" not in primary_tex

    primary_csv = pd.read_csv(tmp_path / "out" / "csv" / "rq1_4_1_1_forecast_metrics_full_primary_test.csv")
    detailed_csv = pd.read_csv(tmp_path / "out" / "csv" / "rq1_4_1_1_forecast_metrics_full_detailed_test.csv")
    assert list(primary_csv.columns) == [
        "target",
        "RLQR",
        "XGB",
        "TFT",
        "best_model",
        "relative_improvement_vs_RLQR_pct",
        "n_obs",
        "quantiles_used",
        "lead_min",
        "lead_max",
    ]
    assert list(detailed_csv.columns) == [
        "target",
        "metric",
        "RLQR",
        "XGB",
        "TFT",
        "best_model",
        "relative_improvement_vs_RLQR_pct",
        "n_obs",
        "quantiles_used",
        "lead_min",
        "lead_max",
    ]


def test_p50_absolute_error_tolerance_curve_uses_common_intersection(tmp_path: Path) -> None:
    joined = tmp_path / "benchmark" / "diagnostics" / "joined_predictions"
    _write_joined(joined / "linear__test__pred_da_price.parquet", _rows("linear", include_third=True))
    _write_joined(joined / "xgb__test__pred_da_price.parquet", _rows("xgb", include_third=True))
    _write_joined(joined / "tft__test__pred_da_price.parquet", _rows("tft", include_third=False))

    curve, summary = build_p50_error_tolerance_outputs(
        benchmark_dir=tmp_path / "benchmark",
        models=[ModelSpec("tft", "TFT"), ModelSpec("xgb", "XGB"), ModelSpec("linear", "RLQR")],
        splits=["test"],
        grid_size=5,
    )

    assert set(curve["n_obs"]) == {2}
    assert set(summary["n_obs"]) == {2}
    for model, part in curve.groupby("model"):
        ordered = part.sort_values("threshold")
        assert ordered["share_within_threshold"].is_monotonic_increasing, model

    tft_zero = curve[(curve["model"] == "TFT") & (curve["threshold"] == 0.0)].iloc[0]
    assert math.isclose(float(tft_zero["share_within_threshold"]), 0.5)
    xgb_one = summary[summary["model"] == "XGB"].iloc[0]
    assert math.isclose(float(xgb_one["share_le_1"]), 1.0)
    rlqr_one = summary[summary["model"] == "RLQR"].iloc[0]
    assert math.isclose(float(rlqr_one["share_le_1"]), 0.5)


def test_p50_absolute_error_tolerance_outputs_and_latex_snippet(tmp_path: Path) -> None:
    metrics, diagnostics = _build_small_metrics(tmp_path)
    curve, summary = build_p50_error_tolerance_outputs(
        benchmark_dir=tmp_path / "benchmark",
        models=[ModelSpec("tft", "TFT"), ModelSpec("xgb", "XGB"), ModelSpec("linear", "RLQR")],
        splits=["test"],
        grid_size=5,
    )
    outputs = write_outputs(
        metrics,
        diagnostics,
        benchmark_dir=tmp_path / "benchmark",
        out_dir=tmp_path / "out",
        split="test",
        p50_tolerance_curve=curve,
        p50_tolerance_summary=summary,
    )
    names = {p.name for p in outputs}
    assert "rq1_4_1_1_price_p50_absolute_error_tolerance_curve.csv" in names
    assert "rq1_4_1_1_da_price_p50_absolute_error_tolerance_curve.png" in names
    assert "rq1_4_1_1_da_price_p50_absolute_error_tolerance_curve.tex" in names
    assert "rq1_4_1_1_price_p50_error_tolerance_summary_test.csv" in names
    assert "rq1_4_1_1_da_price_p50_error_tolerance_summary_test.tex" in names

    tex = (tmp_path / "out" / "latex_figures" / "rq1_4_1_1_da_price_p50_absolute_error_tolerance_curve.tex").read_text(encoding="utf-8")
    assert r"\includegraphics" in tex
    assert r"\caption" in tex
    assert r"\label{fig:da_price_p50_absolute_error_tolerance_curve}" in tex
