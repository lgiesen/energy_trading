from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from scripts.build_final_full_forecast_metrics import (
    ModelSpec,
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
        "TFT",
        "XGB",
        "RLQR",
        "best_model",
        "relative_improvement_vs_RLQR_pct",
        "n_obs",
        "quantiles_used",
        "lead_min",
        "lead_max",
    ]
    assert primary["best_model"].iloc[0] in {"TFT", "XGB", "RLQR"}


def test_detailed_table_contains_all_four_metrics(tmp_path: Path) -> None:
    metrics, _ = _build_small_metrics(tmp_path)
    detailed = build_detailed_table(metrics, split="test")
    assert set(detailed["metric"]) == {"mean_pinball_loss", "mae_p50", "rmse_p50", "bias_p50"}
    bias_row = detailed.loc[detailed["metric"] == "bias_p50"].iloc[0]
    assert bias_row["best_model"] == ""
    assert pd.isna(bias_row["relative_improvement_vs_RLQR_pct"])


def test_outputs_exclude_interval_metrics_from_4_1_1(tmp_path: Path) -> None:
    metrics, diagnostics = _build_small_metrics(tmp_path)
    outputs = write_outputs(metrics, diagnostics, out_dir=tmp_path / "out", split="test")
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
    assert r"\textbf{Target} & \textbf{TFT} & \textbf{XGB} & \textbf{RLQR} & \textbf{Best model} & \textbf{Improvement vs RLQR (\%)} & \textbf{N}" in primary_tex
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
        "TFT",
        "XGB",
        "RLQR",
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
        "TFT",
        "XGB",
        "RLQR",
        "best_model",
        "relative_improvement_vs_RLQR_pct",
        "n_obs",
        "quantiles_used",
        "lead_min",
        "lead_max",
    ]
