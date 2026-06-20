from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_final_per_lead_benchmark import (
    ModelSpec,
    assign_lead_range,
    build_per_lead_outputs,
    build_thesis_range_table,
    write_latex_range_table,
)


MODELS = [ModelSpec("tft", "TFT"), ModelSpec("xgb", "XGB"), ModelSpec("linear", "RLQR")]


def _write_joined(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _rows(model: str, *, extra_lead1: bool = False) -> list[dict]:
    rows = [
        {
            "model": model,
            "split": "test",
            "target": "pred_da_price",
            "target_time_utc": "2025-01-01T01:00:00Z",
            "forecast_time_utc": "2025-01-01T00:00:00Z",
            "lead_time_h": 1,
            "y_true": 2.0,
            "p10": 0.0,
            "p50": 1.0,
            "p90": 3.0,
        },
        {
            "model": model,
            "split": "test",
            "target": "pred_da_price",
            "target_time_utc": "2025-01-01T02:00:00Z",
            "forecast_time_utc": "2025-01-01T00:00:00Z",
            "lead_time_h": 2,
            "y_true": 4.0,
            "p10": 2.0,
            "p50": 5.0,
            "p90": 6.0,
        },
    ]
    if extra_lead1:
        rows.append(
            {
                "model": model,
                "split": "test",
                "target": "pred_da_price",
                "target_time_utc": "2025-01-01T03:00:00Z",
                "forecast_time_utc": "2025-01-01T02:00:00Z",
                "lead_time_h": 1,
                "y_true": 10.0,
                "p10": 9.0,
                "p50": 10.0,
                "p90": 11.0,
            }
        )
    return rows


def _small_benchmark(tmp_path: Path) -> Path:
    joined = tmp_path / "benchmark" / "diagnostics" / "joined_predictions"
    _write_joined(joined / "tft__test__pred_da_price.parquet", _rows("tft"))
    _write_joined(joined / "xgb__test__pred_da_price.parquet", _rows("xgb", extra_lead1=True))
    _write_joined(joined / "linear__test__pred_da_price.parquet", _rows("linear"))
    return tmp_path / "benchmark"


def test_assign_lead_range_boundaries() -> None:
    assert assign_lead_range(1) == "short_h1_8"
    assert assign_lead_range(8) == "short_h1_8"
    assert assign_lead_range(9) == "medium_h9_16"
    assert assign_lead_range(16) == "medium_h9_16"
    assert assign_lead_range(17) == "long_h17_48"
    assert assign_lead_range(48) == "long_h17_48"
    assert assign_lead_range(49) is None


def test_per_lead_metrics_and_common_intersection_by_lead(tmp_path: Path) -> None:
    benchmark = _small_benchmark(tmp_path)
    outputs = build_per_lead_outputs(benchmark_dir=benchmark, models=MODELS, splits=["test"], horizon=3)
    metrics = outputs["metrics"]
    row_counts = outputs["row_counts"]

    xgb_h1 = row_counts[(row_counts["model"] == "xgb") & (row_counts["lead_time_h"] == 1)].iloc[0]
    assert xgb_h1["valid_rows"] == 2
    assert xgb_h1["retained_common_rows"] == 1
    assert xgb_h1["dropped_rows"] == 1
    assert xgb_h1["row_intersection_key"] == "split,target,forecast_time_utc,target_time_utc,lead_time_h"

    lead1 = metrics[(metrics["model"] == "tft") & (metrics["lead_time_h"] == 1)].iloc[0]
    # Pinball values for y=2, p10=0, p50=1, p90=3 are 0.2, 0.5, 0.1.
    assert lead1["mean_pinball_loss"] == pytest.approx(np.mean([0.2, 0.5, 0.1]))
    assert lead1["mae_p50"] == pytest.approx(1.0)
    assert lead1["rmse_p50"] == pytest.approx(1.0)
    assert lead1["coverage_p10_p90"] == pytest.approx(1.0)
    assert outputs["warnings"]["message"].str.contains("Missing lead hours").any()


def test_range_table_and_latex_are_compact(tmp_path: Path) -> None:
    benchmark = _small_benchmark(tmp_path)
    outputs = build_per_lead_outputs(benchmark_dir=benchmark, models=MODELS, splits=["test"], horizon=2)
    table = build_thesis_range_table(outputs["range_summary"], split="test")
    assert list(table.columns) == ["target", "lead_range", "TFT", "XGB", "RLQR", "best_model", "n_obs"]
    assert set(table["lead_range"]) == {"h1-h8"}
    path = write_latex_range_table(table, out_dir=tmp_path, split="test")
    assert path is not None
    tex = path.read_text(encoding="utf-8")
    assert "\\toprule" in tex
    assert "\\midrule" in tex
    assert "\\bottomrule" in tex
    assert "\\textbf{Target}" in tex
    assert "pinball" in tex.lower()
    assert "MAE" not in tex
    assert "RMSE" not in tex
