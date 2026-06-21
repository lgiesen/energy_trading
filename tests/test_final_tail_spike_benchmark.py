from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.build_final_tail_spike_benchmark import (
    ModelSpec,
    compute_target_thresholds,
    regime_masks_for_target,
    select_high_volatility_weeks,
    build_tail_spike_outputs,
    plot_forecast_band_example,
    write_latex_table,
)


MODELS = [ModelSpec("tft", "TFT"), ModelSpec("xgb", "XGB"), ModelSpec("linear", "RLQR")]


def _utc(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts, tz="UTC")


def test_da_tail_threshold_regimes_select_expected_rows() -> None:
    df = pd.DataFrame(
        {
            "target": ["target_da_price"] * 20,
            "y_true": list(range(-10, 10)),
        }
    )
    thresholds = compute_target_thresholds({"target_da_price": df}, threshold_source="test_conditional", epsilon=1e-9)
    masks = regime_masks_for_target(df, thresholds, epsilon=1e-9)
    assert masks["da_abs_tail_top5"].sum() >= 1
    assert df.loc[masks["da_positive_spike_top5"], "y_true"].min() >= thresholds.loc[thresholds["regime"].eq("da_positive_spike_top5"), "threshold_value"].iloc[0]
    assert df.loc[masks["da_negative_spike_bottom5"], "y_true"].max() <= thresholds.loc[thresholds["regime"].eq("da_negative_spike_bottom5"), "threshold_value"].iloc[0]


def test_activation_zero_nonzero_epsilon_regimes() -> None:
    df = pd.DataFrame(
        {
            "target": ["target_afrr_activation_rate_pos"] * 4,
            "y_true": [0.0, 1e-10, 1e-8, 0.5],
        }
    )
    thresholds = compute_target_thresholds({"target_afrr_activation_rate_pos": df}, threshold_source="domain", epsilon=1e-9)
    masks = regime_masks_for_target(df, thresholds, epsilon=1e-9)
    assert masks["activation_zero_or_nearzero"].tolist() == [True, True, False, False]
    assert masks["activation_nonzero"].tolist() == [False, False, True, True]


def test_high_volatility_week_selection() -> None:
    ts = pd.date_range("2025-01-01", periods=14, freq="D", tz="UTC")
    y = [1.0] * 7 + [0.0, 10.0, -10.0, 8.0, -8.0, 6.0, -6.0]
    df = pd.DataFrame(
        {
            "split": "test",
            "target_group": "DA price",
            "target_time_utc": ts,
            "y_true": y,
        }
    )
    weeks = select_high_volatility_weeks(df, top_share=0.5)
    assert not weeks.empty
    assert float(weeks["volatility_score"].max()) > 0.0


def _row(target: str, i: int, y: float) -> dict:
    return {
        "model": "x",
        "split": "test",
        "target": target,
        "forecast_time_utc": _utc(f"2025-01-01 {i:02d}:00"),
        "target_time_utc": _utc(f"2025-01-02 {i:02d}:00"),
        "lead_time_h": 24,
        "y_true": y,
        "p10": y - 1.0,
        "p50": y + 1.0,
        "p90": y + 2.0,
    }


def _write_joined(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _benchmark(tmp_path: Path) -> Path:
    joined = tmp_path / "benchmark" / "diagnostics" / "joined_predictions"
    test_rows = [_row("pred_da_price", i, float(i)) for i in range(10)]
    val_rows = [_row("pred_da_price", i, float(i)) for i in range(10)]
    for model in ["tft", "linear"]:
        _write_joined(joined / f"{model}__test__pred_da_price.parquet", [{**r, "model": model} for r in test_rows])
        _write_joined(joined / f"{model}__val__pred_da_price.parquet", [{**r, "model": model} for r in val_rows])
    _write_joined(joined / "xgb__test__pred_da_price.parquet", [{**r, "model": "xgb"} for r in test_rows[:-1]])
    _write_joined(joined / "xgb__val__pred_da_price.parquet", [{**r, "model": "xgb"} for r in val_rows])
    return tmp_path / "benchmark"


def test_common_row_intersection_and_metrics(tmp_path: Path) -> None:
    outputs = build_tail_spike_outputs(
        benchmark_dir=_benchmark(tmp_path),
        models=MODELS,
        splits=["val", "test"],
        main_split="test",
        epsilon=1e-9,
        derive_forecast_time=False,
    )
    row_counts = outputs["row_counts"]
    tft_rows = row_counts[(row_counts["split"] == "test") & (row_counts["model"] == "tft")]
    assert int(tft_rows["dropped_rows"].sum()) > 0
    metrics = outputs["metrics"]
    row = metrics[(metrics["split"] == "test") & (metrics["model"] == "tft")].iloc[0]
    assert row["mae_p50"] == pytest.approx(1.0)
    assert row["bias_p50"] == pytest.approx(1.0)
    assert "validation" in set(outputs["thresholds"]["threshold_source"])


def test_forecast_band_plot_requires_single_lead_or_snapshot(tmp_path: Path) -> None:
    df = pd.DataFrame([_row("pred_da_price", 0, 1.0), _row("pred_da_price", 1, 2.0)])
    with pytest.raises(ValueError, match="exactly one lead"):
        plot_forecast_band_example(df, out_path=tmp_path / "bad.png")
    with pytest.raises(ValueError, match="exactly one lead"):
        plot_forecast_band_example(df, out_path=tmp_path / "bad.png", lead=24, snapshot="2025-01-01T00:00:00Z")


def test_latex_table_booktabs(tmp_path: Path) -> None:
    outputs = build_tail_spike_outputs(
        benchmark_dir=_benchmark(tmp_path),
        models=MODELS,
        splits=["val", "test"],
        main_split="test",
        epsilon=1e-9,
        derive_forecast_time=False,
    )
    path = write_latex_table(outputs["metrics"], out_dir=tmp_path, split="test")
    assert path is not None
    text = path.read_text(encoding="utf-8")
    assert "\\toprule" in text
    assert "\\midrule" in text
    assert "\\bottomrule" in text
    assert r"\textbf{\shortstack{Target\\group}}" in text
    assert r"\textbf{\shortstack{Best\\model}}" in text
    assert r"\textbf{\shortstack{Main\\issue}}" in text
