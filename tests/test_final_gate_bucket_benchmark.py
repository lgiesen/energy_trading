from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.build_final_gate_bucket_benchmark import (
    ModelSpec,
    bucket_mask,
    build_gate_bucket_outputs,
    write_latex_table,
)


MODELS = [ModelSpec("tft", "TFT"), ModelSpec("xgb", "XGB"), ModelSpec("linear", "RLQR")]


def _utc(local: str) -> pd.Timestamp:
    return pd.Timestamp(local, tz="Europe/Berlin").tz_convert("UTC")


def _row(target: str, forecast_local: str, target_local: str, lead: int, y: float = 10.0) -> dict:
    return {
        "model": "tft",
        "split": "test",
        "target": target,
        "forecast_time_utc": _utc(forecast_local),
        "target_time_utc": _utc(target_local),
        "lead_time_h": lead,
        "y_true": y,
        "p10": y - 2.0,
        "p50": y,
        "p90": y + 2.0,
    }


def test_general_bucket_masks_select_expected_leads() -> None:
    df = pd.DataFrame(
        {
            "target": ["target_da_price"] * 50,
            "lead_time_h": list(range(1, 51)),
            "forecast_time_utc": pd.Timestamp("2025-01-01T00:00:00Z"),
            "target_time_utc": pd.Timestamp("2025-01-01T01:00:00Z"),
        }
    )
    assert set(df.loc[bucket_mask(df, "full_h1_48"), "lead_time_h"]) == set(range(1, 49))
    assert set(df.loc[bucket_mask(df, "short_h1_8"), "lead_time_h"]) == set(range(1, 9))
    assert set(df.loc[bucket_mask(df, "medium_h9_16"), "lead_time_h"]) == set(range(9, 17))
    assert set(df.loc[bucket_mask(df, "long_h17_48"), "lead_time_h"]) == set(range(17, 49))


def test_da_actionable_bucket_uses_11_local_dplus1_and_target_only() -> None:
    rows = [
        _row("target_da_price", "2025-01-01 11:00", f"2025-01-02 {hour:02d}:00", 13 + hour)
        for hour in range(24)
    ]
    rows.append(_row("target_afrr_capacity_price_pos", "2025-01-01 11:00", "2025-01-02 00:00", 13))
    rows.append(_row("target_da_price", "2025-01-01 10:00", "2025-01-02 00:00", 14))
    df = pd.DataFrame(rows)
    selected = df.loc[bucket_mask(df, "actionable_da_dplus1_11")]
    assert set(selected["target"]) == {"target_da_price"}
    assert set(selected["lead_time_h"]) == set(range(13, 37))


def test_bcm_actionable_bucket_uses_08_local_dplus1_and_capacity_only() -> None:
    rows = []
    for target in ["target_afrr_capacity_price_pos", "target_afrr_capacity_price_neg"]:
        rows.extend(
            _row(target, "2025-01-01 08:00", f"2025-01-02 {hour:02d}:00", 16 + hour)
            for hour in range(24)
        )
    rows.append(_row("target_da_price", "2025-01-01 08:00", "2025-01-02 00:00", 16))
    rows.append(_row("target_afrr_capacity_price_pos", "2025-01-01 09:00", "2025-01-02 00:00", 15))
    df = pd.DataFrame(rows)
    selected = df.loc[bucket_mask(df, "actionable_bcm_dplus1_08")]
    assert set(selected["target"]) == {"target_afrr_capacity_price_pos", "target_afrr_capacity_price_neg"}
    assert set(selected["lead_time_h"]) == set(range(16, 40))


def test_bem_actionable_bucket_uses_activation_targets_h1_to_h8() -> None:
    rows = [
        _row("target_afrr_activation_price_vwap_pos", "2025-01-01 08:00", "2025-01-01 09:00", 1),
        _row("target_afrr_activation_rate_neg", "2025-01-01 08:00", "2025-01-01 16:00", 8),
        _row("target_afrr_activation_rate_pos", "2025-01-01 08:00", "2025-01-01 17:00", 9),
        _row("target_da_price", "2025-01-01 08:00", "2025-01-01 09:00", 1),
    ]
    selected = pd.DataFrame(rows).loc[bucket_mask(pd.DataFrame(rows), "actionable_bem_short_h1_8")]
    assert set(selected["target"]) == {"target_afrr_activation_price_vwap_pos", "target_afrr_activation_rate_neg"}


def test_actionable_bucket_local_time_filter_handles_dst_conversion() -> None:
    rows = [
        _row("target_da_price", "2025-03-29 11:00", "2025-03-30 00:00", 13),
        _row("target_da_price", "2025-03-29 12:00", "2025-03-30 00:00", 12),
    ]
    selected = pd.DataFrame(rows).loc[bucket_mask(pd.DataFrame(rows), "actionable_da_dplus1_11")]
    assert len(selected) == 1
    assert int(selected["lead_time_h"].iloc[0]) == 13


def _write_joined(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _benchmark(tmp_path: Path) -> Path:
    joined = tmp_path / "benchmark" / "diagnostics" / "joined_predictions"
    base_rows = [
        _row("target_da_price", "2025-01-01 11:00", "2025-01-02 00:00", 13, y=10.0),
        _row("target_da_price", "2025-01-01 11:00", "2025-01-02 01:00", 14, y=20.0),
    ]
    for model in ["tft", "linear"]:
        rows = [{**r, "model": model, "target": "pred_da_price"} for r in base_rows]
        _write_joined(joined / f"{model}__test__pred_da_price.parquet", rows)
    xgb_rows = [{**r, "model": "xgb", "target": "pred_da_price"} for r in base_rows[:1]]
    _write_joined(joined / "xgb__test__pred_da_price.parquet", xgb_rows)
    return tmp_path / "benchmark"


def _benchmark_missing_forecast_time(tmp_path: Path) -> Path:
    joined = tmp_path / "benchmark_missing_forecast" / "diagnostics" / "joined_predictions"
    rows = [_row("target_da_price", "2025-01-01 11:00", "2025-01-02 00:00", 13, y=10.0)]
    for row in rows:
        row.pop("forecast_time_utc", None)
    for model in ["tft", "xgb", "linear"]:
        _write_joined(joined / f"{model}__test__pred_da_price.parquet", [{**r, "model": model, "target": "pred_da_price"} for r in rows])
    return tmp_path / "benchmark_missing_forecast"


def test_common_row_intersection_and_latex(tmp_path: Path) -> None:
    outputs = build_gate_bucket_outputs(
        benchmark_dir=_benchmark(tmp_path),
        models=MODELS,
        splits=["test"],
        derive_forecast_time=False,
    )
    row_counts = outputs["row_counts"]
    xgb = row_counts[(row_counts["model"] == "xgb") & (row_counts["bucket"] == "full_h1_48")].iloc[0]
    tft = row_counts[(row_counts["model"] == "tft") & (row_counts["bucket"] == "full_h1_48")].iloc[0]
    assert int(xgb["original_rows"]) == 1
    assert int(tft["original_rows"]) == 2
    assert int(tft["retained_common_rows"]) == 1
    assert int(tft["dropped_rows"]) == 1
    assert outputs["metrics"]["model_label"].isin(["TFT", "XGB", "RLQR"]).all()
    tex = write_latex_table(outputs["metrics"], out_dir=tmp_path, split="test")
    assert tex is not None
    content = tex.read_text(encoding="utf-8")
    assert "\\toprule" in content
    assert "\\midrule" in content
    assert "\\bottomrule" in content
    assert "\\textbf{Bucket}" in content


def test_actionable_bucket_missing_timestamps_fail_clearly() -> None:
    df = pd.DataFrame({"target": ["target_da_price"], "lead_time_h": [13], "target_time_utc": [pd.Timestamp("2025-01-02T00:00:00Z")]})
    with pytest.raises(KeyError, match="forecast_time_utc"):
        bucket_mask(df, "actionable_da_dplus1_11")
    df2 = pd.DataFrame({"target": ["target_da_price"], "lead_time_h": [13], "forecast_time_utc": [pd.Timestamp("2025-01-01T10:00:00Z")]})
    with pytest.raises(KeyError, match="target_time_utc"):
        bucket_mask(df2, "actionable_da_dplus1_11")


def test_build_outputs_missing_forecast_time_fails_without_explicit_fallback(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing forecast_time_utc"):
        build_gate_bucket_outputs(
            benchmark_dir=_benchmark_missing_forecast_time(tmp_path),
            models=MODELS,
            splits=["test"],
            derive_forecast_time=False,
        )
    outputs = build_gate_bucket_outputs(
        benchmark_dir=_benchmark_missing_forecast_time(tmp_path),
        models=MODELS,
        splits=["test"],
        derive_forecast_time=True,
    )
    assert outputs["warnings"]["message"].str.contains("forecast_time_utc was derived").any()
