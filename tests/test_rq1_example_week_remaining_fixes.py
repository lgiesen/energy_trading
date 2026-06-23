from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_rq1_example_weeks import (
    WEEK_TIMEZONE,
    _latex_escape,
    _prepare_tail_spike_high_vol_source,
    validate_algorithmic_outputs,
    target_y_axis_label,
    target_y_axis_unit_label,
)


def test_target_y_axis_labels_use_units_and_activation_rate_percent() -> None:
    assert target_y_axis_label("target_da_price") == "DA price (EUR/MWh)"
    assert target_y_axis_label("target_afrr_capacity_price_neg") == "Capacity price - (EUR/MW)"
    assert target_y_axis_label("target_afrr_activation_price_vwap_pos") == "Activation price + (EUR/MWh)"
    assert target_y_axis_label("target_afrr_activation_rate_neg") == "Activation rate - (%)"
    assert target_y_axis_unit_label("target_afrr_activation_rate_neg") == "Activation rate - (%)"
    assert _latex_escape(target_y_axis_unit_label("target_afrr_activation_rate_neg")) == r"Activation rate - (\%)"


def test_tail_spike_source_missing_rank_is_ranked_by_volatility_score(tmp_path: Path) -> None:
    src = pd.DataFrame(
        {
            "split": ["test", "test"],
            "selection_type": ["high_volatility_week", "high_volatility_week"],
            "target": ["target_da_price", "target_da_price"],
            "week_start_utc": ["2025-01-06T00:00:00Z", "2025-01-13T00:00:00Z"],
            "weekly_std": [2.0, 5.0],
        }
    )
    out, warnings = _prepare_tail_spike_high_vol_source(
        src,
        split="test",
        canonical_targets=["target_da_price"],
        source=tmp_path / "tail_spike_selected_weeks.csv",
    )
    assert warnings == []
    best = out.sort_values("selection_rank").iloc[0]
    assert int(best["selection_rank"]) == 1
    assert float(best["volatility_score"]) == 5.0
    assert pd.Timestamp(best["week_end_utc"]) == pd.Timestamp("2025-01-20T00:00:00Z")
    assert best["week_timezone"] == WEEK_TIMEZONE
    assert "UTC week" in best["selection_rule"]


def test_tail_spike_target_group_only_source_is_rejected(tmp_path: Path) -> None:
    src = pd.DataFrame(
        {
            "split": ["test"],
            "selection_type": ["high_volatility_week"],
            "target_group": ["aFRR activation price"],
            "week_start_utc": ["2025-01-06T00:00:00Z"],
            "weekly_std": [10.0],
        }
    )
    out, warnings = _prepare_tail_spike_high_vol_source(
        src,
        split="test",
        canonical_targets=["target_afrr_activation_price_vwap_pos"],
        source=tmp_path / "tail_spike_selected_weeks.csv",
    )
    assert out.empty
    assert warnings
    assert warnings[0]["warning"] == "tail_spike_selected_weeks_is_not_target_specific_recomputed_high_volatility"


def test_validate_algorithmic_outputs_fails_and_logs_missing_figure_path(tmp_path: Path) -> None:
    out_dir = tmp_path / "4_1_6_example_weeks"
    (out_dir / "backup" / "diagnostics").mkdir(parents=True)
    (out_dir / "backup" / "csv").mkdir(parents=True)
    (out_dir / "backup" / "warnings").mkdir(parents=True)

    selected = pd.DataFrame(
        {
            "selection_mode": ["algorithmic"],
            "split": ["test"],
            "selection_type": ["typical_week"],
            "target": ["target_da_price"],
            "target_display": ["DA price"],
            "target_group": ["DA"],
            "week_start_utc": ["2025-01-06T00:00:00Z"],
            "week_end_utc": ["2025-01-13T00:00:00Z"],
            "weekly_std": [1.0],
            "median_weekly_std": [1.0],
            "abs_distance_to_median_std": [0.0],
            "volatility_score": [1.0],
            "n_rows": [168],
            "selection_rank": [1],
            "selection_scope": ["target"],
            "selection_rule": ["Selected test-set UTC week."],
            "source": ["computed"],
            "week_timezone": [WEEK_TIMEZONE],
        }
    )
    selected.to_csv(out_dir / "backup" / "diagnostics" / "example_week_selected_weeks.csv", index=False)

    values = pd.DataFrame(
        {
            "selection_mode": ["algorithmic"],
            "selection_type": ["typical_week"],
            "market_context": ["bem_h1"],
            "forecast_snapshot_rule": ["BEM h1"],
            "split": ["test"],
            "target": ["target_da_price"],
            "target_display": ["DA price"],
            "target_group": ["DA"],
            "y_axis_label": ["DA price (EUR/MWh)"],
            "figure_title": ["DA Price: p50 Forecast"],
            "figure_subtitle": ["Typical week | DA D−1 11:00 Europe/Berlin forecast snapshot"],
            "caption": [
                "Example-week DA price forecasts for the algorithmically selected typical week using the DA D$-1$ 11:00 Europe/Berlin forecast snapshot. The figure compares realized values with p50 forecasts from RLQR, XGB and TFT."
            ],
            "short_caption": ["Example-week DA price forecasts"],
            "market_context_label": ["DA D−1 11:00 Europe/Berlin forecast snapshot"],
            "week_start_utc": ["2025-01-06T00:00:00Z"],
            "week_end_utc": ["2025-01-13T00:00:00Z"],
            "weekly_std": [1.0],
            "median_weekly_std": [1.0],
            "abs_distance_to_median_std": [0.0],
            "volatility_score": [1.0],
            "selection_rank": [1],
            "selection_scope": ["target"],
            "selection_rule": ["Selected test-set UTC week."],
            "source": ["computed"],
            "week_timezone": [WEEK_TIMEZONE],
            "target_time_utc": ["2025-01-06T01:00:00Z"],
            "forecast_time_utc": ["2025-01-06T00:00:00Z"],
            "lead_time_h": [1.0],
            "model": ["xgb"],
            "model_label": ["XGB"],
            "y_true": [1.0],
            "p10": [0.0],
            "p50": [1.0],
            "p90": [2.0],
            "residual_p50": [0.0],
            "abs_error_p50": [0.0],
            "figure_path": [str(out_dir / "result_section" / "figures" / "missing.png")],
        }
    )
    values.to_csv(out_dir / "backup" / "csv" / "example_week_plot_values.csv", index=False)
    values[["figure_path"]].to_csv(out_dir / "backup" / "csv" / "example_week_metrics.csv", index=False)
    (out_dir / "backup" / "warnings" / "example_week_warnings.csv").write_text("", encoding="utf-8")
    (out_dir / "example_week_manifest.json").write_text(
        json.dumps({"artifacts": [{"path": str(out_dir / "result_section" / "figures" / "missing.png"), "artifact_type": "figure"}]}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        validate_algorithmic_outputs(out_dir=out_dir, split="test")

    warnings = pd.read_csv(out_dir / "backup" / "warnings" / "example_week_warnings.csv")
    assert "missing_generated_artifact_path" in set(warnings["warning"])
