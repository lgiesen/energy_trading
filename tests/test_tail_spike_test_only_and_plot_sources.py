from __future__ import annotations

import pandas as pd

from scripts.build_final_tail_spike_benchmark import (
    _regime_definitions,
    compute_target_thresholds,
    regime_masks_for_target,
    relative_pinball_plot_source,
    residual_distribution_source,
    select_example_week,
    select_high_volatility_weeks,
    select_spike_weeks,
    validate_tail_spike_outputs,
    write_outputs,
)


def _rows(split: str, target: str, week_start: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": split,
                "target": target,
                "target_group": "aFRR activation rate",
                "target_time_utc": pd.Timestamp(week_start, tz="UTC") + pd.Timedelta(days=i),
                "y_true": float(v),
            }
            for i, v in enumerate(values)
        ]
    )


def test_tail_spike_selected_weeks_are_test_only_and_activation_rate_zero_q95_not_all_spikes(tmp_path) -> None:
    pos_test = pd.concat(
        [
            _rows("test", "target_afrr_activation_rate_pos", "2025-01-06", [0.0] * 7),
            _rows("test", "target_afrr_activation_rate_pos", "2025-01-13", [0.0, 0.0, 0.0, 0.2, 0.3, 0.4, 0.5]),
        ],
        ignore_index=True,
    )
    neg_test = pd.concat(
        [
            _rows("test", "target_afrr_activation_rate_neg", "2025-01-06", [0.0] * 7),
            _rows("test", "target_afrr_activation_rate_neg", "2025-01-13", [0.0, 0.0, 0.0, 0.8, 0.9, 1.0, 1.0]),
        ],
        ignore_index=True,
    )
    train = _rows("train", "target_afrr_activation_rate_pos", "2024-12-30", [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    val = _rows("val", "target_afrr_activation_rate_neg", "2024-12-30", [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    base = pd.concat([train, val, pos_test, neg_test], ignore_index=True)
    selection_base = base.loc[base["split"].eq("test")].copy()

    thresholds = compute_target_thresholds(
        {target: part for target, part in selection_base.groupby("target")},
        threshold_source="test_conditional",
        epsilon=1e-9,
        source_splits=["test"],
        selection_split="test",
    )
    assert set(thresholds["source_splits"]) == {"test"}
    assert set(thresholds["selection_split"]) == {"test"}

    zero_only = pos_test.loc[pos_test["target_time_utc"].lt(pd.Timestamp("2025-01-13", tz="UTC"))].copy()
    zero_masks = regime_masks_for_target(zero_only, thresholds, epsilon=1e-9)
    assert int(zero_masks["afrr_activation_rate_high_tail_top5"].sum()) == 0

    hv = select_high_volatility_weeks(selection_base, top_share=0.5)
    sp = select_spike_weeks(selection_base, thresholds, epsilon=1e-9, top_share=0.5)
    selected = pd.concat([hv, sp], ignore_index=True)
    assert set(selected["split"]) == {"test"}
    assert {"target_afrr_activation_rate_pos", "target_afrr_activation_rate_neg"} <= set(selected["target"])
    assert set(selected["selection_scope"]) == {"target"}

    week, metric, reason, value = select_example_week(selected, split="test", selection_type="spike_week")
    assert week is not None
    assert week["split"] == "test"
    assert metric == "spike_share"
    assert "highest spike_share" in reason
    assert value >= 0.0

    metrics = pd.DataFrame(
        [
            {
                "split": "test",
                "target": "target_afrr_activation_rate_pos",
                "target_label": "aFRR activation rate +",
                "target_group": "aFRR activation rate",
                "regime": "spike_week",
                "model": model,
                "model_label": label,
                "mean_pinball_loss": loss,
                "n_obs": 7,
            }
            for model, label, loss in [("linear", "RLQR", 2.0), ("xgb", "XGB", 1.0), ("tft", "TFT", 1.5)]
        ]
    )
    points = pd.DataFrame(
        [
            {
                "forecast_time_utc": pd.Timestamp("2025-01-13", tz="UTC"),
                "target_time_utc": pd.Timestamp("2025-01-13", tz="UTC") + pd.Timedelta(hours=i),
                "lead_time_h": 0,
                "y_true": float(i),
                "p10": float(i) - 0.1,
                "p50": float(i) + 0.1,
                "p90": float(i) + 0.2,
                "model": model,
                "model_label": label,
                "split": "test",
                "target": "target_afrr_activation_rate_pos",
                "target_label": "aFRR activation rate +",
                "target_group": "aFRR activation rate",
                "regime": "spike_week",
            }
            for model, label in [("linear", "RLQR"), ("xgb", "XGB"), ("tft", "TFT")]
            for i in range(3)
        ]
    )
    outputs = {
        "metrics": metrics,
        "definitions": _regime_definitions(),
        "thresholds": thresholds,
        "row_counts": pd.DataFrame(),
        "selected_weeks": sp.loc[sp["target"].eq("target_afrr_activation_rate_pos")].head(1).copy(),
        "warnings": pd.DataFrame(columns=["split", "target", "regime", "severity", "message"]),
        "points": points,
    }
    written = write_outputs(outputs, out_dir=tmp_path, split="test", structured_out_dir=None)
    written_names = {p.name for p in written}
    assert "tail_spike_points.csv" in written_names
    assert "tail_spike_points_test.csv" in written_names
    assert "tail_spike_plot_source_relative_pinball.csv" in written_names
    assert "tail_spike_example_week_selection.csv" in written_names
    assert "tail_spike_example_week_plot_values.csv" in written_names
    assert pd.read_csv(tmp_path / "tail_spike_points_test.csv").empty is False
    assert pd.read_csv(tmp_path / "tail_spike_plot_source_relative_pinball.csv").empty is False
    assert pd.read_csv(tmp_path / "tail_spike_example_week_plot_values.csv").empty is False

    validation_warnings = validate_tail_spike_outputs(
        out_dir=tmp_path,
        split="test",
        selected_weeks=outputs["selected_weeks"],
        points_test=pd.read_csv(tmp_path / "tail_spike_points_test.csv"),
        relative_source=relative_pinball_plot_source(metrics, split="test"),
        example_selection=pd.read_csv(tmp_path / "tail_spike_example_week_selection.csv"),
        example_values=pd.read_csv(tmp_path / "tail_spike_example_week_plot_values.csv"),
    )
    assert validation_warnings == []
    residual = residual_distribution_source(points, split="test")
    assert {"residual_p50", "abs_error_p50"} <= set(residual.columns)
