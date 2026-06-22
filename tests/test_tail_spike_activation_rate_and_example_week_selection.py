from __future__ import annotations

import pandas as pd

from scripts.build_final_tail_spike_benchmark import (
    compute_target_thresholds,
    regime_masks_for_target,
    select_example_week,
    select_spike_weeks,
)


def _rate_rows(target: str, week1: list[float], week2: list[float]) -> pd.DataFrame:
    rows = []
    for offset, value in enumerate([*week1, *week2]):
        rows.append(
            {
                "split": "test",
                "target": target,
                "target_group": "aFRR activation rate",
                "target_time_utc": pd.Timestamp("2025-01-06", tz="UTC") + pd.Timedelta(days=offset),
                "y_true": float(value),
            }
        )
    return pd.DataFrame(rows)


def test_activation_rate_spike_weeks_use_high_tail_not_zero_nonzero_regimes() -> None:
    pos = _rate_rows(
        "target_afrr_activation_rate_pos",
        week1=[0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.00],
        week2=[0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
    )
    neg = _rate_rows(
        "target_afrr_activation_rate_neg",
        week1=[0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        week2=[0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97],
    )
    base = pd.concat([pos, neg], ignore_index=True)

    thresholds = compute_target_thresholds(
        {target: part for target, part in base.groupby("target")},
        threshold_source="test_conditional",
        epsilon=1e-9,
    )
    rate_thresholds = thresholds.loc[thresholds["regime"].eq("afrr_activation_rate_high_tail_top5")]
    assert set(rate_thresholds["target"]) == {
        "target_afrr_activation_rate_pos",
        "target_afrr_activation_rate_neg",
    }
    assert set(rate_thresholds["threshold_name"]) == {"activation_rate_high_tail_q95"}

    pos_masks = regime_masks_for_target(pos, thresholds, epsilon=1e-9)
    assert "activation_nonzero" in pos_masks
    assert "activation_zero_or_nearzero" in pos_masks
    assert "afrr_activation_rate_high_tail_top5" in pos_masks

    selected = select_spike_weeks(base, thresholds, epsilon=1e-9, top_share=0.5)
    pos_week = selected.loc[selected["target"].eq("target_afrr_activation_rate_pos")].iloc[0]
    neg_week = selected.loc[selected["target"].eq("target_afrr_activation_rate_neg")].iloc[0]
    assert pos_week["week_start_utc"] == pd.Timestamp("2025-01-06", tz="UTC")
    assert neg_week["week_start_utc"] == pd.Timestamp("2025-01-13", tz="UTC")
    assert set(selected["threshold_name"]) == {"activation_rate_high_tail_q95"}
    assert set(selected["selection_scope"]) == {"target"}
    assert set(selected["selection_rule"]) == {
        "Top 10% weeks by target-specific high-activation-rate tail share and count, selected separately per split and target."
    }


def test_example_week_selection_uses_strongest_metrics_not_first_sorted_row() -> None:
    selected = pd.DataFrame(
        [
            {
                "split": "test",
                "target": "target_afrr_activation_rate_pos",
                "target_display": "aFRR activation rate +",
                "target_group": "aFRR activation rate",
                "week_start_utc": pd.Timestamp("2025-01-06", tz="UTC"),
                "selection_type": "high_volatility_week",
                "volatility_score": 0.1,
                "n_rows": 7,
            },
            {
                "split": "test",
                "target": "target_afrr_activation_rate_neg",
                "target_display": "aFRR activation rate -",
                "target_group": "aFRR activation rate",
                "week_start_utc": pd.Timestamp("2025-01-13", tz="UTC"),
                "selection_type": "high_volatility_week",
                "volatility_score": 0.9,
                "n_rows": 7,
            },
            {
                "split": "test",
                "target": "target_afrr_activation_rate_pos",
                "target_display": "aFRR activation rate +",
                "target_group": "aFRR activation rate",
                "week_start_utc": pd.Timestamp("2025-01-06", tz="UTC"),
                "selection_type": "spike_week",
                "spike_share": 0.2,
                "spike_count": 2,
                "n_rows": 10,
            },
            {
                "split": "test",
                "target": "target_afrr_activation_rate_neg",
                "target_display": "aFRR activation rate -",
                "target_group": "aFRR activation rate",
                "week_start_utc": pd.Timestamp("2025-01-13", tz="UTC"),
                "selection_type": "spike_week",
                "spike_share": 0.2,
                "spike_count": 4,
                "n_rows": 10,
            },
        ]
    )

    hv_row, hv_metric, hv_reason, hv_value = select_example_week(
        selected,
        split="test",
        selection_type="high_volatility_week",
    )
    assert hv_row is not None
    assert hv_row["target"] == "target_afrr_activation_rate_neg"
    assert hv_metric == "volatility_score"
    assert hv_value == 0.9
    assert "highest volatility_score" in hv_reason

    spike_row, spike_metric, spike_reason, spike_value = select_example_week(
        selected,
        split="test",
        selection_type="spike_week",
    )
    assert spike_row is not None
    assert spike_row["target"] == "target_afrr_activation_rate_neg"
    assert spike_metric == "spike_share"
    assert spike_value == 0.2
    assert "highest spike_share and spike_count" in spike_reason
