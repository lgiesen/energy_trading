from __future__ import annotations

import pandas as pd

from scripts.build_final_tail_spike_benchmark import (
    compute_target_thresholds,
    regime_masks_for_target,
    select_high_volatility_weeks,
    select_spike_weeks,
)


def _capacity_rows(target: str, week1: list[float], week2: list[float]) -> pd.DataFrame:
    rows = []
    for offset, value in enumerate([*week1, *week2]):
        rows.append(
            {
                "split": "test",
                "target": target,
                "target_group": "aFRR capacity price",
                "target_time_utc": pd.Timestamp("2025-01-06", tz="UTC") + pd.Timedelta(days=offset),
                "y_true": float(value),
            }
        )
    return pd.DataFrame(rows)


def test_high_volatility_weeks_are_selected_per_capacity_target_direction() -> None:
    pos = _capacity_rows(
        "target_afrr_capacity_price_pos",
        week1=[0, 10, 0, 10, 0, 10, 0],
        week2=[5, 5, 5, 5, 5, 5, 5],
    )
    neg = _capacity_rows(
        "target_afrr_capacity_price_neg",
        week1=[5, 5, 5, 5, 5, 5, 5],
        week2=[0, 20, 0, 20, 0, 20, 0],
    )

    selected = select_high_volatility_weeks(pd.concat([pos, neg], ignore_index=True), top_share=0.5)

    pos_week = selected.loc[selected["target"].eq("target_afrr_capacity_price_pos")].iloc[0]
    neg_week = selected.loc[selected["target"].eq("target_afrr_capacity_price_neg")].iloc[0]
    assert pos_week["week_start_utc"] == pd.Timestamp("2025-01-06", tz="UTC")
    assert neg_week["week_start_utc"] == pd.Timestamp("2025-01-13", tz="UTC")
    assert set(selected["selection_scope"]) == {"target"}
    assert set(selected["target_group"]) == {"aFRR capacity price"}
    assert set(selected["target"]) == {"target_afrr_capacity_price_pos", "target_afrr_capacity_price_neg"}


def test_capacity_price_spike_weeks_use_formal_target_specific_q95_thresholds() -> None:
    pos = _capacity_rows(
        "target_afrr_capacity_price_pos",
        week1=[100, 100, 100, 100, 100, 100, 100],
        week2=[0, 0, 0, 0, 0, 0, 0],
    )
    neg = _capacity_rows(
        "target_afrr_capacity_price_neg",
        week1=[0, 0, 0, 0, 0, 0, 0],
        week2=[200, 200, 200, 200, 200, 200, 200],
    )
    base = pd.concat([pos, neg], ignore_index=True)

    thresholds = compute_target_thresholds(
        {target: part for target, part in base.groupby("target")},
        threshold_source="test_conditional",
        epsilon=1e-9,
    )
    cap_thresholds = thresholds.loc[thresholds["target_group"].eq("aFRR capacity price")]
    assert set(cap_thresholds["target"]) == {
        "target_afrr_capacity_price_pos",
        "target_afrr_capacity_price_neg",
    }
    assert set(cap_thresholds["threshold_name"]) == {"capacity_price_high_tail_q95"}
    assert cap_thresholds.loc[
        cap_thresholds["target"].eq("target_afrr_capacity_price_pos"),
        "threshold_value",
    ].iloc[0] == 100.0
    assert cap_thresholds.loc[
        cap_thresholds["target"].eq("target_afrr_capacity_price_neg"),
        "threshold_value",
    ].iloc[0] == 200.0

    pos_masks = regime_masks_for_target(pos, thresholds, epsilon=1e-9)
    assert "afrr_capacity_price_high_tail_top5" in pos_masks
    assert int(pos_masks["afrr_capacity_price_high_tail_top5"].sum()) == 7

    selected = select_spike_weeks(base, thresholds, epsilon=1e-9, top_share=0.5)
    pos_week = selected.loc[selected["target"].eq("target_afrr_capacity_price_pos")].iloc[0]
    neg_week = selected.loc[selected["target"].eq("target_afrr_capacity_price_neg")].iloc[0]
    assert pos_week["week_start_utc"] == pd.Timestamp("2025-01-06", tz="UTC")
    assert neg_week["week_start_utc"] == pd.Timestamp("2025-01-13", tz="UTC")
    assert set(selected["selection_scope"]) == {"target"}
    assert set(selected["threshold_name"]) == {"capacity_price_high_tail_q95"}
    assert set(selected["selection_rule"]) == {
        "Top 10% weeks by target-specific spike_share and spike_count, selected separately per split and target."
    }
