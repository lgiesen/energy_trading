from __future__ import annotations

import pandas as pd

from scripts.build_rq1_example_weeks import (
    WeekSpec,
    _select_market_actionable_rows,
    select_high_volatility_weeks_algorithmic,
    select_typical_weeks_algorithmic,
)


def _weekly_base(stds: list[float], *, split: str = "test", target: str = "target_da_price") -> pd.DataFrame:
    rows = []
    for week, std in enumerate(stds):
        start = pd.Timestamp("2025-01-06T00:00:00Z") + pd.Timedelta(days=7 * week)
        vals = [-std, std, 0.0, 0.0]
        for hour, value in enumerate(vals):
            rows.append(
                {
                    "split": split,
                    "target": target,
                    "target_display": "DA price",
                    "target_group": "DA price",
                    "target_time_utc": start + pd.Timedelta(hours=hour),
                    "y_true": value,
                    "week_start_utc": start,
                }
            )
    return pd.DataFrame(rows)


def test_typical_week_selection_closest_to_median_weekly_std_per_target() -> None:
    base = _weekly_base([1.0, 4.0, 9.0])
    selected = select_typical_weeks_algorithmic(base)
    row = selected.iloc[0]
    assert row["selection_type"] == "typical_week"
    assert pd.Timestamp(row["week_start_utc"]) == pd.Timestamp("2025-01-13T00:00:00Z")


def test_high_volatility_week_selection_highest_weekly_std_per_target() -> None:
    base = _weekly_base([1.0, 4.0, 9.0])
    selected = select_high_volatility_weeks_algorithmic(base)
    row = selected.iloc[0]
    assert row["selection_type"] == "high_volatility_week"
    assert pd.Timestamp(row["week_start_utc"]) == pd.Timestamp("2025-01-20T00:00:00Z")


def test_week_selection_preserves_split() -> None:
    base = pd.concat([_weekly_base([1.0], split="test"), _weekly_base([99.0], split="val")], ignore_index=True)
    selected = select_high_volatility_weeks_algorithmic(base)
    assert set(selected["split"]) == {"test", "val"}


def test_da_market_actionable_selector_uses_dminus1_1100_berlin() -> None:
    week = WeekSpec("typical_week", "Typical", pd.Timestamp("2025-06-02T00:00:00Z"))
    df = pd.DataFrame(
        {
            "forecast_time_utc": pd.to_datetime(["2025-06-01T09:00:00Z", "2025-06-01T10:00:00Z"], utc=True),
            "target_time_utc": pd.to_datetime(["2025-06-02T00:00:00Z", "2025-06-02T00:00:00Z"], utc=True),
            "lead_time_h": [15.0, 14.0],
            "model": ["xgb", "xgb"],
            "y_true": [1.0, 1.0],
            "p50": [1.0, 2.0],
        }
    )
    selected, _ = _select_market_actionable_rows(
        df,
        spec={"selection": "local_dminus1_snapshot", "forecast_hour": 11, "snapshot_description": "DA D-1 11:00"},
        week=week,
        window_hours=168,
    )
    assert len(selected) == 1
    assert selected["p50"].iloc[0] == 1.0


def test_bcm_market_actionable_selector_uses_dminus1_0800_berlin() -> None:
    week = WeekSpec("high_volatility_week", "High-volatility", pd.Timestamp("2025-06-02T00:00:00Z"))
    df = pd.DataFrame(
        {
            "forecast_time_utc": pd.to_datetime(["2025-06-01T06:00:00Z", "2025-06-01T09:00:00Z"], utc=True),
            "target_time_utc": pd.to_datetime(["2025-06-02T00:00:00Z", "2025-06-02T00:00:00Z"], utc=True),
            "lead_time_h": [18.0, 15.0],
            "model": ["xgb", "xgb"],
            "y_true": [1.0, 1.0],
            "p50": [1.0, 2.0],
        }
    )
    selected, _ = _select_market_actionable_rows(
        df,
        spec={"selection": "local_dminus1_snapshot", "forecast_hour": 8, "snapshot_description": "BCM D-1 08:00"},
        week=week,
        window_hours=168,
    )
    assert len(selected) == 1
    assert selected["p50"].iloc[0] == 1.0


def test_bem_market_actionable_selector_uses_h1() -> None:
    week = WeekSpec("typical_week", "Typical", pd.Timestamp("2025-06-02T00:00:00Z"))
    df = pd.DataFrame(
        {
            "forecast_time_utc": pd.to_datetime(["2025-06-01T23:00:00Z", "2025-06-01T22:00:00Z"], utc=True),
            "target_time_utc": pd.to_datetime(["2025-06-02T00:00:00Z", "2025-06-02T00:00:00Z"], utc=True),
            "lead_time_h": [1.0, 2.0],
            "model": ["xgb", "xgb"],
            "y_true": [1.0, 1.0],
            "p50": [1.0, 2.0],
        }
    )
    selected, _ = _select_market_actionable_rows(
        df,
        spec={"selection": "lead_h1", "lead_h": 1.0, "snapshot_description": "BEM h1"},
        week=week,
        window_hours=168,
    )
    assert len(selected) == 1
    assert selected["lead_time_h"].iloc[0] == 1.0
