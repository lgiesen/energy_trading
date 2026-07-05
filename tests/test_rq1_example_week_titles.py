from __future__ import annotations

import pandas as pd

from scripts.build_rq1_example_weeks import (
    MARKET_ACTIONABLE_SPECS,
    WeekSpec,
    _market_actionable_caption,
    _market_actionable_short_caption,
    _market_actionable_subtitle,
    _market_actionable_target_title,
    _market_actionable_title,
)


def _spec(context: str) -> dict:
    for spec in MARKET_ACTIONABLE_SPECS:
        if spec["market_context"] == context:
            return spec
    raise AssertionError(f"Missing test spec: {context}")


def test_market_actionable_title_uses_short_target_and_p50() -> None:
    target = "target_afrr_activation_price_vwap_pos"
    title = _market_actionable_title(_spec("bem_h1"), _market_actionable_target_title(target))

    assert title == "aFRR Activation Price +: p50 Forecast"
    assert "|" not in title
    assert "Forecast Snapshot" not in title


def test_market_actionable_subtitle_uses_week_and_snapshot_rule() -> None:
    week = WeekSpec("high_volatility_week", "High-volatility", pd.Timestamp("2025-01-06T00:00:00Z"))
    subtitle = _market_actionable_subtitle(week, _spec("bcm_dplus1_08"), "target_afrr_capacity_price_pos")

    assert subtitle == "High-volatility week | BCM D−1 08:00 Europe/Berlin forecast snapshot"


def test_market_actionable_caption_contains_required_context() -> None:
    typical = WeekSpec("typical_week", "Typical", pd.Timestamp("2025-01-06T00:00:00Z"))
    high_vol = WeekSpec("high_volatility_week", "High-volatility", pd.Timestamp("2025-01-06T00:00:00Z"))

    typical_caption = _market_actionable_caption(typical, _spec("da_dminus1_11"), "target_da_price")
    high_vol_caption = _market_actionable_caption(high_vol, _spec("bem_h1"), "target_afrr_activation_price_vwap_pos")

    assert "typical week" in typical_caption
    assert "DA D$-1$ 11:00 Europe/Berlin forecast snapshot" in typical_caption
    assert "high-volatility week" in high_vol_caption
    assert "BEM h1 forecast" in high_vol_caption
    for caption in [typical_caption, high_vol_caption]:
        assert "realized values" in caption
        assert "$p50$ forecasts" in caption
        assert "RLQR, XGB and TFT" in caption


def test_market_actionable_short_caption_is_concise() -> None:
    assert _market_actionable_short_caption("target_afrr_capacity_price_neg") == "Example-week aFRR capacity price negative forecasts"
