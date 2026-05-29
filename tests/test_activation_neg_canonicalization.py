from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from energy_trading.features.build_features import add_explicit_targets, engineer_targets  # noqa: E402
from energy_trading.processing.refine_market_data import canonicalize_activation_price_neg  # noqa: E402
from scripts.train_and_export_runs import AFRR_TARGETS, filter_afrr_targets  # noqa: E402


def test_activation_price_neg_raw_preserved_and_canonical_positive() -> None:
    df = pl.DataFrame({"afrr_activation_price_vwap_neg": [-100.0, -20.0, 5.0]})
    out, stats = canonicalize_activation_price_neg(df)
    assert "afrr_activation_price_vwap_neg_raw" in out.columns
    assert out["afrr_activation_price_vwap_neg_raw"].to_list() == [-100.0, -20.0, 5.0]
    assert out["afrr_activation_price_vwap_neg"].to_list() == [100.0, 20.0, -5.0]
    assert stats["raw_positive_count"] == 1.0


def test_activation_price_neg_transform_uses_minus_raw_not_abs() -> None:
    df = pl.DataFrame({"afrr_activation_price_vwap_neg": [5.0]})
    out, _ = canonicalize_activation_price_neg(df)
    assert float(out["afrr_activation_price_vwap_neg"][0]) == -5.0


def test_activation_price_neg_lags_use_canonical_values() -> None:
    ts = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    df = pl.DataFrame(
        {
            "timestamp_utc": ts,
            "afrr_activation_price_vwap_pos": [10.0, 12.0, 14.0, 16.0],
            "afrr_activation_price_vwap_neg": [100.0, 200.0, 300.0, 400.0],
            "afrr_activated_mwh_pos": [1.0, 1.0, 1.0, 1.0],
            "afrr_activated_mwh_neg": [1.0, 1.0, 1.0, 1.0],
            "da_price": [50.0, 51.0, 52.0, 53.0],
            "activation_rate_ml_pos": [0.1, 0.1, 0.1, 0.1],
            "activation_rate_ml_neg": [0.2, 0.2, 0.2, 0.2],
            "afrr_capacity_price_pos": [1.0, 1.0, 1.0, 1.0],
            "afrr_capacity_price_neg": [2.0, 2.0, 2.0, 2.0],
        }
    )
    out = add_explicit_targets(engineer_targets(df))
    # target is shift(-1) from canonical unlagged source.
    assert float(out["target_afrr_activation_price_vwap_neg"][0]) == 200.0


def test_train_orchestrator_filters_to_single_target() -> None:
    out = filter_afrr_targets(AFRR_TARGETS, "target_afrr_activation_price_vwap_neg")
    assert out == ["target_afrr_activation_price_vwap_neg"]


def test_unaffected_targets_not_retrained_when_target_filter_used() -> None:
    out = filter_afrr_targets(AFRR_TARGETS, "target_afrr_activation_price_vwap_neg")
    assert "target_afrr_capacity_price_neg" not in out
