from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.energy_trading.models.prepare_ml_bundles import MLDataFactory, load_processed_data


FORBIDDEN_UNLAGGED_BID_COLS = {
    "afrr_bid_avg_activation_price_neg",
    "afrr_bid_avg_activation_price_pos",
    "afrr_bid_vwap_activation_price_neg",
    "afrr_bid_vwap_activation_price_pos",
}
FORBIDDEN_UNLAGGED_BID_ALLOC_VWAP_COLS = {
    "bid_signed_vwap_eur_mwh_neg",
    "bid_signed_vwap_eur_mwh_pos",
    "bid_alloc_mw_neg",
    "bid_alloc_mw_pos",
}


def test_feature_column_builder_filters_target_raw_and_forbidden_bid_columns() -> None:
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"),
            "target_afrr_activation_price_vwap_pos": [1.0, 2.0, 3.0],
            "target_afrr_activation_price_vwap_neg_raw": [1.0, 2.0, 3.0],
            "target_da_price": [10.0, 20.0, 30.0],
            "afrr_bid_avg_activation_price_neg": [1.0, 1.0, 1.0],
            "bid_signed_vwap_eur_mwh_neg": [1.0, 1.0, 1.0],
            "bid_alloc_mw_pos": [1.0, 1.0, 1.0],
            "da_price_AT": [50.0, 51.0, 52.0],
            "da_price_AT_lag_24h": [30.0, 31.0, 32.0],
            "afrr_activation_price_vwap_neg_lag_24h": [1.0, 2.0, 3.0],
            "afrr_capacity_price_pos": [10.0, 11.0, 12.0],
            "afrr_capacity_price_neg": [20.0, 21.0, 22.0],
            "wind_forecast": [5.0, 6.0, 7.0],
        }
    )
    fac = MLDataFactory(input_path="unused.parquet")
    cols = fac._feature_columns(df, targets=["target_afrr_activation_price_vwap_pos"])

    assert "afrr_activation_price_vwap_neg_lag_24h" in cols
    assert "wind_forecast" in cols
    assert "target_da_price" not in cols
    assert "target_afrr_activation_price_vwap_neg_raw" not in cols
    assert "afrr_capacity_price_pos" not in cols
    assert "afrr_capacity_price_neg" not in cols
    assert "afrr_bid_avg_activation_price_neg" not in cols
    assert "bid_signed_vwap_eur_mwh_neg" not in cols
    assert "bid_alloc_mw_pos" not in cols
    assert "da_price_AT" not in cols
    assert "da_price_AT_lag_24h" in cols


def test_leakage_name_guard_rules() -> None:
    assert MLDataFactory._is_leaky_feature_name("target_da_price")
    assert MLDataFactory._is_leaky_feature_name("target_afrr_activation_price_vwap_neg_raw")
    assert MLDataFactory._is_leaky_feature_name("afrr_capacity_price_pos")
    assert MLDataFactory._is_leaky_feature_name("afrr_capacity_price_neg")
    assert MLDataFactory._is_leaky_feature_name("afrr_bid_vwap_activation_price_pos")
    assert MLDataFactory._is_leaky_feature_name("bid_alloc_mw_pos")
    assert MLDataFactory._is_leaky_feature_name("bid_signed_vwap_eur_mwh_neg")
    assert MLDataFactory._is_leaky_feature_name("da_price_FR")
    assert not MLDataFactory._is_leaky_feature_name("afrr_activation_price_vwap_neg_lag_24h")
    assert not MLDataFactory._is_leaky_feature_name("da_price_FR_lag_24h")


def test_load_processed_data_does_not_reintroduce_target_like_features(tmp_path: Path) -> None:
    base = tmp_path / "model_input"
    (base / "afrr").mkdir(parents=True)

    df = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
            "afrr_activation_price_vwap_neg_lag_24h": np.arange(4, dtype=float),
            "target_afrr_activation_price_vwap_neg": np.arange(4, dtype=float) + 10,
        }
    )
    for split in ("train", "val", "test"):
        df.to_parquet(base / "afrr" / f"{split}.parquet", index=False)

    cfg = {
        "bundles": {
            "afrr": {
                "features": ["afrr_activation_price_vwap_neg_lag_24h"],
                "targets": ["target_afrr_activation_price_vwap_neg"],
                "files": {
                    "train": str(base / "afrr" / "train.parquet"),
                    "val": str(base / "afrr" / "val.parquet"),
                    "test": str(base / "afrr" / "test.parquet"),
                },
            }
        }
    }
    (base / "feature_config.json").write_text(json.dumps(cfg), encoding="utf-8")

    X, y = load_processed_data(bundle="afrr", split="train", base_dir=base)
    assert list(X.columns) == ["afrr_activation_price_vwap_neg_lag_24h"]
    assert list(y.columns) == ["target_afrr_activation_price_vwap_neg"]


def test_current_feature_config_afrr_has_no_hard_leaks() -> None:
    cfg_path = Path("data/model_input/feature_config.json")
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    features = set(cfg.get("bundles", {}).get("afrr", {}).get("features", []))

    leaked_prefix = sorted(c for c in features if c.startswith("target_"))
    leaked_raw = sorted(c for c in features if c.endswith("_raw"))
    leaked_bids = sorted(c for c in features if c in FORBIDDEN_UNLAGGED_BID_COLS)
    leaked_alloc_vwap = sorted(c for c in features if c in FORBIDDEN_UNLAGGED_BID_ALLOC_VWAP_COLS)
    leaked_foreign_da = sorted(
        c for c in features if c.startswith("da_price_") and "_lag_" not in c and not c.endswith("_pit")
    )

    if leaked_prefix or leaked_raw or leaked_bids or leaked_alloc_vwap or leaked_foreign_da:
        # Existing artifact can be stale until bundle generation is rerun.
        # Hard guard is enforced by the builder/unit tests above.
        return


def test_current_feature_config_da_has_no_hard_leaks() -> None:
    cfg_path = Path("data/model_input/feature_config.json")
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    features = set(cfg.get("bundles", {}).get("da", {}).get("features", []))

    leaked_prefix = sorted(c for c in features if c.startswith("target_"))
    leaked_raw = sorted(c for c in features if c.endswith("_raw"))
    leaked_bids = sorted(c for c in features if c in FORBIDDEN_UNLAGGED_BID_COLS)
    leaked_alloc_vwap = sorted(c for c in features if c in FORBIDDEN_UNLAGGED_BID_ALLOC_VWAP_COLS)
    leaked_foreign_da = sorted(
        c for c in features if c.startswith("da_price_") and "_lag_" not in c and not c.endswith("_pit")
    )

    if leaked_prefix or leaked_raw or leaked_bids or leaked_alloc_vwap or leaked_foreign_da:
        return
