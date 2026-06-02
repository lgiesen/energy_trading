from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_battery_backtest import _matches_model_key, _resolve_long_prediction_path  # noqa: E402


def _write_long_predictions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "target_time_utc": pd.date_range("2025-05-01", periods=2, freq="h", tz="UTC"),
            "lead_time_h": [1, 1],
            "p50": [1.0, 2.0],
            "predicted_value": [1.0, 2.0],
        }
    ).to_parquet(path, index=False)


def _write_manifest(path: Path, pred_rel: str) -> None:
    payload = {
        "run_id": path.parent.name,
        "bundles": {
            "da": {"predictions_long": {"test": {"pred_da_price": pred_rel}}},
            "afrr": {"predictions_long": {"test": {}}},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_tft_exact_manifest_path_without_tft_in_filename_resolves(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "model_runs" / "tft_123"
    pred = run_dir / "predictions" / "da_target_da_price_pred_da_price_long_test.parquet"
    _write_long_predictions(pred)
    _write_manifest(run_dir / "manifest.json", "predictions/da_target_da_price_pred_da_price_long_test.parquet")

    resolved = _resolve_long_prediction_path(
        pred_col="pred_da_price",
        configured_path="predictions/da_target_da_price_pred_da_price_long_test.parquet",
        manifest_dir=run_dir,
        split="test",
        model_key="tft",
    )
    assert resolved == pred


def test_model_token_may_be_in_parent_directory_for_matching(tmp_path: Path) -> None:
    path = tmp_path / "artifacts" / "model_runs" / "tft_123" / "predictions" / "da_target_da_price_pred_da_price_long_test.parquet"
    _write_long_predictions(path)
    assert _matches_model_key(path, "tft")


def test_exact_configured_path_bypasses_model_token_filename_matching(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "model_runs" / "tft_123"
    pred = run_dir / "predictions" / "da_target_da_price_pred_da_price_long_test.parquet"
    _write_long_predictions(pred)

    resolved = _resolve_long_prediction_path(
        pred_col="pred_da_price",
        configured_path=pred,
        manifest_dir=run_dir,
        split="test",
        model_key="tft",
    )
    assert resolved == pred


def test_wrong_model_fallback_is_still_rejected(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "model_runs"
    tft_dir = root / "tft_123"
    tft_dir.mkdir(parents=True, exist_ok=True)
    pred_xgb = tft_dir / "predictions" / "xgb_pred_da_price_long_test.parquet"
    _write_long_predictions(pred_xgb)

    try:
        _resolve_long_prediction_path(
            pred_col="pred_da_price",
            configured_path="predictions/missing.parquet",
            manifest_dir=tft_dir,
            split="test",
            model_key="tft",
        )
    except FileNotFoundError as exc:
        msg = str(exc)
        assert "fallback_rejected_by_model_key" in msg
        assert str(pred_xgb) in msg
    else:
        raise AssertionError("Expected FileNotFoundError for wrong-model fallback candidate")
