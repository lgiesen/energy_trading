from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_battery_backtest import (  # noqa: E402
    _manifest_can_resolve_long_predictions,
    _matches_model_key,
    _normalize_model_choice,
    _preflight_manifest_and_quantiles,
    _resolve_long_prediction_path,
    _resolve_model_manifest,
    parse_args,
)


def _write_long_predictions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "target_time_utc": pd.date_range("2025-05-01", periods=4, freq="h", tz="UTC"),
            "lead_time_h": [1, 1, 1, 1],
            "p50": [1.0, 2.0, 3.0, 4.0],
            "predicted_value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    df.to_parquet(path, index=False)


def _write_manifest(path: Path, pred_rel: str, *, run_id: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": run_id,
        "bundles": {
            "da": {"predictions_long": {"val": {"pred_da_price": pred_rel}, "test": {"pred_da_price": pred_rel}}},
            "afrr": {"predictions_long": {"val": {}, "test": {}}},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_model_alias_normalization() -> None:
    assert _normalize_model_choice("xgb") == ("xgb", "latest_xgboost.json")
    assert _normalize_model_choice("xgboost") == ("xgb", "latest_xgboost.json")
    assert _normalize_model_choice("tft") == ("tft", "latest_tft.json")
    assert _normalize_model_choice("linear") == ("linear", "latest_linear.json")
    assert _normalize_model_choice("rlqr") == ("linear", "latest_linear.json")


def test_cli_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_battery_backtest.py"])
    args = parse_args()
    assert args.split == "test"
    assert args.trading_strategy == "multi"
    assert args.da_quantile_role == "mid"
    assert args.quantile_pairs == "p50-p50"
    assert args.strict_simulation_validity is True
    assert args.final_soc_mode == "hard"
    assert args.clean_output is True


def test_latest_pointer_resolution(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "model_runs"
    run_dir = root / "xgb_123"
    pred = run_dir / "predictions" / "xgboost_da_val_pred_da_price_long.parquet"
    _write_long_predictions(pred)
    _write_manifest(run_dir / "manifest.json", "predictions/xgboost_da_val_pred_da_price_long.parquet", run_id="xgb_123")
    latest = root / "latest_xgboost.json"
    latest.write_text(json.dumps({"manifest_path": "xgb_123/manifest.json", "run_id": "xgb_123"}), encoding="utf-8")

    resolved_path, payload, run_id = _resolve_model_manifest(
        run_manifest_arg="",
        run_id=None,
        model_key="xgb",
        split="val",
        model_runs_root=root,
    )
    assert resolved_path == run_dir / "manifest.json"
    assert run_id == "xgb_123"
    usable, issues = _manifest_can_resolve_long_predictions(payload, resolved_path.parent, "val", "xgb", {"p50"})
    assert usable, issues


def test_copied_latest_manifest_resolution_switches_to_actual_run(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "model_runs"
    run_dir = root / "xgb_123"
    pred = run_dir / "predictions" / "xgboost_da_val_pred_da_price_long.parquet"
    _write_long_predictions(pred)
    _write_manifest(run_dir / "manifest.json", "predictions/xgboost_da_val_pred_da_price_long.parquet", run_id="xgb_123")
    copied_latest = root / "latest_xgboost.json"
    copied_latest.write_text(
        json.dumps(
            {
                "run_id": "xgb_123",
                "bundles": {
                    "da": {"predictions_long": {"val": {"pred_da_price": "predictions/xgboost_da_val_pred_da_price_long.parquet"}}},
                    "afrr": {"predictions_long": {"val": {}}},
                },
            }
        ),
        encoding="utf-8",
    )

    resolved_path, _, _ = _resolve_model_manifest(
        run_manifest_arg=str(copied_latest),
        run_id=None,
        model_key="xgb",
        split="val",
        model_runs_root=root,
    )
    assert resolved_path == run_dir / "manifest.json"


def test_fallback_scan_chooses_newest_usable_manifest(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "model_runs"
    older = root / "xgb_older"
    newer = root / "xgb_newer"
    _write_long_predictions(older / "predictions" / "xgboost_da_val_pred_da_price_long.parquet")
    _write_manifest(older / "manifest.json", "predictions/xgboost_da_val_pred_da_price_long.parquet", run_id="xgb_older")
    _write_long_predictions(newer / "predictions" / "xgboost_da_val_pred_da_price_long.parquet")
    _write_manifest(newer / "manifest.json", "predictions/xgboost_da_val_pred_da_price_long.parquet", run_id="xgb_newer")
    os.utime(older / "manifest.json", (1, 1))
    os.utime(newer / "manifest.json", None)

    resolved_path, _, run_id = _resolve_model_manifest(
        run_manifest_arg="",
        run_id=None,
        model_key="xgb",
        split="val",
        model_runs_root=root,
    )
    assert resolved_path == newer / "manifest.json"
    assert run_id == "xgb_newer"


def test_strict_model_matching() -> None:
    assert _matches_model_key(Path("xgboost_da_val_pred_da_price_long.parquet"), "xgb")
    assert not _matches_model_key(Path("tft_da_val_pred_da_price_long.parquet"), "xgb")
    assert _matches_model_key(Path("tft_afrr_val_pred_afrr_activation_price_pos_long.parquet"), "tft")
    assert not _matches_model_key(Path("xgboost_afrr_val_pred_afrr_activation_price_pos_long.parquet"), "tft")
    assert _matches_model_key(Path("linear_da_val_pred_da_price_long.parquet"), "linear")
    assert _matches_model_key(Path("rlqr_da_val_pred_da_price_long.parquet"), "linear")
    assert not _matches_model_key(Path("tft_da_val_pred_da_price_long.parquet"), "linear")


def test_error_message_lists_manifest_context_and_candidates(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "model_runs"
    run_dir = root / "xgb_123"
    manifest_path = run_dir / "manifest.json"
    payload = _write_manifest(manifest_path, "predictions/xgboost_da_val_pred_da_price_long.parquet", run_id="xgb_123")
    with pytest.raises(FileNotFoundError) as exc:
        _preflight_manifest_and_quantiles(
            manifest_path=manifest_path,
            manifest_payload=payload,
            split="val",
            model_key="xgb",
            manifest_dir=run_dir,
            expected_quantiles={"p50"},
        )
    msg = str(exc.value)
    assert "resolved_manifest_path=" in msg
    assert "manifest_dir=" in msg
    assert "configured path" in msg.lower() or "Configured path" in msg
    assert "model_key=xgb" in msg
    assert "split=val" in msg
    assert "Tried:" in msg


def test_end_to_end_manifest_preflight_smoke(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "model_runs"
    run_dir = root / "xgb_123"
    pred = run_dir / "predictions" / "xgboost_da_val_pred_da_price_long.parquet"
    _write_long_predictions(pred)
    payload = _write_manifest(run_dir / "manifest.json", "predictions/xgboost_da_val_pred_da_price_long.parquet", run_id="xgb_123")

    _preflight_manifest_and_quantiles(
        manifest_path=run_dir / "manifest.json",
        manifest_payload=payload,
        split="val",
        model_key="xgb",
        manifest_dir=run_dir,
        expected_quantiles={"p50"},
    )


def test_model_tft_does_not_pick_xgb_file(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "xgb_123"
    pred_dir = manifest_dir / "predictions"
    _write_long_predictions(pred_dir / "xgboost_da_val_pred_da_price_long.parquet")
    with pytest.raises(FileNotFoundError):
        _resolve_long_prediction_path(
            pred_col="pred_da_price",
            configured_path="predictions/xgboost_da_val_pred_da_price_long.parquet",
            manifest_dir=manifest_dir,
            split="val",
            model_key="tft",
        )
