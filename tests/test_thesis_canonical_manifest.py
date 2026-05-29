from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_combined_manifest_marks_activation_neg_canonical(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    new = tmp_path / "new.json"
    out = tmp_path / "combined.json"
    base.write_text(
        json.dumps(
            {
                "bundles": {"afrr": {"predictions_long": {"test": {"pred_afrr_activation_price_neg": "old.parquet"}}}},
                "simulation": {},
            }
        )
    )
    new.write_text(
        json.dumps(
            {
                "bundles": {"afrr": {"predictions_long": {"test": {"pred_afrr_activation_price_neg": "new.parquet"}}}},
            }
        )
    )
    cp = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(ROOT / "scripts" / "create_thesis_canonical_manifest.py"),
            "--base-manifest",
            str(base),
            "--canonical-neg-manifest",
            str(new),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
    payload = json.loads(out.read_text())
    assert payload["target_value_mode"]["pred_afrr_activation_price_neg"] == "canonical_economic"
    assert "pred_afrr_activation_price_neg" in payload["simulation"]["canonical_economic_targets"]
    assert payload["bundles"]["afrr"]["predictions_long"]["test"]["pred_afrr_activation_price_neg"] == "new.parquet"


def test_combined_manifest_keeps_unaffected_targets_unchanged(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    new = tmp_path / "new.json"
    out = tmp_path / "combined.json"
    base.write_text(
        json.dumps(
            {
                "bundles": {
                    "afrr": {
                        "predictions_long": {
                            "test": {
                                "pred_afrr_activation_price_neg": "old.parquet",
                                "pred_afrr_capacity_price_pos": "cap_pos_old.parquet",
                            }
                        }
                    }
                },
                "simulation": {},
            }
        )
    )
    new.write_text(
        json.dumps(
            {
                "bundles": {"afrr": {"predictions_long": {"test": {"pred_afrr_activation_price_neg": "new.parquet"}}}},
            }
        )
    )
    cp = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(ROOT / "scripts" / "create_thesis_canonical_manifest.py"),
            "--base-manifest",
            str(base),
            "--canonical-neg-manifest",
            str(new),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
    payload = json.loads(out.read_text())
    assert payload["bundles"]["afrr"]["predictions_long"]["test"]["pred_afrr_capacity_price_pos"] == "cap_pos_old.parquet"


def test_combined_manifest_replaces_only_activation_neg(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    new = tmp_path / "new.json"
    out = tmp_path / "combined.json"
    base.write_text(
        json.dumps(
            {
                "bundles": {
                    "afrr": {
                        "predictions_long": {
                            "test": {
                                "pred_afrr_activation_price_neg": "old_neg.parquet",
                                "pred_afrr_activation_price_pos": "old_pos.parquet",
                                "pred_afrr_capacity_price_neg": "old_cap_neg.parquet",
                            }
                        }
                    }
                },
                "simulation": {},
            }
        )
    )
    new.write_text(
        json.dumps(
            {
                "bundles": {
                    "afrr": {
                        "predictions_long": {
                            "test": {
                                "pred_afrr_activation_price_neg": "new_neg.parquet",
                                "pred_afrr_activation_price_pos": "new_pos_should_not_be_used.parquet",
                            }
                        }
                    }
                },
            }
        )
    )
    cp = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(ROOT / "scripts" / "create_thesis_canonical_manifest.py"),
            "--base-manifest",
            str(base),
            "--canonical-neg-manifest",
            str(new),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
    payload = json.loads(out.read_text())
    got = payload["bundles"]["afrr"]["predictions_long"]["test"]
    assert got["pred_afrr_activation_price_neg"] == "new_neg.parquet"
    assert got["pred_afrr_activation_price_pos"] == "old_pos.parquet"
    assert got["pred_afrr_capacity_price_neg"] == "old_cap_neg.parquet"
