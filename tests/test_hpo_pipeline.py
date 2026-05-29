from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_and_export_runs import (  # noqa: E402
    AFRR_TARGETS,
    DA_TARGET,
    hpo_override_cli_args,
    load_hpo_artifact_map,
    validate_hpo_cli_choice,
)


def _mk_hpo_json(path: Path, *, bundle: str, target: str) -> None:
    path.write_text(
        json.dumps(
            {
                "bundle": bundle,
                "target_col": target,
                "best_params": {"learning_rate": 0.1},
                "selection_metric": "tail_upper_mae",
                "best_objective_value": 1.23,
            }
        ),
        encoding="utf-8",
    )


def test_hpo_artifact_map_generation_creates_expected_mapping(tmp_path: Path) -> None:
    out_dir = tmp_path / "hpo"
    out_dir.mkdir(parents=True)
    for bundle, target in [("da", DA_TARGET), *[("afrr", t) for t in AFRR_TARGETS]]:
        _mk_hpo_json(out_dir / f"xgb_optuna_{bundle}_{target}.json", bundle=bundle, target=target)

    out_map = tmp_path / "xgb_map.json"
    cp = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(ROOT / "scripts" / "build_hpo_artifact_map.py"),
            "--model-type",
            "xgboost",
            "--hpo-out-dir",
            str(out_dir),
            "--out",
            str(out_map),
        ],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
    mp = load_hpo_artifact_map(out_map)
    assert mp[DA_TARGET].endswith(f"xgb_optuna_da_{DA_TARGET}.json")
    for t in AFRR_TARGETS:
        assert t in mp
        assert mp[t].endswith(f"xgb_optuna_afrr_{t}.json")


def test_missing_hpo_file_fails_in_full_hpo_mode(tmp_path: Path) -> None:
    out_dir = tmp_path / "hpo"
    out_dir.mkdir(parents=True)
    # Only DA file exists -> should fail.
    _mk_hpo_json(out_dir / f"xgb_optuna_da_{DA_TARGET}.json", bundle="da", target=DA_TARGET)
    out_map = tmp_path / "xgb_map.json"
    cp = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(ROOT / "scripts" / "build_hpo_artifact_map.py"),
            "--model-type",
            "xgboost",
            "--hpo-out-dir",
            str(out_dir),
            "--out",
            str(out_map),
        ],
        capture_output=True,
        text=True,
    )
    assert cp.returncode != 0


def test_train_and_export_runs_uses_target_specific_hpo_artifact_for_each_target() -> None:
    params_da = {"max_depth": 4}
    params_afrr = {"max_depth": 9}
    da_args = hpo_override_cli_args("xgboost", params_da)
    afrr_args = hpo_override_cli_args("xgboost", params_afrr)
    assert da_args != afrr_args
    assert "--max-depth" in da_args and "4" in da_args
    assert "--max-depth" in afrr_args and "9" in afrr_args


def test_da_hpo_artifact_not_reused_for_afrr_target() -> None:
    da_artifact = "artifacts/hpo/xgb_optuna_da_target_da_price.json"
    afrr_artifact = "artifacts/hpo/xgb_optuna_afrr_target_afrr_activation_price_vwap_pos.json"
    assert da_artifact != afrr_artifact


def test_backward_compatible_single_hpo_artifact_still_works_for_single_target() -> None:
    args = hpo_override_cli_args("linear", {"alpha": 0.123, "l1_ratio": 0.4})
    assert "--alpha" in args and "0.123" in args
    assert "--l1-ratio" in args and "0.4" in args


def test_hpo_artifact_and_map_conflict_fails() -> None:
    with pytest.raises(ValueError):
        validate_hpo_cli_choice("a.json", "map.json")


def test_makefile_dry_run_includes_afrr_hpo_commands_for_all_targets() -> None:
    cp = subprocess.run(
        ["make", "-n", "tune-xgb-all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
    out = cp.stdout
    assert "--bundle afrr --target-col \"$TGT\"" in out
    for t in AFRR_TARGETS:
        assert t in out


def test_makefile_dry_run_train_xgb_uses_hpo_map() -> None:
    cp = subprocess.run(
        ["make", "-n", "train-xgb"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "--hpo-artifact-map \"artifacts/hpo/xgb_hpo_artifact_map.json\"" in cp.stdout
