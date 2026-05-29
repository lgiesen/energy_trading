from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
from src.energy_trading.models.train_tft_export import _mean_pinball_from_decay  # noqa: E402
from scripts.tune_linear import _mean_pinball_loss as linear_mean_pinball_loss  # noqa: E402
from scripts.tune_xgboost import _build_cli as xgb_tune_build_cli  # noqa: E402
from scripts.tune_xgboost import _mean_pinball_loss as xgb_mean_pinball_loss  # noqa: E402


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
            sys.executable,
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
            sys.executable,
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
    assert "--hpo-artifact " not in cp.stdout


def test_makefile_dry_run_train_linear_uses_hpo_map_only() -> None:
    cp = subprocess.run(
        ["make", "-n", "train-linear"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "--hpo-artifact-map \"artifacts/hpo/linear_hpo_artifact_map.json\"" in cp.stdout
    assert "--hpo-artifact " not in cp.stdout


def test_makefile_dry_run_train_tft_uses_hpo_map_only() -> None:
    cp = subprocess.run(
        ["make", "-n", "train-tft"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "--hpo-artifact-map \"artifacts/hpo/tft_hpo_artifact_map.json\"" in cp.stdout
    assert "--hpo-artifact " not in cp.stdout


def test_makefile_default_hpo_trial_budgets() -> None:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "XGB_HPO_TRIALS ?= 30" in text
    assert "TFT_HPO_TRIALS ?= 6" in text


def test_makefile_dry_run_tuning_uses_unweighted_default_selection_metrics() -> None:
    cp = subprocess.run(
        ["make", "-n", "tune-all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
    out = cp.stdout
    assert "--selection-metric pinball_mean" in out
    assert "--selection-metric pinball_mean_val" in out
    assert "--hpo-quantiles 0.1,0.5,0.9" in out
    assert "--fallback-metric mae_val" in out
    assert "tail_upper_mae" not in out
    assert "leadtime_pinball_p90_val_weighted" not in out
    assert "--use-tail-weights" not in out


def test_pinball_helpers_perfect_predictions_are_zero() -> None:
    y = np.array([1.0, 2.0, 3.0], dtype=float)
    preds = {0.1: y.copy(), 0.5: y.copy(), 0.9: y.copy()}
    assert xgb_mean_pinball_loss(y, preds) == 0.0
    assert linear_mean_pinball_loss(y, preds) == 0.0


def test_pinball_helpers_worse_predictions_have_larger_loss() -> None:
    y = np.array([1.0, 2.0, 3.0], dtype=float)
    good = {0.1: y.copy(), 0.5: y.copy(), 0.9: y.copy()}
    bad = {0.1: y - 1.0, 0.5: y + 1.0, 0.9: y + 2.0}
    assert xgb_mean_pinball_loss(y, bad) > xgb_mean_pinball_loss(y, good)
    assert linear_mean_pinball_loss(y, bad) > linear_mean_pinball_loss(y, good)


def test_xgb_tune_defaults_pinball_mean_and_quantiles() -> None:
    parser = xgb_tune_build_cli()
    args = parser.parse_args([])
    assert args.selection_metric == "pinball_mean"
    assert args.hpo_quantiles == "0.1,0.5,0.9"
    assert args.use_tail_weights is False


def test_tft_pinball_mean_uses_full_horizon_not_h1_only() -> None:
    decay = pd.DataFrame(
        {
            "lead_time_h": [1.0, 2.0],
            "n": [10.0, 10.0],
            "pinball_p10": [1.0, 3.0],
            "pinball_p50": [2.0, 4.0],
            "pinball_p90": [3.0, 5.0],
        }
    )
    got = _mean_pinball_from_decay(decay)
    # Per-quantile lead-mean = [2,3,4], overall = 3
    assert got == pytest.approx(3.0)


def test_tft_pinball_mean_changes_when_non_h1_lead_changes() -> None:
    decay_a = pd.DataFrame(
        {
            "lead_time_h": [1.0, 2.0],
            "n": [10.0, 10.0],
            "pinball_p10": [1.0, 1.0],
            "pinball_p50": [1.0, 1.0],
            "pinball_p90": [1.0, 1.0],
        }
    )
    decay_b = decay_a.copy()
    decay_b.loc[decay_b["lead_time_h"] == 2.0, ["pinball_p10", "pinball_p50", "pinball_p90"]] = [5.0, 5.0, 5.0]
    assert _mean_pinball_from_decay(decay_b) > _mean_pinball_from_decay(decay_a)


def test_tft_pinball_mean_ignores_invalid_rows_and_requires_positive_n() -> None:
    decay = pd.DataFrame(
        {
            "lead_time_h": [1.0, 2.0, 3.0],
            "n": [10.0, 0.0, np.nan],
            "pinball_p10": [1.0, 99.0, 99.0],
            "pinball_p50": [2.0, np.inf, np.nan],
            "pinball_p90": [3.0, -np.inf, np.nan],
        }
    )
    got = _mean_pinball_from_decay(decay)
    # only lead 1 is valid -> mean(1,2,3)=2
    assert got == pytest.approx(2.0)
