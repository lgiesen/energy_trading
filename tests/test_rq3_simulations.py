from __future__ import annotations

import json
import sys
from pathlib import Path
import subprocess

import pandas as pd
import pytest

import scripts.run_rq3_simulations as rs


def test_builds_one_command_per_model_quantile(tmp_path: Path) -> None:
    ap = rs.build_arg_parser()
    args = ap.parse_args(
        [
            "--simulation-root", str(tmp_path / "sim"),
            "--models", "xgb,tft",
            "--model-manifest", "xgb=tests/test_rq3_simulations.py",
            "--model-manifest", "tft=tests/test_rq3_simulations.py",
            "--quantile-policies", "p10-p10,p50-p50",
        ]
    )
    cmds = rs._build_commands(args, {"xgb": "tests/test_rq3_simulations.py", "tft": "tests/test_rq3_simulations.py"})
    assert len(cmds) == 4
    assert len({c["out_dir"] for c in cmds}) == 4
    assert cmds[0]["command"][0] == sys.executable


def test_missing_manifest_fails_before_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_rq3_simulations.py",
            "--simulation-root", str(tmp_path / "sim"),
            "--models", "xgb,tft",
            "--model-manifest", "xgb=tests/test_rq3_simulations.py",
            "--quantile-policies", "p50-p50",
        ],
    )
    with pytest.raises(ValueError, match="Missing manifests for models"):
        rs.main()


def test_overwrite_adds_clean_output(tmp_path: Path) -> None:
    ap = rs.build_arg_parser()
    args = ap.parse_args(
        [
            "--simulation-root", str(tmp_path / "sim"),
            "--models", "xgb",
            "--model-manifest", "xgb=tests/test_rq3_simulations.py",
            "--quantile-policies", "p50-p50",
            "--overwrite",
        ]
    )
    cmds = rs._build_commands(args, {"xgb": "tests/test_rq3_simulations.py"})
    assert "--clean-output" in cmds[0]["command"]


def test_reuse_existing_marks_reused(tmp_path: Path) -> None:
    out = tmp_path / "sim" / "xgb_multi_p50-p50"
    nested = out / "multi" / "p50_p50"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "backtest_summary.json").write_text(
        json.dumps({"simulation_valid": 1, "thesis_reportable": 1, "invalid_reason": ""}),
        encoding="utf-8",
    )
    pd.DataFrame({"timestamp_utc": ["2025-01-01T00:00:00Z"], "real_pnl_eur": [0.0]}).to_csv(nested / "backtest_hourly.csv", index=False)
    ap = rs.build_arg_parser()
    args = ap.parse_args(
        [
            "--simulation-root", str(tmp_path / "sim"),
            "--models", "xgb",
            "--model-manifest", "xgb=tests/test_rq3_simulations.py",
            "--quantile-policies", "p50-p50",
            "--reuse-existing",
        ]
    )
    cmds = rs._build_commands(args, {"xgb": "tests/test_rq3_simulations.py"})
    assert cmds[0]["status"] == "reused"


def test_nested_artifact_inspection_finds_deepest_pair(tmp_path: Path) -> None:
    out = tmp_path / "sim" / "xgb_multi_p50-p50"
    nested = out / "multi" / "p50_p50"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "backtest_summary.json").write_text(
        json.dumps({"simulation_valid": 1, "thesis_reportable": 1, "invalid_reason": ""}),
        encoding="utf-8",
    )
    pd.DataFrame({"timestamp_utc": ["2025-01-01T00:00:00Z"], "real_pnl_eur": [1.0]}).to_csv(nested / "backtest_hourly.csv", index=False)
    rec = rs.inspect_run_artifacts(
        {
            "model": "xgb",
            "strategy": "multi",
            "quantile_policy": "p50-p50",
            "out_dir": str(out),
            "log_path": str(out / "log.txt"),
            "status": "completed",
            "exit_code": 0,
            "cmd": [],
        }
    )
    assert rec["scenario_output_dir"] == str(nested)
    assert rec["summary_path"] == str(nested / "backtest_summary.json")
    assert rec["hourly_path"] == str(nested / "backtest_hourly.csv")
    assert rec["failure_class"] == "completed_reportable"


def test_preflight_missing_manifest_writes_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = tmp_path / "sim"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_rq3_simulations.py",
            "--simulation-root", str(sim),
            "--models", "xgb,tft",
            "--model-manifest", "xgb=tests/test_rq3_simulations.py",
            "--quantile-policies", "p50-p50",
        ],
    )
    with pytest.raises(ValueError):
        rs.main()
    assert (sim / "run_manifest_rq3.json").exists()
    assert (sim / "run_manifest_rq3.csv").exists()
    assert (sim / "failed_runs.csv").exists()
    assert (sim / "failed_runs.json").exists()
    assert (sim / "failed_runs.md").exists()


def test_help_smoke_works_via_file_path() -> None:
    cp = subprocess.run([sys.executable, "scripts/run_rq3_simulations.py", "--help"], capture_output=True, text=True)
    assert cp.returncode == 0


def test_inspect_run_artifacts_classifies_strict_metric_reconciliation_failure(tmp_path: Path) -> None:
    out = tmp_path / "sim" / "xgb_multi_p50-p70"
    nested = out / "multi" / "p50_p70"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "backtest_summary.json").write_text(
        json.dumps({"simulation_valid": 0, "thesis_reportable": 0, "invalid_reason": "performance_metric_reconciliation"}),
        encoding="utf-8",
    )
    pd.DataFrame({"timestamp_utc": ["2025-01-01T00:00:00Z"], "real_pnl_eur": [0.0]}).to_csv(nested / "backtest_hourly.csv", index=False)
    log_path = out / "log.txt"
    log_path.write_text(
        "RuntimeError: Performance metric reconciliation failed in strict mode: {}\n",
        encoding="utf-8",
    )
    rec = rs.inspect_run_artifacts(
        {
            "model": "xgb",
            "strategy": "multi",
            "quantile_policy": "p50-p70",
            "out_dir": str(out),
            "log_path": str(log_path),
            "status": "completed",
            "exit_code": 1,
            "cmd": [],
        }
    )
    assert rec["failure_class"] == "strict_metric_reconciliation_failed_with_artifacts"
    assert rec["suspected_failure_stage"] == "performance_metric_reconciliation"
