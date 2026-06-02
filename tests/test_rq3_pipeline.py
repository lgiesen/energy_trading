from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_rq3_pipeline as rp


def test_smoke_preset_defaults() -> None:
    ap = rp.build_arg_parser()
    args = ap.parse_args(["--preset", "smoke", "--simulation-root", "artifacts/sim"])
    models, policies, workers = rp._resolved_defaults(args)
    assert models == "xgb"
    assert policies == "p50-p50"
    assert workers == 1


def test_final_preset_defaults() -> None:
    ap = rp.build_arg_parser()
    args = ap.parse_args(["--preset", "final", "--simulation-root", "artifacts/sim"])
    models, policies, workers = rp._resolved_defaults(args)
    assert models == "xgb,tft,linear"
    assert "p10-p10" in policies
    assert workers == 3


def test_custom_requires_models_and_quantiles() -> None:
    ap = rp.build_arg_parser()
    args = ap.parse_args(["--preset", "custom", "--simulation-root", "artifacts/sim"])
    with pytest.raises(ValueError):
        rp._resolved_defaults(args)


def test_pipeline_writes_simulation_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run(cmd, stdout=None, stderr=None, text=None):  # noqa: ANN001
        calls.append({"cmd": cmd, "stdout": stdout})
        if stdout is not None:
            stdout.write("hello log\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_rq3_pipeline.py",
            "--preset", "smoke",
            "--simulation-root", str(tmp_path / "sim"),
            "--model-manifest", "xgb=artifacts/model_runs/latest_xgboost.json",
            "--export-afrr-bin-ev-audit",
            "--allow-invalid-output",
            "--no-strict-simulation-validity",
        ],
    )
    rp.main()
    assert (tmp_path / "sim" / "pipeline_simulation.log").exists()
    assert (tmp_path / "sim" / "pipeline_diagnostics.log").exists()
    assert calls[0]["cmd"][0] == sys.executable
    assert "--da-quantile-role" in calls[0]["cmd"]
    assert "--final-soc-mode" in calls[0]["cmd"]
    assert "--no-strict-simulation-validity" in calls[0]["cmd"]
    assert "--export-afrr-bin-ev-audit" in calls[0]["cmd"]
    assert "--allow-invalid-output" in calls[0]["cmd"]


def test_simulation_failure_writes_pipeline_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(cmd, stdout=None, stderr=None, text=None):  # noqa: ANN001
        if stdout is not None:
            stdout.write("Traceback (most recent call last):\nValueError: boom\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_rq3_pipeline.py",
            "--preset", "smoke",
            "--simulation-root", str(tmp_path / "sim"),
            "--model-manifest", "xgb=artifacts/model_runs/latest_xgboost.json",
        ],
    )
    assert rp.main() == 1
    assert (tmp_path / "sim" / "rq3_pipeline_manifest.json").exists()
    assert (tmp_path / "sim" / "pipeline_failed.md").exists()


def test_allow_partial_only_runs_diagnostics_if_reportable_results_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _fake_run(cmd, stdout=None, stderr=None, text=None):  # noqa: ANN001
        calls.append(list(cmd))
        if "run_rq3_simulations.py" in str(cmd):
            # write a minimal manifest so allow-partial can inspect it
            sim_root = Path(cmd[cmd.index("--simulation-root") + 1])
            (sim_root / "run_manifest_rq3.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "failure_class": "completed_reportable",
                                "thesis_reportable": 1,
                                "summary_path": str(sim_root / "x" / "backtest_summary.json"),
                                "hourly_path": str(sim_root / "x" / "backtest_hourly.csv"),
                                "scenario_output_dir": str(sim_root / "x"),
                                "quantile_policy": "p50-p50",
                                "model": "xgb",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
        if stdout is not None:
            stdout.write("ok\n")
        return SimpleNamespace(returncode=1 if "run_rq3_simulations.py" in str(cmd) else 0)

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_rq3_pipeline.py",
            "--preset", "smoke",
            "--simulation-root", str(tmp_path / "sim"),
            "--model-manifest", "xgb=artifacts/model_runs/latest_xgboost.json",
            "--allow-partial-results",
        ],
    )
    assert rp.main() == 1
    assert any("generate_strategy_diagnostics.py" in " ".join(c) for c in calls)
    assert (tmp_path / "sim" / "pipeline_diagnostics.log").exists()


def test_allow_partial_without_reportable_results_skips_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _fake_run(cmd, stdout=None, stderr=None, text=None):  # noqa: ANN001
        calls.append(list(cmd))
        if "run_rq3_simulations.py" in str(cmd):
            sim_root = Path(cmd[cmd.index("--simulation-root") + 1])
            (sim_root / "run_manifest_rq3.json").write_text(json.dumps({"results": []}), encoding="utf-8")
        if stdout is not None:
            stdout.write("boom\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_rq3_pipeline.py",
            "--preset", "smoke",
            "--simulation-root", str(tmp_path / "sim"),
            "--model-manifest", "xgb=artifacts/model_runs/latest_xgboost.json",
            "--allow-partial-results",
        ],
    )
    assert rp.main() == 1
    assert not any("generate_strategy_diagnostics.py" in " ".join(c) for c in calls)
