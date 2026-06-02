from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_models_csv(models: str) -> list[str]:
    return [m.strip() for m in str(models).split(",") if m.strip()]


def parse_quantile_policies_csv(policies: str) -> list[str]:
    return [p.strip() for p in str(policies).split(",") if p.strip()]


def parse_model_manifest_items(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_qpair(name: str) -> tuple[str, str]:
    m = re.search(r"(p\d{2})[-_](p\d{2})", str(name))
    if m:
        return m.group(1), m.group(2)
    return "", ""


def build_scenario_out_dir(simulation_root: Path, model: str, strategy: str, quantile_policy: str) -> Path:
    return simulation_root / f"{model}_{strategy}_{quantile_policy}"


def _find_nested_artifact_pair(out_dir: Path) -> tuple[Path | None, Path | None, Path]:
    """Find the deepest summary/hourly pair beneath out_dir."""
    candidates: list[tuple[int, Path, Path, Path]] = []
    for summary_path in out_dir.rglob("backtest_summary.json"):
        scenario_dir = summary_path.parent
        hourly_parq = scenario_dir / "backtest_hourly.parquet"
        hourly_csv = scenario_dir / "backtest_hourly.csv"
        hourly_path = hourly_parq if hourly_parq.exists() else (hourly_csv if hourly_csv.exists() else None)
        if hourly_path is None:
            continue
        depth = len(scenario_dir.relative_to(out_dir).parts) if scenario_dir != out_dir else 0
        candidates.append((depth, summary_path, hourly_path, scenario_dir))
    if not candidates:
        return None, None, out_dir
    candidates.sort(key=lambda x: (x[0], str(x[1].parent)), reverse=True)
    _, summary_path, hourly_path, scenario_dir = candidates[0]
    return summary_path, hourly_path, scenario_dir


def _tail_file(path: Path, n_lines: int = 120) -> str:
    if not path.exists():
        return ""
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = txt.splitlines()
    return "\n".join(lines[-n_lines:]) if lines else ""


def _extract_last_traceback(log_text: str) -> dict[str, str]:
    lines = str(log_text).splitlines()
    traceback_tail = ""
    exception_type = ""
    exception_message = ""
    for i in range(len(lines) - 1, -1, -1):
        if "Traceback (most recent call last):" in lines[i]:
            tb = lines[i:]
            traceback_tail = "\n".join(tb)
            for ln in reversed(tb):
                m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", ln.strip())
                if m:
                    exception_type = m.group(1)
                    exception_message = m.group(2)
                    break
            break
    return {
        "traceback_tail": traceback_tail,
        "exception_type": exception_type,
        "exception_message": exception_message,
    }


def inspect_run_artifacts(rec: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(str(rec.get("out_dir", "")))
    summary_path = out_dir / "backtest_summary.json"
    hourly_parq = out_dir / "backtest_hourly.parquet"
    hourly_csv = out_dir / "backtest_hourly.csv"
    hourly_path = hourly_parq if hourly_parq.exists() else (hourly_csv if hourly_csv.exists() else None)
    scenario_output_dir = out_dir
    if summary_path.exists() and hourly_path is not None:
        scenario_output_dir = out_dir
    else:
        nested_summary, nested_hourly, nested_dir = _find_nested_artifact_pair(out_dir)
        if nested_summary is not None and nested_hourly is not None:
            summary_path = nested_summary
            hourly_path = nested_hourly
            scenario_output_dir = nested_dir
    summary_exists = summary_path.exists()
    hourly_exists = hourly_path is not None and Path(hourly_path).exists()
    artifacts_exist = bool(summary_exists or hourly_exists)

    simulation_valid = np.nan
    thesis_reportable = np.nan
    invalid_reason = ""
    if summary_exists:
        try:
            sm = json.loads(summary_path.read_text(encoding="utf-8"))
            simulation_valid = float(pd.to_numeric(pd.Series([sm.get("simulation_valid", np.nan)]), errors="coerce").iloc[0])
            thesis_reportable = float(pd.to_numeric(pd.Series([sm.get("thesis_reportable", np.nan)]), errors="coerce").iloc[0])
            invalid_reason = str(sm.get("invalid_reason", "") or "")
        except Exception:
            pass

    status = str(rec.get("status", "unknown"))
    exit_code = int(rec.get("exit_code", 1))
    if status == "reused":
        failure_class = "reused_reportable" if np.isfinite(thesis_reportable) and thesis_reportable >= 0.5 else "reused_nonreportable"
    elif exit_code == 0 and np.isfinite(thesis_reportable) and thesis_reportable >= 0.5:
        failure_class = "completed_reportable"
    elif exit_code == 0 and artifacts_exist:
        failure_class = "completed_nonreportable"
    elif exit_code != 0 and summary_exists and np.isfinite(thesis_reportable) and thesis_reportable < 0.5:
        failure_class = "strict_invalid_with_artifacts"
    elif exit_code != 0 and artifacts_exist:
        failure_class = "failed_with_artifacts"
    elif exit_code < 0:
        failure_class = "process_crash"
    else:
        failure_class = "failed_without_artifacts"

    log_path = Path(str(rec.get("log_path", "")))
    log_tail = _tail_file(log_path)
    trace = _extract_last_traceback(log_tail)

    suspected_stage = "unknown"
    lt = log_tail.lower()
    if "performance metric reconciliation failed in strict mode" in lt and artifacts_exist:
        failure_class = "strict_metric_reconciliation_failed_with_artifacts"
        suspected_stage = "performance_metric_reconciliation"
    elif "run_battery_backtest.py" in lt:
        suspected_stage = "run_battery_backtest"
    elif "strict validity" in lt:
        suspected_stage = "strict_validity_output_guard"
    elif "forecast warehouse" in lt:
        suspected_stage = "forecast_loading_or_early_run_stage"

    return {
        **rec,
        "job_out_dir": str(out_dir),
        "scenario_output_dir": str(scenario_output_dir),
        "summary_path": str(summary_path),
        "hourly_path": "" if hourly_path is None else str(hourly_path),
        "artifacts_exist": artifacts_exist,
        "summary_exists": summary_exists,
        "hourly_exists": hourly_exists,
        "simulation_valid": simulation_valid,
        "thesis_reportable": thesis_reportable,
        "invalid_reason": invalid_reason,
        "failure_class": failure_class,
        "log_tail": log_tail,
        "suspected_failure_stage": suspected_stage,
        **trace,
    }


def write_failed_run_reports(simulation_root: Path, failed: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    csv_path = simulation_root / "failed_runs.csv"
    json_path = simulation_root / "failed_runs.json"
    md_path = simulation_root / "failed_runs.md"
    pd.DataFrame(failed).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Failed RQ3 Simulation Runs", ""]
    for fr in failed:
        lines += [
            f"## {fr.get('model','')} | {fr.get('strategy','')} | {fr.get('quantile_policy','')}",
            "",
            f"- Exit code: {fr.get('exit_code')}",
            f"- Failure class: {fr.get('failure_class')}",
            f"- Invalid reason: {fr.get('invalid_reason','')}",
            f"- Output dir: {fr.get('out_dir','')}",
            f"- Log path: {fr.get('log_path','')}",
            f"- Exception: {fr.get('exception_type','')}: {fr.get('exception_message','')}",
            "",
            "### Log tail",
            "```text",
            str(fr.get("traceback_tail", "") or fr.get("log_tail", "")),
            "```",
            "",
        ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, json_path, md_path


def run_logged_subprocess(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        cp = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return int(cp.returncode)
