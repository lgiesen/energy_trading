from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _read_summary(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_log_path(root: Path, summary_path: Path) -> Path | None:
    scenario_dir = summary_path.parent
    candidates = [
        scenario_dir / "run.log",
        root / "logs" / f"{scenario_dir.parent.name}_{scenario_dir.name}.log",
        root / "logs" / f"{scenario_dir.name}.log",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _read_log_text(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _extract_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _extract_last_int_pair(pattern: str, text: str) -> int | None:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    try:
        return int(matches[-1][0])
    except Exception:
        return None


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return None if pd.isna(out) else out


def _simulated_days_from_args(summary: dict[str, object]) -> float | None:
    raw = summary.get("command_line_args")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        args = json.loads(raw)
    except Exception:
        return None
    start = pd.to_datetime(args.get("start"), utc=True, errors="coerce")
    end = pd.to_datetime(args.get("end"), utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return None
    return float((end - start).total_seconds() / 86400.0)


def _infer_model(summary: dict[str, object], summary_path: Path) -> str:
    raw = summary.get("command_line_args")
    if isinstance(raw, str) and raw.strip():
        try:
            args = json.loads(raw)
            model = str(args.get("model") or args.get("model_key") or "").strip()
            if model:
                return model
        except Exception:
            pass
    parts = [p.name.lower() for p in summary_path.parents]
    for token in ("xgb", "tft", "linear", "rlqr"):
        if any(token in part for part in parts):
            return "linear" if token == "rlqr" else token
    return ""


def _infer_quantile(summary: dict[str, object], summary_path: Path) -> str:
    raw = summary.get("command_line_args")
    if isinstance(raw, str) and raw.strip():
        try:
            args = json.loads(raw)
            q = str(args.get("quantile_pairs", "")).strip()
            if q:
                return q
        except Exception:
            pass
    return summary_path.parent.name


def summarize_runtime(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.rglob("backtest_summary.json")):
        summary = _read_summary(summary_path)
        if not summary:
            continue
        log_path = _find_log_path(root, summary_path)
        log_text = _read_log_text(log_path)

        strategy = summary_path.parent.parent.name if summary_path.parent.parent != root else ""
        optimizer_steps = _safe_float(summary.get("optimizer_steps"))
        if optimizer_steps is None:
            optimizer_steps = _extract_last_int_pair(r"backtest rolling step (\d+)/(\d+)", log_text)
            optimizer_steps = float(optimizer_steps) if optimizer_steps is not None else None

        simulated_days = _simulated_days_from_args(summary)
        if simulated_days is None:
            simulated_days = _extract_float(r"timeframe_total_days:\s*([0-9.]+)", log_text)

        backtester_run_seconds = _safe_float(summary.get("backtester_run_seconds"))
        if backtester_run_seconds is None:
            backtester_run_seconds = _extract_float(r"END backtester_run \| elapsed=([0-9.]+)s", log_text)

        rows.append(
            {
                "model": _infer_model(summary, summary_path),
                "strategy": strategy,
                "quantile": _infer_quantile(summary, summary_path),
                "simulated_days": simulated_days,
                "rolling_steps": optimizer_steps,
                "backtester_run_seconds": backtester_run_seconds,
                "seconds_per_simulated_day": (
                    backtester_run_seconds / simulated_days
                    if backtester_run_seconds is not None and simulated_days not in (None, 0.0)
                    else None
                ),
                "seconds_per_rolling_step": (
                    backtester_run_seconds / optimizer_steps
                    if backtester_run_seconds is not None and optimizer_steps not in (None, 0.0)
                    else None
                ),
                "simulation_valid": _safe_float(summary.get("simulation_valid")),
                "thesis_reportable": _safe_float(summary.get("thesis_reportable")),
                "summary_path": str(summary_path),
                "log_path": str(log_path) if log_path is not None else "",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize battery backtest runtime from summary JSONs and logs.")
    ap.add_argument("root", type=Path, help="Simulation output root directory.")
    ap.add_argument("--out-csv", type=Path, default=None, help="Optional CSV output path.")
    args = ap.parse_args()

    df = summarize_runtime(args.root)
    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
    else:
        if df.empty:
            print("(no runtime summaries found)")
        else:
            print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
