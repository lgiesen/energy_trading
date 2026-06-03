from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _read_json(path: Path) -> dict[str, object]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _safe_float(v: object, default: float = float("nan")) -> float:
    try:
        out = float(pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0])
    except Exception:
        return default
    return out if pd.notna(out) else default


def _parse_quantile(path: Path, summary: dict[str, object]) -> str:
    for key in ("scenario", "quantile_pair", "quantile_policy"):
        val = str(summary.get(key, "") or "").strip()
        if val:
            return val.replace("_", "-")
    m = re.search(r"p\d{2}[-_]p\d{2}", str(path))
    return m.group(0).replace("_", "-") if m else path.name


def collect_runtime(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.rglob("backtest_summary.json")):
        summary = _read_json(summary_path)
        if not summary:
            continue
        scenario_dir = summary_path.parent
        hours = _safe_float(summary.get("optimized_hours_only_rows", summary.get("n_hours", float("nan"))))
        if not pd.notna(hours) or hours <= 0.0:
            hourly_csv = scenario_dir / "backtest_hourly.csv"
            hourly_parquet = scenario_dir / "backtest_hourly.parquet"
            try:
                if hourly_csv.exists():
                    hours = float(len(pd.read_csv(hourly_csv, usecols=[0])))
                elif hourly_parquet.exists():
                    hours = float(len(pd.read_parquet(hourly_parquet, columns=None)))
            except Exception:
                hours = float("nan")
        optimizer_total = _safe_float(summary.get("optimizer_total_seconds", 0.0), 0.0)
        settlement_pred = _safe_float(summary.get("settlement_predicted_seconds", 0.0), 0.0)
        settlement_real = _safe_float(summary.get("settlement_realized_seconds", 0.0), 0.0)
        output_write = _safe_float(summary.get("output_write_seconds", 0.0), 0.0)
        rows.append(
            {
                "scenario_path": str(scenario_dir),
                "model": summary.get("model_key", summary.get("model", "")),
                "strategy": summary.get("trading_strategy", scenario_dir.parent.name),
                "quantile": _parse_quantile(scenario_dir, summary),
                "hours": hours,
                "optimizer_total_seconds": optimizer_total,
                "optimizer_mean_seconds_per_step": _safe_float(summary.get("optimizer_mean_seconds_per_step", float("nan"))),
                "settlement_seconds": settlement_pred + settlement_real,
                "output_write_seconds": output_write,
                "seconds_per_simulated_hour": (
                    _safe_float(summary.get("backtester_run_seconds", 0.0), 0.0) / hours
                    if pd.notna(hours) and hours > 0.0
                    else float("nan")
                ),
                "debug_dump_count": _safe_float(summary.get("infeasible_debug_dump_count", 0.0), 0.0),
                "simulation_valid": _safe_float(summary.get("simulation_valid", 0.0), 0.0),
                "thesis_reportable": _safe_float(summary.get("thesis_reportable", 0.0), 0.0),
                "invalid_reason": summary.get("invalid_reason", ""),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize battery backtest runtime fields from simulation outputs.")
    parser.add_argument("root", help="Simulation root containing backtest_summary.json files.")
    parser.add_argument("--out-csv", default="", help="Optional CSV output path.")
    parser.add_argument("--out-json", default="", help="Optional JSON output path.")
    args = parser.parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Simulation root not found: {root}")
    df = collect_runtime(root)
    if df.empty:
        print(f"[WARN] No backtest_summary.json files found under: {root}")
        return
    print(df.to_string(index=False))
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(df.to_json(orient="records", indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
