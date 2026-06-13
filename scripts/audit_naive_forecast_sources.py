#!/usr/bin/env python3
"""Audit naive forecast source lineage for one simulation scenario or run root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_SUFFIXES = (
    "_naive_source_timestamp_utc",
    "_naive_source_lag_hours",
    "_naive_source_mode",
)


def _scenario_dirs(root: Path) -> list[Path]:
    if (root / "backtest_summary.json").exists():
        return [root]
    return sorted(p.parent for p in root.glob("*/*/backtest_summary.json"))


def _num(summary: dict[str, object], key: str, default: float = float("nan")) -> float:
    try:
        return float(pd.to_numeric(pd.Series([summary.get(key, default)]), errors="coerce").iloc[0])
    except Exception:
        return default


def _load_hourly(scenario_dir: Path) -> pd.DataFrame:
    for name in ("naive_hourly.parquet", "backtest_hourly.parquet"):
        path = scenario_dir / name
        if path.exists():
            return pd.read_parquet(path)
    return pd.DataFrame()


def _audit_scenario(scenario_dir: Path, *, max_rows: int) -> tuple[bool, list[str]]:
    summary_path = scenario_dir / "backtest_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mode = str(summary.get("naive_forecast_mode", "") or "")
    fallback_count = _num(summary, "naive_forecast_fallback_count", 0.0)
    missing_count = _num(summary, "naive_weekly_source_missing_count", float("nan"))
    min_lag = _num(summary, "naive_forecast_source_min_lag_hours", float("nan"))
    causality = _num(summary, "naive_forecast_causality_violations", 0.0)

    errors: list[str] = []
    warnings: list[str] = []
    if mode != "same_weekday_last_week":
        warnings.append(f"mode_not_weekly:{mode or 'missing'}")
    if causality > 0.0:
        errors.append(f"causality_violations={causality:g}")
    if mode == "same_weekday_last_week":
        if fallback_count > 0.0:
            errors.append(f"weekly_fallback_used={fallback_count:g}")
        if np.isfinite(missing_count) and missing_count > 0.0:
            errors.append(f"weekly_source_missing={missing_count:g}")
        if np.isfinite(min_lag) and min_lag < 168.0 - 1e-6:
            errors.append(f"source_min_lag_hours={min_lag:g}<168")

    hourly = _load_hourly(scenario_dir)
    source_cols = [c for c in hourly.columns if c.endswith(SOURCE_SUFFIXES)] if not hourly.empty else []
    timestamp_col = "timestamp_utc" if "timestamp_utc" in hourly.columns else ""
    if mode == "same_weekday_last_week" and not source_cols:
        warnings.append("row_level_source_diagnostics_unavailable")
    elif mode == "same_weekday_last_week" and timestamp_col:
        ts = pd.to_datetime(hourly[timestamp_col], utc=True, errors="coerce")
        source_ts_cols = [c for c in hourly.columns if c.endswith("_naive_source_timestamp_utc")]
        rows: list[dict[str, object]] = []
        for col in source_ts_cols:
            source_ts = pd.to_datetime(hourly[col], utc=True, errors="coerce")
            expected = ts - pd.Timedelta(days=7)
            mismatch = source_ts.notna() & ts.notna() & (source_ts != expected)
            if mismatch.any():
                errors.append(f"{col}:source_ts_not_delivery_minus_7d_count={int(mismatch.sum())}")
                sample = hourly.loc[mismatch, [timestamp_col, col]].head(max_rows)
                for _, row in sample.iterrows():
                    rows.append(
                        {
                            "timestamp_utc": str(row[timestamp_col]),
                            "source_col": col,
                            "source_timestamp_utc": str(row[col]),
                        }
                    )
        if rows:
            warnings.append("source_mismatch_samples=" + json.dumps(rows, default=str))

    status = "PASS" if not errors else "FAIL"
    lines = [
        f"{status} {scenario_dir}",
        f"  naive_forecast_mode={mode}",
        f"  naive_forecast_causality_violations={causality:g}",
        f"  naive_forecast_fallback_count={fallback_count:g}",
        f"  naive_weekly_source_missing_count={missing_count:g}" if np.isfinite(missing_count) else "  naive_weekly_source_missing_count=missing",
        f"  naive_forecast_source_min_lag_hours={min_lag:g}" if np.isfinite(min_lag) else "  naive_forecast_source_min_lag_hours=missing",
    ]
    lines.extend(f"  ERROR {e}" for e in errors)
    lines.extend(f"  WARN {w}" for w in warnings)
    return not errors, lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Scenario directory or run root containing */*/backtest_summary.json")
    parser.add_argument("--max-rows", type=int, default=10, help="Maximum mismatch sample rows to print per source column.")
    args = parser.parse_args()

    scenarios = _scenario_dirs(args.path)
    if not scenarios:
        raise SystemExit(f"No scenarios found under {args.path}")
    all_ok = True
    for scenario in scenarios:
        ok, lines = _audit_scenario(scenario, max_rows=max(1, int(args.max_rows)))
        all_ok = all_ok and ok
        print("\n".join(lines))
    print(f"naive forecast source audit: {'PASS' if all_ok else 'FAIL'} scenarios_checked={len(scenarios)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
