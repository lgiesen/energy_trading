"""Compare two existing simulation run directories without running simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _summary_paths(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for p in root.rglob("backtest_summary.json"):
        rel = str(p.parent.relative_to(root))
        paths[rel] = p
    return paths


def _load_summary(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _numeric(value: object) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(v):
        return None
    return v


def compare_runs(baseline_dir: Path, candidate_dir: Path) -> pd.DataFrame:
    base = _summary_paths(baseline_dir)
    cand = _summary_paths(candidate_dir)
    keys = sorted(set(base) | set(cand))
    rows: list[dict[str, object]] = []
    metrics = [
        "realized_total_pnl_eur",
        "simulation_valid",
        "thesis_reportable",
        "fallback_used",
        "headroom_violation_count",
        "protected_soc_violation_count",
        "physical_soc_violation_count",
    ]
    for key in keys:
        b = _load_summary(base[key]) if key in base else {}
        c = _load_summary(cand[key]) if key in cand else {}
        row: dict[str, object] = {
            "scenario_path": key,
            "baseline_exists": key in base,
            "candidate_exists": key in cand,
            "baseline_invalid_reason": b.get("invalid_reason", ""),
            "candidate_invalid_reason": c.get("invalid_reason", ""),
        }
        for metric in metrics:
            bv = _numeric(b.get(metric))
            cv = _numeric(c.get(metric))
            row[f"baseline_{metric}"] = bv
            row[f"candidate_{metric}"] = cv
            row[f"delta_{metric}"] = (cv - bv) if bv is not None and cv is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir", required=True, type=Path)
    p.add_argument("--candidate-dir", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    df = compare_runs(args.baseline_dir, args.candidate_dir)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
    else:
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
