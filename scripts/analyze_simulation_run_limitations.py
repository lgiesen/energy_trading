#!/usr/bin/env python3
"""Summarize limitations and invalidity diagnostics for an existing simulation run.

This script is read-only with respect to the simulation run. It collects scenario
summary files, invalid-reason tokens, solver failure diagnostics, and hourly
violation magnitudes into thesis-auditable CSV/JSON/TXT outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RUN_ROOT = Path("artifacts/simulation_runs/thesis_final_multi_2m_20260620T091938Z")
DEFAULT_OUT_ROOT = Path("artifacts/benchmark/simulation_limitations/thesis_final_multi_2m_20260620T091938Z")

REASON_EMPTY = {"", "none", "nan", "null", "[]", "{}"}
HOURLY_SEVERITY_PATTERNS = (
    "violation",
    "shortfall",
    "missed",
    "fallback",
    "infeasible",
    "repair",
    "rejected",
)
HOURLY_EXCLUDE_PATTERNS = (
    "margin",
    "enabled",
    "available",
    "source",
    "reason",
    "policy",
    "checked",
    "passed",
    "required",
    "target",
    "min_mwh",
    "max_mwh",
)
DEBUG_FILES = (
    "optimization_failure_debug.csv",
    "optimization_infeasibility_attribution.csv",
    "solver_failure_diagnostics.csv",
    "hard_final_soc_infeasibility_debug.csv",
    "backtest_milp_event_summary.csv",
)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _split_reasons(value: Any) -> list[str]:
    text = str(value or "").strip()
    if text.lower() in REASON_EMPTY:
        return []
    parts = [p.strip() for p in re.split(r"[,;|]", text) if p.strip()]
    return [p for p in parts if p.lower() not in REASON_EMPTY]


def _scenario_dirs(run_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for summary in run_root.glob("*/multi/*/backtest_summary.json"):
        dirs.append(summary.parent)
    return sorted(dirs)


def _scenario_identity(path: Path, run_root: Path) -> dict[str, str]:
    rel = path.relative_to(run_root)
    top = rel.parts[0] if rel.parts else path.name
    quantile = rel.parts[-1] if len(rel.parts) >= 1 else ""
    if top.startswith("benchmarks_"):
        model = top.replace("benchmarks_", "").upper()
        model_key = top.replace("benchmarks_", "")
    else:
        bits = top.split("_")
        model_key = bits[0] if bits else top
        model = {"linear": "RLQR", "xgb": "XGB", "tft": "TFT"}.get(model_key, model_key.upper())
    return {
        "folder": top,
        "model_key": model_key,
        "model": model,
        "quantile_policy": quantile.replace("_", "-"),
        "scenario_path": str(path),
    }


def _load_scenario_summary(path: Path, run_root: Path) -> dict[str, Any]:
    ident = _scenario_identity(path, run_root)
    backtest = _read_json(path / "backtest_summary.json")
    model_summary = _read_json(path / "model_summary.json")
    perf = _read_csv(path / "performance_metrics.csv")
    perf_row = perf.iloc[0].to_dict() if not perf.empty else {}
    row: dict[str, Any] = {**ident}
    for source in (backtest, model_summary, perf_row):
        for key in [
            "simulation_valid",
            "thesis_reportable",
            "invalid_reason",
            "realized_total_pnl_eur",
            "realized_net_revenue_eur",
            "annualized_realized_net_revenue_eur",
            "predicted_total_pnl_eur",
            "fallback_used",
            "optimizer_fallback_used",
            "infeasible_debug_dump_count",
            "accepted_path_infeasible_debug_dump_count",
            "candidate_infeasible_debug_dump_count",
            "first_infeasible_timestamp_utc",
            "first_fallback_timestamp_utc",
            "first_rolling_window_nonterminal_infeasible_timestamp_utc",
            "terminal_soc_repair_cost_eur",
            "final_soc_shortfall_mwh",
            "min_soc_mwh",
            "max_soc_mwh",
            "final_soc_mwh",
            "target_final_soc_mwh",
            "throughput_mwh_total",
            "real_power_violation_total_mw",
            "real_power_violation_charge_mw",
            "real_power_violation_discharge_mw",
            "real_protected_soc_violation_pos_mwh",
            "real_protected_soc_violation_neg_mwh",
            "real_headroom_violation_pos_mwh",
            "real_headroom_violation_neg_mwh",
            "real_missed_activation_mwh",
            "real_missed_capacity_mw",
        ]:
            if key in source and (key not in row or pd.isna(row.get(key)) or row.get(key) == ""):
                row[key] = source.get(key)
    row["invalid_reason_tokens"] = ",".join(_split_reasons(row.get("invalid_reason", "")))
    row["n_invalid_reason_tokens"] = len(_split_reasons(row.get("invalid_reason", "")))
    row["simulation_valid"] = _safe_float(row.get("simulation_valid"), 0.0)
    row["thesis_reportable"] = _safe_float(row.get("thesis_reportable"), 0.0)
    return row


def _summarize_numeric_series(series: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {
            "count_non_null": 0.0,
            "count_nonzero": 0.0,
            "share_nonzero": 0.0,
            "sum": 0.0,
            "mean": math.nan,
            "p95": math.nan,
            "max": math.nan,
        }
    nonzero = x.abs().gt(1e-12)
    return {
        "count_non_null": float(len(x)),
        "count_nonzero": float(nonzero.sum()),
        "share_nonzero": float(nonzero.mean()),
        "sum": float(x.sum()),
        "mean": float(x.mean()),
        "p95": float(x.quantile(0.95)),
        "max": float(x.max()),
    }


def _hourly_limitation_rows(path: Path, run_root: Path) -> list[dict[str, Any]]:
    ident = _scenario_identity(path, run_root)
    hourly = _read_parquet(path / "backtest_hourly.parquet")
    if hourly.empty:
        hourly = _read_csv(path / "backtest_hourly.csv")
    if hourly.empty:
        return []
    rows: list[dict[str, Any]] = []
    numeric_cols = [
        c for c in hourly.columns
        if any(pattern in c.lower() for pattern in HOURLY_SEVERITY_PATTERNS)
        and not any(pattern in c.lower() for pattern in HOURLY_EXCLUDE_PATTERNS)
        and pd.api.types.is_numeric_dtype(hourly[c])
    ]
    for col in sorted(numeric_cols):
        stats = _summarize_numeric_series(hourly[col])
        if stats["count_nonzero"] <= 0 and not any(token in col.lower() for token in ["fallback", "infeasible"]):
            continue
        rows.append({**ident, "source_file": "backtest_hourly", "limitation_metric": col, **stats})
    return rows


def _reason_long_rows(scenarios: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if scenarios.empty:
        return pd.DataFrame()
    for _, row in scenarios.iterrows():
        tokens = _split_reasons(row.get("invalid_reason", ""))
        if not tokens:
            rows.append(
                {
                    "folder": row.get("folder"),
                    "model": row.get("model"),
                    "quantile_policy": row.get("quantile_policy"),
                    "invalid_reason": "",
                    "simulation_valid": row.get("simulation_valid"),
                    "thesis_reportable": row.get("thesis_reportable"),
                }
            )
        for token in tokens:
            rows.append(
                {
                    "folder": row.get("folder"),
                    "model": row.get("model"),
                    "quantile_policy": row.get("quantile_policy"),
                    "invalid_reason": token,
                    "simulation_valid": row.get("simulation_valid"),
                    "thesis_reportable": row.get("thesis_reportable"),
                }
            )
    return pd.DataFrame(rows)


def _infeasibility_frequency(scenarios: pd.DataFrame, hourly: pd.DataFrame, debug: pd.DataFrame) -> pd.DataFrame:
    if scenarios.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    hourly_wide = pd.DataFrame()
    if not hourly.empty:
        key_metrics = [
            "optimizer_fallback_used",
            "is_fallback_hour",
            "da_candidate_rejected_hourly_lock_infeasible",
            "da_candidate_rejected_locked_commitment_infeasible",
            "da_precommit_da_candidate_rejected_hourly_lock_infeasible",
            "da_precommit_da_candidate_rejected_locked_commitment_infeasible",
        ]
        h = hourly[hourly["limitation_metric"].isin(key_metrics)].copy()
        if not h.empty:
            hourly_wide = h.pivot_table(
                index=["folder", "quantile_policy"],
                columns="limitation_metric",
                values="count_nonzero",
                aggfunc="max",
                fill_value=0,
            ).reset_index()
    debug_wide = pd.DataFrame()
    if not debug.empty:
        d = debug.groupby(["folder", "quantile_policy", "debug_file"], as_index=False)["row_count"].max()
        debug_wide = d.pivot_table(
            index=["folder", "quantile_policy"],
            columns="debug_file",
            values="row_count",
            aggfunc="max",
            fill_value=0,
        ).reset_index()
    for _, row in scenarios.iterrows():
        base = {
            "folder": row.get("folder"),
            "model": row.get("model"),
            "quantile_policy": row.get("quantile_policy"),
            "simulation_valid": row.get("simulation_valid"),
            "invalid_reason": row.get("invalid_reason", ""),
        }
        key = (row.get("folder"), row.get("quantile_policy"))
        for frame in [hourly_wide, debug_wide]:
            if frame.empty:
                continue
            match = frame[(frame["folder"].eq(key[0])) & (frame["quantile_policy"].eq(key[1]))]
            if not match.empty:
                for col, value in match.iloc[0].items():
                    if col not in {"folder", "quantile_policy"}:
                        base[f"{col}_count"] = int(_safe_float(value, 0.0))
        rows.append(base)
    out = pd.DataFrame(rows)
    count_cols = [c for c in out.columns if c.endswith("_count")]
    if count_cols:
        out["total_infeasibility_or_fallback_events_count"] = out[count_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).astype(int)
    return out


def _categorical_count_rows(path: Path, run_root: Path) -> list[dict[str, Any]]:
    ident = _scenario_identity(path, run_root)
    hourly = _read_parquet(path / "backtest_hourly.parquet")
    if hourly.empty:
        hourly = _read_csv(path / "backtest_hourly.csv")
    if hourly.empty:
        return []
    rows: list[dict[str, Any]] = []
    cols = [
        c for c in hourly.columns
        if any(pattern in c.lower() for pattern in ["reason", "driver", "status", "classification", "source"])
        and not pd.api.types.is_numeric_dtype(hourly[c])
    ]
    for col in sorted(cols):
        s = hourly[col].fillna("").astype(str).str.strip()
        s = s[~s.str.lower().isin(REASON_EMPTY)]
        if s.empty:
            continue
        counts = s.value_counts().head(25)
        for value, count in counts.items():
            rows.append({**ident, "source_file": "backtest_hourly", "field": col, "value": value, "count": int(count), "share_of_hours": float(count / len(hourly))})
    return rows


def _debug_file_rows(path: Path, run_root: Path) -> list[dict[str, Any]]:
    ident = _scenario_identity(path, run_root)
    rows: list[dict[str, Any]] = []
    for filename in DEBUG_FILES:
        df = _read_csv(path / filename)
        if df.empty:
            rows.append({**ident, "debug_file": filename, "row_count": 0, "classification_field": "", "classification_value": "", "classification_count": 0})
            continue
        class_cols = [
            c for c in df.columns
            if any(token in c.lower() for token in ["root_cause", "classification", "solver_status", "failure", "reason", "driver"])
        ]
        if not class_cols:
            rows.append({**ident, "debug_file": filename, "row_count": int(len(df)), "classification_field": "", "classification_value": "", "classification_count": int(len(df))})
            continue
        for col in class_cols[:4]:
            counts = df[col].fillna("").astype(str).str.strip()
            counts = counts[~counts.str.lower().isin(REASON_EMPTY)].value_counts().head(20)
            if counts.empty:
                rows.append({**ident, "debug_file": filename, "row_count": int(len(df)), "classification_field": col, "classification_value": "", "classification_count": 0})
            for value, count in counts.items():
                rows.append({**ident, "debug_file": filename, "row_count": int(len(df)), "classification_field": col, "classification_value": value, "classification_count": int(count)})
    return rows


def _invalid_reason_counts(scenarios: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = max(len(scenarios), 1)
    invalid = scenarios[pd.to_numeric(scenarios["simulation_valid"], errors="coerce").fillna(0.0) < 0.5].copy()
    invalid_total = max(len(invalid), 1)
    counter: Counter[str] = Counter()
    by_model: Counter[tuple[str, str]] = Counter()
    for _, row in scenarios.iterrows():
        tokens = _split_reasons(row.get("invalid_reason", ""))
        for token in tokens:
            counter[token] += 1
            by_model[(str(row.get("model", "")), token)] += 1
    for reason, count in counter.most_common():
        models = {model: c for (model, r), c in by_model.items() if r == reason}
        rows.append(
            {
                "invalid_reason": reason,
                "scenario_count": int(count),
                "share_of_all_scenarios": count / total,
                "share_of_invalid_scenarios": count / invalid_total,
                "by_model_json": json.dumps(models, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def _scenario_limitation_score(scenarios: pd.DataFrame, hourly: pd.DataFrame, debug: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    hourly_lookup = {}
    if not hourly.empty:
        for (folder, quantile), part in hourly.groupby(["folder", "quantile_policy"], dropna=False):
            hourly_lookup[(folder, quantile)] = {
                "nonzero_metric_count": int(part["limitation_metric"].nunique()),
                "max_share_nonzero": float(pd.to_numeric(part["share_nonzero"], errors="coerce").max()),
                "max_abs_severity": float(pd.to_numeric(part["max"], errors="coerce").abs().max()),
            }
    debug_lookup = {}
    if not debug.empty:
        for (folder, quantile), part in debug.groupby(["folder", "quantile_policy"], dropna=False):
            debug_lookup[(folder, quantile)] = {
                "debug_files_with_rows": int((pd.to_numeric(part["row_count"], errors="coerce").fillna(0) > 0).sum()),
                "max_debug_file_rows": int(pd.to_numeric(part["row_count"], errors="coerce").fillna(0).max()),
            }
    for _, row in scenarios.iterrows():
        key = (row.get("folder"), row.get("quantile_policy"))
        rows.append(
            {
                "folder": row.get("folder"),
                "model": row.get("model"),
                "quantile_policy": row.get("quantile_policy"),
                "simulation_valid": row.get("simulation_valid"),
                "thesis_reportable": row.get("thesis_reportable"),
                "invalid_reason": row.get("invalid_reason", ""),
                "n_invalid_reason_tokens": row.get("n_invalid_reason_tokens", 0),
                **hourly_lookup.get(key, {"nonzero_metric_count": 0, "max_share_nonzero": 0.0, "max_abs_severity": math.nan}),
                **debug_lookup.get(key, {"debug_files_with_rows": 0, "max_debug_file_rows": 0}),
                "realized_net_revenue_eur": row.get("realized_net_revenue_eur", row.get("realized_total_pnl_eur", math.nan)),
                "annualized_realized_net_revenue_eur": row.get("annualized_realized_net_revenue_eur", math.nan),
            }
        )
    return pd.DataFrame(rows)


def _write_text_report(path: Path, *, run_root: Path, scenarios: pd.DataFrame, reasons: pd.DataFrame, hourly: pd.DataFrame, debug: pd.DataFrame, score: pd.DataFrame) -> None:
    lines: list[str] = []
    n = len(scenarios)
    invalid = int((pd.to_numeric(scenarios["simulation_valid"], errors="coerce").fillna(0.0) < 0.5).sum()) if not scenarios.empty else 0
    reportable = int((pd.to_numeric(scenarios["thesis_reportable"], errors="coerce").fillna(0.0) >= 0.5).sum()) if not scenarios.empty else 0
    lines.extend(
        [
            f"Simulation limitation report",
            f"Run root: {run_root}",
            f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
            "",
            f"Scenarios discovered: {n}",
            f"Simulation-valid scenarios: {n - invalid}",
            f"Invalid scenarios: {invalid}",
            f"Thesis-reportable scenarios: {reportable}",
            "",
            "Top invalidity reasons:",
        ]
    )
    if reasons.empty:
        lines.append("- none")
    else:
        for _, row in reasons.head(15).iterrows():
            lines.append(f"- {row['invalid_reason']}: {int(row['scenario_count'])} scenarios ({row['share_of_all_scenarios']:.1%} of all)")
    lines.extend(["", "Worst scenarios by limitation breadth:"])
    if score.empty:
        lines.append("- none")
    else:
        sort_cols = ["n_invalid_reason_tokens", "nonzero_metric_count", "max_debug_file_rows"]
        s = score.sort_values(sort_cols, ascending=False).head(10)
        for _, row in s.iterrows():
            lines.append(
                f"- {row.get('folder')} {row.get('quantile_policy')}: "
                f"reasons={row.get('n_invalid_reason_tokens')}, hourly_metrics={row.get('nonzero_metric_count')}, "
                f"max_debug_rows={row.get('max_debug_file_rows')}, invalid_reason={row.get('invalid_reason')}"
            )
    lines.extend(["", "Most frequent nonzero hourly limitation metrics:"])
    if hourly.empty:
        lines.append("- none")
    else:
        h = (
            hourly.groupby("limitation_metric", as_index=False)
            .agg(scenarios=("folder", "nunique"), max_share_nonzero=("share_nonzero", "max"), max_value=("max", "max"))
            .sort_values(["scenarios", "max_share_nonzero"], ascending=False)
            .head(20)
        )
        for _, row in h.iterrows():
            lines.append(f"- {row['limitation_metric']}: scenarios={int(row['scenarios'])}, max_share_nonzero={row['max_share_nonzero']:.1%}, max={row['max_value']:.6g}")
    lines.extend(["", "Debug-file coverage:"])
    if debug.empty:
        lines.append("- none")
    else:
        d = debug.groupby("debug_file", as_index=False).agg(scenarios=("folder", "nunique"), max_rows=("row_count", "max"))
        for _, row in d.sort_values("debug_file").iterrows():
            lines.append(f"- {row['debug_file']}: scenarios={int(row['scenarios'])}, max_rows={int(row['max_rows'])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_limitations_report(run_root: Path, out_dir: Path) -> dict[str, str]:
    run_root = run_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_paths = _scenario_dirs(run_root)
    scenario_rows = [_load_scenario_summary(p, run_root) for p in scenario_paths]
    scenarios = pd.DataFrame(scenario_rows)
    hourly = pd.DataFrame([row for p in scenario_paths for row in _hourly_limitation_rows(p, run_root)])
    categorical = pd.DataFrame([row for p in scenario_paths for row in _categorical_count_rows(p, run_root)])
    debug = pd.DataFrame([row for p in scenario_paths for row in _debug_file_rows(p, run_root)])
    reasons = _invalid_reason_counts(scenarios) if not scenarios.empty else pd.DataFrame()
    reason_long = _reason_long_rows(scenarios)
    score = _scenario_limitation_score(scenarios, hourly, debug) if not scenarios.empty else pd.DataFrame()
    infeasibility = _infeasibility_frequency(scenarios, hourly, debug)

    paths = {
        "scenario_limitations": out_dir / "scenario_limitations.csv",
        "scenario_invalid_reasons_long": out_dir / "scenario_invalid_reasons_long.csv",
        "invalid_reason_counts": out_dir / "invalid_reason_counts.csv",
        "infeasibility_frequency": out_dir / "infeasibility_frequency.csv",
        "hourly_limitation_severity": out_dir / "hourly_limitation_severity.csv",
        "hourly_categorical_counts": out_dir / "hourly_categorical_counts.csv",
        "debug_file_summary": out_dir / "debug_file_summary.csv",
        "scenario_limitation_score": out_dir / "scenario_limitation_score.csv",
        "limitations_report": out_dir / "limitations_report.txt",
        "limitations_manifest": out_dir / "limitations_manifest.json",
    }
    scenarios.to_csv(paths["scenario_limitations"], index=False)
    reason_long.to_csv(paths["scenario_invalid_reasons_long"], index=False)
    reasons.to_csv(paths["invalid_reason_counts"], index=False)
    infeasibility.to_csv(paths["infeasibility_frequency"], index=False)
    hourly.to_csv(paths["hourly_limitation_severity"], index=False)
    categorical.to_csv(paths["hourly_categorical_counts"], index=False)
    debug.to_csv(paths["debug_file_summary"], index=False)
    score.to_csv(paths["scenario_limitation_score"], index=False)
    _write_text_report(paths["limitations_report"], run_root=run_root, scenarios=scenarios, reasons=reasons, hourly=hourly, debug=debug, score=score)
    manifest = {
        "schema_version": "simulation_limitations_v1",
        "run_root": str(run_root),
        "out_dir": str(out_dir.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_count": int(len(scenarios)),
        "outputs": {k: str(v) for k, v in paths.items() if k != "limitations_manifest"},
        "methodological_notes": [
            "The script reads existing artifacts only; it does not rerun simulations.",
            "Invalidity causes come from stored invalid_reason fields and diagnostic files.",
            "Severity uses available proxies: nonzero hourly diagnostic counts, maxima/sums, debug-file row counts, and solver/root-cause classifications.",
        ],
    }
    paths["limitations_manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze limitations and invalidity causes for an existing simulation run.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = build_limitations_report(Path(args.run_root), Path(args.out_dir))
    print(f"[OK] Simulation limitation report written: {paths['limitations_report']}")
    for name, path in paths.items():
        print(f"[OK] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
