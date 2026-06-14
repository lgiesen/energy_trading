#!/usr/bin/env python3
"""Read-only audit for BCM model-vs-naive revenue and side-selection gaps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _num(df: pd.DataFrame, *candidates: str) -> pd.Series:
    col = _first_existing(df.columns, candidates)
    if col is None:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _text(df: pd.DataFrame, *candidates: str) -> pd.Series:
    col = _first_existing(df.columns, candidates)
    if col is None:
        return pd.Series("", index=df.index, dtype=object)
    return df[col].fillna("").astype(str)


def _scenario_dirs(root: Path) -> list[Path]:
    if (root / "backtest_hourly.parquet").exists():
        return [root]
    return sorted({p.parent for p in root.rglob("backtest_hourly.parquet")})


def _load_hourly(scenario_dir: Path) -> pd.DataFrame:
    hourly = pd.read_parquet(scenario_dir / "backtest_hourly.parquet")
    naive_path = scenario_dir / "naive_hourly.parquet"
    if naive_path.exists() and "timestamp_utc" in hourly.columns:
        naive = pd.read_parquet(naive_path)
        if "timestamp_utc" in naive.columns:
            naive_cols = [c for c in naive.columns if c == "timestamp_utc" or c.startswith("naive_")]
            missing = [c for c in naive_cols if c not in hourly.columns]
            if missing:
                hourly = hourly.merge(naive[["timestamp_utc", *missing]], on="timestamp_utc", how="left")
    return hourly


def _load_summary(scenario_dir: Path) -> dict[str, object]:
    path = scenario_dir / "backtest_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _first_nonempty(values: pd.Series) -> str:
    for value in values.astype(str):
        value = value.strip()
        if value and value.lower() not in {"nan", "none", "0", "0.0"}:
            return value
    return ""


def _classify(row: pd.Series) -> str:
    if float(row.get("naive_simulation_valid", 1.0)) < 0.5:
        return "benchmark_invalid"
    if float(row["model_fallback_hours"]) > 0.0:
        return "model_fallback_contaminated"
    if float(row["naive_minus_model_total_revenue_eur"]) <= 0.0:
        return "model_revenue_ge_naive"
    model_zero = str(row.get("model_zero_reason", "")).lower()
    if any(token in model_zero for token in ("headroom", "infeasible", "derate", "zero")):
        return "model_guard_or_precommit_derate"
    if float(row["naive_locked_pos_mw_sum"]) > float(row["model_locked_pos_mw_sum"]):
        return "naive_more_pos_exposure"
    if float(row["naive_locked_neg_mw_sum"]) > float(row["model_locked_neg_mw_sum"]):
        return "naive_more_neg_exposure"
    return "model_forecast_or_side_choice"


def build_bcm_model_vs_naive_audit(scenario_dir: str | Path) -> pd.DataFrame:
    scenario_dir = Path(scenario_dir)
    summary = _load_summary(scenario_dir)
    hourly = _load_hourly(scenario_dir)
    if "timestamp_utc" not in hourly.columns:
        raise ValueError(f"{scenario_dir}: missing timestamp_utc")
    df = hourly.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.loc[df["timestamp_utc"].notna()].copy()
    block_col = _first_existing(
        df.columns,
        ["bcm_capacity_block_id", "bcm_product_block_id", "reserve_product_block_id"],
    )
    if block_col is None:
        df["_audit_block_id"] = df["timestamp_utc"].dt.floor("4h").astype(str)
        block_col = "_audit_block_id"

    records: list[dict[str, object]] = []
    for block_id, grp in df.groupby(block_col, dropna=False, sort=True):
        if str(block_id).strip().lower() in {"", "nan", "none"}:
            continue
        model_cap = _num(grp, "real_revenue_capacity_eur", "real_bcm_capacity_revenue_eur").sum()
        model_act = _num(grp, "real_bcm_linked_activation_revenue_eur").sum()
        naive_cap = _num(grp, "naive_bcm_capacity_revenue_eur").sum()
        naive_act = _num(grp, "naive_bcm_linked_activation_revenue_eur").sum()
        rec = {
            "scenario_dir": str(scenario_dir),
            "bcm_capacity_block_id": str(block_id),
            "delivery_start_utc": grp["timestamp_utc"].min().isoformat(),
            "delivery_end_utc": (grp["timestamp_utc"].max() + pd.Timedelta(hours=1)).isoformat(),
            "model_capacity_revenue_eur": float(model_cap),
            "model_activation_revenue_eur": float(model_act),
            "model_total_bcm_revenue_eur": float(model_cap + model_act),
            "naive_capacity_revenue_eur": float(naive_cap),
            "naive_activation_revenue_eur": float(naive_act),
            "naive_total_bcm_revenue_eur": float(naive_cap + naive_act),
            "naive_minus_model_total_revenue_eur": float((naive_cap + naive_act) - (model_cap + model_act)),
            "model_locked_pos_mw_sum": float(_num(grp, "real_locked_bcm_capacity_pos_mw", "locked_bcm_capacity_pos_mw").sum()),
            "model_locked_neg_mw_sum": float(_num(grp, "real_locked_bcm_capacity_neg_mw", "locked_bcm_capacity_neg_mw").sum()),
            "naive_locked_pos_mw_sum": float(_num(grp, "naive_locked_bcm_capacity_pos_mw").sum()),
            "naive_locked_neg_mw_sum": float(_num(grp, "naive_locked_bcm_capacity_neg_mw").sum()),
            "model_ev_pos_mean": float(_num(grp, "real_bcm_ev_pos", "bcm_ev_pos").replace(0.0, np.nan).mean(skipna=True)),
            "model_ev_neg_mean": float(_num(grp, "real_bcm_ev_neg", "bcm_ev_neg").replace(0.0, np.nan).mean(skipna=True)),
            "naive_ev_pos_mean": float(_num(grp, "naive_bcm_ev_pos").replace(0.0, np.nan).mean(skipna=True)),
            "naive_ev_neg_mean": float(_num(grp, "naive_bcm_ev_neg").replace(0.0, np.nan).mean(skipna=True)),
            "model_zero_reason": _first_nonempty(_text(grp, "real_bcm_zero_reason", "bcm_zero_reason")),
            "model_precommit_zero_reason": _first_nonempty(
                _text(grp, "real_bcm_precommit_zero_reason", "bcm_precommit_zero_reason")
            ),
            "naive_zero_reason": _first_nonempty(_text(grp, "naive_bcm_zero_reason")),
            "naive_simulation_valid": float(summary.get("naive_simulation_valid", 1.0) or 0.0),
            "naive_invalid_reason": str(summary.get("naive_invalid_reason", "") or "none"),
            "rhpf_simulation_valid": float(summary.get("rhpf_simulation_valid", 1.0) or 0.0),
            "rhpf_invalid_reason": str(summary.get("rhpf_invalid_reason", "") or "none"),
            "model_fallback_hours": float(_num(grp, "optimizer_fallback_used", "real_optimizer_fallback_used").gt(0.5).sum()),
        }
        rec["classification"] = _classify(pd.Series(rec))
        records.append(rec)
    audit = pd.DataFrame(records)
    if not audit.empty:
        audit = audit.sort_values("naive_minus_model_total_revenue_eur", ascending=False).reset_index(drop=True)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    frames = [build_bcm_model_vs_naive_audit(path) for path in _scenario_dirs(args.run_dir)]
    audit = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    if audit.empty:
        print("BCM model-vs-naive audit: no comparable rows")
        return
    print("BCM model-vs-naive audit")
    print(f"rows: {len(audit)}")
    print(f"naive_minus_model_total_revenue_eur: {audit['naive_minus_model_total_revenue_eur'].sum():.2f}")
    print(audit["classification"].value_counts(dropna=False).to_string())
    cols = [
        "scenario_dir",
        "bcm_capacity_block_id",
        "delivery_start_utc",
        "naive_minus_model_total_revenue_eur",
        "model_total_bcm_revenue_eur",
        "naive_total_bcm_revenue_eur",
        "model_locked_pos_mw_sum",
        "model_locked_neg_mw_sum",
        "naive_locked_pos_mw_sum",
        "naive_locked_neg_mw_sum",
        "classification",
        "model_zero_reason",
        "model_precommit_zero_reason",
    ]
    print(audit.loc[:, cols].head(max(0, int(args.max_rows))).to_string(index=False))
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(args.csv, index=False)
        print(f"wrote: {args.csv}")


if __name__ == "__main__":
    main()
