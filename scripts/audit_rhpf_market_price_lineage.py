#!/usr/bin/env python3
"""Read-only audit for RHPF market-price bid lineage across DA, BCM, and BEM.

The audit checks whether rolling perfect foresight submits prices equal to the
true market/cutoff prices whenever it submits nonzero MW. It reads existing
simulation artifacts only and does not run backtests or mutate artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SIDES = ("pos", "neg")


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _num_series(df: pd.DataFrame, candidates: Iterable[str], default: float = np.nan) -> pd.Series:
    col = _first_existing(df.columns, candidates)
    if col is None:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _str_series(df: pd.DataFrame, candidates: Iterable[str], default: str = "") -> pd.Series:
    col = _first_existing(df.columns, candidates)
    if col is None:
        return pd.Series(default, index=df.index, dtype="object")
    return df[col].fillna(default).astype(str)


def _finite(value: Any) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value_f if math.isfinite(value_f) else float("nan")


def _nonzero(value: Any, tol: float) -> bool:
    value_f = _finite(value)
    return math.isfinite(value_f) and abs(value_f) > tol


def _close(a: Any, b: Any, tol: float) -> bool:
    a_f = _finite(a)
    b_f = _finite(b)
    return math.isfinite(a_f) and math.isfinite(b_f) and abs(a_f - b_f) <= tol


def _group_value(series: pd.Series, *, prefer: str = "max") -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    nonzero = values[np.abs(values) > 1e-9]
    sample = nonzero if not nonzero.empty else values
    if prefer == "mean":
        return float(sample.mean())
    if prefer == "min":
        return float(sample.min())
    return float(sample.max())


def _group_text(series: pd.Series) -> str:
    for value in series.astype(str):
        value = str(value).strip()
        if value and value.lower() not in {"nan", "none", "0", "0.0"}:
            return value
    return ""


def _scenario_dirs(run_dir: Path) -> list[Path]:
    if (run_dir / "backtest_summary.json").exists() or (run_dir / "backtest_hourly.parquet").exists():
        return [run_dir]
    dirs = sorted({p.parent for p in run_dir.rglob("backtest_summary.json")})
    if dirs:
        return dirs
    return sorted({p.parent for p in run_dir.rglob("backtest_hourly.parquet")})


def _load_hourly(scenario_dir: Path) -> pd.DataFrame:
    hourly_path = scenario_dir / "backtest_hourly.parquet"
    if not hourly_path.exists():
        raise FileNotFoundError(f"missing {hourly_path}")
    hourly = pd.read_parquet(hourly_path)
    pf_path = scenario_dir / "rolling_pf_hourly.parquet"
    if pf_path.exists() and "timestamp_utc" in hourly.columns:
        pf = pd.read_parquet(pf_path)
        if "timestamp_utc" in pf.columns:
            pf_cols = [c for c in pf.columns if c == "timestamp_utc" or c.startswith("perfect_foresight_")]
            missing = [c for c in pf_cols if c not in hourly.columns]
            if missing:
                hourly = hourly.merge(pf[["timestamp_utc", *missing]], on="timestamp_utc", how="left")
    return hourly


def classify_da_lineage_row(row: pd.Series, *, tol: float = 1e-9) -> str:
    submitted = max(0.0, _finite(row.get("submitted_buy_mw"))) + max(0.0, _finite(row.get("submitted_sell_mw")))
    rejected = max(0.0, _finite(row.get("price_rejected_buy_mwh"))) + max(
        0.0, _finite(row.get("price_rejected_sell_mwh"))
    )
    policy = str(row.get("execution_policy", "") or "").strip().lower()
    price_taker = _finite(row.get("price_taker_mode"))
    if submitted <= tol:
        return "da_no_bid_physical_or_economic"
    if rejected > tol or (policy and policy != "price_taker") or price_taker < 0.5:
        return "da_bug_limit_price_used_in_rhpf"
    return "da_ok_price_taker"


def classify_bcm_lineage_row(row: pd.Series, *, tol: float = 1e-9) -> str:
    submitted = max(0.0, _finite(row.get("rhpf_submitted_mw")))
    awarded = max(0.0, _finite(row.get("rhpf_awarded_mw")))
    bid = _finite(row.get("rhpf_bid_price_eur_per_mw_h"))
    cutoff = _finite(row.get("true_market_price_eur_per_mw_h"))
    zero_reason = str(row.get("rhpf_zero_reason", "") or "").lower()
    headroom_reason = str(row.get("rhpf_headroom_rejection_reason", "") or "").lower()
    if awarded > tol and math.isfinite(bid) and math.isfinite(cutoff) and bid > cutoff + tol:
        return "bcm_auction_bug_awarded_above_cutoff"
    if submitted > tol and math.isfinite(bid) and math.isfinite(cutoff):
        if abs(bid - cutoff) <= tol:
            return "bcm_ok_rhpf_bids_cutoff"
        if bid < cutoff - tol:
            return "bcm_rhpf_underbids_cutoff"
        return "bcm_rhpf_bid_above_cutoff_rejected"
    if submitted <= tol and ("headroom" in zero_reason or "headroom" in headroom_reason):
        return "bcm_no_bid_due_to_headroom_or_side_choice"
    if submitted <= tol:
        return "bcm_no_bid_due_to_headroom_or_side_choice"
    return "bcm_not_comparable"


def classify_bem_lineage_row(row: pd.Series, *, tol: float = 1e-9) -> str:
    submitted = max(0.0, _finite(row.get("rhpf_submitted_mw")))
    accepted = max(0.0, _finite(row.get("rhpf_accepted")))
    bid = _finite(row.get("rhpf_bid_price_eur_per_mwh"))
    true_price = _finite(row.get("true_market_price_eur_per_mwh"))
    guard_reason = str(row.get("rhpf_guard_reason", "") or "").lower()
    if submitted <= tol:
        return "bem_no_bid_due_to_physical_guard" if "headroom" in guard_reason else "bem_no_bid_physical_or_economic"
    if math.isfinite(bid) and math.isfinite(true_price) and abs(bid - true_price) <= tol:
        return "bem_ok_rhpf_bids_true_activation_price"
    if accepted > tol:
        return "bem_accepted_despite_wrong_price"
    return "bem_rhpf_underbids_or_overbids_true_activation_price"


def _build_da_rows(df: pd.DataFrame, scenario_dir: Path, tol: float) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "scenario_dir": str(scenario_dir),
            "market": "DA",
            "timestamp_utc": pd.to_datetime(df.get("timestamp_utc"), utc=True, errors="coerce"),
            "side": "both",
            "submitted_buy_mw": _num_series(df, ["perfect_foresight_submitted_da_buy_mw"]),
            "submitted_sell_mw": _num_series(df, ["perfect_foresight_submitted_da_sell_mw"]),
            "price_rejected_buy_mwh": _num_series(df, ["perfect_foresight_da_price_rejected_buy_mwh"]),
            "price_rejected_sell_mwh": _num_series(df, ["perfect_foresight_da_price_rejected_sell_mwh"]),
            "execution_policy": _str_series(df, ["perfect_foresight_da_execution_policy", "rolling_pf_da_execution_policy"]),
            "price_taker_mode": _num_series(df, ["perfect_foresight_da_price_taker_mode", "rolling_pf_da_price_taker_mode"]),
            "true_market_price_eur_per_mwh": _num_series(df, ["true_da_price", "da_price"]),
            "rhpf_bid_price_eur_per_mwh": _num_series(
                df,
                ["perfect_foresight_submitted_da_buy_price_eur_mwh", "perfect_foresight_submitted_da_sell_price_eur_mwh"],
            ),
        }
    )
    out["classification"] = out.apply(lambda r: classify_da_lineage_row(r, tol=tol), axis=1)
    out["price_diff_eur"] = np.nan
    return out


def _build_bcm_rows(df: pd.DataFrame, scenario_dir: Path, tol: float) -> pd.DataFrame:
    work = df.copy()
    work["timestamp_utc"] = pd.to_datetime(work.get("timestamp_utc"), utc=True, errors="coerce")
    block_col = _first_existing(work.columns, ["bcm_capacity_block_id", "bcm_product_block_id", "reserve_product_block_id"])
    if block_col is None:
        work["_audit_block_id"] = work["timestamp_utc"].dt.floor("4h").astype(str)
        block_col = "_audit_block_id"
    records: list[dict[str, Any]] = []
    for block_id, blk in work.dropna(subset=["timestamp_utc"]).groupby(block_col, dropna=False, sort=True):
        if str(block_id).lower() in {"", "nan", "none"}:
            continue
        for side in SIDES:
            record = {
                "scenario_dir": str(scenario_dir),
                "market": "BCM",
                "timestamp_utc": str(blk["timestamp_utc"].min()),
                "bcm_capacity_block_id": str(block_id),
                "side": side,
                "rhpf_submitted_mw": _group_value(
                    _num_series(blk, [f"perfect_foresight_submitted_bcm_capacity_{side}_mw"])
                ),
                "rhpf_bid_price_eur_per_mw_h": _group_value(
                    _num_series(
                        blk,
                        [
                            f"perfect_foresight_bcm_capacity_bid_price_{side}_eur_per_mw_h",
                            f"perfect_foresight_bcm_lockbook_capacity_bid_price_{side}_eur_per_mw_h",
                            f"perfect_foresight_bcm_settlement_capacity_price_resolved_{side}_eur_per_mw_h",
                            f"perfect_foresight_settlement_cap_bid_price_{side}_eur_mw",
                        ],
                    ),
                    prefer="mean",
                ),
                "true_market_price_eur_per_mw_h": _group_value(
                    _num_series(blk, [f"bcm_capacity_cutoff_price_{side}_eur_per_mw_h"]), prefer="mean"
                ),
                "rhpf_awarded_mw": _group_value(
                    _num_series(
                        blk,
                        [f"perfect_foresight_awarded_capacity_{side}_mw", f"perfect_foresight_locked_bcm_capacity_{side}_mw"],
                    )
                ),
                "rhpf_zero_reason": _group_text(
                    _str_series(blk, ["perfect_foresight_bcm_precommit_zero_reason", "perfect_foresight_bcm_zero_reason"])
                ),
                "rhpf_headroom_rejection_reason": "",
            }
            required = _group_value(_num_series(blk, [f"perfect_foresight_required_headroom_{side}_mwh"]))
            available = _group_value(_num_series(blk, [f"perfect_foresight_available_headroom_{side}_mwh"]))
            if math.isfinite(required) and math.isfinite(available) and required > available + tol:
                record["rhpf_headroom_rejection_reason"] = "headroom_infeasible"
            record["classification"] = classify_bcm_lineage_row(pd.Series(record), tol=tol)
            record["price_diff_eur"] = _finite(record["rhpf_bid_price_eur_per_mw_h"]) - _finite(
                record["true_market_price_eur_per_mw_h"]
            )
            records.append(record)
    return pd.DataFrame.from_records(records)


def _build_bem_rows(df: pd.DataFrame, scenario_dir: Path, tol: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    timestamps = pd.to_datetime(df.get("timestamp_utc"), utc=True, errors="coerce")
    for side in SIDES:
        side_label = "pos" if side == "pos" else "neg"
        submitted = _num_series(
            df,
            [
                f"perfect_foresight_bem_only_submitted_{side_label}_mw",
                f"perfect_foresight_afrr_bem_submitted_{side_label}_mw",
                f"perfect_foresight_submitted_afrr_{side_label}_mw",
            ],
        )
        bid = _num_series(
            df,
            [
                f"perfect_foresight_bem_submitted_activation_price_{side_label}_eur_per_mwh",
                f"perfect_foresight_bem_activation_bid_price_{side_label}_eur_per_mwh",
                f"bem_submitted_activation_price_{side_label}_eur_per_mwh",
                f"bem_activation_bid_price_{side_label}_eur_per_mwh",
                f"perfect_foresight_bcm_activation_bid_price_{side_label}",
                f"perfect_foresight_executed_afrr_act_{side_label}_bin_0_price_eur_mwh",
                f"perfect_foresight_submitted_afrr_{side_label}_bin_0_price_eur_mw",
            ],
        )
        true_price = _num_series(
            df,
            [
                f"perfect_foresight_true_activation_price_{side_label}",
                f"perfect_foresight_bcm_true_activation_price_{side_label}",
                f"true_activation_price_{side_label}",
                f"bcm_true_activation_price_{side_label}",
                f"afrr_activation_price_vwap_{side_label}",
            ],
        )
        accepted = _num_series(df, [f"perfect_foresight_afrr_act_{side_label}_accepted"])
        guard_reason = _str_series(
            df,
            [
                f"perfect_foresight_bem_guard_zero_reason_{side_label}",
                "perfect_foresight_bem_only_headroom_guard_reason",
            ],
        )
        for idx in df.index:
            record = {
                "scenario_dir": str(scenario_dir),
                "market": "BEM",
                "timestamp_utc": str(timestamps.loc[idx]) if idx in timestamps.index else "",
                "side": side_label,
                "rhpf_submitted_mw": _finite(submitted.loc[idx]),
                "rhpf_bid_price_eur_per_mwh": _finite(bid.loc[idx]),
                "true_market_price_eur_per_mwh": _finite(true_price.loc[idx]),
                "rhpf_accepted": _finite(accepted.loc[idx]),
                "rhpf_guard_reason": str(guard_reason.loc[idx]),
            }
            record["classification"] = classify_bem_lineage_row(pd.Series(record), tol=tol)
            record["price_diff_eur"] = _finite(record["rhpf_bid_price_eur_per_mwh"]) - _finite(
                record["true_market_price_eur_per_mwh"]
            )
            rows.append(record)
    return pd.DataFrame.from_records(rows)


def build_rhpf_market_price_lineage_audit(
    scenario_dir: str | Path,
    *,
    tol: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scenario_dir = Path(scenario_dir)
    df = _load_hourly(scenario_dir)
    frames = [_build_da_rows(df, scenario_dir, tol), _build_bcm_rows(df, scenario_dir, tol), _build_bem_rows(df, scenario_dir, tol)]
    audit = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if audit.empty:
        return audit, {"scenario_dir": str(scenario_dir), "rows": 0, "status": "NO_DATA"}
    mismatch_classes = {
        "da_bug_limit_price_used_in_rhpf",
        "bcm_rhpf_underbids_cutoff",
        "bcm_rhpf_bid_above_cutoff_rejected",
        "bcm_auction_bug_awarded_above_cutoff",
        "bem_rhpf_underbids_or_overbids_true_activation_price",
        "bem_accepted_despite_wrong_price",
    }
    counts = audit["classification"].value_counts().to_dict()
    mismatch_rows = int(audit["classification"].isin(mismatch_classes).sum())
    auction_bug_rows = int((audit["classification"] == "bcm_auction_bug_awarded_above_cutoff").sum())
    summary: dict[str, Any] = {
        "scenario_dir": str(scenario_dir),
        "rows": int(len(audit)),
        "mismatch_rows": mismatch_rows,
        "auction_bug_rows": auction_bug_rows,
        "status": "FAIL" if mismatch_rows > 0 else "PASS",
    }
    for key, value in sorted(counts.items()):
        summary[f"count_{key}"] = int(value)
    return audit, summary


def _print_table(df: pd.DataFrame, max_rows: int) -> None:
    if df.empty:
        print("No audit rows found.")
        return
    cols = [c for c in ["market", "timestamp_utc", "bcm_capacity_block_id", "side", "rhpf_submitted_mw", "rhpf_bid_price_eur_per_mwh", "rhpf_bid_price_eur_per_mw_h", "true_market_price_eur_per_mwh", "true_market_price_eur_per_mw_h", "price_diff_eur", "classification"] if c in df.columns]
    print(df[cols].head(max_rows).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run root or scenario directory")
    parser.add_argument("--format", choices=["table", "csv", "json"], default="table")
    parser.add_argument("--max-rows", type=int, default=60)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    scenario_dirs = _scenario_dirs(args.run_dir)
    if not scenario_dirs:
        raise SystemExit(f"No scenario artifacts found below {args.run_dir}")
    audits: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for scenario_dir in scenario_dirs:
        audit, summary = build_rhpf_market_price_lineage_audit(scenario_dir, tol=float(args.tolerance))
        audits.append(audit)
        summaries.append(summary)
    all_audit = pd.concat([a for a in audits if not a.empty], ignore_index=True) if any(not a.empty for a in audits) else pd.DataFrame()
    failed = any(s.get("status") == "FAIL" for s in summaries)

    if args.format == "csv":
        all_audit.to_csv(index=False)
    elif args.format == "json":
        print(json.dumps({"status": "FAIL" if failed else "PASS", "summaries": summaries, "rows": all_audit.to_dict("records")}, indent=2, default=str))
    else:
        print(f"RHPF market-price lineage audit: {'FAIL' if failed else 'PASS'} scenarios_checked={len(summaries)}")
        print(pd.DataFrame(summaries).to_string(index=False))
        if not all_audit.empty:
            suspicious = all_audit[~all_audit["classification"].astype(str).str.contains("_ok_|no_bid", regex=True)]
            if not suspicious.empty:
                print("\nMarket-price mismatch rows:")
                _print_table(suspicious, int(args.max_rows))
            no_bid = all_audit[all_audit["classification"].astype(str).str.contains("no_bid", regex=False)]
            if not no_bid.empty:
                print("\nNo-bid explanatory rows:")
                _print_table(no_bid, min(20, int(args.max_rows)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
