#!/usr/bin/env python3
"""Read-only audit for BCM capacity auction and RHPF pay-as-bid bid lineage.

The audit checks block/side-level capacity bids against the true capacity cutoff.
It does not run simulations and does not modify artifacts.
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


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


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


def classify_bcm_capacity_auction_row(row: pd.Series, *, tol: float = 1e-9) -> str:
    """Classify one block/side audit row.

    The strict auction bug condition is awarded MW with submitted bid price above
    the true cutoff. A model bid higher than RHPF is not a bug when it still sits
    below the cutoff.
    """
    model_awarded = _finite_or_nan(row.get("model_awarded_mw"))
    rhpf_awarded = _finite_or_nan(row.get("rhpf_awarded_mw"))
    model_bid = _finite_or_nan(row.get("model_bid_price_eur_per_mw_h"))
    rhpf_bid = _finite_or_nan(row.get("rhpf_bid_price_eur_per_mw_h"))
    cutoff = _finite_or_nan(row.get("true_capacity_cutoff_price_eur_per_mw_h"))
    rhpf_submitted = _finite_or_nan(row.get("rhpf_submitted_mw"))
    rhpf_zero_reason = str(row.get("rhpf_zero_reason", "") or "").lower()
    rhpf_headroom_reason = str(row.get("rhpf_headroom_rejection_reason", "") or "").lower()

    if model_awarded > tol and math.isfinite(model_bid) and math.isfinite(cutoff) and model_bid > cutoff + tol:
        return "auction_bug_model_awarded_above_cutoff"
    if rhpf_awarded > tol and math.isfinite(rhpf_bid) and math.isfinite(cutoff) and rhpf_bid > cutoff + tol:
        return "auction_bug_rhpf_awarded_above_cutoff"
    if rhpf_submitted > tol and math.isfinite(rhpf_bid) and math.isfinite(cutoff) and rhpf_bid < cutoff - tol:
        return "rhpf_underbids_cutoff"
    if rhpf_submitted <= tol and ("headroom" in rhpf_zero_reason or "headroom" in rhpf_headroom_reason):
        return "rhpf_no_bid_due_to_physical_guard"
    if rhpf_submitted <= tol and model_awarded > tol:
        return "rhpf_no_bid_due_to_side_choice"
    if (
        model_awarded > tol
        and math.isfinite(model_bid)
        and math.isfinite(rhpf_bid)
        and math.isfinite(cutoff)
        and model_bid > rhpf_bid + tol
        and model_bid <= cutoff + tol
    ):
        return "model_bid_higher_than_rhpf_but_valid"
    if model_awarded > tol and math.isfinite(model_bid) and math.isfinite(cutoff) and model_bid <= cutoff + tol:
        return "auction_ok_model_bid_below_cutoff"
    return "no_award_or_not_comparable"


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
    if pf_path.exists():
        pf = pd.read_parquet(pf_path)
        if "timestamp_utc" in hourly.columns and "timestamp_utc" in pf.columns:
            pf_cols = [c for c in pf.columns if c == "timestamp_utc" or c.startswith("perfect_foresight_")]
            missing_pf_cols = [c for c in pf_cols if c not in hourly.columns]
            if missing_pf_cols:
                hourly = hourly.merge(
                    pf[["timestamp_utc", *missing_pf_cols]],
                    on="timestamp_utc",
                    how="left",
                    suffixes=("", "_pf_file"),
                )
    return hourly


def build_bcm_capacity_auction_audit(
    scenario_dir: str | Path,
    *,
    tol: float = 1e-9,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scenario_dir = Path(scenario_dir)
    hourly = _load_hourly(scenario_dir)
    if "timestamp_utc" not in hourly.columns:
        raise ValueError(f"{scenario_dir}: backtest_hourly.parquet has no timestamp_utc column")

    df = hourly.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    block_col = _first_existing(
        df.columns,
        ["bcm_capacity_block_id", "bcm_product_block_id", "reserve_product_block_id"],
    )
    if block_col is None:
        df["_audit_block_id"] = df["timestamp_utc"].dt.floor("4h").astype(str)
        block_col = "_audit_block_id"

    if "bcm_capacity_block_start_utc" in df.columns:
        df["_delivery_start_utc"] = pd.to_datetime(df["bcm_capacity_block_start_utc"], utc=True, errors="coerce")
    else:
        df["_delivery_start_utc"] = df.groupby(block_col)["timestamp_utc"].transform("min")
    if "bcm_capacity_block_end_utc" in df.columns:
        df["_delivery_end_utc"] = pd.to_datetime(df["bcm_capacity_block_end_utc"], utc=True, errors="coerce")
    else:
        df["_delivery_end_utc"] = df.groupby(block_col)["timestamp_utc"].transform("max") + pd.Timedelta(hours=1)

    records: list[dict[str, Any]] = []
    for block_id, blk in df.dropna(subset=["timestamp_utc"]).groupby(block_col, dropna=False, sort=True):
        if str(block_id).lower() in {"", "nan", "none"}:
            continue
        for side in SIDES:
            cutoff = _group_value(
                _num_series(
                    blk,
                    [
                        f"bcm_capacity_cutoff_price_{side}_eur_per_mw_h",
                        "bcm_capacity_cutoff_price_eur_per_mw_h",
                    ],
                ),
                prefer="mean",
            )
            model_submitted = _group_value(
                _num_series(
                    blk,
                    [
                        f"real_submitted_bcm_capacity_{side}_mw",
                        f"submitted_bcm_capacity_{side}_mw",
                        f"submitted_reserve_{side}_mw",
                    ],
                )
            )
            model_bid = _group_value(
                _num_series(
                    blk,
                    [
                        f"real_bcm_capacity_bid_price_{side}_eur_per_mw_h",
                        f"real_bcm_lockbook_capacity_bid_price_{side}_eur_per_mw_h",
                        f"real_bcm_settlement_capacity_price_resolved_{side}_eur_per_mw_h",
                        f"bcm_capacity_bid_price_{side}_eur_per_mw_h",
                        f"bcm_lockbook_capacity_bid_price_{side}_eur_per_mw_h",
                        f"settlement_cap_bid_price_{side}_eur_mw",
                    ],
                ),
                prefer="mean",
            )
            model_awarded = _group_value(
                _num_series(
                    blk,
                    [
                        f"real_awarded_capacity_{side}_mw",
                        f"real_locked_bcm_capacity_{side}_mw",
                        f"bcm_awarded_capacity_{side}_mw",
                        f"locked_bcm_capacity_{side}_mw",
                    ],
                )
            )
            rhpf_submitted = _group_value(
                _num_series(
                    blk,
                    [
                        f"perfect_foresight_submitted_bcm_capacity_{side}_mw",
                        f"pf_submitted_bcm_capacity_{side}_mw",
                    ],
                )
            )
            rhpf_bid = _group_value(
                _num_series(
                    blk,
                    [
                        f"perfect_foresight_bcm_capacity_bid_price_{side}_eur_per_mw_h",
                        f"perfect_foresight_bcm_lockbook_capacity_bid_price_{side}_eur_per_mw_h",
                        f"perfect_foresight_bcm_settlement_capacity_price_resolved_{side}_eur_per_mw_h",
                        f"perfect_foresight_settlement_cap_bid_price_{side}_eur_mw",
                        f"perfect_foresight_bcm_p70_capacity_bid_price_{side}_eur_per_mw_h",
                    ],
                ),
                prefer="mean",
            )
            rhpf_awarded = _group_value(
                _num_series(
                    blk,
                    [
                        f"perfect_foresight_awarded_capacity_{side}_mw",
                        f"perfect_foresight_locked_bcm_capacity_{side}_mw",
                        f"perfect_foresight_bcm_locked_{side}_mw",
                    ],
                )
            )
            zero_reason = _group_text(
                _str_series(
                    blk,
                    [
                        "perfect_foresight_bcm_precommit_zero_reason",
                        "perfect_foresight_bcm_zero_reason",
                    ],
                )
            )
            required = _group_value(
                _num_series(
                    blk,
                    [f"perfect_foresight_required_headroom_{side}_mwh", f"required_headroom_{side}_mwh"],
                )
            )
            available = _group_value(
                _num_series(
                    blk,
                    [f"perfect_foresight_available_headroom_{side}_mwh", f"available_headroom_{side}_mwh"],
                )
            )
            headroom_reason = ""
            if math.isfinite(required) and math.isfinite(available) and required > available + tol:
                headroom_reason = "headroom_infeasible"
            if "headroom" in zero_reason.lower():
                headroom_reason = zero_reason

            record = {
                "scenario_dir": str(scenario_dir),
                "bcm_capacity_block_id": str(block_id),
                "side": side,
                "delivery_start_utc": str(pd.to_datetime(blk["_delivery_start_utc"], utc=True, errors="coerce").dropna().min()),
                "delivery_end_utc": str(pd.to_datetime(blk["_delivery_end_utc"], utc=True, errors="coerce").dropna().max()),
                "true_capacity_cutoff_price_eur_per_mw_h": cutoff,
                "model_submitted_mw": model_submitted,
                "model_bid_price_eur_per_mw_h": model_bid,
                "model_awarded_mw": model_awarded,
                "rhpf_submitted_mw": rhpf_submitted,
                "rhpf_bid_price_eur_per_mw_h": rhpf_bid,
                "rhpf_awarded_mw": rhpf_awarded,
                "rhpf_zero_reason": zero_reason,
                "rhpf_headroom_rejection_reason": headroom_reason,
            }
            record.update(_boolean_flags(record, tol=tol))
            record["classification"] = classify_bcm_capacity_auction_row(pd.Series(record), tol=tol)
            records.append(record)

    audit = pd.DataFrame.from_records(records)
    if audit.empty:
        summary = {"scenario_dir": str(scenario_dir), "rows": 0, "status": "NO_DATA"}
        return audit, summary

    bug_mask = audit["classification"].astype(str).str.startswith("auction_bug_")
    summary = {
        "scenario_dir": str(scenario_dir),
        "rows": int(len(audit)),
        "auction_bug_rows": int(bug_mask.sum()),
        "rhpf_underbids_cutoff_rows": int((audit["classification"] == "rhpf_underbids_cutoff").sum()),
        "model_bid_higher_than_rhpf_but_valid_rows": int(
            (audit["classification"] == "model_bid_higher_than_rhpf_but_valid").sum()
        ),
        "rhpf_no_bid_due_to_physical_guard_rows": int(
            (audit["classification"] == "rhpf_no_bid_due_to_physical_guard").sum()
        ),
        "rhpf_no_bid_due_to_side_choice_rows": int(
            (audit["classification"] == "rhpf_no_bid_due_to_side_choice").sum()
        ),
        "status": "FAIL" if bool(bug_mask.any()) else "PASS",
    }
    return audit, summary


def _boolean_flags(record: dict[str, Any], *, tol: float) -> dict[str, Any]:
    cutoff = _finite_or_nan(record.get("true_capacity_cutoff_price_eur_per_mw_h"))
    model_bid = _finite_or_nan(record.get("model_bid_price_eur_per_mw_h"))
    rhpf_bid = _finite_or_nan(record.get("rhpf_bid_price_eur_per_mw_h"))
    model_awarded = _finite_or_nan(record.get("model_awarded_mw"))
    rhpf_awarded = _finite_or_nan(record.get("rhpf_awarded_mw"))
    rhpf_submitted = _finite_or_nan(record.get("rhpf_submitted_mw"))
    model_bid_le_cutoff = bool(math.isfinite(model_bid) and math.isfinite(cutoff) and model_bid <= cutoff + tol)
    rhpf_bid_le_cutoff = bool(math.isfinite(rhpf_bid) and math.isfinite(cutoff) and rhpf_bid <= cutoff + tol)
    model_awarded_with_bid_above_cutoff = bool(
        model_awarded > tol and math.isfinite(model_bid) and math.isfinite(cutoff) and model_bid > cutoff + tol
    )
    rhpf_awarded_with_bid_above_cutoff = bool(
        rhpf_awarded > tol and math.isfinite(rhpf_bid) and math.isfinite(cutoff) and rhpf_bid > cutoff + tol
    )
    model_bid_gt_rhpf_bid = bool(math.isfinite(model_bid) and math.isfinite(rhpf_bid) and model_bid > rhpf_bid + tol)
    return {
        "model_bid_le_cutoff": float(model_bid_le_cutoff),
        "rhpf_bid_le_cutoff": float(rhpf_bid_le_cutoff),
        "model_awarded_with_bid_above_cutoff": float(model_awarded_with_bid_above_cutoff),
        "rhpf_awarded_with_bid_above_cutoff": float(rhpf_awarded_with_bid_above_cutoff),
        "model_bid_gt_rhpf_bid": float(model_bid_gt_rhpf_bid),
        "model_bid_gt_rhpf_bid_but_le_cutoff": float(model_bid_gt_rhpf_bid and model_bid_le_cutoff),
        "rhpf_submitted_but_under_cutoff": float(
            rhpf_submitted > tol and math.isfinite(rhpf_bid) and math.isfinite(cutoff) and rhpf_bid < cutoff - tol
        ),
        "rhpf_no_bid_or_opposite_side": float(rhpf_submitted <= tol and model_awarded > tol),
    }


def _print_table(df: pd.DataFrame, *, max_rows: int) -> None:
    if df.empty:
        print("No BCM auction audit rows found.")
        return
    cols = [
        "bcm_capacity_block_id",
        "side",
        "true_capacity_cutoff_price_eur_per_mw_h",
        "model_bid_price_eur_per_mw_h",
        "model_awarded_mw",
        "rhpf_bid_price_eur_per_mw_h",
        "rhpf_awarded_mw",
        "classification",
    ]
    print(df[cols].head(max_rows).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory or scenario directory to audit")
    parser.add_argument("--format", choices=["table", "csv", "json"], default="table")
    parser.add_argument("--max-rows", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    scenario_dirs = _scenario_dirs(args.run_dir)
    if not scenario_dirs:
        raise SystemExit(f"No scenario artifacts found below {args.run_dir}")

    audits: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for scenario_dir in scenario_dirs:
        audit, summary = build_bcm_capacity_auction_audit(scenario_dir, tol=float(args.tolerance))
        audits.append(audit)
        summaries.append(summary)

    all_audit = pd.concat([a for a in audits if not a.empty], ignore_index=True) if any(not a.empty for a in audits) else pd.DataFrame()
    failed = any(s.get("status") == "FAIL" for s in summaries)

    if args.format == "csv":
        all_audit.to_csv(index=False)
    elif args.format == "json":
        payload = {"status": "FAIL" if failed else "PASS", "summaries": summaries, "rows": all_audit.to_dict("records")}
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"BCM capacity auction audit: {'FAIL' if failed else 'PASS'} scenarios_checked={len(summaries)}")
        print(pd.DataFrame(summaries).to_string(index=False))
        suspicious = all_audit[
            all_audit.get("classification", pd.Series(dtype=str)).isin(
                [
                    "auction_bug_model_awarded_above_cutoff",
                    "auction_bug_rhpf_awarded_above_cutoff",
                    "rhpf_underbids_cutoff",
                    "model_bid_higher_than_rhpf_but_valid",
                    "rhpf_no_bid_due_to_physical_guard",
                ]
            )
        ] if not all_audit.empty else all_audit
        if not suspicious.empty:
            print("\nSuspicious / explanatory rows:")
            _print_table(suspicious, max_rows=int(args.max_rows))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
