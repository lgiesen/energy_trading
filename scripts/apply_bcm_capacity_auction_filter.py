#!/usr/bin/env python3
"""Post-process BCM capacity awards against realized capacity prices.

This tool is intentionally post-simulation accounting. It does not re-optimize or
re-run settlement. It flags/removes capacity remuneration for BCM capacity bids
whose submitted pay-as-bid capacity price exceeds the realized market capacity
price in the same hour and writes adjusted artifacts to a separate output root.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class ScenarioAdjustment:
    model: str
    quantile: str
    scenario_dir: str
    rows: int
    rejected_pos_hours: int
    rejected_neg_hours: int
    rejected_pos_mwh: float
    rejected_neg_mwh: float
    removed_capacity_revenue_pos_eur: float
    removed_capacity_revenue_neg_eur: float
    removed_linked_activation_revenue_pos_eur: float
    removed_linked_activation_revenue_neg_eur: float
    original_model_pnl_eur: float
    auction_adjusted_model_pnl_eur: float
    pnl_delta_eur: float
    output_hourly: str


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return pd.to_numeric(obj, errors="coerce")


def _series_sum(df: pd.DataFrame, candidates: Iterable[str]) -> float:
    for col in candidates:
        if col in df.columns:
            return float(_num(df, col).fillna(0.0).sum())
    return 0.0


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _infer_model_quantile(model_hourly_path: Path, input_root: Path) -> tuple[str, str]:
    rel = model_hourly_path.relative_to(input_root)
    # Expected: <model>_<pXX>/multi/<pXX_pXX>/model_hourly.parquet
    run_label = rel.parts[0] if len(rel.parts) >= 1 else "unknown"
    quantile = rel.parts[2] if len(rel.parts) >= 3 else "unknown"
    model = run_label.rsplit("_p", 1)[0] if "_p" in run_label else run_label
    return model, quantile


def _read_summary_pnl(scenario_dir: Path, hourly: pd.DataFrame) -> float:
    for name in ("backtest_summary.json", "model_summary.json"):
        path = scenario_dir / name
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for key in (
                "model_total_pnl_eur",
                "real_total_pnl_eur",
                "total_pnl_eur",
                "pnl_eur",
            ):
                if key in data:
                    try:
                        value = float(data[key])
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        return value
    return _series_sum(hourly, ["real_pnl_eur", "pnl_eur"])


def adjust_hourly(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    pos_bid_col = _first_existing(out, [
        "real_bcm_capacity_bid_price_pos_eur_per_mw_h",
        "real_settlement_cap_bid_price_pos_eur_mw",
        "bcm_capacity_bid_price_pos_eur_per_mw_h",
    ])
    neg_bid_col = _first_existing(out, [
        "real_bcm_capacity_bid_price_neg_eur_per_mw_h",
        "real_settlement_cap_bid_price_neg_eur_mw",
        "bcm_capacity_bid_price_neg_eur_per_mw_h",
    ])
    pos_true_col = _first_existing(out, [
        "real_bcm_capacity_cutoff_price_pos_eur_per_mw_h",
        "bcm_capacity_cutoff_price_pos_eur_per_mw_h",
        "target_afrr_capacity_price_pos",
        "aFRR_Capacity_Price_EUR_MW_Pos",
    ])
    neg_true_col = _first_existing(out, [
        "real_bcm_capacity_cutoff_price_neg_eur_per_mw_h",
        "bcm_capacity_cutoff_price_neg_eur_per_mw_h",
        "target_afrr_capacity_price_neg",
        "aFRR_Capacity_Price_EUR_MW_Neg",
    ])
    if not all([pos_bid_col, neg_bid_col, pos_true_col, neg_true_col]):
        missing = {
            "pos_bid_col": pos_bid_col,
            "neg_bid_col": neg_bid_col,
            "pos_true_col": pos_true_col,
            "neg_true_col": neg_true_col,
        }
        raise ValueError(f"Missing required BCM auction columns: {missing}")

    pos_bid = _num(out, pos_bid_col)
    neg_bid = _num(out, neg_bid_col)
    pos_true = _num(out, pos_true_col)
    neg_true = _num(out, neg_true_col)
    pos_bcm_mw = _num(out, "real_executed_bcm_capacity_pos_mw", 0.0).fillna(0.0)
    neg_bcm_mw = _num(out, "real_executed_bcm_capacity_neg_mw", 0.0).fillna(0.0)
    pos_cap_rev = _num(out, "real_bcm_capacity_revenue_pos_eur", 0.0).fillna(0.0)
    neg_cap_rev = _num(out, "real_bcm_capacity_revenue_neg_eur", 0.0).fillna(0.0)
    pos_act_rev = _num(out, "real_bcm_linked_pos_activation_revenue_eur", 0.0).fillna(0.0)
    neg_act_rev = _num(out, "real_bcm_linked_neg_activation_revenue_eur", 0.0).fillna(0.0)

    reject_pos = pos_bcm_mw.gt(1e-12) & pos_bid.notna() & pos_true.notna() & pos_bid.gt(pos_true + 1e-9)
    reject_neg = neg_bcm_mw.gt(1e-12) & neg_bid.notna() & neg_true.notna() & neg_bid.gt(neg_true + 1e-9)

    removed_pos_cap = pos_cap_rev.where(reject_pos, 0.0)
    removed_neg_cap = neg_cap_rev.where(reject_neg, 0.0)
    removed_pos_act = pos_act_rev.where(reject_pos, 0.0)
    removed_neg_act = neg_act_rev.where(reject_neg, 0.0)
    removed_total = removed_pos_cap + removed_neg_cap + removed_pos_act + removed_neg_act

    out["bcm_capacity_auction_filter_rejected_pos"] = reject_pos.astype(float)
    out["bcm_capacity_auction_filter_rejected_neg"] = reject_neg.astype(float)
    out["bcm_capacity_auction_filter_true_price_pos_eur_per_mw_h"] = pos_true
    out["bcm_capacity_auction_filter_true_price_neg_eur_per_mw_h"] = neg_true
    out["bcm_capacity_auction_filter_bid_minus_true_pos_eur_per_mw_h"] = pos_bid - pos_true
    out["bcm_capacity_auction_filter_bid_minus_true_neg_eur_per_mw_h"] = neg_bid - neg_true
    out["bcm_capacity_auction_filter_removed_capacity_revenue_pos_eur"] = removed_pos_cap
    out["bcm_capacity_auction_filter_removed_capacity_revenue_neg_eur"] = removed_neg_cap
    out["bcm_capacity_auction_filter_removed_linked_activation_revenue_pos_eur"] = removed_pos_act
    out["bcm_capacity_auction_filter_removed_linked_activation_revenue_neg_eur"] = removed_neg_act
    out["bcm_capacity_auction_filter_removed_total_revenue_eur"] = removed_total

    # Adjusted counterfactual columns. Original simulator columns remain intact.
    out["auction_adjusted_real_bcm_capacity_revenue_pos_eur"] = pos_cap_rev - removed_pos_cap
    out["auction_adjusted_real_bcm_capacity_revenue_neg_eur"] = neg_cap_rev - removed_neg_cap
    out["auction_adjusted_real_bcm_capacity_revenue_eur"] = (
        out["auction_adjusted_real_bcm_capacity_revenue_pos_eur"]
        + out["auction_adjusted_real_bcm_capacity_revenue_neg_eur"]
    )
    out["auction_adjusted_real_bcm_linked_pos_activation_revenue_eur"] = pos_act_rev - removed_pos_act
    out["auction_adjusted_real_bcm_linked_neg_activation_revenue_eur"] = neg_act_rev - removed_neg_act
    out["auction_adjusted_real_bcm_linked_activation_revenue_eur"] = (
        out["auction_adjusted_real_bcm_linked_pos_activation_revenue_eur"]
        + out["auction_adjusted_real_bcm_linked_neg_activation_revenue_eur"]
    )
    if "real_pnl_eur" in out.columns:
        out["auction_adjusted_real_pnl_eur"] = _num(out, "real_pnl_eur", 0.0).fillna(0.0) - removed_total
    if "real_net_cashflow_eur" in out.columns:
        out["auction_adjusted_real_net_cashflow_eur"] = _num(out, "real_net_cashflow_eur", 0.0).fillna(0.0) - removed_total

    # Null settlement-facing BCM reserve/revenue columns in the adjusted view only.
    for side, mask in (("pos", reject_pos), ("neg", reject_neg)):
        for col in [
            f"real_submitted_bcm_capacity_{side}_mw",
            f"real_locked_bcm_capacity_{side}_mw",
            f"real_executed_bcm_capacity_{side}_mw",
            f"real_bcm_locked_{side}_mw",
            f"real_bcm_{side}_obligation_mw",
            f"real_fixed_reserve_obligation_{side}_mw",
            f"fixed_reserve_obligation_{side}_mw",
            f"bcm_to_bem_energy_obligation_{side}_mw",
            f"real_bcm_linked_{side}_activation_mwh",
            f"real_bcm_linked_{side}_activation_revenue_eur",
        ]:
            if col in out.columns:
                out[f"auction_adjusted_{col}"] = _num(out, col, 0.0).fillna(0.0).mask(mask, 0.0)

    stats = {
        "rejected_pos_hours": int(reject_pos.sum()),
        "rejected_neg_hours": int(reject_neg.sum()),
        "rejected_pos_mwh": float(pos_bcm_mw.where(reject_pos, 0.0).sum()),
        "rejected_neg_mwh": float(neg_bcm_mw.where(reject_neg, 0.0).sum()),
        "removed_capacity_revenue_pos_eur": float(removed_pos_cap.sum()),
        "removed_capacity_revenue_neg_eur": float(removed_neg_cap.sum()),
        "removed_linked_activation_revenue_pos_eur": float(removed_pos_act.sum()),
        "removed_linked_activation_revenue_neg_eur": float(removed_neg_act.sum()),
        "removed_total_revenue_eur": float(removed_total.sum()),
        "true_price_column_pos": str(pos_true_col),
        "true_price_column_neg": str(neg_true_col),
        "bid_price_column_pos": str(pos_bid_col),
        "bid_price_column_neg": str(neg_bid_col),
    }
    return out, stats


def process(input_root: Path, output_root: Path, *, dry_run: bool) -> list[ScenarioAdjustment]:
    hourly_paths = sorted(input_root.glob("*_p*/multi/*/model_hourly.parquet"))
    if not hourly_paths:
        raise FileNotFoundError(f"No model_hourly.parquet files found below {input_root}")
    rows: list[ScenarioAdjustment] = []
    for hourly_path in hourly_paths:
        model, quantile = _infer_model_quantile(hourly_path, input_root)
        scenario_dir = hourly_path.parent
        hourly = pd.read_parquet(hourly_path)
        adjusted, stats = adjust_hourly(hourly)
        original_pnl = _read_summary_pnl(scenario_dir, hourly)
        adjusted_pnl = original_pnl - float(stats["removed_total_revenue_eur"])
        rel_hourly = hourly_path.relative_to(input_root)
        out_hourly = output_root / rel_hourly.with_name("model_hourly_auction_adjusted.parquet")
        if not dry_run:
            out_hourly.parent.mkdir(parents=True, exist_ok=True)
            adjusted.to_parquet(out_hourly, index=False)
            scenario_summary = {
                "source_model_hourly": str(hourly_path),
                "postprocess_type": "bcm_capacity_bid_above_true_price_filter",
                "scope_note": (
                    "Post-simulation counterfactual. Removes capacity and linked activation revenue for rejected BCM capacity "
                    "hours. It does not re-optimize, replay SoC, or recompute downstream ID/aux/degradation decisions."
                ),
                "model": model,
                "quantile": quantile,
                "original_model_pnl_eur": original_pnl,
                "auction_adjusted_model_pnl_eur": adjusted_pnl,
                **stats,
            }
            (out_hourly.parent / "bcm_capacity_auction_adjustment_summary.json").write_text(
                json.dumps(scenario_summary, indent=2), encoding="utf-8"
            )
        rows.append(
            ScenarioAdjustment(
                model=model,
                quantile=quantile,
                scenario_dir=str(scenario_dir),
                rows=int(len(hourly)),
                rejected_pos_hours=int(stats["rejected_pos_hours"]),
                rejected_neg_hours=int(stats["rejected_neg_hours"]),
                rejected_pos_mwh=float(stats["rejected_pos_mwh"]),
                rejected_neg_mwh=float(stats["rejected_neg_mwh"]),
                removed_capacity_revenue_pos_eur=float(stats["removed_capacity_revenue_pos_eur"]),
                removed_capacity_revenue_neg_eur=float(stats["removed_capacity_revenue_neg_eur"]),
                removed_linked_activation_revenue_pos_eur=float(stats["removed_linked_activation_revenue_pos_eur"]),
                removed_linked_activation_revenue_neg_eur=float(stats["removed_linked_activation_revenue_neg_eur"]),
                original_model_pnl_eur=float(original_pnl),
                auction_adjusted_model_pnl_eur=float(adjusted_pnl),
                pnl_delta_eur=float(adjusted_pnl - original_pnl),
                output_hourly=str(out_hourly),
            )
        )
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame([asdict(r) for r in rows])
        table.to_csv(output_root / "bcm_capacity_auction_adjusted_overview.csv", index=False)
        table.to_parquet(output_root / "bcm_capacity_auction_adjusted_overview.parquet", index=False)
        meta = {
            "input_root": str(input_root),
            "output_root": str(output_root),
            "scenario_count": len(rows),
            "postprocess_type": "bcm_capacity_bid_above_true_price_filter",
            "not_recomputed": [
                "SoC path",
                "future optimizer decisions",
                "ID repairs",
                "auxiliary consumption",
                "degradation and transaction costs after changed activation/SoC",
                "validity flags",
            ],
        }
        (output_root / "bcm_capacity_auction_adjustment_manifest.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply post-simulation BCM capacity auction filter to a run root.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    input_root = args.input_root
    output_root = args.output_root or input_root.with_name(input_root.name + "_bcm_auction_adjusted")
    rows = process(input_root, output_root, dry_run=bool(args.dry_run))
    table = pd.DataFrame([asdict(r) for r in rows])
    display_cols = [
        "model", "quantile", "rejected_pos_hours", "rejected_neg_hours",
        "rejected_pos_mwh", "rejected_neg_mwh",
        "removed_capacity_revenue_pos_eur", "removed_capacity_revenue_neg_eur",
        "removed_linked_activation_revenue_pos_eur", "removed_linked_activation_revenue_neg_eur",
        "original_model_pnl_eur", "auction_adjusted_model_pnl_eur", "pnl_delta_eur",
    ]
    print(table[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\ninput_root={input_root}")
    print(f"output_root={output_root}")
    print(f"dry_run={bool(args.dry_run)}")


if __name__ == "__main__":
    main()
