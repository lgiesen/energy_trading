#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_trading.models.prepare_ml_bundles import load_processed_data  # noqa: E402

EXPLICIT_SUSPICIOUS = {
    "afrr_bid_avg_activation_price_neg",
    "afrr_bid_avg_activation_price_pos",
    "afrr_bid_vwap_activation_price_neg",
    "afrr_bid_vwap_activation_price_pos",
    "bid_signed_vwap_eur_mwh_neg",
    "bid_signed_vwap_eur_mwh_pos",
    "bid_alloc_mw_neg",
    "bid_alloc_mw_pos",
    "target_afrr_activation_price_vwap_pos_raw",
    "target_afrr_activation_price_vwap_neg_raw",
    "target_da_price",
}

FOREIGN_DA_RE = re.compile(r"^da_price_[A-Z0-9]+$")

PATTERNS = [
    re.compile(r"^bid_"),
    re.compile(r".*_alloc_.*"),
    re.compile(r".*_awarded_.*"),
    re.compile(r".*_award_.*"),
    re.compile(r".*_vwap_.*"),
    re.compile(r".*_clearing_.*"),
    re.compile(r".*_auction_.*"),
    FOREIGN_DA_RE,
]


def _auto_suspicious(cols: list[str]) -> set[str]:
    out: set[str] = set()
    for c in cols:
        if c.startswith("target_"):
            out.add(c)
        if c.endswith("_raw"):
            out.add(c)
        if "activation_price" in c and "_lag_" not in c:
            out.add(c)
        if any(p.match(c) for p in PATTERNS):
            out.add(c)
    return out


def _timing_semantics(col: str) -> tuple[str, str, str]:
    if re.search(r"_lag_\d+h$", col):
        return "historical_lagged_value", "likely_yes", "lagged feature"
    if col.startswith("target_"):
        return "target_column", "no", "explicit supervised label"
    if col.endswith("_raw"):
        return "raw_target_like", "no", "raw target-like provenance column"
    if col in {"bid_alloc_mw_neg", "bid_alloc_mw_pos"}:
        return "allocation_or_award_stat", "unknown", "allocation/award-like; likely ex-post"
    if col in {"bid_signed_vwap_eur_mwh_neg", "bid_signed_vwap_eur_mwh_pos"}:
        return "bid_vwap_stat", "unknown", "bid VWAP-like settlement statistic"
    if col.startswith("afrr_bid_"):
        return "bid_or_auction_result", "unknown", "aFRR bid/auction derived feature"
    if FOREIGN_DA_RE.match(col):
        return "foreign_da_realized_same_hour", "unknown", "foreign DA price without explicit PIT suffix"
    return "unknown_or_derived", "unknown", "timing not proven by name"


def _verdict(col: str, row: dict[str, object]) -> tuple[str, str]:
    if row["in_feature_config_targets"]:
        return "ok", "target column (y), not feature"

    used = row["used_in_X_train_xgb"] or row["used_in_X_train_linear"] or row["used_in_X_train_tft"]
    if not used:
        return "ok", "not used in model X features"
    if re.search(r"_lag_\\d+h$", col):
        return "ok", "lagged feature in use; causality depends on lag construction pipeline"

    if col.startswith("target_"):
        return "hard_leak", "target_* entered model features"
    if col.endswith("_raw"):
        return "hard_leak", "*_raw entered model features"
    if col in {
        "afrr_bid_avg_activation_price_neg",
        "afrr_bid_avg_activation_price_pos",
        "afrr_bid_vwap_activation_price_neg",
        "afrr_bid_vwap_activation_price_pos",
        "bid_alloc_mw_neg",
        "bid_alloc_mw_pos",
        "bid_signed_vwap_eur_mwh_neg",
        "bid_signed_vwap_eur_mwh_pos",
    }:
        return "hard_leak", "unlagged bid/allocation VWAP/activation stat entered model features"
    if FOREIGN_DA_RE.match(col):
        return "hard_leak", "foreign DA same-hour price in X without PIT/lag proof"
    if "activation_price" in col and "_lag_" not in col:
        return "warn", "unlagged activation-price feature used; timing proof required"
    return "warn", "suspicious feature used; timing ambiguous"


def _audit_bundle(cfg: dict, bundle: str, base_dir: Path) -> tuple[pd.DataFrame, int]:
    bcfg = cfg["bundles"][bundle]
    train_path = Path(bcfg["files"]["train"])
    train_df = pd.read_parquet(train_path)
    all_cols = list(train_df.columns)
    feat_cols = list(bcfg.get("features", []))
    target_cols = list(bcfg.get("targets", []))

    suspicious = sorted(set(EXPLICIT_SUSPICIOUS) | _auto_suspicious(all_cols) | _auto_suspicious(feat_cols))

    xgb_x, _ = load_processed_data(bundle=bundle, split="train", base_dir=base_dir)
    linear_x, _ = load_processed_data(bundle=bundle, split="train", base_dir=base_dir)
    tft_x_cols = set(feat_cols)

    rows: list[dict[str, object]] = []
    hard_leak_count = 0
    for col in suspicious:
        sem, known, sem_reason = _timing_semantics(col)
        rec: dict[str, object] = {
            "column": col,
            "bundle": bundle,
            "exists_in_train": col in all_cols,
            "in_feature_config_features": col in feat_cols,
            "in_feature_config_targets": col in target_cols,
            "used_in_X_train_xgb": col in xgb_x.columns,
            "used_in_X_train_linear": col in linear_x.columns,
            "used_in_X_train_tft": col in tft_x_cols,
            "known_at_prediction_time": known,
            "timing_semantics": sem,
            "timing_reason": sem_reason,
        }
        verdict, reason = _verdict(col, rec)
        rec["verdict"] = verdict
        rec["reason"] = reason
        if verdict == "hard_leak":
            hard_leak_count += 1
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values(["verdict", "column"]).reset_index(drop=True)
    return out, hard_leak_count


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit potential feature leakage for DA/aFRR bundle feature matrices.")
    ap.add_argument("--feature-config", default="data/model_input/feature_config.json")
    ap.add_argument("--bundle", default="afrr", choices=["da", "afrr", "both"])
    ap.add_argument("--base-dir", default="data/model_input")
    args = ap.parse_args()

    cfg_path = Path(args.feature_config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base_dir = Path(args.base_dir)

    bundles = ["da", "afrr"] if args.bundle == "both" else [args.bundle]
    total_hard = 0
    for b in bundles:
        out, hard = _audit_bundle(cfg, b, base_dir)
        total_hard += hard
        print(out.to_string(index=False))
        print("")

    if total_hard > 0:
        print(f"[FAIL] hard leakage findings: {total_hard}")
        return 2
    print("[OK] no hard leakage findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
