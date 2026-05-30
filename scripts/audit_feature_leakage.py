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
    "target_afrr_activation_price_vwap_pos_raw",
    "target_afrr_activation_price_vwap_neg_raw",
    "target_da_price",
}


def _auto_suspicious(cols: list[str]) -> set[str]:
    out: set[str] = set()
    for c in cols:
        if c.startswith("target_"):
            out.add(c)
        if c.endswith("_raw"):
            out.add(c)
        if "activation_price" in c and "_lag_" not in c:
            out.add(c)
        if "bid_avg_activation_price" in c or "bid_vwap_activation_price" in c:
            out.add(c)
    return out


def _verdict(col: str, row: dict[str, object]) -> tuple[str, str]:
    if row["in_feature_config_targets"]:
        return "ok", "target column (y), not feature"
    if row["used_in_X_train_xgb"] or row["used_in_X_train_linear"] or row["used_in_X_train_tft"]:
        if col.startswith("target_"):
            return "hard_leak", "target_* entered model features"
        if col.endswith("_raw"):
            return "hard_leak", "*_raw entered model features"
        if col in EXPLICIT_SUSPICIOUS:
            return "hard_leak", "explicitly forbidden unlagged bid/raw feature in model features"
        if "activation_price" in col and "_lag_" not in col:
            return "warn", "unlagged activation-price feature used; timing justification required"
        return "warn", "suspicious feature is used by at least one model"
    if re.search(r"_lag_\d+h$", col):
        return "ok", "lagged feature; causally valid if lag construction is correct"
    return "ok", "not used in model X features"


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit potential feature leakage for bundle feature matrices.")
    ap.add_argument("--feature-config", default="data/model_input/feature_config.json")
    ap.add_argument("--bundle", default="afrr", choices=["da", "afrr"])
    ap.add_argument("--base-dir", default="data/model_input")
    args = ap.parse_args()

    cfg_path = Path(args.feature_config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    bcfg = cfg["bundles"][args.bundle]

    train_path = Path(bcfg["files"]["train"])
    train_df = pd.read_parquet(train_path)
    all_cols = list(train_df.columns)
    feat_cols = list(bcfg.get("features", []))
    target_cols = list(bcfg.get("targets", []))

    suspicious = sorted(set(EXPLICIT_SUSPICIOUS) | _auto_suspicious(all_cols) | _auto_suspicious(feat_cols))

    xgb_x, _ = load_processed_data(bundle=args.bundle, split="train", base_dir=args.base_dir)
    linear_x, _ = load_processed_data(bundle=args.bundle, split="train", base_dir=args.base_dir)

    # TFT starts from feature_config features and applies classification/pruning.
    tft_x_cols = set(feat_cols)

    rows: list[dict[str, object]] = []
    hard_leak_count = 0
    for col in suspicious:
        rec: dict[str, object] = {
            "column": col,
            "exists_in_afrr_train": col in all_cols,
            "in_feature_config_features": col in feat_cols,
            "in_feature_config_targets": col in target_cols,
            "used_in_X_train_xgb": col in xgb_x.columns,
            "used_in_X_train_linear": col in linear_x.columns,
            "used_in_X_train_tft": col in tft_x_cols,
        }
        verdict, reason = _verdict(col, rec)
        rec["verdict"] = verdict
        rec["reason"] = reason
        if verdict == "hard_leak":
            hard_leak_count += 1
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values(["verdict", "column"]).reset_index(drop=True)
    print(out.to_string(index=False))
    if hard_leak_count > 0:
        print(f"\n[FAIL] hard leakage findings: {hard_leak_count}")
        return 2
    print("\n[OK] no hard leakage findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
