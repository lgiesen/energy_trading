#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

DA_TARGET = "target_da_price"
AFRR_TARGETS = [
    "target_afrr_activation_price_vwap_pos",
    "target_afrr_activation_price_vwap_neg",
    "target_afrr_activation_rate_pos",
    "target_afrr_activation_rate_neg",
    "target_afrr_capacity_price_pos",
    "target_afrr_capacity_price_neg",
]


def _expected_filename(model_type: str, bundle: str, target_col: str) -> str:
    if model_type == "xgboost":
        return f"xgb_optuna_{bundle}_{target_col}.json"
    if model_type == "linear":
        return f"linear_sgd_tuning_{bundle}_{target_col}.json"
    if model_type == "tft":
        return f"tft_optuna_{bundle}_{target_col}.json"
    raise ValueError(f"Unsupported model-type: {model_type}")


def main() -> int:
    p = argparse.ArgumentParser(description="Build target->HPO artifact map for DA+aFRR.")
    p.add_argument("--model-type", choices=["xgboost", "linear", "tft"], required=True)
    p.add_argument("--hpo-out-dir", default="artifacts/hpo")
    p.add_argument("--out", required=True)
    p.add_argument("--allow-missing", action="store_true", help="Do not fail on missing artifacts.")
    args = p.parse_args()

    out_dir = Path(args.hpo_out_dir)
    mapping: dict[str, str] = {}
    missing: list[str] = []

    targets = [("da", DA_TARGET), *[("afrr", t) for t in AFRR_TARGETS]]
    for bundle, target in targets:
        fp = out_dir / _expected_filename(args.model_type, bundle, target)
        if not fp.exists():
            missing.append(str(fp))
        mapping[target] = str(fp)

    if missing and not args.allow_missing:
        raise FileNotFoundError("Missing HPO artifacts:\n" + "\n".join(missing))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"[OK] wrote HPO artifact map: {out_path}")
    if missing:
        print(f"[WARN] missing entries: {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
