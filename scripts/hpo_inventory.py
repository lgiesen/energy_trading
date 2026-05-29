#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
MODELS = ["xgboost", "linear", "tft"]


def _json_name(model: str, bundle: str, target: str) -> str:
    if model == "xgboost":
        return f"xgb_optuna_{bundle}_{target}.json"
    if model == "linear":
        return f"linear_sgd_tuning_{bundle}_{target}.json"
    if model == "tft":
        return f"tft_optuna_{bundle}_{target}.json"
    raise ValueError(model)


def _csv_name(model: str, bundle: str, target: str) -> str:
    return _json_name(model, bundle, target).replace(".json", "_trials.csv")


def main() -> int:
    p = argparse.ArgumentParser(description="Inventory/validate HPO artifacts for all model-target pairs.")
    p.add_argument("--hpo-out-dir", default="artifacts/hpo")
    p.add_argument("--out-csv", default="artifacts/hpo/hpo_inventory.csv")
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.hpo_out_dir)
    rows: list[dict[str, object]] = []
    failures: list[str] = []

    for model in MODELS:
        for bundle, target in [("da", DA_TARGET), *[("afrr", t) for t in AFRR_TARGETS]]:
            hpo_json = out_dir / _json_name(model, bundle, target)
            trials_csv = out_dir / _csv_name(model, bundle, target)
            exists = hpo_json.exists()
            has_best = False
            best_obj = ""
            sel_metric = ""
            if exists:
                try:
                    payload = json.loads(hpo_json.read_text(encoding="utf-8"))
                    has_best = isinstance(payload.get("best_params"), dict)
                    best_obj = payload.get("best_objective_value", "")
                    sel_metric = payload.get("selection_metric", "")
                    got_bundle = str(payload.get("bundle", "")).strip()
                    got_target = str(payload.get("target_col", "")).strip()
                    if got_bundle and got_bundle != bundle:
                        failures.append(f"{hpo_json}: bundle mismatch expected={bundle} got={got_bundle}")
                    if got_target and got_target != target:
                        failures.append(f"{hpo_json}: target mismatch expected={target} got={got_target}")
                except Exception as exc:
                    failures.append(f"{hpo_json}: unreadable JSON ({exc})")
            if not exists:
                failures.append(f"missing: {hpo_json}")
            elif not has_best:
                failures.append(f"missing best_params: {hpo_json}")

            rows.append(
                {
                    "model_type": model,
                    "bundle": bundle,
                    "target_col": target,
                    "hpo_json": str(hpo_json),
                    "trials_csv": str(trials_csv),
                    "exists": int(exists),
                    "has_best_params": int(has_best),
                    "best_objective_value": best_obj,
                    "selection_metric": sel_metric,
                }
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model_type",
                "bundle",
                "target_col",
                "hpo_json",
                "trials_csv",
                "exists",
                "has_best_params",
                "best_objective_value",
                "selection_metric",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] wrote HPO inventory: {out_csv}")
    if failures:
        print("[WARN] validation issues:")
        for m in failures:
            print("-", m)
    if args.validate and failures:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
