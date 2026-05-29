#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description="Create combined thesis manifest with canonical activation-neg target")
    p.add_argument("--base-manifest", required=True, help="Existing full manifest used as baseline")
    p.add_argument("--canonical-neg-manifest", required=True, help="Manifest containing retrained activation-neg artifacts")
    p.add_argument("--out", required=True, help="Output combined manifest path")
    args = p.parse_args()

    base = _load(Path(args.base_manifest))
    neg = _load(Path(args.canonical_neg_manifest))

    out = dict(base)
    out.setdefault("bundles", {}).setdefault("afrr", {})
    out.setdefault("target_value_mode", {})

    neg_long = neg.get("bundles", {}).get("afrr", {}).get("predictions_long", {})
    base_long = out["bundles"]["afrr"].get("predictions_long", {})
    for split, m in neg_long.items():
        if not isinstance(m, dict):
            continue
        if "pred_afrr_activation_price_neg" not in m:
            continue
        base_long.setdefault(split, {})
        base_long[split]["pred_afrr_activation_price_neg"] = m["pred_afrr_activation_price_neg"]
    out["bundles"]["afrr"]["predictions_long"] = base_long

    out["target_value_mode"]["pred_afrr_activation_price_neg"] = "canonical_economic"
    sim = out.setdefault("simulation", {})
    canonical_targets = set(sim.get("canonical_economic_targets", []) or [])
    canonical_targets.add("pred_afrr_activation_price_neg")
    sim["canonical_economic_targets"] = sorted(canonical_targets)
    transformed_targets = set(sim.get("transformed_targets", []) or [])
    transformed_targets.add("pred_afrr_activation_price_neg")
    sim["transformed_targets"] = sorted(transformed_targets)

    out.setdefault("thesis", {})
    out["thesis"]["combined_manifest_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    out["thesis"]["base_manifest"] = str(Path(args.base_manifest).resolve())
    out["thesis"]["canonical_neg_manifest"] = str(Path(args.canonical_neg_manifest).resolve())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] wrote combined manifest: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
