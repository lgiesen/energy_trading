#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _as_float(x: object) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _close(a: object, b: object, tol: float) -> bool:
    af = _as_float(a)
    bf = _as_float(b)
    if af is not None and bf is not None:
        return abs(af - bf) <= tol
    return str(a) == str(b)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_metrics_payload(metrics: dict, *, tol: float) -> list[str]:
    errs: list[str] = []
    hpo_best = metrics.get("hpo_best_params", {})
    if not isinstance(hpo_best, dict) or not hpo_best:
        return errs
    effective = metrics.get("effective_model_params")
    if isinstance(effective, dict):
        for k, hv in hpo_best.items():
            if k in effective and not _close(hv, effective[k], tol):
                errs.append(f"effective_model_params mismatch {k}: hpo={hv} effective={effective[k]}")
    per_target_policy = metrics.get("per_target_policy")
    if isinstance(per_target_policy, dict):
        for tgt, pol in per_target_policy.items():
            if not isinstance(pol, dict):
                continue
            xgb = pol.get("xgb_params")
            if isinstance(xgb, dict):
                for k, hv in hpo_best.items():
                    if k in xgb and not _close(hv, xgb[k], tol):
                        errs.append(f"{tgt}.xgb_params mismatch {k}: hpo={hv} effective={xgb[k]}")
    return errs


def main() -> int:
    p = argparse.ArgumentParser(description="Verify HPO best_params were applied in training outputs.")
    p.add_argument("--run-manifest", required=True)
    p.add_argument("--hpo-artifact-map", default="")
    p.add_argument("--tol", type=float, default=1e-9)
    args = p.parse_args()

    manifest = _load_json(Path(args.run_manifest))
    errs: list[str] = []
    for bundle in manifest.get("bundles", {}).values():
        metrics_paths = bundle.get("metrics_paths", [])
        if isinstance(metrics_paths, str):
            metrics_paths = [metrics_paths]
        for mp in metrics_paths:
            m = _load_json(Path(mp))
            errs.extend([f"{mp}: {e}" for e in _verify_metrics_payload(m, tol=float(args.tol))])

    if errs:
        print("[FAIL] HPO verification mismatches:")
        for e in errs:
            print(" -", e)
        return 1
    print("[OK] HPO params applied check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
