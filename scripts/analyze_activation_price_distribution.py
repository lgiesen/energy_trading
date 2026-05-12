"""Plot true vs predicted aFRR activation-price distributions.

Usage:
  python3 scripts/analyze_activation_price_distribution.py \
    --run-manifest artifacts/model_runs/<run_id>/manifest.json \
    --split test \
    --out-dir artifacts/reports/activation_price_distribution
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resolve_manifest(path: Path) -> tuple[Path, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "manifest_path" in payload:
        path = Path(str(payload["manifest_path"]))
        payload = json.loads(path.read_text(encoding="utf-8"))
    return path, payload


def _resolve_path(configured: str, manifest_dir: Path) -> Path:
    p = Path(configured)
    if p.is_absolute():
        return p
    cands = [
        (manifest_dir / p).resolve(),
        (Path.cwd() / p).resolve(),
    ]
    for c in cands:
        if c.exists():
            return c
    return cands[0]


def _load_pred_long(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    col = "p50" if "p50" in df.columns else "predicted_value"
    if col not in df.columns:
        raise KeyError(f"Missing p50/predicted_value in {path}")
    return pd.to_numeric(df[col], errors="coerce")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-manifest", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--out-dir", default="artifacts/reports/activation_price_distribution")
    args = p.parse_args()

    manifest_path, payload = _resolve_manifest(Path(args.run_manifest))
    manifest_dir = manifest_path.parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_cfg = payload.get("ground_truth", {})
    gt_path = _resolve_path(str(gt_cfg.get("default_path", "")), manifest_dir)
    truth = pd.read_parquet(gt_path)

    long_map = {
        **payload.get("bundles", {}).get("da", {}).get("predictions_long", {}).get(args.split, {}),
        **payload.get("bundles", {}).get("afrr", {}).get("predictions_long", {}).get(args.split, {}),
    }
    key_pos = "pred_afrr_activation_price_pos"
    key_neg = "pred_afrr_activation_price_neg"
    if key_pos not in long_map or key_neg not in long_map:
        raise KeyError(f"Missing activation price long predictions for split={args.split}")

    pred_pos = _load_pred_long(_resolve_path(str(long_map[key_pos]), manifest_dir))
    pred_neg = _load_pred_long(_resolve_path(str(long_map[key_neg]), manifest_dir))
    true_pos = pd.to_numeric(truth.get("afrr_activation_price_vwap_pos"), errors="coerce")
    true_neg = pd.to_numeric(truth.get("afrr_activation_price_vwap_neg"), errors="coerce")

    for side, true_s, pred_s in [
        ("pos", true_pos, pred_pos),
        ("neg", true_neg, pred_neg),
    ]:
        t = true_s[np.isfinite(true_s.to_numpy(dtype=float))]
        pr = pred_s[np.isfinite(pred_s.to_numpy(dtype=float))]
        if t.empty or pr.empty:
            continue
        q = {
            "true_q01": float(t.quantile(0.01)),
            "true_q99": float(t.quantile(0.99)),
            "pred_q01": float(pr.quantile(0.01)),
            "pred_q99": float(pr.quantile(0.99)),
            "true_mean": float(t.mean()),
            "pred_mean": float(pr.mean()),
        }
        (out_dir / f"activation_price_{side}_summary.json").write_text(
            json.dumps(q, indent=2), encoding="utf-8"
        )

        plt.figure(figsize=(10, 5))
        plt.hist(t, bins=200, alpha=0.5, label="true", density=True)
        plt.hist(pr, bins=200, alpha=0.5, label="predicted", density=True)
        plt.yscale("log")
        plt.title(f"aFRR activation price distribution ({side})")
        plt.xlabel("EUR/MWh")
        plt.ylabel("Density (log scale)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"activation_price_{side}_hist.png", dpi=160)
        plt.close()

    print(f"[OK] Distribution diagnostics written to {out_dir}")


if __name__ == "__main__":
    main()

