#!/usr/bin/env python3
"""Build a hybrid/champion-by-target prediction table for simulation.

Terminology:
- "Hybrid model" and "Ensemble" are both acceptable.
- This script implements a *champion-by-target ensemble*:
  choose best model per prediction target and combine into one backtest table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

PRED_TARGETS = [
    "pred_da_price",
    "pred_afrr_capacity_price_pos",
    "pred_afrr_capacity_price_neg",
    "pred_afrr_activation_price_pos",
    "pred_afrr_activation_price_neg",
    "pred_afrr_activation_rate_pos",
    "pred_afrr_activation_rate_neg",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if "manifest_path" in payload:
        raw = Path(str(payload["manifest_path"]))
        p2 = raw if raw.is_absolute() else (path.parent / raw)
        if p2.exists():
            payload = _read_json(p2)
    return payload


def _resolve_pred_path(configured: str, manifest_path: Path) -> Path:
    p = Path(configured)
    if p.exists():
        return p
    mdir = manifest_path.parent
    cands = [mdir / p.name, mdir / "predictions" / p.name]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"Prediction file not found: {configured} (manifest={manifest_path})")


def _backtest_table_from_manifest(manifest_path: Path, split: str) -> pd.DataFrame:
    payload = _load_manifest(manifest_path)
    bundles = payload.get("bundles", {})
    da_cfg = bundles.get("da", {}).get("predictions", {}).get(split)
    afrr_cfg = bundles.get("afrr", {}).get("predictions", {}).get(split)
    if not da_cfg or not afrr_cfg:
        raise KeyError(f"Missing da/afrr predictions for split={split} in {manifest_path}")

    da_path = _resolve_pred_path(str(da_cfg), manifest_path)
    afrr_path = _resolve_pred_path(str(afrr_cfg), manifest_path)

    da_df = pd.read_parquet(da_path)
    afrr_df = pd.read_parquet(afrr_path)
    if "timestamp_utc" not in da_df.columns or "timestamp_utc" not in afrr_df.columns:
        raise KeyError("timestamp_utc missing in source prediction files")

    out = da_df.merge(afrr_df, on="timestamp_utc", how="inner").sort_values("timestamp_utc").reset_index(drop=True)
    return out


def _find_quantile_cols(df: pd.DataFrame, pred_col: str) -> list[str]:
    pref = f"{pred_col}_p"
    return sorted([c for c in df.columns if c.startswith(pref)])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build champion-by-target hybrid prediction table.")
    p.add_argument(
        "--recommendation-csv",
        default="artifacts/benchmarks/final_report/recommendation_per_target_test.csv",
        help="CSV from generate_final_benchmark_report.py containing prediction_column, model_label, run_dir.",
    )
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument(
        "--out",
        default="artifacts/simulation_runs/hybrid/test/backtest_table_test.parquet",
        help="Output hybrid backtest table parquet.",
    )
    p.add_argument(
        "--manifest-filename",
        default="manifest.json",
        help="Manifest filename inside run_dir (default: manifest.json).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reco = pd.read_csv(args.recommendation_csv)
    need = {"prediction_column", "model_label", "run_dir"}
    miss = need - set(reco.columns)
    if miss:
        raise KeyError(f"Missing columns in recommendation CSV: {sorted(miss)}")

    reco = reco[reco["prediction_column"].isin(PRED_TARGETS)].copy()
    if reco.empty:
        raise RuntimeError("No supported prediction targets found in recommendation CSV")

    # one winner per target (if duplicates, take first row order)
    reco = reco.drop_duplicates(subset=["prediction_column"], keep="first").reset_index(drop=True)

    # Load source tables for each unique run_dir once.
    cache: dict[str, pd.DataFrame] = {}
    for rd in sorted(set(reco["run_dir"].astype(str))):
        mpath = Path(rd) / args.manifest_filename
        if not mpath.exists():
            raise FileNotFoundError(f"Manifest not found: {mpath}")
        cache[rd] = _backtest_table_from_manifest(mpath, split=args.split)

    # Use first source as base table (keeps true columns + timestamp + optional extras).
    base_rd = str(reco.iloc[0]["run_dir"])
    hybrid = cache[base_rd].copy()

    # Swap in champion prediction columns per target.
    audit_rows: list[dict[str, Any]] = []
    for _, rr in reco.iterrows():
        pred_col = str(rr["prediction_column"])
        rd = str(rr["run_dir"])
        src = cache[rd]

        # align by timestamp
        cols = ["timestamp_utc", pred_col]
        qcols = _find_quantile_cols(src, pred_col)
        cols.extend(qcols)
        missing_src = [c for c in cols if c not in src.columns]
        if missing_src:
            raise KeyError(f"Source run_dir={rd} missing columns for {pred_col}: {missing_src}")

        part = src[cols].copy()
        rename_map = {c: f"{c}__src" for c in cols if c != "timestamp_utc"}
        part = part.rename(columns=rename_map)
        hybrid = hybrid.drop(columns=[c for c in cols if c in hybrid.columns and c != "timestamp_utc"], errors="ignore")
        hybrid = hybrid.merge(part, on="timestamp_utc", how="left")
        for c in cols:
            if c == "timestamp_utc":
                continue
            hybrid[c] = hybrid[f"{c}__src"]
            hybrid.drop(columns=[f"{c}__src"], inplace=True)

        audit_rows.append(
            {
                "prediction_column": pred_col,
                "model_label": str(rr.get("model_label", "")),
                "run_dir": rd,
                "quantile_cols_included": ",".join(qcols),
            }
        )

    hybrid = hybrid.sort_values("timestamp_utc").reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hybrid.to_parquet(out, index=False)

    audit = pd.DataFrame(audit_rows)
    audit_path = out.parent / "hybrid_target_mapping.csv"
    audit.to_csv(audit_path, index=False)

    print("[OK] Hybrid champion-by-target table created.")
    print(f"- split: {args.split}")
    print(f"- rows: {len(hybrid)}")
    print(f"- out: {out}")
    print(f"- mapping: {audit_path}")


if __name__ == "__main__":
    main()
