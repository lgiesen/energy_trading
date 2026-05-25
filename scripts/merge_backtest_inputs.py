"""Merge DA and aFRR prediction files into a single simulation input table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _resolve_manifest(path: Path) -> tuple[dict, str | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    run_id = payload.get("run_id")
    manifest_path = path
    if "manifest_path" in payload:
        raw = Path(str(payload["manifest_path"]))
        manifest_path = raw if raw.is_absolute() else (path.parent / raw)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = run_id or payload.get("run_id")
    return payload, run_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge DA and aFRR prediction tables for simulation.")
    p.add_argument("--model-key", choices=["xgboost", "linear", "tft"], default="xgboost")
    p.add_argument("--run-manifest", default="")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument(
        "--out",
        default="",
        help="Output path for merged backtest table. If empty: artifacts/simulation_runs/<run_id>/<split>/backtest_table_<split>.parquet",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_manifest = args.run_manifest.strip()
    if not run_manifest:
        run_manifest = f"artifacts/model_runs/latest_{args.model_key}.json"
    manifest, run_id = _resolve_manifest(Path(run_manifest))

    run_manifest_path = Path(run_manifest)
    payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if "manifest_path" in payload:
        raw = Path(str(payload["manifest_path"]))
        manifest_file = raw if raw.is_absolute() else (run_manifest_path.parent / raw)
    else:
        manifest_file = run_manifest_path
    manifest_dir = manifest_file.parent

    da_pred = Path(manifest["bundles"]["da"]["predictions"][args.split])
    afrr_pred = Path(manifest["bundles"]["afrr"]["predictions"][args.split])
    if not da_pred.is_absolute():
        da_pred = manifest_dir / da_pred
    if not afrr_pred.is_absolute():
        afrr_pred = manifest_dir / afrr_pred

    if args.out.strip():
        out = Path(args.out)
    else:
        rid = run_id or "manual"
        out = Path("artifacts/simulation_runs") / rid / args.split / f"backtest_table_{args.split}.parquet"

    out.parent.mkdir(parents=True, exist_ok=True)

    da_df = pd.read_parquet(da_pred)
    afrr_df = pd.read_parquet(afrr_pred)
    merged = da_df.merge(afrr_df, on="timestamp_utc", how="inner")
    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)
    merged.to_parquet(out, index=False)

    print("[OK] Backtest input table merged.")
    print(f"- run_id: {run_id}")
    print(f"- split: {args.split}")
    print(f"- rows: {len(merged)}")
    print(f"- out: {out}")


if __name__ == "__main__":
    main()
