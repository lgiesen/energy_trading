"""Merge DA and aFRR prediction files into a single simulation input table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _resolve_manifest(path: Path) -> tuple[dict, str | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    run_id = payload.get("run_id")
    if "manifest_path" in payload:
        path = Path(payload["manifest_path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_id = run_id or payload.get("run_id")
    return payload, run_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge DA and aFRR prediction tables for simulation.")
    p.add_argument("--run-manifest", default="artifacts/model_runs/latest.json")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument(
        "--out",
        default="",
        help="Output path for merged backtest table. If empty: artifacts/simulation_runs/<run_id>/<split>/backtest_table_<split>.parquet",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest, run_id = _resolve_manifest(Path(args.run_manifest))

    da_pred = Path(manifest["bundles"]["da"]["predictions"][args.split])
    afrr_pred = Path(manifest["bundles"]["afrr"]["predictions"][args.split])

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
