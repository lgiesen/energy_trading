#!/usr/bin/env python3
"""Diagnose null gaps that can collapse time range during feature truncation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_repo_root() -> Path:
    root = Path.cwd().resolve()
    if (root / "src").exists():
        return root
    for parent in root.parents:
        if (parent / "src").exists():
            return parent
    raise RuntimeError("Could not resolve REPO_ROOT (directory containing 'src').")


REPO_ROOT = _resolve_repo_root()


def _format_ts(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return "None"
    return str(ts)


def analyze_null_gaps(df: pd.DataFrame, ts_col: str = "timestamp_utc") -> pd.DataFrame:
    if ts_col not in df.columns:
        raise KeyError(f"Missing required timestamp column: {ts_col}")

    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError(f"Timestamp column '{ts_col}' contains invalid values.")

    num = df.select_dtypes(include=[np.number]).copy()
    if num.empty:
        raise ValueError("No numeric columns found.")

    total_rows = len(df)
    rows = []
    for col in num.columns:
        s = pd.to_numeric(num[col], errors="coerce")
        valid = s.notna()
        null_count = int((~valid).sum())
        null_pct = float(null_count / total_rows * 100.0) if total_rows else 0.0
        first_valid = ts[valid].min() if valid.any() else pd.NaT
        last_valid = ts[valid].max() if valid.any() else pd.NaT
        rows.append(
            {
                "column": col,
                "null_count": null_count,
                "null_percentage": round(null_pct, 4),
                "first_valid_timestamp": first_valid,
                "last_valid_timestamp": last_valid,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["null_count", "column"], ascending=[False, True]).reset_index(drop=True)
    return out


def find_bottlenecks(profile: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    valid_first = profile.dropna(subset=["first_valid_timestamp"]).copy()
    valid_last = profile.dropna(subset=["last_valid_timestamp"]).copy()
    if valid_first.empty or valid_last.empty:
        raise ValueError("No valid first/last timestamps available for bottleneck analysis.")

    # Latest first_valid_timestamp => strongest front-cut driver.
    start_bottleneck = valid_first.sort_values(
        ["first_valid_timestamp", "null_count", "column"],
        ascending=[False, False, True],
    ).iloc[0]

    # Earliest last_valid_timestamp => strongest tail-cut driver.
    end_bottleneck = valid_last.sort_values(
        ["last_valid_timestamp", "null_count", "column"],
        ascending=[True, False, True],
    ).iloc[0]

    return start_bottleneck, end_bottleneck


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug null gaps in transformed parquet before feature truncation.")
    p.add_argument(
        "--path",
        default="data/processed/all_data_transformed.parquet",
        help="Path to pre-feature transformed parquet.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many most-problematic columns (by null_count) to print.",
    )
    p.add_argument(
        "--out-csv",
        default=None,
        help="Optional CSV output path for full null profile.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = (REPO_ROOT / args.path).resolve() if not Path(args.path).is_absolute() else Path(args.path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_path}")

    df = pd.read_parquet(in_path, engine="pyarrow")
    profile = analyze_null_gaps(df, ts_col="timestamp_utc")
    start_bottle, end_bottle = find_bottlenecks(profile)

    print(f"[INFO] Input: {in_path}")
    print(f"[INFO] Rows: {len(df)} | Numeric columns: {profile.shape[0]}")
    print("\nTop-10 problematische Spalten (nach null_count):")
    print(
        profile.head(args.top_k)[
            ["column", "null_count", "null_percentage", "first_valid_timestamp", "last_valid_timestamp"]
        ].to_string(index=False)
    )

    print(
        "\nKritischer Flaschenhals Start: "
        f"Spalte [{start_bottle['column']}] beginnt erst am [{_format_ts(start_bottle['first_valid_timestamp'])}]."
    )
    print(
        "Kritischer Flaschenhals Ende: "
        f"Spalte [{end_bottle['column']}] endet bereits am [{_format_ts(end_bottle['last_valid_timestamp'])}]."
    )

    if args.out_csv:
        out_path = (REPO_ROOT / args.out_csv).resolve() if not Path(args.out_csv).is_absolute() else Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        profile.to_csv(out_path, index=False)
        print(f"[INFO] Full null profile saved to: {out_path}")


if __name__ == "__main__":
    main()
