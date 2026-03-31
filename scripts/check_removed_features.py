"""Audit check: ensure removed legacy feature families are absent.

Usage:
    ./.venv/bin/python scripts/check_removed_features.py \
        --path data/processed/all_data_transformed.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that no columns contain 'reconstructed' or 'grid_share'."
    )
    parser.add_argument(
        "--path",
        default="data/processed/all_data_transformed.parquet",
        help="Parquet path to scan (default: data/processed/all_data_transformed.parquet).",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")

    cols = list(pq.ParquetFile(path).schema.names)
    forbidden = [
        c for c in cols
        if ("reconstructed" in c.lower()) or ("grid_share" in c.lower())
    ]
    if forbidden:
        print("❌ FEHLER: Verbotene Legacy-Spalten gefunden:")
        for c in sorted(forbidden):
            print(f"- {c}")
        raise SystemExit(1)

    print(f"✅ OK: Keine 'reconstructed'/'grid_share'-Spalten in {path}")
    print(f"[INFO] Columns scanned: {len(cols)}")


if __name__ == "__main__":
    main()

