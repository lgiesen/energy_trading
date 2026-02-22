"""Wrapper: collect raw data and run missing-value handling.

Usage:
    ./.venv/bin/python scripts/collect_and_clean.py \
        --start 2020-11-30T23:00:00Z --end 2026-01-01T02:00:00Z \
        --raw-out data/processed/all_data.parquet \
        --clean-out data/processed/all_data_clean.parquet
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("INFO:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect raw data and then clean missing values.")
    parser.add_argument("--start", required=True, help="Start ISO8601 (UTC).")
    parser.add_argument("--end", required=True, help="End ISO8601 (UTC).")
    parser.add_argument("--raw-out", default="data/processed/all_data.parquet", help="Output path for raw merged data.")
    parser.add_argument("--clean-out", default="data/processed/all_data_clean.parquet", help="Output path for cleaned data.")
    parser.add_argument(
        "--smard-download-market-data-csv",
        action="store_true",
        help="Deprecated: SMARD CSV download is default. Kept for compatibility.",
    )
    parser.add_argument(
        "--smard-skip-market-data-csv",
        action="store_true",
        help="Skip SMARD market-data CSV download during fetch.",
    )
    parser.add_argument(
        "--smard-market-data-out",
        default=None,
        help="Optional path for SMARD market-data CSV output.",
    )
    args = parser.parse_args()

    python_bin = Path(".venv/bin/python")
    if not python_bin.exists():
        raise RuntimeError(".venv/bin/python not found. Activate venv or create it first.")

    _run([
        str(python_bin),
        "scripts/collect_and_merge_all_data.py",
        "--start", args.start,
        "--end", args.end,
        *( ["--smard-download-market-data-csv"] if args.smard_download_market_data_csv else [] ),
        *( ["--smard-skip-market-data-csv"] if args.smard_skip_market_data_csv else [] ),
        *( ["--smard-market-data-out", args.smard_market_data_out] if args.smard_market_data_out else [] ),
    ])

    _run([
        str(python_bin),
        "-m", "energy_trading.processing.handle_missing_values",
        "--in", args.raw_out,
        "--out", args.clean_out,
    ])


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: command failed with exit code {exc.returncode}")
        sys.exit(exc.returncode)
