"""Collect all data sources and merge them into one parquet.

Runs:
- ENTSO-E
- Day-ahead prices (Energy Charts)
- Netztransparenz
- SMARD
- Commodities (Yahoo Finance)
- Merge all parquets into one file
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import logging
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def run(cmd: list[str]):
    LOGGER.info("-> %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fetch all datasets and merge in one go.")
    parser.add_argument("--start", default="2020-12-01T00:00:00Z", help="Start date (UTC ISO8601).")
    parser.add_argument("--end", default="2026-01-01T02:00:00Z", help="End date (UTC ISO8601).")
    parser.add_argument("--out-dir", default="data", help="Output directory for individual parquets.")
    parser.add_argument("--merged", default="data/all_data.parquet", help="Merged parquet output.")
    parser.add_argument("--skip-commodities", action="store_true", help="Skip commodities fetch.")
    parser.add_argument("--skip-smard", action="store_true", help="Skip SMARD fetch.")
    parser.add_argument("--entsoe-timeout", dest="entsoe_timeout", type=int, default=120, help="ENTSO-E HTTP timeout seconds (default 120).")
    parser.add_argument("--entsoe-chunk-days", dest="entsoe_chunk_days", type=int, default=90, help="ENTSO-E chunk size in days to avoid timeouts (default 90).")
    parser.add_argument("--entsoe-chunk-sleep", dest="entsoe_chunk_sleep", type=float, default=1.0, help="Sleep seconds between ENTSO-E chunks (default 1.0).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    # ENTSO-E (requires ENTSOE_API_KEY env)
    run([
        py, "-m", "ingestion.fetch_entsoe",
        "--start", args.start,
        "--end", args.end,
        "--timeout", str(args.entsoe_timeout),
        "--chunk-days", str(args.entsoe_chunk_days),
        "--chunk-sleep", str(args.entsoe_chunk_sleep),
        "--out", str(out_dir / "entsoe.parquet"),
    ])

    # Day-ahead prices
    run([
        py, "-m", "ingestion.fetch_energy_charts_prices",
        "--start", args.start,
        "--end", args.end,
        "--out", str(out_dir / "energy_charts.parquet"),
    ])

    # Netztransparenz (assumes env creds configured)
    run([
        py, "-m", "ingestion.fetch_netztransparenz",
        "--start", args.start,
        "--end", args.end,
        "--chunk-days", "60",
        "--chunk-sleep", "1",
        "--out", str(out_dir / "netztransparenz.parquet"),
    ])

    # SMARD
    if not args.skip_smard:
        run([
            py, "-m", "ingestion.fetch_smard",
            "--start", args.start,
            "--end", args.end,
            "--out", str(out_dir / "smard.parquet"),
        ])

    # yfinance commodities
    if not args.skip_commodities:
        run([
            py, "-m", "ingestion.fetch_yfinance",
            "--start", args.start,
            "--end", args.end,
            "--out", str(out_dir / "commodities.parquet"),
        ])

    # Regelleistung aFRR (hourly, includes net import/export + MOL slope)
    run([
        py, "-m", "ingestion.fetch_regelleistung",
        "--start", args.start,
        "--end", args.end,
        "--out", str(out_dir / "regelleistung.parquet"),
    ])

    # Merge all parquets in out_dir
    run([
        py, "-m", "ingestion.merge_data",
        "--data-dir", str(out_dir),
        "--out", str(Path(args.merged)),
    ])

    LOGGER.info("All tasks completed.")


if __name__ == "__main__":
    main()
