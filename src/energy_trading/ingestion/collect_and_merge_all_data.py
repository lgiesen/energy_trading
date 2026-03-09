"""Collect all data sources and merge them into one parquet.

Usage:
    ./.venv/bin/python scripts/collect_and_merge_all_data.py --start 2020-11-30T23:00:00Z --end 2026-01-01T02:00:00Z

Runs:
- ENTSO-E
- Day-ahead prices (Energy Charts)
- Netztransparenz
- SMARD
- Optional SMARD market-data CSV (installed_capacity.csv)
- Commodities (Yahoo Finance)
- Merge all parquets into one file
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def run(cmd: list[str]):
    LOGGER.info("-> %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fetch all datasets and merge in one go.")
    parser.add_argument("--start", default="2020-11-30T23:00:00Z", help="Start date (UTC ISO8601).")
    parser.add_argument("--end", default="2026-01-01T02:00:00Z", help="End date (UTC ISO8601).")
    parser.add_argument("--out-dir", default="data/raw", help="Output directory for individual parquets.")
    parser.add_argument("--merged", default="data/processed/all_data.parquet", help="Merged parquet output.")
    parser.add_argument("--skip-commodities", action="store_true", help="Skip commodities fetch.")
    parser.add_argument("--skip-smard", action="store_true", help="Skip SMARD fetch.")
    parser.add_argument(
        "--smard-download-market-data-csv",
        action="store_true",
        help="Deprecated: SMARD CSV download is now default. Kept for compatibility.",
    )
    parser.add_argument(
        "--smard-skip-market-data-csv",
        action="store_true",
        help="Skip SMARD market-data CSV download (installed_capacity.csv).",
    )
    parser.add_argument(
        "--smard-market-data-out",
        default=None,
        help="Optional path for SMARD market-data CSV output (default: <out-dir>/installed_capacity.csv).",
    )
    # Kept for backward compatibility (currently unused by fetch_entsoe).
    parser.add_argument("--entsoe-timeout", dest="entsoe_timeout", type=int, default=120, help="(unused) ENTSO-E HTTP timeout seconds.")
    parser.add_argument("--entsoe-chunk-days", dest="entsoe_chunk_days", type=int, default=90, help="(unused) ENTSO-E chunk size in days.")
    parser.add_argument("--entsoe-chunk-sleep", dest="entsoe_chunk_sleep", type=float, default=1.0, help="(unused) Sleep seconds between ENTSO-E chunks.")
    parser.add_argument("--entsoe-chunk-months", type=int, default=3, help="ENTSO-E chunk size in months.")
    parser.add_argument("--entsoe-workers", type=int, default=3, help="ENTSO-E parallel workers.")
    parser.add_argument(
        "--clip-start",
        default=None,
        help="Clip start passed to merge_data (defaults to --start).",
    )
    parser.add_argument(
        "--clip-end",
        default=None,
        help="Clip end passed to merge_data (defaults to --end).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    # ENTSO-E (requires ENTSOE_API_KEY env)
    run([
        py, "-m", "energy_trading.ingestion.fetch_entsoe",
        "--start", args.start,
        "--end", args.end,
        "--out", str(out_dir / "entsoe.parquet"),
        "--chunk-months", str(args.entsoe_chunk_months),
        "--workers", str(args.entsoe_workers),
    ])

    # Day-ahead prices
    run([
        py, "-m", "energy_trading.ingestion.fetch_energy_charts",
        "--start", args.start,
        "--end", args.end,
        "--out", str(out_dir / "energy_charts.parquet"),
    ])

    # Netztransparenz (assumes env creds configured)
    run([
        py, "-m", "energy_trading.ingestion.fetch_netztransparenz",
        "--start", args.start,
        "--end", args.end,
        "--chunk-days", "60",
        "--chunk-sleep", "1",
        "--out", str(out_dir / "netztransparenz.parquet"),
    ])

    # SMARD
    if not args.skip_smard:
        smard_cmd = [
            py, "-m", "energy_trading.ingestion.fetch_smard",
            "--start", args.start,
            "--end", args.end,
            "--out", str(out_dir / "smard.parquet"),
        ]
        if args.smard_download_market_data_csv:
            pass
        if args.smard_skip_market_data_csv:
            smard_cmd.extend(["--skip-market-data-csv"])
        if not args.smard_skip_market_data_csv:
            market_data_out = args.smard_market_data_out or str(out_dir / "installed_capacity.csv")
            smard_cmd.extend(["--market-data-out", market_data_out])
        run(smard_cmd)

    # yfinance commodities (write to yfinance.parquet)
    if not args.skip_commodities:
        run([
            py, "-m", "energy_trading.ingestion.fetch_yfinance",
            "--start", args.start,
            "--end", args.end,
            "--out", str(out_dir / "yfinance.parquet"),
        ])

    # Regelleistung aFRR (hourly, includes net import/export + MOL slope)
    run([
        py, "-m", "energy_trading.ingestion.fetch_regelleistung",
        "--start", args.start,
        "--end", args.end,
        "--out", str(out_dir / "regelleistung.parquet"),
        "--skip-hourly",
    ])

    # Upsample 15-min Regelleistung prices to hourly parquet (replaces in-script resample).
    run([
        py, "-m", "energy_trading.processing.upsample_regelleistung",
        "--prices-15-in", str(out_dir / "prices_15min.parquet"),
        "--out", str(out_dir / "regelleistung.parquet"),
        "--start", args.start,
        "--end", args.end,
        "--mol-dir", str(out_dir),
    ])

    # Merge all parquets in out_dir
    clip_start = args.clip_start or args.start
    clip_end = args.clip_end or args.end
    run([
        py, "-m", "energy_trading.ingestion.merge_data",
        "--data-dir", str(out_dir),
        "--out", str(Path(args.merged)),
        "--clip-start", clip_start,
        "--clip-end", clip_end,
    ])

    LOGGER.info("All tasks completed.")


if __name__ == "__main__":
    main()
