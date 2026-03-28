#!/usr/bin/env python3
"""End-to-end pipeline wrapper: fetch -> merge -> refine -> prune -> transform -> features.

Usage:
    ./.venv/bin/python scripts/run_pipeline.py --start 2020-11-30T23:00:00Z --end 2026-03-01T02:00:00Z
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def run(cmd: list[str]) -> None:
    LOGGER.info("-> %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run full data pipeline (fetch -> features).")
    parser.add_argument("--start", default="2020-11-30T23:00:00Z", help="Start date (UTC ISO8601).")
    parser.add_argument("--end", default="2026-03-01T02:00:00Z", help="End date (UTC ISO8601).")
    parser.add_argument("--out-dir", default="data/raw", help="Output directory for raw parquets.")
    parser.add_argument("--merged", default="data/processed/all_data.parquet", help="Merged parquet output.")
    parser.add_argument("--skip-commodities", action="store_true", help="Skip commodities fetch.")
    parser.add_argument("--skip-smard", action="store_true", help="Skip SMARD fetch.")
    parser.add_argument(
        "--smard-download-market-data-csv",
        action="store_true",
        help="Deprecated: SMARD CSV download is default. Kept for compatibility.",
    )
    parser.add_argument(
        "--smard-skip-market-data-csv",
        action="store_true",
        help="Skip SMARD market-data CSV download during SMARD fetch.",
    )
    parser.add_argument(
        "--smard-market-data-out",
        default=None,
        help="Optional path for SMARD market-data CSV output.",
    )
    parser.add_argument("--skip-features", action="store_true", help="Stop after transform step.")
    parser.add_argument("--entsoe-chunk-months", type=int, default=1, help="ENTSO-E chunk size in months.")
    parser.add_argument("--entsoe-workers", type=int, default=1, help="ENTSO-E parallel workers.")
    args = parser.parse_args()

    py = sys.executable
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Fetch + merge raw data
    run([
        py, "scripts/collect_and_merge_all_data.py",
        "--start", args.start,
        "--end", args.end,
        "--out-dir", str(out_dir),
        "--merged", args.merged,
        "--entsoe-chunk-months", str(args.entsoe_chunk_months),
        "--entsoe-workers", str(args.entsoe_workers),
        *( ["--skip-commodities"] if args.skip_commodities else [] ),
        *( ["--skip-smard"] if args.skip_smard else [] ),
        *( ["--smard-download-market-data-csv"] if args.smard_download_market_data_csv else [] ),
        *( ["--smard-skip-market-data-csv"] if args.smard_skip_market_data_csv else [] ),
        *( ["--smard-market-data-out", args.smard_market_data_out] if args.smard_market_data_out else [] ),
    ])

    # 2) Prune redundant columns on refined output.
    refined_path = "data/processed/all_data_refined.parquet"
    pruned_path = "data/processed/all_data_pruned.parquet"
    run([
        py, "-m", "energy_trading.processing.drop_redundant_features",
        "--in", refined_path,
        "--out", pruned_path,
    ])

    # 3) Transform
    transformed_path = "data/processed/all_data_transformed.parquet"
    run([
        py, "-m", "energy_trading.processing.transform_data",
        "--in", pruned_path,
        "--out", transformed_path,
    ])

    if args.skip_features:
        LOGGER.info("Skipping feature generation (--skip-features).")
        return

    # 4) Build features
    run([
        py, "-m", "energy_trading.features.build_features",
        "--in", transformed_path,
        "--out", "data/features/all_data_features.parquet",
    ])

    LOGGER.info("Pipeline completed.")


if __name__ == "__main__":
    main()
