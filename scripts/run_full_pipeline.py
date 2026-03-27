#!/usr/bin/env python3
"""Single-command end-to-end pipeline runner (fetch -> final features).

Usage:
    ./.venv/bin/python scripts/run_full_pipeline.py \
        --start 2020-11-30T23:00:00Z \
        --end 2026-01-01T02:00:00Z
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _run(cmd: list[str]) -> None:
    LOGGER.info("-> %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Run full pipeline in one command (fetch/merge/refine/prune/transform/features)."
    )
    parser.add_argument("--start", default="2020-11-30T23:00:00Z", help="Start date (UTC ISO8601).")
    parser.add_argument("--end", default="2026-01-01T02:00:00Z", help="End date (UTC ISO8601).")
    parser.add_argument("--out-dir", default="data/raw", help="Output directory for raw parquets.")
    parser.add_argument("--merged", default="data/processed/all_data.parquet", help="Merged parquet output.")
    parser.add_argument(
        "--refined", default="data/processed/all_data_refined.parquet", help="Refined parquet output."
    )
    parser.add_argument(
        "--pruned", default="data/processed/all_data_pruned.parquet", help="Pruned parquet output."
    )
    parser.add_argument(
        "--transformed",
        default="data/processed/all_data_transformed.parquet",
        help="Transformed parquet output.",
    )
    parser.add_argument(
        "--features",
        default="data/features/all_data_features.parquet",
        help="Final features parquet output.",
    )
    parser.add_argument("--skip-commodities", action="store_true", help="Skip commodities fetch.")
    parser.add_argument("--skip-smard", action="store_true", help="Skip SMARD fetch.")
    parser.add_argument(
        "--skip-bid-activation-prices",
        action="store_true",
        help="Skip anonymous-bid activation price reconstruction in fetch_regelleistung.",
    )
    parser.add_argument(
        "--smard-download-market-data-csv",
        action="store_true",
        help="Deprecated: SMARD CSV download is default. Kept for compatibility.",
    )
    parser.add_argument(
        "--smard-skip-market-data-csv",
        action="store_true",
        help="Skip SMARD market-data CSV download.",
    )
    parser.add_argument(
        "--smard-market-data-out",
        default=None,
        help="Optional path for SMARD market-data CSV output.",
    )
    parser.add_argument("--entsoe-chunk-months", type=int, default=3, help="ENTSO-E chunk size in months.")
    parser.add_argument("--entsoe-workers", type=int, default=3, help="ENTSO-E parallel workers.")
    parser.add_argument(
        "--clip-start",
        default=None,
        help="Optional clip start passed to merge_data (defaults to --start).",
    )
    parser.add_argument(
        "--clip-end",
        default=None,
        help="Optional clip end passed to merge_data (defaults to --end).",
    )
    parser.add_argument("--skip-features", action="store_true", help="Stop after transform step.")
    parser.add_argument("--verify-lags", action="store_true", help="Run scripts/verify_lags.py at the end.")
    args = parser.parse_args()

    py = sys.executable
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.merged).parent.mkdir(parents=True, exist_ok=True)
    Path(args.refined).parent.mkdir(parents=True, exist_ok=True)
    Path(args.pruned).parent.mkdir(parents=True, exist_ok=True)
    Path(args.transformed).parent.mkdir(parents=True, exist_ok=True)
    Path(args.features).parent.mkdir(parents=True, exist_ok=True)

    # 1) Fetch, merge, refine.
    collect_cmd = [
        py,
        "scripts/collect_and_merge_all_data.py",
        "--start",
        args.start,
        "--end",
        args.end,
        "--out-dir",
        args.out_dir,
        "--merged",
        args.merged,
        "--refined",
        args.refined,
        "--entsoe-chunk-months",
        str(args.entsoe_chunk_months),
        "--entsoe-workers",
        str(args.entsoe_workers),
    ]
    if args.skip_commodities:
        collect_cmd.append("--skip-commodities")
    if args.skip_smard:
        collect_cmd.append("--skip-smard")
    if args.skip_bid_activation_prices:
        collect_cmd.append("--skip-bid-activation-prices")
    if args.smard_download_market_data_csv:
        collect_cmd.append("--smard-download-market-data-csv")
    if args.smard_skip_market_data_csv:
        collect_cmd.append("--smard-skip-market-data-csv")
    if args.smard_market_data_out:
        collect_cmd.extend(["--smard-market-data-out", args.smard_market_data_out])
    if args.clip_start:
        collect_cmd.extend(["--clip-start", args.clip_start])
    if args.clip_end:
        collect_cmd.extend(["--clip-end", args.clip_end])
    _run(collect_cmd)

    # 2) Drop/prune redundant columns.
    _run(
        [
            py,
            "-m",
            "energy_trading.processing.drop_redundant_features",
            "--in",
            args.refined,
            "--out",
            args.pruned,
        ]
    )

    # 3) Transform.
    _run(
        [
            py,
            "-m",
            "energy_trading.processing.transform_data",
            "--in",
            args.pruned,
            "--out",
            args.transformed,
        ]
    )

    if args.skip_features:
        LOGGER.info("Stopping after transform (--skip-features).")
        return

    # 4) Build final features.
    _run(
        [
            py,
            "-m",
            "energy_trading.features.build_features",
            "--in",
            args.transformed,
            "--out",
            args.features,
        ]
    )

    # 5) Optional lag verification.
    if args.verify_lags:
        _run([py, "scripts/verify_lags.py", "--path", args.features])

    LOGGER.info("Full pipeline completed.")


if __name__ == "__main__":
    main()
