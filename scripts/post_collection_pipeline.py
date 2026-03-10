#!/usr/bin/env python3
"""Post-collection pipeline wrapper: merge -> clean -> transform -> features.

Usage:
    ./.venv/bin/python scripts/post_collection_pipeline.py \
        --clip-start 2020-11-30T23:00:00Z \
        --clip-end 2025-12-31T23:00:00Z
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
        description="Run post-collection processing (merge -> clean -> transform -> features)."
    )
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing raw parquet files.")
    parser.add_argument(
        "--merged",
        default="data/processed/all_data.parquet",
        help="Merged parquet output.",
    )
    parser.add_argument(
        "--clean-out",
        default="data/processed/all_data_clean.parquet",
        help="Cleaned parquet output.",
    )
    parser.add_argument(
        "--transformed-out",
        default="data/processed/all_data_transformed.parquet",
        help="Transformed parquet output.",
    )
    parser.add_argument(
        "--features-out",
        default="data/features/all_data_features.parquet",
        help="Features parquet output.",
    )
    parser.add_argument(
        "--clip-start",
        default="2020-11-30T23:00:00Z",
        help="Clip start (timezone-aware recommended, UTC Z).",
    )
    parser.add_argument(
        "--clip-end",
        default="2025-12-31T23:00:00Z",
        help="Clip end (timezone-aware recommended, UTC Z).",
    )
    parser.add_argument(
        "--resample-freq",
        default="1h",
        help="Resample frequency passed to merge_data (default: 1h).",
    )
    parser.add_argument(
        "--skip-transform",
        action="store_true",
        help="Stop after cleaning step.",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Stop after transform step.",
    )
    args = parser.parse_args()

    py = sys.executable
    Path(args.merged).parent.mkdir(parents=True, exist_ok=True)
    Path(args.clean_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.transformed_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.features_out).parent.mkdir(parents=True, exist_ok=True)

    # 1) Merge collected raw files.
    merge_cmd = [
        py,
        "-m",
        "energy_trading.ingestion.merge_data",
        "--data-dir",
        args.data_dir,
        "--out",
        args.merged,
        "--clip-start",
        args.clip_start,
        "--clip-end",
        args.clip_end,
    ]
    if args.resample_freq is not None:
        merge_cmd.extend(["--resample-freq", args.resample_freq])
    _run(merge_cmd)

    # 2) Handle missing values.
    _run(
        [
            py,
            "-m",
            "energy_trading.processing.handle_missing_values",
            "--in",
            args.merged,
            "--out",
            args.clean_out,
        ]
    )

    if args.skip_transform:
        LOGGER.info("Stopping after cleaning (--skip-transform).")
        return

    # 3) Transform.
    _run(
        [
            py,
            "-m",
            "energy_trading.processing.transform_data",
            "--in",
            args.clean_out,
            "--out",
            args.transformed_out,
        ]
    )

    if args.skip_features:
        LOGGER.info("Stopping after transform (--skip-features).")
        return

    # 4) Build features.
    _run(
        [
            py,
            "-m",
            "energy_trading.features.build_features",
            "--in",
            args.transformed_out,
            "--out",
            args.features_out,
        ]
    )

    LOGGER.info("Post-collection pipeline completed.")


if __name__ == "__main__":
    main()
